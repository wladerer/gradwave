"""ASE Calculator interface (Layer C).

Supports energy, forces, and stress (fixed-basis, via the differentiable
radial transforms in pseudo/radial_torch.py), so variable-cell relaxation
through ase.filters.FrechetCellFilter works. The usual plane-wave caveat
applies: relaxing the cell at fixed ecut carries Pulay (basis-incompleteness)
pressure — converge ecut or re-relax at the new cell.

Geometry setup (grids, form-factor tables) is cached and reused when only
positions change, which is the common case during a relaxation; any cell
change triggers a full re-setup.

Every calculate() computes energy, forces, AND stress in one pass off the
converged SCF state (the stress is one extra autograd backward — far cheaper
than a second calculate), and a geometry guard skips the SCF outright when
the incoming atoms match the state the stored result was computed at. ASE's
FrechetCellFilter asks for forces and stress as two separate get_property
calls; without both pieces every variable-cell BFGS step would pay a second
warm SCF plus a full forces evaluation just to append the stress.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from beartype import beartype
from jaxtyping import Complex, Float, jaxtyped
from typing_extensions import override

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE, SpinXC
from gradwave.dtypes import RDTYPE
from gradwave.grids import FFTGrid, GSphere
from gradwave.postscf.forces import forces as hf_forces
from gradwave.scf.loop import System, scf, setup_system

if TYPE_CHECKING:
    from gradwave.core.hubbard import HubbardManifold
    from gradwave.core.xc.base import XCFunctional
    from gradwave.pseudo.upf import UPFData
    from gradwave.pseudo.upf_paw import PAWData
    from gradwave.scf.loop import SCFResult
    from gradwave.scf.results import USPPResult
    from gradwave.scf.uspp_setup import USPPSystem

_XC = {"lda": LDA_PW92, "pbe": PBE, "r2scan": R2SCAN}
# Spin-polarized (nspin=2) counterparts, keyed identically — same registry the
# api's _spin_setup uses, so the calculator's collinear path matches task: scf.
_SPIN_XC = {"lda": LSDA_PW92, "pbe": SpinPBE, "r2scan": SpinR2SCAN}

# The exact-match tuple `_state_key()` builds for the (geometry, parameters)
# state an SCF result is valid at (see its docstring below).
_StateKey = tuple[bytes, bytes, bytes, tuple[str, ...], bytes, str, str]


def _fft_grid(system: System | USPPSystem) -> FFTGrid:
    """`System`/`USPPSystem` both declare `grid` as bare `object` (their
    modules stay import-light with respect to grids.py); every actual
    instance is an `FFTGrid` — this names that fact for the type checker
    instead of re-asserting it at each access site."""
    return cast("FFTGrid", system.grid)


@jaxtyped(typechecker=beartype)
def _resample_fft_axis(
    box: Complex[torch.Tensor, "..."], axis: int, n_old: int, n_new: int
) -> Complex[torch.Tensor, "..."]:
    """Move an FFT-box axis from n_old to n_new points, matching by integer
    Miller index. Both boxes use numpy/torch FFT ordering (fftfreq): the
    coefficient with Miller index m lives at array position m mod n. Copy every
    coefficient whose Miller index is representable in BOTH boxes; new positions
    with no old counterpart stay zero (high-G zero-fill), old ones that no
    longer fit are dropped (truncation)."""
    if n_old == n_new:
        return box
    m_old = np.fft.fftfreq(n_old, d=1.0 / n_old).astype(np.int64)  # pos → Miller
    m_new = np.fft.fftfreq(n_new, d=1.0 / n_new).astype(np.int64)
    pos_of_m = {int(m): i for i, m in enumerate(m_old)}
    src, dst = [], []
    for j, m in enumerate(m_new):
        i = pos_of_m.get(int(m))
        if i is not None:
            src.append(i)
            dst.append(j)
    dev = box.device
    src_t = torch.as_tensor(src, dtype=torch.long, device=dev)
    dst_t = torch.as_tensor(dst, dtype=torch.long, device=dev)
    out_shape = list(box.shape)
    out_shape[axis] = n_new
    out = torch.zeros(out_shape, dtype=box.dtype, device=dev)
    out.index_copy_(axis, dst_t, box.index_select(axis, src_t))
    return out


@jaxtyped(typechecker=beartype)
def _remap_density_to_grid(
    rho_old: Float[torch.Tensor, "n1 n2 n3"],
    old_grid: FFTGrid,
    new_grid: FFTGrid,
    n_electrons: float,
) -> Float[torch.Tensor, "m1 m2 m3"]:
    """Remap a real-space density from ``old_grid``'s FFT box onto ``new_grid``'s
    via G-space, for a warm start that survives an ecut-fixed FFT-grid resize
    (variable-cell relaxation strains the cell, so the box dims change).

    FFT ρ to its Fourier-series coefficients, re-place them by integer Miller
    index (zero-fill new high-G components, drop ones that no longer fit),
    inverse-FFT on the new box, then renormalize the integral to ``n_electrons``
    on the new cell volume — which also absorbs the volume-change scaling that a
    same-grid warm start applies explicitly. dtype and device are preserved."""
    from gradwave.core.fftbox import g_to_r_box, r_to_g

    old_shape = tuple(old_grid.shape)
    new_shape = tuple(new_grid.shape)
    cg = r_to_g(rho_old)  # Fourier-series coefficients on the old box (complex)
    for axis in range(3):
        cg = _resample_fft_axis(cg, axis, old_shape[axis], new_shape[axis])
    rho_new = g_to_r_box(cg, real=True).to(dtype=rho_old.dtype)
    vol = float(new_grid.volume)
    # G-space truncation can dip the reconstruction slightly negative between
    # atoms; floor it (as SAD does) then pin the integral to N_e exactly.
    rho_new = torch.clamp(rho_new, min=1e-12)
    return rho_new * (float(n_electrons) / (rho_new.mean() * vol))


def _remap_coeffs_to_spheres(
    coeffs_old: list[torch.Tensor] | None,
    spheres_old: list[GSphere],
    spheres_new: list[GSphere],
) -> list[torch.Tensor] | None:
    """(Not @jaxtyped: each list element is legitimately a differently-shaped
    (nb, npw_k) block — npw varies per k-point — and jaxtyping's list[Array]
    support enforces one consistent size per named axis across every element,
    which would reject exactly the varying-npw case this function exists to
    handle. See _resample_fft_axis / _remap_density_to_grid for the
    single-array cases where that consistency check is exactly what we want.)

    Remap per-k band plane-wave coefficients from one G-sphere set onto
    another by integer Miller-index matching, so the previous ionic step's
    eigenvectors survive an ecut-fixed G-sphere resize (variable-cell relaxation
    strains the cell; the set of |k+G|² ≤ ecut plane waves changes per k). Each
    band's coefficient at Miller index m moves to that index's slot in the new
    sphere; new components (no old counterpart) are zero-filled, old ones that no
    longer fit are dropped — the plane-wave analogue of ``_remap_density_to_grid``.

    Returns a new list-per-k of ``(nb, npw_new)`` coefficient blocks, or None
    when the k-point sets are structurally incompatible (different k count, or a
    per-k fractional-k mismatch, e.g. a symmetry-group change reshaped the IBZ) —
    the caller then bails to a fresh orbital start."""
    if coeffs_old is None or len(coeffs_old) != len(spheres_new) \
            or len(spheres_old) != len(spheres_new):
        return None
    out = []
    for ck, sph_o, sph_n in zip(coeffs_old, spheres_old, spheres_new,
                                strict=True):
        if ck is None or not np.allclose(np.asarray(sph_o.k_frac, dtype=float),
                                         np.asarray(sph_n.k_frac, dtype=float)):
            return None
        m_o = sph_o.miller.detach().cpu().numpy()
        m_n = sph_n.miller.detach().cpu().numpy()
        pos_of = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(m_o)}
        src, dst = [], []
        for j, (a, b, c) in enumerate(m_n):
            i = pos_of.get((int(a), int(b), int(c)))
            if i is not None:
                src.append(i)
                dst.append(j)
        nb = ck.shape[0]
        new_c = torch.zeros(nb, m_n.shape[0], dtype=ck.dtype, device=ck.device)
        if src:
            src_t = torch.as_tensor(src, dtype=torch.long, device=ck.device)
            dst_t = torch.as_tensor(dst, dtype=torch.long, device=ck.device)
            new_c.index_copy_(1, dst_t, ck.detach().index_select(1, src_t))
        out.append(new_c)
    return out


class GradWave(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress",
                              "magmom"]
    # ase.calculators.calculator.Calculator declares these as `None` at
    # __init__ time (its BaseCalculator sets `self.atoms = None` before
    # `calculate()` first populates it), so its own inferred attribute type
    # includes `None`. By the time GradWave's __init__ returns, ASE's
    # Calculator.__init__ has already resolved `self.parameters` to a real
    # dict (falling back to get_default_parameters()); `self.atoms` is set on
    # the first calculate(). Redeclaring the narrower type here documents
    # that contract for every method below instead of re-asserting it at
    # each access site.
    atoms: Atoms
    parameters: dict[str, Any]

    def __init__(
        self,
        *,
        ecut: float,
        pseudopotentials: dict[str, str],  # element → UPF path
        xc: str = "pbe",
        ecutrho: float | None = None,  # density cutoff (USPP/PAW); default 4×ecut
        kpts: Sequence[int] = (1, 1, 1),
        kshift: Sequence[int] = (0, 0, 0),
        smearing: str = "none",
        width: float = 0.1,
        nbands: int | None = None,
        use_symmetry: bool = True,
        nspin: int = 1,  # 1 (restricted) or 2 (collinear); auto-bumps to 2 when
        # the atoms carry nonzero initial magnetic moments (see _resolve_spin)
        tot_magnetization: float | None = None,  # fix M=N↑−N↓ for a no-smearing
        # nspin=2 run; None → derive from the initial moments (smearing="none")
        # or let the shared Fermi level find it when a smearing is set
        max_iter: int = 100,
        etol: float = 1e-8,
        rhotol: float = 1e-7,
        diago_tol: float = 1e-9,
        mixing_scheme: str = "pulay",  # USPP/PAW path only (NC scf is Pulay)
        mixing_alpha: float = 0.7,
        mixing_history: int | None = None,  # None → solver's per-scheme default
        mixing_kerker: bool | None = None,  # None → auto (on iff smeared)
        device: str = "cpu",
        compile_xc: bool = False,
        eigensolver: str = "davidson",  # davidson | chebyshev (NC path only)
        precond: str = "kerker",  # kerker | local_tf (NC and USPP/PAW paths)
        reuse_wavefunctions: bool = True,  # seed the Davidson eigensolver from the
        # previous ionic step's eigenvectors (alongside the density warm start);
        # False falls back to a density-only warm start (cold orbital seed)
        dispersion: bool | dict[str, Any] | None = None,  # opt-in D3(BJ):
        # True/False, or a dict of overrides
        hubbard: Iterable[object] | None = None,  # DFT+U: list of per-species
        # manifolds, each an object/dict with .species (element symbol) / .l /
        # .u [eV] / optional .j; None → off. Norm-conserving only through the
        # calculator (USPP/PAW +U stress is not implemented and the
        # calculator always evaluates stress for a cell).
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if nspin not in (1, 2):
            # collinear spin (nspin=2) threads through the norm-conserving SCF,
            # forces, and stress below; noncollinear/SOC has no calculator path
            raise ValueError(
                "GradWave supports nspin=1 (spin-restricted) and nspin=2 "
                "(collinear spin); noncollinear/spin-orbit has no calculator "
                "path yet (use task: magnetism via the api)")
        self.parameters.update(
            dict(ecut=ecut, ecutrho=ecutrho, xc=xc, kpts=tuple(kpts),
                 kshift=tuple(kshift), smearing=smearing, width=width,
                 nbands=nbands, use_symmetry=use_symmetry, nspin=nspin,
                 tot_magnetization=tot_magnetization,
                 max_iter=max_iter, etol=etol, rhotol=rhotol,
                 diago_tol=diago_tol, mixing_scheme=mixing_scheme,
                 mixing_alpha=mixing_alpha, mixing_history=mixing_history,
                 mixing_kerker=mixing_kerker, eigensolver=eigensolver,
                 precond=precond, reuse_wavefunctions=reuse_wavefunctions)
        )
        # Opt-in D3(BJ) dispersion, mirroring inputs.DispersionParams: True →
        # enabled with defaults (functional = the SCF xc); a dict overrides any
        # of functional/cutoff/cn_cutoff/s6/s8/a1/a2; None/False → off. Stored on
        # self.parameters so toggling it invalidates ASE's cached results.
        self._dispersion: dict[str, Any] | None
        if dispersion in (None, False):
            self._dispersion = None
        elif dispersion is True:
            self._dispersion = {}
        elif isinstance(dispersion, dict):
            self._dispersion = dict(dispersion)
        else:
            raise ValueError(
                "dispersion must be True/False or a dict of D3(BJ)/D4(BJ) "
                "overrides (method, functional, cutoff, cn_cutoff, s6, s8, a1, "
                "a2; D4 also accepts charge)")
        if self._dispersion is not None:
            method = str(self._dispersion.get("method", "d3")).lower()
            if method not in ("d3", "d4"):
                raise ValueError(
                    f"dispersion method must be 'd3' or 'd4', got {method!r}")
        self.parameters["dispersion"] = self._dispersion
        # DFT+U manifolds normalized to (element, l, U, J) tuples and kept
        # element-keyed; resolved to species indices per calculate() (where the
        # current atoms' symbol ordering is known). Accepts objects with
        # .species/.l/.u/.j (inputs.HubbardManifoldSpec) or plain dicts.
        def _norm(m: object) -> tuple[str, int, float, float]:
            if isinstance(m, dict):
                sp, ll, u, j = m.get("species"), m.get("l"), m.get("u"), m.get("j", 0.0)
            else:
                sp = getattr(m, "species", None)
                ll = getattr(m, "l", None)
                u = getattr(m, "u", None)
                j = getattr(m, "j", 0.0)
            # both branches yield gradually-typed (dict/attr-lookup) values;
            # cast documents that str()/int()/float() below are exactly the
            # runtime validation doing the real type-narrowing work.
            return (str(cast(Any, sp)), int(cast(Any, ll)),
                    float(cast(Any, u)), float(cast(Any, j)))
        self._hubbard: list[tuple[str, int, float, float]] | None = (
            None if not hubbard else [_norm(m) for m in hubbard])
        self.parameters["hubbard"] = (None if self._hubbard is None
                                      else tuple(self._hubbard))
        self._pseudo_paths: dict[str, str] = dict(pseudopotentials)
        self._upf_cache: dict[str, UPFData | PAWData] = {}
        self._system: System | USPPSystem | None = None
        self._system_key: tuple[tuple[float, ...], tuple[str, ...]] | None = None
        self._device = device
        self._compile_xc = compile_xc
        self._verbose = verbose
        self.last_result: SCFResult | USPPResult | None = None
        self._scf_state: _StateKey | None = None  # the geometry/params the
        # stored SCF state was built at
        self._cached_results: dict[str, Any] = {}  # full results dict paired
        # with _scf_state
        self._warm_start_remaps = 0  # count of grid-shape-change density remaps

    def _make_xc(self, nspin: int = 1) -> XCFunctional | SpinXC:
        """Instantiate the XC functional, opting into the compiled real-valued
        energy_density path when compile_xc is set. nspin=2 selects the
        spin-polarized (LSDA / SpinPBE / SpinR2SCAN) variant so the collinear
        SCF, forces, and stress all see (ρ↑, ρ↓).
        The functional degrades to eager on any toolchain gap, so this is safe
        to leave on. It pays only for XC-heavy, CPU-bound work (PAW one-center
        loop, response HVPs, learned-XC training), not a plain FFT-bound SCF."""
        xc = (_SPIN_XC if nspin == 2 else _XC)[self.parameters["xc"]]()
        if self._compile_xc:
            xc.enable_compile()
        return xc

    def _resolve_spin(
        self, atoms: Atoms, system: System | USPPSystem
    ) -> tuple[int, list[float] | None, float | None]:
        """Effective (nspin, start_mag, tot_magnetization) from ASE's initial
        magnetic moments. nspin=2 whenever the user asked for it (nspin=2) or an
        atom carries a nonzero initial moment; the per-atom μ_B moments become
        the polarization fractions scf's start_mag expects (m = μ / Z_valence),
        and their sum fixes the spin moment for a no-smearing (fixed-spin-moment)
        run — with a smearing the shared Fermi level finds the moment instead."""
        magmoms = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
        if self.parameters["nspin"] != 2 and not np.any(magmoms != 0.0):
            return 1, None, None
        z_val = system.charges.detach().cpu().numpy().astype(float)
        start_mag = [m / z if z > 0 else 0.0
                     for m, z in zip(magmoms, z_val, strict=True)]
        tot_mag = self.parameters["tot_magnetization"]
        if tot_mag is None and self.parameters["smearing"] == "none":
            tot_mag = float(magmoms.sum())
        return 2, start_mag, None if tot_mag is None else float(tot_mag)

    def _resolve_hubbard(self, atoms: Atoms) -> list[HubbardManifold] | None:
        """DFT+U manifolds as ``core.hubbard.HubbardManifold`` with the element
        symbols resolved to species indices for the current atoms, or None. The
        index matches the SCF setup's ``sorted(set(symbols))`` species ordering."""
        if self._hubbard is None:
            return None
        from gradwave.core.hubbard import HubbardManifold

        species = sorted(set(atoms.get_chemical_symbols()))
        out = []
        for sp, l, u, j in self._hubbard:
            if sp not in species:
                raise ValueError(
                    f"DFT+U manifold species {sp!r} is not in the structure")
            out.append(HubbardManifold(species=species.index(sp), l=l, u=u, j=j))
        return out

    def _upf(self, symbol: str) -> UPFData | PAWData:
        # shared loader (NC / USPP-PAW detection) lives in api; keep the
        # per-instance cache so a relaxation parses each pseudo once
        if symbol not in self._upf_cache:
            from gradwave.api import _load_upf

            self._upf_cache[symbol] = _load_upf(self._pseudo_paths[symbol])
        return self._upf_cache[symbol]

    def _is_uspp(self, species: Iterable[str]) -> bool:
        from gradwave.api import _is_uspp

        return _is_uspp([self._upf(s) for s in species])

    def _apply_dispersion(self, system: System | USPPSystem) -> None:
        """Fold the opt-in D3(BJ)/D4(BJ) correction into the reported energy,
        forces, and stress — the calculator analog of ``api._apply_dispersion``.

        The energy is folded into ``res.energies.dispersion`` so ``free_energy``
        (which the calculator reports) carries it; the dispersion forces/stress
        are added on top of the SCF Hellmann–Feynman terms. Both are geometric,
        autograd contributions from ``postscf/dispersion.py`` (D3) or
        ``postscf/dispersion_d4.py`` (D4, charge-dependent EEQ C6 with a periodic
        Ewald charge model); no reimplementation here. ``method`` (``'d3'`` by
        default, or ``'d4'``) selects the model; D4 also reads a ``charge`` knob.
        Degrades to a no-op when the element set is uncovered or no BJ preset
        exists for the functional, matching the api's ``{'available': False}``."""
        if self._dispersion is None:
            return
        d = self._dispersion
        method = str(d.get("method", "d3")).lower()
        functional = d.get("functional") or self.parameters["xc"]
        positions = system.positions.detach().to(RDTYPE)
        cell = np.asarray(_fft_grid(system).cell, dtype=np.float64)
        z = [int(v) for v in self.atoms.get_atomic_numbers()]
        # each branch calls its OWN energy/forces/stress trio right after
        # building the matching Config type (rather than joining afterward,
        # which would leave a same-named-but-different-signature callable
        # unioned with an incompatible D3Config | D4Config argument type).
        try:
            if method == "d4":
                from gradwave.postscf.dispersion_d4 import (
                    D4Config,
                    dispersion_energy,
                    dispersion_forces,
                    dispersion_stress,
                )
                cfg_d4 = D4Config.resolve(
                    functional, charge=float(d.get("charge", 0.0)),
                    cutoff_ang=d.get("cutoff", 21.2),
                    cn_cutoff_ang=d.get("cn_cutoff", 10.6),
                    s6=d.get("s6"), s8=d.get("s8"), a1=d.get("a1"), a2=d.get("a2"),
                )
                cell_t = torch.as_tensor(cell, dtype=RDTYPE, device=positions.device)
                e = dispersion_energy(positions, cell_t, z, cfg_d4)
                f = dispersion_forces(positions, cell, z, cfg_d4)
                sig = (dispersion_stress(positions, cell, z, cfg_d4)
                       if "stress" in self.results else None)
            else:
                from gradwave.postscf.dispersion import (
                    D3Config,
                    dispersion_energy,
                    dispersion_forces,
                    dispersion_stress,
                )
                cfg_d3 = D3Config.resolve(
                    functional,
                    cutoff_ang=d.get("cutoff", 21.2),
                    cn_cutoff_ang=d.get("cn_cutoff", 10.6),
                    s6=d.get("s6"), s8=d.get("s8"), a1=d.get("a1"), a2=d.get("a2"),
                )
                cell_t = torch.as_tensor(cell, dtype=RDTYPE, device=positions.device)
                e = dispersion_energy(positions, cell_t, z, cfg_d3)
                f = dispersion_forces(positions, cell, z, cfg_d3)
                sig = (dispersion_stress(positions, cell, z, cfg_d3)
                       if "stress" in self.results else None)
        except (ValueError, NotImplementedError):
            return

        res = self.last_result
        assert res is not None  # calculate() always sets it before dispersion runs
        res.energies.dispersion = (
            res.energies.dispersion + e.detach().to(positions.device))
        self.results["energy"] = float(res.energies.free_energy)
        self.results["free_energy"] = float(res.energies.free_energy)
        self.results["forces"] = self.results["forces"] + f.cpu().numpy()
        if sig is not None and "stress" in self.results:
            s = sig.cpu().numpy()
            self.results["stress"] = self.results["stress"] + np.array([
                s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])

    def _get_system(self, atoms: Atoms) -> System:
        """With use_symmetry off, positions-only updates reuse the cached
        System (its tables are phase-free; positions enter through structure
        factors built per solve). With use_symmetry on the IBZ k-mesh and the
        density symmetrizer are position-dependent — a moved atom can break a
        spacegroup op — so the system is rebuilt every call, letting spglib
        re-detect the current configuration's group. The expensive rebuilds
        (RhoSymmetrizer index/phase maps) are memoized inside the setup layer
        keyed on (grid shape, op set) and reused whenever the op set is
        unchanged (the #121 memoization). Reusing the position-swapped cache
        here instead would keep the previous geometry's IBZ, shifting warm-start
        energies by up to ~0.3 eV once a move breaks a symmetry op (#128)."""
        symbols = atoms.get_chemical_symbols()
        species = sorted(set(symbols))
        # DFT+U needs the full spatial BZ: an IBZ-folded mesh under-counts the
        # correlated occupation matrix (see api.build_system). Force symmetry off
        # whenever +U is on, matching the api path and the validated regime.
        use_sym = self.parameters["use_symmetry"] and self._hubbard is None
        key = (tuple(np.round(atoms.cell.array, 12).ravel()), tuple(symbols))
        if (not use_sym and self._system is not None
                and key == self._system_key):
            # this cache is shared with _get_uspp_system; _get_system is only
            # ever called on the norm-conserving path, so a hit here is
            # always a previously-cached System, never a USPPSystem.
            assert isinstance(self._system, System)
            return dataclasses.replace(
                self._system,
                positions=torch.as_tensor(atoms.get_positions(), dtype=RDTYPE).to(
                    self._system.positions.device),
            )
        system = setup_system(
            cell=atoms.cell.array,
            positions=atoms.get_positions(),
            species_of_atom=[species.index(s) for s in symbols],
            # this is the norm-conserving path (calculate() already routed on
            # _is_uspp before reaching here), so every pseudo is a UPFData.
            upfs=[cast("UPFData", self._upf(s)) for s in species],
            ecut=self.parameters["ecut"],
            kmesh=self.parameters["kpts"],
            kshift=self.parameters["kshift"],
            nbands=self.parameters["nbands"],
            use_symmetry=use_sym,
        ).to(self._device)
        self._system, self._system_key = system, key
        return system

    def _warm_start(
        self, system: System | USPPSystem, nspin: int = 1
    ) -> SCFResult | USPPResult | dict[str, Any] | None:
        """Seed the next solve from the previous converged state (same
        FFT grid — positions-only moves during a relaxation/MD qualify),
        with QE-style atomic extrapolation when the atoms moved: the
        superposition-of-atoms part of ρ travels with the atoms, the
        bonding remainder is reused. Plain reuse is nearly worthless
        under motion (measured: 8 vs 9 iterations for a 6 mÅ move —
        the seed error is first-order in displacement) while the
        extrapolated seed keeps the ~2-iteration warm restart.

        Restricted to nspin=1: the extrapolated seed below builds a single
        (total-density) channel, so a collinear-spin run cold-starts rather
        than reuse a mismatched seed.

        When the FFT grid resizes (variable-cell relaxation strains the cell,
        so the ecut-fixed box dims change) the previous density no longer lives
        on this grid. Rather than cold-start, remap it in G-space onto the new
        grid (``_remap_density_to_grid``) and warm-start from that — worth 4-8×
        fewer SCF iterations per variable-cell step."""
        from gradwave.scf.guess import sad_density
        from gradwave.scf.uspp import USPPSystem

        prev = self.last_result
        if prev is None or nspin != 1 or getattr(prev, "nspin", 1) != 1:
            return None
        # reuse_wavefunctions gates ONLY the orbital seed; the density (and USPP
        # becsum) warm start is unconditional. False → density-only warm start
        # (cold orbital seed), the pre-reuse behavior.
        reuse = self.parameters["reuse_wavefunctions"]
        is_uspp = prev.formalism == "uspp"
        # USPPResult.system is declared `object` (scf/results.py keeps that
        # module import-light); is_uspp above already reflects exactly this
        # runtime fact, so the narrowing here is not a new assumption.
        prev_sys = prev.system
        assert isinstance(prev_sys, (System, USPPSystem))
        if tuple(_fft_grid(prev_sys).shape) != tuple(_fft_grid(system).shape):
            self._warm_start_remaps += 1
            rho = _remap_density_to_grid(
                prev.rho.detach(), _fft_grid(prev_sys), _fft_grid(system),
                system.n_electrons)
            # remap the eigenvectors onto the new G-spheres too (Miller-index
            # match, zero-fill new high-G components) so the Davidson seed
            # survives the grid change; None if the k-set is incompatible.
            coeffs = (_remap_coeffs_to_spheres(
                prev.coeffs, prev_sys.spheres, system.spheres)
                if reuse else None)
            if is_uspp:
                # becsum (rho_ij_atoms) is per-atom and grid-independent, so it
                # rides along; swap in the new system so warm_start_densities
                # sees a matching grid, and carry the remapped orbitals.
                return dataclasses.replace(prev, rho=rho, system=system,
                                           coeffs=coeffs)
            return {"system": system, "nspin": 1, "rho": rho,
                    "rho_spin": None, "coeffs": coeffs}
        pos_new = system.positions
        pos_old = prev_sys.positions.to(pos_new.device)
        if float((pos_new - pos_old).abs().max()) < 1e-12:
            return prev if reuse else dataclasses.replace(prev, coeffs=None)
        tabs = prev_sys.paws if isinstance(prev_sys, USPPSystem) else prev_sys.upfs
        soa = prev_sys.species_of_atom
        ne = prev_sys.n_electrons
        delta = (sad_density(_fft_grid(system), pos_new, soa, tabs, ne)
                 - sad_density(_fft_grid(system), pos_old, soa, tabs, ne))
        rho = prev.rho.detach() + delta
        if is_uspp:
            # becsum is per-atom and rides along as-is; orbitals live on the same
            # G-spheres (positions-only move), so they seed the Davidson directly
            return dataclasses.replace(prev, rho=rho,
                                       coeffs=prev.coeffs if reuse else None)
        return {"system": prev_sys, "nspin": 1, "rho": rho, "rho_spin": None,
                "coeffs": prev.coeffs if reuse else None}

    def _state_key(self, atoms: Atoms) -> _StateKey:
        """Exact-match key for the (geometry, parameters) state an SCF result
        is valid at: positions/cell/pbc/symbols/initial moments plus every
        calculator parameter (dispersion rides on self.parameters) and the
        device. Exact bytes — any genuine move changes them."""
        return (
            atoms.get_positions().tobytes(),
            atoms.cell.array.tobytes(),
            atoms.get_pbc().tobytes(),
            tuple(atoms.get_chemical_symbols()),
            np.asarray(atoms.get_initial_magnetic_moments(),
                       dtype=float).tobytes(),
            repr(sorted(self.parameters.items(), key=lambda kv: kv[0])),
            self._device,
        )

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        key = self._state_key(self.atoms)
        if self.last_result is not None and key == self._scf_state:
            # Unchanged geometry/parameters: the stored SCF state is still
            # valid. ASE re-enters calculate() only when a property is missing
            # from self.results (its reset() clears them); the single pass
            # below populates every property, so restoring the paired dict
            # answers any request without re-running the SCF.
            self.results.update(self._cached_results)
            return
        symbols = self.atoms.get_chemical_symbols()
        if self._is_uspp(sorted(set(symbols))):
            self._calculate_uspp()
        else:
            self._calculate_nc()
        self._scf_state = key
        self._cached_results = dict(self.results)

    def _calculate_nc(self) -> None:
        """Norm-conserving route: one SCF, then energy, forces, and (for a
        periodic cell) stress in the same pass."""
        p = self.parameters
        system = self._get_system(self.atoms)
        # collinear-spin channel from the atoms' initial moments (nspin=1 when
        # unmagnetized); start_mag/tot_magnetization seed and (optionally) pin M
        nspin, start_mag, tot_mag = self._resolve_spin(self.atoms, system)
        xc = self._make_xc(nspin)
        # NC scf takes an int mixing_history (None isn't accepted); omit it so
        # the solver's own default stands when the user left it unset
        mix_kw = ({} if p["mixing_history"] is None
                  else {"mixing_history": p["mixing_history"]})
        manifolds = self._resolve_hubbard(self.atoms)
        # scf()/forces() declare xc: XCFunctional (nspin=1 only in their own
        # signature); the nspin=2 SpinXC variants both share XCFunctional's
        # interface (CompilableXC, torch.nn.Module) and are the SCF's own
        # long-standing collinear-spin contract (see _make_xc / the SPIN_XC
        # registries in api.py) — the cast documents that, not a workaround.
        res = scf(
            system, cast("XCFunctional", xc),
            smearing=p["smearing"], width=p["width"],
            max_iter=p["max_iter"], etol=p["etol"], rhotol=p["rhotol"],
            mixing_alpha=p["mixing_alpha"], kerker=p["mixing_kerker"],
            diago_tol=p["diago_tol"], verbose=self._verbose,
            eigensolver=p["eigensolver"], precond=p["precond"],
            nspin=nspin, start_mag=start_mag, tot_magnetization=tot_mag,
            hubbard=manifolds,
            # _warm_start is shared with the USPP path, so its declared return
            # spans both; scf() only ever receives the NC-shaped subset here
            # (is_uspp gates it internally) — the cast documents that, not a
            # workaround.
            start_from=cast(Any, self._warm_start(system, nspin)), **mix_kw,
        )
        if not res.converged:
            raise RuntimeError("gradwave SCF did not converge")
        self.last_result = res
        self.results["energy"] = float(res.energies.free_energy)  # consistent forces
        self.results["free_energy"] = float(res.energies.free_energy)
        if nspin == 2:
            # total spin moment M = N↑−N↓ (μ_B); cheap scalar off the result
            self.results["magmom"] = float(res.mag_total)
        # xc is used only when the system carries an NLCC core charge (the
        # core-correction force term, spin-resolved for nspin=2); ignored for
        # valence-only species. Reuse the (spin-matched) functional above.
        f = hf_forces(res, xc=cast("XCFunctional", xc))
        if manifolds is not None:
            # +U Hellmann-Feynman force through the atomic-orbital projector
            # phases, additive to the KB/local/Ewald force (nspin 1 and 2).
            from gradwave.postscf.forces import hubbard_force

            f = f + hubbard_force(res, manifolds)
        self.results["forces"] = f.cpu().numpy()
        if self.atoms.pbc.all():
            # unconditional (not gated on the requested properties): one
            # autograd backward over the stored SCF state, far cheaper than
            # the warm SCF + forces a re-entrant stress request used to cost
            from gradwave.postscf.stress import stress as hf_stress

            sig = hf_stress(res, xc, manifolds=manifolds).cpu().numpy()
            # ASE Voigt order (xx, yy, zz, yz, xz, xy); ASE's convention is
            # +(1/Ω)∂E/∂ε, same as ours
            self.results["stress"] = np.array([
                sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1],
            ])
        self._apply_dispersion(system)

    def _get_uspp_system(self, atoms: Atoms) -> USPPSystem:
        """With use_symmetry off, positions-only updates reuse the cached
        USPPSystem (its tables are phase-free; positions enter through
        structure factors built per solve). With use_symmetry on the density
        symmetrizer and the IBZ k-mesh are position-dependent, so the system
        is rebuilt every call — spglib then finds the current configuration's
        group (dropping to time-reversal-only when a move breaks it). The
        expensive per-step pieces of that rebuild (RhoSymmetrizer maps,
        becsum D blocks, aug form-factor tables) are memoized inside the
        setup layer and reused whenever the op set / G set is unchanged."""
        from gradwave.scf.uspp import USPPSystem, setup_uspp

        p = self.parameters
        symbols = atoms.get_chemical_symbols()
        species = sorted(set(symbols))
        # DFT+U needs the full spatial BZ (an IBZ-folded mesh under-counts the
        # correlated occupation matrix), so force symmetry off when +U is on —
        # matching the NC ground-state path and api.build_system.
        use_sym = p["use_symmetry"] and self._hubbard is None
        key = (tuple(np.round(atoms.cell.array, 12).ravel()), tuple(symbols))
        if (not use_sym and self._system is not None
                and key == self._system_key):
            # this cache is shared with _get_system; _get_uspp_system is only
            # ever called on the USPP/PAW path, so a hit here is always a
            # previously-cached USPPSystem, never a plain System.
            assert isinstance(self._system, USPPSystem)
            return dataclasses.replace(
                self._system,
                positions=torch.as_tensor(atoms.get_positions(), dtype=RDTYPE).to(
                    self._system.positions.device),
            )
        system = setup_uspp(
            atoms.cell.array, atoms.get_positions(),
            [species.index(s) for s in symbols],
            # this is the USPP/PAW path (calculate() already routed on
            # _is_uspp before reaching here), so every pseudo is a PAWData.
            [cast("PAWData", self._upf(s)) for s in species],
            ecut=p["ecut"], kmesh=p["kpts"], nbands=p["nbands"],
            ecutrho=p.get("ecutrho"), use_symmetry=use_sym,
        ).to(self._device)
        self._system, self._system_key = system, key
        return system

    def _calculate_uspp(self) -> None:
        """USPP/PAW route (nspin 1 or 2): one SCF, then energy, forces, and (for
        a periodic cell) stress in the same pass."""
        from gradwave.scf.uspp import scf_uspp

        p = self.parameters
        system = self._get_uspp_system(self.atoms)
        # collinear-spin channel from the atoms' initial moments (nspin=1 when
        # unmagnetized), exactly as the NC path resolves it. scf_uspp,
        # forces_uspp, and stress_uspp all thread nspin=2 (per-spin becsum /
        # rho / occupation channels), so the calculator just passes it through.
        nspin, start_mag, _tot_mag = self._resolve_spin(self.atoms, system)
        if p["eigensolver"] != "davidson":
            raise ValueError(
                "eigensolver='chebyshev' is norm-conserving only; the USPP/PAW "
                "generalized S-metric problem is not supported yet")
        # DFT+U (Dudarev) is wired end-to-end for USPP/PAW: the S-metric SCF
        # (scf_uspp), the +U force (forces_uspp reads it off the result), and
        # now the +U stress (stress_uspp adds the strained S-dressed occupation
        # term), so relaxation / EOS / elastic constants run here too.
        manifolds = self._resolve_hubbard(self.atoms)
        xc = self._make_xc(nspin)
        # scf_uspp has no tot_magnetization pin (no fixed-spin-moment mode); the
        # start_mag seed and a shared Fermi level find the moment.
        # scf_uspp takes mixing_history=None natively (per-scheme default)
        res = scf_uspp(system, xc, nspin=nspin, start_mag=start_mag,
                       smearing=p["smearing"],
                       width=p["width"], max_iter=p["max_iter"],
                       etol=p["etol"], rhotol=p["rhotol"],
                       diago_tol=p["diago_tol"], mixing_scheme=p["mixing_scheme"],
                       mixing_alpha=p["mixing_alpha"],
                       mixing_history=p["mixing_history"],
                       mixing_kerker=p["mixing_kerker"], precond=p["precond"],
                       hubbard=manifolds, verbose=self._verbose,
                       # _warm_start is shared with the NC path; scf_uspp only
                       # ever receives the USPP-shaped subset here (is_uspp
                       # gates it internally) — the cast documents that.
                       start_from=cast(Any, self._warm_start(system, nspin)))
        if not res.converged:
            raise RuntimeError("gradwave USPP SCF did not converge")
        self.last_result = res
        self.results["energy"] = float(res.energies.free_energy)
        self.results["free_energy"] = float(res.energies.free_energy)
        if nspin == 2:
            # always set (never None) on the nspin=2 path; results.py keeps
            # the field Optional for the shared nspin=1 default.
            self.results["magmom"] = float(cast(float, res.mag_total))
        from gradwave.postscf.paw_forces import forces_uspp

        # forces_uspp/stress_uspp declare res: dict (postscf/paw_forces.py,
        # paw_stress.py keep those modules import-light) but actually consume
        # USPPResult via the _DictBridge duck-typing (scf/results.py) — same
        # established pattern as the hf_forces cast above, not a workaround.
        self.results["forces"] = forces_uspp(cast("dict[str, Any]", res), xc).cpu().numpy()
        if self.atoms.pbc.all():
            from gradwave.postscf.paw_stress import stress_uspp

            sig = stress_uspp(cast("dict[str, Any]", res), xc).cpu().numpy()
            self.results["stress"] = np.array([
                sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1],
            ])
        self._apply_dispersion(system)
