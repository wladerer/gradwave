"""Python API mirroring the YAML input (Layer C).

run() executes a task and writes three files into the output directory:
<task>.json (the machine-readable summary — the parsing target),
<task>.out (the human-readable report) and, for SCF tasks, checkpoint.pt
(restartable state, wavefunctions excluded unless requested). The same
summary dict feeds gradwave.analysis for pandas/matplotlib work.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE, SpinXC
from gradwave.inputs import Input, InputError, VolumetricParams

if TYPE_CHECKING:
    from ase import Atoms

    from gradwave.calculator import GradWave
    from gradwave.core.hubbard import HubbardManifold
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.distributed import DistKContext
    from gradwave.grids import FFTGrid
    from gradwave.postscf.magnetism import MagneticReport
    from gradwave.pseudo.upf import UPFData
    from gradwave.pseudo.upf_paw import PAWData
    from gradwave.scf.loop import SCFResult, System
    from gradwave.scf.noncollinear import NCResult
    from gradwave.scf.results import USPPNCResult, USPPResult
    from gradwave.scf.uspp_setup import USPPSystem

    # every SCF driver's converged-state result, the type `_get()`/
    # `build_summary()` duck-type over via getattr (field sets differ; see
    # `_get`'s docstring) rather than isinstance branching.
    SCFLike = SCFResult | NCResult | USPPResult | USPPNCResult

XC_REGISTRY: dict[str, type[XCFunctional]] = {"lda": LDA_PW92, "pbe": PBE,
                                              "r2scan": R2SCAN}
SPIN_XC_REGISTRY: dict[str, type[SpinXC]] = {"lda": LSDA_PW92, "pbe": SpinPBE,
                                             "r2scan": SpinR2SCAN}
_OCC_TOL = 1e-6
# the collinear/NC solvers build a fixed-length Pulay history and need an int
# (None is not accepted, unlike the USPP path); this names the default the api
# forwards, matching the per-scheme default those solvers use internally
_DEFAULT_MIXING_HISTORY = 8
# UPFs are static for a run; cache by path so build_summary / run_scf /
# _error_estimate_block / _parameters_block parse each pseudo once, not 3-4×
_UPF_CACHE: dict[str, UPFData | PAWData] = {}

logger = logging.getLogger(__name__)


def _mixing_scheme(inp: Input) -> str | None:
    """The mixing scheme the api forwards to the SCF driver. The input default
    ``auto`` maps to None so each formalism's own resolver picks its
    evidence-backed default (johnson for USPP/PAW and for collinear-spin nspin=2
    norm-conserving, pulay otherwise); an explicit pulay|broyden|johnson passes
    through unchanged. Shared by run_scf and _build_relax_calc so both entry
    points defer to the same resolvers. See scf.loop._resolve_mixing_scheme and
    scf.uspp_loop._resolve_uspp_mixing_scheme."""
    scheme = inp.scf.mixing.scheme
    return None if scheme == "auto" else scheme


def _load_upf(path: str | Path) -> UPFData | PAWData:
    """Parse a UPF of either family (NC via upf.py, USPP/PAW via
    upf_paw.py — same detection the ASE calculator uses), cached by path."""
    key = str(path)
    cached = _UPF_CACHE.get(key)
    if cached is not None:
        return cached
    from gradwave.pseudo.upf import parse_upf

    try:
        upf = parse_upf(path)
    except ValueError as err:
        if "norm-conserving" not in str(err):
            raise
        from gradwave.pseudo.upf_paw import parse_upf_paw

        upf = parse_upf_paw(path)
    if logger.isEnabledFor(logging.DEBUG):
        # projection orbitals (NC: pswfc, US/PAW: chi) gate COHP/PDOS analysis —
        # SG15 ONCV fixtures ship none, PseudoDojo/PAW do
        orbitals = getattr(upf, "pswfc", None)
        if orbitals is None:
            orbitals = getattr(upf, "chi", [])
        logger.debug(
            "loaded pseudo %s: %s element=%s, n_proj=%s, projection_orbitals=%d, "
            "core_correction=%s", key, type(upf).__name__,
            getattr(upf, "element", "?"), getattr(upf, "n_proj", "?"),
            len(orbitals), getattr(upf, "core_correction", "?"))
    _UPF_CACHE[key] = upf
    return upf


def _species_upfs(
    inp: Input,
) -> tuple[list[str], list[UPFData | PAWData], list[int]]:
    symbols = inp.atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    upfs = [_load_upf(inp.pseudo_dir / inp.pseudo_map[s]) for s in species]
    species_of_atom = [species.index(s) for s in symbols]
    return species, upfs, species_of_atom


def _hubbard_manifolds(inp: Input) -> list[HubbardManifold] | None:
    """The input's `hubbard` block as a list of ``core.hubbard.HubbardManifold``
    (species RESOLVED to the setup's integer index), or None when +U is off. The
    index is ``sorted(set(symbols)).index(element)`` — the same species ordering
    ``build_system`` / ``_species_upfs`` use, so the manifolds line up with the
    system's ``species_of_atom``/``upfs``."""
    if not inp.hubbard.enabled:
        return None
    from gradwave.core.hubbard import HubbardManifold

    species = sorted(set(inp.atoms.get_chemical_symbols()))
    return [HubbardManifold(species=species.index(m.species), l=m.l, u=m.u, j=m.j)
            for m in inp.hubbard.manifolds]


def _is_uspp(upfs: Iterable[UPFData | PAWData]) -> bool:
    from gradwave.pseudo.upf_paw import PAWData

    kinds = {isinstance(u, PAWData) for u in upfs}
    if len(kinds) > 1:
        raise ValueError("mixing NC and USPP/PAW pseudopotentials is not "
                         "supported")
    return kinds.pop()


def _as_paws(upfs: list[UPFData | PAWData]) -> list[PAWData]:
    """`_is_uspp(upfs)` already guarantees every element is a `PAWData` (it
    raises on a mixed NC/USPP set); this documents that fact for the type
    checker at each USPP-branch call into `setup_uspp`."""
    return cast("list[PAWData]", upfs)


def _as_upfs(upfs: list[UPFData | PAWData]) -> list[UPFData]:
    """The norm-conserving analogue of `_as_paws`."""
    return cast("list[UPFData]", upfs)


def _fft_grid(system: System | USPPSystem) -> FFTGrid:
    """Both `System.grid` and `USPPSystem.grid` are `FFTGrid`; this just names
    the shared field for callers that hold the `System | USPPSystem` union."""
    return system.grid


def build_system(inp: Input) -> System | USPPSystem:
    """The Layer-B system for this input, NC or USPP/PAW by UPF kind."""
    species, upfs, species_of_atom = _species_upfs(inp)
    # DFT+U builds the correlated occupation matrix n^I_{mm'} from only the
    # k-points in the mesh. An IBZ-folded mesh under-counts it: the manifold
    # projector's m-components mix under the star's rotations, so a single IBZ
    # representative is not the star-averaged matrix and Tr[n(1−n)] (the Dudarev
    # energy) comes out wrong. The occupation matrix would need star-symmetrizing;
    # until then +U runs on the full spatial BZ (the regime the +U machinery is
    # validated in — the NiO/Si references use no IBZ reduction). Time reversal
    # is kept (n at −k is n*, and |n_{mm'}|² is TR-invariant, so E_U is exact).
    hubbard = inp.hubbard.enabled
    if _is_uspp(upfs):
        from gradwave.scf.uspp import setup_uspp

        return setup_uspp(
            inp.atoms.cell.array, inp.atoms.get_positions(), species_of_atom,
            _as_paws(upfs), ecut=inp.ecut, kmesh=inp.kpoints.mesh,
            ecutrho=inp.ecutrho, nbands=inp.nbands,
            use_symmetry=inp.symmetry and not hubbard,
        )
    from gradwave.scf.loop import setup_system

    # a hybrid's multi-k Fock sum runs over the WHOLE BZ (q = k−k′), so a
    # symmetry-folded or time-reversal-reduced mesh is an invalid quadrature —
    # force the full BZ (at Γ this is a no-op). Otherwise: a magnetic spinor
    # breaks k ≡ −k (TR flips m⃗); a nonmagnetic spinor (SOC only) keeps Kramers.
    hybrid = inp.hybrid.enabled
    return setup_system(
        cell=inp.atoms.cell.array,
        positions=inp.atoms.get_positions(),
        species_of_atom=species_of_atom,
        upfs=_as_upfs(upfs),
        ecut=inp.ecut,
        kmesh=inp.kpoints.mesh,
        kshift=inp.kpoints.shift,
        nbands=inp.nbands,
        use_symmetry=inp.symmetry and not hybrid and not hubbard,
        time_reversal=(not hybrid
                       and not (inp.noncollinear and not inp.nonmagnetic)),
    )


def _spin_setup(inp: Input) -> tuple[SpinXC, list[float]]:
    xc = SPIN_XC_REGISTRY[inp.xc]()
    symbols = inp.atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    mags = [float((inp.start_mag or {}).get(s, 0.5)) for s in species]
    return xc, mags


def run_scf(
    inp: Input,
    system: System | USPPSystem | None = None,
    verbose: bool = True,
    start_from: Any = None,
) -> SCFResult | NCResult | USPPResult:
    """Run the SCF for either formalism. Returns the native result
    (SCFResult for NC, USPPResult for USPP/PAW).

    ``start_from`` warm-starts the density from a previous converged result
    (the volume-scan chain in ``run_eos`` uses it); when None the checkpoint in
    ``inp.restart`` is used instead, if any."""
    _species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    system = system or build_system(inp)
    if inp.device != "cpu":
        system = system.to(inp.device)
    dist_ctx: DistKContext | None = None
    if inp.distributed:
        if inp.hybrid.enabled or inp.noncollinear:
            raise NotImplementedError(
                "distributed (k-point-sharded) SCF is implemented for the "
                "norm-conserving and USPP/PAW collinear paths (DFT+U "
                "included) — hybrid functionals and the noncollinear/SOC SCF "
                "are documented follow-ups (see docs/manual/distributed.md)"
            )
        from gradwave.distributed import init_from_env, shard_system, shard_uspp_system

        info = init_from_env()
        if info is not None:
            rank, world_size, group = info
            # symmetry (IBZ) is rejected inside the shard_* call for either
            # formalism (build with symmetry: false) — nothing to gate here.
            if uspp:
                system, dist_ctx = shard_uspp_system(
                    cast("USPPSystem", system), rank, world_size, group)
            else:
                system, dist_ctx = shard_system(
                    cast("System", system), rank, world_size, group)
            # every rank still computes the identical, correctly-reduced
            # result (see scf.loop.scf / scf.uspp_loop.scf_uspp dist_ctx
            # handling) — quiet every rank but 0 purely to avoid world_size-fold
            # duplicated chatter
            verbose = verbose and rank == 0
    if inp.hybrid.enabled:
        return _run_scf_hybrid(inp, system, verbose, start_from, uspp)
    if inp.noncollinear:
        return _run_scf_noncollinear(inp, system, verbose)
    if inp.nspin == 2:
        xc, mags = _spin_setup(inp)
    else:
        xc = XC_REGISTRY[inp.xc]()
        mags = None

    if start_from is None and inp.restart is not None:
        from gradwave.checkpoint import as_start_from, load_checkpoint

        start_from = as_start_from(load_checkpoint(inp.restart))

    kerker = inp.scf.mixing.kerker
    kerker = None if kerker == "auto" else bool(kerker)
    # dict(...) infers a value type from this mixed bag of kwargs (int | str |
    # float | bool | list[float] | None); explicit dict[str, Any] avoids that
    # union being checked, key-blind, against scf()/scf_uspp()'s **kwargs
    # below (the whole point of merging these here is to share one literal
    # between both call sites without duplicating every argument name twice).
    common: dict[str, Any] = dict(
        nspin=inp.nspin, start_mag=mags,
        smearing=inp.smearing.type, width=inp.smearing.width,
        max_iter=inp.scf.max_iter, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
        mixing_alpha=inp.scf.mixing.alpha, precond=inp.scf.mixing.precond,
        # auto → None so each formalism's resolver (scf / scf_uspp) picks its
        # evidence-backed default; both branches consume it identically here.
        mixing_scheme=_mixing_scheme(inp),
        diago_tol=inp.scf.diago_tol, verbose=verbose,
        # energy-metric convergence gate (opt-in via scf.convergence: "energy").
        # Both the NC scf() and USPP scf_uspp() take these kwargs; the default
        # "density" leaves the density gate bit-for-bit unchanged.
        energy_metric=(inp.scf.convergence == "energy"), entol=inp.scf.entol,
    )
    # DFT+U: the same manifold list feeds the NC and USPP/PAW SCF (both take a
    # `hubbard=` kwarg); species already resolved to the setup's integer index.
    manifolds = _hubbard_manifolds(inp)
    if manifolds is not None:
        common["hubbard"] = manifolds
        # +U convergence aids (both collinear drivers take these kwarg names);
        # defaults (β=1.0, ramp off) leave today's numbers bit-for-bit
        common["hub_occ_mix"] = inp.hubbard.occ_mix
        common["hub_u_ramp_iters"] = inp.hubbard.u_ramp_iters
    if inp.tot_magnetization is not None and not uspp:
        # fixed spin moment: a collinear nspin=2, no-smearing pin (see
        # scf/loop.py:_check_scf_args). The USPP path has no such argument.
        common["tot_magnetization"] = inp.tot_magnetization
    # uspp (from _is_uspp(upfs) above) already tells us which concrete type
    # `system` is — the two branches below just name that for the checker.
    if uspp:
        from gradwave.scf.uspp import scf_uspp

        # scf_uspp has no eigensolver knob — the S-metric problem is Davidson-only.
        # Reject a chebyshev request here, mirroring the calculator's rejection
        # (calculator.calculate), rather than silently ignoring it.
        if inp.scf.eigensolver != "davidson":
            raise InputError(
                "scf.eigensolver='chebyshev' is norm-conserving only; the "
                "USPP/PAW generalized S-metric problem is not supported yet")
        # history=None keeps the per-scheme default (johnson 12, else 8);
        # mixing_scheme rides in `common` (shared with the NC branch below)
        return scf_uspp(cast("USPPSystem", system), xc,
                        mixing_history=inp.scf.mixing.history,
                        mixing_kerker=kerker, start_from=start_from,
                        dist_ctx=dist_ctx, **common)
    from gradwave.scf.loop import scf

    return scf(cast("System", system), cast("XCFunctional", xc),
               kerker=kerker, start_from=start_from,
               eigensolver=inp.scf.eigensolver,
               mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
               dist_ctx=dist_ctx,
               **common)


def _run_scf_noncollinear(
    inp: Input, system: System | USPPSystem, verbose: bool
) -> NCResult:
    """A plain non-collinear (spinor) SCF for task: scf with
    noncollinear: true. Builds a NoncollinearXC from inp.xc (as run_magnetism
    does), seeds the atomic moments along +z from start_mag (or warm-starts
    from a checkpoint's m⃗ field), and returns the NCResult."""
    import torch

    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.scf.noncollinear import scf_noncollinear

    xc = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())

    if inp.nonmagnetic:
        # spin-orbit only: pin m⃗ ≡ 0 (no seed, no spurious moment)
        mag_vec_init = torch.zeros((len(inp.atoms), 3), dtype=torch.float64)
    elif inp.restart is not None:
        from gradwave.checkpoint import load_checkpoint, nc_mag_seed

        mag_vec_init = nc_mag_seed(load_checkpoint(inp.restart), system)
    else:
        # high-spin seed along +z; per-species magnitude from start_mag (a
        # moment fraction ~ scale on the SAD magnetization), default 1.0
        symbols = inp.atoms.get_chemical_symbols()
        z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        scales = [float((inp.start_mag or {}).get(s, 1.0)) for s in symbols]
        mag_vec_init = torch.stack([s * z for s in scales])  # (na, 3)

    # NC SCF requires a real smearing scheme (spinor bands hold one electron)
    smtype = inp.smearing.type if inp.smearing.type != "none" else "gaussian"
    # DFT+U: noncollinear/spin-orbit +U (the 2×2 spin-block occupation matrix,
    # core.hubbard) — same manifold list the collinear/USPP paths take.
    manifolds = _hubbard_manifolds(inp)
    # run_scf routes noncollinear: true to this (norm-conserving) driver
    # regardless of pseudopotential kind (see inputs.py's HubbardParams
    # docstring) — scf_noncollinear's own System-only annotation is the
    # narrow spot, not this call.
    return scf_noncollinear(
        cast("System", system), xc, mag_vec_init=mag_vec_init,
        smearing=smtype, width=inp.smearing.width,
        max_iter=inp.scf.max_iter, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
        # magnetic-channel spinor knobs (scf.magnetic); the spinor driver
        # resolves its (ρ, m⃗) mixing independently of scf.mixing.scheme, so
        # these are the only path to them from an input file.
        mag_mixer=inp.scf.magnetic.mixer,
        spin_precond=inp.scf.magnetic.spin_precond,
        mag_mixing_alpha=inp.scf.magnetic.mixing_alpha,
        mag_diago_schedule=inp.scf.magnetic.diago_schedule,
        # energy-metric convergence gate (opt-in via scf.convergence: "energy");
        # the default "density" leaves the residual gate bit-for-bit unchanged.
        energy_metric=(inp.scf.convergence == "energy"), entol=inp.scf.entol,
        diago_tol=inp.scf.diago_tol, verbose=verbose,
        nonmagnetic=inp.nonmagnetic,
        hubbard=manifolds,
    )


def _run_scf_hybrid(
    inp: Input,
    system: System | USPPSystem,
    verbose: bool,
    start_from: Any,
    uspp: bool,
) -> SCFResult:
    """Self-consistent PBE0-form / screened hybrid SCF (xc: pbe0 | hse).

    Dispatches to ``postscf.hybrid.hybrid_scf``, which scales the semilocal PBE
    exchange by (1−α) and adds α·E_x^Fock through the SCF ``fock`` hook (the
    multi-k build, full BZ). Norm-conserving, nspin=1 (the input layer already
    rejected the other combinations); the system was built full-BZ in
    ``build_system``. Returns the same SCFResult as a plain NC run."""
    if uspp:
        raise NotImplementedError(
            "hybrid functionals need norm-conserving pseudopotentials "
            "(the Fock hook builds on the norm-conserving exchange path)")
    from gradwave.postscf.hybrid import hybrid_scf

    if start_from is None and inp.restart is not None:
        from gradwave.checkpoint import as_start_from, load_checkpoint

        start_from = as_start_from(load_checkpoint(inp.restart))

    hy = inp.hybrid
    kerker = inp.scf.mixing.kerker
    kerker = None if kerker == "auto" else bool(kerker)
    omega = hy.omega if hy.mode != "full" else None
    # the uspp guard above rules out USPPSystem; hybrid_scf is norm-conserving only.
    return hybrid_scf(
        cast("System", system), alpha=hy.alpha, mode=hy.mode, omega=omega,
        smearing=inp.smearing.type, width=inp.smearing.width,
        max_iter=inp.scf.max_iter, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
        kerker=kerker, diago_tol=inp.scf.diago_tol,
        start_from=start_from, verbose=verbose)


def _get(res: SCFLike, key: str, default: Any = None) -> Any:
    """Attribute read with a default: every SCF driver returns a result
    dataclass, but the field sets differ (e.g. NCResult has no nspin)."""
    return getattr(res, key, default)


def _gap(eigenvalues: Any, occupations: Any, nspin: int) -> float | None:
    """HOMO-LUMO gap over all k and spins, None when any occupation is
    fractional (metals/smeared systems have no meaningful scalar gap)."""
    import numpy as np

    e = np.asarray(eigenvalues, dtype=float).reshape(-1)
    f = np.asarray(occupations, dtype=float).reshape(-1)
    f_full = 2.0 / nspin
    frac = (f > _OCC_TOL) & (np.abs(f - f_full) > _OCC_TOL)
    if frac.any() or not (f > _OCC_TOL).any() or not (f <= _OCC_TOL).any():
        return None
    homo = e[f > _OCC_TOL].max()
    lumo = e[f <= _OCC_TOL].min()
    return float(lumo - homo) if lumo > homo else 0.0


def _xc_label(inp: Input) -> str:
    """The functional name for the report: the hybrid label (pbe0/hse) when a
    hybrid is enabled, else the plain semilocal xc."""
    return inp.hybrid.name if inp.hybrid.enabled else inp.xc


def build_summary(res: SCFLike, inp: Input, task: str,
                  runtime_s: float | None = None,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The unified machine-readable summary for a task run."""
    from gradwave import __version__
    from gradwave.checkpoint import energies_eV_dict

    system = _get(res, "system")
    e = _get(res, "energies")
    nspin = int(_get(res, "nspin", 1) or 1)
    eig = _get(res, "eigenvalues")
    occ = _get(res, "occupations")
    species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    # a non-collinear NCResult carries an integrated moment vector but no
    # occupations (spinor bands each hold one electron); the gap/occupations
    # blocks degrade gracefully below.
    mag_vec = _get(res, "mag_vec")
    is_ncmag = _get(res, "formalism") == "noncollinear"

    import math

    def _finite(x: Any) -> float | None:
        # the first iteration records dE = inf; bare Infinity is not
        # valid strict JSON, so non-finite maps to null
        return None if x is None or not math.isfinite(x) else float(x)

    trace: list[dict[str, Any]] = [
        {"iter": h["iter"], "free_energy_eV": float(h["free_energy"]),
         "dE_eV": _finite(h["dE"]), "drho": float(h["res"]),
         **({"t_s": round(float(h["t"]), 3)} if "t" in h else {}),
         # spinor energy-metric gate (scf.convergence: energy): the per-iteration
         # estimate and its charge / longitudinal / transverse decomposition,
         # recorded only on the noncollinear driver's own history (it has no
         # SCFRecorder). Absent on the density-gate path and the other formalisms.
         **({"energy_metric_eV": float(h["energy_metric_eV"]),
             "energy_metric_charge_eV": float(h["energy_metric_charge_eV"]),
             "energy_metric_longitudinal_eV": float(h["energy_metric_longitudinal_eV"]),
             "energy_metric_transverse_eV": float(h["energy_metric_transverse_eV"])}
            if h.get("energy_metric_eV") is not None else {})}
        for h in (_get(res, "history") or [])
    ]
    scf_block: dict[str, Any] = {
        "converged": bool(_get(res, "converged")),
        "n_iter": int(_get(res, "n_iter")),
        "fermi_eV": None if _get(res, "fermi") is None
        else float(_get(res, "fermi")),
        "gap_eV": None if occ is None else _gap(eig.tolist(), occ.tolist(), nspin),
        "energies_eV": {
            **energies_eV_dict(e),
            "fock": float(getattr(e, "fock", 0.0)),
            "e0": float(0.5 * (e.total + e.free_energy)),
        },
        "free_energy_per_atom_eV": float(e.free_energy) / len(system.positions),
        "trace": trace,
    }
    if nspin == 2:
        scf_block["total_magnetization_muB"] = float(_get(res, "mag_total", 0.0))
        scf_block["absolute_magnetization_muB"] = float(_get(res, "mag_abs", 0.0))
    if is_ncmag:
        mv = [float(x) for x in mag_vec]
        scf_block["magnetization_vector_muB"] = mv
        scf_block["total_magnetization_muB"] = float((sum(x * x for x in mv)) ** 0.5)
        scf_block["absolute_magnetization_muB"] = float(_get(res, "mag_abs", 0.0))

    # convergence diagnostics: final residuals against the thresholds, the
    # geometric decay rate q of the energy residual (small q = fast, clean
    # convergence), and whether the run warm-started from a checkpoint
    _des = [abs(h["dE_eV"]) for h in trace
            if h.get("dE_eV") is not None and h["dE_eV"] != 0.0]
    _ratios = [_des[i] / _des[i - 1] for i in range(1, len(_des)) if _des[i - 1] > 0]
    q = None
    if _ratios:
        _tail = sorted(_ratios[-4:])
        q = float(_tail[len(_tail) // 2])  # median of the last few ratios
    _final = trace[-1] if trace else {}
    scf_block["convergence"] = {
        "criterion": inp.scf.convergence,
        "final_dE_eV": _final.get("dE_eV"),
        "final_drho": _final.get("drho"),
        "etol_eV": float(inp.scf.etol),
        "rhotol": float(inp.scf.rhotol),
        "ratio_q": q,
        "warm_started": inp.restart is not None,
        # the per-iteration energy-metric value is surfaced under
        # scf_diagnostics.energy_metric_eV (the recorder block) on the collinear
        # and USPP/PAW paths; here we record the active criterion and threshold,
        # plus the final estimate for the spinor path (which has no recorder).
        **({"entol_eV": float(inp.scf.entol)}
           if inp.scf.convergence == "energy" else {}),
        **({"final_energy_metric_eV": _final.get("energy_metric_eV"),
            "final_energy_metric_charge_eV": _final.get("energy_metric_charge_eV"),
            "final_energy_metric_longitudinal_eV":
                _final.get("energy_metric_longitudinal_eV"),
            "final_energy_metric_transverse_eV":
                _final.get("energy_metric_transverse_eV")}
           if _final.get("energy_metric_eV") is not None else {}),
    }

    summary = {
        "code": {"name": "gradwave", "version": __version__,
                 "created": datetime.datetime.now().isoformat(timespec="seconds")},
        "task": task,
        "structure": _structure_block(inp),
        "parameters": {
            "formalism": "noncollinear" if is_ncmag else (
                "uspp/paw" if uspp else "nc"),
            "xc": _xc_label(inp),
            "ecut_eV": float(inp.ecut),
            "ecutrho_eV": float(inp.ecutrho) if (uspp and inp.ecutrho) else None,
            "kmesh": list(inp.kpoints.mesh),
            "nk": len(system.kweights),
            "nk_total": int(math.prod(inp.kpoints.mesh)),
            "kweights": [float(w) for w in system.kweights],
            "nspin": nspin,
            "smearing": inp.smearing.type,
            "width_eV": float(inp.smearing.width),
            "symmetry": bool(inp.symmetry),
            "mixing": {
                "scheme": inp.scf.mixing.scheme,
                "alpha": float(inp.scf.mixing.alpha),
                "history": inp.scf.mixing.history,
                "kerker": inp.scf.mixing.kerker,
                "kerker_used": _get(res, "kerker_used"),
                "precond": inp.scf.mixing.precond,
            },
            "n_electrons": float(system.n_electrons),
            "nbands": int(system.nbands),
            "fft_grid": list(system.grid.shape),
            "npw": int(system.spheres[0].npw),
            "pseudos": {s: inp.pseudo_map[s] for s in species},
            **({"hubbard": [
                {"species": m.species, "l": m.l, "U_eV": m.u, "J_eV": m.j}
                for m in inp.hubbard.manifolds]} if inp.hubbard.enabled else {}),
        },
        "scf": scf_block,
        "eigenvalues_eV": eig.tolist(),
        "occupations": [] if occ is None else occ.tolist(),
    }
    # SCF flight-recorder diagnostics (scf.recorder): a compact block of the
    # per-iteration convergence health — heuristic tags, long-wavelength
    # (sloshing) residual fraction, band-reordering count, per-iteration wall
    # time. Present for the recording drivers (NC / USPP-PAW collinear); absent
    # for the noncollinear path, which does not record.
    _rec = _get(res, "recorder")
    if _rec is not None and getattr(_rec, "iters", None):
        summary["scf_diagnostics"] = _rec.summarize()
    if runtime_s is not None:
        summary["runtime_s"] = round(float(runtime_s), 2)
    if extra:
        summary.update(extra)
    return summary


# Above this estimated Pulay pressure a vc-relax is running on a severely
# underconverged basis: the (under-estimating) correction can no longer be
# trusted to reach the energy-consistent cell, so the driver warns loudly.
# Scale: converged runs sit ≪1 GPa; 12 Ry silicon ~1-2 GPa; 40 Ry ONCV-oxygen
# quartz (the #217 collapse) ~40 GPa.
_PULAY_WARN_GPA = 5.0


def _pulay_correction_unsupported(inp: Input) -> str | None:
    """Reason the Pulay stress correction cannot run this input, or None.

    Mirrors ``estimate_pressure_error``'s contract (and the calculator's
    shifted-mesh guard): norm-conserving, full (unreduced) Γ-centered k-set,
    no DFT+U, scalar-relativistic. Checked only for a variable-cell relax —
    a fixed-cell relax never applies the correction."""
    _species, upfs, _soa = _species_upfs(inp)
    if _is_uspp(upfs):
        return "USPP/PAW pseudopotentials (the stress-error estimator is NC-only)"
    if inp.symmetry:
        return ("symmetry: true (the estimator's frozen strained rebuild needs "
                "the full k-point set; set symmetry: false to enable)")
    if tuple(inp.kpoints.shift) != (0, 0, 0):
        return "shifted k-mesh (the estimator rebuilds a Γ-centered mesh only)"
    if inp.hubbard.enabled:
        return "DFT+U (pressure error with +U not implemented)"
    if inp.noncollinear:
        return "noncollinear/SOC (no calculator path)"
    if any(b.j is not None for u in upfs for b in u.betas):
        return "fully-relativistic pseudopotentials"
    return None


def _resolve_pulay_correction(inp: Input, verbose: bool = True) -> bool:
    """Whether the vc-relax calculator applies the Pulay pressure correction.

    ``relax.pulay_correction`` None (auto) → on whenever supported, with a
    visible warning naming the reason when it is not (the #217 trap — a
    vc-relax whose stress silently carries the full basis-incompleteness
    bias should never again look routine); True → required, InputError when
    unsupported; False → off. Fixed-cell relax: always False (the stress
    drives nothing)."""
    if not inp.relax.cell or inp.relax.pulay_correction is False:
        return False
    reason = _pulay_correction_unsupported(inp)
    if reason is None:
        return True
    if inp.relax.pulay_correction is True:
        raise InputError(f"relax.pulay_correction: true is unsupported here: {reason}")
    logger.warning("vc-relax without Pulay stress correction: %s", reason)
    if verbose:
        print(f"  relax: Pulay stress correction unavailable ({reason}) — the "
              "reported stress carries uncorrected basis-incompleteness "
              "(Pulay) pressure; converge ecut before trusting the relaxed "
              "cell", flush=True)
    return False


def _build_relax_calc(inp: Input, verbose: bool = True) -> GradWave:
    """The GradWave calculator a relaxation drives — shared by the nested
    engine and by ``joint``'s final consistent-energy/forces/stress SCF at the
    relaxed geometry (so both report ASE-calculator numbers, not the joint
    functional's fixed-basis energy)."""
    from gradwave.calculator import GradWave

    kerker = inp.scf.mixing.kerker
    kerker = None if kerker == "auto" else bool(kerker)
    return GradWave(
        pulay_stress_correction=_resolve_pulay_correction(inp, verbose),
        ecut=inp.ecut,
        pseudopotentials={s: str(inp.pseudo_dir / f)
                          for s, f in inp.pseudo_map.items()},
        xc=inp.xc,
        ecutrho=inp.ecutrho,
        kpts=inp.kpoints.mesh,
        kshift=inp.kpoints.shift,
        smearing=inp.smearing.type,
        width=inp.smearing.width,
        nbands=inp.nbands,
        use_symmetry=inp.symmetry,
        nspin=inp.nspin,
        tot_magnetization=inp.tot_magnetization,
        max_iter=inp.scf.max_iter,
        etol=inp.scf.etol,
        rhotol=inp.scf.rhotol,
        diago_tol=inp.scf.diago_tol,
        mixing_scheme=_mixing_scheme(inp),
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history,
        mixing_kerker=kerker,
        eigensolver=inp.scf.eigensolver,
        precond=inp.scf.mixing.precond,
        hubbard=list(inp.hubbard.manifolds) if inp.hubbard.enabled else None,
        hub_occ_mix=inp.hubbard.occ_mix,
        hub_u_ramp_iters=inp.hubbard.u_ramp_iters,
        extrapolation=inp.relax.extrapolation,
        pulay_solver=inp.relax.pulay_solver,
        device=inp.device,
        verbose=False,
    )


def run_relax(
    inp: Input, verbose: bool = True
) -> tuple[dict[str, Any], Atoms, list[Atoms]]:
    """Relax with ASE, returning (relax block, final atoms, per-step ASE frames).

    ``relax.method`` selects the engine. ``"nested"`` (default) runs a full SCF
    inside every BFGS/FIRE geometry step — robust for every formalism, spin, and
    metals. ``"joint"`` (opt-in) descends on (strain, positions, orbitals) at
    once via ``opt.joint.joint_relax`` for far fewer Hamiltonian applications;
    it is norm-conserving-insulator only and transparently falls back to the
    nested engine on an unsupported system (USPP/PAW, spin, smearing) or if the
    joint descent does not converge. The frames carry energy and forces
    (SinglePointCalculator) so the caller can write an extxyz trajectory.

    Selective dynamics (``inp.fixed``) only the nested engine honours — the joint
    and newton engines relax all degrees of freedom — so a ``joint``/``newton``
    request with fixed atoms falls back to nested with the reason recorded in the
    ``relax`` block (``requested_method``/``fallback_reason``)."""
    if inp.relax.method in ("joint", "newton"):
        if inp.fixed is not None:
            if verbose:
                print(f"  relax: selective dynamics (structure.fixed) is not "
                      f"supported by the {inp.relax.method} engine — using "
                      f"nested", flush=True)
            relax, atoms, frames = _relax_nested(inp, verbose)
            relax["requested_method"] = inp.relax.method
            relax["fallback_reason"] = (
                "selective dynamics (structure.fixed) requires the nested engine")
            return relax, atoms, frames
        out = (_relax_newton(inp, verbose) if inp.relax.method == "newton"
               else _relax_joint(inp, verbose))
        if out is not None:
            return out
        if verbose:
            print(f"  relax: {inp.relax.method} engine unavailable or "
                  "non-converged — falling back to nested SCF+BFGS", flush=True)
    return _relax_nested(inp, verbose)


def _relax_nested(
    inp: Input, verbose: bool = True
) -> tuple[dict[str, Any], Atoms, list[Atoms]]:
    from ase.optimize import BFGS, FIRE

    atoms = inp.atoms.copy()
    if inp.fixed is not None:
        # Selective dynamics: fix each atom's masked axes in FRACTIONAL space, so
        # held atoms ride the cell under a variable-cell relax. ASE applies the
        # constraint inside get_forces (zeroing the held components before the
        # optimizer sees them and before its fmax gate) and inside the position
        # update (so a fully-fixed atom never moves). FixScaled masks the same
        # axes for a fixed cell too, where fractional and Cartesian coincide.
        from ase.constraints import FixScaled

        atoms.set_constraint([
            FixScaled(i, mask=tuple(bool(b) for b in inp.fixed[i]))
            for i in range(len(atoms)) if bool(inp.fixed[i].any())
        ])
    atoms.calc = _build_relax_calc(inp, verbose)
    opt_cls = {"fire": FIRE, "bfgs": BFGS}[inp.relax.optimizer]
    target: Atoms | FrechetCellFilter = atoms
    if inp.relax.cell:
        from ase.filters import FrechetCellFilter

        # ASE stress/pressure is eV/Å³; the user knob is GPa. The filter adds
        # the cell degrees of freedom, so opt.run(fmax) then gates BOTH the
        # atomic forces and the stress (external pressure subtracted).
        gpa_to_ev_a3 = 1.0 / 160.21766208
        target = FrechetCellFilter(
            atoms, scalar_pressure=inp.relax.pressure * gpa_to_ev_a3)
    # ASE's Optimizer.__init__ declares atoms: Atoms, but at runtime accepts
    # any Atoms-like object implementing get_positions/get_forces/etc. —
    # every ase.filters wrapper (FrechetCellFilter included) is meant to be
    # passed here; the stub just doesn't spell out that duck-typed contract.
    opt = opt_cls(cast("Atoms", target), logfile=None)  # our own per-step line below
    trajectory: list[dict[str, Any]] = []
    frames: list[Atoms] = []  # ASE Atoms per step, energy+forces frozen for extxyz output

    def _record() -> None:
        import numpy as np
        from ase.calculators.singlepoint import SinglePointCalculator

        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        fmax = float(np.linalg.norm(forces, axis=1).max())
        entry: dict[str, Any] = {
            "step": opt.nsteps,
            "energy_eV": energy,
            "fmax_eV_ang": fmax,
            "positions_ang": atoms.get_positions().tolist(),
            "cell_ang": atoms.cell.array.tolist(),
        }
        # the calculator caches its last SCF, so each relax step records how
        # many SCF iterations it took and whether it converged
        scf_res = getattr(atoms.calc, "last_result", None)
        if scf_res is not None:
            entry["scf_iter"] = int(getattr(scf_res, "n_iter", 0))
            entry["scf_converged"] = bool(getattr(scf_res, "converged", True))
        pulay_gpa = getattr(atoms.calc, "last_pulay_pressure_gpa", None)
        if pulay_gpa is not None:
            entry["pulay_pressure_GPa"] = round(float(pulay_gpa), 4)
        trajectory.append(entry)
        if verbose:
            sc = ((f" · SCF {entry['scf_iter']} it"
                   + ("" if entry.get("scf_converged", True) else " (NOT conv.)"))
                  if "scf_iter" in entry else "")
            pl = (f" · Pulay {pulay_gpa:+.2f} GPa" if pulay_gpa is not None
                  else "")
            print(f"  relax step {opt.nsteps:>3d} · E = {energy:+.8f} eV"
                  f" · fmax = {fmax:.5f} eV/Å{sc}{pl}", flush=True)
        frame = atoms.copy()
        sp_kw: dict[str, Any] = {"energy": energy, "forces": forces}
        if inp.relax.cell:
            sp_kw["stress"] = atoms.get_stress()
        frame.calc = SinglePointCalculator(frame, **sp_kw)
        frame.info["step"] = opt.nsteps
        frames.append(frame)

    opt.attach(_record)
    converged = opt.run(fmax=inp.relax.fmax, steps=inp.relax.max_steps)
    import numpy as np

    relax: dict[str, Any] = {
        "converged": bool(converged),
        "method": "nested",
        "n_steps": opt.nsteps,
        "optimizer": inp.relax.optimizer,
        "cell_relaxed": bool(inp.relax.cell),
        "fmax_target_eV_ang": inp.relax.fmax,
        "energy_eV": float(atoms.get_potential_energy()),
        "fmax_eV_ang": float(
            np.linalg.norm(atoms.get_forces(), axis=1).max()),
        "max_displacement_ang": float(np.linalg.norm(
            atoms.get_positions() - inp.atoms.get_positions(),
            axis=1).max()),
        "species": atoms.get_chemical_symbols(),
        "positions_ang": atoms.get_positions().tolist(),
        "cell_ang": atoms.cell.array.tolist(),
        "volume_ang3": float(atoms.get_volume()),
        "trajectory": trajectory,
    }
    scf_iters = [s["scf_iter"] for s in trajectory if "scf_iter" in s]
    if scf_iters:
        relax["scf_iter_per_step"] = scf_iters
        relax["scf_total_iter"] = int(sum(scf_iters))
        relax["scf_all_converged"] = all(
            s.get("scf_converged", True) for s in trajectory)
    relax["extrapolation"] = inp.relax.extrapolation
    if getattr(atoms.calc, "_density_clamped", False):
        # the extrapolated density dipped negative on at least one step and was
        # clamped to zero then renormalized to N_e (a benign, recorded fallback)
        relax["extrapolation_density_clamped"] = True
    if trajectory:
        relax["energy_change_eV"] = (
            float(atoms.get_potential_energy()) - trajectory[0]["energy_eV"])
        relax["volume_change_ang3"] = (
            float(atoms.get_volume()) - float(abs(np.linalg.det(
                np.asarray(trajectory[0]["cell_ang"])))))
    last = getattr(atoms.calc, "last_result", None)
    if last is not None and getattr(last, "system", None) is not None:
        relax["nk_ibz"] = len(last.system.kweights)
    if inp.relax.cell:
        relax["max_stress_eV_ang3"] = float(np.abs(atoms.get_stress()).max())
        relax["pressure_GPa"] = inp.relax.pressure
        pulay_final = getattr(atoms.calc, "last_pulay_pressure_gpa", None)
        relax["pulay_correction"] = pulay_final is not None
        if pulay_final is not None:
            relax["pulay_pressure_GPa_final"] = round(float(pulay_final), 4)
            if abs(pulay_final) > _PULAY_WARN_GPA and verbose:
                print(f"  relax: WARNING — estimated Pulay pressure "
                      f"{pulay_final:+.1f} GPa exceeds {_PULAY_WARN_GPA:.0f} GPa: "
                      "the stress at this ecut is severely underconverged. The "
                      "correction reduces but does not eliminate the bias "
                      "(first-order estimate, ~0.5-0.75x); increase ecut before "
                      "trusting the relaxed cell.", flush=True)
    if inp.fixed is not None:
        # record the selective-dynamics mask (True = axis held fixed) so the
        # output documents which degrees of freedom were frozen
        relax["fixed"] = inp.fixed.tolist()
        relax["n_fixed_atoms"] = int(inp.fixed.any(axis=1).sum())
    return relax, atoms, frames


def _joint_supported(inp: Input, upfs: list[UPFData | PAWData]) -> str | None:
    """Reason the joint engine cannot run this input, or None if it can.

    Joint descent (opt/joint.py) is the norm-conserving, nspin=1, fixed-integer-
    occupation prototype: insulators only. Everything outside that contract
    (USPP/PAW overlap, spin, smeared/metallic occupations, external pressure)
    routes to the nested engine instead — see the coverage matrix in
    docs/design/joint-geometry-electronic.md."""
    if _is_uspp(upfs):
        return "USPP/PAW pseudopotential (generalized S-overlap not on the joint graph)"
    if inp.nspin == 2 or inp.noncollinear:
        return "spin-polarized / noncollinear (joint prototype is nspin=1)"
    if inp.smearing.type != "none":
        return f"smearing={inp.smearing.type!r} (metals reintroduce level crossings)"
    if inp.relax.pressure != 0.0:
        return "external pressure (joint functional minimizes E, not enthalpy)"
    if inp.fixed is not None:
        return ("selective dynamics (structure.fixed) — the joint/newton engines "
                "relax all degrees of freedom; use method=nested to hold atoms")
    return None


def _relax_joint(
    inp: Input, verbose: bool = True
) -> tuple[dict[str, Any], Atoms, list[Atoms]] | None:
    """Joint (strain, positions, orbitals) descent as a relax engine.

    Returns the same ``(relax, atoms, frames)`` tuple as the nested engine, or
    ``None`` to signal the caller should fall back to nested — either because
    the system is outside the joint contract or because the descent did not
    converge. Convergence gates are mapped from the ASE calculator's: the force
    tolerance is ``relax.fmax`` and (variable cell) the stress tolerance is
    ``fmax/Ω`` — the FrechetCellFilter treats σ·Ω as a generalized force gated
    by the same scalar. The final energy/forces/stress are recomputed with one
    calculator SCF at the relaxed geometry so they are ASE-consistent (not the
    joint functional's fixed-basis value) and ``last_result`` is populated for
    downstream error estimates."""
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator

    from gradwave.opt.joint import joint_relax

    species, upfs, species_of_atom = _species_upfs(inp)
    reason = _joint_supported(inp, upfs)
    if reason is not None:
        logger.info("relax method=joint not applicable: %s", reason)
        return None

    cell0 = inp.atoms.cell.array.copy()
    pos0 = inp.atoms.get_positions().copy()
    fix_cell = not inp.relax.cell
    omega = float(abs(np.linalg.det(cell0)))
    smax = inp.relax.fmax / omega  # σ·Ω is the FrechetCellFilter cell "force"
    try:
        res = joint_relax(
            cell0, pos0, species_of_atom, upfs, XC_REGISTRY[inp.xc](),
            ecut=inp.ecut, kmesh=inp.kpoints.mesh, fmax=inp.relax.fmax,
            smax=smax, max_closures=40 * inp.relax.max_steps,
            fix_cell=fix_cell, device=inp.device, verbose=verbose,
        )
    except (ValueError, RuntimeError) as exc:  # torch LinAlgError ⊂ RuntimeError
        logger.warning("joint relax failed (%s); falling back to nested", exc)
        return None
    if not res.converged:
        logger.info("joint relax did not converge in %d closures; falling back",
                    res.n_closures)
        return None

    # final ASE-consistent energy/forces/stress at the relaxed geometry
    atoms = inp.atoms.copy()
    atoms.set_cell(res.cell, scale_atoms=False)
    atoms.set_positions(res.positions)
    atoms.calc = _build_relax_calc(inp, verbose)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float(np.linalg.norm(forces, axis=1).max())
    sp_kw: dict[str, Any] = {"energy": energy, "forces": forces}
    if inp.relax.cell:
        sp_kw["stress"] = atoms.get_stress()
    frame = atoms.copy()
    frame.calc = SinglePointCalculator(frame, **sp_kw)
    frame.info["step"] = res.n_closures
    last = getattr(atoms.calc, "last_result", None)

    relax: dict[str, Any] = {
        "converged": True,
        "method": "joint",
        "n_steps": res.n_cycles,             # basis-rebuild cycles (outer loop)
        "n_closures": res.n_closures,        # L-BFGS energy+grad evaluations
        "optimizer": "lbfgs",
        "cell_relaxed": bool(inp.relax.cell),
        "fmax_target_eV_ang": inp.relax.fmax,
        "energy_eV": energy,
        "fmax_eV_ang": fmax_final,
        "max_displacement_ang": float(np.linalg.norm(
            atoms.get_positions() - inp.atoms.get_positions(), axis=1).max()),
        "species": atoms.get_chemical_symbols(),
        "positions_ang": atoms.get_positions().tolist(),
        "cell_ang": atoms.cell.array.tolist(),
        "volume_ang3": float(atoms.get_volume()),
        # H-apply provenance — the reason to use this engine
        "h_applies": int(res.h_equiv),
        "h_seed": int(res.h_seed),
    }
    if last is not None:
        relax["scf_iter_final"] = int(getattr(last, "n_iter", 0))
        if getattr(last, "system", None) is not None:
            relax["nk_ibz"] = len(last.system.kweights)
    if inp.relax.cell:
        relax["max_stress_eV_ang3"] = float(np.abs(atoms.get_stress()).max())
        relax["pressure_GPa"] = inp.relax.pressure
    if verbose:
        print(f"  relax: joint engine converged — E = {energy:+.8f} eV · "
              f"fmax = {fmax_final:.5f} eV/Å · {res.n_cycles} cycles / "
              f"{res.n_closures} closures · {res.h_equiv} H-applies", flush=True)
    return relax, atoms, [frame]


def _relax_newton(
    inp: Input, verbose: bool = True
) -> tuple[dict[str, Any], Atoms, list[Atoms]] | None:
    """Exact-Hvp Steihaug trust-region Newton-CG as a relax engine.

    Same contract, guard, and nested fallback as ``_relax_joint`` (returns
    ``None`` to fall back); the only difference is the inner optimizer
    (``opt.newton.newton_cg_relax``) and the H-apply provenance it reports
    (grad/Hvp/trial counts instead of L-BFGS closures). Final energy/forces/
    stress are recomputed with one calculator SCF at the relaxed geometry so
    the reported numbers are ASE-consistent."""
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator

    from gradwave.opt.newton import newton_cg_relax

    species, upfs, species_of_atom = _species_upfs(inp)
    reason = _joint_supported(inp, upfs)
    if reason is not None:
        logger.info("relax method=newton not applicable: %s", reason)
        return None

    cell0 = inp.atoms.cell.array.copy()
    pos0 = inp.atoms.get_positions().copy()
    fix_cell = not inp.relax.cell
    omega = float(abs(np.linalg.det(cell0)))
    smax = inp.relax.fmax / omega
    try:
        res = newton_cg_relax(
            cell0, pos0, species_of_atom, upfs, XC_REGISTRY[inp.xc](),
            ecut=inp.ecut, kmesh=inp.kpoints.mesh, fmax=inp.relax.fmax,
            smax=smax, max_newton=inp.relax.max_steps, fix_cell=fix_cell,
            device=inp.device, verbose=verbose,
        )
    except (ValueError, RuntimeError) as exc:  # torch LinAlgError ⊂ RuntimeError
        logger.warning("newton relax failed (%s); falling back to nested", exc)
        return None
    if not res.converged:
        logger.info("newton relax did not converge (%d Newton steps); falling "
                    "back", res.n_newton)
        return None

    atoms = inp.atoms.copy()
    atoms.set_cell(res.cell, scale_atoms=False)
    atoms.set_positions(res.positions)
    atoms.calc = _build_relax_calc(inp, verbose)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float(np.linalg.norm(forces, axis=1).max())
    sp_kw: dict[str, Any] = {"energy": energy, "forces": forces}
    if inp.relax.cell:
        sp_kw["stress"] = atoms.get_stress()
    frame = atoms.copy()
    frame.calc = SinglePointCalculator(frame, **sp_kw)
    frame.info["step"] = res.n_newton
    last = getattr(atoms.calc, "last_result", None)

    relax: dict[str, Any] = {
        "converged": True,
        "method": "newton",
        "n_steps": res.n_cycles,
        "n_newton": res.n_newton,
        "n_grad": res.n_grad,
        "n_hvp": res.n_hvp,
        "optimizer": "steihaug-newton-cg",
        "cell_relaxed": bool(inp.relax.cell),
        "fmax_target_eV_ang": inp.relax.fmax,
        "energy_eV": energy,
        "fmax_eV_ang": fmax_final,
        "max_displacement_ang": float(np.linalg.norm(
            atoms.get_positions() - inp.atoms.get_positions(), axis=1).max()),
        "species": atoms.get_chemical_symbols(),
        "positions_ang": atoms.get_positions().tolist(),
        "cell_ang": atoms.cell.array.tolist(),
        "volume_ang3": float(atoms.get_volume()),
        "h_applies": int(res.h_equiv),
        "h_seed": int(res.h_seed),
    }
    if last is not None:
        relax["scf_iter_final"] = int(getattr(last, "n_iter", 0))
        if getattr(last, "system", None) is not None:
            relax["nk_ibz"] = len(last.system.kweights)
    if inp.relax.cell:
        relax["max_stress_eV_ang3"] = float(np.abs(atoms.get_stress()).max())
        relax["pressure_GPa"] = inp.relax.pressure
    if verbose:
        print(f"  relax: newton-cg engine converged — E = {energy:+.8f} eV · "
              f"fmax = {fmax_final:.5f} eV/Å · {res.n_newton} steps · "
              f"{res.n_hvp} Hvp · {res.h_equiv} H-applies", flush=True)
    return relax, atoms, [frame]


def run_eos(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Isotropic volume scan + 3rd-order Birch-Murnaghan fit → V0, B0, B0'.

    For each factor in ``inp.eos.scales`` the cell is scaled isotropically
    (a → a·s^(1/3), fractional coordinates fixed) and the SCF re-converged. All
    volumes share ONE FFT grid, pinned to the elementwise max over the scan, so
    E(V) carries no grid-discontinuity steps; each volume warm-starts from the
    previous converged density (the cheap, branch-stable EOS chain). Returns the
    ``eos`` summary block."""
    import numpy as np

    from gradwave.postscf.eos import EV_A3_TO_GPA, fit_bm3

    _species, upfs, species_of_atom = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    scales = list(inp.eos.scales)
    cell0 = np.asarray(inp.atoms.cell.array, dtype=float)
    frac = inp.atoms.get_scaled_positions()
    natoms = len(inp.atoms)

    def _build_at(
        scale: float, fft_shape: tuple[int, ...] | None
    ) -> tuple[System | USPPSystem, Any]:
        cell = cell0 * scale ** (1.0 / 3.0)
        pos = frac @ cell
        if uspp:
            from gradwave.scf.uspp import setup_uspp

            return setup_uspp(
                cell, pos, species_of_atom, _as_paws(upfs), ecut=inp.ecut,
                kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho, nbands=inp.nbands,
                use_symmetry=inp.symmetry, fft_shape=fft_shape), cell
        from gradwave.scf.loop import setup_system

        return setup_system(
            cell=cell, positions=pos, species_of_atom=species_of_atom,
            upfs=_as_upfs(upfs), ecut=inp.ecut, kmesh=inp.kpoints.mesh,
            kshift=inp.kpoints.shift, nbands=inp.nbands,
            use_symmetry=inp.symmetry, fft_shape=fft_shape), cell

    # pass 1: natural FFT grid per volume, then pin the elementwise max so
    # every volume shares one grid (larger cells otherwise pick a finer grid)
    dims = [tuple(_fft_grid(_build_at(s, None)[0]).shape) for s in scales]
    fixed = tuple(max(d[i] for d in dims) for i in range(3))
    if verbose:
        print(f"eos: {len(scales)} volumes on fixed FFT grid {fixed}", flush=True)

    prev = None
    volumes, energies, converged = [], [], []
    ekind = inp.eos.energy
    for s in scales:
        sysd, cell = _build_at(s, fixed)
        res = run_scf(inp, system=sysd, verbose=False, start_from=prev)
        prev = res
        e = float(getattr(res.energies, ekind))
        vol = float(abs(np.linalg.det(cell)))
        conv = bool(getattr(res, "converged", True))
        volumes.append(vol)
        energies.append(e)
        converged.append(conv)
        if verbose:
            tag = "" if conv else "  (NOT converged)"
            print(f"  s={s:.3f}  V={vol / natoms:8.4f} Å³/at  "
                  f"E={e / natoms:+.6f} eV/at{tag}", flush=True)

    v_at = np.array(volumes) / natoms
    e_at = np.array(energies) / natoms
    fit = fit_bm3(v_at, e_at)
    block: dict[str, Any] = {
        "scales": scales,
        "energy_kind": ekind,
        "n_atoms": natoms,
        "volumes_ang3_per_atom": v_at.tolist(),
        "energies_eV_per_atom": e_at.tolist(),
        "fft_grid": list(fixed),
        "v0_ang3_per_atom": fit.v0,
        "b0_GPa": fit.b0_GPa,
        "b0_prime": fit.b0_prime,
        "e0_eV_per_atom": fit.e0,
        "rms_residual_eV_per_atom": fit.rms_residual_eV,
        "b0_eV_ang3": fit.b0,
        "ev_a3_to_gpa": EV_A3_TO_GPA,
        "all_converged": all(converged),
    }
    if verbose:
        print(f"eos: V0={fit.v0:.4f} Å³/at  B0={fit.b0_GPa:.2f} GPa  "
              f"B0'={fit.b0_prime:.3f}", flush=True)
    return block


def run_elastic(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Elastic constants: FD of the analytic stress over the six Voigt strains
    → the 6×6 stiffness C and Voigt–Reuss–Hill moduli.

    ``elastic.mode`` picks the tensor. ``"clamped"`` (default) deforms the cell
    with fractional coordinates fixed and re-relaxes only the electrons.
    ``"relaxed"`` additionally relaxes the internal coordinates at every
    strained cell (fixed-cell BFGS on the analytic forces, gated by
    ``elastic.fmax``) before reading the stress, giving the relaxed-ion tensor
    C_ij = dσ_i/dε_j along the relaxed path — the physical constants for
    crystals whose compliance is dominated by symmetry-allowed internal
    displacement (quartz, diamond/zincblende shear). The input geometry is
    assumed to be the equilibrium one; the residual reference fmax is reported
    so a non-equilibrium start is visible in the output.

    Each strain runs on one FFT grid pinned across the scan; every strained SCF
    warm-starts from the unstrained reference (in relaxed mode, each ionic step
    warm-starts from the previous one). Norm-conserving (``postscf.stress``) and
    USPP/PAW (``postscf.paw_stress``) are both handled, for nspin=1 and
    collinear nspin=2 (the fixed-basis stress sums per spin channel — see
    postscf.stress).

    Fully-relativistic (spin-orbit) spinor runs (``noncollinear: true`` with
    j-resolved pseudos) are handled on the norm-conserving path:
    ``postscf.stress.stress`` differentiates the spinor strained energy
    (``_energy_strained_fr``), so the same FD-of-analytic-stress driver folds it
    into C. DFT+U on that path is a feature boundary (#142), and the USPP/PAW
    spinor stress has no path yet, so both are rejected below."""
    import numpy as np

    from gradwave.postscf.elastic import (
        elastic_tensor,
        is_mechanically_stable,
        moduli_from_cij,
    )

    _species, upfs, species_of_atom = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    is_fr = any(b.j is not None for u in upfs for b in u.betas)
    relaxed_ion = inp.elastic.mode == "relaxed"
    if relaxed_ion and inp.noncollinear:
        raise NotImplementedError(
            "relaxed-ion elastic constants need the collinear force path — "
            "noncollinear/spinor forces are not implemented (use "
            "elastic.mode: clamped)")
    if inp.noncollinear:
        if uspp:
            raise NotImplementedError(
                "elastic constants for noncollinear USPP/PAW runs are not "
                "supported (no spinor USPP/PAW stress path)")
        if not is_fr:
            raise NotImplementedError(
                "elastic constants for a magnetic non-collinear run without "
                "spin-orbit need fully-relativistic (j-resolved) pseudos — the "
                "scalar stress path has no magnetic-spinor route")
        if inp.hubbard.enabled:
            raise NotImplementedError(
                "DFT+U elastic constants on the spin-orbit path (feature "
                "boundary, #142)")

    if inp.noncollinear:
        from gradwave.core.xc.noncollinear import NoncollinearXC
        xc = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())
    elif inp.nspin == 2:
        xc = SPIN_XC_REGISTRY[inp.xc]()
    else:
        xc = XC_REGISTRY[inp.xc]()
    if uspp:
        from gradwave.postscf.paw_stress import stress_uspp as _stress
    else:
        from gradwave.postscf.stress import stress as _stress

    cell0 = np.asarray(inp.atoms.cell.array, dtype=float)
    frac = inp.atoms.get_scaled_positions()
    natoms = len(inp.atoms)
    h = inp.elastic.strain

    # a magnetic spinor breaks k ≡ −k (TR flips m⃗); a nonmagnetic spinor
    # (SOC only) keeps Kramers, matching build_system's time-reversal logic.
    time_reversal = not (inp.noncollinear and not inp.nonmagnetic)

    def _build(
        cell: Any, fft_shape: tuple[int, ...] | None, pos: Any = None
    ) -> System | USPPSystem:
        if pos is None:
            pos = frac @ cell
        if uspp:
            from gradwave.scf.uspp import setup_uspp

            return setup_uspp(
                cell, pos, species_of_atom, _as_paws(upfs), ecut=inp.ecut,
                kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho, nbands=inp.nbands,
                use_symmetry=inp.symmetry, fft_shape=fft_shape)
        from gradwave.scf.loop import setup_system

        return setup_system(
            cell=cell, positions=pos, species_of_atom=species_of_atom,
            upfs=_as_upfs(upfs), ecut=inp.ecut, kmesh=inp.kpoints.mesh,
            kshift=inp.kpoints.shift, nbands=inp.nbands,
            use_symmetry=inp.symmetry, time_reversal=time_reversal,
            fft_shape=fft_shape)

    # pin one FFT grid: the +h strains give the largest cells / finest grids
    from gradwave.postscf.elastic import voigt_strain_tensor

    probe = [cell0] + [cell0 @ (np.eye(3) + voigt_strain_tensor(j, h)).T
                       for j in range(6)]
    fixed = tuple(max(int(_fft_grid(_build(c, None)).shape[i]) for c in probe)
                  for i in range(3))
    if verbose:
        print(f"elastic: {inp.elastic.mode}-ion, strain h={h}, "
              f"fixed FFT grid {fixed}", flush=True)

    # reference SCF once — warm-start seed and residual-stress readout
    ref = run_scf(inp, system=_build(cell0, fixed), verbose=False)
    converged = [bool(getattr(ref, "converged", True))]
    # _stress is stress_uspp (res: USPPResult, xc: XCFunctional | SpinXC) or
    # stress (res: SCFResult | NCResult, xc: XCFunctional | SpinXC |
    # NoncollinearXC) depending on the branch above; ty can't correlate which
    # member of that function union is bound with which result/xc value this
    # call actually has (the uspp/noncollinear guards above are runtime, not
    # visible to the type checker across the import-time branch), so the seam
    # needs Any on both arguments, not a specific cast.
    sigma_ref = _stress(cast(Any, ref), cast(Any, xc)).detach().cpu().numpy()

    # relaxed-ion machinery: analytic forces for the per-strain fixed-cell BFGS
    ref_fmax: float | None = None
    relax_steps: list[int] = []
    relax_conv: list[bool] = []
    if relaxed_ion:
        if uspp:
            from gradwave.postscf.paw_forces import forces_uspp

            def _forces(res: Any) -> Any:
                return forces_uspp(res, cast(Any, xc)).detach().cpu().numpy()
        else:
            from gradwave.postscf.forces import forces as _forces_nc

            def _forces(res: Any) -> Any:
                return _forces_nc(res, xc=cast(Any, xc)).detach().cpu().numpy()

        ref_fmax = float(np.linalg.norm(_forces(ref), axis=1).max())
        if verbose and ref_fmax > inp.elastic.fmax:
            print(f"elastic: reference fmax {ref_fmax:.4f} eV/Å exceeds the "
                  f"relax gate {inp.elastic.fmax} — input geometry is not at "
                  "equilibrium, C is about the current state", flush=True)

    def _relax_ions(cell: Any) -> Any:
        """Fixed-cell BFGS on the analytic forces at a strained cell; returns
        the SCF result at the relaxed geometry (each ionic step warm-starts
        from the previous one, the first from the unstrained reference)."""
        from ase import Atoms as _AseAtoms
        from ase.calculators.calculator import Calculator, all_changes
        from ase.optimize import BFGS
        from typing_extensions import override

        state: dict[str, Any] = {"prev": ref, "res": None}

        class _FixedCellCalc(Calculator):
            implemented_properties = ["energy", "forces"]

            # ASE's signature (atoms/properties/system_changes defaults) — the
            # base class drives this; only the SCF+forces body is ours.
            @override
            def calculate(self, atoms: Any = None,
                          properties: Any = ("energy",),
                          system_changes: Any = all_changes) -> None:
                super().calculate(atoms, properties, system_changes)
                pos = cast("Atoms", self.atoms).get_positions()
                res = run_scf(
                    inp, system=_build(cell, fixed, pos=pos),
                    verbose=False, start_from=state["prev"])
                state["prev"] = state["res"] = res
                converged.append(bool(getattr(res, "converged", True)))
                self.results = {
                    "energy": float(res.energies.total),
                    "forces": _forces(res),
                }

        atoms = _AseAtoms(inp.atoms.get_chemical_symbols(), scaled_positions=frac,
                          cell=np.asarray(cell), pbc=True)
        atoms.calc = _FixedCellCalc()
        opt = BFGS(atoms, logfile=None)
        ok = bool(opt.run(fmax=inp.elastic.fmax, steps=inp.elastic.max_steps))
        relax_steps.append(int(opt.nsteps))
        relax_conv.append(ok)
        if verbose:
            fnow = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            print(f"elastic: strain {len(relax_steps):>2d}/12 ions relaxed in "
                  f"{opt.nsteps} steps (fmax {fnow:.4f} eV/Å"
                  f"{'' if ok else ', NOT converged'})", flush=True)
        return state["res"]

    def _stress_at(eps: Any) -> Any:
        cell = cell0 @ (np.eye(3) + eps).T
        if relaxed_ion:
            res = _relax_ions(cell)
        else:
            res = run_scf(inp, system=_build(cell, fixed), verbose=False,
                          start_from=ref)
            converged.append(bool(getattr(res, "converged", True)))
        return _stress(cast(Any, res), cast(Any, xc)).detach().cpu().numpy()

    c = elastic_tensor(_stress_at, h=h)
    mod = moduli_from_cij(c)
    resid_gpa = float(np.abs(sigma_ref).max()) * 160.2176634
    block: dict[str, Any] = {
        "strain": h,
        "mode": inp.elastic.mode,
        "n_atoms": natoms,
        "formalism": "uspp/paw" if uspp else "nc",
        "c_GPa": c.tolist(),
        "bulk_modulus_GPa": {"voigt": mod.bulk_voigt, "reuss": mod.bulk_reuss,
                             "hill": mod.bulk_hill},
        "shear_modulus_GPa": {"voigt": mod.shear_voigt, "reuss": mod.shear_reuss,
                              "hill": mod.shear_hill},
        "young_modulus_GPa": mod.young,
        "poisson_ratio": mod.poisson,
        "mechanically_stable": is_mechanically_stable(c),
        "residual_stress_GPa": resid_gpa,
        "all_converged": all(converged),
    }
    if relaxed_ion:
        block["relax_fmax"] = inp.elastic.fmax
        block["ref_fmax_eV_ang"] = ref_fmax
        # 12 entries in FD order: strain j = 0..5, +h then −h for each
        block["relax_steps"] = relax_steps
        block["relax_all_converged"] = all(relax_conv)
    if verbose:
        print(f"elastic: K={mod.bulk_hill:.1f}  G={mod.shear_hill:.1f}  "
              f"E={mod.young:.1f} GPa  ν={mod.poisson:.3f}  "
              f"stable={block['mechanically_stable']}", flush=True)
    return block


def run_phonons(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Supercell finite-displacement phonons: dispersion along a q-path + a
    phonon DOS on a q-mesh.

    Builds the diagonal supercell, displaces only the primitive home-cell atoms
    (the translational reduction — 6·N_prim SCFs regardless of supercell size,
    each warm-started from the undisplaced reference), folds the force constants
    to D(q) and diagonalizes. Norm-conserving OR ultrasoft/PAW, nspin ∈ {1, 2}
    (the analytic forces sum per spin channel — see postscf.forces /
    postscf.paw_forces)."""
    import numpy as np

    from gradwave.postscf.phonons_supercell import (
        build_supercell,
        dispersion,
        force_constants_home,
        phonon_dos,
    )

    if inp.noncollinear:
        raise NotImplementedError(
            "supercell phonons do not support noncollinear/spinor runs "
            "(the finite-displacement force fold uses the collinear force path)")
    _species, upfs, species_of_atom = _species_upfs(inp)
    uspp = _is_uspp(upfs)

    if inp.nspin == 2:
        xc, mags = _spin_setup(inp)
    else:
        xc, mags = XC_REGISTRY[inp.xc](), None
    cell = np.asarray(inp.atoms.cell.array, dtype=float)
    positions = inp.atoms.get_positions()
    masses = inp.atoms.get_masses()  # primitive-atom masses [amu]
    n = inp.phonons.supercell
    scmap = build_supercell(cell, positions, species_of_atom, n)
    # fold the primitive k-mesh by the supercell size (equivalent BZ sampling)
    ksuper = tuple(max(1, inp.kpoints.mesh[i] // n[i]) for i in range(3))

    def make_scf(pos_sc: Any, start_from: Any = None) -> SCFResult | USPPResult:
        if uspp:
            from gradwave.scf.uspp import scf_uspp, setup_uspp

            usystem = setup_uspp(
                scmap.cell_super, pos_sc, scmap.species_super, _as_paws(upfs),
                ecut=inp.ecut, kmesh=ksuper, ecutrho=inp.ecutrho,
                use_symmetry=False)
            if inp.device != "cpu":
                usystem = usystem.to(inp.device)
            # scf_uspp has no tot_magnetization pin (see run_scf); a zero
            # start_mag is the nonmagnetic seed for nspin=2.
            return scf_uspp(
                usystem, xc, nspin=inp.nspin,
                start_mag=mags,
                smearing=inp.smearing.type, width=inp.smearing.width,
                etol=inp.scf.etol, rhotol=inp.scf.rhotol,
                mixing_alpha=inp.scf.mixing.alpha,
                mixing_history=inp.scf.mixing.history,
                diago_tol=inp.scf.diago_tol, start_from=start_from,
                verbose=False)
        from gradwave.scf.loop import scf, setup_system

        system = setup_system(
            scmap.cell_super, pos_sc, scmap.species_super, _as_upfs(upfs),
            ecut=inp.ecut, kmesh=ksuper, kshift=inp.kpoints.shift,
            use_symmetry=False)
        if inp.device != "cpu":
            system = system.to(inp.device)
        # fixed spin moment: a collinear nspin=2, no-smearing pin (see run_scf)
        spin_kw: dict[str, Any] = (
            {"tot_magnetization": inp.tot_magnetization}
            if inp.nspin == 2 and inp.tot_magnetization is not None else {})
        return scf(system, cast("XCFunctional", xc), nspin=inp.nspin,
                   start_mag=mags,
                   smearing=inp.smearing.type, width=inp.smearing.width,
                   etol=inp.scf.etol, rhotol=inp.scf.rhotol,
                   mixing_alpha=inp.scf.mixing.alpha,
                   mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
                   precond=inp.scf.mixing.precond,
                   diago_tol=inp.scf.diago_tol, start_from=start_from, verbose=False,
                   **spin_kw)

    if verbose:
        print(f"phonons: {tuple(n)} supercell ({scmap.n_sc} atoms), displacing "
              f"{scmap.n_prim} home atoms → {6 * scmap.n_prim} SCFs, k-mesh {ksuper}",
              flush=True)
    # xc is needed by the force path only for NLCC species (spin-resolved for
    # nspin=2); it is ignored for valence-only pseudos. USPP/PAW routes the fold
    # through the augmentation-aware paw_forces.forces_uspp; NC uses the default.
    force_fn: Callable[[Any], Any] | None = None
    if uspp:
        from gradwave.postscf.paw_forces import forces_uspp

        def force_fn(res: Any) -> Any:
            return forces_uspp(res, xc)
    phi = force_constants_home(make_scf, scmap, h=inp.phonons.displacement,
                               xc=xc, force_fn=force_fn, verbose=verbose)

    bp = inp.atoms.cell.bandpath(path=inp.phonons.path or None,
                                 npoints=inp.phonons.npoints)
    qpts = np.asarray(bp.kpts)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    freqs = dispersion(phi, scmap, masses, qpts)  # (nq, 3·N_prim) [cm⁻¹]
    block = {
        "supercell": list(n),
        "n_atoms_supercell": scmap.n_sc,
        "displacement_ang": inp.phonons.displacement,
        "kmesh_supercell": list(ksuper),
        "qpts_frac": qpts.tolist(),
        "x": np.asarray(x).tolist(),
        "labels": list(zip(xticks.tolist(), list(xlabels), strict=True)),
        "frequencies_cm1": freqs.tolist(),
        "min_frequency_cm1": float(freqs.min()),
    }
    if min(inp.phonons.dos_mesh) > 0:
        from gradwave.kpoints import monkhorst_pack

        qmesh, weights = monkhorst_pack(inp.phonons.dos_mesh)
        grid, dos = phonon_dos(phi, scmap, masses, qmesh, weights,
                               width=inp.phonons.dos_width)
        block["dos"] = {"frequency_cm1": grid.tolist(), "dos": dos.tolist(),
                        "mesh": list(inp.phonons.dos_mesh)}
    if verbose:
        print(f"phonons: min frequency {freqs.min():.1f} cm⁻¹ "
              f"({'has imaginary modes' if freqs.min() < -1.0 else 'all real'})",
              flush=True)
    return block


def _bands_reference(res: SCFLike) -> float:
    """Band-plot energy zero: the Fermi level for a metal (any partially filled
    state), else the valence-band maximum. Mirrors postscf.bands.band_structure
    so the USPP path reports the same reference the NC path does."""
    import numpy as np

    occ = np.asarray(_get(res, "occupations"), dtype=float)
    eig = np.asarray(_get(res, "eigenvalues"), dtype=float)
    nspin = int(_get(res, "nspin", 1) or 1)
    g = 2.0 if nspin == 1 else 1.0
    is_metal = bool(((occ > _OCC_TOL) & (occ < g - _OCC_TOL)).any())
    if is_metal:
        return float(_get(res, "fermi"))
    return float(eig[occ > _OCC_TOL].max())


def _bands_uspp_block(inp: Input, res: SCFLike, verbose: bool) -> dict[str, Any]:
    """USPP/PAW band structure along an ASE k-path via postscf.uspp_bands. The NC
    ``bands_along_ase_path`` builds the path and returns a BandStructure; the
    USPP solver instead takes an explicit k-list and returns bare eigenvalues, so
    the ASE path (k-points, axis, special-point labels) and the energy reference
    are assembled here to match the NC bands block shape."""
    import numpy as np

    from gradwave.postscf.uspp_bands import bands_uspp

    bp = inp.atoms.cell.bandpath(path=inp.bands.path or None,
                                 npoints=inp.bands.npoints)
    kpts = np.asarray(bp.kpts, dtype=float)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    xc = SPIN_XC_REGISTRY[inp.xc]() if inp.nspin == 2 else XC_REGISTRY[inp.xc]()
    if verbose:
        print(f"bands (USPP/PAW): {len(kpts)} k-points along the path", flush=True)
    # this block only ever runs for a USPP/PAW result (see _bands_uspp_block's
    # caller); res's static SCFLike type is wider than what's true here.
    eig = bands_uspp(cast("USPPResult", res), xc, kpts,
                     nbands=inp.bands.nbands).detach().cpu().numpy()
    bands: dict[str, Any] = {
        "kpts_frac": kpts.tolist(),
        "x": np.asarray(x).tolist(),
        "labels": list(zip(xticks.tolist(), list(xlabels), strict=True)),
        "eigenvalues_eV": eig.tolist(),
        "reference_eV": _bands_reference(res),
    }
    return {"bands": bands}


def _bands_extra(inp: Input, res: SCFLike, verbose: bool) -> dict[str, Any]:
    from gradwave.postscf.bands import bands_along_ase_path

    _species, upfs, _soa = _species_upfs(inp)
    if _is_uspp(upfs):
        return _bands_uspp_block(inp, res, verbose)

    # bands_along_ase_path declares res: SCFResult; this branch is reached
    # only when _is_uspp(upfs) is False above, so res is the norm-conserving
    # SCFResult (the USPP/PAW and noncollinear formalisms route to
    # _bands_uspp_block / a different task entirely).
    bs = bands_along_ase_path(
        cast("SCFResult", res), inp.atoms, path=inp.bands.path,
        npoints=inp.bands.npoints, nbands=inp.bands.nbands, verbose=verbose,
    )
    # bands_along_ase_path always populates x/labels (the BandStructure
    # dataclass is shared with the lower-level band_structure(), which
    # leaves them unset — see postscf/bands.py).
    assert bs.x is not None
    assert bs.labels is not None
    bands: dict[str, Any] = {
        "kpts_frac": bs.kpts_frac.tolist(),
        "x": bs.x.tolist(),
        "labels": bs.labels,
        "eigenvalues_eV": bs.eigenvalues.tolist(),
        "reference_eV": bs.reference,
    }
    if inp.bands.irreps:
        import numpy as np

        from gradwave.postscf.irreps import band_irreps

        cache: dict[Any, Any] = {}
        ann: list[dict[str, Any]] = []
        for xt, lab in bs.labels:
            idx = int(np.argmin(np.abs(np.asarray(bs.x) - xt)))
            kf_exact = bs.kpts_frac[idx]  # full precision — rounding here
            # shrinks the little group at threshold (1/3 vs 0.33333333)
            key = tuple(np.round(kf_exact, 8))
            if key not in cache:
                # same "reached only for the norm-conserving SCFResult" guard
                # as the bands_along_ase_path call above.
                cache[key] = band_irreps(
                    cast("SCFResult", res), kf_exact, nbands=inp.bands.nbands)
            ann.append({
                "x": float(xt), "name": lab,
                "clusters": [
                    {"e": float(np.mean(c.energies)), "label": c.label,
                     "dim": c.dim, "warning": c.warning}
                    for c in cache[key].clusters
                ],
            })
        bands["irreps"] = ann
    return {"bands": bands}


def _error_estimate_xc(inp: Input) -> NoncollinearXC | SpinXC | XCFunctional:
    """The functional object the post-SCF estimators need to rebuild operators.

    Non-collinear runs need a ``NoncollinearXC`` (the exchange field enters the
    spinor Hamiltonian); collinear nspin=2 needs the spin functional; nspin=1 the
    plain one.
    """
    if inp.noncollinear:
        from gradwave.core.xc.noncollinear import NoncollinearXC

        return NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())
    return _spin_setup(inp)[0] if inp.nspin == 2 else XC_REGISTRY[inp.xc]()


def _error_estimate_block(res: SCFLike, inp: Input) -> dict[str, Any]:
    """Post-SCF plane-wave (Ecut) discretization-error estimate for the output.

    Cheap post-processing (no larger SCF): the first-order complement correction
    of Cancès et al. gives the estimated basis-set error in the energy (a
    definite lowering), the density, the Kohn-Sham eigenvalues / band gap, and --
    for norm-conserving collinear runs -- the Hellmann-Feynman forces. Covers
    norm-conserving and USPP/PAW (nspin=1, 2) and the non-collinear/SOC spinor
    formalism. Reported as an indicator, not a rigorous bound. Degrades
    gracefully when the run's formalism/settings are outside coverage.
    """
    from gradwave.postscf.discretization_error import (
        estimate_density_error,
        estimate_eigenvalue_error,
        estimate_force_error,
        estimate_gap_error,
    )

    _species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    is_nc = bool(inp.noncollinear)
    xc = _error_estimate_xc(inp)
    system = _get(res, "system")
    nspin = int(_get(res, "nspin", 1) or 1)
    natom = len(system.positions)
    grid = system.grid
    vol, npts = grid.volume, grid.n_points
    nelec = float(system.n_electrons)
    # postscf.discretization_error / convergence_error declare `res:
    # SCFResult`, but (per this function's own docstring) duck-type over
    # every formalism via getattr, exactly like `_get` above — the
    # annotations there are simply narrower than the runtime contract.
    res_nc = cast("SCFResult", res)
    # a non-collinear SCF always runs with a real smearing scheme (spinor bands
    # hold one electron); a "none" request maps to gaussian, as the run does.
    nc_scheme = ("gaussian" if inp.smearing.type == "none" else inp.smearing.type)
    dens_kw: dict[str, Any] = (
        dict(smearing=nc_scheme, width=inp.smearing.width) if is_nc else {})
    try:
        err = estimate_density_error(res_nc, xc=xc, **dens_kw)
    except NotImplementedError as e:
        return {"available": False, "reason": str(e)}

    drho = err.drho
    free_e = float(_get(res, "energies").free_energy)
    block: dict[str, Any] = {
        "available": True,
        "method": "Cances first-order complement (post-SCF)",
        "ecut_eV": err.ecut,
        "ecut_large_eV": err.ecut_large,
        "denergy_eV": float(err.denergy),
        "denergy_meV_per_atom": float(err.denergy) / natom * 1e3,
        "free_energy_extrapolated_eV": free_e + float(err.denergy),
        "drho_L1_per_electron": float(drho.abs().sum()) * vol / npts / nelec,
        "int_drho": float(drho.sum()) * vol / npts,
        "note": "first-order estimate, indicative not a rigorous bound",
    }
    # force error: norm-conserving collinear (no NLCC) or USPP/PAW (nspin=1, 2,
    # incl. NLCC, no +U). The spinor force terms in P(eps) are not assembled.
    if uspp:
        force_ok = not is_nc and _get(res, "hub_sites") is None
    else:
        force_ok = (not is_nc and nspin in (1, 2)
                    and getattr(system, "rho_core", None) is None)
    if force_ok:
        try:
            # force_ok is False whenever is_nc, so xc is never a NoncollinearXC
            # here — same "narrower annotation than the force_ok-guarded
            # runtime contract" seam as res_nc above.
            fe = estimate_force_error(
                res_nc, err, xc=cast("XCFunctional | SpinXC | None", xc)
            ).norm(dim=1)
            block["force_error_max_eV_ang"] = float(fe.max())
            block["force_error_rms_eV_ang"] = float((fe ** 2).mean().sqrt())
        except NotImplementedError as exc:
            block["force_error"] = {"available": False, "reason": str(exc)}
    # band-gap error (insulators; NC/USPP/PAW now covered, skipped for metals).
    try:
        eig_kw: dict[str, Any] = (
            dict(smearing=nc_scheme, width=inp.smearing.width) if is_nc else {})
        eige = estimate_eigenvalue_error(res_nc, ecut_large=err.ecut_large, xc=xc,
                                         **eig_kw)
        gap_kw: dict[str, Any] = {}
        if is_nc:
            # NCResult carries no occupations; recompute (degeneracy 1) and set
            # the metal/insulator threshold to half of one-electron filling.
            gap_kw = dict(occupations=_nc_occupations(cast("NCResult", res),
                                                      nc_scheme,
                                                      inp.smearing.width),
                          occ_threshold=0.5)
        gap = estimate_gap_error(res_nc, eige, **gap_kw)
        block["gap_eV"] = gap["gap_eV"]
        block["gap_extrapolated_eV"] = gap["gap_extrapolated_eV"]
        block["dgap_eV"] = gap["dgap_eV"]
    except (NotImplementedError, ValueError) as exc:
        block["gap_error"] = {"available": False, "reason": str(exc)}
    # other numerical convergence errors (SCF self-consistency, smearing). These
    # are separate axes from the basis-set error; k-point sampling needs a mesh
    # sweep (estimate_kpoint_error) and is not reachable from one run.
    from gradwave.postscf.convergence_error import (
        estimate_scf_error,
        estimate_smearing_error,
    )
    # SCF self-consistency error is extrapolated from the energy trajectory, so
    # it is available for every system. The collinear response kernel
    # (K_Hxc/chi0) adds an optional second-order diagnostic where it applies
    # (norm-conserving collinear); USPP/PAW and the spinor formalism have no such
    # primitive exposed yet, so xc is only passed on the supported path.
    scf_xc = xc if (not uspp and not is_nc) else None
    try:
        # scf_xc is never a NoncollinearXC (it's None whenever is_nc) —
        # narrower annotation than the guarded runtime contract, as above.
        scfe = estimate_scf_error(res_nc, cast("XCFunctional | None", scf_xc))
        sc: dict[str, Any] = {
            "denergy_eV": scfe.denergy,
            "denergy_meV_per_atom": scfe.denergy / natom * 1e3,
            "residual_L1_per_electron": scfe.residual_norm,
            "reliable": scfe.reliable,
            "ratio": scfe.ratio,
            "energy_converged_estimate_eV": scfe.energy_converged_estimate,
            "method": "energy-trajectory extrapolation",
        }
        if scfe.denergy_response is not None:
            sc["denergy_response_eV"] = scfe.denergy_response
            sc["screened"] = scfe.screened
            sc["note"] = ("response diagnostic is not sign-definite; the "
                          "headline denergy is the trajectory extrapolation")
        block["scf_convergence"] = sc
    except (NotImplementedError, ValueError, AttributeError) as exc:
        logger.debug("scf_convergence estimate skipped: %r", exc)
    try:
        # estimate_smearing_error reads res.energies — an attribute on every
        # result dataclass, USPP/PAW included
        sme = estimate_smearing_error(
            res_nc, scheme=nc_scheme if is_nc else inp.smearing.type,
            width=inp.smearing.width)
        block["smearing"] = {
            "scheme": sme.scheme,
            "dsmearing_eV": sme.dsmearing,
            "energy_extrapolated_eV": sme.energy_extrapolated,
            "residual_bound_eV": sme.half_width,
            "note": sme.note,
        }
    except (NotImplementedError, ValueError) as exc:
        # record the reason under a distinct key: the human report reads
        # block["smearing"] eagerly, so an available:False entry there would
        # break it (a fixed-occupation run raises here every time)
        block["smearing_error"] = {"available": False, "reason": str(exc)}
    # rolled-up numerical-energy error: the reachable terms (basis-set Ecut, SCF
    # self-consistency, smearing) add only to leading order because the axes
    # couple, so this is an indicative sum rather than a rigorous total. k-point
    # sampling is not reachable from a single run and is excluded.
    terms = {"ecut": abs(float(block["denergy_eV"]))}
    if isinstance(block.get("scf_convergence"), dict):
        terms["scf"] = abs(float(block["scf_convergence"]["denergy_eV"]))
    if isinstance(block.get("smearing"), dict):
        terms["smearing"] = abs(float(block["smearing"]["dsmearing_eV"]))
    total = sum(terms.values())
    block["numerical_energy_error"] = {
        "total_eV": total,
        "total_meV_per_atom": total / natom * 1e3,
        "terms_eV": terms,
        "note": "leading-order sum of the reachable terms; k-point sampling excluded",
    }
    return block


def _nc_occupations(res: NCResult, scheme: str, width: float) -> list[Any]:
    """Per-k occupations of a spinor (NCResult) run, recomputed for the gap tool.

    NCResult stores neither the occupations nor the smearing width, so rebuild
    them from the stored eigenvalues at degeneracy 1.0 (one electron per spinor
    band), the same recipe the SCF uses.
    """
    from gradwave.core.occupations import (
        SCHEMES,
        find_fermi,
        occupations_and_entropy,
    )

    system = res.system
    eps = res.eigenvalues
    mu = find_fermi(eps, system.kweights, SCHEMES[scheme], width,
                    system.n_electrons, degeneracy=1.0)
    occ, _ = occupations_and_entropy(eps, mu, SCHEMES[scheme], width,
                                     degeneracy=1.0)
    return [occ[ik] for ik in range(eps.shape[0])]


def _apply_dispersion(res: SCFLike, inp: Input) -> dict[str, Any]:
    """Compute the D3(BJ)/D4(BJ) correction, fold its energy into ``res.energies``
    (so the reported total/free energy include it), and return the summary block
    (energy, forces, stress, resolved damping). ``dispersion.method`` selects
    ``'d3'`` (default) or ``'d4'`` (charge-dependent EEQ C6, periodic Ewald charge
    model). Degrades to ``{'available': False}`` when the element set is uncovered
    or no BJ preset exists for the functional."""
    import numpy as np
    import torch

    dp = inp.dispersion
    method = dp.method.lower()
    system = _get(res, "system")
    positions = system.positions.detach().to(torch.float64)
    cell = np.asarray(system.grid.cell, dtype=np.float64)
    z = [int(v) for v in inp.atoms.get_atomic_numbers()]
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
                dp.functional or inp.xc, charge=dp.charge,
                cutoff_ang=dp.cutoff, cn_cutoff_ang=dp.cn_cutoff,
                s6=dp.s6, s8=dp.s8, a1=dp.a1, a2=dp.a2,
            )
            cell_t = torch.as_tensor(cell, dtype=torch.float64, device=positions.device)
            e = dispersion_energy(positions, cell_t, z, cfg_d4)
            forces = dispersion_forces(positions, cell, z, cfg_d4)
            stress = dispersion_stress(positions, cell, z, cfg_d4)
            cfg = cfg_d4
        else:
            from gradwave.postscf.dispersion import (
                D3Config,
                dispersion_energy,
                dispersion_forces,
                dispersion_stress,
            )
            cfg_d3 = D3Config.resolve(
                dp.functional or inp.xc,
                cutoff_ang=dp.cutoff, cn_cutoff_ang=dp.cn_cutoff,
                s6=dp.s6, s8=dp.s8, a1=dp.a1, a2=dp.a2,
            )
            cell_t = torch.as_tensor(cell, dtype=torch.float64, device=positions.device)
            e = dispersion_energy(positions, cell_t, z, cfg_d3)
            forces = dispersion_forces(positions, cell, z, cfg_d3)
            stress = dispersion_stress(positions, cell, z, cfg_d3)
            cfg = cfg_d3
    except (ValueError, NotImplementedError) as err:
        return {"available": False, "reason": str(err)}

    # fold the energy into the breakdown; total/free_energy pick it up
    res.energies.dispersion = e.detach().to(positions.device)
    return {
        "available": True,
        "method": f"{method}-bj",
        "functional": (dp.functional or inp.xc).lower(),
        "damping": {"s6": cfg.s6, "s8": cfg.s8, "a1": cfg.a1, "a2_bohr": cfg.a2},
        "energy_eV": float(e),
        "energy_per_atom_eV": float(e) / len(z),
        "forces_eV_ang": forces.detach().cpu().tolist(),
        "stress_eV_ang3": stress.detach().cpu().tolist(),
    }


def _pdos_summary_block(res: SCFLike, inp: Input) -> dict[str, Any]:
    """Löwdin projected-DOS block for the summary JSON. Returns a graceful
    ``{'available': False, ...}`` when the pseudopotentials omit PP_PSWFC."""
    from gradwave.postscf.pdos import projected_dos
    p = inp.projections
    try:
        # projected_dos declares SCFResult | USPPResult and raises
        # NotImplementedError for anything else (NCResult/USPPNCResult) via
        # its own _unpack_result — caught below, so the wider SCFLike here
        # is a safe runtime seam, not a real type mismatch.
        block = projected_dos(cast("SCFResult | USPPResult", res),
                              group_by=p.group_by, width=p.width,
                              npoints=p.npoints).to_dict()
        return block
    except (ValueError, NotImplementedError) as err:
        return {"available": False, "reason": str(err)}


def _cohp_summary_block(res: SCFLike, inp: Input) -> dict[str, Any]:
    """Crystal Orbital Hamilton Population block for the summary JSON, computed
    alongside the PDOS when ``projections.cohp`` is enabled. Returns a graceful
    ``{'available': False, ...}`` when the pseudopotentials omit PP_PSWFC or the
    formalism is out of coverage."""
    from gradwave.postscf.cohp import cohp
    c = inp.projections.cohp
    pairs = None if c.pairs is None else [tuple(p) for p in c.pairs]
    try:
        # cohp declares SCFResult | USPPResult and raises NotImplementedError
        # for anything else via its own _unpack_result — caught below (the
        # noncollinear/SOC cohp_noncollinear/cohp_soc entry points aren't
        # wired into this summary path yet), so the wider SCFLike here is a
        # safe runtime seam, not a real type mismatch.
        block = cohp(cast("SCFResult | USPPResult", res), pairs=pairs,
                     rcut=c.rcut, width=c.width, npoints=c.npoints).to_dict()
        block["available"] = True
        return block
    except (ValueError, NotImplementedError) as err:
        return {"available": False, "reason": str(err)}


def run_magnetism(inp: Input, verbose: bool = True) -> MagneticReport:
    """Characterize the magnetism of the input system (task: magnetism). Builds a
    non-collinear XC from inp.xc, runs `characterize_magnetism`, and returns the
    MagneticReport."""
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.postscf.magnetism import characterize_magnetism

    system = build_system(inp)
    if inp.device != "cpu":
        system = system.to(inp.device)
    xc = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())
    m = inp.magnetism
    smtype = inp.smearing.type if inp.smearing.type != "none" else "gaussian"
    return characterize_magnetism(
        system, xc, exchange=m.exchange, ref_atom=m.ref_atom, lam=m.lam,
        delta=m.delta, seed_scale=m.seed_scale, smearing=smtype,
        width=inp.smearing.width, max_iter=inp.scf.max_iter, etol=inp.scf.etol,
        rhotol=inp.scf.rhotol, mixing_alpha=inp.scf.mixing.alpha, verbose=verbose)


def _write_volumetric(
    res: SCFLike, spec: VolumetricParams, outdir: Path, verbose: bool
) -> dict[str, Any]:
    """Write the requested volumetric fields (.cube/.xsf/CHGCAR) and return an
    {label: filename} map for summary["outputs"]. A field that the result type
    does not support (e.g. ELF on a noncollinear run) is skipped with a warning
    rather than losing the finished run."""
    from gradwave.postscf import volumetric as vol

    ext = "." + spec.format
    jobs = []
    if spec.density:
        jobs.append(("density", f"density{ext}", lambda p: vol.write_density(res, p)))
    if spec.elf:
        jobs.append(("elf", f"elf{ext}", lambda p: vol.write_elf(res, p)))
    if spec.magnetization:
        jobs.append(("magnetization", f"magnetization{ext}",
                     lambda p: vol.write_magnetization(res, p)))
    for band, kpt in spec.bands:
        label = f"parchg_b{band}_k{kpt}"
        jobs.append((label, f"{label}{ext}",
                     lambda p, b=band, k=kpt: vol.write_band_density(res, p, band=b, kpoint=k)))

    written = {}
    for label, name, write in jobs:
        try:
            produced = write(outdir / name)
            # a writer may emit several files (e.g. spin-resolved ELF → up/dn);
            # record their actual names, else the single fixed name
            written[label] = ([Path(p).name for p in produced]
                              if isinstance(produced, (list, tuple)) else name)
        except (NotImplementedError, ValueError) as exc:
            if verbose:
                print(f"skipped {label}: {exc}")
    return written


def _base_summary(inp: Input, task: str) -> dict[str, Any]:
    """The lightweight summary scaffold shared by tasks that carry no
    SCFResult (relax, magnetism, eos, elastic, phonons). SCF-derived tasks
    use build_summary() instead. Callers append their per-task result block
    and a trailing "runtime_s" so the serialized key order stays
    code/task/structure/parameters/<block>/runtime_s."""
    from gradwave import __version__

    return {
        "code": {"name": "gradwave", "version": __version__,
                 "created": datetime.datetime.now().isoformat(
                     timespec="seconds")},
        "task": task,
        "structure": _structure_block(inp),
        "parameters": _parameters_block(inp),
    }


# post-SCF tasks whose run() branch is a bare "run it, wrap the result" —
# collapsed into one data-driven branch below
_POSTSCF_RUNNERS = {"eos": run_eos, "elastic": run_elastic,
                    "phonons": run_phonons}


def run(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Execute inp.task and write <task>.json, <task>.out and (for SCF
    state) checkpoint.pt into inp.output_dir.

    Under ``inp.distributed`` (a torchrun launch), every rank computes the
    identical, correctly-reduced result (see ``scf.loop.scf``'s ``dist_ctx``
    handling), so file writing below is gated to rank 0 to avoid every rank
    racing to write the same files."""
    from gradwave.output import write_output
    from gradwave.runinfo import ProcessMeter, machine_snapshot, provenance_block

    if inp.distributed and inp.task not in ("scf", "bands"):
        raise NotImplementedError(
            f"distributed: true is only wired for task: scf | bands so far "
            f"(got task: {inp.task!r}) — relax/eos/elastic/phonons/magnetism "
            f"don't route through the k-point-sharded SCF path yet (see "
            f"docs/manual/distributed.md)"
        )
    snap = machine_snapshot()
    meter = ProcessMeter()
    t0 = time.time()
    res = None
    _frames = None
    if inp.task == "scf":
        res = run_scf(inp, verbose=verbose)
        disp_block = _apply_dispersion(res, inp) if inp.dispersion.enabled else None
        summary = build_summary(res, inp, "scf", runtime_s=time.time() - t0)
        if disp_block is not None:
            summary["dispersion"] = disp_block
        if inp.error_estimate:
            summary["error_estimate"] = _error_estimate_block(res, inp)
        if inp.projections.enabled:
            summary["pdos"] = _pdos_summary_block(res, inp)
        if inp.projections.cohp.enabled:
            summary["cohp"] = _cohp_summary_block(res, inp)
    elif inp.task == "relax":
        relax, _atoms, _frames = run_relax(inp, verbose=verbose)
        summary = _base_summary(inp, "relax")
        summary["relax"] = relax
        summary["runtime_s"] = round(time.time() - t0, 2)
        # error estimate at the FINAL geometry: the calculator caches its
        # last converged SCF, so the estimate describes the relaxed state
        res_final = getattr(_atoms.calc, "last_result", None)
        if inp.error_estimate and res_final is not None:
            summary["error_estimate"] = _error_estimate_block(res_final, inp)
    elif inp.task == "bands":
        res = run_scf(inp, verbose=verbose)
        summary = build_summary(res, inp, "bands",
                                extra=_bands_extra(inp, res, verbose),
                                runtime_s=time.time() - t0)
        if inp.error_estimate:
            summary["error_estimate"] = _error_estimate_block(res, inp)
    elif inp.task == "magnetism":
        # no error_estimate block here: magnetism runs are spinor SCFs,
        # outside every estimator's coverage (it would always be
        # available: false)
        report = run_magnetism(inp, verbose=verbose)
        if verbose:
            print(report.summary())
        summary = _base_summary(inp, "magnetism")
        summary["magnetism"] = {
            "ordering": report.ordering,
            "total_moment_muB": round(report.total_moment, 4),
            "atomic_moments_muB": report.moment_magnitudes,
            "moment_vectors_muB": report.moment_vectors,
            "exchange_J_meV": None if report.exchange_J is None else
            {str(i): round(J * 1000, 3) for i, J in report.exchange_J.items()},
            "dmi_meV": None if report.dmi is None else
            {str(i): round(d * 1000, 4) for i, d in report.dmi.items()},
            "curie_temperature_mfa_K": None if report.curie_temperature_mfa is None
            else round(report.curie_temperature_mfa),
        }
        summary["runtime_s"] = round(time.time() - t0, 2)
    elif inp.task in _POSTSCF_RUNNERS:
        block = _POSTSCF_RUNNERS[inp.task](inp, verbose=verbose)
        summary = _base_summary(inp, inp.task)
        summary[inp.task] = block
        summary["runtime_s"] = round(time.time() - t0, 2)
    else:
        raise ValueError(
            f"unknown task {inp.task!r} "
            f"(scf | relax | bands | magnetism | eos | elastic | phonons)")

    if inp.distributed:
        from gradwave.distributed import current_rank, maybe_destroy_process_group

        # All collective communication finished inside run_scf's result gather;
        # from here on every rank works purely locally (file writing below is
        # rank-0-only). Tear the process group down now so a torchrun launch
        # exits cleanly instead of leaking it (#216). Guarded (no-op if never
        # initialized), so the single-process path never touches it.
        maybe_destroy_process_group()
        if current_rank() != 0:
            return summary

    outdir = Path(inp.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if res is not None and inp.output_checkpoint:
        from gradwave.checkpoint import save_checkpoint

        ck = save_checkpoint(res, outdir / "checkpoint.pt",
                             wavefunctions=inp.output_wavefunctions)
        outputs["checkpoint"] = ck.name
    if inp.task == "relax" and _frames:
        from ase.io import write as ase_write

        ase_write(str(outdir / "relax.xyz"), _frames, format="extxyz")
        outputs["trajectory"] = "relax.xyz"
    if res is not None and inp.output_volumetric.any():
        outputs.update(_write_volumetric(res, inp.output_volumetric, outdir, verbose))
    # SCF flight-recorder sidecar: the full per-iteration trace, opted in via
    # scf.trace. The compact diagnostics ride in the JSON regardless; this is the
    # heavy per-iteration corpus (schema-versioned scf_trace.json). Rank-0 only,
    # inheriting the gate above.
    _rec = getattr(res, "recorder", None) if res is not None else None
    if inp.scf.trace and _rec is not None and getattr(_rec, "iters", None):
        (outdir / "scf_trace.json").write_text(json.dumps(_rec.to_trace_dict(), indent=1))
        outputs["scf_trace"] = "scf_trace.json"
    summary["provenance"] = provenance_block(snap, meter)
    summary["outputs"] = {**outputs, "json": f"{inp.task}.json",
                          "report": f"{inp.task}.out"}
    (outdir / f"{inp.task}.json").write_text(json.dumps(summary, indent=1))
    write_output(summary, outdir / f"{inp.task}.out")
    if verbose:
        print(f"wrote {outdir / inp.task}.json / .out"
              + (" / checkpoint.pt" if "checkpoint" in outputs else ""))
    return summary


def _structure_block(inp: Input) -> dict[str, Any]:
    import numpy as np

    vol = float(abs(np.linalg.det(inp.atoms.cell.array)))
    block = {
        "cell_ang": inp.atoms.cell.array.tolist(),
        "positions_ang": inp.atoms.get_positions().tolist(),
        "species": inp.atoms.get_chemical_symbols(),
        "n_atoms": len(inp.atoms),
        "volume_ang3": vol,
        # 1 amu/Å³ = 1.66053906660 g/cm³
        "density_g_cm3": float(inp.atoms.get_masses().sum() * 1.66053906660 / vol),
    }
    try:
        import spglib
    except ImportError:
        return block
    try:
        ds = spglib.get_symmetry_dataset(
            (inp.atoms.cell.array, inp.atoms.get_scaled_positions(),
             inp.atoms.get_atomic_numbers()), symprec=1e-5)
        if ds is None:
            # a degenerate/near-singular cell; drop the field rather than
            # swallow an unrelated bug below
            return block
        block["spacegroup"] = f"{ds.international} ({ds.number})"
        block["pointgroup"] = ds.pointgroup
        block["n_symops"] = len(ds.rotations)
    except (TypeError, spglib.SpglibError):
        pass
    return block


def _parameters_block(inp: Input) -> dict[str, Any]:
    """Parameters block for the tasks written without a materialized System
    (relax, magnetism). The magnetism run is always the non-collinear/spinor
    formalism (matching build_summary's convention); relax follows the
    pseudopotential family."""
    import math

    species, upfs, _soa = _species_upfs(inp)
    if inp.task == "magnetism":
        formalism = "noncollinear"
    else:
        formalism = "uspp/paw" if _is_uspp(upfs) else "nc"
    return {
        "formalism": formalism,
        "xc": _xc_label(inp),
        "ecut_eV": float(inp.ecut),
        "ecutrho_eV": None,
        "kmesh": list(inp.kpoints.mesh),
        "nk": None,
        "nk_total": int(math.prod(inp.kpoints.mesh)),
        "kweights": None,
        "nspin": inp.nspin,
        "smearing": inp.smearing.type,
        "width_eV": float(inp.smearing.width),
        "symmetry": bool(inp.symmetry),
        "mixing": {
            "scheme": inp.scf.mixing.scheme,
            "alpha": float(inp.scf.mixing.alpha),
            "history": inp.scf.mixing.history,
            "kerker": inp.scf.mixing.kerker,
            "kerker_used": None,
            "precond": inp.scf.mixing.precond,
        },
        "pseudos": {s: inp.pseudo_map[s] for s in species},
    }
