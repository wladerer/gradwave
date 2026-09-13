"""UPF loading (path-cached) and Layer-B system construction."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gradwave.api._common import SPIN_XC_REGISTRY, time_reversal_ok
from gradwave.core.xc.spin import SpinXC
from gradwave.inputs import Input

if TYPE_CHECKING:
    from ase import Atoms

    from gradwave.core.hubbard import HubbardManifold
    from gradwave.grids import FFTGrid
    from gradwave.pseudo.upf import UPFData
    from gradwave.pseudo.upf_paw import PAWData
    from gradwave.scf.loop import System
    from gradwave.scf.uspp_setup import USPPSystem

logger = logging.getLogger(__name__)


# UPFs are static for a run; cache by path so build_summary / run_scf /
# _error_estimate_block / _parameters_block parse each pseudo once, not 3-4×
_UPF_CACHE: dict[str, UPFData | PAWData] = {}


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


def build_scaled_system(
    inp: Input,
    upfs: list[UPFData | PAWData],
    uspp: bool,
    species_of_atom: list[int],
    cell: Any,
    positions: Any,
    *,
    fft_shape: tuple[int, ...] | None,
    time_reversal: bool = True,
) -> System | USPPSystem:
    """Build the System/USPPSystem for a given (cell, positions) on a PINNED FFT
    grid — the shared body behind the eos volume spokes and the elastic strain
    spokes. Both scan a deformed cell at fixed ecut/kmesh/nbands and must reuse
    one FFT box across the scan (so E(V)/σ(ε) carry no grid-discontinuity step),
    which is exactly what ``fft_shape`` pins. NC vs USPP/PAW by the ``uspp`` flag.

    This is a module-level function (not a closure) so the SeedPool worker path
    can call it in a spawned process — it must stay picklable. The serial
    closures and the workers in both drivers route through here.

    ``time_reversal`` is threaded to the norm-conserving ``setup_system`` only
    (the USPP/PAW path derives Kramers internally): the isotropic eos scan
    leaves it at the default True, while the elastic scan passes the
    spinor-aware value from ``time_reversal_ok``."""
    if uspp:
        from gradwave.scf.uspp import setup_uspp

        return setup_uspp(
            cell, positions, species_of_atom, _as_paws(upfs), ecut=inp.ecut,
            kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho, nbands=inp.nbands,
            use_symmetry=inp.symmetry, fft_shape=fft_shape)
    from gradwave.scf.loop import setup_system

    return setup_system(
        cell=cell, positions=positions, species_of_atom=species_of_atom,
        upfs=_as_upfs(upfs), ecut=inp.ecut, kmesh=inp.kpoints.mesh,
        kshift=inp.kpoints.shift, nbands=inp.nbands,
        use_symmetry=inp.symmetry, time_reversal=time_reversal,
        fft_shape=fft_shape)


def _resolve_kmesh(inp: Input) -> tuple[int, int, int]:
    """The effective Monkhorst–Pack mesh for this input.

    ``kpoints.kspacing`` (when set) derives a slab-aware anisotropic mesh from the
    cell via ``kpoints.slab_kmesh`` — the detected vacuum axis is pinned to a
    single Γ point. Otherwise the explicit ``kpoints.mesh`` is used verbatim."""
    if inp.kpoints.kspacing is None:
        return tuple(inp.kpoints.mesh)
    from gradwave.kpoints import slab_kmesh

    return slab_kmesh(
        inp.atoms.cell.array, inp.atoms.get_positions(),
        float(inp.kpoints.kspacing))


def build_system(inp: Input) -> System | USPPSystem:
    """The Layer-B system for this input, NC or USPP/PAW by UPF kind."""
    species, upfs, species_of_atom = _species_upfs(inp)
    # Slab vacuum auto-sizer (opt-in, ESM only): trim the open axis to the SAD
    # density tail before the grid is built, so the box is frozen for the whole
    # solve. A no-op unless enabled and both gates pass (see api._slab).
    from gradwave.api._slab import resolve_slab_box

    box = resolve_slab_box(inp, upfs, species_of_atom)
    cell, positions = box.cell, box.positions
    kmesh = _resolve_kmesh(inp)
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
            cell, positions, species_of_atom,
            _as_paws(upfs), ecut=inp.ecut, kmesh=kmesh,
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
        cell=cell,
        positions=positions,
        species_of_atom=species_of_atom,
        upfs=_as_upfs(upfs),
        ecut=inp.ecut,
        kmesh=kmesh,
        kshift=inp.kpoints.shift,
        nbands=inp.nbands,
        use_symmetry=inp.symmetry and not hybrid and not hubbard,
        time_reversal=not hybrid and time_reversal_ok(inp),
    )


def trim_slab_vacuum(
    atoms: Atoms,
    pseudopotentials: dict[str, str],
    ecut: float,
    *,
    boundary: str = "open_z",
    target: str = "energy",
    min_vacuum: float = 3.0,
    vacuum_tol: float | None = None,
    vacuum_margin: float | None = None,
    npw_gate: int = 8000,
    vacuum_fraction_gate: float = 0.3,
) -> Atoms:
    """Return a copy of ``atoms`` with the open (vacuum) axis trimmed to the SAD
    density tail — the ASE-facing entry to the slab vacuum auto-sizer.

    Under an open-boundary (ESM) run the vacuum-normal electrostatics are
    box-independent *where the density has decayed to zero at the box edge*, so
    excess vacuum beyond the physical tail is FFT/plane-wave waste. Call this once
    on a slab (or slab+adsorbate) built with generous vacuum, then hand the
    returned Atoms to :class:`~gradwave.calculator.GradWave` with the *same*
    ``boundary`` and ``ecut``: every downstream SCF, force and (fixed-cell)
    relaxation step then runs in the smaller, consistent box. The trim is a
    forward hyperparameter (set once, frozen for the solve, like ``ecut``) — the
    vacuum-normal stress is meaningless for a slab, so there is no
    autograd/relaxation coupling as long as the cell is held fixed.

    **This is a controllable approximation, not an exact transform.** The box
    edge is placed where the *SAD* planar density falls below ``vacuum_tol``;
    trimming more aggressively (a looser ``vacuum_tol``, needed before the box
    actually shrinks for a diffuse-tail material like Al) moves the ESM boundary
    into non-zero density and shifts the total energy — measured on Al: ~tens of
    meV/atom at a ~2× npw cut, hundreds of meV/atom to > 1 eV beyond. Validate
    the energy error for your material and pick ``vacuum_tol`` for the box-size /
    accuracy tradeoff you want. The default (``target="energy"`` → 1e-4 e/Å³) is
    conservative — for a light delocalised metal it often declines to trim rather
    than corrupt the energy. Returns the input unchanged (no trim) unless
    ``boundary`` is open *and* both gates pass (arithmetic-bound
    ``npw ≳ npw_gate``, vacuum-dominated ``vacuum_fraction ≳ vacuum_fraction_gate``).
    ``target="workfunction"`` uses the conservative plateau-limited margin.

    ``ecut`` is in eV (the same units the ASE calculator takes). Mirrors the
    ``slab.vacuum_autosize`` knob on the ``Input``/api path (``api._slab``)."""
    from ase import Atoms as _Atoms

    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    upfs = [_load_upf(pseudopotentials[s]) for s in species]
    species_of_atom = [species.index(s) for s in symbols]

    from gradwave.api._slab import resolve_slab_box_geom

    box = resolve_slab_box_geom(
        atoms.cell.array,
        atoms.get_positions(),
        boundary=boundary,
        ecut=ecut,
        upfs=upfs,
        species_of_atom=species_of_atom,
        vacuum_autosize=True,
        vacuum_target=target,
        vacuum_tol=vacuum_tol,
        vacuum_margin=vacuum_margin,
        min_vacuum=min_vacuum,
        npw_gate=npw_gate,
        vacuum_fraction_gate=vacuum_fraction_gate,
    )
    if not box.trimmed:
        logger.info("trim_slab_vacuum: no trim (%s)", box.reason)
        return atoms
    trimmed = _Atoms(
        symbols=symbols,
        positions=box.positions,
        cell=box.cell,
        pbc=atoms.pbc,
    )
    # carry constraints/tags/momenta so an ASE relaxation (FixAtoms on the slab,
    # add_adsorbate tags) keeps working unchanged in the trimmed box.
    trimmed.set_constraint(atoms.constraints)
    trimmed.set_tags(atoms.get_tags())
    return trimmed


def _spin_setup(inp: Input) -> tuple[SpinXC, list[float]]:
    xc = SPIN_XC_REGISTRY[inp.xc]()
    symbols = inp.atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    mags = [float((inp.start_mag or {}).get(s, 0.5)) for s in species]
    return xc, mags
