"""UPF loading (path-cached) and Layer-B system construction."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from gradwave.api._common import SPIN_XC_REGISTRY, time_reversal_ok
from gradwave.core.xc.spin import SpinXC
from gradwave.inputs import Input

if TYPE_CHECKING:

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


def build_system(inp: Input) -> System | USPPSystem:
    """The Layer-B system for this input, NC or USPP/PAW by UPF kind."""
    species, upfs, species_of_atom = _species_upfs(inp)
    # Slab vacuum auto-sizer (opt-in, ESM only): trim the open axis to the SAD
    # density tail before the grid is built, so the box is frozen for the whole
    # solve. A no-op unless enabled and both gates pass (see api._slab).
    from gradwave.api._slab import resolve_slab_box

    box = resolve_slab_box(inp, upfs, species_of_atom)
    cell, positions = box.cell, box.positions
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
        cell=cell,
        positions=positions,
        species_of_atom=species_of_atom,
        upfs=_as_upfs(upfs),
        ecut=inp.ecut,
        kmesh=inp.kpoints.mesh,
        kshift=inp.kpoints.shift,
        nbands=inp.nbands,
        use_symmetry=inp.symmetry and not hybrid and not hubbard,
        time_reversal=not hybrid and time_reversal_ok(inp),
    )


def _spin_setup(inp: Input) -> tuple[SpinXC, list[float]]:
    xc = SPIN_XC_REGISTRY[inp.xc]()
    symbols = inp.atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    mags = [float((inp.start_mag or {}).get(s, 0.5)) for s in species]
    return xc, mags
