"""run_eos and its parallel spoke workers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from gradwave.api.scf import run_scf
from gradwave.api.system import _as_paws, _as_upfs, _fft_grid, _is_uspp, _species_upfs
from gradwave.inputs import Input

if TYPE_CHECKING:

    from gradwave.scf.loop import System
    from gradwave.scf.uspp_setup import USPPSystem

logger = logging.getLogger(__name__)


# --- eos (volume spokes) ---------------------------------------------------
def _eos_rebuild(inp: Input) -> tuple[Any, bool, Any]:
    _species, upfs, soa = _species_upfs(inp)
    return upfs, _is_uspp(upfs), soa


def _eos_build(
    inp: Input, upfs: Any, uspp: bool, soa: Any, scale: float, fixed: Any,
) -> tuple[System | USPPSystem, Any]:
    """Isotropically-scaled system on the pinned grid (mirrors run_eos'
    ``_build_at``); returns ``(system, cell)``."""
    import numpy as np

    cell0 = np.asarray(inp.atoms.cell.array, dtype=float)
    cell = cell0 * scale ** (1.0 / 3.0)
    pos = inp.atoms.get_scaled_positions() @ cell
    if uspp:
        from gradwave.scf.uspp import setup_uspp

        return setup_uspp(
            cell, pos, soa, _as_paws(upfs), ecut=inp.ecut,
            kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho, nbands=inp.nbands,
            use_symmetry=inp.symmetry, fft_shape=fixed), cell
    from gradwave.scf.loop import setup_system

    return setup_system(
        cell=cell, positions=pos, species_of_atom=soa, upfs=_as_upfs(upfs),
        ecut=inp.ecut, kmesh=inp.kpoints.mesh, kshift=inp.kpoints.shift,
        nbands=inp.nbands, use_symmetry=inp.symmetry, fft_shape=fixed), cell


class _EosSpoke(NamedTuple):
    inp: Input
    scale: float
    fixed: Any
    ckpt_path: str
    idx: int


def _eos_spoke_worker(spoke: _EosSpoke) -> tuple[int, float, float, bool]:
    """Worker: build the scaled volume, warm-start from the reference volume's
    checkpoint, run the SCF and return ``(idx, volume, energy, converged)``."""
    import numpy as np

    from gradwave.io.checkpoint import as_start_from, load_checkpoint

    upfs, uspp, soa = _eos_rebuild(spoke.inp)
    system, cell = _eos_build(spoke.inp, upfs, uspp, soa, spoke.scale, spoke.fixed)
    start_from = as_start_from(load_checkpoint(spoke.ckpt_path))
    res = run_scf(spoke.inp, system=system, verbose=False, start_from=start_from)
    e = float(getattr(res.energies, spoke.inp.eos.energy))
    vol = float(abs(np.linalg.det(cell)))
    return spoke.idx, vol, e, bool(getattr(res, "converged", True))


def run_eos(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Isotropic volume scan + 3rd-order Birch-Murnaghan fit → V0, B0, B0'.

    For each factor in ``inp.eos.scales`` the cell is scaled isotropically
    (a → a·s^(1/3), fractional coordinates fixed) and the SCF re-converged. All
    volumes share ONE FFT grid, pinned to the elementwise max over the scan, so
    E(V) carries no grid-discontinuity steps; each volume warm-starts from the
    previous converged density (the cheap, branch-stable EOS chain). Returns the
    ``eos`` summary block.

    Under ``distributed: true`` each per-volume ``run_scf`` shards the k-mesh
    across ranks (the volume-scan chain is a series of warm-started SCFs, and
    ``run_scf`` already routes the sharded path). The FFT box is pinned across
    the scan, so the shard rebuilds deterministically per volume from
    ``(nk, rank, world_size)``. Every rank fits the identical E(V), so only the
    printing is quieted below.

    ``eos.n_workers > 1`` (SeedPool) evaluates the volumes across that many
    worker processes, each warm-started from the reference volume (nearest 1.0)
    rather than the serial neighbour chain, so E(V) matches the serial fit to SCF
    tolerance (not bit-for-bit). Forward-only: a differentiable EOS must keep
    ``n_workers=1``. Forced serial under ``distributed`` (k-sharding already owns
    the parallelism there)."""
    import numpy as np

    from gradwave.postscf.eos import EV_A3_TO_GPA, fit_bm3

    if inp.distributed:
        # every rank scans the identical (full-mesh-reduced) E(V); quiet all but
        # rank 0 to avoid world_size-fold duplicated per-volume lines
        from gradwave.distributed import current_rank

        verbose = verbose and current_rank() == 0
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

    ekind = inp.eos.energy
    # SeedPool routes the volumes through worker processes; forced serial under
    # `distributed` (that path already shards k across ranks — the two
    # parallelisms do not compose in v1).
    n_workers = 1 if inp.distributed else (inp.eos.n_workers or 1)

    def _eos_print(s: float, vol: float, e: float, conv: bool) -> None:
        if verbose:
            tag = "" if conv else "  (NOT converged)"
            print(f"  s={s:.3f}  V={vol / natoms:8.4f} Å³/at  "
                  f"E={e / natoms:+.6f} eV/at{tag}", flush=True)

    if n_workers > 1:
        # reference = the volume nearest 1.0 (cheapest cold start); every other
        # volume warm-starts from ITS checkpoint (a shared seed, not the serial
        # neighbour chain), so E(V) matches the serial fit to SCF tolerance.
        import os
        import tempfile

        from gradwave.io.checkpoint import save_checkpoint
        from gradwave.postscf.seedpool import map_spokes

        ref_idx = int(np.argmin([abs(s - 1.0) for s in scales]))
        ref_sys, ref_cell = _build_at(scales[ref_idx], fixed)
        ref = run_scf(inp, system=ref_sys, verbose=False, start_from=None)
        volumes = [0.0] * len(scales)
        energies = [0.0] * len(scales)
        converged = [False] * len(scales)
        volumes[ref_idx] = float(abs(np.linalg.det(ref_cell)))
        energies[ref_idx] = float(getattr(ref.energies, ekind))
        converged[ref_idx] = bool(getattr(ref, "converged", True))
        with tempfile.TemporaryDirectory(prefix="gw_seedpool_") as td:
            ckpt = os.path.join(td, "ref.ckpt")
            save_checkpoint(ref, ckpt)
            spokes = [_EosSpoke(inp, s, fixed, ckpt, i)
                      for i, s in enumerate(scales) if i != ref_idx]
            out = map_spokes(_eos_spoke_worker, spokes,
                             n_workers=n_workers, verbose=verbose)
        for idx, vol, e, conv in out:
            volumes[idx] = vol
            energies[idx] = e
            converged[idx] = conv
        for s, vol, e, conv in zip(scales, volumes, energies, converged,
                                   strict=True):
            _eos_print(s, vol, e, conv)
    else:
        prev = None
        volumes, energies, converged = [], [], []
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
            _eos_print(s, vol, e, conv)

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
