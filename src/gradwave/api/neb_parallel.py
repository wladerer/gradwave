"""SeedPool-parallel CI-NEB band (Increment 1).

The interior images' SCFs are independent within a band step, so they fan out
across worker processes (``postscf.seedpool.map_spokes``) exactly like the
phonon displacement spokes. Each worker runs one image's forward SCF —
warm-started from that image's own checkpoint saved on the previous band step —
and returns its energy and forces. The band is then advanced with the pure
projector ``gradwave.opt.neb.neb_forces`` (validated against analytic surfaces)
and a FIRE step.

forward-only: worker autograd graphs do not cross the process boundary, so this
path returns plain numbers; a differentiable NEB must use the serial
(``n_workers=1``) driver. Norm-conserving / USPP-PAW, nspin=1 only in v1 (the
worker rebuilds a spin-unpolarized SCF); a spin-polarized request runs serially
(handled by the caller).

The worker uses ``use_symmetry=False`` for every image: a displaced band image
generally breaks the crystal symmetry, so reducing its k-set by the ideal
space group would be unsound. This is a k-sampling choice only; the converged
forces agree with the serial (calculator) path to SCF tolerance.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gradwave.api._common import _DEFAULT_MIXING_HISTORY, XC_REGISTRY
from gradwave.api.system import _as_paws, _as_upfs, _is_uspp, _species_upfs
from gradwave.inputs import Input

if TYPE_CHECKING:
    from ase import Atoms
    from ase.mep import NEB

logger = logging.getLogger(__name__)


class _NebSpoke(NamedTuple):
    inp: Input
    cell: Any                 # (3,3) fixed cell [Å], numpy
    species_of_atom: list[int]
    positions: Any            # (natoms, 3) image positions [Å], numpy
    ckpt_in: str | None       # warm-start checkpoint path (None = cold)
    ckpt_out: str            # where this worker writes its checkpoint for reuse
    index: int               # image index in the band


def _neb_image_worker(spoke: _NebSpoke) -> tuple[int, float, Any]:
    """One image SCF → ``(index, free_energy_eV, forces[natoms,3])``.

    Warm-starts from ``ckpt_in`` when present and writes its converged state to
    ``ckpt_out`` so the next band step reuses it (per-image density warm-start,
    the parallel analogue of the serial calculator's cross-step extrapolation).
    """
    from gradwave.io.checkpoint import (
        as_start_from,
        load_checkpoint,
        save_checkpoint,
    )
    from gradwave.postscf.forces import forces as nc_forces
    from gradwave.scf.loop import scf, setup_system

    inp = spoke.inp
    _species, upfs, _soa = _species_upfs(inp)
    if _is_uspp(upfs):
        from gradwave.postscf.paw_forces import forces_uspp
        from gradwave.scf.uspp import scf_uspp, setup_uspp

        usystem = setup_uspp(
            spoke.cell, spoke.positions, spoke.species_of_atom, _as_paws(upfs),
            ecut=inp.ecut, kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho,
            use_symmetry=False)
        if inp.device != "cpu":
            usystem = usystem.to(inp.device)
        start = (as_start_from(load_checkpoint(spoke.ckpt_in))
                 if spoke.ckpt_in else None)
        res = scf_uspp(
            usystem, XC_REGISTRY[inp.xc](), nspin=1,
            smearing=inp.smearing.type, width=inp.smearing.width,
            etol=inp.scf.etol, rhotol=inp.scf.rhotol,
            mixing_alpha=inp.scf.mixing.alpha,
            mixing_history=inp.scf.mixing.history,
            diago_tol=inp.scf.diago_tol, start_from=start, verbose=False)
        f = forces_uspp(res, XC_REGISTRY[inp.xc]())
    else:
        system = setup_system(
            spoke.cell, spoke.positions, spoke.species_of_atom, _as_upfs(upfs),
            ecut=inp.ecut, kmesh=inp.kpoints.mesh, kshift=inp.kpoints.shift,
            use_symmetry=False)
        if inp.device != "cpu":
            system = system.to(inp.device)
        start = (as_start_from(load_checkpoint(spoke.ckpt_in))
                 if spoke.ckpt_in else None)
        res = scf(
            system, XC_REGISTRY[inp.xc](), nspin=1,
            smearing=inp.smearing.type, width=inp.smearing.width,
            etol=inp.scf.etol, rhotol=inp.scf.rhotol,
            mixing_alpha=inp.scf.mixing.alpha,
            mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
            precond=inp.scf.mixing.precond, diago_tol=inp.scf.diago_tol,
            start_from=start, verbose=False)
        f = nc_forces(res)
    save_checkpoint(res, spoke.ckpt_out)
    energy = float(res.energies.free_energy)
    return spoke.index, energy, f.detach().cpu().numpy()


def run_neb_parallel(
    inp: Input, images: list[Atoms], band: NEB, verbose: bool = True,
) -> tuple[dict[str, Any], list[Atoms]]:
    """FIRE-drive the interpolated ``band`` with per-step parallel image SCFs."""
    from gradwave.api.neb import _frames, _mep_block
    from gradwave.opt.neb import neb_forces
    from gradwave.postscf.seedpool import map_spokes

    p = inp.neb
    n = len(images)
    _species, _upfs, species_of_atom = _species_upfs(inp)
    cell = np.asarray(inp.atoms.cell.array, dtype=float)
    pos = np.array([img.get_positions() for img in images])  # (n, natoms, 3)

    # FIRE state on the interior images only (endpoints held fixed)
    vel = np.zeros_like(pos)
    dt, dt_max = 0.1, 1.0
    alpha, alpha0 = 0.1, 0.1
    finc, fdec, falpha, n_min = 1.1, 0.5, 0.99, 5
    npos = 0

    with tempfile.TemporaryDirectory(prefix="gw_neb_") as td:
        ckpt_in: list[str | None] = [None] * n
        ckpt_a = [os.path.join(td, f"img_{i}_a.ckpt") for i in range(n)]
        ckpt_b = [os.path.join(td, f"img_{i}_b.ckpt") for i in range(n)]
        energies = np.zeros(n)
        forces = np.zeros_like(pos)
        t0 = time.perf_counter()
        converged = False
        fmax = float("inf")
        step = 0

        def _evaluate(use_a: bool) -> None:
            # evaluate ALL images (endpoints included) so the band has endpoint
            # energies; write checkpoints to the alternating slot for reuse.
            ck_out = ckpt_a if use_a else ckpt_b
            spokes = [
                _NebSpoke(inp, cell, species_of_atom, pos[i], ckpt_in[i],
                          ck_out[i], i)
                for i in range(n)
            ]
            out = map_spokes(_neb_image_worker, spokes,
                             n_workers=p.n_workers, verbose=verbose)
            for idx, e, f in out:
                energies[idx] = e
                forces[idx] = f
                ckpt_in[idx] = ck_out[idx]

        for step in range(p.max_steps):
            _evaluate(use_a=(step % 2 == 0))
            proj = neb_forces(pos, energies, forces, p.spring_k, climb=p.climb)
            fmax = float(np.linalg.norm(proj.reshape(n, -1), axis=1).max())
            if verbose:
                e_barrier = float(np.max(energies)) - float(energies[0])
                print(f"  neb step {step:>3d} · E_barrier = {e_barrier:+.4f} eV"
                      f" · fmax = {fmax:.4f} eV/Å · "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)
            if fmax < p.fmax:
                converged = True
                break
            # FIRE integration on the projected forces
            pdot = float((proj * vel).sum())
            if pdot > 0.0:
                npos += 1
                vnorm = np.linalg.norm(vel)
                fnorm = np.linalg.norm(proj)
                if fnorm > 0:
                    vel = (1.0 - alpha) * vel + alpha * (vnorm / fnorm) * proj
                if npos > n_min:
                    dt = min(dt * finc, dt_max)
                    alpha *= falpha
            else:
                npos = 0
                dt *= fdec
                alpha = alpha0
                vel[:] = 0.0
            vel = vel + dt * proj
            pos = pos + dt * vel  # endpoints have zero projected force → fixed

        for i, img in enumerate(images):
            img.set_positions(pos[i])

    e_ini, e_fin = float(energies[0]), float(energies[-1])
    # the parallel path always FIRE-integrates the projected forces, regardless of
    # inp.neb.optimizer (which only drives the serial ASE optimizer).
    block = _mep_block(images, [float(e) for e in energies], e_ini, e_fin,
                       fmax, converged, step + 1, inp, optimizer="fire")
    if verbose:
        print(f"neb: barrier E_a = {block['barrier_eV']:+.4f} eV "
              f"(TS = image {block['ts_image']}/{n - 1}), reaction ΔE = "
              f"{block['reaction_energy_eV']:+.4f} eV — "
              f"{'converged' if converged else 'NOT converged'} "
              f"[{p.n_workers}-worker SeedPool]", flush=True)
    frames = _frames(images, [float(e) for e in energies], forces=forces)
    return block, frames
