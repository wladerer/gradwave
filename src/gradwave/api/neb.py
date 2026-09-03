"""run_neb — climbing-image nudged-elastic-band transition states.

A band of images between two relaxed endpoints (IS, FS) is built by linear or
IDPP interpolation, each interior image is driven by its own GradWave
calculator, and the band is relaxed to the minimum-energy path with the
climbing-image NEB (improved-tangent). The reported barrier is
``E_a = E(TS) − E(IS)`` where the TS is the highest-energy image.

This is a *driver*: the band math is ASE's mature ``ase.mep.NEB`` (its
``improvedtangent`` method is the same formulation validated cheaply and
independently against analytic 2D surfaces in ``gradwave.opt.neb`` /
``tests/unit/test_neb_projector.py``), and the physics — forces, the slab stack
(slab k-mesh, per-image density warm-start across band steps via the ordinary
SCF calculator's extrapolation) — is the shipped SCF calculator. Each image
keeps its own calculator for the whole optimization, so its density warm-starts
from its previous band step exactly as an ionic relaxation warm-starts across
geometry steps.

``neb.n_workers > 1`` fans the interior images' SCFs out across worker processes
each band step (SeedPool), each warm-started from that image's own previous
checkpoint, and drives the band with the pure-function projector
``gradwave.opt.neb.neb_forces`` + FIRE. It is forward-only (worker autograd
graphs do not cross processes), so a differentiable NEB must stay at
``n_workers=1``.

Planned next increment (the differentiable flagship): ``dE_a/dλ`` for a
parameter λ (a Hubbard U, an XC coefficient) in ONE backward pass evaluated only
at the converged IS and TS — the geometry term vanishes by stationarity
(∂E/∂R = 0 there), so the barrier derivative is ∂E(TS)/∂λ − ∂E(IS)/∂λ with no
need to differentiate through the band relaxation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from gradwave.inputs import Input, InputError

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)


def _read_final(inp: Input) -> Atoms:
    """Load and validate the final-state geometry against the initial state."""
    from ase.io import read as ase_read

    p = inp.neb
    if p.final is None:  # guarded at parse time; defensive here
        raise InputError("task: neb requires neb.final")
    try:
        fin = ase_read(str(p.final), format=p.final_format, index=p.final_index)
    except FileNotFoundError:
        raise InputError(f"neb.final file not found: {p.final}") from None
    except Exception as e:  # ase raises a grab-bag of read errors
        raise InputError(f"could not read neb.final {p.final}: {e}") from None
    if isinstance(fin, list):  # a bad index selected multiple frames
        raise InputError(
            f"neb.final {p.final} selected {len(fin)} frames; set neb.final_index")
    ini = inp.atoms
    if len(fin) != len(ini):
        raise InputError(
            f"neb.final has {len(fin)} atoms but the initial structure has "
            f"{len(ini)} — the endpoints must share atom count and order")
    if fin.get_chemical_symbols() != ini.get_chemical_symbols():
        raise InputError(
            "neb.final species/order differ from the initial structure; the two "
            "endpoints must list the same atoms in the same order")
    if not np.allclose(np.asarray(fin.cell.array), np.asarray(ini.cell.array),
                       atol=1e-6):
        logger.warning(
            "neb.final cell differs from the initial cell; NEB assumes a fixed "
            "cell — using the initial cell for every image")
    return fin


def _mep_block(images: list[Atoms], energies: list[float],
               e_ini: float, e_fin: float, fmax: float, converged: bool,
               n_steps: int, inp: Input, optimizer: str | None = None) -> dict[str, Any]:
    """Assemble the MEP / barrier summary block from a relaxed band."""
    n = len(images)
    ts = int(np.argmax(energies))
    pos = [img.get_positions() for img in images]
    # reaction coordinate: cumulative inter-image Cartesian distance, normalized
    seg = [0.0] + [float(np.linalg.norm(pos[i] - pos[i - 1])) for i in range(1, n)]
    rc = np.cumsum(seg)
    rc = (rc / rc[-1]).tolist() if rc[-1] > 0 else rc.tolist()
    barrier = energies[ts] - e_ini
    return {
        "n_images": n,
        "interpolation": inp.neb.interpolation,
        "spring_k": inp.neb.spring_k,
        "climb": inp.neb.climb,
        "optimizer": optimizer if optimizer is not None else inp.neb.optimizer,
        "converged": bool(converged),
        "n_steps": int(n_steps),
        "fmax_target_eV_ang": inp.neb.fmax,
        "fmax_eV_ang": float(fmax),
        "ts_image": ts,
        "barrier_eV": float(barrier),
        "reverse_barrier_eV": float(energies[ts] - e_fin),
        "reaction_energy_eV": float(e_fin - e_ini),
        "energy_initial_eV": float(e_ini),
        "energy_final_eV": float(e_fin),
        "ts_is_endpoint": ts in (0, n - 1),
        "image_energies_eV": [float(e) for e in energies],
        "image_energies_rel_eV": [float(e - e_ini) for e in energies],
        "reaction_coordinate": rc,
        "ts_positions_ang": pos[ts].tolist(),
    }


def _frames(images: list[Atoms], energies: list[float],
            forces: np.ndarray | None = None) -> list[Atoms]:
    """One ASE frame per image, energy/forces frozen for an extxyz MEP write.

    ``forces`` supplies precomputed per-image forces (the parallel path, which already
    has them from the worker pool); when ``None`` they are recomputed from each image's
    attached calculator (the serial path)."""
    from ase.calculators.singlepoint import SinglePointCalculator

    out: list[Atoms] = []
    for i, img in enumerate(images):
        fr = img.copy()
        f = img.get_forces(apply_constraint=False) if forces is None else forces[i]
        fr.calc = SinglePointCalculator(fr, energy=float(energies[i]), forces=f)
        fr.info["image"] = i
        out.append(fr)
    return out


def run_neb(inp: Input, verbose: bool = True) -> tuple[dict[str, Any], list[Atoms]]:
    """Climbing-image NEB between the input (IS) and ``neb.final`` (FS).

    Returns ``(neb_block, frames)`` — the barrier/MEP summary and one ASE frame
    per image (energy + forces) for the extxyz trajectory the dispatcher writes
    to ``neb.xyz``.
    """
    from ase.mep import NEB
    from ase.optimize import BFGS, FIRE

    from gradwave.api.relax import _build_relax_calc

    p = inp.neb
    # the SeedPool workers rebuild an nspin=1 SCF only in v1, so force serial for
    # spin-polarized bands rather than silently computing spin-unpolarized energies.
    n_workers = p.n_workers
    if inp.nspin == 2 and n_workers > 1:
        logger.info("neb: n_workers>1 not wired for nspin=2 — running serially")
        n_workers = 1
    ini = inp.atoms.copy()
    fin = _read_final(inp)
    n = p.n_images

    # build the band and its initial path (endpoints fixed, interior copies)
    images: list[Atoms] = [ini]
    images += [ini.copy() for _ in range(n - 2)]
    images += [fin]
    band = NEB(images, k=p.spring_k, climb=p.climb, method="improvedtangent")
    band.interpolate(method=p.interpolation)
    if verbose:
        print(f"neb: {n}-image band ({p.interpolation} interpolation), "
              f"climbing image {'on' if p.climb else 'off'}, k={p.spring_k} "
              f"eV/Å², optimizer {p.optimizer}", flush=True)

    if n_workers > 1:
        from gradwave.api.neb_parallel import run_neb_parallel

        return run_neb_parallel(inp, images, band, verbose=verbose)

    # attach one fresh calculator per image — each retains its own density
    # warm-start history across the band's optimization steps
    for img in images:
        img.calc = _build_relax_calc(inp, verbose=False)
    # endpoint energies (computed once; the endpoints are never displaced)
    e_ini = float(images[0].get_potential_energy())
    e_fin = float(images[-1].get_potential_energy())

    opt_cls = {"fire": FIRE, "bfgs": BFGS}[p.optimizer]
    # ASE optimizers accept any Atoms-like object with get_positions/get_forces;
    # NEB implements that duck-typed contract (as ase.filters wrappers do), but
    # the stub only spells out Atoms — same cast as api.relax's FrechetCellFilter.
    opt = opt_cls(cast("Atoms", band), logfile=None)
    t0 = time.perf_counter()
    step_state: dict[str, Any] = {"n": 0}

    def _log() -> None:
        step_state["n"] = opt.nsteps
        if not verbose:
            return
        energies = [e_ini]
        energies += [float(images[i].get_potential_energy()) for i in range(1, n - 1)]
        energies += [e_fin]
        fmax = float(np.linalg.norm(band.get_forces(), axis=1).max())
        emax = max(energies)
        print(f"  neb step {opt.nsteps:>3d} · E_barrier = {emax - e_ini:+.4f} eV"
              f" · fmax = {fmax:.4f} eV/Å · {time.perf_counter() - t0:.1f}s",
              flush=True)

    opt.attach(_log)
    converged = bool(opt.run(fmax=p.fmax, steps=p.max_steps))

    energies = [e_ini]
    energies += [float(images[i].get_potential_energy()) for i in range(1, n - 1)]
    energies += [e_fin]
    fmax = float(np.linalg.norm(band.get_forces(), axis=1).max())
    block = _mep_block(images, energies, e_ini, e_fin, fmax, converged,
                       opt.nsteps, inp)
    if verbose:
        print(f"neb: barrier E_a = {block['barrier_eV']:+.4f} eV "
              f"(TS = image {block['ts_image']}/{n - 1}), "
              f"reaction ΔE = {block['reaction_energy_eV']:+.4f} eV — "
              f"{'converged' if converged else 'NOT converged'}", flush=True)
        if block["ts_is_endpoint"]:
            print("neb: WARNING — the highest-energy image is an endpoint; the "
                  "band has no interior barrier (endpoints may not be minima, or "
                  "n_images is too small to resolve the saddle)", flush=True)
    return block, _frames(images, energies)
