"""run_surface_energy: γ from a slab-thickness sweep (Fiorentini-Methfessel)."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from gradwave.api.scf import run_scf
from gradwave.inputs import Input

logger = logging.getLogger(__name__)


def run_surface_energy(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Surface energy γ from a slab-thickness sweep (task: surface_energy).

    For each slab in ``inp.surface_energy.slabs`` (a geometry file + its layer
    count) the SCF is re-converged, then ``E_slab(N) = 2·γ·A + N·E_bulk`` is fit
    by least squares (``postscf.surface_energy.surface_energy_fm``) and γ is read
    off the intercept. All slabs must share the same in-plane surface area A (only
    the thickness / c-axis grows); a mismatch raises. Every slab is built from the
    same pseudopotentials / ecut / k-mesh as the top-level input (swap only the
    geometry), so set a slab-appropriate k-mesh (n_z = 1). Returns the
    ``surface_energy`` summary block."""
    import numpy as np
    from ase.io import read as ase_read

    from gradwave.postscf.surface_energy import EV_A2_TO_JM2, surface_energy_fm

    se = inp.surface_energy
    ekind = se.energy
    n_layers: list[float] = []
    energies: list[float] = []
    areas: list[float] = []
    converged: list[bool] = []
    files: list[str] = []

    for point in se.slabs:
        atoms = ase_read(str(point.structure))
        if isinstance(atoms, list):  # ase_read may return a trajectory; take the last frame
            atoms = atoms[-1]
        # build this slab's system from the shared pseudos/ecut/kmesh — only the
        # geometry changes (task forced to scf so run_scf takes the plain path)
        sub = dataclasses.replace(inp, atoms=atoms, task="scf")
        res = run_scf(sub, verbose=False)
        e = float(getattr(res.energies, ekind))
        cell = np.asarray(atoms.cell.array, dtype=float)
        area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        conv = bool(getattr(res, "converged", True))
        n_layers.append(float(point.n_layers))
        energies.append(e)
        areas.append(area)
        converged.append(conv)
        files.append(point.structure.name)
        if verbose:
            tag = "" if conv else "  (NOT converged)"
            print(f"  {point.structure.name}: N={point.n_layers:g}  "
                  f"A={area:8.4f} Å²  E={e:+.6f} eV{tag}", flush=True)

    # all slabs must expose the same face: one area feeds the fit
    a0 = areas[0]
    if any(abs(a - a0) > 1e-4 * a0 for a in areas):
        raise ValueError(
            f"surface_energy: the slabs do not share an in-plane area "
            f"(got {[round(a, 4) for a in areas]} Å²) — every thickness must "
            f"keep the same surface cell; only the c-axis / layer count grows")

    fit = surface_energy_fm(n_layers, energies, a0, n_surfaces=se.n_surfaces)
    block: dict[str, Any] = {
        "method": "Fiorentini-Methfessel slab-thickness fit",
        "energy_kind": ekind,
        "n_surfaces": se.n_surfaces,
        "slabs": files,
        "n_layers": n_layers,
        "energies_eV": energies,
        "area_ang2": a0,
        "gamma_eV_ang2": fit.gamma,
        "gamma_J_m2": fit.gamma * EV_A2_TO_JM2,
        "gamma_mJ_m2": fit.gamma * EV_A2_TO_JM2 * 1e3,
        "e_bulk_eV_per_layer": fit.e_bulk,
        "intercept_eV": fit.intercept,
        "rms_residual_eV": fit.rms_residual_eV,
        "all_converged": all(converged),
    }
    if verbose:
        print(f"surface_energy: γ = {fit.gamma:.6f} eV/Å² "
              f"({fit.gamma * EV_A2_TO_JM2:.4f} J/m²), "
              f"E_bulk = {fit.e_bulk:+.6f} eV/layer", flush=True)
    return block
