"""run_qha: quasi-harmonic thermodynamics from a phonon-per-volume sweep."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from gradwave.api.phonons import run_phonons
from gradwave.api.scf import run_scf
from gradwave.inputs import Input

logger = logging.getLogger(__name__)


def run_qha(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Quasi-harmonic thermodynamics (task: qha): G(T), V(T), thermal expansion.

    For each isotropically-scaled volume in ``inp.qha.scales`` the static SCF
    energy E(V) and a phonon DOS are computed (the DOS reuses the ``phonons``
    supercell / DOS-mesh settings), then the quasi-harmonic Gibbs free energy is
    fit over the temperature grid (``postscf.qha.qha``). Returns the ``qha``
    summary block: V(T), G(T), B(T), Cv(T), the thermal-expansion coefficient
    α(T), Cp(T) and the Grüneisen parameter γ(T)."""
    import numpy as np
    from ase import Atoms

    from gradwave.constants import EV_A3_TO_GPA
    from gradwave.postscf.qha import qha

    q = inp.qha
    if min(inp.phonons.dos_mesh) <= 0:
        raise ValueError(
            "task: qha needs a phonon DOS at each volume — set phonons.dos_mesh "
            "> 0 (it is (0,0,0), which skips the DOS)")

    cell0 = np.asarray(inp.atoms.cell.array, dtype=float)
    frac = inp.atoms.get_scaled_positions()
    symbols = inp.atoms.get_chemical_symbols()
    temps = np.asarray(q.temperatures, dtype=float)
    pressure_eV_A3 = q.pressure_GPa / EV_A3_TO_GPA

    volumes: list[float] = []
    energies: list[float] = []
    dos_per_volume: list[tuple[Any, Any]] = []
    all_conv = True

    for s in q.scales:
        cell = cell0 * s ** (1.0 / 3.0)
        atoms = Atoms(symbols, scaled_positions=frac, cell=cell, pbc=True)
        sub = dataclasses.replace(inp, atoms=atoms)
        # static E(V)
        res = run_scf(dataclasses.replace(sub, task="scf"), verbose=False)
        e = float(getattr(res.energies, q.energy))
        all_conv = all_conv and bool(res.converged)
        # phonon DOS at this volume (reuses the phonons block settings)
        ph = run_phonons(dataclasses.replace(sub, task="phonons"), verbose=False)
        dos = ph.get("dos")
        if dos is None:
            raise ValueError(
                "the phonon calculation produced no DOS (phonons.dos_mesh must "
                "be > 0 for a qha run)")
        grid = np.asarray(dos["frequency_cm1"], dtype=float)
        dvals = np.asarray(dos["dos"], dtype=float)
        vol = float(abs(np.linalg.det(cell)))
        volumes.append(vol)
        energies.append(e)
        dos_per_volume.append((grid, dvals))
        if verbose:
            print(f"  s={s:.3f}  V={vol:8.4f} Å³  E={e:+.6f} eV", flush=True)

    result = qha(np.asarray(volumes), np.asarray(energies), dos_per_volume,
                 temps, pressure=pressure_eV_A3)

    alpha = result.thermal_expansion()
    cp = result.heat_capacity_p()
    gamma = result.gruneisen()
    natoms = len(inp.atoms)
    block: dict[str, Any] = {
        "method": "quasi-harmonic approximation (phonon-DOS per volume)",
        "energy_kind": q.energy,
        "pressure_GPa": q.pressure_GPa,
        "n_atoms": natoms,
        "scales": list(q.scales),
        "volumes_ang3": volumes,
        "static_energies_eV": energies,
        "temperatures_K": temps.tolist(),
        "volume_T_ang3": result.volume.tolist(),
        "gibbs_T_eV": result.gibbs.tolist(),
        "bulk_modulus_T_GPa": (result.bulk_modulus * EV_A3_TO_GPA).tolist(),
        "cv_T_eV_per_K": result.cv.tolist(),
        "cp_T_eV_per_K": cp.tolist(),
        "thermal_expansion_T_per_K": alpha.tolist(),
        "gruneisen_T": gamma.tolist(),
        "all_converged": all_conv,
    }
    if verbose:
        print(f"qha: {len(q.scales)} volumes × {len(temps)} T; "
              f"V(300K)≈{float(np.interp(300.0, temps, result.volume)):.3f} Å³",
              flush=True)
    return block
