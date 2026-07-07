"""Heisenberg exchange constants from collinear ordering energetics (M-agnetics).

Classical Heisenberg mapping per magnetic atom:

    E_c/atom = E0 − ½ Σ_shells J_s · S_cs,
    S_cs = (1/N_at) Σ_i Σ_{j ∈ shell s} σ_i σ_j          (σ = ±1)

Given ≥ n_shells+1 collinear configurations (their converged energies and
spin patterns), the signed neighbor sums S_cs are computed geometrically
and [E0, J_s] solved by least squares. Positive J = ferromagnetic coupling.

This is the ENERGY-MAPPING route (works with today's collinear code). The
transverse-response route (J from second derivatives of E w.r.t. moment
rotations through the implicit-diff machinery) needs the non-collinear
phase and will slot in beside this as a cross-check.

Mean-field Curie/Néel estimate: k_B T_c^MF = (1/3) Σ_j J_0j  (overestimates
by ~30% typically — reported for orientation only).
"""

from __future__ import annotations

import numpy as np

from gradwave.constants import KB_EV


def shell_sums(cell, positions, spins, shell_cutoffs) -> np.ndarray:
    """S_s = (1/N) Σ_i Σ_{j∈shell s} σ_i σ_j for one configuration.

    shell_cutoffs: increasing distances [d1, d2, ...]; shell s collects
    neighbors with d_{s-1} < r ≤ d_s (d_0 = 0.1 Å).
    """
    cell = np.asarray(cell, float)
    pos = np.asarray(positions, float)
    spins = np.asarray(spins, float)
    na = len(pos)
    reach = int(np.ceil(max(shell_cutoffs) / min(np.linalg.norm(cell, axis=1)))) + 1
    images = np.array([
        [i, j, k] for i in range(-reach, reach + 1)
        for j in range(-reach, reach + 1) for k in range(-reach, reach + 1)
    ]) @ cell

    edges = [0.1] + list(shell_cutoffs)
    sums = np.zeros(len(shell_cutoffs))
    for i in range(na):
        d = np.linalg.norm(pos[None, :, :] + images[:, None, :] - pos[i], axis=-1)
        sgn = spins[i] * spins[None, :]
        for s in range(len(shell_cutoffs)):
            mask = (d > edges[s]) & (d <= edges[s + 1] + 1e-6)
            sums[s] += (sgn * mask.astype(float)).sum()
    return sums / na


def heisenberg_fit(configs, shell_cutoffs):
    """configs: [(E_per_atom_eV, cell, positions, spins)]. Returns (E0, J[eV], resid)."""
    a_rows, b = [], []
    for e_at, cell, pos, spins in configs:
        s = shell_sums(cell, pos, spins, shell_cutoffs)
        a_rows.append(np.concatenate([[1.0], -0.5 * s]))
        b.append(e_at)
    a = np.array(a_rows)
    sol, res, *_ = np.linalg.lstsq(a, np.array(b), rcond=None)
    resid = float(np.sqrt(res[0])) if len(res) else 0.0
    return float(sol[0]), sol[1:], resid


def mean_field_tc(j_shell_ev, coordination) -> float:
    """k_B T^MF = (1/3)Σ z_s J_s → T in K."""
    return float(sum(z * j for z, j in zip(coordination, j_shell_ev, strict=True))
                 / (3.0 * KB_EV))
