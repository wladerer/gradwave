"""FLAPW LAPW assembly: the empty-lattice correctness gate (V=0 -> free-electron bands)."""

from __future__ import annotations

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.flapw import build_matrices, build_matrices_multi, log_mesh, solve_geneig


def _free_electron(kfrac, L, ecut, nbands):
    b = 2 * np.pi / L
    nmax = int(np.ceil(np.sqrt(ecut / HBAR2_2M) / b)) + 1
    e = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = b * (np.array([i, j, m]) + np.asarray(kfrac))
                if HBAR2_2M * (kg @ kg) <= ecut:
                    e.append(HBAR2_2M * (kg @ kg))
    return np.sort(e)[:nbands]


def test_empty_lattice_free_electron():
    """With V=0 in the muffin tin, the LAPW bands must reproduce ½|k+G|² = HBAR2_2M|k+G|²."""
    L, R, lmax, ecut = 6.0, 1.0, 6, 60.0
    r, dx = log_mesh(1e-4, R + 1.0, 1200)
    v = torch.zeros_like(r)
    El = {lg: 2.5 for lg in range(lmax + 1)}       # linearization energy in the band range
    for kfrac in ([0.0, 0.0, 0.0], [0.1, 0.0, 0.0]):
        H, S, _ = build_matrices(kfrac, L, R, lmax, El, ecut, r, dx, v)
        ev = solve_geneig(0.5 * (H + H.T), S, 6)
        ref = _free_electron(kfrac, L, ecut, 6)
        assert float(np.abs(ev - ref).max()) < 1e-2       # meV-level (lmax + log-mesh limited)


def test_multi_atom_empty_lattice_free_electron():
    """Two empty muffin tins at CsCl positions still give free-electron bands — the multi-atom
    correctness gate (structure phases e^{iΔk·τ} + two augmentation blocks, complex Hermitian)."""
    L, R, lmax, ecut = 6.0, 1.0, 6, 55.0
    r, dx = log_mesh(1e-4, R + 1.0, 1200)
    v = torch.zeros_like(r)
    El = {lg: 2.5 for lg in range(lmax + 1)}
    species = {"A": {"R": R, "v": v, "El": El}}
    atoms = [([0.0, 0.0, 0.0], "A"), ([0.5 * L, 0.5 * L, 0.5 * L], "A")]
    for kfrac in ([0.0, 0.0, 0.0], [0.1, 0.2, 0.0]):
        H, S, _ = build_matrices_multi(kfrac, L, atoms, lmax, ecut, r, dx, species)
        ev = solve_geneig(H, S, 6)
        ref = _free_electron(kfrac, L, ecut, 6)
        assert float(np.abs(ev - ref).max()) < 1e-2
