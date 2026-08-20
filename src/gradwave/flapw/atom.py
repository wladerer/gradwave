"""Self-consistent all-electron atom (spherical KS), the reference for the FLAPW muffin-tin.

Solves the radial Kohn-Sham equations for a closed-shell atom with the tridiagonal eigensolver
(``flapw.radial.radial_eigs_tridiag``), radial-Poisson Hartree, LDA XC, and Anderson mixing. The
converged spherical potential feeds the crystal FLAPW muffin tins; the eigenvalues are the atomic
limit the crystal SCF must reduce to.

Verified against the NIST LDA atomic reference eigenvalues (physics.nist.gov/PhysRefData/DFTdata):
Be 1s to 0.002 eV, He/Ne valence to ~0.1-0.4 eV (deep cores are O(dx²) mesh-limited).
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import Tensor

from gradwave.constants import E2, HARTREE_EV
from gradwave.flapw.coulomb import hartree
from gradwave.flapw.functionals import vxc_lda
from gradwave.flapw.mixing import anderson_next
from gradwave.flapw.radial import radial_eigs_tridiag

# closed-shell configurations: symbol -> (Z, [(n, l, occupation), ...])
CONFIG: dict[str, tuple[float, list[tuple[int, int, int]]]] = {
    "He": (2.0, [(1, 0, 2)]),
    "Be": (4.0, [(1, 0, 2), (2, 0, 2)]),
    "O": (8.0, [(1, 0, 2), (2, 0, 2), (2, 1, 4)]),      # open-shell 2p⁴ (spherically averaged)
    "Ne": (10.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6)]),
    "Ti": (22.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6), (3, 2, 2), (4, 0, 2)]),
}
# NIST LDA atomic reference KS eigenvalues (eV), converted from Hartree.
NIST_LDA_EV: dict[str, dict[str, float]] = {
    "He": {"1s": -0.5704 * HARTREE_EV},
    "Be": {"1s": -3.8551 * HARTREE_EV, "2s": -0.20565 * HARTREE_EV},
    "Ne": {"1s": -30.305 * HARTREE_EV, "2s": -1.3230 * HARTREE_EV, "2p": -0.49928 * HARTREE_EV},
    "Ti": {"1s": -177.276643 * HARTREE_EV, "2s": -19.457901 * HARTREE_EV,
           "2p": -16.285339 * HARTREE_EV, "3s": -2.258007 * HARTREE_EV,
           "3p": -1.422947 * HARTREE_EV, "3d": -0.170010 * HARTREE_EV,
           "4s": -0.167106 * HARTREE_EV},
}


def atomic_scf(symbol: str, r: Tensor, dx: float, iters: int = 60, tol: float = 1e-4,
               m: int = 5) -> tuple[dict[str, float], Tensor]:
    """Self-consistent atom. Returns the converged KS eigenvalues (eV) and the potential v (eV)."""
    Z, occ = CONFIG[symbol]
    by_l: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for n, l, f in occ:
        by_l[l].append((n, f))
    v = -Z * E2 / r
    hist_v: list[Tensor] = []
    hist_r: list[Tensor] = []
    eigs: dict[str, float] = {}
    for _ in range(iters):
        rho = torch.zeros_like(r)
        new_eigs: dict[str, float] = {}
        for l, states in by_l.items():
            kmax = max(n - l for n, _ in states)
            energies, u = radial_eigs_tridiag(l, r, dx, v, kmax)
            for n, f in states:
                uj = torch.tensor(u[:, n - l - 1], dtype=torch.float64)
                rho = rho + f * (uj * uj) / (4 * math.pi * r * r)
                new_eigs[f"{n}{'spdf'[l]}"] = float(energies[n - l - 1])
        vnew = -Z * E2 / r + hartree(rho, r, dx) + vxc_lda(rho)
        hist_v.append(v)
        hist_r.append(vnew - v)
        if len(hist_r) > m + 1:
            hist_v, hist_r = hist_v[-(m + 1):], hist_r[-(m + 1):]
        conv = bool(eigs) and max(abs(new_eigs[k] - eigs.get(k, 0.0)) for k in new_eigs) < tol
        eigs = new_eigs
        if conv:
            break
        v = anderson_next(hist_v, hist_r, beta=0.5, m=m)
    return eigs, v
