"""Blocks genuinely shared between the NC and USPP/PAW SCF loops
(refactor stage 4, deliberately minimal — full loop unification is
deferred until the S=1 overhead is measured AND NC maintenance hurts;
see docs/refactor_plan.md)."""

from __future__ import annotations

import torch

from gradwave.core.occupations import (
    SCHEMES,
    find_fermi,
    fixed_occupations,
    occupations_and_entropy,
)
from gradwave.dtypes import RDTYPE


def shared_fermi_occupations(eigs_s, kweights, smearing, width, n_electrons,
                             nspin, device):
    """Occupations, Fermi level, and entropy term for per-spin eigenvalue
    stacks with a SHARED Fermi level (both spin channels fill from one μ;
    the spin degeneracy g = 2 for nspin=1, 1 per channel otherwise).

    Returns (occ_s per spin, mu float, entropy_term tensor). smearing
    "none" gives fixed occupations (nspin=1 only — a spin system needs a
    shared Fermi level to exchange charge between channels)."""
    g_spin = 2 if nspin == 1 else 1
    if smearing == "none":
        if nspin != 1:
            raise ValueError("nspin=2 requires smearing (shared Fermi level)")
        occ_s = [fixed_occupations(eigs_s[0], n_electrons)]
        mu = float(eigs_s[0][:, int(n_electrons // 2) - 1].max())
        entropy_term = torch.zeros((), dtype=RDTYPE, device=device)
        return occ_s, mu, entropy_term
    scheme = SCHEMES[smearing]
    eigs_cat = torch.cat(eigs_s, dim=0)  # (nspin·nk, nb)
    kw_cat = torch.cat([kweights] * nspin)
    mu = float(find_fermi(eigs_cat, kw_cat, scheme, width, n_electrons,
                          degeneracy=g_spin))
    # NB: bare torch.tensor(mu) would be float32 and shift N_e by ~1e-7
    mu_t = torch.tensor(mu, dtype=RDTYPE, device=device)
    occ_s, ent = [], torch.zeros((), dtype=RDTYPE, device=device)
    for isp in range(nspin):
        o, s_ent = occupations_and_entropy(eigs_s[isp], mu_t, scheme, width,
                                           degeneracy=g_spin)
        occ_s.append(o)
        ent = ent - width * (g_spin * kweights[:, None] * s_ent).sum()
    return occ_s, mu, ent
