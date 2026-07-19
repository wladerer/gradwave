"""Smearing entropy is the Legendre partner of the occupation function.

For every smearing scheme, the band free energy at fixed electron count,

    F(ε) = Σ_k w_k Σ_n f_nk ε_nk  −  σ Σ_k w_k Σ_n g·s_nk,    Σ w f = N,

must satisfy dF/dε_nk = w_k f_nk exactly: the per-state entropy s(x) has to
obey s'(x) = x·f'(x) so the explicit ε-dependence beyond the leading w·f
collapses to μ·(dN/dε) = 0 through the implicit Fermi level. This is the
identity that makes smeared forces Hellmann-Feynman forces of F (Mermin);
a mis-derived entropy expression (the classic trap for Methfessel-Paxton
and Marzari-Vanderbilt) breaks it at first order while leaving occupations
— and therefore every converged-density check — untouched.

Gated on the SCF's own assembly (`shared_fermi_occupations`), including the
nspin=2 shared-μ bookkeeping where perturbing one channel re-partitions
charge between both.
"""

import pytest
import torch

from gradwave.core.occupations import SCHEMES
from gradwave.dtypes import RDTYPE
from gradwave.scf.common import shared_fermi_occupations


def _eigs(nk, nb, seed, scale=4.0):
    gen = torch.Generator().manual_seed(seed)
    return scale * torch.randn(nk, nb, generator=gen, dtype=RDTYPE)


def _free_energy(eigs_s, kw, smearing, width, n_el, nspin):
    occ_s, _, ent = shared_fermi_occupations(
        eigs_s, kw, smearing, width, n_el, nspin, eigs_s[0].device)
    e_band = sum((kw[:, None] * occ_s[isp] * eigs_s[isp]).sum()
                 for isp in range(nspin))
    return float(e_band + ent), occ_s


@pytest.mark.parametrize("smearing", list(SCHEMES))
@pytest.mark.parametrize("nspin", [1, 2])
def test_dF_deps_equals_weighted_occupation(smearing, nspin):
    nk, nb, width, h = 3, 6, 0.15, 1e-5
    n_el = 5.0 if nspin == 1 else 4.0  # odd/frac fillings, μ inside the band
    kw = torch.tensor([0.5, 0.3, 0.2], dtype=RDTYPE)
    eigs_s = [_eigs(nk, nb, seed=17 + isp) for isp in range(nspin)]

    _, occ_s = _free_energy(eigs_s, kw, smearing, width, n_el, nspin)
    g = 2.0 if nspin == 1 else 1.0
    # always include the most fractional state — that is where the entropy
    # chain is maximally active (far from μ the identity is trivially w·f)
    ik_f, ib_f = divmod(int((occ_s[0] - 0.5 * g).abs().argmin()),
                        occ_s[0].shape[1])
    for isp, ik, ib in [(0, ik_f, ib_f), (0, 1, 4), (0, 2, 0),
                        (nspin - 1, 1, 1)]:
        pert = []
        for sgn in (+h, -h):
            es = [e.clone() for e in eigs_s]
            es[isp][ik, ib] += sgn
            pert.append(_free_energy(es, kw, smearing, width, n_el, nspin)[0])
        fd = (pert[0] - pert[1]) / (2 * h)
        exact = float(kw[ik] * occ_s[isp][ik, ib])
        assert abs(fd - exact) < 1e-7 * max(1.0, abs(exact)), (
            f"{smearing}, nspin={nspin}, state ({isp},{ik},{ib}): "
            f"dF/de = {fd:.10f} vs w·f = {exact:.10f}")


def test_identity_has_teeth():
    """A deliberately wrong entropy (gaussian occupations with fermi-dirac
    entropy) must break the identity at first order."""
    from gradwave.core.occupations import find_fermi

    nk, nb, width, h = 3, 6, 0.15, 1e-5
    kw = torch.tensor([0.5, 0.3, 0.2], dtype=RDTYPE)
    eigs = _eigs(nk, nb, seed=17)
    gauss, fd_scheme = SCHEMES["gaussian"], SCHEMES["fermi-dirac"]

    def f_wrong(e):
        mu = torch.as_tensor(find_fermi(e, kw, gauss, width, 5.0), dtype=RDTYPE)
        x = (e - mu) / width
        occ = 2.0 * gauss.occupation(x)
        ent = -width * (2.0 * kw[:, None] * fd_scheme.entropy(x)).sum()
        return float((kw[:, None] * occ * e).sum() + ent), occ

    _, occ = f_wrong(eigs)
    # perturb the most fractional state — far from μ both entropies vanish
    # and even a wrong pair looks consistent
    ik, ib = divmod(int((occ - 1.0).abs().argmin()), occ.shape[1])
    ep, em = eigs.clone(), eigs.clone()
    ep[ik, ib] += h
    em[ik, ib] -= h
    fd = (f_wrong(ep)[0] - f_wrong(em)[0]) / (2 * h)
    assert abs(fd - float(kw[ik] * occ[ik, ib])) > 1e-3
