"""Smeared fixed-spin-moment occupations (scf.common.fsm_smeared_occupations).

The smeared FSM mode pins M = N↑ − N↓ with TWO Fermi levels — μ↑ solves the
up-channel count N↑ = (N_e+M)/2 and μ↓ the down-channel count — instead of the
integer per-channel fill the no-smearing FSM uses. The key physics: along the
fixed-moment curve ∂F/∂M = (μ↑ − μ↓)/2, so at M equal to the UNCONSTRAINED
moment the two Fermi levels coincide and the FSM solve reproduces the shared-μ
occupations, energy, and entropy exactly. These tests pin that algebra on
synthetic eigenvalues (no SCF); the SCF-level gate lives in
tests/integration/test_fsm_smeared.py.
"""

import pytest
import torch

from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.dtypes import RDTYPE
from gradwave.scf.common import fsm_smeared_occupations, shared_fermi_occupations

NK, NB = 4, 40
N_E = 20.0
WIDTH = 0.25


def _eigs(seed, spread=8.0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    e = torch.rand(NK, NB, generator=g, dtype=torch.float64) * spread + shift
    return torch.sort(e.to(RDTYPE), dim=1).values


def _setup():
    # dense metallic spectra (level spacing << width, so N(μ) is strictly
    # monotone near the solutions and μ is unique), slightly split channels
    # like an exchange-split metal
    eigs_s = [_eigs(7, shift=-0.4), _eigs(13, shift=+0.4)]
    kw = torch.full((NK,), 1.0 / NK, dtype=RDTYPE)
    return eigs_s, kw


@pytest.mark.parametrize("smearing", ["gaussian", "mp1", "fermi-dirac", "cold"])
@pytest.mark.parametrize("m", [0.0, 0.7, 1.5, -0.9])
def test_fsm_pins_the_moment(smearing, m):
    """Each channel's smeared count hits its target, so N and M are exact."""
    eigs_s, kw = _setup()
    occ_s, (mu_up, mu_dn), _ = fsm_smeared_occupations(
        eigs_s, kw, smearing, WIDTH, N_E, m, torch.device("cpu"))
    n_up = float((kw[:, None] * occ_s[0]).sum())
    n_dn = float((kw[:, None] * occ_s[1]).sum())
    assert n_up + n_dn == pytest.approx(N_E, abs=1e-9)
    assert n_up - n_dn == pytest.approx(m, abs=1e-9)


@pytest.mark.parametrize("smearing", ["gaussian", "fermi-dirac"])
def test_fermi_split_monotone_in_moment(smearing):
    """Raising M raises μ↑ and lowers μ↓ (monotone occupations ⇒ each channel's
    Fermi level moves with its electron count) — the FSM field h = (μ↑−μ↓)/2
    grows along the E(M) curve."""
    eigs_s, kw = _setup()
    dev = torch.device("cpu")
    _, (up_lo, dn_lo), _ = fsm_smeared_occupations(
        eigs_s, kw, smearing, WIDTH, N_E, 0.4, dev)
    _, (up_hi, dn_hi), _ = fsm_smeared_occupations(
        eigs_s, kw, smearing, WIDTH, N_E, 1.6, dev)
    assert up_hi > up_lo
    assert dn_hi < dn_lo


@pytest.mark.parametrize("smearing", ["gaussian", "mp1", "fermi-dirac"])
def test_fsm_at_free_moment_reproduces_shared_mu(smearing):
    """At M = the unconstrained (shared-μ) moment, μ↑ = μ↓ = μ_shared and the
    occupations and entropy match the shared-Fermi solve — ∂F/∂M = (μ↑−μ↓)/2
    vanishes exactly at the free minimum."""
    eigs_s, kw = _setup()
    dev = torch.device("cpu")
    occ_ref, mu_ref, ent_ref = shared_fermi_occupations(
        eigs_s, kw, smearing, WIDTH, N_E, 2, dev)
    m0 = float((kw[:, None] * (occ_ref[0] - occ_ref[1])).sum())
    occ_s, (mu_up, mu_dn), ent = fsm_smeared_occupations(
        eigs_s, kw, smearing, WIDTH, N_E, m0, dev)
    assert mu_up == pytest.approx(mu_ref, abs=1e-8)
    assert mu_dn == pytest.approx(mu_ref, abs=1e-8)
    for sp in range(2):
        assert torch.allclose(occ_s[sp], occ_ref[sp], atol=1e-8)
    assert float(ent) == pytest.approx(float(ent_ref), abs=1e-9)


def test_shared_fermi_delegates_with_tot_magnetization():
    """shared_fermi_occupations(nspin=2, smearing, tot_magnetization) runs the
    FSM solve and reports μ as the mean of the pair (a float, so existing
    consumers are untouched)."""
    eigs_s, kw = _setup()
    dev = torch.device("cpu")
    m = 1.2
    occ_a, mu, ent_a = shared_fermi_occupations(
        eigs_s, kw, "mp1", WIDTH, N_E, 2, dev, tot_magnetization=m)
    occ_b, (mu_up, mu_dn), ent_b = fsm_smeared_occupations(
        eigs_s, kw, "mp1", WIDTH, N_E, m, dev)
    assert isinstance(mu, float)
    assert mu == pytest.approx(0.5 * (mu_up + mu_dn), abs=1e-12)
    for sp in range(2):
        assert torch.equal(occ_a[sp], occ_b[sp])
    assert float(ent_a) == pytest.approx(float(ent_b), abs=1e-14)


def test_none_path_unchanged():
    """tot_magnetization=None keeps the original shared-μ algorithm: one Fermi
    level from the concatenated channels, per-channel occupations/entropy at
    degeneracy 1 (guards the new branch from touching the default path)."""
    eigs_s, kw = _setup()
    dev = torch.device("cpu")
    occ_s, mu, ent = shared_fermi_occupations(
        eigs_s, kw, "gaussian", WIDTH, N_E, 2, dev)
    scheme = SCHEMES["gaussian"]
    mu_ref = float(find_fermi(torch.cat(eigs_s, dim=0), torch.cat([kw, kw]),
                              scheme, WIDTH, N_E, degeneracy=1.0))
    assert mu == pytest.approx(mu_ref, abs=1e-12)
    mu_t = torch.tensor(mu_ref, dtype=RDTYPE)
    ent_ref = torch.zeros((), dtype=RDTYPE)
    for sp in range(2):
        o, s = occupations_and_entropy(eigs_s[sp], mu_t, scheme, WIDTH,
                                       degeneracy=1.0)
        assert torch.equal(occ_s[sp], o)
        ent_ref = ent_ref - WIDTH * (kw[:, None] * s).sum()
    assert float(ent) == pytest.approx(float(ent_ref), abs=1e-14)


def test_fsm_rejects_overlarge_moment():
    eigs_s, kw = _setup()
    with pytest.raises(ValueError, match="exceeds n_electrons"):
        fsm_smeared_occupations(eigs_s, kw, "gaussian", WIDTH, N_E, N_E + 2.0,
                                torch.device("cpu"))
