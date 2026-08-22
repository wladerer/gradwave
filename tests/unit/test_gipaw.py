"""GIPAW Phase 1 (ground-state) shielding terms: frozen-core Lamb + diamagnetic
augmentation, on the PAW partial-wave path. Pure post-processing over the PAW
fixtures (no SCF) — fast tier."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradwave.postscf.gipaw import (
    PAWOnSite,
    _ang_L_matrices,
    augmentation_charge_residual,
    core_lamb_shielding,
    ground_state_shielding,
)
from gradwave.postscf.kgeometry_nmr import sigma_shielding
from gradwave.pseudo.upf_paw import parse_upf_paw
from tests.helpers import pseudo

SI_PAW = "Si.pbe-n-kjpaw_psl.1.0.0.UPF"
O_PAW = "O.pbe-n-kjpaw_psl.1.0.0.UPF"


def _ref_becsum(paw) -> torch.Tensor:
    """Free-atom reference on-site density matrix: diag(paw_occ) spread over the
    (2l+1) m-channels — the m-expanded becsum layout."""
    onsite = PAWOnSite.from_paw(paw)
    n = onsite.n_mexp
    d = np.zeros((n, n))
    for a, (i, l, _m) in enumerate(onsite.idx):
        d[a, a] = paw.paw_occ[i] / (2 * l + 1)
    return torch.as_tensor(d)


@pytest.mark.parametrize(
    ("fname", "n_core", "flapw_core_ppm"),
    [(SI_PAW, 10.0, 811.2), (O_PAW, 2.0, 269.5)],
)
def test_core_lamb_vs_flapw(fname, n_core, flapw_core_ppm):
    """σ_core from the PAW AE core density cross-validates the FLAPW all-electron
    -core Lamb term (the "two halves meet at the core" milestone) to a few %,
    and the AE core integrates to the exact electron count."""
    paw = parse_upf_paw(pseudo(fname))
    ne = 4.0 * np.pi * float(np.sum(paw.ae_core_rho * paw.r**2 * paw.rab))
    assert ne == pytest.approx(n_core, abs=1e-3)
    sig = core_lamb_shielding(paw)
    assert sig == pytest.approx(flapw_core_ppm, rel=0.05)  # few-% Tier-A agreement


@pytest.mark.parametrize("fname", [SI_PAW, O_PAW])
def test_augmentation_charge_selfconsistency(fname):
    """PAW-path bridge: the partial-wave reconstruction of the parsed augmentation
    charges ∫Q_ij d³r (a completeness/norm relation) agrees to ~1e-9."""
    paw = parse_upf_paw(pseudo(fname))
    assert augmentation_charge_residual(paw) < 1e-7


@pytest.mark.parametrize("fname", [SI_PAW, O_PAW])
def test_dia_angular_tensor_isotropic_trace(fname):
    """The diamagnetic angular tensor's isotropic trace is exact: (1/3)Σ_α M_αα
    = (2/3)δ_IJ (the Lamb reduction of δ_αβ − r̂_α r̂_β)."""
    onsite = PAWOnSite.from_paw(parse_upf_paw(pseudo(fname)))
    tr = onsite.dia_ang[0, 0] + onsite.dia_ang[1, 1] + onsite.dia_ang[2, 2]
    nlm = tr.shape[0]
    assert np.abs(tr / 3.0 - (2.0 / 3.0) * np.eye(nlm)).max() < 1e-12


@pytest.mark.parametrize(
    ("fname", "iso_ppm"), [(SI_PAW, 0.9), (O_PAW, 14.8)]
)
def test_dia_aug_free_atom(fname, iso_ppm):
    """σ_dia_aug for the free-atom reference density matrix: the full 3×3 tensor
    is isotropic (spherical-atom null-test, analogous to the cubic-EFG null),
    and its trace matches the direct AE−PS valence Lamb difference. Literature
    sanity: ¹⁷O's near-nucleus AE−PS valence correction (~15 ppm) is an order of
    magnitude above ²⁹Si's (~1 ppm) [Yates-Pickard-Mauri, PRB 76, 024401]."""
    paw = parse_upf_paw(pseudo(fname))
    onsite = PAWOnSite.from_paw(paw)
    d = _ref_becsum(paw)
    tensor = onsite.dia_aug_tensor(d).numpy()
    diag = np.diag(tensor)
    off = np.abs(tensor - np.diag(diag)).max()
    assert off < 1e-10  # spherical free atom → isotropic tensor
    assert np.ptp(diag) < 1e-10
    assert onsite.dia_aug_iso(d) == pytest.approx(iso_ppm, abs=0.2)
    assert onsite.dia_aug_iso(d) == pytest.approx(float(np.mean(diag)))


def test_ground_state_shielding_multisite():
    """The per-site assembler broadcasts species terms to atoms and returns core,
    the dia_aug tensor, and its isotropic trace."""
    paw = parse_upf_paw(pseudo(SI_PAW))
    d = _ref_becsum(paw)
    out = ground_state_shielding([paw], [0, 0], [d, d])
    assert out["core"].shape == (2,)
    assert out["dia_aug"].shape == (2, 3, 3)
    assert out["dia_aug_iso"].shape == (2,)
    assert float(out["core"][0]) == pytest.approx(core_lamb_shielding(paw))
    assert float(out["dia_aug_iso"][1]) == pytest.approx(0.913, abs=0.1)


def test_ang_L_algebra():
    """The real-harmonic angular-momentum matrices satisfy the su(2) algebra,
    are Hermitian, block-diagonal with L² = l(l+1), and reproduce the analytic
    L_z action — the operator-correctness gate for σ_para_aug."""
    lx, ly, lz = _ang_L_matrices(3)
    for la in (lx, ly, lz):
        assert np.abs(la - la.conj().T).max() < 1e-9  # Hermitian
        assert np.abs(la.real).max() < 1e-9  # purely imaginary
    assert np.abs((lx @ ly - ly @ lx) - 1j * lz).max() < 1e-6
    assert np.abs((ly @ lz - lz @ ly) - 1j * lx).max() < 1e-6
    assert np.abs((lz @ lx - lx @ lz) - 1j * ly).max() < 1e-6
    l2 = lx @ lx + ly @ ly + lz @ lz
    ll1 = np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(4)])
    assert np.abs(l2 - np.diag(ll1.astype(float)).astype(complex)).max() < 1e-6
    # l=1 real slots: 1=(1,0)~z, 2=(1,1c)~x, 3=(1,1s)~y.  L_z: x → +i y
    assert lz[3, 2] == pytest.approx(1j, abs=1e-6)
    assert lz[2, 3] == pytest.approx(-1j, abs=1e-6)


@pytest.mark.parametrize("fname", [SI_PAW, O_PAW])
def test_para_aug_operator_antihermitian(fname):
    """The on-site paramagnetic operator F^α = ⟨φ|L_α/r³|φ⟩−PS is purely imaginary
    and antisymmetric (L_α is imaginary-Hermitian, the radial factor real
    symmetric), so its diagonal vanishes — the algebraic root of the closed-shell
    null. It is also non-trivial (the p-channel AE−PS ⟨1/r³⟩ is nonzero)."""
    onsite = PAWOnSite.from_paw(parse_upf_paw(pseudo(fname)))
    f = onsite.para_aug_operator()
    n = onsite.n_mexp
    assert f.shape == (3, n, n)
    scale = max(np.abs(f).max(), 1.0)
    for a in range(3):
        assert np.abs(f[a].real).max() < 1e-9 * scale  # purely imaginary
        assert np.abs(f[a] + f[a].T).max() < 1e-9 * scale  # antisymmetric
        assert np.abs(np.diag(f[a])).max() < 1e-9 * scale  # zero diagonal
    assert np.abs(f).max() > 1e-6  # p-channel operator is live


@pytest.mark.parametrize("fname", [SI_PAW, O_PAW])
def test_para_aug_nulls_and_reality(fname):
    """σ_para_aug is real for any cross density, and vanishes for the two
    paramagnetic-null inputs: an m-diagonal (spherical, closed-shell) cross and a
    real-symmetric cross (no orbital current). A generic complex-antisymmetric
    cross gives a nonzero tensor (the operator is not identically zero)."""
    onsite = PAWOnSite.from_paw(parse_upf_paw(pseudo(fname)))
    n = onsite.n_mexp
    rng = np.random.default_rng(0)
    # spherical: diagonal per field → zero (F has zero diagonal)
    diag = np.stack([np.diag(rng.standard_normal(n)) for _ in range(3)])
    s_diag = onsite.para_aug_tensor(torch.as_tensor(diag.astype(complex)))
    assert np.abs(s_diag.numpy()).max() < 1e-9
    # real symmetric cross → F(imag antisym)·X(real sym) is imaginary → Re = 0
    sym = rng.standard_normal((3, n, n))
    sym = sym + sym.transpose(0, 2, 1)
    s_sym = onsite.para_aug_tensor(torch.as_tensor(sym.astype(complex)))
    assert np.abs(s_sym.numpy()).max() < 1e-9
    # live: contracting the operator with itself is a genuine nonzero response
    f = onsite.para_aug_operator()
    s_live = onsite.para_aug_tensor(torch.as_tensor(f)).numpy()
    assert np.isrealobj(s_live) and s_live.shape == (3, 3)
    assert np.abs(s_live).max() > 1e-3


def test_sigma_shielding_core_ppm_wired():
    """``sigma_shielding`` exposes the opt-in per-site ``core_ppm`` offset (the
    σ_core wiring). Signature/contract check — the full tensor path needs an SCF
    (integration tier)."""
    import inspect

    sig = inspect.signature(sigma_shielding)
    assert "core_ppm" in sig.parameters
    assert sig.parameters["core_ppm"].default is None
