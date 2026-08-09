"""USPP/PAW meta-GGA (τ): the PAW one-center τ augmentation and the bare-
ultrasoft hard gate.

The one-center kinetic-energy density τ¹ = ½ Σ_ij ρ_ij ∇φ_i·∇φ_j is a new
bilinear form in the becsum; these tests pin that it genuinely enters the
one-center energy and that its contribution to the on-site potential
ddd_ij = ∂E_1c/∂ρ_ij is exact (autograd == finite difference). The smooth-τ
Hamiltonian wiring is exercised end-to-end by tests/integration/
test_uspp_metagga_scf.py.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.paw_onsite import OneCenter

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


def _rho(paw, seed, scale=0.03):
    """Physically-shaped random becsum: atomic occupations + symmetric noise
    (the same generator the batched one-center test uses)."""
    nm = sum(2 * b.l + 1 for b in paw.betas)
    m0 = torch.zeros(nm, nm, dtype=torch.float64)
    col = 0
    for i, b in enumerate(paw.betas):
        for _m in range(2 * b.l + 1):
            m0[col, col] = paw.paw_occ[i] / (2 * b.l + 1)
            col += 1
    gen = torch.Generator().manual_seed(seed)
    p = scale * torch.randn(nm, nm, generator=gen, dtype=torch.float64)
    return m0 + (p + p.T) / 2


def test_onecenter_tau_contributes_and_ddd_matches_fd():
    """The meta-GGA one-center energy depends on τ (it differs from the GGA
    one-center at the same becsum), and its ddd carries the τ term exactly:
    the autograd ∂E_1c/∂ρ_ij matches a central finite difference."""
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    b = _rho(paw, 100)
    e_pbe, _ = OneCenter(paw, PBE()).energy_and_ddd(b)
    oc = OneCenter(paw, R2SCAN())
    e_r2, ddd = oc.energy_and_ddd(b)
    assert abs(e_r2 - e_pbe) > 1e-3  # τ genuinely enters the one-center energy

    eps = 1e-6
    for (i, j) in [(0, 0), (1, 1), (0, 2), (3, 4)]:
        d = torch.zeros_like(b)
        d[i, j] = 1.0
        if i != j:
            d[j, i] = 1.0  # becsum is symmetric — perturb the pair
        ep, _ = oc.energy_and_ddd(b + eps * d)
        em, _ = oc.energy_and_ddd(b - eps * d)
        fd = (ep - em) / (2 * eps)
        analytic = float(ddd[i, j]) + (float(ddd[j, i]) if i != j else 0.0)
        assert abs(fd - analytic) < 1e-4, f"ddd[{i},{j}]: {analytic} vs FD {fd}"


def test_onecenter_tau_spin_ddd_matches_fd():
    """Spin one-center τ: the per-channel ddd matches finite difference for
    SpinR2SCAN (τ_↑ / τ_↓ built from the per-spin becsum)."""
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    up, dn = 0.55 * _rho(paw, 200), 0.45 * _rho(paw, 300)
    oc = OneCenter(paw, SpinR2SCAN())
    _, (ddu, _ddd) = oc.energy_and_ddd([up, dn])
    eps = 1e-6
    d = torch.zeros_like(up)
    d[1, 1] = 1.0
    ep, _ = oc.energy_and_ddd([up + eps * d, dn])
    em, _ = oc.energy_and_ddd([up - eps * d, dn])
    fd = (ep - em) / (2 * eps)
    assert abs(fd - float(ddu[1, 1])) < 1e-4


def test_bare_ultrasoft_metagga_rejected():
    """meta-GGA needs the PAW one-center sphere for the AE/PS τ correction; a
    bare (non-PAW) ultrasoft pseudo must be rejected up front, not silently run
    as an uncontrolled smooth-only approximation."""
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    ry = 13.605693122994
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
    assert not paw.is_paw  # genuinely bare ultrasoft
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]) @ cell
    system = setup_uspp(cell, pos, [0, 0], [paw], ecut=12 * ry, kmesh=(1, 1, 1))
    with pytest.raises(NotImplementedError, match="meta-GGA requires PAW"):
        scf_uspp(system, R2SCAN(), verbose=False, max_iter=1)
