"""The non-collinear one-center (PAW) exchange-correlation (scf/paw_noncollinear.py).

Fast (no SCF — just the on-site radial×angular quadrature). Two exact handles plus
a derivative check: the non-collinear on-site XC must reduce to the collinear
OneCenter._exc_t when m⃗ ∥ ẑ, be independent of the m⃗ direction (rotation invariant
without spin-orbit), and its autograd B-field must match a finite difference of the
energy."""

import numpy as np
import torch

from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.paw_noncollinear import (
    onsite_nc_energy_and_ddd,
    onsite_nc_energy_and_field,
    onsite_nc_exc,
    spinor_onsite_becsum,
)
from gradwave.scf.paw_onsite import OneCenter
from tests.helpers import pseudo

PSEUDO = pseudo("Si.pbe-n-kjpaw_psl.1.0.0.UPF")


def _oc():
    paw = parse_upf_paw(PSEUDO)
    return OneCenter(paw, LSDA_PW92()), paw


def _rho(paw, nm, seed, scale):
    m0 = torch.zeros(nm, nm, dtype=torch.float64)
    col = 0
    for i, b in enumerate(paw.betas):
        for _ in range(2 * b.l + 1):
            m0[col, col] = paw.paw_occ[i] / (2 * b.l + 1)
            col += 1
    g = torch.Generator().manual_seed(seed)
    p = scale * torch.randn(nm, nm, generator=g, dtype=torch.float64)
    return m0 + (p + p.T) / 2


def test_onsite_nc_xc_collinear_limit_and_rotation():
    oc, paw = _oc()
    nm = sum(2 * b.l + 1 for b in paw.betas)
    rup = _rho(paw, nm, 1, 0.02) * 1.15
    rdn = _rho(paw, nm, 2, 0.02) * 0.85
    for what in ("ae", "ps"):
        up_lm, dn_lm = oc.rho_lm_t(rup, what), oc.rho_lm_t(rdn, what)
        e_col = float(oc._exc_t([up_lm, dn_lm], what))
        n_lm, mz_lm = up_lm + dn_lm, up_lm - dn_lm
        z = torch.zeros_like(n_lm)
        e_z = float(onsite_nc_exc(oc, [n_lm, z, z, mz_lm], what))     # m ∥ ẑ
        e_x = float(onsite_nc_exc(oc, [n_lm, mz_lm, z, z], what))     # m ∥ x̂
        s = 1 / np.sqrt(3)
        e_d = float(onsite_nc_exc(oc, [n_lm, mz_lm * s, mz_lm * s, mz_lm * s], what))
        assert abs(e_z - e_col) < 1e-9, f"{what}: collinear limit {e_z} vs {e_col}"
        assert abs(e_z - e_x) < 1e-9 and abs(e_z - e_d) < 1e-9, "rotation invariance"


def test_onsite_nc_xc_field_matches_fd():
    oc, paw = _oc()
    nm = sum(2 * b.l + 1 for b in paw.betas)
    up_lm = oc.rho_lm_t(_rho(paw, nm, 1, 0.02) * 1.15, "ae")
    dn_lm = oc.rho_lm_t(_rho(paw, nm, 2, 0.02) * 0.85, "ae")
    n_lm, mz_lm = up_lm + dn_lm, up_lm - dn_lm
    z = torch.zeros_like(n_lm)
    _, grads = onsite_nc_energy_and_field(oc, [n_lm, z, z, mz_lm], "ae")
    ad = float((grads[3] * mz_lm).sum())                 # dE/d(mz scale), autograd
    eps = 1e-5
    ep = float(onsite_nc_exc(oc, [n_lm, z, z, mz_lm * (1 + eps)], "ae"))
    em = float(onsite_nc_exc(oc, [n_lm, z, z, mz_lm * (1 - eps)], "ae"))
    assert abs(ad / ((ep - em) / (2 * eps)) - 1) < 1e-4


def test_onsite_nc_onecenter_ddd_collinear_limit():
    """The full one-center corrector (Hartree + XC) and its 2×2 on-site potential:
    in the collinear limit the energy matches the collinear energy_and_ddd, and the
    2×2 ddd reduces to [ddd_up, ddd_down] via ddd_n ± ddd_mz with zero off-diagonal."""
    oc, paw = _oc()
    nm = sum(2 * b.l + 1 for b in paw.betas)
    rup = _rho(paw, nm, 1, 0.02) * 1.15
    rdn = _rho(paw, nm, 2, 0.02) * 0.85
    e_col, (ddd_up, ddd_dn) = oc.energy_and_ddd([rup, rdn])
    n_ij, mz_ij = rup + rdn, rup - rdn
    zero = torch.zeros_like(n_ij)
    e_nc, (dn_, dmx, dmy, dmz) = onsite_nc_energy_and_ddd(oc, [n_ij, zero, zero, mz_ij])
    assert abs(e_nc - e_col) < 1e-8
    assert float((dn_ + dmz - ddd_up).abs().max()) < 1e-8
    assert float((dn_ - dmz - ddd_dn).abs().max()) < 1e-8
    assert float(dmx.abs().max()) < 1e-8 and float(dmy.abs().max()) < 1e-8


def test_spinor_onsite_becsum_hermitian_and_collinear_limit():
    """The 2×2 on-site becsum Pauli decomposition: every channel is Hermitian, and a
    pure-up spinor gives n = mz = the collinear up-becsum with zero off-diagonal."""
    torch.manual_seed(0)
    nb, npr = 6, 4
    bu = torch.randn(nb, npr, dtype=torch.complex128)
    bd = torch.randn(nb, npr, dtype=torch.complex128)
    w = torch.rand(nb, dtype=torch.float64)
    for c in spinor_onsite_becsum(bu, bd, w):
        assert float((c - c.conj().T).abs().max()) < 1e-12
    n0, mx0, my0, mz0 = spinor_onsite_becsum(bu, torch.zeros_like(bd), w)
    r = torch.einsum("b,bi,bj->ij", w, bu.conj(), bu)
    r = 0.5 * (r + r.conj().T)
    assert float((n0 - r).abs().max()) < 1e-12 and float((mz0 - r).abs().max()) < 1e-12
    assert float(mx0.abs().max()) < 1e-12 and float(my0.abs().max()) < 1e-12
