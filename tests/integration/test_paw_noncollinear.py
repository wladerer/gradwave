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
from gradwave.scf.paw_noncollinear import onsite_nc_energy_and_field, onsite_nc_exc
from gradwave.scf.paw_onsite import OneCenter

PSEUDO = "tests/fixtures/qe/pseudos/Si.pbe-n-kjpaw_psl.1.0.0.UPF"


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
