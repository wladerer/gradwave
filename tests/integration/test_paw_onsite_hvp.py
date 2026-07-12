"""One-center torch chain (task #58 milestone 2, phase A).

The dense-T ρ_lm maps + torch radial Poisson must reproduce the numpy
reference implementations exactly; ddd must be the exact energy derivative
(becsum-space FD of e1c_t); the Hessian-vector product must match FD of ddd.
These are the building blocks of the USPP/PAW self-consistent adjoint.
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.paw_onsite import OneCenter

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


def _rho(paw, seed, scale=0.03):
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


def test_torch_chain_matches_numpy_reference():
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    oc = OneCenter(paw, PBE())
    rho = _rho(paw, 7)
    for what in ("ae", "ps"):
        ref = oc.rho_lm(rho.numpy(), what)
        new = oc.rho_lm_t(rho, what).numpy()
        assert np.abs(ref - new).max() < 1e-13 * max(1.0, np.abs(ref).max())
    rl = oc.rho_lm(rho.numpy(), "ae")
    v_ref, e_ref = oc.hartree(rl)
    v_new, e_new = oc.hartree_t(torch.as_tensor(rl))
    assert np.abs(v_ref - v_new.numpy()).max() < 1e-12 * max(1.0, np.abs(v_ref).max())
    assert abs(e_ref - float(e_new)) < 1e-12 * max(1.0, abs(e_ref))


def test_ddd_is_exact_energy_derivative():
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    oc = OneCenter(paw, PBE())
    rho = _rho(paw, 7)
    _, ddd = oc.energy_and_ddd(rho)
    gen = torch.Generator().manual_seed(23)
    v = torch.randn(*rho.shape, generator=gen, dtype=torch.float64)
    v = (v + v.T) / 2
    eps = 1e-5
    ep, _ = oc.energy_and_ddd(rho + eps * v)
    em, _ = oc.energy_and_ddd(rho - eps * v)
    fd = (ep - em) / (2 * eps)
    an = float((ddd * v).sum())
    assert abs(an - fd) < 1e-7 * max(1.0, abs(fd)), (an, fd)


def test_hvp_vs_fd_of_ddd():
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    for xc, spin in ((PBE(), False), (SpinPBE(), True)):
        oc = OneCenter(paw, xc)
        rho = _rho(paw, 7)
        gen = torch.Generator().manual_seed(23)
        v = torch.randn(*rho.shape, generator=gen, dtype=torch.float64)
        v = (v + v.T) / 2
        if spin:
            arg = [rho / 2 + 0.01 * _rho(paw, 11), rho / 2 - 0.01 * _rho(paw, 11)]
            vec = [v, -0.5 * v]
        else:
            arg, vec = rho, v
        hv = oc.hvp_becsum(arg, vec)
        eps = 1e-5
        if spin:
            _, dp = oc.energy_and_ddd([a + eps * w for a, w in zip(arg, vec, strict=True)])
            _, dm = oc.energy_and_ddd([a - eps * w for a, w in zip(arg, vec, strict=True)])
            for h, p, m in zip(hv, dp, dm, strict=True):
                fd = (p - m) / (2 * eps)
                assert float((h - fd).abs().max()) < 2e-6 * max(
                    1.0, float(fd.abs().max()))
        else:
            _, dp = oc.energy_and_ddd(arg + eps * vec)
            _, dm = oc.energy_and_ddd(arg - eps * vec)
            fd = (dp - dm) / (2 * eps)
            assert float((hv - fd).abs().max()) < 2e-6 * max(
                1.0, float(fd.abs().max()))
