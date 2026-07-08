"""Differentiable SBT (pseudo/radial_torch.py) vs the numpy reference."""

from pathlib import Path

import numpy as np
import torch

from gradwave.pseudo import radial
from gradwave.pseudo.local import vloc_of_g
from gradwave.pseudo.radial_torch import RadialTables, jl_t, sbt_t, simpson_weights
from gradwave.pseudo.upf import parse_upf

PSE = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"


def test_jl_matches_reference():
    x = np.concatenate([np.linspace(0.0, 8.0, 2001), [1e-8, 3.999999, 4.000001, 50.0]])
    for l in range(4):
        ref = radial.sph_jl(l, x)
        got = jl_t(l, torch.tensor(x)).numpy()
        assert np.abs(got - ref).max() < 1e-14, l


def test_jl4_via_recurrence():
    x = np.linspace(4.01, 60.0, 500)
    j4 = (7.0 / x) * radial.sph_jl(3, x) - radial.sph_jl(2, x)
    got = jl_t(4, torch.tensor(x)).numpy()
    assert np.abs(got - j4).max() < 1e-13


def test_sbt_values_and_derivative():
    upf = parse_upf(PSE / "Si_ONCV_PBE-1.2.upf")
    q = np.linspace(1e-3, 12.0, 41)
    tab = RadialTables(upf)
    for i, b in enumerate(upf.betas):
        n = b.cutoff_idx
        ref = radial.sbt(b.l, b.rbeta * upf.r[:n], upf.r[:n], upf.rab[:n], q)
        qt = torch.tensor(q, requires_grad=True)
        got = tab.beta_of_g(i, qt)
        assert np.abs(got.detach().numpy() - ref).max() < 1e-13

        (grad,) = torch.autograd.grad(got.sum(), qt)
        d = 1e-6
        fd = (radial.sbt(b.l, b.rbeta * upf.r[:n], upf.r[:n], upf.rab[:n], q + d)
              - radial.sbt(b.l, b.rbeta * upf.r[:n], upf.r[:n], upf.rab[:n], q - d)) / (2 * d)
        assert np.abs(grad.numpy() - fd).max() < 1e-8


def test_vloc_table_matches_reference():
    for name in ("Si_ONCV_PBE-1.2.upf", "PD_Ni_PBE.upf"):
        upf = parse_upf(PSE / name)
        q = np.linspace(0.05, 15.0, 37)
        tab = RadialTables(upf)
        ref = vloc_of_g(upf, q)
        got = tab.vloc_of_g(torch.tensor(q)).numpy()
        assert np.abs(got - ref).max() < 1e-11, name


def test_simpson_weights_match_reference():
    for n in (245, 246):  # odd and even closures
        rab = np.linspace(0.01, 0.02, n)
        f = np.exp(-np.linspace(0, 4, n))
        ref = radial.simpson(f, rab)
        got = float((torch.tensor(f) * torch.tensor(simpson_weights(rab))).sum())
        assert abs(got - ref) < 1e-15


def test_qe_msh_truncation():
    """PD_Ni's mesh reaches 13.7 bohr; the local channel must stop at QE's
    10-bohr msh (rigid −7.9 meV v_loc(0) offset otherwise — see upf.py)."""
    upf = parse_upf(PSE / "PD_Ni_PBE.upf")
    assert upf.msh == 1001
    assert upf.msh < len(upf.r)
    q = np.array([1.0, 5.0])
    tab = RadialTables(upf)
    assert np.abs(tab.vloc_of_g(torch.tensor(q)).numpy() - vloc_of_g(upf, q)).max() < 1e-11


def test_sbt_analytic_gaussian():
    """SBT of a Gaussian against the closed form, including through backward:
    ∫ e^{−ar²} j₀(qr) r² dr = (√π/4a^{3/2}) e^{−q²/4a}."""
    a = 1.7
    r = np.linspace(0, 12.0, 4001)
    rab = np.full_like(r, r[1] - r[0])
    g = np.exp(-a * r**2) * r**2
    q = torch.tensor(np.linspace(0.1, 6.0, 23), requires_grad=True)
    got = sbt_t(0, torch.tensor(g), torch.tensor(r), torch.tensor(simpson_weights(rab)), q)
    ref = np.sqrt(np.pi) / (4 * a**1.5) * np.exp(-(q.detach().numpy() ** 2) / (4 * a))
    assert np.abs(got.detach().numpy() - ref).max() < 1e-12
    (grad,) = torch.autograd.grad(got.sum(), q)
    dref = ref * (-q.detach().numpy() / (2 * a))
    assert np.abs(grad.numpy() - dref).max() < 1e-11
