"""Bit-accuracy gate for the opt-in compiled XC path (compile_xc).

The XC transcendental chain is real-valued and fuses under Inductor, unlike the
complex FFT-bound Hamiltonian apply (see docs/torch-compile.md). enable_compile()
routes energy_density through torch.compile with an eager fallback. These tests
pin the contract that matters regardless of whether the host toolchain is present:

- With the toolchain present, the compiled forward and v_xc match eager to machine
  precision (the compiled callable is bit-accurate, not approximate), and the
  guarded f_xc HVP runs eager and matches exactly.
- With the toolchain absent (no openssl on PATH on a stock NixOS checkout, no
  Triton libcuda), the path latches to eager and returns the identical result, so
  the flag is always safe to leave on.

The f_xc HVP is a double backward. torch.compile with aot_autograd cannot double
backward, so the response and HVP call sites wrap their xc.energy() in xc_eager().
The test below exercises the guarded path and asserts it matches eager. The GGA
v_xc test covers the real SCF case where sigma is a function of rho in the outer
graph, which the compiled forward and its single backward handle correctly.
"""

import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE

# torch.compile tracing runs before the toolchain check even fails, so the first
# enable_compile() call costs on the order of a minute whether or not the host
# compiler is present. That is minutes-class, so this gate runs in the slow tier.
pytestmark = pytest.mark.slow

# A compiled result that latched to eager is bitwise identical, so use a tight
# tolerance that both the compiled (machine-precision) and eager-fallback (exact)
# cases satisfy.
RTOL = 1e-12


def _rho(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n**3, dtype=torch.float64, generator=g) * 0.5 + 0.05


def _sigma(n=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n**3, dtype=torch.float64, generator=g) * 0.1


def _vxc(xc, rho, sigma):
    rho = rho.detach().requires_grad_(True)
    s = sigma.detach().requires_grad_(True) if sigma is not None else None
    e = xc.eval_energy_density(rho, s).sum()
    return torch.autograd.grad(e, rho)[0]


def _fxc_hvp(xc, rho, sigma, vec):
    rho = rho.detach().requires_grad_(True)
    s = sigma.detach().requires_grad_(True) if sigma is not None else None
    e = xc.eval_energy_density(rho, s).sum()
    g = torch.autograd.grad(e, rho, create_graph=True)[0]
    return torch.autograd.grad(g, rho, grad_outputs=vec)[0]


def _rel(a, b):
    return (a - b).abs().max().item() / b.abs().max().item()


def test_compiled_forward_and_vxc_match_eager():
    rho, sigma, = _rho(), _sigma()
    for cls, needs in [(LDA_PW92, False), (PBE, True)]:
        s = sigma if needs else None
        eager, comp = cls(), cls().enable_compile()

        e_eager = eager.eval_energy_density(rho, s)
        e_comp = comp.eval_energy_density(rho, s)
        assert _rel(e_comp, e_eager) < RTOL, cls.__name__

        v_eager = _vxc(eager, rho, s)
        v_comp = _vxc(comp, rho, s)
        assert _rel(v_comp, v_eager) < RTOL, cls.__name__


def test_guarded_fxc_hvp_matches_eager():
    # Double backward is unsupported by compiled aot_autograd, so the f_xc HVP
    # sites wrap their xc.energy() in xc_eager(). Under that guard a compile-
    # enabled functional runs eager and its f_xc HVP matches eager exactly. This
    # is the contract the response and HVP code depends on.
    from gradwave.core.xc.base import xc_eager

    rho, sigma = _rho(), _sigma()
    vec = _rho(seed=7)
    for cls, needs in [(LDA_PW92, False), (PBE, True)]:
        s = sigma if needs else None
        eager, comp = cls(), cls().enable_compile()
        h_eager = _fxc_hvp(eager, rho, s, vec)
        with xc_eager():
            h_comp = _fxc_hvp(comp, rho, s, vec)
        assert _rel(h_comp, h_eager) < RTOL, cls.__name__


def test_compiled_gga_vxc_with_rho_coupled_sigma():
    # Real SCF GGA path: sigma = f(rho) in the outer graph, so v_xc = dE/drho
    # gets a contribution through sigma. The compiled forward returns partials wrt
    # both rho and sigma and the outer autograd composes them. This is the case a
    # naive custom-Function wrapper got wrong, so pin it.
    base = _rho()
    eager, comp = PBE(), PBE().enable_compile()

    def vxc_coupled(xc):
        rho = base.detach().requires_grad_(True)
        sigma = rho * rho * 0.3 + 0.01  # sigma depends on rho, in-graph
        e = xc.eval_energy_density(rho, sigma).sum()
        return torch.autograd.grad(e, rho)[0]

    assert _rel(vxc_coupled(comp), vxc_coupled(eager)) < RTOL


def test_compiled_spin_vxc_matches_eager():
    ru, rd = _rho(seed=2), _rho(seed=3)
    suu, sdd, stt = _sigma(seed=4), _sigma(seed=5), _sigma(seed=6)
    for cls, needs in [(LSDA_PW92, False), (SpinPBE, True)]:
        eager, comp = cls(), cls().enable_compile()

        def go(xc):
            a = ru.detach().requires_grad_(True)
            b = rd.detach().requires_grad_(True)
            if needs:
                args = (a, b, suu.detach().requires_grad_(True),
                        sdd.detach().requires_grad_(True),
                        stt.detach().requires_grad_(True))
            else:
                args = (a, b)
            e = xc.eval_energy_density(*args).sum()
            return torch.autograd.grad(e, (a, b))

        ge, gc = go(eager), go(comp)
        assert _rel(gc[0], ge[0]) < RTOL, cls.__name__
        assert _rel(gc[1], ge[1]) < RTOL, cls.__name__


def test_disable_and_eager_guard_round_trip():
    from gradwave.core.xc.base import xc_eager

    rho, sigma = _rho(), _sigma()
    xc = PBE().enable_compile()
    ref = PBE().eval_energy_density(rho, sigma)

    # xc_eager() forces the eager branch even with compile on.
    with xc_eager():
        e_guarded = xc.eval_energy_density(rho, sigma)
    assert _rel(e_guarded, ref) < RTOL

    # disable_compile() returns to plain eager permanently.
    xc.disable_compile()
    e_off = xc.eval_energy_density(rho, sigma)
    assert _rel(e_off, ref) < RTOL
