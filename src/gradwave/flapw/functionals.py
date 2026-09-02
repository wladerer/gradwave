"""LDA exchange-correlation for the FLAPW stack (eV/Å units).

Slater exchange + Perdew-Wang 1992 correlation. ``vxc_lda`` returns the XC potential in eV for a
number density in e/Å³.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradwave.constants import BOHR_ANG, E2, HARTREE_EV


def _pw92_ec(rs: Tensor) -> Tensor:
    """Perdew-Wang 1992 correlation energy per particle ε_c(r_s) in Hartree (spin-unpolarized).

    The shared kernel of :func:`vxc_lda` and :func:`fxc_lda`; each caller prepares ``rs`` with its
    own autograd wiring (a detached leaf for the potential, a ρ-chained graph for the kernel) and
    handles the ``r_s`` derivative itself — this returns only ε_c(r_s), bit-identically for both.
    """
    A, a1 = 0.031091, 0.21370
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    denom = 2 * A * (b1 * rs**0.5 + b2 * rs + b3 * rs**1.5 + b4 * rs**2)
    return -2 * A * (1 + a1 * rs) * torch.log(1 + 1.0 / denom)


def vxc_lda(rho: Tensor) -> Tensor:
    """LDA XC potential (eV): Slater exchange + PW92 correlation. ``rho`` in e/Å³.

    Exchange ``V_x = -(3/π)^{1/3} e² ρ^{1/3}``; correlation via the PW92 parametrization with
    ``V_c = ε_c - (r_s/3) dε_c/dr_s`` (the derivative taken with autograd).
    """
    rho = rho.clamp_min(1e-12)
    vx = -((3.0 / math.pi) ** (1.0 / 3.0)) * E2 * rho ** (1.0 / 3.0)
    rho_au = rho * BOHR_ANG**3
    rs = (3.0 / (4.0 * math.pi * rho_au)) ** (1.0 / 3.0)
    rs = rs.detach().clone().requires_grad_(True)
    ec = _pw92_ec(rs)
    (dec,) = torch.autograd.grad(ec.sum(), rs)
    vc_ha = (ec - (rs / 3.0) * dec).detach()
    return vx + vc_ha * HARTREE_EV


def fxc_lda(rho: Tensor) -> Tensor:
    """LDA XC kernel ``f_xc = dV_xc/dρ`` (eV·Å³), elementwise — the density-response kernel that
    screens the FLAPW-DFPT density perturbation (``K_Hxc`` = Hartree ⊕ this). Same Slater-x + PW92-c
    parametrization as :func:`vxc_lda`; unlike ``vxc_lda`` (which detaches the internal ``r_s``
    derivative and so is not a clean autograd path to the kernel) the correlation second
    derivative is taken here through a *connected* graph — ``ρ`` is the leaf, ``r_s(ρ)`` chains,
    and the inner ``dε_c/dr_s`` is built with ``create_graph=True`` so the outer ``d/dρ`` sees it.
    Validated elementwise against a central finite difference of :func:`vxc_lda` to ~1e-10."""
    rho = rho.clamp_min(1e-12).detach().clone().requires_grad_(True)
    vx = -((3.0 / math.pi) ** (1.0 / 3.0)) * E2 * rho ** (1.0 / 3.0)
    rho_au = rho * BOHR_ANG**3
    rs = (3.0 / (4.0 * math.pi * rho_au)) ** (1.0 / 3.0)
    ec = _pw92_ec(rs)  # shared PW92 kernel; rs chains ρ so the outer d/dρ sees it
    (dec,) = torch.autograd.grad(ec.sum(), rs, create_graph=True)
    vc_ha = ec - (rs / 3.0) * dec
    vxc = vx + vc_ha * HARTREE_EV
    (fxc,) = torch.autograd.grad(vxc.sum(), rho)
    return fxc.detach()
