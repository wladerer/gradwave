"""Phase 1 FLAPW-DFPT: the torch-differentiable EFG evaluation + moving muffin-tin term.

Fast, pure-primitive (no SCF) gates for :mod:`gradwave.flapw.efg_torch`:

* value parity — the torch EFG eval reproduces the numpy reference V_zz to round-off;
* the FROZEN-DENSITY gradcheck — autograd dV_zz/dtau|_rho == frozen central finite
  difference (delta-halving confirms the linear regime), the decisive proof that the
  explicit moving-muffin-tin partial is captured correctly;
* the Theta(G) term is LOAD-BEARING — omitting the moving-boundary Pulay phase
  (``include_pulay=False``) collapses the autograd gradient to zero and breaks the
  gradcheck, exactly the ``grad_mask_autograd == 0`` control of the gate-A toy.

The state here is synthetic-but-valid (a real interstitial grid, real l=2 multipoles),
so the whole file runs in well under a second single-threaded. The real converged-rutile
numbers live in ``experiments/autoapw/efg_frozen_gradcheck.py`` (asus).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gradwave.constants import E2
from gradwave.flapw.efg import efg_tensor_full, interstitial_l2_boundary
from gradwave.flapw.efg_torch import build_boundary_ctx, efg_vzz_torch

_LSET2 = [(2, m) for m in range(-2, 3)]


def _synthetic_state():
    """A deterministic, valid frozen EFG state: log radial mesh, complex l=2 multipoles with a
    broken degeneracy (nonzero eta, no eigenvalue crossing near tau0), a real interstitial grid,
    a small own-sphere q^MT, an orthorhombic (rutile-like) cell, and an off-grid atom center."""
    rng = np.random.default_rng(20260821)
    R = 0.824                                             # O-like muffin-tin radius (Angstrom)
    nr = 300
    rr = np.geomspace(1e-4, R, nr)
    dx = math.log(rr[1] / rr[0])                          # constant log spacing
    drw = rr * dx
    env = rr**2 * np.exp(-((rr / (0.3 * R)) ** 2))
    amp = {0: 0.05 + 0j, 1: 0.02 + 0.01j, 2: 0.03 - 0.015j}
    rho = {}
    for m in range(-2, 3):
        a = amp[abs(m)] if m >= 0 else ((-1) ** m) * np.conj(amp[abs(m)])
        rho[(2, m)] = a * env
    rho[(0, 0)] = (0.4 * np.exp(-rr / (0.5 * R))).astype(complex)

    nfft = 12
    v_grid = rng.standard_normal((nfft, nfft, nfft))      # a real interstitial Coulomb grid
    cell = np.array([4.5936, 4.5936, 2.9585])             # rutile a,a,c in Angstrom
    qmt = {(2, 0): 0.01 + 0j, (2, 1): 0.005 - 0.002j, (2, 2): -0.008 + 0.003j}
    center = np.array([1.31, 1.77, 0.93])                 # off-grid atom position (Angstrom)
    return dict(rho=rho, rr=rr, drw=drw, v_grid=v_grid, cell=cell, R=R, qmt=qmt, center=center)


def _numpy_vzz(st, center):
    """The numpy reference V_zz, mirroring gradwave.flapw.scf._efg_from_multipoles for one atom."""
    R = st["R"]
    v_bc = interstitial_l2_boundary(st["v_grid"], center, R, st["cell"])
    v_bc = {m: v_bc[m] - (4 * math.pi * E2 / 5.0) * st["qmt"].get((2, m), 0.0) / R**3
            for m in range(-2, 3)}
    _tensor, v_zz, _eta = efg_tensor_full(st["rho"], st["rr"], st["drw"], v_bc, R)
    return v_zz


def _torch_inputs(st, device="cpu"):
    rho_t = {(2, m): torch.as_tensor(st["rho"][(2, m)], dtype=torch.complex128, device=device)
             for m in range(-2, 3)}
    rr_t = torch.as_tensor(st["rr"], dtype=torch.float64, device=device)
    drw_t = torch.as_tensor(st["drw"], dtype=torch.float64, device=device)
    ctx = build_boundary_ctx(st["v_grid"], st["R"], st["cell"], _LSET2, device=device)
    return rho_t, rr_t, drw_t, ctx


def test_torch_efg_value_matches_numpy():
    """The torch EFG eval reproduces the numpy reference V_zz to round-off (both branches pick the
    same |eigenvalue|-max root)."""
    torch.set_num_threads(1)
    st = _synthetic_state()
    rho_t, rr_t, drw_t, ctx = _torch_inputs(st)
    center = torch.as_tensor(st["center"], dtype=torch.float64)
    v_zz_torch = float(efg_vzz_torch(rho_t, rr_t, drw_t, ctx, center, st["R"], qmt=st["qmt"]))
    v_zz_numpy = _numpy_vzz(st, st["center"])
    assert v_zz_torch == pytest.approx(v_zz_numpy, rel=1e-10, abs=1e-10)


def test_frozen_density_gradcheck():
    """FROZEN-DENSITY gradcheck: autograd dV_zz/dtau|_rho == frozen central finite difference, with
    delta-halving confirming the O(delta^2) linear regime. This is the Phase-1 deliverable gate."""
    torch.set_num_threads(1)
    st = _synthetic_state()
    rho_t, rr_t, drw_t, ctx = _torch_inputs(st)

    center = torch.tensor(st["center"], dtype=torch.float64, requires_grad=True)
    v_zz = efg_vzz_torch(rho_t, rr_t, drw_t, ctx, center, st["R"], qmt=st["qmt"])
    (grad_ad,) = torch.autograd.grad(v_zz, center)

    def fd(delta):
        g = torch.zeros(3, dtype=torch.float64)
        for j in range(3):
            cp = center.detach().clone()
            cp[j] += delta
            cm = center.detach().clone()
            cm[j] -= delta
            vp = efg_vzz_torch(rho_t, rr_t, drw_t, ctx, cp, st["R"], qmt=st["qmt"]).detach()
            vm = efg_vzz_torch(rho_t, rr_t, drw_t, ctx, cm, st["R"], qmt=st["qmt"]).detach()
            g[j] = (vp - vm) / (2 * delta)
        return g

    grad_fd1 = fd(1e-4)
    grad_fd2 = fd(5e-5)
    err1 = float((grad_ad - grad_fd1).norm())
    err2 = float((grad_ad - grad_fd2).norm())

    # the gradient must be genuinely nonzero (otherwise the test is vacuous)
    assert float(grad_ad.norm()) > 1e-6
    # autograd matches the frozen FD, and halving delta shrinks the residual ~4x (O(delta^2))
    assert err1 < 1e-4 * max(float(grad_ad.norm()), 1.0)
    assert err2 < 0.4 * err1


def test_pulay_term_is_load_bearing():
    """OMITTING the moving muffin-tin Theta(G) phase (include_pulay=False) collapses the autograd
    gradient to zero, so it no longer matches the frozen finite difference. This is the control that
    proves the term is real and correctly captured (mirrors gate-A's grad_mask_autograd == 0)."""
    torch.set_num_threads(1)
    st = _synthetic_state()
    rho_t, rr_t, drw_t, ctx = _torch_inputs(st)

    center = torch.tensor(st["center"], dtype=torch.float64, requires_grad=True)
    v_zz_nop = efg_vzz_torch(rho_t, rr_t, drw_t, ctx, center, st["R"], qmt=st["qmt"],
                             include_pulay=False)
    # With the moving-boundary phase omitted, tau enters NOWHERE: V_zz is a graph constant, so it
    # carries no grad_fn and the autograd gradient is identically zero.
    assert not v_zz_nop.requires_grad
    grad_nop = torch.zeros(3, dtype=torch.float64)

    # frozen FD of the TRUE (Pulay-on) function — the physical response the omission misses
    def fd_true(delta=1e-4):
        g = torch.zeros(3, dtype=torch.float64)
        for j in range(3):
            cp = center.detach().clone()
            cp[j] += delta
            cm = center.detach().clone()
            cm[j] -= delta
            vp = efg_vzz_torch(rho_t, rr_t, drw_t, ctx, cp, st["R"], qmt=st["qmt"]).detach()
            vm = efg_vzz_torch(rho_t, rr_t, drw_t, ctx, cm, st["R"], qmt=st["qmt"]).detach()
            g[j] = (vp - vm) / (2 * delta)
        return g

    fd_norm = float(fd_true().norm())
    assert float(grad_nop.norm()) < 1e-12          # autograd sees no tau-dependence
    assert fd_norm > 1e-4                            # the omitted term is large and load-bearing
