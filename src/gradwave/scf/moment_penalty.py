"""Differentiable atomic-moment penalties for constrained non-collinear DFT.

Both the constraining field that enters the spinor Hamiltonian and the
direction-torque that rotates target moments are obtained by autograd on a
single scalar penalty E_p(M, ê). Swapping the penalty form is therefore a
one-line change with no hand-derived gradients to keep in sync, and the SCF
field and the config-search torque are guaranteed consistent because they
differentiate the same function.

Penalty modes
-------------
"perp"    E_p = λ Σ_I |M_I^⊥|²            (Ma-Dudarev direction penalty)
          M_I^⊥ = M_I − (M_I·ê_I) ê_I. Constrains only the *direction* of each
          atomic moment. It is minimized (E_p → 0) by M_I → 0, so a frustrated
          magnet can satisfy it for free by demagnetizing — the "magnitude
          problem": a strongly-coupled pair forced to a large relative angle
          collapses its moments instead of holding them apart.

"vector"  E_p = λ Σ_I |M_I − m0_I ê_I|²   (full moment-vector target)
          Pins both direction *and* magnitude toward m0_I ê_I. Demagnetization
          now costs λ m0_I², so a moment forced to a large relative angle is
          held at full magnitude rather than collapsing. m0_I is a target
          magnitude [μB], e.g. the unconstrained self-consistent |M_I|.

The field the SCF adds to the exchange field b_xc = δE/δm⃗ is
B_c(r) = δE_p/δm⃗ = Σ_I (∂E_p/∂M_I) w_I(r), since M_I = ∫ w_I m⃗ dr. The
config-search gradient is the envelope derivative dW/dê_I = ∂E_p/∂ê_I taken at
the converged (fixed) moment, projected transverse to ê_I.
"""

from __future__ import annotations

import torch


def penalty_energy(M, dirs, lam: float, mode: str = "perp", target_mag=None):
    """Scalar penalty E_p [eV]. M and dirs are (na,3); target_mag is a per-atom
    magnitude (na,) or a scalar (broadcast to all atoms), required for
    mode='vector'."""
    if mode == "perp":
        m_par = (M * dirs).sum(-1, keepdim=True) * dirs   # (M·ê) ê
        return lam * ((M - m_par) ** 2).sum()
    if mode == "vector":
        if target_mag is None:
            raise ValueError("mode='vector' needs target_mag (per-atom |M| target)")
        tm = torch.as_tensor(target_mag, dtype=M.dtype, device=M.device)
        if tm.ndim == 0:
            tm = tm.expand(M.shape[0])
        return lam * ((M - tm.unsqueeze(-1) * dirs) ** 2).sum()
    raise ValueError(f"unknown constraint mode {mode!r}")


def field_coeff(m_at, dirs, lam: float, mode: str = "perp", target_mag=None):
    """Per-atom constraining field g_I = ∂E_p/∂M_I, shape (na,3). The real-space
    field added to b_xc is B_c(r) = Σ_I g_I w_I(r). Computed by autograd so it
    always tracks `penalty_energy`. Safe to call inside a torch.no_grad SCF loop:
    it enables grad locally on the tiny (na,3) moment tensor."""
    with torch.enable_grad():
        M = m_at.detach().requires_grad_(True)
        ep = penalty_energy(M, dirs, lam, mode, target_mag)
        (g,) = torch.autograd.grad(ep, M)
    return g


def direction_gradient(M, dirs, lam: float, mode: str = "perp", target_mag=None):
    """Envelope gradient dW/dê_I = ∂E_p/∂ê_I, projected transverse to ê_I,
    shape (na,3). Held at the converged moment M (detached). The descent torque
    that rotates ê_I downhill in energy is its negative."""
    with torch.enable_grad():
        d = dirs.detach().requires_grad_(True)
        ep = penalty_energy(M.detach(), d, lam, mode, target_mag)
        (g,) = torch.autograd.grad(ep, d)
    return g - (g * dirs).sum(-1, keepdim=True) * dirs
