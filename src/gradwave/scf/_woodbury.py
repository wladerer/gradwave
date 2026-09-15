"""The low-rank Woodbury inverse shared by the two χ₀-preconditioners.

Both ``scf.spin_precond.StonerSpinPrecond`` (the Stoner m-channel
(I − χ₀^diag K_mm)⁻¹) and ``scf.subspace_chi0.WoodburyPrecond`` (the coupled
charge+spin subspace χ₀ preconditioner) apply the SAME operator
``(1 − U diag(c) W†)⁻¹`` via the Sherman-Morrison-Woodbury identity: factor the
small r×r capacitance matrix once, then solve it per residual. The two classes
differ only in how they BUILD the low-rank columns (U, W, c) and in their
channel bookkeeping (single m-channel vs stacked up/dn with a G=0 pin) — the
inverse algebra itself is identical, and lives here so a change to it (or a bug
in it) has one home.

Density-sphere pairing convention: ⟨a, b⟩ = Ω Σ_G â* b̂, so the cell volume Ω
is carried into diag(1/c) here (not into the columns).
"""

from __future__ import annotations

import torch

from gradwave.dtypes import CDTYPE


def woodbury_factor(u_g: torch.Tensor, w_g: torch.Tensor, cvals: torch.Tensor,
                    volume: float):
    """LU-factor the r×r capacitance ``A = diag(1/(Ω c)) − W†U`` for the low-rank
    inverse ``(1 − U diag(c) W†)⁻¹``. Returns the ``torch.linalg.lu_factor``
    tuple, consumed by :func:`woodbury_apply`.

    ``u_g`` / ``w_g``: (r, ng) codensity columns φ̂_p and their K-images
    (K φ_p)^ on the density sphere. ``cvals``: (r,) real dyad weights c_p.
    ``volume``: Ω (the pairing factor)."""
    d_inv = torch.diag(1.0 / (volume * cvals.to(CDTYPE)))
    a = d_inv - torch.einsum("ag,bg->ab", w_g.conj(), u_g)
    return torch.linalg.lu_factor(a)


def woodbury_apply(u_g: torch.Tensor, w_g: torch.Tensor, a_lu,
                   r: torch.Tensor) -> torch.Tensor:
    """Apply ``(1 − U diag(c) W†)⁻¹`` to a sphere vector ``r`` given the factored
    capacitance ``a_lu`` from :func:`woodbury_factor`. Solves the small r×r
    system for the projection and adds back the rank-r correction."""
    proj = torch.einsum("ag,g->a", w_g.conj(), r)
    sol = torch.linalg.lu_solve(*a_lu, proj[:, None])[:, 0]
    return r + torch.einsum("ag,a->g", u_g, sol)
