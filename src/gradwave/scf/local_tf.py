"""Local Thomas–Fermi density preconditioner (position-dependent Kerker).

The bare Kerker filter R̃(G) = R(G)·G²/(G²+q0²) screens long-wavelength charge
sloshing with a SINGLE screening length 1/q0. That is the right operator for a
bulk metal, where the density is roughly uniform, but it is the wrong operator
for an inhomogeneous cell (a slab, a molecule, anything with vacuum): a single
q0 either over-screens the vacuum, where the density cannot respond and the
long-wavelength modes must be free to move, or under-screens the dense region.
Measured on a 4-layer Al(100) slab the constant Kerker at the bulk q0 was WORSE
than turning it off (21 vs 18 iterations), while the best constant q0 (0.5) took
17 — no single value is right everywhere.

The fix, following Quantum ESPRESSO's ``mixing_mode='local-TF'``, is to let the
screening wavevector track the local density,

    q²(r) = min( q²_TF(r),  q0_max² ),   q²_TF(r) = (4/π) k_F(r) / a0² ,
    k_F(r) = (3π² n(r))^{1/3}  (atomic units),

so the metal is screened at the (capped) bulk value and the vacuum, where
n(r) → 0, is left unscreened. The cap is load-bearing: the *uncapped* Thomas–
Fermi wavevector over-screens even a bulk metal (fcc Al: full TF q ≈ 2.5 Å⁻¹
converges in 11 iterations against 9 at the bulk 1.1 Å⁻¹), so q0_max is set to
the same value the bare Kerker already uses and the local screening never
exceeds it. In the constant-density limit q²(r) → q0_max² and this operator
reproduces the bare Kerker filter to round-off.

The operator is not diagonal in either representation, so it is applied by a
short conjugate-gradient solve of the screened-Poisson equation, a handful of
box FFTs per mixing step (QE's ``approx_screening2`` has the same cost). The
CG state is warm-started across SCF iterations, where the residual changes
slowly, so the amortized cost is a couple of FFT pairs per SCF step.

Preconditioner form.  P = -∇² (-∇² + Q̂)⁻¹ = I − Q̂(-∇² + Q̂)⁻¹, with Q̂ the
multiplication by q²(r).  P·R = R − Q̂u where (-∇² + Q̂)u = R.  For constant
q²(r) = q0², u = R/(G²+q0²) and P·R = R·G²/(G²+q0²) exactly.  The G=0 component
is preserved automatically: integrating the screened-Poisson equation over the
cell gives (Q̂u)(0) = R(0) = 0, so P leaves the pinned charge untouched (it is
re-zeroed after the solve as cheap insurance against CG round-off).
"""

from __future__ import annotations

import torch

from gradwave.constants import BOHR_ANG as _BOHR  # shared CODATA Bohr radius [Å]
from gradwave.core.fftbox import g_to_r_box


class LocalTFPrecond:
    """Position-dependent Thomas–Fermi Kerker preconditioner on the density
    sphere.  Construct once per SCF (it caches the box Laplacian and the
    sphere↔box index map), call :meth:`set_density` each iteration with the
    current real-space total density, then use the instance as the mixer's
    ``precond_op``: ``P·r`` for a residual ``r`` over the density sphere."""

    def __init__(self, g2_box: torch.Tensor, shape, mask_flat: torch.Tensor,
                 q0_max: float = 1.1, cg_iters: int = 12, cg_tol: float = 1e-3):
        # g2_box: (n_points,) |G|² over the FULL box [Å⁻²]; shape: box dims;
        # mask_flat: bool (n_points,) selecting the density sphere.
        self.g2 = g2_box.reshape(-1)
        self.shape = tuple(shape)
        self.n_points = self.g2.shape[0]
        self.mask = mask_flat.reshape(-1)
        self.q0_max2 = float(q0_max) ** 2
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol
        self.q2_r = None      # (*shape,) real, q²(r) [Å⁻²]
        self._u_warm = None   # (n_points,) complex, warm-started CG solution

    def set_density(self, rho_r: torch.Tensor) -> None:
        """Build q²(r) from the current total density n(r) [e/Å³]."""
        n = rho_r.reshape(self.shape).clamp_min(0.0)
        # k_F in atomic units from n in bohr⁻³, then q²_TF = (4/π) k_F in bohr⁻²,
        # converted to Å⁻². Everything below the vacuum floor screens at ~0.
        n_bohr = n * (_BOHR ** 3)
        kf = (3.0 * torch.pi ** 2 * n_bohr).clamp_min(0.0) ** (1.0 / 3.0)  # bohr⁻¹
        q2_bohr = (4.0 / torch.pi) * kf                                    # bohr⁻²
        q2_ang = q2_bohr / (_BOHR ** 2)                                    # Å⁻²
        self.q2_r = q2_ang.clamp_max(self.q0_max2)

    def _apply_operator(self, u_box: torch.Tensor) -> torch.Tensor:
        """(-∇² + Q̂) u in box G-space.  u_box, out: (n_points,) complex."""
        lap = self.g2 * u_box
        u_r = g_to_r_box(u_box.reshape(self.shape))
        qu = torch.fft.fftn(self.q2_r * u_r, dim=(-3, -2, -1)).reshape(-1) / self.n_points
        return lap + qu

    def __call__(self, r_sphere: torch.Tensor) -> torch.Tensor:
        """P·r for a residual r over the density sphere (returns the same
        sphere layout).  ``r_sphere`` may be a single density block; callers
        that mix several blocks (e.g. the magnetization channel) apply this to
        the total block only."""
        if self.q2_r is None:
            raise RuntimeError("LocalTFPrecond.set_density must be called each "
                               "iteration before applying the preconditioner")
        # scatter the sphere residual into the full box
        rhs = torch.zeros(self.n_points, dtype=r_sphere.dtype, device=r_sphere.device)
        rhs[self.mask] = r_sphere

        # PRECONDITIONED conjugate gradient on the SPD screened-Poisson operator,
        # warm-started. The diagonal preconditioner M⁻¹ = 1/(G²+q0_max²) is the
        # EXACT inverse of the operator in the constant-density limit (q²(r) →
        # q0_max²), so PCG converges in one step there and reproduces the bare
        # Kerker filter exactly; the position-dependent part of q²(r) is a small
        # perturbation, so a handful of iterations suffice everywhere else.
        # Unpreconditioned CG cannot solve this — the operator's condition number
        # is G²_max/q0² and the high-G components stay under-solved, degrading P
        # into a poor Kerker approximation (measured: bulk Al 9→15 iterations).
        minv = 1.0 / (self.g2 + self.q0_max2)
        u = (self._u_warm if self._u_warm is not None
             and self._u_warm.shape == rhs.shape
             else torch.zeros_like(rhs))
        resid = rhs - self._apply_operator(u)
        z = minv * resid
        p = z.clone()
        rz_old = (resid.conj() @ z).real
        rhs_norm2 = (rhs.conj() @ rhs).real.clamp_min(1e-300)
        for _ in range(self.cg_iters):
            if (resid.conj() @ resid).real / rhs_norm2 < self.cg_tol ** 2:
                break
            ap = self._apply_operator(p)
            denom = (p.conj() @ ap).real
            if denom <= 0:
                break
            alpha = rz_old / denom
            u = u + alpha * p
            resid = resid - alpha * ap
            z = minv * resid
            rz_new = (resid.conj() @ z).real
            p = z + (rz_new / rz_old) * p
            rz_old = rz_new
        self._u_warm = u

        # P·R = R − Q̂u, evaluated on the box then gathered back to the sphere
        u_r = g_to_r_box(u.reshape(self.shape))
        qu = torch.fft.fftn(self.q2_r * u_r, dim=(-3, -2, -1)).reshape(-1) / self.n_points
        pr_box = rhs - qu
        pr_box[0] = 0.0  # pin G=0 (preserved analytically; guard CG round-off)
        return pr_box[self.mask]
