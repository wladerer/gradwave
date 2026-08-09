"""Density mixing: linear, Kerker-preconditioned, Pulay/Anderson (Layer B).

Mixing operates on ρ(G) over the density sphere (complex vectors). The G=0
component is pinned (both ρ_in and ρ_out integrate to N_e, so the residual
there is zero by construction — checked).

Kerker: R̃(G) = R(G)·G²/(G² + q0²) suppresses long-wavelength charge
sloshing in metals; q0 default 1.1 Å⁻¹ (~0.58 bohr⁻¹). Off for insulators
(it slows their convergence).

Pulay: minimize ‖Σ c_i R_i‖ with Σ c_i = 1 in the Kerker-weighted inner
product ⟨R, R'⟩ = Σ_G Re[R*R']/(G² + q0²) (QE-style metric emphasizing
long-range components), via a bordered linear system; restart on
ill-conditioning.
"""

from __future__ import annotations

import logging

import torch
from typing_extensions import override

logger = logging.getLogger(__name__)


class _DampedMixerBase:
    """Shared plumbing for every mixer: the damped, optionally
    Kerker-preconditioned plain step P·r with per-component step scaling
    and the per-iteration extra_precond hook (Stoner spin preconditioner).
    Subclasses own only their history state and step() algorithm."""

    def __init__(
        self,
        g2: torch.Tensor,  # (n,) |G|² per component; 0 on non-grid blocks
        alpha: float = 0.7,
        history: int = 8,
        kerker: bool = False,
        q0: float = 1.1,
        check_g0: bool = True,
        kerker_mask: torch.Tensor | None = None,  # per-component bool; None → kerker on all
        step_scale: torch.Tensor | float | None = None,  # per-component or global multiplier
    ) -> None:
        self.g2 = g2
        self.alpha = alpha
        self.history = history
        self.kerker = kerker
        self.q0 = q0
        self.check_g0 = check_g0
        self.kerker_mask = kerker_mask
        self.step_scale = step_scale
        self.extra_precond = None
        # optional position-dependent preconditioner on the density-total block
        # (local Thomas–Fermi). When set it REPLACES the constant Kerker factor
        # on that block; other blocks (magnetization, becsum) keep plain damping.
        self.precond_op = None
        self.precond_slice = None  # slice of r the precond_op acts on (None → all)

    def _damped(self, r: torch.Tensor) -> torch.Tensor:
        if self.extra_precond is not None:
            r = self.extra_precond(r)
        if self.precond_op is not None:
            if self.precond_slice is None:
                out = self.alpha * self.precond_op(r)
            else:
                out = self.alpha * r.clone()
                sl = self.precond_slice
                out[sl] = self.alpha * self.precond_op(r[sl])
        elif not self.kerker:
            out = self.alpha * r
        else:
            fac = self.g2 / (self.g2 + self.q0**2)
            if self.kerker_mask is not None:
                fac = torch.where(self.kerker_mask, fac, torch.ones_like(fac))
            out = self.alpha * fac * r
        if self.step_scale is not None:
            out = out * self.step_scale
        return out

    @property
    def block_mult(self) -> None:
        """Per-block adaptive-damping multipliers {block_id: multiplier} for
        the driver's result dict, or None when this mixer has no per-block
        adaptation (only PulayMixer with adapt_blocks tracks them)."""
        return None


class BroydenMixer(_DampedMixerBase):
    """Limited-memory Broyden's second method (QE mixing_mode='plain').

    Maintains an approximate inverse Jacobian B of the residual map through
    sequential rank-one secant updates B y_i = s_i (s = Δρ_in, y = Δres),
    with the damped Kerker-preconditioned step as the seed B₀ = −αP. The
    step is the quasi-Newton ρ − B·res. Where Pulay/Anderson minimizes over
    a residual history and falls back to the PLAIN damped step outside the
    span of stored residual differences, Broyden's compounded updates keep
    a directional gain estimate — the difference that matters for modes the
    plain step amplifies (FM metals near the Stoner instability: per-block
    scalar damping heuristics either over-damp or ride the stability
    boundary, measured on fcc Ni; the secant update captures the expansive
    direction's gain and inverts it).

    Raw (s, y) pairs are stored and the update stack is rebuilt
    sequentially each step, so dropping the oldest pair is exact
    limited-memory Broyden on the window (m² dot products per step)."""

    def __init__(self, g2: torch.Tensor, **kw) -> None:
        super().__init__(g2, **kw)
        self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._prev_in: torch.Tensor | None = None
        self._prev_res: torch.Tensor | None = None

    def _apply_b(
        self, v: torch.Tensor, us: list[torch.Tensor], ys: list[torch.Tensor]
    ) -> torch.Tensor:
        """B v with B = −αP + Σ u_i ⟨y_i|·⟩ (rank-one secant corrections)."""
        out = -self._damped(v)
        for u, y in zip(us, ys, strict=True):
            out = out + u * (y.conj() @ v)
        return out

    def reset(self) -> None:
        self._pairs.clear()
        self._prev_in = None
        self._prev_res = None

    def step(self, rho_in: torch.Tensor, rho_out: torch.Tensor) -> torch.Tensor:
        res = rho_out - rho_in
        if self.check_g0 and res[0].abs() >= 1e-8:
            raise ValueError("G=0 residual nonzero")
        if self._prev_in is not None:
            # invariant: _prev_in and _prev_res are always set together
            # (reset()/__init__ clear both, step()'s tail sets both)
            assert self._prev_res is not None
            s = rho_in - self._prev_in
            y = res - self._prev_res
            yy = float((y.conj() @ y).real)
            rr = float((res.conj() @ res).real)
            # near-degenerate y (tiny relative to the current residual)
            # gives an exploding rank-one term — skip the pair
            if yy > 1e-12 * max(rr, 1e-300):
                self._pairs.append((s, y))
                if len(self._pairs) > self.history:
                    self._pairs.pop(0)
        us: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for s, y in self._pairs:
            by = self._apply_b(y, us, ys)
            us.append((s - by) / (y.conj() @ y))
            ys.append(y)
        self._prev_in, self._prev_res = rho_in, res
        return rho_in - self._apply_b(res, us, ys)


class JohnsonMixer(_DampedMixerBase):
    """Johnson's modified Broyden method (PRB 38, 12807) — the QE scheme.

    Multisecant inverse-Jacobian update over normalized residual
    differences with heavy Tikhonov regularization (w0): the step is

        x' = x + P f − Σ_l γ_l u_l,   γ = (w0² I + A)⁻¹ c,
        A_ij = ⟨ΔF̂_i, ΔF̂_j⟩,  c_i = ⟨ΔF̂_i, f⟩,
        u_i = (P ΔF_i + Δx_i)/|ΔF_i|,   ΔF̂ = ΔF/|ΔF|,

    with P the damped (Kerker-preconditioned) step. Three properties the
    plain sequential Broyden lacks, and why QE's version survives the
    wild early iterations that made the unweighted variant diverge on FM
    Ni: pair normalization (scale invariance across a residual range of
    10⁴), w0-regularized simultaneous solve (near-parallel garbage pairs
    are damped, not exactly enforced), and no compounding of stale
    corrections.

    Measured Ni defaults (etol 1e-6 energy criterion, default alpha):
    history 8 → 44 iterations, 12 → 27, 16 → 26, 25 → 27, so 12 is the
    default (saturation); Kerker ON beats OFF (44 vs 58); the Coulomb
    metric option does not converge this system and stays non-default."""

    def __init__(self, g2: torch.Tensor, history: int = 12, w0: float = 0.01,
                 metric_w: torch.Tensor | None = None, **kw) -> None:
        # metric_w: per-component inner-product weights (QE rho_ddot uses
        # the Coulomb metric; None → plain l2)
        super().__init__(g2, history=history, **kw)
        self.w0 = w0
        self.metric_w = metric_w
        self._df: list[torch.Tensor] = []
        self._u: list[torch.Tensor] = []
        self._prev_in: torch.Tensor | None = None
        self._prev_f: torch.Tensor | None = None

    def reset(self) -> None:
        self._df.clear()
        self._u.clear()
        self._prev_in = None
        self._prev_f = None

    def step(self, rho_in: torch.Tensor, rho_out: torch.Tensor) -> torch.Tensor:
        f = rho_out - rho_in
        if self.check_g0 and f[0].abs() >= 1e-8:
            raise ValueError("G=0 residual nonzero")
        if self._prev_in is not None:
            # invariant: _prev_in and _prev_f are always set together
            # (reset()/__init__ clear both, step()'s tail sets both)
            assert self._prev_f is not None
            df = f - self._prev_f
            nrm = float(torch.linalg.norm(df))
            if nrm > 1e-14:
                self._df.append(df / nrm)
                self._u.append((self._damped(df) + (rho_in - self._prev_in))
                               / nrm)
                if len(self._df) > self.history:
                    self._df.pop(0)
                    self._u.pop(0)
        self._prev_in, self._prev_f = rho_in, f

        x_new = rho_in + self._damped(f)
        m = len(self._df)
        if m:
            dfm = torch.stack(self._df)
            if self.metric_w is not None:
                a = torch.einsum("ig,g,jg->ij", dfm.conj(), self.metric_w,
                                 dfm).real
                c = torch.einsum("ig,g,g->i", dfm.conj(), self.metric_w,
                                 f).real
            else:
                a = torch.einsum("ig,jg->ij", dfm.conj(), dfm).real
                c = torch.einsum("ig,g->i", dfm.conj(), f).real
            beta = torch.linalg.solve(
                self.w0**2 * torch.eye(m, dtype=a.dtype, device=a.device)
                + a, c)
            for lam, u in zip(beta.tolist(), self._u, strict=True):
                x_new = x_new - lam * u
        return x_new


class PulayMixer(_DampedMixerBase):
    def __init__(
        self,
        g2: torch.Tensor,
        coeff_cap: float | None = None,  # ℓ₁ bound on DIIS coefficients (see step())
        step_cap: float | None = None,  # ‖Δρ‖ bound in damped-step units (see step())
        adapt_blocks: torch.Tensor | None = None,  # (n,) int block ids → per-block adaptive damp
        adapt_floor: float = 0.05,  # smallest adaptive multiplier
        **kw,
    ) -> None:
        super().__init__(g2, **kw)
        self.coeff_cap = coeff_cap
        self.step_cap = step_cap
        self.adapt_blocks = adapt_blocks
        self.adapt_floor = adapt_floor
        self._block_masks: list[tuple[int, torch.Tensor]] | None = None
        self._block_mult: dict[int, float] | None = None
        self._mult_vec: torch.Tensor | None = None
        self._prev_bnorm: dict[int, float] | None = None
        self._global_mult = 1.0
        self._gnorm_hist: list[float] = []
        if adapt_blocks is not None:
            ids = torch.unique(adapt_blocks).tolist()
            self._block_masks = [(b, adapt_blocks == b) for b in ids]
            self._block_mult = dict.fromkeys(ids, 1.0)
        self._rho_in: list[torch.Tensor] = []
        self._res: list[torch.Tensor] = []

    @override
    def _damped(self, r: torch.Tensor) -> torch.Tensor:
        out = super()._damped(r)
        if self._mult_vec is not None:
            out = out * self._mult_vec
        return out

    @property
    @override
    def block_mult(self) -> dict[int, float] | None:
        return dict(self._block_mult) if self._block_mult else None

    def _adapt(self, res: torch.Tensor) -> None:
        """Per-block gain tracking. A block whose residual grows across
        iterations is locally expansive under the current step (FM metals:
        the magnetization channel near a wrong moment — Stoner curvature);
        cut its damped-step multiplier by the observed growth. Multipliers
        are monotone non-increasing and KEPT across reset(): they encode
        Jacobian gain, not history, and the post-reset plain damped step is
        exactly where an expansive mode diverges. Known limitation, and why
        this is opt-in: transient startup growth (the wild first iterations
        from a SAD start) also fires the reduction, and without recovery
        the run can end over-damped — good enough to hold the FM branch at
        the default alpha, not good enough to replace hand-set damping for
        tight convergence (recovery rules were tried and ride the
        stability boundary instead; a Broyden-class update that learns the
        actual Jacobian is the principled successor)."""
        # invariant: only called from step() under `if self._block_masks is
        # not None`, and _block_masks/_block_mult are always set together
        # (__init__'s adapt_blocks branch sets both, neither is ever cleared)
        assert self._block_masks is not None and self._block_mult is not None
        w = 1.0 / (self.g2 + self.q0**2)
        bnorm: dict[int, float] = {}
        changed = False
        for b, mask in self._block_masks:
            r = res[mask]
            bnorm[b] = float((r.conj() * r * w[mask]).sum().real) ** 0.5
        if self._prev_bnorm is not None:
            total = sum(bnorm.values())
            for b, _ in self._block_masks:
                prev = self._prev_bnorm.get(b, 0.0)
                if bnorm[b] < 1e-12 * max(total, 1e-300) or prev <= 0.0:
                    continue
                g = bnorm[b] / prev
                mult = self._block_mult[b]
                if g > 1.2:
                    self._block_mult[b] = max(mult / min(g, 4.0),
                                              self.adapt_floor)
                    changed = True
                # NO recovery: multipliers are monotone non-increasing.
                # Any recovery rule re-inflates the step once the mode
                # contracts, parks the effective gain at ~1, and the
                # residual plateaus instead of falling (FM Ni: |drho| stuck
                # at 2e-1 with x1.15 immediate recovery, 5e-2 with slow
                # hysteresis recovery). The multiplier encodes the block's
                # Jacobian gain, which does not shrink as the run converges;
                # DIIS supplies the acceleration once the plain step is
                # stable.
        self._prev_bnorm = bnorm
        # plateau-triggered GLOBAL damping: per-block multipliers cannot
        # stabilize an expansive mode that straddles blocks (FM Ni hovers at
        # |drho| ~2e-2 with the m-block floored while rho/becsum run at full
        # step). If the best residual of the last window is no better than
        # 0.8x the window before, the whole map is riding its stability
        # boundary — halve the global step, monotonically, exactly the
        # by-hand mixing_alpha reduction encoded as a rule.
        self._gnorm_hist.append(sum(bnorm.values()))
        if len(self._gnorm_hist) >= 16:
            recent = min(self._gnorm_hist[-8:])
            before = min(self._gnorm_hist[-16:-8])
            if recent > 0.8 * before and self._global_mult > self.adapt_floor:
                self._global_mult = max(0.5 * self._global_mult,
                                        self.adapt_floor)
                self._gnorm_hist.clear()
                changed = True
        if changed or (self._mult_vec is None
                       and (self._global_mult != 1.0
                            or any(m != 1.0
                                   for m in self._block_mult.values()))):
            vec = torch.full_like(self.g2, self._global_mult)
            for b, mask in self._block_masks:
                vec = torch.where(mask, self._global_mult
                                  * self._block_mult[b], vec)
            self._mult_vec = vec

    def _metric(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w = 1.0 / (self.g2 + self.q0**2)
        return (a.conj() * b * w).sum().real

    def reset(self) -> None:
        self._rho_in.clear()
        self._res.clear()
        self._prev_bnorm = None  # block multipliers survive (see _adapt)
        self._gnorm_hist.clear()

    def step(self, rho_in: torch.Tensor, rho_out: torch.Tensor) -> torch.Tensor:
        """Next ρ_in(G) from the current (ρ_in, ρ_out) pair."""
        res = rho_out - rho_in
        if self.check_g0 and res[0].abs() >= 1e-8:
            raise ValueError("G=0 residual nonzero — density not normalized")
        if self._block_masks is not None:
            self._adapt(res)

        self._rho_in.append(rho_in)
        self._res.append(res)
        if len(self._res) > self.history:
            self._rho_in.pop(0)
            self._res.pop(0)

        # stale-history filter: entries whose residual is far larger than the
        # current one carry curvature information from a region the iteration
        # has left; keeping them distorts the DIIS extrapolation near
        # convergence (the NiO+U tail was dominated by this)
        r_now = self._metric(res, res)
        while len(self._res) > 2:
            r_old = self._metric(self._res[0], self._res[0])
            if r_old > 1e8 * r_now:
                self._rho_in.pop(0)
                self._res.pop(0)
            else:
                break

        while True:
            m = len(self._res)
            if m == 1:
                return rho_in + self._damped(res)

            # bordered system: [B 1; 1ᵀ 0][c; λ] = [0; 1], B_ij = <R_i, R_j>.
            # Solve in the diagonal-normalized basis (B̃ = D⁻¹BD⁻¹, D = √diag B)
            # so the Tikhonov term is scale-invariant: residual norms span many
            # orders across the history, and a regularizer scaled to the raw
            # matrix would swamp the newest (smallest-residual) entries.
            b0 = torch.zeros((m, m), dtype=torch.float64, device=rho_in.device)
            for i in range(m):
                for j in range(i, m):
                    bij = self._metric(self._res[i], self._res[j])
                    b0[i, j] = b0[j, i] = bij
            d = torch.sqrt(b0.diagonal().clamp_min(1e-300))
            bn = b0 / d[:, None] / d[None, :]
            bn = bn + 1e-10 * torch.eye(m, dtype=torch.float64, device=rho_in.device)

            b = torch.zeros((m + 1, m + 1), dtype=torch.float64, device=rho_in.device)
            b[:m, :m] = bn
            b[:m, m] = 1.0 / d
            b[m, :m] = 1.0 / d
            rhs = torch.zeros(m + 1, dtype=torch.float64, device=rho_in.device)
            rhs[m] = 1.0

            # residual ill-conditioning → drop the OLDEST entry and retry
            # (a full reset discards curvature the next steps need). A flag
            # (not raise/except) so a genuine linalg.solve failure below is
            # not swallowed as ill-conditioning.
            cond = torch.linalg.cond(bn)
            if not torch.isfinite(cond) or cond > 1e14:
                logger.debug(
                    "Pulay: cond(B)=%.3e > 1e14, dropping oldest of %d history "
                    "entries and retrying", float(cond), m)
                self._rho_in.pop(0)
                self._res.pop(0)
                continue
            coeff = torch.linalg.solve(b, rhs)[:m] / d
            break

        # Σc=1 bounds nothing: near-parallel early residuals admit large ±c
        # whose ρ_opt extrapolates far outside the region where the SCF map
        # is linear (Ni₂ spin: first DIIS step sent |R| 37→615). Blend toward
        # the pure newest-point step (c = e_last, still Σc=1) until ‖c‖₁ fits.
        # OPT-IN (default None): stiff metals NEED >cap quasi-Newton steps —
        # capped, FM Ni's on-site mode (damped gain > 1) limit-cycles forever.
        cnorm = coeff.abs().sum()
        if self.coeff_cap is not None and cnorm > self.coeff_cap:
            theta = (self.coeff_cap - 1.0) / (cnorm - 1.0)
            logger.debug(
                "Pulay coeff-cap: |c|_1=%.3g > cap=%.3g, blending toward the "
                "newest-point step (theta=%.3g)", float(cnorm), self.coeff_cap,
                float(theta))
            e_last = torch.zeros_like(coeff)
            e_last[-1] = 1.0
            coeff = theta * coeff + (1.0 - theta) * e_last

        rho_opt = sum(c * r for c, r in zip(coeff, self._rho_in, strict=True))
        res_opt = sum(c * r for c, r in zip(coeff, self._res, strict=True))
        rho_new = rho_opt + self._damped(res_opt)

        # step-norm trust region: the coefficient cap bounds the extrapolation
        # weights, but ‖Δρ‖ can still leave the linear-response region when
        # the residual itself is large (FM metals: the magnetization channel
        # is locally EXPANSIVE near a wrong moment — Stoner curvature — so
        # early residuals grow until DIIS learns the Jacobian; unbounded steps
        # blow up first). Scale the step to step_cap × the damped-step norm —
        # the scale empirically inside the linear region.
        if self.step_cap is not None:
            step = rho_new - rho_in
            lim = self.step_cap * torch.linalg.norm(self._damped(res))
            snorm = torch.linalg.norm(step)
            if snorm > lim:
                logger.debug(
                    "Pulay step-cap: |step|=%.3e > trust limit=%.3e, scaling "
                    "into the linear-response region", float(snorm), float(lim))
                rho_new = rho_in + step * (lim / snorm)
        return rho_new


class SoftModeDeflatingMixer:
    """Deflate the constant-µ soft mode (the total-charge / G=0 channel) out of
    the density mixer and treat it with a dedicated, stabilized charge update.

    Under a constant-potential (grand-canonical) SCF the electron count N floats
    to hold the Fermi level µ, so the uniform-charge component ρ(G=0) = N/vol is a
    live mixing variable (``float_charge`` disables the usual G=0-conservation
    pin). On a high-DOS transition metal that channel is a SOFT MODE of the SCF
    charge response (1 − χ₀K): its loop gain a = dρ_out(0)/dρ_in(0) ≈
    −(dN/dµ)·(dV/dN) is large and negative — adding electrons raises the potential
    at fixed µ, which lowers N_out — so the plain fixed-point map 1 + α(a − 1) has
    magnitude > 1 and any linear/DIIS mixing of it oscillates or diverges (Na
    crawls ~130 iterations at moderate |µ − µ0|; Pt/Ni fail outright). The
    mixer-wide α that converges the G ≠ 0 modes is simply too large for this one
    stiff, negative-gain coordinate. Broadening the smearing only shrinks |a|; the
    principled cure is to DEFLATE the mode and treat it on its own terms.

    This is the ground-state analogue of the χ₀-response deflation in
    ``solvers.deflation`` (#263): that path extracts the soft subspace of the
    explicit ``M = K·χ`` matvec and removes it with an exact coarse solve while
    Anderson smooths the complement. Here the SCF Jacobian is available only
    implicitly through the (ρ_in, ρ_out) stream and the soft subspace is the
    physically known rank-1 uniform-charge direction, so per iteration we:

    * hide the soft channel from the wrapped mixer (zero its G=0 residual) so its
      DIIS / Broyden history and metric stay in the well-conditioned complement
      (G ≠ 0) — which then converges as the neutral fixed-N run does, instead of
      being polluted by the Kerker-over-weighted (1/q0²) G=0 residual; and
    * update the scalar total-charge coordinate ρ0 = ρ(G=0) on its own. When a
      clean secant slope j0 = dr0/dρ0 in the physical screening regime (j0 < 0,
      the negative-gain channel) is available, take the (under-relaxed) Newton
      step ρ0 − r0/j0 — the exact 1-D coarse solve, which converges the stiff
      channel in a few steps regardless of |a|. Otherwise fall back to a SMALL
      fixed damping α0·r0: with |a| large and negative, α0 ≲ 1/|a| makes the
      channel a contraction on its own even when the mixer-wide α would not, so
      the charge stays bounded and creeps to the fixed point. A noisy secant (the
      G ≠ 0 density is still co-varying early on, contaminating dr0) is REJECTED
      rather than trusted — an out-of-regime j0 is exactly what makes a naive
      Newton bolt — and a fixed relative trust cap bounds any single step.

    The wrapper forwards every other attribute/method to the inner mixer, so the
    driver drives it identically (``step``/``extra_precond``/``reset``/…)."""

    def __init__(self, inner: PulayMixer | BroydenMixer | JohnsonMixer,
                 g2: torch.Tensor, alpha0: float,
                 *, damp: float = 0.1, relax: float = 0.5, j0_min: float = 0.3,
                 max_frac: float = 0.05) -> None:
        # inner: the wrapped mixer (Pulay/Broyden/Johnson) handling the complement.
        # g2: per-component |G|² of the mixing vector; the total-charge G=0
        # coordinate is its (first, min-g2) zero — argmin picks it robustly.
        # alpha0: the mixer-wide α (kept for reference / provenance; the charge
        # channel uses its OWN small `damp`, since alpha0 is what destabilizes it).
        self._inner = inner
        self._i0 = int(torch.argmin(g2))
        self._alpha0 = float(alpha0)
        self._damp = float(damp)      # small fixed step for the stiff G=0 channel
        self._relax = float(relax)    # under-relaxation of the Newton step
        self._j0_min = float(j0_min)  # require j0 < −j0_min (physical screening regime)
        self._max_frac = float(max_frac)  # per-step cap as a fraction of |ρ0|
        self._p0_prev: torch.Tensor | None = None
        self._r0_prev: torch.Tensor | None = None

    def __getattr__(self, name: str):
        # Delegate anything not defined here (q0, precond_op, block_mult, …) to the
        # wrapped mixer. Only fires for names missing on self; _inner is set first
        # in __init__, so this never recurses on it.
        return getattr(self._inner, name)

    @property
    def extra_precond(self):
        return self._inner.extra_precond

    @extra_precond.setter
    def extra_precond(self, value) -> None:
        self._inner.extra_precond = value

    def reset(self) -> None:
        self._inner.reset()
        self._p0_prev = None
        self._r0_prev = None

    def _capped(self, p0: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Move ρ0 toward ``target`` but no further than max_frac·|ρ0| — a fixed
        trust region (no dependence on the last step, so it cannot self-amplify)."""
        step = target - p0
        lim = self._max_frac * p0.abs().clamp_min(1e-12)
        mag = step.abs()
        if float(mag) > float(lim):
            step = step * (lim / mag)
        return p0 + step

    def _charge_update(self, p0: torch.Tensor, r0: torch.Tensor) -> torch.Tensor:
        """Stabilized update for the scalar total-charge coordinate: an
        under-relaxed Newton step when a clean in-regime secant slope is
        available, else a small fixed-damping step."""
        damped = p0 + self._damp * r0
        if self._p0_prev is None:
            return self._capped(p0, damped)
        # _p0_prev and _r0_prev are set together (step()'s tail, cleared jointly
        # in __init__/reset), so _r0_prev is non-None here too.
        assert self._r0_prev is not None
        dp = p0 - self._p0_prev
        dr = r0 - self._r0_prev
        if float(dp.abs()) < 1e-12 or float(dr.abs()) < 1e-16:
            return self._capped(p0, damped)
        j0 = dr / dp  # secant slope dr0/dρ0 ≈ a − 1 (physically < −1: screening)
        # Trust the Newton step only when j0 is clearly in the negative-gain
        # screening regime; a near-zero or positive j0 is secant noise from the
        # still-converging G ≠ 0 density and would send Newton the wrong way.
        if not bool(torch.isfinite(j0)) or float(j0) > -self._j0_min:
            return self._capped(p0, damped)
        newton = p0 - self._relax * r0 / j0
        return self._capped(p0, newton)

    def step(self, rho_in: torch.Tensor, rho_out: torch.Tensor) -> torch.Tensor:
        i0 = self._i0
        # the uniform charge is real (∫ρ = N); carry only the real part.
        p0 = rho_in[i0].real.clone()
        r0 = (rho_out[i0] - rho_in[i0]).real.clone()
        # hide the soft channel from the inner mixer: a zero G=0 residual keeps it
        # out of the DIIS metric (where the Kerker weight 1/q0² would over-weight
        # it) and out of the Broyden/Anderson history.
        inner_in = rho_in.clone()
        inner_out = rho_out.clone()
        inner_out[i0] = inner_in[i0]
        rho_new = self._inner.step(inner_in, inner_out).clone()
        # overwrite the soft coordinate with the deflated charge update.
        rho_new[i0] = self._charge_update(p0, r0).to(rho_new.dtype)
        self._p0_prev, self._r0_prev = p0, r0
        return rho_new
