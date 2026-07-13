"""Density mixing: linear, Kerker-preconditioned, Pulay/Anderson (Layer B).

Mixing operates on ρ(G) over the density sphere (complex vectors). The G=0
component is pinned (both ρ_in and ρ_out integrate to N_e, so the residual
there is zero by construction — asserted).

Kerker: R̃(G) = R(G)·G²/(G² + q0²) suppresses long-wavelength charge
sloshing in metals; q0 default 1.1 Å⁻¹ (~0.58 bohr⁻¹). Off for insulators
(it slows their convergence).

Pulay: minimize ‖Σ c_i R_i‖ with Σ c_i = 1 in the Kerker-weighted inner
product ⟨R, R'⟩ = Σ_G Re[R*R']/(G² + q0²) (QE-style metric emphasizing
long-range components), via a bordered linear system; restart on
ill-conditioning.
"""

from __future__ import annotations

import torch


class BroydenMixer:
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

    def __init__(
        self,
        g2: torch.Tensor,
        alpha: float = 0.7,
        history: int = 8,
        kerker: bool = False,
        q0: float = 1.1,
        check_g0: bool = True,
        kerker_mask=None,
        step_scale=None,
    ):
        self.g2 = g2
        self.alpha = alpha
        self.history = history
        self.kerker = kerker
        self.q0 = q0
        self.check_g0 = check_g0
        self.kerker_mask = kerker_mask
        self.step_scale = step_scale
        self.extra_precond = None  # per-iteration hook (Stoner spin precond)
        self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._prev_in = None
        self._prev_res = None

    def _damped(self, r: torch.Tensor) -> torch.Tensor:
        if self.extra_precond is not None:
            r = self.extra_precond(r)
        if not self.kerker:
            out = self.alpha * r
        else:
            fac = self.g2 / (self.g2 + self.q0**2)
            if self.kerker_mask is not None:
                fac = torch.where(self.kerker_mask, fac, torch.ones_like(fac))
            out = self.alpha * fac * r
        if self.step_scale is not None:
            out = out * self.step_scale
        return out

    def _apply_b(self, v, us, ys):
        """B v with B = −αP + Σ u_i ⟨y_i|·⟩ (rank-one secant corrections)."""
        out = -self._damped(v)
        for u, y in zip(us, ys, strict=True):
            out = out + u * (y.conj() @ v)
        return out

    def reset(self):
        self._pairs.clear()
        self._prev_in = None
        self._prev_res = None

    def step(self, rho_in: torch.Tensor, rho_out: torch.Tensor) -> torch.Tensor:
        res = rho_out - rho_in
        if self.check_g0:
            assert res[0].abs() < 1e-8, "G=0 residual nonzero"
        if self._prev_in is not None:
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


class JohnsonMixer:
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
    corrections."""

    def __init__(
        self,
        g2: torch.Tensor,
        alpha: float = 0.7,
        history: int = 8,
        kerker: bool = False,
        q0: float = 1.1,
        check_g0: bool = True,
        kerker_mask=None,
        step_scale=None,
        w0: float = 0.01,
    ):
        self.g2 = g2
        self.alpha = alpha
        self.history = history
        self.kerker = kerker
        self.q0 = q0
        self.check_g0 = check_g0
        self.kerker_mask = kerker_mask
        self.step_scale = step_scale
        self.w0 = w0
        self.extra_precond = None
        self._df: list[torch.Tensor] = []
        self._u: list[torch.Tensor] = []
        self._prev_in = None
        self._prev_f = None

    _damped = BroydenMixer._damped

    def reset(self):
        self._df.clear()
        self._u.clear()
        self._prev_in = None
        self._prev_f = None

    def step(self, rho_in: torch.Tensor, rho_out: torch.Tensor) -> torch.Tensor:
        f = rho_out - rho_in
        if self.check_g0:
            assert f[0].abs() < 1e-8, "G=0 residual nonzero"
        if self._prev_in is not None:
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
            a = torch.einsum("ig,jg->ij", dfm.conj(), dfm).real
            c = torch.einsum("ig,g->i", dfm.conj(), f).real
            beta = torch.linalg.solve(
                self.w0**2 * torch.eye(m, dtype=a.dtype, device=a.device)
                + a, c)
            for lam, u in zip(beta.tolist(), self._u, strict=True):
                x_new = x_new - lam * u
        return x_new


class PulayMixer:
    def __init__(
        self,
        g2: torch.Tensor,  # (nG,) |G|² over the density sphere, G=0 first entry allowed
        alpha: float = 0.7,
        history: int = 8,
        kerker: bool = False,
        q0: float = 1.1,
        check_g0: bool = True,
        kerker_mask=None,  # per-component bool; None → kerker applies to all
        step_scale=None,  # per-component multiplier on the damped step (None → 1)
        coeff_cap: float | None = None,  # ℓ₁ bound on DIIS coefficients (see step())
        step_cap: float | None = None,  # ‖Δρ‖ bound in damped-step units (see step())
        adapt_blocks=None,  # (n,) int block ids → per-block adaptive damping
        adapt_floor: float = 0.05,  # smallest adaptive multiplier
    ):
        self.check_g0 = check_g0
        self.kerker_mask = kerker_mask
        self.step_scale = step_scale
        self.g2 = g2
        self.alpha = alpha
        self.history = history
        self.kerker = kerker
        self.q0 = q0
        self.coeff_cap = coeff_cap
        self.step_cap = step_cap
        self.adapt_blocks = adapt_blocks
        self.adapt_floor = adapt_floor
        self._block_masks = None
        self._block_mult = None
        self._mult_vec = None
        self._prev_bnorm = None
        self._global_mult = 1.0
        self._gnorm_hist: list[float] = []
        if adapt_blocks is not None:
            ids = torch.unique(adapt_blocks).tolist()
            self._block_masks = [(b, adapt_blocks == b) for b in ids]
            self._block_mult = {b: 1.0 for b in ids}
        self.extra_precond = None  # per-iteration hook (Stoner spin precond)
        self._rho_in: list[torch.Tensor] = []
        self._res: list[torch.Tensor] = []

    def _precondition(self, r: torch.Tensor) -> torch.Tensor:
        if self.extra_precond is not None:
            r = self.extra_precond(r)
        if not self.kerker:
            out = self.alpha * r
        else:
            fac = self.g2 / (self.g2 + self.q0**2)
            if self.kerker_mask is not None:
                fac = torch.where(self.kerker_mask, fac, torch.ones_like(fac))
            out = self.alpha * fac * r
        if self.step_scale is not None:
            out = out * self.step_scale
        if self._mult_vec is not None:
            out = out * self._mult_vec
        return out

    def _adapt(self, res: torch.Tensor):
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
        w = 1.0 / (self.g2 + self.q0**2)
        bnorm, changed = {}, False
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

    def reset(self):
        self._rho_in.clear()
        self._res.clear()
        self._prev_bnorm = None  # block multipliers survive (see _adapt)
        self._gnorm_hist.clear()

    def step(self, rho_in: torch.Tensor, rho_out: torch.Tensor) -> torch.Tensor:
        """Next ρ_in(G) from the current (ρ_in, ρ_out) pair."""
        res = rho_out - rho_in
        if self.check_g0:
            assert res[0].abs() < 1e-8, "G=0 residual nonzero — density not normalized"
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
                return rho_in + self._precondition(res)

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
            # (a full reset discards curvature the next steps need)
            try:
                cond = torch.linalg.cond(bn)
                if not torch.isfinite(cond) or cond > 1e14:
                    raise RuntimeError
                coeff = torch.linalg.solve(b, rhs)[:m] / d
            except RuntimeError:
                self._rho_in.pop(0)
                self._res.pop(0)
                continue
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
            e_last = torch.zeros_like(coeff)
            e_last[-1] = 1.0
            coeff = theta * coeff + (1.0 - theta) * e_last

        rho_opt = sum(c * r for c, r in zip(coeff, self._rho_in, strict=True))
        res_opt = sum(c * r for c, r in zip(coeff, self._res, strict=True))
        rho_new = rho_opt + self._precondition(res_opt)

        # step-norm trust region: the coefficient cap bounds the extrapolation
        # weights, but ‖Δρ‖ can still leave the linear-response region when
        # the residual itself is large (FM metals: the magnetization channel
        # is locally EXPANSIVE near a wrong moment — Stoner curvature — so
        # early residuals grow until DIIS learns the Jacobian; unbounded steps
        # blow up first). Scale the step to step_cap × the damped-step norm —
        # the scale empirically inside the linear region.
        if self.step_cap is not None:
            step = rho_new - rho_in
            lim = self.step_cap * torch.linalg.norm(self._precondition(res))
            snorm = torch.linalg.norm(step)
            if snorm > lim:
                rho_new = rho_in + step * (lim / snorm)
        return rho_new
