"""Learned radial density preconditioner (multi-pole Kerker).

The bare Kerker filter R̃(G) = R(G)·G²/(G²+q0²) is the long-wavelength, single-
pole approximation to the exact linear-response preconditioner P = ε⁻¹, with
ε(G) = 1 − v_c(G)·χ₀(G) the (shell-averaged) dielectric function. One pole q0 is
the right shape when the SCF residual map is diagonal in G with a single response
length, i.e. a homogeneous metal; it is the wrong shape when the response carries
more than one length scale (two-component intermetallics, semicore + valence
screening, inhomogeneous cells whose G-space response profile is not a single
Lorentzian). `docs/manual/wisdom.md` records the design stance this module acts
on: prefer a preconditioner (an operator on the residual) to step-size control,
and the principled operator is the χ₀-diagonal one Kerker crudely approximates.

This module replaces the single pole with a learned sum of poles,

    f_θ(G²) = Σ_i w_i · G²/(G² + q_i²),   w_i ≥ 0,  q_i² ≥ 0,

applied per density-sphere component exactly where the mixer applies Kerker (as
`mixer.precond_op`; the driver multiplies by the mixing step α). Two properties
carry over from Kerker for free and are load-bearing, not incidental:

- f_θ(0) = 0, so the pinned G=0 charge is untouched (every term has a G²
  numerator). A learned preconditioner cannot leak charge into the conserved
  mode by construction.
- The fixed point is unchanged. A preconditioner reshapes the *path* to self-
  consistency, never the solution; convergence is still gated on the true
  residual (`scf` checks |ρ_out − ρ_in|). So the accuracy risk of a bad learned
  filter is zero — only the iteration count moves.

K=1 with w=1, q1=q0 reproduces bare Kerker to round-off, so the learned filter is
a strict generalization and the single-pole result is always in its hypothesis
class.

Fitting (`fit_multipole`) is where the differentiable solver earns its keep. The
error of preconditioned linear mixing evolves, component-wise in the diagonal-in-G
model, as

    e_{n+1}(G) = [1 − α·f_θ(G²)·d(G)]·e_n(G),   d(G) = 1 − j(G),

with j(G) the SCF residual map's diagonal gain and d(G) its response denominator
(d → 1/f is the one-step-optimal filter). `fit_multipole` unrolls this recurrence
for a fixed number of steps and backpropagates ‖e_N‖ through it to the pole
weights and positions — training the preconditioner against the solver's own
linearized response, which no non-differentiable DFT code can do without finite
differences. `response_from_residuals` estimates d(G) per |G|-shell from a short
plain-mixing SCF's residual history (res_{n+1}/res_n = 1 − α·d in the same model),
so the whole loop — probe, fit, deploy — runs on real solver output.
"""

from __future__ import annotations

from typing import Any

import torch

from gradwave.dtypes import RDTYPE


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) = y, for seeding a parameter at a target value."""
    import math
    return math.log(math.expm1(y)) if y < 20 else y


class MultipoleKerkerPrecond:
    """A learned radial density preconditioner f_θ(G²)·R over the density sphere.

    Construct with :meth:`kerker` (single-pole, matches the bare filter) or
    :meth:`init_poles` (K poles, log-spaced seed), fit the poles with
    :func:`fit_multipole`, then hand the instance to ``scf(..., precond_op=P)`` or
    assign it to ``mixer.precond_op``. Callable: ``P(r)`` returns f_θ(g²)·r on the
    same layout as ``r``. The parameters live in ``P.params`` (a list of leaf
    tensors) for an optimizer; call :meth:`detach_` before deploying in a solve so
    no autograd graph is retained through the SCF."""

    def __init__(self, g2: torch.Tensor, w_raw: torch.Tensor, logq2: torch.Tensor,
                 c_raw: torch.Tensor | None = None) -> None:
        # g2: (n,) |G|² per component the filter acts on [Å⁻²]; buffer, no grad.
        self.g2 = g2.detach()
        self.w_raw = w_raw          # (K,) softplus⁻¹ weights (leaf, may need grad)
        self.logq2 = logq2          # (K,) log pole positions log(q_i²) (leaf)
        # optional G=0-alive constant w0 = sigmoid(c_raw) ∈ (0,1). None → f(0)=0,
        # the charge form (pinned total charge untouched). A nonzero w0 is for the
        # MAGNETIZATION channel: the uniform moment must move (so f(0)≠0) but the
        # near-critical Stoner mode must be damped (so w0<1) — see fit_multipole.
        self.c_raw = c_raw

    # -- construction ---------------------------------------------------------
    @classmethod
    def kerker(cls, g2: torch.Tensor, q0: float = 1.1) -> MultipoleKerkerPrecond:
        """Single pole reproducing the bare Kerker filter G²/(G²+q0²)."""
        import math
        w_raw = torch.tensor([_inv_softplus(1.0)], dtype=RDTYPE, device=g2.device)
        logq2 = torch.tensor([2.0 * math.log(float(q0))],
                             dtype=RDTYPE, device=g2.device)
        return cls(g2, w_raw, logq2)

    @classmethod
    def init_poles(cls, g2: torch.Tensor, n_poles: int = 3,
                   q_min: float = 0.3, q_max: float = 3.0,
                   requires_grad: bool = True,
                   const: bool = False, const_init: float = 0.5
                   ) -> MultipoleKerkerPrecond:
        """K poles log-spaced over [q_min, q_max] Å⁻¹, unit total weight seed.
        ``const=True`` adds the learnable G=0-alive term w0 seeded at ``const_init``
        (the magnetization-channel form)."""
        import math
        q = torch.logspace(float(torch.log10(torch.tensor(q_min))),
                           float(torch.log10(torch.tensor(q_max))),
                           n_poles, dtype=RDTYPE, device=g2.device)
        logq2 = (2.0 * torch.log(q)).clone()
        w_raw = torch.full((n_poles,), _inv_softplus(1.0 / n_poles),
                           dtype=RDTYPE, device=g2.device)
        c_raw = None
        if const:
            logit = math.log(const_init / (1.0 - const_init))
            c_raw = torch.tensor(logit, dtype=RDTYPE, device=g2.device)
            c_raw.requires_grad_(requires_grad)
        w_raw.requires_grad_(requires_grad)
        logq2.requires_grad_(requires_grad)
        return cls(g2, w_raw, logq2, c_raw)

    # -- parametrization ------------------------------------------------------
    @property
    def params(self) -> list[torch.Tensor]:
        p = [self.w_raw, self.logq2]
        if self.c_raw is not None:
            p.append(self.c_raw)
        return p

    def weights(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.w_raw)

    def q2(self) -> torch.Tensor:
        return torch.exp(self.logq2)

    def const_val(self) -> torch.Tensor:
        """The G=0-alive constant w0 ∈ (0,1), or 0 when disabled."""
        if self.c_raw is None:
            return torch.zeros((), dtype=RDTYPE, device=self.g2.device)
        return torch.sigmoid(self.c_raw)

    def filter_vals(self, g2: torch.Tensor | None = None) -> torch.Tensor:
        """f_θ(g²) per component. Without a const term g²=0 → 0 exactly (pinned
        charge preserved); with it, f(0) = w0 ∈ (0,1) (magnetization channel)."""
        g2 = self.g2 if g2 is None else g2
        w, q2 = self.weights(), self.q2()
        gg = g2[:, None]
        base = (w * gg / (gg + q2)).sum(dim=-1)          # Σ_i w_i g²/(g²+q_i²)
        if self.c_raw is not None:
            base = base + self.const_val()
        return base

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        """P·r = f_θ(g²)·r on the density sphere (matches Kerker's fac·r)."""
        fac = self.filter_vals().to(r.real.dtype if r.is_complex() else r.dtype)
        return fac * r

    def rebind(self, g2: torch.Tensor) -> MultipoleKerkerPrecond:
        """Same learned poles on a new |G|² grid (fit on shell centers, deploy on
        the per-component density sphere). The poles are analytic in G², so this
        is exact, not a resampling."""
        return MultipoleKerkerPrecond(g2, self.w_raw, self.logq2, self.c_raw)

    def detach_(self) -> MultipoleKerkerPrecond:
        """Drop autograd tracking on the poles for use inside a solve."""
        self.w_raw = self.w_raw.detach()
        self.logq2 = self.logq2.detach()
        if self.c_raw is not None:
            self.c_raw = self.c_raw.detach()
        return self

    def summary(self) -> str:
        w = self.weights().detach().tolist()
        q = self.q2().detach().sqrt().tolist()
        poles = ", ".join(f"w={wi:.3f}@q={qi:.3f}Å⁻¹"
                          for wi, qi in zip(w, q, strict=True))
        w0 = "" if self.c_raw is None else f"w0={float(self.const_val()):.3f} + "
        return f"MultipoleKerkerPrecond[{w0}{poles}]"


class BlockPrecond:
    """Composite preconditioner over a packed multi-block mixing vector: apply a
    separate operator to each contiguous block.

    Used on the collinear nspin=2 (total, magnetization) vector — bare Kerker on
    the charge-total block, a learned G=0-alive filter on the magnetization block —
    so the charge channel is untouched while the spin channel gets its own operator.
    ``acts_on = "grid"`` tells the driver to slice the whole grid part (every nspin
    block) to this op rather than the total block alone."""

    acts_on = "grid"

    def __init__(self, blocks) -> None:
        # blocks: list of (n_components, callable | None). None → identity (plain
        # damping) on that block. Segment lengths must sum to the sliced vector.
        self.blocks = list(blocks)

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        out, i = [], 0
        for n, op in self.blocks:
            seg = r[i:i + n]
            out.append(op(seg) if op is not None else seg)
            i += n
        return torch.cat(out) if len(out) > 1 else out[0]


def spectral_radius(f_vals: torch.Tensor, d: torch.Tensor,
                    alpha: float) -> torch.Tensor:
    """Worst-component amplification max_G |1 − α·f(G²)·d(G)| of the diagonal
    preconditioned-mixing iteration matrix. The asymptotic convergence rate:
    residual falls by this factor per SCF step, so iterations-to-tol scale as
    log(tol)/log(ρ). Reported by the benchmarks alongside real n_iter."""
    return (1.0 - alpha * f_vals * d).abs().max()


def _diis_unroll_logres(f: torch.Tensor, d: torch.Tensor, metric: torch.Tensor,
                        alpha: float, n_unroll: int, history: int) -> torch.Tensor:
    """log ‖res_N‖ after unrolling Pulay DIIS in the diagonal model, differentiable
    in the filter values f.

    In the diagonal linear model res_i = d⊙e_i, and a Pulay step is
    e_new = (1 − α f d) ⊙ ē with ē = Σ c_i e_i the DIIS extrapolation whose
    coefficients minimize ‖Σ c_i res_i‖²_metric under Σc_i = 1 (the same bordered,
    diagonal-normalized, Tikhonov-regularized solve mixing.PulayMixer runs). The
    coefficients do not depend on f, but the e-history they extrapolate does, so f
    is trained to accelerate the modes DIIS's finite history does NOT already
    kill — not to duplicate its low-G work."""
    S = d.shape[0]
    amp = 1.0 - alpha * f * d                      # (S,) preconditioned step factor
    wq = metric * d * d                            # residual metric weight ⊙ d²
    E = [torch.ones(S, dtype=d.dtype, device=d.device)]
    eye = torch.eye
    for _ in range(n_unroll):
        m = len(E)
        Emat = torch.stack(E)                      # (m, S)
        if m == 1:
            ebar = E[0]
        else:
            b = (Emat * wq) @ Emat.T               # B_ij = ⟨res_i, res_j⟩_metric
            diag = b.diagonal().clamp_min(1e-300).sqrt()
            bn = b / diag[:, None] / diag[None, :]
            bn = bn + 1e-10 * eye(m, dtype=d.dtype, device=d.device)
            bordered = torch.zeros(m + 1, m + 1, dtype=d.dtype, device=d.device)
            bordered[:m, :m] = bn
            bordered[:m, m] = 1.0 / diag
            bordered[m, :m] = 1.0 / diag
            rhs = torch.zeros(m + 1, dtype=d.dtype, device=d.device)
            rhs[m] = 1.0
            c = torch.linalg.solve(bordered, rhs)[:m] / diag
            ebar = (c[:, None] * Emat).sum(0)
        E.append(amp * ebar)
        if len(E) > history:
            E.pop(0)
    eN = E[-1]
    return 0.5 * torch.log((wq * eN * eN).sum().clamp_min(1e-300))


def fit_multipole(g2_shell: torch.Tensor, d_shell: torch.Tensor, *,
                  n_poles: int = 3, alpha: float = 0.7, n_unroll: int = 40,
                  steps: int = 400, lr: float = 0.05,
                  q_min: float = 0.3, q_max: float = 3.0,
                  seed_q: list[float] | None = None,
                  q_floor: float | None = None, q_ceil: float | None = None,
                  weight: torch.Tensor | None = None,
                  mixer: str = "plain", history: int = 8, q0: float = 1.1,
                  verbose: bool = False) -> tuple[MultipoleKerkerPrecond, dict[str, Any]]:
    """Fit multi-pole Kerker poles to a per-shell response denominator d(G).

    Differentiates the unrolled mixing residual through the solver's linearized
    response to the pole weights and positions. Returns the fitted preconditioner
    (built on ``g2_shell``; rebind to the density sphere for deployment) and a
    history dict with the loss curve and initial/final spectral radius.

    Args:
        g2_shell: (S,) representative |G|² per shell [Å⁻²].
        d_shell:  (S,) response denominator d = 1 − j per shell (real).
        seed_q:   optional explicit initial pole positions [Å⁻¹] (length overrides
                  ``n_poles``); the multi-seed driver passes a deterministic set of
                  these. ``None`` seeds ``n_poles`` log-spaced over [q_min, q_max].
        q_floor:  optional hard lower bound [Å⁻¹] on every pole position, projected
                  after each optimizer step. Set by the robust driver to the lowest
                  |G| shell with trustworthy probe data so a pole cannot chase
                  lowest-shell noise below where d(G) is measured.
        q_ceil:   optional hard upper bound [Å⁻¹] on every pole position.
        weight:   (S,) per-shell importance (shell multiplicity or probe-quality
                  confidence); defaults to uniform. In ``mixer="diis"`` it also
                  enters the Pulay metric. A zero-weight shell is excluded.
        mixer:    ``"plain"`` unrolls damped linear mixing e_{n+1}=(1−αfd)e_n and
                  minimizes the worst-shell rate (a smooth max over shells). This
                  is the right target when the deployment mixer is plain damping.
                  ``"diis"`` unrolls Pulay DIIS (history ``history``, Kerker metric
                  wavevector ``q0``) so the filter is trained to complement the
                  DIIS the real ``scf`` runs, not to duplicate its low-G work.
    """
    import math
    g2_shell = g2_shell.to(RDTYPE)
    d_shell = d_shell.to(RDTYPE)
    raw_w = (torch.ones_like(g2_shell) if weight is None
             else weight.to(RDTYPE)).clamp_min(0.0)
    w_shell = raw_w / raw_w.sum().clamp_min(1e-30)          # loss weight (plain)
    metric = raw_w / (g2_shell + q0**2)                     # Pulay metric (diis)
    if mixer not in ("plain", "diis"):
        raise ValueError("mixer must be 'plain' or 'diis'")

    if seed_q is not None:
        q = torch.tensor(list(seed_q), dtype=RDTYPE, device=g2_shell.device)
        logq2 = (2.0 * torch.log(q)).clone().requires_grad_(True)
        w_raw = torch.full((len(seed_q),), _inv_softplus(1.0 / len(seed_q)),
                           dtype=RDTYPE, device=g2_shell.device
                           ).requires_grad_(True)
        P = MultipoleKerkerPrecond(g2_shell, w_raw, logq2)
    else:
        P = MultipoleKerkerPrecond.init_poles(g2_shell, n_poles, q_min, q_max,
                                              requires_grad=True)
    lo = 2.0 * math.log(q_floor) if q_floor is not None else None
    hi = 2.0 * math.log(q_ceil) if q_ceil is not None else None

    def _project() -> None:
        if lo is None and hi is None:
            return
        with torch.no_grad():
            P.logq2.clamp_(min=lo if lo is not None else float("-inf"),
                           max=hi if hi is not None else float("inf"))

    _project()  # keep the seed inside the bounds too
    opt = torch.optim.Adam(P.params, lr=lr)
    e0 = torch.ones_like(g2_shell)
    rho_init = float(spectral_radius(P.filter_vals().detach(), d_shell, alpha))
    loss_hist: list[float] = []

    for it in range(steps):
        opt.zero_grad()
        f = P.filter_vals()
        if mixer == "diis":
            loss = _diis_unroll_logres(f, d_shell, metric, alpha, n_unroll,
                                       history)
        else:
            amp = (1.0 - alpha * f * d_shell)      # (S,) per-shell error factor
            # e_N = amp**N ⊙ e0; log|amp| avoids under/overflow and the weighted
            # logsumexp is a smooth surrogate for the worst-shell rate the
            # spectral radius takes as a hard max.
            log_amp = torch.log(amp.abs().clamp_min(1e-12))
            log_eN = n_unroll * log_amp + torch.log(e0)
            loss = torch.logsumexp(log_eN + torch.log(w_shell.clamp_min(1e-30)),
                                   dim=0)
        loss.backward()
        opt.step()
        _project()
        loss_hist.append(float(loss.detach()))
        if verbose and (it % max(1, steps // 10) == 0):
            print(f"  fit {it:4d}  loss={float(loss):+.4f}  "
                  f"rho={float(spectral_radius(P.filter_vals().detach(), d_shell, alpha)):.4f}")

    P.detach_()
    rho_final = float(spectral_radius(P.filter_vals(), d_shell, alpha))
    return P, {"loss": loss_hist, "rho_init": rho_init, "rho_final": rho_final}


def diis_objective(P: MultipoleKerkerPrecond, g2_shell: torch.Tensor,
                   d_shell: torch.Tensor, *, alpha: float = 0.7, q0: float = 1.1,
                   n_unroll: int = 30, history: int = 8,
                   weight: torch.Tensor | None = None) -> float:
    """The unrolled-Pulay-DIIS objective log‖res_N‖ for a filter on a response.

    The single number the robust driver ranks candidates by: lower is a smaller
    predicted post-DIIS residual. Evaluated with the SAME metric the fit and the
    real ``mixing.PulayMixer`` use (Kerker weight 1/(g²+q0²)), so Kerker and every
    learned candidate are scored on one common yardstick.
    """
    g2_shell = g2_shell.to(RDTYPE)
    d_shell = d_shell.to(RDTYPE)
    raw_w = (torch.ones_like(g2_shell) if weight is None
             else weight.to(RDTYPE)).clamp_min(0.0)
    metric = raw_w / (g2_shell + q0**2)
    f = P.filter_vals(g2_shell).detach()
    return float(_diis_unroll_logres(f, d_shell, metric, alpha, n_unroll, history))


def _seed_positions(k: int, q_lo: float, q_hi: float, n_seeds: int
                    ) -> list[list[float]]:
    """Deterministic multi-seed pole placements for a K-pole fit: ``n_seeds``
    anchor scales log-spaced across [q_lo, q_hi], each seeding K poles log-spaced
    over a sub-band around its anchor. No RNG — the same input gives the same
    seeds every call, so selection cannot flip on RNG-state leakage."""
    import math
    lo, hi = math.log10(q_lo), math.log10(q_hi)
    anchors = torch.logspace(lo, hi, n_seeds).tolist()
    half = 0.25 * (hi - lo)                       # sub-band half-width in decades
    seeds: list[list[float]] = []
    for a in anchors:
        la = math.log10(a)
        if k == 1:
            seeds.append([a])
        else:
            sub_lo = max(lo, la - half)
            sub_hi = min(hi, la + half)
            seeds.append(torch.logspace(sub_lo, sub_hi, k).tolist())
    return seeds


def fit_multipole_robust(g2_shell: torch.Tensor, d_shell: torch.Tensor, *,
                         alpha: float = 0.7, q0: float = 1.1,
                         n_unroll: int = 30, history: int = 8, steps: int = 500,
                         lr: float = 0.05, k_max: int = 3, n_seeds: int = 5,
                         q_min: float = 0.3, q_max: float = 3.0,
                         weight: torch.Tensor | None = None,
                         quality: torch.Tensor | None = None,
                         model_tol: float = 0.75, abstain_margin: float = 1.5,
                         rel_margin: float = 0.0, verbose: bool = False
                         ) -> tuple[MultipoleKerkerPrecond, dict[str, Any]]:
    """Robust multi-pole fit: multi-seed, quality-weighted, model-selected, with a
    Kerker abstention gate. Hardens :func:`fit_multipole` against the one-iteration
    fragility where floating-point noise in the probe steered the deterministic
    optimizer into a worse local minimum.

    Four mechanisms, all deterministic:

    1. **Multi-seed.** Each K is fit from ``n_seeds`` deterministic initial pole
       placements spread over the pole range (:func:`_seed_positions`); the
       candidate minimizing the true unrolled-DIIS objective wins. Selecting by the
       actual objective, not by which seed the optimizer happened to start from,
       is what makes the choice noise-stable.
    2. **Probe-quality weighting.** ``quality`` (from
       ``response_from_residuals(..., return_quality=True)``) downweights or
       excludes shells whose d(G) saturates the clamp or scatters widely, and sets
       a pole floor at the lowest trustworthy shell so no pole lands in the
       untrusted low-|G| noise.
    3. **Pole-count model selection.** Bare Kerker (single pole at ``q0``) is always
       a candidate. Fit K=1..``k_max``; pick the SMALLEST K whose best objective is
       within ``model_tol`` of the overall best, so equal-objective fits prefer
       fewer poles (recorded as ``info["selected_k_model"]``).
    4. **Abstention gate.** A single pole is Kerker's own shape, and reshaping it
       (moving w or q off the default) is exactly the noise-level tweak the
       fragility record shows flipping on last-bit rounding — so a K=1 selection
       always deploys BARE Kerker. A K≥2 selection deploys only if it beats the
       best single-pole candidate (bare Kerker or the fitted one-pole, whichever
       is lower) by more than ``max(abstain_margin, rel_margin·|obj_ref|)``, i.e.
       only when the response carries structure a single pole provably cannot
       express, by a margin decisively above fit noise. Otherwise: bare Kerker.
       A noise-level win or loss becomes a guaranteed tie by construction —
       losing to Kerker is impossible at fit time. Two measured calibrations
       behind the defaults: (a) the diagonal model always predicts a large gain
       for merely rescaling Kerker's single pole (10-20 log units even on a
       single-scale response), which is model optimism the real DIIS mixer
       already captures — hence the best-single-pole reference, not bare Kerker;
       (b) across two-scale/three-scale/single-scale/flat synthetic responses
       the K≥2 headroom over the fitted single pole is |Δ| ≲ 0.7 log units
       (DIIS's finite-history extrapolation absorbs smooth radial multi-scale
       structure in the diagonal model), while 1e-14 probe noise moves the
       candidate objectives by up to ~0.75 — so ``abstain_margin=1.5`` sits
       above both, and only a decisively non-single-pole response deploys
       (measured example: a two-stage screening d(G) stepping down at two
       separated scales, K=2 headroom ≈ 20-27).

    Returns the chosen preconditioner (on ``g2_shell``; rebind to deploy) and an
    info dict recording every candidate's objective, the model-selected and
    deployed pole counts, and whether the gate abstained.
    """
    g2_shell = g2_shell.to(RDTYPE)
    d_shell = d_shell.to(RDTYPE)

    # weight for the loss/metric: prefer the probe-quality confidence, fall back to
    # the shell multiplicity, then to uniform. Guard an all-zero quality vector.
    w = quality if quality is not None else weight
    if w is not None:
        w = w.to(RDTYPE).clamp_min(0.0)
        if float(w.sum()) <= 0.0:
            w = weight if weight is not None else None
            w = w.to(RDTYPE).clamp_min(0.0) if w is not None else None

    # pole floor: lowest |G| shell with trustworthy probe data (quality>0). Kills
    # the q≈0.03 Å⁻¹ overfit-to-lowest-shell-noise failure mode structurally.
    q_lo = q_min
    if quality is not None:
        trust = quality.to(RDTYPE) > 0.0
        if bool(trust.any()):
            q_lo = max(q_min, float(g2_shell[trust].min().clamp_min(1e-12).sqrt()))
    q_hi = q_max

    def _obj(P: MultipoleKerkerPrecond) -> float:
        return diis_objective(P, g2_shell, d_shell, alpha=alpha, q0=q0,
                              n_unroll=n_unroll, history=history, weight=w)

    # candidate 0: bare Kerker, always present, the abstention reference.
    kerker = MultipoleKerkerPrecond.kerker(g2_shell, q0)
    j_kerker = _obj(kerker)
    cand: list[dict[str, Any]] = [{"k": 1, "kind": "kerker", "P": kerker, "obj": j_kerker}]

    for k in range(1, k_max + 1):
        best_k: dict[str, Any] | None = None
        for seed_q in _seed_positions(k, q_lo, q_hi, n_seeds):
            P, _ = fit_multipole(g2_shell, d_shell, alpha=alpha, mixer="diis",
                                 history=history, q0=q0, n_unroll=n_unroll,
                                 steps=steps, lr=lr, seed_q=seed_q,
                                 q_floor=q_lo, q_ceil=q_hi, weight=w)
            j = _obj(P)
            if best_k is None or j < best_k["obj"]:
                best_k = {"k": k, "kind": "fit", "P": P, "obj": j}
        assert best_k is not None
        cand.append(best_k)
        if verbose:
            print(f"  K={k}  best obj={best_k['obj']:+.4f}")

    j_best = min(c["obj"] for c in cand)
    # smallest K within model_tol of the best objective (Kerker's K=1 included).
    within = [c for c in cand if c["obj"] <= j_best + model_tol]
    k_sel = min(c["k"] for c in within)
    sel = min((c for c in within if c["k"] == k_sel), key=lambda c: c["obj"])

    # abstention gate. Reference: the best single-pole candidate (bare Kerker or
    # the fitted one-pole) — single-pole reshaping gains are model optimism, not
    # deployable wins. Deploy a K≥2 fit only when it clears the margin over that
    # reference; a K=1 selection always deploys bare Kerker.
    j_ref = min(c["obj"] for c in cand if c["k"] == 1)
    margin = max(abstain_margin, rel_margin * abs(j_ref))
    abstained = False
    chosen = sel
    if sel["kind"] != "kerker" and (sel["k"] == 1
                                    or (j_ref - sel["obj"]) <= margin):
        chosen = cand[0]                                   # return bare Kerker
        abstained = True

    info = {
        "obj_kerker": j_kerker,
        "obj_ref": j_ref,
        "obj_best": j_best,
        "obj_deployed": chosen["obj"],
        "margin": margin,
        "selected_k_model": sel["k"],
        "selected_k": chosen["k"],
        "selected_kind": chosen["kind"],
        "abstained": abstained,
        "q_floor": q_lo,
        "candidates": [{"k": c["k"], "kind": c["kind"], "obj": c["obj"]}
                       for c in cand],
    }
    if verbose:
        gate = "abstain→kerker" if abstained else f"deploy K={chosen['k']}"
        print(f"  robust: kerker={j_kerker:+.4f} best={j_best:+.4f} "
              f"[{gate}]  {chosen['P'].summary()}")
    return chosen["P"], info


def response_from_residuals(res_hist: list[torch.Tensor], g2: torch.Tensor,
                            alpha: float, n_bins: int = 40, skip: int = 2,
                            precond_fac: torch.Tensor | None = None,
                            return_quality: bool = False):
    """Estimate the per-shell response denominator d(G) from a short SCF's
    residual history.

    In the diagonal model a mixing step gives res_{n+1}(G)/res_n(G) = 1 − α·P(G)·d(G),
    where P(G) is whatever preconditioner ran during the probe. With plain damping
    (``precond_fac=None``, P=1) this inverts directly to d = (1 − ratio)/α. Probing
    a d-band or semicore metal with plain damping charge-sloshes, so the robust
    path is to probe with Kerker ON and pass ``precond_fac`` = the Kerker factor
    G²/(G²+q0²) per component; the estimator divides it back out to recover the
    bare d(G). Averaging the complex ratio's real part over consecutive iterations
    and |G|-shells de-noises it; the first ``skip`` pairs are dropped (loose early
    diago tolerance). Shells where the probe preconditioner is too small to invert
    reliably (P < 0.05) are excluded.

    With ``return_quality=True`` a fourth tensor is returned: a per-shell confidence
    weight = count·reliability, where reliability falls with the spread of the
    per-sample d estimates within the shell (a wide scatter is unreliable probe
    data) and is zeroed for shells whose mean d(G) saturates the [0, 2] stable-mode
    clamp (the estimate ran off the invertible range and carries no information).
    The robust fit uses this both as the loss/metric weight and to set the pole
    floor at the lowest shell that is actually trustworthy — which structurally
    forbids placing a pole in the untrusted low-|G| noise.

    Returns (g2_shell, d_shell, count[, quality]): representative |G|² [Å⁻²],
    estimated d, the component count per shell (the default fit ``weight``), and —
    when requested — the confidence weight. G=0 is excluded (pinned).
    """
    g2 = g2.reshape(-1)
    nz = g2 > 1e-12
    pf = None
    if precond_fac is not None:
        nz = nz & (precond_fac.reshape(-1) > 0.05)
        pf = precond_fac.reshape(-1).clamp_min(1e-3)
    gmax = float(g2[g2 > 1e-12].max())
    edges = torch.linspace(0.0, gmax * (1 + 1e-6), n_bins + 1,
                           dtype=RDTYPE, device=g2.device)
    idx = torch.bucketize(g2, edges) - 1
    idx = idx.clamp(0, n_bins - 1)

    d_acc = torch.zeros(n_bins, dtype=RDTYPE, device=g2.device)
    d2_acc = torch.zeros(n_bins, dtype=RDTYPE, device=g2.device)
    n_acc = torch.zeros(n_bins, dtype=RDTYPE, device=g2.device)
    for a, b in zip(res_hist[skip:-1], res_hist[skip + 1:], strict=True):
        ratio = (b / a.masked_fill(a.abs() < 1e-14, 1.0)).real  # 1 − α P d per comp
        d_comp = (1.0 - ratio) / alpha
        if pf is not None:
            d_comp = d_comp / pf                    # divide the probe Kerker out
        keep = nz & torch.isfinite(d_comp)
        vals = d_comp[keep].to(RDTYPE)
        d_acc.index_add_(0, idx[keep], vals)
        d2_acc.index_add_(0, idx[keep], vals * vals)
        n_acc.index_add_(0, idx[keep], torch.ones(int(keep.sum()),
                         dtype=RDTYPE, device=g2.device))

    full = n_acc > 0
    n_f = n_acc[full]
    raw_mean = d_acc[full] / n_f                           # pre-clamp shell mean
    d_shell = raw_mean.clamp(0.0, 2.0)                     # stable-mode range
    centers = 0.5 * (edges[:-1] + edges[1:])[full]
    if not return_quality:
        return centers, d_shell, n_f

    # per-shell spread of the individual d samples; std_scale ~ typical d, so a
    # scatter comparable to d itself roughly halves the shell's confidence.
    var = (d2_acc[full] / n_f - raw_mean * raw_mean).clamp_min(0.0)
    std_scale = 0.25
    reliability = 1.0 / (1.0 + (var.sqrt() / std_scale) ** 2)
    eps = 1e-3
    trust = (raw_mean > eps) & (raw_mean < 2.0 - eps) & (n_f >= 2)
    quality = torch.where(trust, n_f * reliability, torch.zeros_like(n_f))
    return centers, d_shell, n_f, quality
