"""Fitted contracted Slater-type-orbital (STO) radial basis for the COHP
projector -- a LOBSTER-style minimal local basis (see
docs/plans/cohp-contracted-basis.md).

docs/plans/cohp-contracted-basis.md's "Basis diffuseness" follow-up measured
that Knizia IAO (postscf/cohp.py's ``_iao_projectors_k``) does NOT shrink the
projector's real-space extent -- it fixes occupied-space *completeness*
(charge_spilling -> 0), which turns out to be the wrong lever: IAO is built
to reproduce the occupied KS states exactly, including whatever bonding
character extends onto neighbouring atoms, so nothing in its construction
penalizes spatial spread. A naive radial truncation (Gaussian-damping the
PP_PSWFC tail) fails for the same underlying reason in reverse:
charge_spilling blows past 20% before the COHP magnitude approaches the
target, meaning the "fix" only works by no longer representing the occupied
states.

LOBSTER's own contracted-STO tables (Bunge / pbeVaspFit2015-style) sidestep
this by fitting the basis at the FREE-ATOM level -- a curve fit of a minimal
Slater-type radial function to the isolated atom's own valence orbital --
entirely independent of any crystal SCF state. Because the fit target is the
atom, not the crystal's occupied manifold, the result cannot re-diffuse to
chase completeness the way IAO does: there is no crystal bonding character in
the free atom to absorb in the first place.

gradwave already carries this exact free-atom target for every element with
pseudopotential support: the UPF file's PP_PSWFC/PP_AECHI pseudo-atomic
valence orbital (``pseudo.upf.AtomicOrbital.rchi = r * R_nl(r)``, the same
table already consumed by ``postscf/pdos.py``'s and ``postscf/cohp.py``'s
``basis="pswfc"``/``"iao"`` paths). This module fits a short contraction of a
few STO primitives -- fixed principal quantum number n (the orbital's own
shell, parsed off its UPF label, e.g. "2P" -> n=2), free exponents zeta_i
(optimized in log-space to stay positive), free linear coefficients c_i -- to
that same table by Adam (the exponents enter nonlinearly), the same
differentiable-fit pattern as ``scf.learned_precond.fit_multipole``: freeze
nothing crystal-dependent, unroll a cheap forward model, backprop into a
handful of scalar parameters.

The result is a radial function that reproduces the PSWFC shape well inside
the pseudization radius (where the chemistry is) while shedding its long
tail via a small number of exponentials rather than the full radial-mesh
degrees of freedom of the raw UPF table -- the actual lever the diffuseness
diagnosis called for, unlike IAO or truncation. Whether it closes the
diamond-vs-LOBSTER magnitude gap is an open, measured question (no LOBSTER
binary/fixture is available in-tree to calibrate against, per the plan doc);
this module is the fitting machinery the calibration needs, not a claim that
it already matches LOBSTER's number.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from gradwave.dtypes import RDTYPE

_LABEL_N_RE = re.compile(r"^(\d+)")


def n_from_label(label: str) -> int:
    """Principal quantum number from a UPF orbital label ("2P" -> 2, "4S" ->
    4). Raises if the label doesn't start with an integer -- every PP_PSWFC/
    PP_AECHI label observed in the fixture tree does."""
    m = _LABEL_N_RE.match(label)
    if m is None:
        raise ValueError(f"orbital label {label!r} does not start with a principal quantum number")
    return int(m.group(1))


@dataclass
class ContractedSTO:
    """R(r) = sum_i c_i/N_i * r^(n-1) * exp(-zeta_i r), a short contraction of
    Slater primitives sharing one principal quantum number ``n`` (the
    orbital's own shell) and angular momentum ``l``. ``rchi(r)`` returns
    r*R(r), the same convention as ``pseudo.upf.AtomicOrbital.rchi``, so it
    drops straight into ``pseudo.radial.sbt`` / ``postscf.pdos._ao_projectors_k``
    in place of the raw UPF table."""

    n: int
    l: int
    zeta: torch.Tensor  # (nprim,) > 0
    c: torch.Tensor     # (nprim,) contraction coefficients

    def _norm(self) -> torch.Tensor:
        # <r^(n-1)e^-zr | r^(n-1)e^-zr> with weight r^2 dr (the physical
        # radial norm) = integral_0^inf r^(2n) e^(-2 zeta r) dr
        #              = (2n)! / (2 zeta)^(2n+1)
        two_n = 2 * self.n
        log_fact = math.lgamma(two_n + 1)
        return torch.exp(0.5 * (log_fact - (two_n + 1) * torch.log(2.0 * self.zeta)))

    def rchi(self, r: torch.Tensor) -> torch.Tensor:
        """r * R(r) on an arbitrary radial grid r (>=0), shape matching r."""
        norm = self._norm()  # (nprim,)
        # r^n * exp(-zeta r) = r * [r^(n-1) exp(-zeta r)] -- already the r*R form
        primitives = r[None, :].pow(self.n) * torch.exp(-self.zeta[:, None] * r[None, :])
        return ((self.c / norm)[:, None] * primitives).sum(dim=0)


def fit_contracted_sto(
    r: torch.Tensor, rab: torch.Tensor, rchi_ref: torch.Tensor, *,
    n: int, l: int, n_primitives: int = 3, steps: int = 800, lr: float = 0.05,
    seed: int = 0, tail_reg: float = 1e-3,
) -> tuple[ContractedSTO, float]:
    """Fit a ``ContractedSTO(n, l, ...)`` to a reference ``rchi(r)`` table
    (e.g. a UPF PP_PSWFC/PP_AECHI free-atom orbital), the SHAPE where the
    orbital actually carries amplitude, by Adam-then-LBFGS minimizing a
    density-weighted L2 residual. Exponents are optimized in log-space to
    stay positive; the fit is renormalized to unit norm (integral rchi^2 dr
    = 1, the convention ``pseudo.upf.AtomicOrbital`` documents) after
    optimization -- a slightly-mis-normalized good SHAPE is a better local
    optimum to keep than an exactly-normalized worse one, so normalization is
    enforced as a final rescale rather than folded into the loss.

    Weighting matters more than it looks: a plain ``rab``-weighted L2 (the
    natural quadrature measure) reproduces the reference's diffuse numerical
    TAIL almost exactly (measured on diamond-C's 2P PP_PSWFC: fitted 1%-tail
    radius 5.05 A vs. the raw table's 5.04 A) -- on a UPF log-radial mesh
    ``rab`` GROWS with r, so a few outer near-zero-amplitude points dominate
    the sum and the optimizer "cheats" with one anomalously soft primitive
    that chases the tail instead of contracting the basis, defeating the
    entire point (docs/plans/cohp-contracted-basis.md's basis-diffuseness
    fix). Weighting by ``rab * rchi_ref^2`` instead (a relative-shape, not
    absolute, objective) removes the reward for tail-chasing, and a small
    ``tail_reg * sum(1/zeta)`` penalty then actively pulls the otherwise-
    underdetermined tail toward MORE localized (larger-zeta) primitives --
    operationalizing LOBSTER's own stated principle that its "bonding
    methods are bound to minimal basis sets on purpose" rather than leaving
    basis extent to chance.

    Returns ``(fitted ContractedSTO, RMS residual on the density-weighted
    grid)`` -- the residual is a fit-quality diagnostic, not itself a
    SCF/postscf quantity; a large residual means ``n_primitives`` is too
    small to trust projected DOS/COHP built from the fit."""
    if not torch.is_grad_enabled():
        # callers like postscf.cohp.cohp() run under @torch.no_grad() for
        # the SCF-state math; this fit needs autograd regardless of the
        # ambient context, exactly like scf.spin_precond._stoner_kernel_diag's
        # torch.enable_grad() escape for the same reason.
        with torch.enable_grad():
            return fit_contracted_sto(
                r, rab, rchi_ref, n=n, l=l, n_primitives=n_primitives,
                steps=steps, lr=lr, seed=seed, tail_reg=tail_reg)
    torch.manual_seed(seed)
    r = r.to(RDTYPE)
    rab = rab.to(RDTYPE)
    rchi_ref = rchi_ref.to(RDTYPE)

    # geometric exponent spread bracketing the reference orbital's own
    # inverse extent (from where |rchi| peaks) -- a generic, no-tuning start
    r_peak = max(float(r[torch.argmax(rchi_ref.abs())]), 0.3)
    zeta_lo, zeta_hi = 0.5 / r_peak, 4.0 / r_peak
    if n_primitives == 1:
        zeta_init = torch.tensor([math.sqrt(zeta_lo * zeta_hi)], dtype=RDTYPE)
    else:
        zeta_init = torch.logspace(
            math.log10(zeta_lo), math.log10(zeta_hi), n_primitives, dtype=RDTYPE)
    log_zeta = zeta_init.log().clone().requires_grad_(True)
    c = (0.5 + torch.rand(n_primitives, dtype=RDTYPE)).requires_grad_(True)

    opt = torch.optim.Adam([log_zeta, c], lr=lr)
    peak2 = rchi_ref.pow(2).max()
    w = rab.clamp_min(0.0) * (rchi_ref.pow(2) + 1e-4 * peak2)
    w_sum = w.sum()

    def _loss() -> torch.Tensor:
        zeta = log_zeta.exp()
        sto = ContractedSTO(n=n, l=l, zeta=zeta, c=c)
        resid = sto.rchi(r) - rchi_ref
        fit_term = (w * resid.pow(2)).sum() / w_sum
        reg_term = tail_reg * (1.0 / zeta).sum()
        return fit_term + reg_term

    for _ in range(steps):
        opt.zero_grad()
        _loss().backward()
        opt.step()

    # Adam plateaus well short of convergence on this stiff, low-dimensional
    # nonlinear least-squares problem (exponent fitting is notoriously
    # ill-conditioned); a short LBFGS polish from Adam's neighborhood tightens
    # the fit by 1-2 orders of magnitude at negligible extra cost (3-6
    # scalars, closed-form gradients via autograd).
    lbfgs = torch.optim.LBFGS([log_zeta, c], max_iter=200, line_search_fn="strong_wolfe")

    def _closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss = _loss()
        loss.backward()
        return loss

    lbfgs.step(_closure)

    # physical normalization uses PLAIN dr quadrature, not the density weighting above
    plain_w = rab.clamp_min(0.0)
    with torch.no_grad():
        sto = ContractedSTO(n=n, l=l, zeta=log_zeta.exp(), c=c)
        norm2 = (plain_w * sto.rchi(r).pow(2)).sum()
        sto = ContractedSTO(n=n, l=l, zeta=sto.zeta, c=sto.c / norm2.sqrt())
        resid = sto.rchi(r) - rchi_ref
        rmse = float(((w * resid.pow(2)).sum() / w_sum).sqrt())
    return sto, rmse
