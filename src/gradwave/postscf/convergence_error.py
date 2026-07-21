"""Non-basis numerical error estimates: SCF, k-point, and smearing.

The plane-wave (Ecut) error in ``postscf.discretization_error`` is one term in
the numerical error budget. The other convergence errors -- how far the reported
energy sits from the fully self-consistent, dense-k, zero-temperature Kohn-Sham
value for the SAME functional -- are estimated here. None of these touch the XC
model error, which is not reachable from inside a run (see docs/ideas.md).

Three terms, each with a different structure:

  SCF convergence error. Stopping the iteration at finite ``rhotol`` leaves the
  reported free energy a little above the fully self-consistent value E_inf. The
  robust estimate reads this distance straight off the recorded energy
  trajectory (``res.history``): in the convergence basin the tail is geometric,
  E_i - E_inf ~ q^i, so the unobserved remainder sums to
  E_inf - E_last ~ dE_last * q / (1 - q) with q the ratio of the last energy
  steps. This needs one run and no response solve, and -- because it only reads
  the recorded energies -- it works for every system (metal, spin, symmetry,
  USPP, noncollinear), reporting a non-negative ``denergy``.

  A second-order response form is also available as a DIAGNOSTIC when ``xc`` and
  the collinear response primitives are supplied. Because the energy is
  stationary at the fixed point the error is second order in the density
  residual r = rho_out - rho_in, and the exact form is
  (1/2)<x | (K_Hxc - chi0^-1) | x> with x = (1 - chi0 K_Hxc)^-1 r the
  dielectric-dressed residual. The code can only form the historical
  (1/2)<r | K_Hxc (1 - chi0 K_Hxc)^-1 | r>, which omits the chi0^-1
  kinetic-response term (it needs a near-singular chi0^-1 solve; see
  docs/ideas.md) and is therefore NOT sign-definite. It is kept as a diagnostic
  and never drives the headline estimate.

  Smearing error. A finite electronic temperature sigma reports the free energy
  F = E - sigma*S instead of the sigma->0 energy. The scheme-matched
  extrapolation E0 = (E + F)/2 cancels the leading entropy-order term for every
  smearing this code supports (each is a matched occupation/entropy pair). The
  reported free energy differs from E0 by (F - E)/2 = -sigma*S/2. Be careful:
  this is a variational-basis extrapolation, valid only because the (f, s) pair
  is constructed for it; a physical finite-temperature Fermi-Dirac run does not
  want it removed, and a fixed-occupation run has no smearing error at all.

  k-point sampling error. BZ integration is a quadrature, not a truncated
  variational space, so the complement/second-order structure does not transfer.
  It is reached instead by mesh extrapolation: fit E(N_k) -> E_inf and report the
  residual of the finest mesh. This one needs more than one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import torch

from gradwave.core.occupations import SCHEMES
from gradwave.dtypes import RDTYPE

# DysonNotConverged moved to postscf._response with the shared Dyson solver;
# re-exported here for the callers that import it from this module.
from gradwave.postscf._response import DysonNotConverged, dyson_fixed_point
from gradwave.scf.implicit import apply_chi0, apply_k_hxc
from gradwave.scf.loop import SCFResult

__all__ = [
    "DysonNotConverged",
    "ScfConvergenceError",
    "SmearingError",
    "estimate_kpoint_error",
    "estimate_scf_error",
    "estimate_scf_error_bracket",
    "estimate_smearing_error",
]


# --------------------------------------------------------------------------- #
#  Smearing / electronic-temperature error                                    #
# --------------------------------------------------------------------------- #


# Per-scheme note on the (E+F)/2 extrapolation. Every scheme in occupations.py
# is a matched (occupation, entropy) pair, so E0 = (E+F)/2 cancels the leading
# term; what differs is how large the residual is and whether the finite-T
# result is itself the physical target.
_SMEAR_NOTES = {
    "gaussian": "Gaussian: E0=(E+F)/2 cancels the O(sigma^2) term; residual O(sigma^4).",
    "mp1": "Methfessel-Paxton order 1: entropy already O(sigma^4)-small, E0 is a "
           "high-order estimate (sigma*S may be negative).",
    "cold": "Marzari-Vanderbilt cold: entropy tiny by construction, E ~ F ~ E0.",
    "fermi-dirac": "Fermi-Dirac: E0 extrapolates to T=0; for a deliberately "
                   "physical-temperature run the finite-T free energy F is the "
                   "intended answer and this correction should NOT be applied.",
}


@dataclass
class SmearingError:
    """Estimated electronic-temperature (smearing) error of the reported energy.

    ``dsmearing`` is the correction from the reported free energy F to the
    sigma->0 extrapolation E0 = (E + F)/2, i.e. E0 = free_energy + dsmearing.
    ``half_width`` = |sigma*S|/2 bounds the residual error remaining in E0.
    """

    scheme: str
    width: float
    entropy_term: float          # -sigma*S [eV] (EnergyBreakdown.smearing)
    kohn_sham_energy: float      # E [eV]
    free_energy: float           # F = E - sigma*S [eV], the reported energy
    energy_extrapolated: float   # E0 = (E + F)/2 [eV], the sigma->0 estimate
    dsmearing: float             # E0 - F [eV], correction to reach sigma->0
    half_width: float            # |sigma*S|/2 [eV], residual bound on E0
    note: str = ""


def estimate_smearing_error(res: SCFResult, *, scheme: str,
                            width: float | None = None) -> SmearingError:
    """Estimate the finite-smearing error from a single converged SCF.

    The entropy term ``-sigma*S`` is already carried on ``res.energies``, so this
    is exposure and extrapolation, not new computation. Returns the sigma->0
    energy E0 = (E + F)/2 and the correction dsmearing = E0 - F.

    ``scheme`` is the smearing name (``fermi-dirac`` | ``gaussian`` | ``mp1`` |
    ``cold``). Raises ValueError for ``none`` / fixed occupations (no smearing
    error) or a negligible entropy term (an insulator run with a token width).
    """
    if scheme is None or scheme == "none":
        raise ValueError("no smearing (fixed occupations): no smearing error")
    if scheme not in SCHEMES:
        raise ValueError(f"unknown smearing scheme {scheme!r}")
    e = res.energies
    entropy_term = float(e.smearing)   # = -sigma*S
    ks = float(e.total)                # E
    free = float(e.free_energy)        # F = E - sigma*S
    e0 = float(e.e0)                   # E0 = (E + F)/2 = E - sigma*S/2
    if abs(entropy_term) < 1e-10:
        raise ValueError(
            "smearing entropy is negligible (gapped bands / no partial "
            "occupation): no meaningful smearing error")
    return SmearingError(
        scheme=scheme,
        width=float(width) if width is not None else float("nan"),
        entropy_term=entropy_term,
        kohn_sham_energy=ks,
        free_energy=free,
        energy_extrapolated=e0,
        dsmearing=e0 - free,
        half_width=0.5 * abs(entropy_term),
        note=_SMEAR_NOTES.get(scheme, ""),
    )


# --------------------------------------------------------------------------- #
#  SCF convergence error                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class ScfConvergenceError:
    """Estimated energy error from stopping the SCF at finite tolerance.

    ``denergy`` (>= 0) is the robust headline: the magnitude of the distance
    between the reported free energy and the fully self-consistent value E_inf,
    extrapolated from the SCF energy trajectory as a geometric tail (see
    ``_extrapolate_energy_tail``). ``energy_converged_estimate`` is that
    extrapolated E_inf. Because it reads only the recorded energies it needs no
    response solve and works for every system. ``reliable`` is False when the
    tail is too short or not geometrically converging; then ``denergy`` falls
    back to the last |dF|, a crude upper proxy -- treat it as order-of-magnitude
    only. ``ratio`` is the fitted geometric ratio and ``n_tail`` the number of
    energy steps it used.

    The response-based second-order form is exposed only as a DIAGNOSTIC, and
    only when ``xc`` (and the collinear response primitives) are supplied:
    ``denergy_response`` is the historical 1/2<r|K_Hxc (1-chi0 K)^-1|r> (screened
    if ``screened`` else the unscreened 1/2<r|K_Hxc|r>), and ``denergy_unscreened``
    the unscreened value. NEITHER is sign-definite: the exact second-order error
    is 1/2<x|(K_Hxc - chi0^-1)|x> and both diagnostics omit the chi0^-1
    kinetic-response term, so they can come out negative. They are kept for
    analysis and never drive ``denergy``; see docs/ideas.md and the module
    docstring for the derivation.
    """

    denergy: float               # eV, robust extrapolated |F - E_inf| (>= 0)
    energy_converged_estimate: float  # eV, extrapolated E_inf
    residual_norm: float         # int|r| per electron (nan if no residual stored)
    reliable: bool               # True if the geometric extrapolation is trustworthy
    ratio: float                 # fitted geometric ratio q of the energy tail (nan if unused)
    n_tail: int                  # energy steps used in the extrapolation
    last_de: float               # last |dF| from the SCF history (crude fallback proxy)
    screened: bool               # True if the response diagnostic used the screened form
    denergy_response: float | None    # eV, response diagnostic; NOT sign-definite (None if unset)
    denergy_unscreened: float | None  # eV, unscreened response diagnostic (None if unset)


def _extrapolate_energy_tail(history, *, min_iter: int = 4, max_tail: int = 4):
    """Estimate (E_inf - E_last, |E_inf - E_last|, q, n_tail, reliable) from the
    tail of an SCF free-energy trajectory.

    Models the converged tail as geometric, E_i - E_inf ~ q^i, and sums the
    unobserved remainder: E_inf - E_last ~ dE_last * q / (1 - q), with q the
    median of the last few consecutive energy steps. A SIGNED q handles both
    monotone (q > 0) and oscillatory (q < 0) convergence. ``reliable`` is False
    when the history is shorter than ``min_iter``, when fewer than two step
    ratios are available, or when the tail is not clearly contracting
    (|q| >= 0.95); in those cases it returns the last signed step as a crude
    upper proxy for the remainder.
    """
    energies = [float(h["free_energy"]) for h in history]
    steps = [energies[i] - energies[i - 1] for i in range(1, len(energies))]
    last = steps[-1] if steps else 0.0
    if last == 0.0:                                  # already at the energy floor
        return 0.0, 0.0, float("nan"), 0, True
    ratios = [steps[i] / steps[i - 1]
              for i in range(1, len(steps)) if steps[i - 1] != 0.0]
    if len(ratios) < 2:                              # too short to fit a ratio
        return last, abs(last), float("nan"), len(steps), False
    tail = ratios[-max_tail:]
    q = median(tail)
    if not abs(q) < 0.999:                           # not contracting: proxy only
        return last, abs(last), q, len(tail), False
    remaining = last * q / (1.0 - q)
    reliable = (len(energies) >= min_iter and abs(q) < 0.95
                and all(abs(rt) < 1.0 for rt in tail))
    return remaining, abs(remaining), q, len(tail), reliable


@torch.no_grad()
def estimate_scf_error(res: SCFResult, xc=None, *,
                       screened: bool | None = None,
                       dyson_beta: float = 0.4,
                       dyson_tol: float = 1e-7,
                       dyson_max_iter: int = 80) -> ScfConvergenceError:
    """Estimate the SCF (self-consistency) energy error of a converged run.

    The headline ``denergy`` extrapolates the recorded energy trajectory
    (``res.history``) as a geometric tail -- see ``_extrapolate_energy_tail``.
    It needs one run, no response solve, and works for every system, returning a
    non-negative distance to the fully self-consistent energy.

    ``xc`` is optional. When supplied (and the collinear response primitives
    apply -- insulator, nspin=1, no symmetry), the second-order response
    DIAGNOSTIC is also computed from the stored residual r = rho_out - rho_in
    (``res.drho_scf``): the unscreened 1/2<r|K_Hxc|r> and, unless
    ``screened=False``, the screened 1/2<x|K_Hxc|r> with x = (1 - chi0 K)^-1 r.
    Both are reported as ``denergy_response``/``denergy_unscreened`` and are NOT
    sign-definite (they omit the chi0^-1 term); they never drive ``denergy``.
    ``screened=True`` forces the screened diagnostic (re-raising if chi0 is
    unavailable); ``screened=None`` tries it and falls back to the unscreened one.

    Cross-check: for a ground-truth number run the SCF to a loose and a tight
    tolerance and pass both to ``estimate_scf_error_bracket``.
    """
    if not res.history:
        raise ValueError(
            "no SCF history stored on this result: cannot extrapolate the "
            "self-consistency energy error")
    remaining, denergy, q, n_tail, reliable = _extrapolate_energy_tail(res.history)
    free = float(res.energies.free_energy)
    last_de = float(res.history[-1]["dE"])

    # Optional response diagnostic (not sign-definite; never the headline).
    denergy_response = None
    denergy_unscreened = None
    used_screened = False
    res_norm = float("nan")
    if xc is not None and res.drho_scf is not None:
        grid = res.system.grid
        cell = grid.volume / grid.n_points
        r = res.drho_scf.to(RDTYPE)
        kr = apply_k_hxc(res, xc, r)                 # physical potential [eV] of r
        denergy_unscreened = 0.5 * float((r * kr).sum()) * cell
        denergy_response = denergy_unscreened
        if screened is not False:
            try:
                x = _dyson_solve(res, xc, r, beta=dyson_beta, tol=dyson_tol,
                                 max_iter=dyson_max_iter)
                denergy_response = 0.5 * float((x * kr).sum()) * cell
                used_screened = True
            except (NotImplementedError, DysonNotConverged):
                if screened is True:
                    raise
        nelec = float(res.system.n_electrons)
        res_norm = float(r.abs().sum()) * cell / nelec
    elif screened is True:
        raise ValueError(
            "screened=True needs xc and a stored residual (res.drho_scf); "
            "pass xc and re-run the SCF with the current gradwave to populate it")

    return ScfConvergenceError(
        denergy=denergy,
        energy_converged_estimate=free + remaining,
        residual_norm=res_norm,
        reliable=reliable,
        ratio=q,
        n_tail=n_tail,
        last_de=last_de,
        screened=used_screened,
        denergy_response=denergy_response,
        denergy_unscreened=denergy_unscreened,
    )


def estimate_scf_error_bracket(res_loose: SCFResult, res_tight: SCFResult) -> dict:
    """Ground-truth SCF error from a loose/tight pair of runs of the SAME system.

    Returns the measured energy gap F_loose - F_tight (the reported energy's
    distance above the tighter reference) alongside the loose run's extrapolated
    estimate, so the single-run extrapolation can be checked against a real
    second run. ``denergy`` is the measured gap; ``denergy_estimated`` is the
    loose run's ``estimate_scf_error(res_loose).denergy``; ``ratio`` is their
    quotient (near 1 when the estimate is on the money).

    Run the tight SCF to a tolerance well below the loose one so F_tight is a
    faithful stand-in for E_inf; the two runs must share cell, k-mesh, cutoff,
    and functional.
    """
    f_loose = float(res_loose.energies.free_energy)
    f_tight = float(res_tight.energies.free_energy)
    measured = f_loose - f_tight
    est = estimate_scf_error(res_loose)
    denom = abs(measured) if abs(measured) > 0.0 else float("nan")
    return {
        "denergy": measured,
        "denergy_estimated": est.denergy,
        "energy_converged_estimate_eV": est.energy_converged_estimate,
        "free_energy_tight_eV": f_tight,
        "reliable": est.reliable,
        "ratio": est.denergy / denom,
    }


@torch.no_grad()
def _dyson_solve(res, xc, r, *, beta, tol, max_iter):
    """x = (1 - chi0 K_Hxc)^-1 r by damped fixed-point iteration.

    Solves x = r + chi0[K_Hxc[x]], the same operator ``discretization_error``'s
    Dyson dressing uses, here applied to the SCF residual. chi0 restricts this
    to nspin=1 insulators with use_symmetry=False (apply_chi0 raises otherwise).
    Defaults (beta 0.4, tol 1e-7, max_iter 80 at the caller) and the
    DysonNotConverged raise are this site's historical behavior; see
    ``dyson_fixed_point``'s note on the divergence between the former copies.
    """

    def _fail(step):
        raise DysonNotConverged(
            f"Dyson solve not converged ({step:.2e} after {max_iter} iters)")

    return dyson_fixed_point(
        lambda x: apply_chi0(res, apply_k_hxc(res, xc, x)), r,
        beta=beta, tol=tol, max_iter=max_iter, on_fail=_fail)


# --------------------------------------------------------------------------- #
#  k-point sampling error                                                      #
# --------------------------------------------------------------------------- #


def estimate_kpoint_error(nkpts, energies, *,
                          exponent: float | None = None) -> dict:
    """Extrapolate a BZ-quadrature (k-point) error from a mesh sweep.

    Fits E(N_k) = E_inf + c * N_k^-p to energies at increasing k-meshes and
    reports the extrapolated dense-k limit and the residual error of the finest
    mesh. BZ integration is a quadrature, so this needs several runs, not one;
    the complement/second-order trick that reaches the basis-set error does not
    transfer to a sampling error.

    Parameters
    ----------
    nkpts : sequence of int
        Total BZ k-point count of each run (e.g. N^3 for an N*N*N mesh). Use a
        consistent measure across the sweep; the fit only sees ratios.
    energies : sequence of float
        Free energies [eV], one per mesh, aligned with ``nkpts``.
    exponent : float, optional
        Fix the convergence exponent p. With >= 3 points p is fit from the data
        and this overrides it; with exactly 2 points p is required (default 2.0).

    Returns a dict with ``e_infinity_eV``, ``error_eV`` (E_finest - E_inf, the
    residual of the densest mesh), ``exponent``, and the per-mesh signed errors.

    Note (smearing coupling): a metal's k-convergence rate depends on the
    smearing width -- the Fermi-surface discontinuity is what makes it algebraic
    rather than exponential -- so extrapolate at a FIXED width and treat a
    width change as a separate axis.
    """
    pts = [float(n) for n in nkpts]
    en = [float(e) for e in energies]
    if len(pts) != len(en):
        raise ValueError("nkpts and energies must have equal length")
    if len(pts) < 2:
        raise ValueError("need at least two meshes to extrapolate")
    order = sorted(range(len(pts)), key=lambda i: pts[i])
    pts = [pts[i] for i in order]
    en = [en[i] for i in order]
    if len(set(pts)) != len(pts):
        raise ValueError("duplicate k-point counts in the sweep")

    if len(pts) == 2:
        p = 2.0 if exponent is None else float(exponent)
        e_inf = _extrapolate_two(pts, en, p)
    else:
        p = _fit_exponent(pts, en) if exponent is None else float(exponent)
        e_inf = _extrapolate_fit(pts, en, p)

    per_mesh = [{"nk": int(n), "energy_eV": e, "error_eV": e - e_inf}
                for n, e in zip(pts, en, strict=True)]
    return {
        "e_infinity_eV": e_inf,
        "error_eV": en[-1] - e_inf,        # residual of the finest mesh
        "exponent": p,
        "n_meshes": len(pts),
        "per_mesh": per_mesh,
    }


def _extrapolate_two(pts, en, p):
    a0, a1 = pts[0] ** (-p), pts[1] ** (-p)
    c = (en[0] - en[1]) / (a0 - a1)
    return en[1] - c * a1


def _extrapolate_fit(pts, en, p):
    # least-squares E_i = E_inf + c * a_i with a_i = N_i^-p (linear in E_inf, c)
    a = [n ** (-p) for n in pts]
    m = len(pts)
    sa = sum(a)
    saa = sum(ai * ai for ai in a)
    se = sum(en)
    sae = sum(ai * ei for ai, ei in zip(a, en, strict=True))
    det = m * saa - sa * sa
    if abs(det) < 1e-300:
        return en[-1]
    e_inf = (saa * se - sa * sae) / det
    return e_inf


def _fit_exponent(pts, en):
    """Estimate p from the three finest points by matching difference ratios.

    E_i = E_inf + c N_i^-p gives (E1-E2)/(E2-E3) = (N1^-p - N2^-p)/(N2^-p - N3^-p),
    a monotone function of p; bisect for it. Falls back to p=2 on a degenerate
    (non-monotone / metallic-noise) triple.
    """
    n1, n2, n3 = pts[-3], pts[-2], pts[-1]
    e1, e2, e3 = en[-3], en[-2], en[-1]
    d12, d23 = e1 - e2, e2 - e3
    if d23 == 0.0 or (d12 / d23) <= 0.0:
        return 2.0
    target = d12 / d23

    def ratio(p):
        return (n1 ** (-p) - n2 ** (-p)) / (n2 ** (-p) - n3 ** (-p))

    lo, hi = 0.3, 8.0
    rlo, rhi = ratio(lo), ratio(hi)
    if (rlo - target) * (rhi - target) > 0:
        return 2.0  # target outside the bracket: fall back
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (ratio(mid) - target) * (rlo - target) <= 0:
            hi = mid
        else:
            lo, rlo = mid, ratio(mid)
    p = 0.5 * (lo + hi)
    return max(0.3, min(8.0, p))
