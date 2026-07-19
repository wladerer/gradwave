"""Non-basis numerical error estimates: SCF, k-point, and smearing.

The plane-wave (Ecut) error in ``postscf.discretization_error`` is one term in
the numerical error budget. The other convergence errors -- how far the reported
energy sits from the fully self-consistent, dense-k, zero-temperature Kohn-Sham
value for the SAME functional -- are estimated here. None of these touch the XC
model error, which is not reachable from inside a run (see docs/ideas.md).

Three terms, each with a different structure:

  SCF convergence error. Stopping the iteration at finite ``rhotol`` leaves a
  density residual r = rho_out - rho_in. Because the energy is stationary at the
  fixed point, the energy error is second order in r,

      dE_scf = (1/2) <r | K_Hxc | (1 - chi0 K_Hxc)^-1 r>,

  the Hartree-XC energy of the residual density screened by the SCF dielectric
  operator. Both operators are the response primitives ``scf/implicit.py``
  already exposes for the Dyson dressing, so no new SCF is taken. The screened
  form needs chi0 (insulator, nspin=1, no symmetry); elsewhere the unscreened
  (1/2)<r | K_Hxc | r> is reported as an overestimate.

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

import math
from dataclasses import dataclass

import torch

from gradwave.core.occupations import SCHEMES
from gradwave.dtypes import RDTYPE
from gradwave.scf.implicit import apply_chi0, apply_k_hxc
from gradwave.scf.loop import SCFResult


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

    ``denergy`` (>= 0) is the estimated distance of the reported energy above
    the fully self-consistent value: E_converged ~= free_energy - denergy. It is
    the screened second-order form when available, else the unscreened
    overestimate (``screened`` records which). ``residual_norm`` is the L1 norm
    of the density residual per electron, the same convergence proxy the SCF
    prints.
    """

    denergy: float               # eV, screened if available else unscreened
    denergy_unscreened: float    # eV, (1/2)<r|K_Hxc|r>, always an overestimate
    residual_norm: float         # int|r| per electron
    screened: bool               # True if the dielectric-screened value is used
    energy_converged_estimate: float  # free_energy - denergy [eV]
    last_de: float               # last |dF| from the SCF history (bracket cross-check)


@torch.no_grad()
def estimate_scf_error(res: SCFResult, xc, *,
                       screened: bool | None = None,
                       dyson_beta: float = 0.4,
                       dyson_tol: float = 1e-7,
                       dyson_max_iter: int = 80) -> ScfConvergenceError:
    """Estimate the SCF (self-consistency) energy error of a converged run.

    Uses the stored last-step density residual r = rho_out - rho_in
    (``res.drho_scf``) and the Hartree-XC kernel K_Hxc. The unscreened estimate
    (1/2)<r|K_Hxc|r> is always available; the screened estimate divides out the
    SCF dielectric operator via one Dyson solve x = (1 - chi0 K_Hxc)^-1 r and
    forms (1/2)<x|K_Hxc|r>, which requires chi0 (insulator, nspin=1, no
    symmetry). ``screened=None`` tries the screened form and falls back to the
    unscreened one; ``screened=False`` forces the cheap overestimate.

    Cross-check: run the SCF to a loose and a tight ``rhotol`` and compare the
    loose ``denergy`` against F_loose - F_tight (the way the eigenvalue error
    cross-checks against ``denergy``).
    """
    if res.drho_scf is None:
        raise ValueError(
            "no SCF residual stored on this result (drho_scf is None): re-run "
            "the SCF with the current gradwave to populate it")
    system = res.system
    grid = system.grid
    vol, npts = grid.volume, grid.n_points
    r = res.drho_scf.to(RDTYPE)

    kr = apply_k_hxc(res, xc, r)                 # physical potential [eV] of r
    cell = vol / npts
    denergy_unscreened = 0.5 * float((r * kr).sum()) * cell

    used_screened = False
    denergy = denergy_unscreened
    if screened is not False:
        try:
            x = _dyson_solve(res, xc, r, beta=dyson_beta, tol=dyson_tol,
                             max_iter=dyson_max_iter)
            denergy = 0.5 * float((x * kr).sum()) * cell
            used_screened = True
        except (NotImplementedError, RuntimeError):
            if screened is True:
                raise
    nelec = float(system.n_electrons)
    res_norm = float(r.abs().sum()) * cell / nelec
    last_de = float(res.history[-1]["dE"]) if res.history else float("nan")
    free = float(res.energies.free_energy)
    return ScfConvergenceError(
        denergy=denergy,
        denergy_unscreened=denergy_unscreened,
        residual_norm=res_norm,
        screened=used_screened,
        energy_converged_estimate=free - denergy,
        last_de=last_de,
    )


@torch.no_grad()
def _dyson_solve(res, xc, r, *, beta, tol, max_iter):
    """x = (1 - chi0 K_Hxc)^-1 r by damped fixed-point iteration.

    Solves x = r + chi0[K_Hxc[x]], the same operator ``discretization_error``'s
    Dyson dressing uses, here applied to the SCF residual. chi0 restricts this
    to nspin=1 insulators with use_symmetry=False (apply_chi0 raises otherwise).
    """
    x = r.clone()
    for _ in range(max_iter):
        x_new = r + apply_chi0(res, apply_k_hxc(res, xc, x))
        denom = max(1.0, float(torch.linalg.norm(x)))
        step = float(torch.linalg.norm(x_new - x)) / denom
        x = x + beta * (x_new - x)
        if step < tol:
            return x
    raise RuntimeError(f"Dyson solve not converged ({step:.2e} after {max_iter} iters)")


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
                for n, e in zip(pts, en)]
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
    sae = sum(ai * ei for ai, ei in zip(a, en))
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
