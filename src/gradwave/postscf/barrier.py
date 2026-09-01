"""Barrier sensitivity to a Hamiltonian parameter — dE_a/dλ in one backward pass.

A reaction barrier is ``E_a(λ) = E_TS(λ, R_TS(λ)) − E_IS(λ, R_IS(λ))`` where λ
is a Hamiltonian parameter (a Hubbard U, an XC coefficient, or — grand
canonically — the electrode potential) and ``R_IS``/``R_TS`` are the relaxed
initial-state minimum and transition-state saddle. By the chain rule::

    dE_a/dλ = [∂E/∂λ + ∂E/∂R · dR/dλ]_TS − [∂E/∂λ + ∂E/∂R · dR/dλ]_IS

At **converged** geometries both endpoints are stationary — the IS is a minimum
and the TS is a first-order saddle, so ``∂E/∂R = ∇_R E = 0`` at each. This is the
Hellmann–Feynman / envelope theorem: the geometry-response term
``∂E/∂R · dR/dλ`` vanishes *despite* ``dR/dλ ≠ 0`` (the relaxed geometry does
move with λ), because the energy is flat in R at a stationary point. Hence

    dE_a/dλ = ∂E_TS/∂λ − ∂E_IS/∂λ                                       (envelope)

obtainable as ONE autograd backward w.r.t. λ at each *fixed* converged geometry
— instead of finite-differencing entire NEB re-runs at λ ± δ. That is the
flagship: the sensitivity of a barrier to a Hamiltonian parameter for the cost
of two single-point parameter gradients.

Two regimes are supported:

- **Canonical** (fixed electron count N): λ is an XC coefficient
  (``barrier_parameter_sensitivity(..., xc=...)`` via
  :func:`gradwave.core.xc.learnable.energy_param_grads`) or a Hubbard U
  (``..., manifolds=...`` via
  :func:`gradwave.postscf.hubbard_u.energy_derivative_u`). Both param-gradient
  routes are themselves envelope-theorem results *in the electronic variables*
  (dE/dθ = ∂E_xc/∂θ at fixed converged density; dE/dU = ½Tr[n(1−n)] at fixed
  converged occupations), so this composes two envelope theorems: one over the
  density, one over the nuclei.

- **Grand-canonical** (constant-µ ESM electrode, ``boundary='open_z_metal'``,
  ``target_mu`` set): N floats and the barrier lives in the grand potential
  ``Ω = E − µN``. The grand-canonical Hellmann–Feynman relation ``dΩ/dµ = −N``
  is exact at a state stationary in geometry *and* electron number, so

      dΩ_a/dµ = −(N_TS − N_IS) = −ΔN‡,   dΩ_a/dU = +ΔN‡

  (U = −µ up to the reference offset, e = 1 in eV/V units). The barrier's
  dependence on electrode potential is governed entirely by the excess electron
  count of the TS relative to the IS — the flagship ``dE_a/dU``
  (:func:`barrier_potential_sensitivity`). It composes with the thermodynamic
  ``dΔG/dU`` into the full ``d(activation free energy)/dU`` of electrocatalysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gradwave.core.hubbard import HubbardManifold
    from gradwave.core.xc.base import XCFunctional
    from gradwave.core.xc.spin import SpinXC


@dataclass
class BarrierSensitivity:
    """dE_a/dλ for a canonical (fixed-N) barrier, plus the pieces it came from.

    ``dEa_dtheta`` maps each XC parameter name to ∂E_TS/∂θ − ∂E_IS/∂θ [eV per
    the parameter's unit]; ``dEa_dU`` is ∂E_TS/∂U − ∂E_IS/∂U [dimensionless,
    eV/eV] summed over Hubbard manifolds. ``grad_is``/``grad_ts`` expose the
    per-endpoint gradients so a caller can see which endpoint drives the
    sensitivity.
    """

    barrier_eV: float
    energy_is_eV: float
    energy_ts_eV: float
    dEa_dtheta: dict[str, float] = field(default_factory=dict)
    dEa_dU: float | None = None
    grad_is: dict[str, float] = field(default_factory=dict)
    grad_ts: dict[str, float] = field(default_factory=dict)
    grad_is_U: float | None = None
    grad_ts_U: float | None = None


@dataclass
class PotentialSensitivity:
    """dΩ_a/dµ and dE_a/dU for a grand-canonical (constant-µ) barrier.

    ``domega_a_dmu = −ΔN`` is the exact grand-canonical envelope derivative;
    ``dEa_dU = +ΔN`` is the same result on the electrode-potential axis
    (U = −µ, e = 1). ``delta_N`` is N_TS − N_IS, the excess electron count of the
    transition state — the physical driver of the potential dependence.
    ``barrier_omega_eV`` is the grand-potential barrier Ω_TS − Ω_IS at the shared
    µ.
    """

    barrier_omega_eV: float
    omega_is_eV: float
    omega_ts_eV: float
    mu_eV: float
    n_is: float
    n_ts: float
    delta_N: float
    domega_a_dmu: float
    dEa_dU: float


def _require_same_geometry_count(res_is, res_ts) -> None:
    n_is = len(res_is.system.positions)
    n_ts = len(res_ts.system.positions)
    if n_is != n_ts:
        raise ValueError(
            f"IS has {n_is} atoms but TS has {n_ts}; the two endpoints of a "
            "barrier must share atom count and order")


def barrier_parameter_sensitivity(
    res_is,
    res_ts,
    *,
    xc: XCFunctional | SpinXC | None = None,
    manifolds: list[HubbardManifold] | None = None,
) -> BarrierSensitivity:
    """dE_a/dλ = ∂E_TS/∂λ − ∂E_IS/∂λ at converged IS/TS geometries.

    ``res_is`` (a relaxed minimum) and ``res_ts`` (a relaxed first-order saddle)
    are converged :class:`~gradwave.scf.loop.SCFResult`\\ s — the endpoints of a
    NEB run or supplied directly. Their geometries must be *stationary*
    (∇_R E ≈ 0) for the envelope theorem to hold; converge forces tightly.

    Pass ``xc`` (the same functional the SCFs ran with, carrying learnable
    parameters) to get ∂E_a/∂θ for each XC coefficient, and/or ``manifolds`` (the
    DFT+U Hubbard manifolds) to get ∂E_a/∂U. At least one must be given.

    Returns a :class:`BarrierSensitivity`. The gradients are single autograd
    backward passes at fixed geometry — no NEB re-run, no finite difference.
    """
    if xc is None and manifolds is None:
        raise ValueError(
            "give xc= (for dE_a/dθ over XC coefficients) and/or manifolds= "
            "(for dE_a/dU over Hubbard manifolds)")
    _require_same_geometry_count(res_is, res_ts)

    e_is = float(res_is.energies.total)
    e_ts = float(res_ts.energies.total)
    out = BarrierSensitivity(
        barrier_eV=e_ts - e_is, energy_is_eV=e_is, energy_ts_eV=e_ts)

    if xc is not None:
        from gradwave.core.xc.learnable import energy_param_grads

        g_is = energy_param_grads(res_is, xc)
        g_ts = energy_param_grads(res_ts, xc)
        for name in g_ts:
            gi = g_is.get(name)
            gt = g_ts.get(name)
            if gi is None or gt is None:  # allow_unused: an inactive parameter
                continue
            fi, ft = float(gi), float(gt)
            out.grad_is[name] = fi
            out.grad_ts[name] = ft
            out.dEa_dtheta[name] = ft - fi

    if manifolds is not None:
        from gradwave.postscf.hubbard_u import energy_derivative_u

        gi = energy_derivative_u(res_is, manifolds)
        gt = energy_derivative_u(res_ts, manifolds)
        out.grad_is_U = gi
        out.grad_ts_U = gt
        out.dEa_dU = gt - gi

    return out


def grand_potential(res) -> float:
    """Ω = F − µN [eV] for a constant-µ (grand-canonical) SCF result.

    F = ``res.energies.free_energy`` (E − σS) and µ = ``res.fermi`` (the fixed
    electrode Fermi level under ``target_mu``); N = ``res.n_electrons`` floats.
    """
    return float(res.energies.free_energy) - float(res.fermi) * float(res.n_electrons)


def barrier_potential_sensitivity(res_is, res_ts) -> PotentialSensitivity:
    """dΩ_a/dµ = −ΔN and dE_a/dU = +ΔN for a constant-µ barrier.

    ``res_is`` and ``res_ts`` are converged constant-µ (``boundary=
    'open_z_metal'``, ``target_mu`` set) results at the **same** µ. By the
    grand-canonical Hellmann–Feynman relation ``dΩ/dµ = −N`` (exact at a state
    stationary in geometry and electron number), the barrier's potential
    sensitivity is fixed by ΔN = N_TS − N_IS, the excess charge of the transition
    state — no differentiation through the constant-µ band relaxation.
    """
    _require_same_geometry_count(res_is, res_ts)
    mu_is, mu_ts = float(res_is.fermi), float(res_ts.fermi)
    if abs(mu_is - mu_ts) > 1e-6:
        raise ValueError(
            f"IS µ={mu_is:.6f} eV and TS µ={mu_ts:.6f} eV differ; a "
            "grand-canonical barrier compares the two endpoints at the SAME "
            "electrode potential (target_mu)")
    n_is = float(res_is.n_electrons)
    n_ts = float(res_ts.n_electrons)
    delta_N = n_ts - n_is
    om_is = grand_potential(res_is)
    om_ts = grand_potential(res_ts)
    return PotentialSensitivity(
        barrier_omega_eV=om_ts - om_is,
        omega_is_eV=om_is,
        omega_ts_eV=om_ts,
        mu_eV=mu_is,
        n_is=n_is,
        n_ts=n_ts,
        delta_N=delta_N,
        domega_a_dmu=-delta_N,
        dEa_dU=delta_N,
    )
