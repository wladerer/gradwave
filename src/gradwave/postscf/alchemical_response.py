"""Composition derivatives of response functions (the second-order tier).

The first-order alchemical gradients (energy, band gap, arbitrary density
functional; ``scf.alchemical``) each ride a single first-order response dρ/dλ. A
RESPONSE function — the macroscopic dielectric tensor ε∞, the Born effective
charges Z* — is itself a second derivative of the energy (ε∞ ∝ d²E/dℰ²), so its
composition derivative dε∞/dλ is a mixed THIRD-order derivative d³E/(dλ dℰ²).

``alchemical_dielectric_gradient`` gives it by a central finite difference of the
E-field DFPT (``postscf.dielectric.dielectric_born``) along the composition path.
For a third-order quantity this is the reliable and standard route: the analytic
mixed derivative is a 2n+1 build carrying a third functional derivative of E_xc
(the ∂_λ K_Hxc term) and the subtle periodic-field position operator, and is a
documented follow-up (docs/ideas.md). The finite difference is the ground truth
that analytic form would be validated against.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def alchemical_dielectric_gradient(
    converge: Callable[[float], Any],
    xc: Any,
    lam: float,
    *,
    h: float = 0.02,
    **dielectric_kw: Any,
) -> dict[str, torch.Tensor | float | dict[str, Any]]:
    """d(ε∞, Z*)/dλ: the composition derivative of the macroscopic dielectric
    tensor and the Born effective charges, by central FD of the E-field DFPT.

    ``converge`` is a callable ``lam -> converged SCFResult`` — an alchemical
    substitution SCF (nspin=1 insulator, use_symmetry=False, the dielectric_born
    regime). ``xc`` is the functional, ``lam`` the point to differentiate at,
    ``h`` the FD step in composition. Extra keyword args pass through to
    ``dielectric_born`` (``dk``, ``cg_tol``, ``max_outer``, ...).

    Returns a dict:
      ``d_eps``     (3,3)     dε∞/dλ [1/λ]
      ``d_eps_iso`` float     d(⅓ Tr ε∞)/dλ
      ``d_born``    (na,3,3)  dZ*/dλ [e/λ]
    and the two endpoint dielectric results under ``plus``/``minus`` for
    inspection (h-convergence, the underlying ε∞ values).
    """
    from gradwave.postscf.dielectric import dielectric_born

    rp = dielectric_born(converge(lam + h), xc, **dielectric_kw)
    rm = dielectric_born(converge(lam - h), xc, **dielectric_kw)
    two_h = 2.0 * float(h)
    return {
        "d_eps": (rp["eps"] - rm["eps"]) / two_h,
        "d_eps_iso": (float(rp["eps_iso"]) - float(rm["eps_iso"])) / two_h,
        "d_born": (rp["born"] - rm["born"]) / two_h,
        "plus": rp,
        "minus": rm,
    }
