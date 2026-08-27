"""Born effective charges Z* by finite differences of the Berry-phase polarization.

The Born (dynamical) effective-charge tensor of atom ``kappa`` is the mixed
derivative of the macroscopic polarization with respect to that atom's
displacement,

    Z*_{kappa, a b} = (V / e) dP_a / du_{kappa b},

evaluated at zero macroscopic field. Here P comes from the King-Smith--Vanderbilt
Berry phase (:mod:`gradwave.postscf.polarization`), so Z* is obtained by
displacing each atom by +/- a small step along each Cartesian axis, recomputing
the reduced polarization, and taking the central difference. Because the
displacements are tiny the polarization branch is continuous: each displaced run
is unwrapped onto the branch of the undisplaced reference (per reciprocal
direction), so no polarization quantum is crossed.

The acoustic sum rule sum_kappa Z*_kappa = 0 (a rigid translation of all atoms
carries no charge) is reported as a diagnostic and can optionally be enforced by
subtracting the mean.

Insulators only, nspin in {1, 2}. This is the finite-difference route; an
E-field DFPT alternative lives in :mod:`gradwave.postscf.dielectric`
(``dielectric_born``), and an autograd probe is provided by
:func:`born_charges_autograd`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from gradwave.dtypes import RDTYPE
from gradwave.postscf.polarization import Polarization, berry_phase_polarization

if TYPE_CHECKING:
    from gradwave.scf.loop import SCFResult


def _polarization_at(
    scf_fn: Callable[[np.ndarray], SCFResult],
    positions: np.ndarray,
    mesh: tuple[int, int, int],
    ref_phases: Tensor | None,
) -> Polarization:
    res = scf_fn(positions)
    return berry_phase_polarization(res, mesh, unwrap_reference=ref_phases)


def born_effective_charges(
    scf_fn: Callable[[np.ndarray], SCFResult],
    positions: np.ndarray,
    mesh: tuple[int, int, int],
    *,
    step: float = 2.0e-3,
    enforce_asr: bool = False,
    verbose: bool = False,
) -> dict[str, Tensor]:
    """Born effective charges Z* (na, 3, 3) by central finite differences.

    ``scf_fn(positions)`` runs a converged insulating SCF at the given Cartesian
    positions [Angstrom] on the FULL unshifted ``mesh`` (``use_symmetry=False``)
    and returns its :class:`~gradwave.scf.loop.SCFResult`. ``step`` is the
    displacement [Angstrom] (central difference, so each atom/axis costs two
    SCFs). The polarization of every displaced run is unwrapped onto the
    undisplaced reference branch, so the quantum is never crossed.

    Returns a dict with:

    - ``born`` (na, 3, 3): ``Z*_{kappa, a b} = V dP_a / du_{kappa b}`` (a =
      polarization/Cartesian component, b = displacement axis), dimensionless
      (units of e).
    - ``asr`` (3, 3): the acoustic-sum-rule residual ``sum_kappa Z*_kappa``.
    - ``asr_max`` (0-dim): the largest absolute ASR component (a scalar
      diagnostic of self-consistency / k-mesh convergence).

    With ``enforce_asr`` the residual is spread equally over the atoms so the
    sum is exactly zero.
    """
    positions = np.asarray(positions, dtype=float)
    na = positions.shape[0]

    ref = _polarization_at(scf_fn, positions, mesh, None)
    volume = ref.volume
    ref_phases = ref.berry_phases

    born = torch.zeros(na, 3, 3, dtype=RDTYPE)
    for kappa in range(na):
        for axis in range(3):
            pos_p = positions.copy()
            pos_m = positions.copy()
            pos_p[kappa, axis] += step
            pos_m[kappa, axis] -= step
            pol_p = _polarization_at(scf_fn, pos_p, mesh, ref_phases)
            pol_m = _polarization_at(scf_fn, pos_m, mesh, ref_phases)
            # dP_a/du : central difference of the physical vector P/e (1/Ang^2)
            dp = (pol_p.vector() - pol_m.vector()) / (2.0 * step)  # (3,) over a
            born[kappa, :, axis] = volume * dp
            if verbose:
                print(f"  Z* atom {kappa} axis {axis} done", flush=True)

    asr = born.sum(dim=0)  # (3, 3)
    asr_max = asr.abs().max()
    if enforce_asr:
        born = born - asr.unsqueeze(0) / na
    return {"born": born, "asr": asr, "asr_max": asr_max}


def born_charges_autograd(
    polarization_of_positions: Callable[[Tensor], Tensor],
    positions: Tensor,
    volume: float,
) -> Tensor:
    """Z* (na, 3, 3) by reverse-mode autograd of P through the SCF (experimental).

    ``polarization_of_positions(pos)`` must return the physical polarization
    vector ``P / e`` (3,) [1 / Angstrom^2] as a differentiable function of the
    (na, 3) Cartesian ``positions`` tensor (``requires_grad=True``). The Jacobian
    ``dP_a/du_{kappa b}`` is taken by three backward passes (one per Cartesian
    component of P), giving ``Z*_{kappa, a b} = V dP_a/du_{kappa b}``.

    This is the documented autograd bonus: the finite-difference
    :func:`born_effective_charges` is the validated deliverable. Differentiating
    the Berry phase through ``torch.linalg.slogdet`` and the SCF fixed point is
    only meaningful when the whole polarization path was built on live (grad-
    carrying) coefficients; use for small cells / cross-checks.
    """
    pos = positions.detach().clone().to(RDTYPE).requires_grad_(True)
    p_vec = polarization_of_positions(pos)  # (3,)
    na = pos.shape[0]
    born = torch.zeros(na, 3, 3, dtype=RDTYPE)
    for a in range(3):
        grad = torch.autograd.grad(p_vec[a], pos, retain_graph=(a < 2))[0]  # (na,3)
        born[:, a, :] = volume * grad
    return born
