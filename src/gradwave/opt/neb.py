"""Nudged-elastic-band force projector (improved tangent + climbing image).

A *pure function* over a band of images: given each image's Cartesian
positions, potential energy and true (PES) force, it returns the projected NEB
force that a band optimizer descends. It contains no SCF, no ASE, no autograd —
just the Henkelman band math — so it can be validated cheaply against analytic
2D surfaces (Müller–Brown, LEPS) where the saddle is known in closed form, and
later reused as the differentiable inner kernel of a ``dE_a/dλ`` flagship.

The formulation is the *improved tangent* of Henkelman & Jónsson,
J. Chem. Phys. 113, 9978 (2000), with the *climbing image* of Henkelman,
Uberuaga & Jónsson, J. Chem. Phys. 113, 9901 (2000): the highest-energy interior
image has its spring force removed and its true parallel force inverted, so it
climbs the potential along the band onto the saddle while the remaining images
bracket the minimum-energy path.

Conventions
-----------
* ``forces`` are the *true* forces −∂E/∂R (what an ASE calculator returns), not
  gradients.
* The two endpoints are held fixed; their returned NEB force is exactly zero.
* ``spring_k`` may be a scalar (uniform springs) or a length-``n_images``
  sequence read as the per-image spring constant ``k_i`` on the spring joining
  image ``i-1`` and ``i`` (the standard variable-spring form).
"""

from __future__ import annotations

import numpy as np

__all__ = ["neb_forces", "neb_tangent"]


def _as_band(a: np.ndarray) -> np.ndarray:
    """Coerce a band array to float64 ``(n_images, n_dof)`` (flattening atoms)."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 3:  # (n_images, n_atoms, 3)
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 2:  # (n_images, n_dof)
        return arr
    raise ValueError(
        f"band array must be (n_images, n_atoms, 3) or (n_images, n_dof), "
        f"got shape {arr.shape}")


def neb_tangent(
    positions: np.ndarray, energies: np.ndarray
) -> np.ndarray:
    """Improved (energy-weighted) unit tangents for every interior image.

    Returns ``(n_images, n_dof)`` with the endpoint rows zero. The tangent at an
    interior image ``i`` follows the uphill neighbour when the image sits on a
    monotonic stretch of the band, and blends both neighbour segments weighted by
    the larger/smaller energy difference at a local extremum — the construction
    that removes the artificial kinks the plain bisector tangent produces near
    the barrier top.
    """
    R = _as_band(positions)
    E = np.asarray(energies, dtype=np.float64).reshape(-1)
    n = R.shape[0]
    if E.shape[0] != n:
        raise ValueError(
            f"energies length {E.shape[0]} != n_images {n}")
    tau = np.zeros_like(R)
    for i in range(1, n - 1):
        tau_plus = R[i + 1] - R[i]     # segment to the next image
        tau_minus = R[i] - R[i - 1]    # segment from the previous image
        e_next, e_here, e_prev = E[i + 1], E[i], E[i - 1]
        if e_next > e_here > e_prev:
            t = tau_plus
        elif e_next < e_here < e_prev:
            t = tau_minus
        else:
            # local maximum or minimum along the band: energy-weighted blend
            dv_max = max(abs(e_next - e_here), abs(e_prev - e_here))
            dv_min = min(abs(e_next - e_here), abs(e_prev - e_here))
            if e_next > e_prev:
                t = tau_plus * dv_max + tau_minus * dv_min
            else:
                t = tau_plus * dv_min + tau_minus * dv_max
        norm = np.linalg.norm(t)
        if norm > 0.0:
            t = t / norm
        tau[i] = t
    return tau


def neb_forces(
    positions_images: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    spring_k: float | np.ndarray = 0.1,
    *,
    climb: bool = True,
    climb_index: int | None = None,
) -> np.ndarray:
    """Projected NEB force for every image of a band (endpoints zeroed).

    Parameters
    ----------
    positions_images
        ``(n_images, n_atoms, 3)`` or ``(n_images, n_dof)`` Cartesian positions.
    energies
        ``(n_images,)`` potential energy of each image.
    forces
        True forces −∂E/∂R, same shape as ``positions_images``.
    spring_k
        Scalar spring constant, or a per-image sequence ``k_i`` (the spring
        between image ``i-1`` and ``i``).
    climb
        When true, the highest-energy interior image climbs (spring force
        dropped, parallel true force inverted). When false, a plain CI-free NEB.
    climb_index
        Force a specific interior image to climb; default is the argmax of
        ``energies`` over the interior images.

    Returns
    -------
    np.ndarray
        Projected forces, the same shape as ``forces``. Rows 0 and ``n_images-1``
        (the fixed endpoints) are exactly zero.

    Notes
    -----
    For an interior, non-climbing image ``i`` with unit tangent ``τ̂``:

        F_i^NEB = (F_i − (F_i·τ̂) τ̂)  +  (k_{i+1}|R_{i+1}−R_i| − k_i|R_i−R_{i-1}|) τ̂

    the first term is the true force with its band-parallel part removed, the
    second the spring force projected onto the tangent (improved-tangent spring,
    which suppresses corner-cutting). The climbing image instead feels

        F_c = F_c − 2 (F_c·τ̂) τ̂

    with no spring contribution.
    """
    R = _as_band(positions_images)
    F = _as_band(forces)
    E = np.asarray(energies, dtype=np.float64).reshape(-1)
    n = R.shape[0]
    if F.shape != R.shape:
        raise ValueError(
            f"forces shape {F.shape} != positions shape {R.shape}")
    if n < 3:
        # only endpoints (or fewer): nothing to relax
        return np.zeros_like(forces, dtype=np.float64)

    k = np.asarray(spring_k, dtype=np.float64)
    if k.ndim == 0:
        k = np.full(n, float(k))
    elif k.shape[0] != n:
        raise ValueError(
            f"spring_k sequence length {k.shape[0]} != n_images {n}")

    tau = neb_tangent(R, E)

    ci = None
    if climb:
        ci = int(climb_index) if climb_index is not None \
            else 1 + int(np.argmax(E[1:-1]))

    out = np.zeros_like(R)
    for i in range(1, n - 1):
        t = tau[i]
        f_par = float(F[i] @ t)          # true force along the tangent
        if i == ci:
            # climbing image: invert the parallel component, no spring
            out[i] = F[i] - 2.0 * f_par * t
            continue
        f_perp = F[i] - f_par * t        # true force perpendicular to the band
        # improved-tangent spring: k(|τ⁺| − |τ⁻|) along τ̂ (variable-k form)
        d_plus = np.linalg.norm(R[i + 1] - R[i])
        d_minus = np.linalg.norm(R[i] - R[i - 1])
        f_spring = (k[i + 1] * d_plus - k[i] * d_minus) * t
        out[i] = f_perp + f_spring

    return out.reshape(np.asarray(forces).shape)
