"""Differentiable composition design (phase 5).

Fits a differentiable surrogate E(lambda) over per-site composition weights and
optimizes composition toward a target property. The surrogate is a cluster
expansion truncated at pairs, a quadratic in the site weights, so it is exact for
a quadratic landscape and a controlled local model otherwise. Gradient
supervision makes each DFT sample carry 1 + na constraints, the energy and the
per-site alchemical gradient, so the fit is data-efficient.

The samples come from the rigorous engine, setup_alchemical_system plus scf plus
alchemical_energy_gradient. The corners (integer lambda) are real DFT, so this
never averages a potential and is not the virtual crystal approximation.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.dtypes import RDTYPE


class CompositionSurrogate:
    """Quadratic pair-cluster surrogate E(lam) = c0 + b . lam + lam . A . lam,
    differentiable in the per-site weights lam."""

    def __init__(self, c0: float | np.float64, b: np.ndarray, A: np.ndarray) -> None:
        self.c0 = torch.as_tensor(c0, dtype=RDTYPE)
        self.b = torch.as_tensor(b, dtype=RDTYPE)
        self.A = torch.as_tensor(A, dtype=RDTYPE)  # symmetric (na, na)

    def energy(self, lam: torch.Tensor | np.ndarray) -> torch.Tensor:
        lam = torch.as_tensor(lam, dtype=RDTYPE)
        return self.c0 + self.b @ lam + lam @ self.A @ lam

    def gradient(self, lam: np.ndarray) -> torch.Tensor:
        lam_t = torch.as_tensor(lam, dtype=RDTYPE)
        return self.b + 2.0 * (self.A @ lam_t)


def _symmetric_from_upper(upper: np.ndarray, na: int,
                          iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Symmetric (na, na) matrix from its upper-triangle entries `upper` laid out
    on the index pair `iu = np.triu_indices(na)`."""
    A = np.zeros((na, na))
    A[iu] = upper
    return A + A.T - np.diag(np.diag(A))


def _model(params: np.ndarray, lam: np.ndarray, na: int,
           iu: tuple[np.ndarray, np.ndarray]) -> tuple[np.float64, np.ndarray]:
    """Evaluate the quadratic model and its per-site gradient for a flat
    parameter vector [c0, b(na), A_upper]. Linear in params, so the same routine
    with unit parameter vectors builds the least-squares design matrix."""
    c0 = params[0]
    b = params[1:1 + na]
    A = _symmetric_from_upper(params[1 + na:], na, iu)
    e = c0 + b @ lam + lam @ A @ lam
    g = b + 2.0 * (A @ lam)
    return e, g


def fit_surrogate(lams: np.ndarray, energies: list[float],
                  grads: list[np.ndarray] | None=None) -> CompositionSurrogate:
    """Least-squares fit of the quadratic surrogate. lams is (m, na), energies is
    (m,), and grads is an optional (m, na) of per-site dE/dlambda. Gradient rows
    add na equations per sample, so a couple of samples pin the model."""
    lams = np.asarray(lams, dtype=float)
    en = np.asarray(energies, dtype=float)
    m, na = lams.shape
    iu = np.triu_indices(na)
    n_par = 1 + na + len(iu[0])

    design, rhs = [], []
    for s in range(m):
        e_cols, g_cols = [], []
        for p in range(n_par):
            unit = np.zeros(n_par)
            unit[p] = 1.0
            e_p, g_p = _model(unit, lams[s], na, iu)
            e_cols.append(e_p)
            g_cols.append(g_p)
        design.append(e_cols)
        rhs.append(en[s])
        if grads is not None:
            g_arr = np.asarray(grads, dtype=float)
            for k in range(na):
                design.append([g_cols[p][k] for p in range(n_par)])
                rhs.append(g_arr[s, k])

    params, *_ = np.linalg.lstsq(np.asarray(design), np.asarray(rhs), rcond=None)
    A = _symmetric_from_upper(params[1 + na:], na, iu)
    return CompositionSurrogate(params[0], params[1:1 + na], A)


def optimize_composition(surrogate: CompositionSurrogate, x0: np.ndarray,
                         target: float | None=None, bounds: tuple[float, float]=(0.0, 1.0),
                         steps: int=400, lr: float=0.05) -> torch.Tensor:
    """Optimize the per-site composition on the surrogate. With target=None the
    energy is minimized, otherwise (E - target)^2 is minimized. lam is projected
    to the bounds each step. Returns the optimal per-site weights."""
    lam = torch.as_tensor(x0, dtype=RDTYPE).clone().requires_grad_(True)
    opt = torch.optim.Adam([lam], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        e = surrogate.energy(lam)
        loss = e if target is None else (e - float(target)) ** 2
        loss.backward()
        opt.step()
        with torch.no_grad():
            lam.clamp_(bounds[0], bounds[1])
    return lam.detach()


def sample_alchemical(cell, positions, upf_a, upf_b, lam, xc, *, ecut,
                      scf_kwargs=None, **setup_kwargs):
    """One DFT sample for the surrogate fit, the energy and per-site alchemical
    gradient at composition lam. Bridges the rigorous engine to the design loop."""
    from gradwave.scf.alchemical import alchemical_energy_gradient, setup_alchemical_system
    from gradwave.scf.loop import scf

    system = setup_alchemical_system(cell, positions, upf_a, upf_b, lam,
                                     ecut=ecut, **setup_kwargs)
    res = scf(system, xc, **(scf_kwargs or {}))
    e = float(res.energies.free_energy)
    grad = alchemical_energy_gradient(res, lam, xc=xc).numpy()
    return e, grad
