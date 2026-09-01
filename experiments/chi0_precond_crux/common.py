"""Shared builders for the χ₀/dielectric quasi-Newton preconditioning crux.

This module wires ONLY shipped machinery (no new physics):

- system builders: fcc-Al bulk, Al(100) slabs at fixed vacuum, bcc-Fe FM
  (nspin=2) — the growing-inhomogeneous and stiff-magnetic series;
- ``ExactDielectricPrecond``: the "teacher" preconditioner. It applies the
  EXACT inverse dielectric ε_ρ⁻¹ = (1 − χ₀·K_Hxc)⁻¹ to the SCF density
  residual, i.e. one Newton step of the outer fixed point. χ₀ is the shipped
  ``scf.implicit.apply_chi0`` (a full Sternheimer solve), K_Hxc the shipped
  ``apply_k_hxc``; the inverse is a shipped ``anderson_solve`` of the density-
  space dielectric system. A truncated/sketched student (the WoodburyPrecond
  M1 would build) can never beat this exact operator, so the A/B iteration
  cut it produces BOUNDS the whole low-rank cluster.

The density Newton derivation: the SCF residual map is g(ρ) = F(ρ) − ρ with
Jacobian g' = J_ρ − I, J_ρ = χ₀·K_Hxc (density → potential via K_Hxc, then
potential → density via χ₀). The Newton step is δρ = −g'⁻¹ g = (I − χ₀K_Hxc)⁻¹ R,
and ``anderson_solve(m_apply, R)`` with ``m_apply = χ₀∘K_Hxc`` returns exactly
that. Applied as ``PulayMixer.precond_op`` it replaces Kerker/local-TF's
approximate ε⁻¹ with the exact one.

nspin=2: the mixer packs grid channels in the (total, magnetization) basis
(``scf.layout.MixLayout``) while χ₀/K_Hxc act in the (up, down) basis. The
conversion is done by reusing ``MixLayout.unpack``/``pack`` verbatim (the same
FFT normalization the SCF itself uses), so the operator is applied by a
similarity transform T·P_updn·T⁻¹ that is correct by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase.build import fcc100

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.implicit import apply_chi0, apply_k_hxc
from gradwave.scf.layout import MixLayout
from gradwave.scf.loop import setup_system
from gradwave.solvers.deflation import anderson_solve

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]  # worktree/repo root
PSEUDO = _ROOT / "tests" / "fixtures" / "qe" / "pseudos"


# --------------------------------------------------------------------------
# System builders (shipped setup_system; use_symmetry=False so apply_chi0 is
# unambiguously valid on a general — non-totally-symmetric — residual field).
# --------------------------------------------------------------------------
def al_bulk_system(ecut_ry: float = 25.0, kmesh=(8, 8, 8),
                   use_symmetry: bool = False):
    upf = parse_upf(PSEUDO / "Al_ONCV_PBE-1.2.upf")
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    system = setup_system(cell, np.array([[0.0, 0, 0]]), [0], [upf],
                          ecut=ecut_ry * RY, kmesh=kmesh, nbands=12,
                          use_symmetry=use_symmetry)
    return system, 1


def al_slab_system(nlayers: int, ecut_ry: float = 25.0, kmesh=(4, 4, 1),
                   vacuum: float = 8.0, use_symmetry: bool = False):
    """Al(100) slab, ``nlayers`` layers at FIXED vacuum — the growing
    inhomogeneous series (bulk-like interior + two metal/vacuum interfaces)."""
    upf = parse_upf(PSEUDO / "Al_ONCV_PBE-1.2.upf")
    slab = fcc100("Al", size=(1, 1, nlayers), a=4.05, vacuum=vacuum)
    natoms = len(slab)
    system = setup_system(np.array(slab.cell), slab.get_positions(),
                          [0] * natoms, [upf], ecut=ecut_ry * RY, kmesh=kmesh,
                          nbands=6 * natoms, use_symmetry=use_symmetry)
    return system, natoms


def fe_fm_system(a: float = 2.87, ecut_ry: float = 45.0, kmesh=(4, 4, 4),
                 use_symmetry: bool = False):
    """bcc Fe, collinear ferromagnet (nspin=2). The stiff-magnetic case: the
    coupled Stoner mode limits convergence and an LDOS/scalar preconditioner
    cannot represent it."""
    fe = parse_upf(PSEUDO / "Fe_ONCV_PBE-1.2.upf")
    cell = a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    system = setup_system(cell, np.zeros((1, 3)), [0], [fe], ecut=ecut_ry * RY,
                          kmesh=kmesh, nbands=12, use_symmetry=use_symmetry)
    return system, 1


# --------------------------------------------------------------------------
# The exact-dielectric "teacher" preconditioner.
# --------------------------------------------------------------------------
class ExactDielectricPrecond:
    """P·r = (1 − χ₀·K_Hxc)⁻¹ r — one exact Newton step of the outer SCF.

    Frozen at a reference ``res`` (a converged SCFResult): χ₀ and K_Hxc are
    built from that fixed diagonalization, so this is the strongest possible
    dielectric preconditioner (exact at the fixed point). Any low-rank student
    approximates THIS operator, so the iteration cut it delivers is the ceiling
    for the whole cluster.

    Set as ``scf(..., precond_op=<this>)``. ``acts_on = "grid"`` makes the SCF
    hand it the whole grid block ([total, mag] for nspin=2), so BOTH the charge
    and the coupled magnetization response are preconditioned (a charge-only
    filter cannot represent the Stoner mode — that is the point).
    """

    acts_on = "grid"

    def __init__(self, res, xc, *, chi0_tol: float = 1e-6, chi0_max_iter: int = 200,
                 inner_tol: float = 1e-6, inner_max_iter: int = 30,
                 beta: float = 0.4, history: int = 8) -> None:
        self.res = res
        self.xc = xc
        self.nspin = int(getattr(res, "nspin", 1))
        self.chi0_tol = chi0_tol
        self.chi0_max_iter = chi0_max_iter
        self.inner_tol = inner_tol
        self.inner_max_iter = inner_max_iter
        self.beta = beta
        self.history = history
        # reconstruct the SCF's own packed-vector layout (nbec=0, NC path)
        self.layout = MixLayout(res.system.grid, self.nspin, [])
        # telemetry
        self.n_calls = 0
        self.inner_iters: list[int] = []

    def _m_apply(self, z: torch.Tensor) -> torch.Tensor:
        """Density-space Jacobian J_ρ·z = χ₀·(K_Hxc·z)."""
        khxc = apply_k_hxc(self.res, self.xc, z)
        return apply_chi0(self.res, khxc, tol=self.chi0_tol,
                          max_iter=self.chi0_max_iter)

    def __call__(self, r_grid: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        # (total, mag) sphere residual → per-spin real-space density residual
        rho_spin, _ = self.layout.unpack(r_grid)
        vbar = (rho_spin[0] if self.nspin == 1
                else torch.stack([rho_spin[0], rho_spin[1]]))
        sol = anderson_solve(self._m_apply, vbar, beta=self.beta,
                             tol=self.inner_tol, max_iter=self.inner_max_iter,
                             history=self.history)
        self.inner_iters.append(sol.n_iter)
        out = [sol.u] if self.nspin == 1 else [sol.u[0], sol.u[1]]
        # per-spin real-space Newton step → (total, mag) sphere vector
        return self.layout.pack(out)


def xc_for(nspin: int):
    """The functional used both in the SCF and the response operator."""
    return SpinPBE() if nspin == 2 else PBE()
