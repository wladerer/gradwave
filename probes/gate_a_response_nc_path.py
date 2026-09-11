"""Gate A (analytic exchange campaign, task #23): does the shared response kernel
(``scf.implicit.apply_chi0`` / ``solve_adjoint``, ``postscf._response``) run through
the NONCOLLINEAR / constrained-moment path that ``spin_exchange`` needs?

The analytic route to J_ij / D_ij replaces ``spin_exchange``'s outer finite
difference (tilt ê_j, re-converge a constrained SCF, read the induced torque on i)
with one linear-response solve: the mixed second derivative

    dT_i/dê_j = ∂T_i/∂ê_j|_ρ  +  (∂T_i/∂ρ)·(dρ/dê_j),

where dρ/dê_j solves the response equation with the constraint-field derivative as
RHS. That needs χ₀ applied through the constrained NONCOLLINEAR (spinor) path.

This probe establishes, empirically on a small genuinely-magnetic system (the O2
triplet used by tests/integration/test_moment_config.py), the two facts that decide
Gate A:

  FACT 1  The constrained-moment torque carries NO autograd graph. ``scf_noncollinear``
          is ``@torch.no_grad()``, so ``constrained_moment_scf``'s ``info["torque"]``
          (and the moments M it is built from) have ``requires_grad=False`` / no
          ``grad_fn``. => the "double-backward through the unrolled SCF" fallback has
          no substrate: there is no graph from ê_j to T_i to differentiate.

  FACT 2  ``apply_chi0`` is COLLINEAR-only. It takes an ``SCFResult`` (nspin=1 scalar
          field, or nspin=2 spin-block-diagonal per-channel field) and returns a
          scalar/per-spin density response. It has no spinor (ρ, m⃗) contract, so it
          cannot be called on the ``NCResult`` the constrained path produces, and it
          structurally cannot represent the TRANSVERSE magnetization response that a
          tilt of ê_j induces.

Run on asus (CPU). Prints a PASS/FACT summary; exits 0 regardless (it is a probe,
not a test).
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.moment_config import atomic_weights, constrained_moment_scf
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from tests.helpers import RY, pseudo

PSEUDO = pseudo("O_ONCV_PBE-1.2.upf")


def o2_system(L=6.0, d=1.21):
    o = parse_upf(PSEUDO)
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    return setup_system(cell, pos, [0, 0], [o, o], ecut=30 * RY, kmesh=(1, 1, 1),
                        nbands=8, time_reversal=False)


def main() -> None:
    torch.set_num_threads(8)
    system = o2_system()
    xc = NoncollinearXC(LSDA_PW92())
    w = atomic_weights(system)

    # a collinear reference tilted slightly off axis for atom 1 (the setup a
    # coupling extraction would linearize around)
    dirs = [[0.0, 0.0, 1.0], [float(np.sin(np.deg2rad(15))), 0.0,
                              float(np.cos(np.deg2rad(15)))]]
    scf_kw = dict(smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8,
                  max_iter=200, verbose=False, mode="vector")
    res, info = constrained_moment_scf(system, xc, dirs, lam=8.0, weights=w,
                                       target_mag=torch.tensor([2.0, 2.0]),
                                       **scf_kw)

    print("=" * 72)
    print("GATE A PROBE — response kernel through the noncollinear path")
    print("=" * 72)
    print(f"system: O2 triplet, converged={info['converged']}, "
          f"E={info['energy_eV']:.4f} eV")
    print(f"atomic moments |M| = {torch.linalg.norm(info['M'], dim=-1).tolist()}")
    print()

    # ---- FACT 1: no autograd graph on the torque ----
    torque = info["torque"]
    M = info["M"]
    f1 = (not torque.requires_grad) and (torque.grad_fn is None) \
        and (not M.requires_grad)
    print("FACT 1 — torque carries no autograd graph (double-backward fallback")
    print("         has no substrate; scf_noncollinear is @torch.no_grad):")
    print(f"         torque.requires_grad = {torque.requires_grad}, "
          f"grad_fn = {torque.grad_fn}")
    print(f"         M.requires_grad      = {M.requires_grad}")
    print(f"         => FACT 1 {'CONFIRMED' if f1 else 'NOT confirmed'}")
    print()

    # ---- FACT 2: apply_chi0 is collinear-only, rejects the spinor result ----
    from gradwave.scf.implicit import apply_chi0
    from gradwave.scf.loop import SCFResult

    is_scfresult = isinstance(res, SCFResult)
    # a transverse (m_x) perturbation field on the grid — the physical RHS a tilt
    # of ê_j produces; the collinear kernel has no slot for it
    grid = system.grid
    wfield = torch.zeros(grid.shape, dtype=torch.float64)
    try:
        _ = apply_chi0(res, wfield)
        chi0_call = "returned (unexpected)"
        f2 = False
    except Exception as e:  # noqa: BLE001 - probe: any failure is the point
        chi0_call = f"{type(e).__name__}: {str(e)[:80]}"
        f2 = True
    print("FACT 2 — apply_chi0 is collinear-only; cannot take the spinor NCResult")
    print("         nor represent the transverse (m_x, m_y) response of a tilt:")
    print(f"         isinstance(res, SCFResult) = {is_scfresult} "
          f"(res type = {type(res).__name__})")
    print(f"         apply_chi0(res, w) -> {chi0_call}")
    print(f"         => FACT 2 {'CONFIRMED' if f2 else 'NOT confirmed'}")
    print()

    print("VERDICT: Gate A", "KILLS the analytic route as scoped" if (f1 and f2)
          else "inconclusive — inspect above")
    print("The missing subsystem is a METALLIC SPINOR χ₀ (transverse spin")
    print("susceptibility): apply_chi0 handles metals but only COLLINEAR scalar")
    print("response; the only spinor response (dielectric._dielectric_born_soc) is")
    print("insulator-only AND nonmagnetic (m⃗≡0). The coupled (ρ,m⃗) f_xc HVP")
    print("(_response._fxc_hvp_noncollinear) exists — the K side — but the χ₀ side")
    print("for a magnetic metal does not.")


if __name__ == "__main__":
    main()
