"""Attack 1: gradient-ascent vs random sampling on the FFT eggbox artifact.

Hypothesis under test: because gradwave is differentiable end to end, a
disagreement / artifact functional D(tau) can be MAXIMIZED by gradient ascent to
find worst-case configurations that equal-budget random sampling misses.

Calibration target here is a real, known artifact: the FFT grid breaks continuous
translation invariance, so rigidly translating all atoms by tau within one grid
voxel makes the net force Sum_a F_a deviate from zero (it must be exactly zero for
an isolated rigid translation of a periodic crystal). We take

    D(tau) = |Sum_a F_a|   (net force magnitude, eV/Ang)

as a differentiable function of the rigid shift tau.

Differentiability regime (stated explicitly):
  We RE-CONVERGE the SCF at every evaluation and treat the converged density,
  orbitals and occupations as FIXED (detached) for that evaluation, then take the
  explicit partial derivative d D / d tau through the Hellmann-Feynman energy
  assembly (Ewald + local structure factor + nonlocal projector phases) that
  postscf.forces.forces() itself differentiates. This is exactly the HF
  stationarity regime forces() uses (dE/dtau at fixed psi, rho). Concretely: for a
  rigid shift s, every atom position is pos0_a + s, so
      Sum_a F_a = - Sum_a dE/dpos_a = - dE/ds,
  and D(s) = |dE/ds|. Gradient ascent needs d D / ds, i.e. one more autograd pass
  (a Hessian-vector-like term) through the same detached-density energy. Cost per
  evaluation is therefore ONE SCF (the autograd passes over the frozen-density
  energy are negligible). Equal budget = equal SCF count.

Usage:
    uv run python experiments/adversarial_probe/attack1_eggbox.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import becp, projectors
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import _stack_dij, scf, setup_system

torch.set_num_threads(2)  # shared 8-core laptop; be a good neighbour
torch.manual_seed(0)
RNG = np.random.default_rng(0)

RY = 13.605693122994
FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "qe" / "pseudos"
UPF = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")

# 2-atom Si FCC primitive cell.
A = 5.43
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
CELL = A / 2 * FCC
POS0 = np.array([[0.0, 0.0, 0.0], [A / 4, A / 4, A / 4]])

# DELIBERATELY loose cutoff so the eggbox is visible, symmetry OFF (a rigid shift
# breaks the crystal symmetry, and symmetrize_forces would project out exactly the
# net-force artifact we are trying to measure).
ECUT = 12.0 * RY
KMESH = (2, 2, 2)

SCF_KW = dict(smearing="none", etol=1e-8, rhotol=1e-7, verbose=False, max_iter=100)

_SCF_COUNT = 0
_VOXEL: torch.Tensor | None = None


def _run_scf(pos: np.ndarray):
    global _SCF_COUNT
    _SCF_COUNT += 1
    system = setup_system(
        CELL, pos, [0, 0], [UPF], ecut=ECUT, kmesh=KMESH, use_symmetry=False
    )
    return scf(system, PBE(), **SCF_KW)


def _voxel_matrix(shape) -> torch.Tensor:
    """Rows = real-space voxel vectors cell_i / N_i (one eggbox period per axis)."""
    cell = torch.as_tensor(CELL, dtype=torch.float64)
    n = torch.as_tensor(shape, dtype=torch.float64)
    return cell / n[:, None]


def _ensure_voxel() -> tuple[torch.Tensor, tuple[int, int, int]]:
    global _VOXEL
    probe = setup_system(
        CELL, POS0, [0, 0], [UPF], ecut=ECUT, kmesh=KMESH, use_symmetry=False
    )
    shape = tuple(int(x) for x in probe.grid.shape)
    if _VOXEL is None:
        _VOXEL = _voxel_matrix(shape)
    return _VOXEL, shape


def net_force_value_and_grad(u: np.ndarray, want_grad: bool):
    """One SCF at rigid fractional-voxel shift u in [0,1)^3.

    Returns (D, E_free, grad_D_du_or_None, net_force_vec, fft_shape).
    D = |Sum_a F_a| via the frozen-density HF assembly, differentiable in u.
    """
    V, shape = _ensure_voxel()
    u_t = torch.tensor(u, dtype=torch.float64, requires_grad=True)
    s = u_t @ V  # (3,) Cartesian shift, carries graph on u
    pos0_t = torch.as_tensor(POS0, dtype=torch.float64)
    pos = pos0_t + s  # (na, 3), graph on u

    res = _run_scf(pos.detach().numpy())
    system = res.system
    grid = system.grid

    # --- frozen-density Hellmann-Feynman energy assembly (mirrors forces()) ---
    rho_g = r_to_g(res.rho.detach().to(torch.complex128))
    projs = [projectors(pd, pos) for pd in system.proj_data]
    dij, kw = _stack_dij(system), system.kweights
    coeffs = [c.detach() for c in res.coeffs]  # nspin=1
    occ = res.occupations.detach()
    becps = [becp(projs[ik], coeffs[ik]) for ik in range(len(coeffs))]
    e_nl = nonlocal_energy(becps, dij, occ, kw)
    vloc_g = local_potential_g(
        pos, system.species_index, system.vloc_tables, grid.g_cart, grid.volume
    )
    e_pos = (
        local_energy(rho_g, vloc_g, grid.volume)
        + e_nl
        + ewald_energy(pos, system.charges, grid.cell)
    )

    # net force = -dE/ds = -sum_a dE/dpos_a
    (g_pos,) = torch.autograd.grad(e_pos, pos, create_graph=want_grad)
    net_f = -g_pos.sum(dim=0)  # (3,)
    D = net_f.norm()

    grad_D_du = None
    if want_grad:
        (grad_D_du,) = torch.autograd.grad(D, u_t)
        grad_D_du = grad_D_du.detach().numpy()

    e_free = float(res.energies.free_energy)
    return (float(D.detach()), e_free, grad_D_du,
            net_f.detach().numpy(), shape)


def cross_check_net_force():
    """Confirm the differentiable net-force value equals postscf.forces (summed,
    remove_net=False) at a nonzero shift: the frozen-density assembly here is the
    same object the validated force path differentiates."""
    u = np.array([0.37, 0.21, 0.63])
    D_diff, _, _, netf_diff, _ = net_force_value_and_grad(u, want_grad=False)
    V, _ = _ensure_voxel()
    s = torch.as_tensor(u, dtype=torch.float64) @ V
    pos = (torch.as_tensor(POS0, dtype=torch.float64) + s).numpy()
    res = _run_scf(pos)
    netf_lib = forces(res, remove_net=False).sum(dim=0).detach().numpy()
    return D_diff, netf_diff, netf_lib, float(np.linalg.norm(netf_lib))


def wrap01(u: np.ndarray) -> np.ndarray:
    return np.mod(u, 1.0)


def random_search(budget: int):
    best = {"D": -1.0}
    trace = []
    for _ in range(budget):
        u = RNG.random(3)
        D, e_free, _, netf, shape = net_force_value_and_grad(u, want_grad=False)
        trace.append({"u": u.tolist(), "D": D, "E": e_free})
        if D > best["D"]:
            best = {"D": D, "u": u.tolist(), "netf": netf.tolist(), "E": e_free,
                    "shape": list(shape)}
    return best, trace


def ascent_search(n_starts: int, n_steps: int, step: float):
    """Gradient ascent using the frozen-density gradient DIRECTION only.

    The frozen-density analytic gradient over-predicts the true (re-converged)
    gradient magnitude by ~1e3x (see gradient_reliability_check), so its raw
    magnitude is unusable for a step size — a normal lr*grad step overshoots the
    voxel by orders of magnitude. We therefore normalize to a unit direction and
    take a fixed step in u-space (fraction of one voxel). This is the honest
    fair-budget test: does following the cheap differentiable direction beat
    random at equal SCF count, given the magnitude is wrong?"""
    best = {"D": -1.0}
    step_log = []
    for st in range(n_starts):
        u = RNG.random(3)
        for k in range(n_steps):
            D, e_free, grad, netf, shape = net_force_value_and_grad(u, want_grad=True)
            gnorm = float(np.linalg.norm(grad))
            step_log.append({"start": st, "step": k, "D": D,
                             "grad_norm": gnorm, "u": u.tolist()})
            if D > best["D"]:
                best = {"D": D, "u": u.tolist(), "netf": netf.tolist(),
                        "E": e_free, "shape": list(shape)}
            gdir = grad / (gnorm + 1e-30)
            u = wrap01(u + step * gdir)  # normalized-direction ascent
    return best, step_log


def gradient_reliability_check(points):
    """Compare the frozen-density analytic directional derivative of D against a
    central finite difference of the RE-CONVERGED D (the true gradient). Records
    the ratio: if it is ~1 the frozen-density gradient is the true gradient; if it
    is large the density-response term (missing from the frozen partial) dominates
    dD/du and the analytic gradient magnitude is not followable."""
    h = 1e-3
    checks = []
    for u in points:
        u = np.asarray(u, float)
        _D0, _, grad, _, _ = net_force_value_and_grad(u, want_grad=True)
        gnorm = float(np.linalg.norm(grad))
        gdir = grad / (gnorm + 1e-30)
        Dp, _, _, _, _ = net_force_value_and_grad(u + h * gdir, want_grad=False)
        Dm, _, _, _, _ = net_force_value_and_grad(u - h * gdir, want_grad=False)
        fd = (Dp - Dm) / (2 * h)
        checks.append({
            "u": u.tolist(),
            "analytic_dir_deriv": gnorm,
            "fd_dir_deriv_reconverged": float(fd),
            "ratio_analytic_over_fd": gnorm / (abs(fd) + 1e-30),
            "same_sign": bool(gnorm * fd > 0),
        })
    return checks


def eggbox_scan(n: int):
    """1-D diagonal scan across one voxel: energy spread (eggbox amplitude) and
    net-force profile."""
    scan = []
    for i in range(n):
        t = i / (n - 1)
        D, e_free, _, _, _ = net_force_value_and_grad(np.array([t, t, t]),
                                                       want_grad=False)
        scan.append({"t": t, "D": D, "E": e_free})
    return scan


def main():
    t0 = time.time()
    out = {}

    _D, _, _, _, shape = net_force_value_and_grad(np.zeros(3), want_grad=False)
    dx = np.linalg.norm(CELL[0]) / shape[0]
    out["setup"] = {
        "cell_a_ang": A, "ecut_Ry": ECUT / RY, "kmesh": list(KMESH),
        "fft_shape": list(shape), "grid_spacing_ang_axis0": dx,
        "voxel_vectors_ang": _VOXEL.tolist(),
        "note": "symmetry OFF, smearing none, nspin=1, forces remove_net=False",
    }

    D_diff, netf_diff, netf_lib, D_lib = cross_check_net_force()
    out["cross_check"] = {
        "D_differentiable": D_diff, "D_library_forces": D_lib,
        "abs_diff": abs(D_diff - D_lib),
        "netf_diff": netf_diff.tolist(), "netf_lib": netf_lib.tolist(),
    }
    print(f"[cross-check] D_diff={D_diff:.6e}  D_lib={D_lib:.6e}  "
          f"|diff|={abs(D_diff-D_lib):.2e}")

    # gradient-reliability diagnostic (separate from the 30/30 budget): is the
    # cheap frozen-density gradient the true gradient of the re-converged D?
    print("[grad-check] frozen-density vs re-converged directional derivative ...")
    grad_checks = gradient_reliability_check(
        [[0.30, 0.40, 0.50], [0.65, 0.15, 0.85], [0.10, 0.90, 0.45]]
    )
    for c in grad_checks:
        print(f"  u={c['u']}  analytic={c['analytic_dir_deriv']:.3f}  "
              f"fd={c['fd_dir_deriv_reconverged']:.4f}  "
              f"ratio={c['ratio_analytic_over_fd']:.1f}  same_sign={c['same_sign']}")
    out["gradient_reliability"] = grad_checks

    BUDGET = 30
    print(f"[random] {BUDGET} SCFs ...")
    rnd_best, _ = random_search(BUDGET)
    print(f"[random] max D = {rnd_best['D']:.6e} eV/Ang at u={rnd_best['u']}")

    print("[ascent] 3 starts x 10 steps = 30 SCFs (normalized direction) ...")
    asc_best, step_log = ascent_search(3, 10, step=0.08)
    print(f"[ascent] max D = {asc_best['D']:.6e} eV/Ang at u={asc_best['u']}")

    print("[eggbox scan] 15-pt diagonal ...")
    scan = eggbox_scan(15)
    e_vals = np.array([p["E"] for p in scan])
    d_vals = np.array([p["D"] for p in scan])
    eggbox_E_meV = float((e_vals.max() - e_vals.min()) * 1000.0)

    out["random"] = {"best": rnd_best, "n_scf": BUDGET}
    out["ascent"] = {"best": asc_best, "n_scf": 30, "steps": step_log}
    out["eggbox_scan"] = {
        "points": scan, "energy_spread_meV": eggbox_E_meV,
        "max_netforce_eV_ang": float(d_vals.max()),
    }
    out["comparison"] = {
        "budget_scf_each": BUDGET,
        "random_maxD": rnd_best["D"], "ascent_maxD": asc_best["D"],
        "ratio_ascent_over_random": asc_best["D"] / rnd_best["D"],
    }
    out["total_scf_count"] = _SCF_COUNT
    out["wall_s"] = time.time() - t0

    dst = Path(__file__).resolve().parent / "attack1_results.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {_SCF_COUNT} total SCFs, {out['wall_s']:.0f}s -> {dst}")
    print(f"  eggbox energy spread : {eggbox_E_meV:.4f} meV over one voxel "
          f"(grid spacing {dx:.4f} Ang)")
    print(f"  random  max|SumF|    : {rnd_best['D']:.4e} eV/Ang")
    print(f"  ascent  max|SumF|    : {asc_best['D']:.4e} eV/Ang")
    print(f"  ascent/random ratio  : "
          f"{out['comparison']['ratio_ascent_over_random']:.3f}")


if __name__ == "__main__":
    main()
