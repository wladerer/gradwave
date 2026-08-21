"""FLAPW-DFPT Phase 1 — the FROZEN-DENSITY EFG gradcheck on a converged rutile TiO2 state.

Converges rutile TiO2 (the campaign's stabilised recipe: kerker screen, cold, smearing 0),
freezes the converged aspherical density rho_LM, the interstitial Coulomb grid v_hart and the
own-sphere q^MT, then for the representative Ti (a0) and O (a2) compares TWO ways of getting the
EXPLICIT partial dV_zz/dtau|_rho:

  (a) autograd through gradwave.flapw.efg_torch.efg_vzz_torch (the torch EFG eval with the moving
      muffin-tin Theta(G) surface phase), and
  (b) a frozen central finite difference of the SAME evaluation at tau +/- delta (density held
      fixed, no re-SCF), at delta and delta/2 to confirm the O(delta^2) linear regime.

Also runs the load-bearing control: with the Theta(G) phase omitted (include_pulay=False) the
autograd gradient collapses to zero and no longer matches the frozen FD.

This is NOT the full self-consistent dV_zz/du (that needs Phase 2's chi_0, the density response
drho/du). It validates the explicit half in isolation and reports its magnitude so the frozen /
density-response split is quantified. Heavy — run on asus:

    uv run python experiments/autoapw/efg_frozen_gradcheck.py

Env knobs (defaults in parens): FG_ECUT(250) FG_FPLMAX(4) FG_LMAX(3) FG_K(2) FG_KERKER(0.7)
FG_TOL(1e-5) FG_ITERS(160) FG_DELTA(1e-3).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from _common import A_BOHR, ATOMS, RADII, U

import gradwave.flapw.scf as scf
from gradwave.constants import BOHR_ANG
from gradwave.flapw.efg_torch import build_boundary_ctx, efg_vzz_torch
from gradwave.flapw.scf import crystal_scf_multi

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

_LSET2 = [(2, m) for m in range(-2, 3)]


def _envf(name, default):
    return float(os.environ.get(name, str(default)))


def _envi(name, default):
    return int(os.environ.get(name, str(default)))


def _capture_efg_inputs():
    """Wrap scf._efg_from_multipoles so a converged efg=True run hands back the exact frozen
    ingredients (rho_LM, v_hart, q^MT, positions, radial meshes) it used to build info['efg']."""
    grabbed: dict = {}
    orig = scf._efg_from_multipoles

    def wrapper(rho_by_key, v_hart, acart, keys, R_by_key, rr_by_key, dx, A, qmt_by_sphere=None):
        grabbed.update(rho_by_key=rho_by_key, v_hart=v_hart, acart=acart, keys=keys,
                       R_by_key=R_by_key, rr_by_key=rr_by_key, dx=dx, A=A, qmt=qmt_by_sphere)
        return orig(rho_by_key, v_hart, acart, keys, R_by_key, rr_by_key, dx, A,
                    qmt_by_sphere=qmt_by_sphere)

    scf._efg_from_multipoles = wrapper
    return grabbed, (lambda: setattr(scf, "_efg_from_multipoles", orig))


def _torch_state(grabbed, ai, key):
    rho = {(2, m): torch.as_tensor(grabbed["rho_by_key"][key][(2, m)], dtype=torch.complex128)
           for m in range(-2, 3)}
    rr = torch.as_tensor(np.asarray(grabbed["rr_by_key"][key], dtype=float), dtype=torch.float64)
    drw = rr * grabbed["dx"]
    R = float(grabbed["R_by_key"][key])
    qmt = grabbed["qmt"][ai] if grabbed["qmt"] else {}
    center0 = np.asarray(grabbed["acart"][ai][0], dtype=float)
    ctx = build_boundary_ctx(grabbed["v_hart"], R, grabbed["A"], _LSET2)
    return rho, rr, drw, R, qmt, center0, ctx


def _fd_grad(fn, center0, delta):
    g = np.zeros(3)
    for j in range(3):
        cp = center0.copy()
        cp[j] += delta
        cm = center0.copy()
        cm[j] -= delta
        g[j] = (fn(cp) - fn(cm)) / (2 * delta)
    return g


def _gradcheck_atom(grabbed, ai, key, delta):
    rho, rr, drw, R, qmt, center0, ctx = _torch_state(grabbed, ai, key)

    def vzz(center_np, include_pulay=True):
        c = torch.as_tensor(center_np, dtype=torch.float64)
        return efg_vzz_torch(rho, rr, drw, ctx, c, R, qmt=qmt, include_pulay=include_pulay)

    # autograd dV_zz/dtau|_rho
    center = torch.tensor(center0, dtype=torch.float64, requires_grad=True)
    v0 = efg_vzz_torch(rho, rr, drw, ctx, center, R, qmt=qmt)
    (grad_ad,) = torch.autograd.grad(v0, center)
    grad_ad = grad_ad.detach().numpy()

    # frozen central finite differences (delta, delta/2)
    fd1 = _fd_grad(lambda c: float(vzz(c).detach()), center0, delta)
    fd2 = _fd_grad(lambda c: float(vzz(c).detach()), center0, delta / 2)

    # load-bearing control: omit the Theta(G) phase -> autograd is identically zero
    center_np = torch.tensor(center0, dtype=torch.float64, requires_grad=True)
    v_nop = efg_vzz_torch(rho, rr, drw, ctx, center_np, R, qmt=qmt, include_pulay=False)
    grad_nop = (np.zeros(3) if not v_nop.requires_grad
                else torch.autograd.grad(v_nop, center_np)[0].detach().numpy())

    return {
        "key": key, "R": R, "center0_ang": center0.tolist(),
        "v_zz_torch": float(v0.detach()),
        "grad_autograd": grad_ad.tolist(),
        "grad_fd_delta": fd1.tolist(),
        "grad_fd_halfdelta": fd2.tolist(),
        "err_autograd_vs_fd_delta": float(np.linalg.norm(grad_ad - fd1)),
        "err_autograd_vs_fd_halfdelta": float(np.linalg.norm(grad_ad - fd2)),
        "grad_norm": float(np.linalg.norm(grad_ad)),
        "grad_nopulay_norm": float(np.linalg.norm(grad_nop)),
    }


def main():
    ecut = _envf("FG_ECUT", 250.0)
    fplmax = _envi("FG_FPLMAX", 4)
    lmax = _envi("FG_LMAX", 3)
    k = _envi("FG_K", 2)
    kerker = _envf("FG_KERKER", 0.7)
    tol = _envf("FG_TOL", 1e-5)
    iters = _envi("FG_ITERS", 160)
    delta = _envf("FG_DELTA", 1e-3)

    cfg = dict(a_bohr=A_BOHR, atoms=ATOMS, radii=RADII, ecut=ecut, lmax=lmax,
               kmesh=(k, k, k), smearing=0.0, fullpot=True, fullpot_lmax=fplmax,
               use_symmetry=True, kerker=kerker, tol=tol, iters=iters, efg=True,
               subspace_reuse=False)
    print(f"config: ecut={ecut} lmax={lmax} fplmax={fplmax} k={k}^3 kerker={kerker} "
          f"tol={tol} iters={iters} delta={delta}", flush=True)

    grabbed, restore = _capture_efg_inputs()
    try:
        _bands, info = crystal_scf_multi(**cfg)
    finally:
        restore()

    conv = info.get("e_fermi")
    efg = info["efg"]
    print(f"converged: e_fermi={conv}", flush=True)
    for kk in ("a0", "a2"):
        print(f"  numpy info efg {kk}: V_zz={efg[kk]['V_zz']:+.4f} eta={efg[kk]['eta']:.3f}",
              flush=True)

    # rutile fractional->cartesian displacement of O under the internal parameter u:
    # O at (u,u,0) -> cart (u a, u a, 0); dtau/du = (a_A, a_A, 0). Ti at (0,0,0) is fixed under u.
    a_ang = A_BOHR[0] * BOHR_ANG
    du_dir_O = np.array([a_ang, a_ang, 0.0])

    results = {"config": {k: (v if k != "atoms" else None) for k, v in cfg.items()},
               "U": U, "a_ang": a_ang, "atoms": []}
    for ai, key in ((0, "a0"), (2, "a2")):
        rec = _gradcheck_atom(grabbed, ai, key, delta)
        rec["site"] = "Ti" if key == "a0" else "O"
        # match torch eval against the numpy reference at tau0
        rec["v_zz_numpy_ref"] = float(efg[key]["V_zz"])
        rec["v_zz_torch_vs_numpy_abs"] = abs(rec["v_zz_torch"] - rec["v_zz_numpy_ref"])
        # project the frozen cartesian partial onto the O u-displacement direction (Ti: fixed -> NA)
        g = np.asarray(rec["grad_autograd"])
        rec["dVzz_du_frozen_ownboundary"] = (float(g @ du_dir_O) if key == "a2" else 0.0)
        results["atoms"].append(rec)

        print(f"\n== {rec['site']} ({key}) ==", flush=True)
        print(f"  V_zz torch={rec['v_zz_torch']:+.6f}  numpy={rec['v_zz_numpy_ref']:+.6f}  "
              f"|diff|={rec['v_zz_torch_vs_numpy_abs']:.2e}", flush=True)
        print(f"  grad autograd   = {np.array2string(g, precision=3)}", flush=True)
        print(f"  grad FD  (d)    = {np.array2string(np.array(rec['grad_fd_delta']), precision=3)}",
              flush=True)
        print(f"  grad FD  (d/2)  = "
              f"{np.array2string(np.array(rec['grad_fd_halfdelta']), precision=3)}", flush=True)
        ratio = rec["err_autograd_vs_fd_delta"] / max(rec["err_autograd_vs_fd_halfdelta"], 1e-30)
        print(f"  ||ad-fd(d)||={rec['err_autograd_vs_fd_delta']:.3e}  "
              f"||ad-fd(d/2)||={rec['err_autograd_vs_fd_halfdelta']:.3e}  "
              f"(halving ratio ~{ratio:.1f}x)", flush=True)
        print(f"  ||grad|| (Pulay on) ={rec['grad_norm']:.3e}   "
              f"||grad|| (Pulay OFF)={rec['grad_nopulay_norm']:.3e}  <- load-bearing control",
              flush=True)
        if key == "a2":
            print(f"  frozen own-boundary dV_zz/du (O) = "
                  f"{rec['dVzz_du_frozen_ownboundary']:+.1f} eV/A^2 per unit u   "
                  f"(full FD ref +1485; remainder is Phase-2 chi_0)", flush=True)

    out = Path(__file__).with_name("results_frozen_gradcheck.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
