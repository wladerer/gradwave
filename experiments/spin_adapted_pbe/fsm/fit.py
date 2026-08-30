"""Assemble the FSM E(M) scans into curves, minima, and the μ₁ refit + LOO.

Reads the raw per-(system, μ₁) JSON written by scan.py, and for each curve:

  - cubic-spline F(M) through the converged points → m* (argmin on a fine
    grid), F(m*), and the FSM spin stiffness d²F/dM² at m*;
  - a local quartic fit on the ±3-point window around the discrete minimum as
    a cross-check on the spline argmin;
  - the constraining field μ↑ − μ↓ per point (2·∂F/∂M — a consistency probe).

Sanity gate (Phase-0 bug detector, run before any fitting): the PBE (μ₁=0)
curve's argmin must agree with the unconstrained SCF moment to ≤0.02 μB per
system.

The refit interpolates each system's minimum position m*(μ₁) with a quadratic
through the three sampled μ₁ values and minimizes the uniform-weight squared
moment error against experiment; leave-one-out refits on 4 systems and
predicts the 5th. No SCFs are run here — use scan.py --mu1 <fitted> (or
confirm.py) for the direct confirmation runs.

  uv run python experiments/spin_adapted_pbe/fsm/fit.py \
      --raw experiments/spin_adapted_pbe/fsm/raw \
      --out experiments/spin_adapted_pbe/fsm/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

MU1_GRID = [0.0, -0.06, -0.10]
SYSTEMS = ["Fe", "Ni", "Co", "FeCo", "Co2MnSi"]
EXP = {"Fe": 2.22, "Ni": 0.61, "Co": 1.60, "FeCo": 4.60, "Co2MnSi": 5.00}
REF_MU1_408 = -0.0610  # #408's 3-system fit, the value under re-test


def load_curve(raw: Path, key: str, mu1: float) -> dict:
    path = raw / f"{key}_mu1_{mu1:+.3f}.json"
    return json.loads(path.read_text())


def curve_minimum(rec: dict) -> dict:
    """m*, F(m*), and d²F/dM² from the E(M) points (converged only)."""
    pts = [(float(m), p) for m, p in rec["curve"].items() if p["converged"]]
    pts.sort()
    ms = np.array([m for m, _ in pts])
    fs = np.array([p["F"] for _, p in pts])
    dmu = np.array([p["mu_up"] - p["mu_dn"] for _, p in pts])
    if len(ms) < 4:
        raise RuntimeError(f"{rec['system']} mu1={rec['mu1']}: only {len(ms)} "
                           "converged points")
    spl = CubicSpline(ms, fs)
    grid = np.linspace(ms[0], ms[-1], 4001)
    i = int(np.argmin(spl(grid)))
    m_star, f_star = float(grid[i]), float(spl(grid[i]))
    curv = float(spl(grid[i], 2))  # eV/muB^2
    edge = i in (0, len(grid) - 1)
    # cross-check: local quartic on the +-3 window around the discrete min
    j = int(np.argmin(fs))
    lo, hi = max(0, j - 3), min(len(ms), j + 4)
    m_quart = None
    if hi - lo >= 5:
        c = np.polyfit(ms[lo:hi], fs[lo:hi], 4)
        r = np.roots(np.polyder(c))
        r = [float(x.real) for x in r
             if abs(x.imag) < 1e-9 and ms[lo] <= x.real <= ms[hi - 1]]
        if r:
            m_quart = min(r, key=lambda x: np.polyval(c, x))
    return {
        "m_star": m_star, "F_star": f_star, "curvature_eV_muB2": curv,
        "argmin_at_edge": edge, "m_star_quartic": m_quart,
        "n_points": len(ms),
        "points_M": ms.tolist(), "points_F": fs.tolist(),
        "points_dmu": dmu.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="experiments/spin_adapted_pbe/fsm/raw")
    ap.add_argument("--out", default="experiments/spin_adapted_pbe/fsm/results.json")
    ap.add_argument("--mu1-grid", type=float, nargs="*", default=MU1_GRID)
    args = ap.parse_args()
    raw = Path(args.raw)
    if 0.0 not in args.mu1_grid:
        raise SystemExit("--mu1-grid must include 0.0 — the PBE (mu1=0) curve "
                         "anchors the sanity gate and the PBE columns")

    out: dict = {"exp": EXP, "mu1_grid": args.mu1_grid,
                 "ref_mu1_408": REF_MU1_408, "curves": {}, "sanity": {}}

    # 1. curves + minima
    mstar: dict[str, dict[float, float]] = {}
    for key in SYSTEMS:
        out["curves"][key] = {}
        mstar[key] = {}
        for mu1 in args.mu1_grid:
            rec = load_curve(raw, key, mu1)
            cm = curve_minimum(rec)
            cm["free"] = rec["free"]
            out["curves"][key][f"{mu1:+.3f}"] = cm
            mstar[key][mu1] = cm["m_star"]

    # 2. sanity gate: PBE argmin vs unconstrained moment (<= 0.02 muB)
    ok = True
    for key in SYSTEMS:
        cm = out["curves"][key]["+0.000"]
        m_free = abs(cm["free"]["m"])
        d = abs(m_free - cm["m_star"])
        out["sanity"][key] = {"m_free": m_free, "m_star_pbe": cm["m_star"],
                              "diff": d, "pass": bool(d <= 0.02)}
        if d > 0.02:
            ok = False
            print(f"SANITY FAIL {key}: |m_free - m*| = {d:.4f} > 0.02")
    out["sanity"]["all_pass"] = ok
    if not ok:
        Path(args.out).write_text(json.dumps(out, indent=2))
        raise SystemExit("sanity gate failed — diagnose before fitting "
                         "(Phase-0 bug signal); partial results written")

    # 3. m*(mu1) interpolants (quadratic through the sampled mu1 values)
    models = {k: np.poly1d(np.polyfit(args.mu1_grid,
                                      [mstar[k][m] for m in args.mu1_grid], 2))
              for k in SYSTEMS}

    def fit(keys: list[str]) -> dict:
        def loss(mu1):
            return sum((float(models[k](mu1)) - EXP[k]) ** 2 for k in keys)
        lo, hi = min(args.mu1_grid), max(args.mu1_grid)
        sol = minimize_scalar(loss, bounds=(lo, hi), method="bounded")
        mu1 = float(sol.x)
        return {"mu1": mu1, "loss": float(sol.fun),
                "predicted": {k: float(models[k](mu1)) for k in keys}}

    full = fit(SYSTEMS)
    full["per_system"] = {}
    for k in SYSTEMS:
        pbe = mstar[k][0.0]
        pred = float(models[k](full["mu1"]))
        full["per_system"][k] = {
            "pbe": pbe, "refit": pred, "exp": EXP[k],
            "err_pbe": pbe - EXP[k], "err_refit": pred - EXP[k]}
    full["mae_pbe"] = float(np.mean([abs(v["err_pbe"])
                                     for v in full["per_system"].values()]))
    full["mae_refit"] = float(np.mean([abs(v["err_refit"])
                                       for v in full["per_system"].values()]))
    full["moved_vs_408"] = full["mu1"] - REF_MU1_408
    out["fit"] = full

    # 4. leave-one-out
    loo = {}
    for held in SYSTEMS:
        res = fit([k for k in SYSTEMS if k != held])
        pred = float(models[held](res["mu1"]))
        loo[held] = {"mu1": res["mu1"], "predicted": pred, "exp": EXP[held],
                     "err": pred - EXP[held],
                     "err_pbe": mstar[held][0.0] - EXP[held]}
    out["loo"] = loo
    out["loo_mae"] = float(np.mean([abs(v["err"]) for v in loo.values()]))
    out["loo_mae_pbe"] = float(np.mean([abs(v["err_pbe"]) for v in loo.values()]))

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"mu1* = {full['mu1']:+.5f}  (408: {REF_MU1_408:+.4f}, "
          f"moved {full['moved_vs_408']:+.5f})")
    print(f"MAE: PBE {full['mae_pbe']:.4f} -> refit {full['mae_refit']:.4f} muB")
    for k in SYSTEMS:
        v = full["per_system"][k]
        print(f"  {k:8s} PBE {v['pbe']:.4f}  refit {v['refit']:.4f}  "
              f"exp {v['exp']:.2f}  err {v['err_refit']:+.4f}")
    print("LOO held-out errors:")
    for k in SYSTEMS:
        print(f"  {k:8s} err {loo[k]['err']:+.4f} (PBE {loo[k]['err_pbe']:+.4f}) "
              f"at mu1={loo[k]['mu1']:+.5f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
