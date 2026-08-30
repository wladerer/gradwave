"""Assemble the FSM E(M) scans into curves, minima, and the μ₁ refit + LOO.

Reads the raw per-(system, μ₁) JSON written by scan.py, and for each curve:

  - the CANONICAL minimum m* from the constraining field: μ↑ − μ↓ = 2·∂F/∂M
    crosses zero at the minimum, and a root of the (shape-preserving PCHIP)
    field interpolant is far better conditioned than the argmin of a shallow,
    asymmetric F(M) well on a 0.15–0.3 μB grid (measured: Ni field-root agrees
    with the unconstrained moment to 5e-4 μB where the F-spline argmin missed
    by 0.06). The FSM spin stiffness d²F/dM² = ½·d(μ↑−μ↓)/dM at the root.
    A half-metal kink (Co₂MnSi: the field JUMPS through zero at the
    Slater-Pauling integer) is detected by the crossing slope exceeding 5× the
    median inter-point slope; the sampled lowest-F point is then the minimum
    and the stiffness is reported as the (one-sided) crossing slope with a
    kink flag;
  - cubic-spline F(M) argmin and a local quartic as cross-checks.

Sanity gate (Phase-0 bug detector, run before any fitting): the field-root m*
must agree with the unconstrained SCF moment to ≤0.02 μB for EVERY
(system, μ₁) curve.

The refit interpolates each system's minimum position m*(μ₁) with a quadratic
through the three sampled μ₁ values and minimizes the uniform-weight squared
moment error against experiment; leave-one-out refits on 4 systems and
predicts the 5th. The canonical m*(μ₁) samples are the UNCONSTRAINED moments
m_free(μ₁) — SCF-exact minima of the same functional, with no grid-
interpolation error; the sanity gate above certifies the FSM curves agree with
them, and the field-root interpolant rides along as a cross-check. No SCFs run
here — scan.py --mu1 <fitted> / residual.py do the direct confirmation.

  uv run python experiments/spin_adapted_pbe/fsm/fit.py \
      --raw experiments/spin_adapted_pbe/fsm/raw \
      --out experiments/spin_adapted_pbe/fsm/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq, minimize_scalar

MU1_GRID = [0.0, -0.06, -0.10]
SYSTEMS = ["Fe", "Ni", "Co", "FeCo", "Co2MnSi"]
EXP = {"Fe": 2.22, "Ni": 0.61, "Co": 1.60, "FeCo": 4.60, "Co2MnSi": 5.00}
REF_MU1_408 = -0.0610  # #408's 3-system fit, the value under re-test


def load_curve(raw: Path, key: str, mu1: float) -> dict:
    path = raw / f"{key}_mu1_{mu1:+.3f}.json"
    return json.loads(path.read_text())


def curve_minimum(rec: dict) -> dict:
    """m*, F(m*), and the stiffness d²F/dM² from the E(M) points.

    Canonical m*: zero of the PCHIP interpolant of the constraining field
    μ↑−μ↓ (= 2·∂F/∂M), taking the sign crossing nearest the discrete F argmin;
    at a half-metal kink (crossing slope > 5× the median slope) the sampled
    lowest-F point is the minimum. Spline/quartic argmins ride along as
    cross-checks."""
    pts = [(float(m), p) for m, p in rec["curve"].items() if p["converged"]]
    pts.sort()
    ms = np.array([m for m, _ in pts])
    fs = np.array([p["F"] for _, p in pts])
    dmu = np.array([p["mu_up"] - p["mu_dn"] for _, p in pts])
    if len(ms) < 4:
        raise RuntimeError(f"{rec['system']} mu1={rec['mu1']}: only {len(ms)} "
                           "converged points")

    # --- canonical: field root -------------------------------------------
    j = int(np.argmin(fs))
    crossings = [i for i in range(len(ms) - 1) if dmu[i] < 0.0 <= dmu[i + 1]]
    field = PchipInterpolator(ms, dmu)
    kink = False
    if crossings:
        i = min(crossings, key=lambda i: abs(0.5 * (ms[i] + ms[i + 1]) - ms[j]))
        slope_c = (dmu[i + 1] - dmu[i]) / (ms[i + 1] - ms[i])
        typical = float(np.median(np.abs(np.diff(dmu) / np.diff(ms))))
        kink = bool(slope_c > 5.0 * typical)
        # half-metal kink: ∂F/∂M jumps through zero — the sampled lowest-F
        # point IS the minimum (the Slater-Pauling plateau edge)
        m_field = (float(ms[j]) if kink
                   else float(brentq(field, ms[i], ms[i + 1])))
        stiff = 0.5 * (float(field(m_field, 1)) if not kink else slope_c)
    else:  # minimum at the grid edge — flagged below
        m_field, stiff = float(ms[j]), None

    # --- cross-checks: F-spline argmin + local quartic --------------------
    spl = CubicSpline(ms, fs)
    grid = np.linspace(ms[0], ms[-1], 4001)
    i_min = int(np.argmin(spl(grid)))
    m_spline = float(grid[i_min])
    lo, hi = max(0, j - 3), min(len(ms), j + 4)
    m_quart = None
    if hi - lo >= 5:
        c = np.polyfit(ms[lo:hi], fs[lo:hi], 4)
        roots = np.roots(np.polyder(c))
        cand = [float(x.real) for x in roots
                if abs(x.imag) < 1e-9 and ms[lo] <= x.real <= ms[hi - 1]]
        if cand:
            m_quart = min(cand, key=lambda x: np.polyval(c, x))

    return {
        "m_star": m_field, "F_star": float(spl(m_field)),
        "stiffness_eV_muB2": stiff, "kink": kink,
        "m_star_spline": m_spline, "m_star_quartic": m_quart,
        "argmin_at_edge": bool(j in (0, len(ms) - 1)) and not crossings,
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

    # 1. curves + minima; canonical fit samples are the SCF-exact free moments
    mfree: dict[str, dict[float, float]] = {}
    mfield: dict[str, dict[float, float]] = {}
    for key in SYSTEMS:
        out["curves"][key] = {}
        mfree[key], mfield[key] = {}, {}
        for mu1 in args.mu1_grid:
            rec = load_curve(raw, key, mu1)
            cm = curve_minimum(rec)
            cm["free"] = rec["free"]
            out["curves"][key][f"{mu1:+.3f}"] = cm
            mfree[key][mu1] = abs(rec["free"]["m"])
            mfield[key][mu1] = cm["m_star"]

    # 2. sanity gate: field-root argmin vs unconstrained moment, EVERY curve
    ok = True
    for key in SYSTEMS:
        out["sanity"][key] = {}
        for mu1 in args.mu1_grid:
            d = abs(mfree[key][mu1] - mfield[key][mu1])
            out["sanity"][key][f"{mu1:+.3f}"] = {
                "m_free": mfree[key][mu1], "m_star_field": mfield[key][mu1],
                "diff": d, "pass": bool(d <= 0.02)}
            if d > 0.02:
                ok = False
                print(f"SANITY FAIL {key} mu1={mu1:+.3f}: "
                      f"|m_free - m*| = {d:.4f} > 0.02")
    out["sanity"]["all_pass"] = ok
    if not ok:
        Path(args.out).write_text(json.dumps(out, indent=2))
        raise SystemExit("sanity gate failed — diagnose before fitting "
                         "(Phase-0 bug signal); partial results written")

    # 3. m*(mu1) interpolants (quadratic through the sampled mu1 values)
    def _models(samples):
        return {k: np.poly1d(np.polyfit(args.mu1_grid,
                                        [samples[k][m] for m in args.mu1_grid], 2))
                for k in SYSTEMS}

    models = _models(mfree)          # canonical: SCF-exact free minima
    models_field = _models(mfield)   # cross-check: FSM field roots

    def fit(keys: list[str], mdl) -> dict:
        def loss(mu1):
            return sum((float(mdl[k](mu1)) - EXP[k]) ** 2 for k in keys)
        lo, hi = min(args.mu1_grid), max(args.mu1_grid)
        sol = minimize_scalar(loss, bounds=(lo, hi), method="bounded")
        mu1 = float(sol.x)
        return {"mu1": mu1, "loss": float(sol.fun),
                "predicted": {k: float(mdl[k](mu1)) for k in keys}}

    full = fit(SYSTEMS, models)
    full["mu1_field_crosscheck"] = fit(SYSTEMS, models_field)["mu1"]
    full["per_system"] = {}
    for k in SYSTEMS:
        pbe = mfree[k][0.0]
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
        res = fit([k for k in SYSTEMS if k != held], models)
        pred = float(models[held](res["mu1"]))
        loo[held] = {"mu1": res["mu1"], "predicted": pred, "exp": EXP[held],
                     "err": pred - EXP[held],
                     "err_pbe": mfree[held][0.0] - EXP[held]}
    out["loo"] = loo
    out["loo_mae"] = float(np.mean([abs(v["err"]) for v in loo.values()]))
    out["loo_mae_pbe"] = float(np.mean([abs(v["err_pbe"]) for v in loo.values()]))

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"mu1* = {full['mu1']:+.5f}  (408: {REF_MU1_408:+.4f}, "
          f"moved {full['moved_vs_408']:+.5f}; field cross-check "
          f"{full['mu1_field_crosscheck']:+.5f})")
    print(f"MAE: PBE {full['mae_pbe']:.4f} -> refit {full['mae_refit']:.4f} muB")
    for k in SYSTEMS:
        v = full["per_system"][k]
        cm = out["curves"][k]["+0.000"]
        stiff = cm["stiffness_eV_muB2"]
        print(f"  {k:8s} PBE {v['pbe']:.4f}  refit {v['refit']:.4f}  "
              f"exp {v['exp']:.2f}  err {v['err_refit']:+.4f}  "
              f"stiffness {stiff if stiff is None else round(stiff, 3)}"
              f"{' (kink)' if cm['kink'] else ''}")
    print("LOO held-out errors:")
    for k in SYSTEMS:
        print(f"  {k:8s} err {loo[k]['err']:+.4f} (PBE {loo[k]['err_pbe']:+.4f}) "
              f"at mu1={loo[k]['mu1']:+.5f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
