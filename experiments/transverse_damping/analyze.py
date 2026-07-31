"""Read the study's traces/summary and print the tables the README needs.

  - transverse amplification: geometric growth rate of dm_x over iterations
    10..40 (baseline vs damped), the campaign's 3x/iteration signature
  - floor table: dm, dm_x, dm_z, dn, and the driver's res-gate floor per run
  - canted alignment: pair_angle over the first 8 iterations (kill-criterion)

Usage: uv run python experiments/transverse_damping/analyze.py [host]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HOST = sys.argv[1] if len(sys.argv) > 1 else "nixos"
RES = Path(__file__).resolve().parent / "results" / HOST


def load_trace(name):
    p = RES / f"trace_{name}.jsonl"
    if not p.exists():
        return None
    return [json.loads(x) for x in p.read_text().splitlines()]


def load_summary():
    p = RES / "summary.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    # de-dup by name, keep last
    d = {}
    for r in rows:
        d[r["name"]] = r
    return d


def growth_rate(trace, key="dm_x", lo=10, hi=40):
    """geometric mean per-iteration growth factor of `key` over [lo, hi]."""
    vals = [(r["it"], r.get(key)) for r in trace
            if lo <= r["it"] <= hi and r.get(key) and r[key] > 0]
    if len(vals) < 3:
        return float("nan")
    it0, v0 = vals[0]
    it1, v1 = vals[-1]
    return (v1 / v0) ** (1.0 / (it1 - it0))


def main():
    summ = load_summary()
    print(f"# results host={HOST}  ({len(summ)} runs)\n")

    print("## Floor table")
    hdr = ("name", "conv", "nit", "F[eV]", "|m|", "dn_fl", "dm_fl",
           "dmx_fl", "dmz_fl", "res_gate")
    print(("{:<24}" + "{:>6}{:>5}" + "{:>14}{:>7}" + "{:>9}" * 5).format(*hdr))
    for name, r in summ.items():
        print(("{:<24}{:>6}{:>5}{:>14.6f}{:>7.3f}"
               + "{:>9}" * 5).format(
            name[:24], "Y" if r["converged"] else "n", r["n_iter"],
            r["free_energy"], r["mag_abs"],
            _e(r.get("dn_floor")), _e(r.get("dm_floor")),
            _e(r.get("dm_x_floor")), _e(r.get("dm_z_floor")),
            _e(r.get("res_gate_floor"))))

    print("\n## Transverse amplification (geom. growth/iter of dm_x, it 10..40)")
    for name in summ:
        tr = load_trace(name)
        if tr is None:
            continue
        g = growth_rate(tr, "dm_x")
        gp = growth_rate(tr, "dm_perp") if any("dm_perp" in x for x in tr) else None
        # peak dm_x
        dmx = [x.get("dm_x", 0) for x in tr]
        print(f"  {name:<24} growth/iter={g:6.3f}  peak dm_x={max(dmx):.2e}"
              + (f"  perp={gp:.3f}" if gp else ""))

    print("\n## Canted alignment (pair_angle deg, first 8 iters) [kill-criterion]")
    for name, r in summ.items():
        if "pair_angle_traj" not in r:
            continue
        traj = r["pair_angle_traj"]
        mtraj = r.get("m_abs_g0_traj")
        print(f"  {name:<24} angle={_fl(traj)}")
        print(f"  {'':<24} |m_tot|={_fl(mtraj)}  pair_angle_floor="
              f"{_e(r.get('pair_angle_floor'))}")


def _e(x):
    if x is None or (isinstance(x, float) and x != x):
        return "---"
    return f"{x:.2e}"


def _fl(lst):
    if not lst:
        return "---"
    return "[" + ", ".join(f"{v:.1f}" if isinstance(v, (int, float)) else "--"
                           for v in lst) + "]"


if __name__ == "__main__":
    main()
