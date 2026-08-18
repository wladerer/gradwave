"""Generate an SCF convergence-telemetry dataset for solver research.

Runs the hardness-graded suite (gradwave.bench.SUITE) across a mixing-method
sweep, records per-iteration flight-recorder telemetry, and writes one JSON per
run into an output directory. With the [bench] extra it also flattens everything
into a tidy Parquet table (one row per iteration) and prints a per-run outcome
summary — the surface a model / symbolic regression consumes.

    uv run python benchmarks/scf_telemetry.py --out bench_runs
    uv run python benchmarks/scf_telemetry.py --out bench_runs --cases Al,Cu --quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from gradwave.bench import SUITE, run_case, sweep_methods
from gradwave.bench.harness import DEFAULT_METHODS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_runs", help="output directory for run JSONs")
    ap.add_argument("--cases", default="", help="comma list to subset (default: all)")
    ap.add_argument("--quick", action="store_true", help="alpha in {0.3,0.7}, pulay only")
    args = ap.parse_args()

    cases = SUITE
    if args.cases:
        want = {c.strip() for c in args.cases.split(",")}
        cases = [c for c in SUITE if c.name in want]
    axes = DEFAULT_METHODS
    if args.quick:
        axes = dict(mixing_scheme=["pulay"], mixing_alpha=[0.3, 0.7], kerker=[True])
    methods = sweep_methods(axes)

    out = Path(args.out)
    print(f"suite: {[c.name for c in cases]}  ×  {len(methods)} methods "
          f"= {len(cases) * len(methods)} runs  ->  {out}/", flush=True)
    t0 = time.perf_counter()
    records = []
    for case in cases:
        for method in methods:
            rec = run_case(case, method, run_dir=out)
            records.append(rec)
            o = rec.outcome
            tags = ",".join(t["tag"] for t in o.get("diagnosis", [])) or "-"
            if o.get("error"):
                status = o["error"]
            elif o.get("converged"):
                status = (f"conv n={o.get('n_iter')} {o.get('wall_s')}s "
                          f"res={o.get('final_residual'):.1e}")
            else:
                status = f"NOT-CONV n={o.get('n_iter')} res={o.get('final_residual')}"
            print(f"  {case.name:12s} {rec.method['mixing_scheme']:8s} "
                  f"a={rec.method['mixing_alpha']}  {status}  [{tags}]", flush=True)
    print(f"\n{len(records)} runs in {time.perf_counter() - t0:.0f}s", flush=True)

    # optional analysis layer (needs [bench])
    try:
        from gradwave.bench.analyze import load_runs, load_runs_summary
        pq = out / "iterations.parquet"
        df = load_runs(out)
        df.to_parquet(pq, index=False)
        summ = load_runs_summary(out)
        print(f"\nflattened {len(df)} iteration-rows -> {pq}", flush=True)
        conv = summ[summ["converged"] == True]  # noqa: E712
        if len(conv):
            print("\nfastest converging method per case (by wall_s):", flush=True)
            best = conv.loc[conv.groupby("case")["wall_s"].idxmin()]
            for _, r in best.iterrows():
                print(f"  {r['case']:12s} {r['scheme']:8s} a={r['alpha']}  "
                      f"n_iter={r['n_iter']}  {r['wall_s']}s", flush=True)
    except ImportError:
        print("\n[bench] extra not installed — JSON runs written, skipping Parquet/summary.",
              flush=True)


if __name__ == "__main__":
    main()
