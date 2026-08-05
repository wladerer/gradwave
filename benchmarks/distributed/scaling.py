"""k-point-sharding scaling sweep: wall time vs torchrun rank count.

Runs the same fcc Al SCF (al_scf.yaml: 8x8x8 full mesh = 512 k, symmetry off)
at 1, 2, 4, and 8 ranks through scripts/gradwave_distributed.sh, with a fixed
OMP thread count per rank so the sweep isolates the k-parallel speedup from
intra-op threading. Checks that every rank count converges to the same free
energy, then writes scaling.json and scaling.png.

Wall time is the end-to-end torchrun wall clock (process launch, setup, SCF,
output), the number a user experiences — not just the SCF loop.

Run from the repo root (the 8-rank point wants >= 16 cores):
    uv run python benchmarks/distributed/scaling.py [--ranks 1,2,4,8] [--threads 2]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent


def run_one(nproc: int, threads: int) -> dict:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    # same launch as scripts/gradwave_distributed.sh, with the uv project
    # overridable so an offline box can point at an already-synced env
    # (GRADWAVE_UV_PROJECT=~/github/gradwave GRADWAVE_UV_NO_SYNC=1)
    uv = ["uv", "run", "--project", env.get("GRADWAVE_UV_PROJECT", str(ROOT))]
    if env.get("GRADWAVE_UV_NO_SYNC"):
        uv.append("--no-sync")
    t0 = time.monotonic()
    subprocess.run(
        [*uv, "torchrun", f"--nproc-per-node={nproc}",
         "--master-addr=127.0.0.1", "--master-port=29500",
         "-m", "gradwave.cli", "run", str(HERE / "al_scf.yaml")],
        cwd=ROOT, env=env, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wall = time.monotonic() - t0
    summary = json.loads((HERE / "out_scaling" / "scf.json").read_text())
    assert summary["scf"]["converged"], f"nproc={nproc} did not converge"
    return {
        "nproc": nproc,
        "threads_per_rank": threads,
        "wall_s": wall,
        "runtime_s": summary["runtime_s"],
        "n_iter": summary["scf"]["n_iter"],
        "free_energy_eV": summary["scf"]["energies_eV"]["free_energy"],
    }


def plot(rows: list[dict], png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["nproc"] for r in rows]
    # runtime_s is gradwave's own end-to-end wall clock (parse → SCF →
    # outputs), the per-run number that scales with the k-shard; wall_s adds
    # the torchrun + interpreter launch on top and is kept in scaling.json.
    w = [r["runtime_s"] for r in rows]
    ideal = [w[0] / (ni / n[0]) for ni in n]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(n, ideal, ls="--", lw=1.2, color="#52514e", alpha=0.7,
            label="ideal 1/N")
    ax.plot(n, w, marker="o", ms=6, lw=2, color="#2a78d6", label="measured")
    for ni, wi in zip(n, w, strict=True):
        ax.annotate(f"{wi:.0f} s", (ni, wi), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color="#2a2a28")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("torchrun ranks (k-set shards)")
    ax.set_ylabel("run wall clock [s]")
    ax.set_title(f"fcc Al, 1728 k, {rows[0]['threads_per_rank']} threads/rank")
    ax.grid(True, which="both", lw=0.4, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    print(f"wrote {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", default="1,2,4,8")
    ap.add_argument("--threads", type=int, default=2,
                    help="OMP threads per rank, fixed across the sweep")
    args = ap.parse_args()

    rows = []
    for nproc in [int(s) for s in args.ranks.split(",")]:
        print(f"nproc={nproc} ...", flush=True)
        row = run_one(nproc, args.threads)
        print(f"  wall {row['wall_s']:.1f} s, n_iter {row['n_iter']}, "
              f"F {row['free_energy_eV']:.8f} eV", flush=True)
        rows.append(row)

    f0 = rows[0]["free_energy_eV"]
    for r in rows[1:]:
        df = abs(r["free_energy_eV"] - f0)
        assert df < 1e-6, f"free energy drifted across ranks: {df:.2e} eV"

    (HERE / "scaling.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {HERE / 'scaling.json'}")
    plot(rows, HERE / "scaling.png")


if __name__ == "__main__":
    main()
