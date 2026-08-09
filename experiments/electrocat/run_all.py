"""Run all four pairs (*H, *CO on Pt(111), Au(111)) and print a summary table.

Runs sequentially — a single H100 shouldn't have two heavy GPU jobs contending.
Each pair is resumable (run_pair writes results/<pair>.json incrementally), so a
re-run continues where a drop left off.

    uv run python run_all.py            # production (GPU)
    uv run python run_all.py --debug    # tiny CPU sanity of the whole pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import run_pair  # noqa: E402

PAIRS = [("Pt", "H"), ("Pt", "CO"), ("Au", "H"), ("Au", "CO")]


def main(debug: bool = False) -> None:
    dbg = config.DEBUG if debug else None
    rows = []
    for metal, ads in PAIRS:
        print(f"\n===== {metal}(111) + *{ads} =====", flush=True)
        r = run_pair.run_pair(metal, ads, dbg)
        a = r["adsorption"]
        rows.append((f"{metal} *{ads}", a["best_site"], a["e_ads"], a["dg_ads"]))

    print("\n================ SUMMARY ================")
    print(f"{'system':<10}{'site':<8}{'ΔE [eV]':>10}{'ΔG [eV]':>10}")
    for name, site, de, dg in rows:
        print(f"{name:<10}{site:<8}{de:>10.3f}{dg:>10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    main(ap.parse_args().debug)
