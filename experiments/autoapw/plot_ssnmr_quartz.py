"""Render the alpha-quartz 29Si MAS spectrum PNG from a quartz_spectrum.json
produced by the ssNMR demo run (quartz_gate.py on asus). Uses the canonical
io.analysis helper so the axis convention matches every other gradwave plot.

Usage: uv run python experiments/autoapw/plot_ssnmr_quartz.py <spectrum.json> <out.png>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gradwave.io.analysis import plot_nmr_spectrum


def main(json_path: str, out_png: str) -> int:
    d = json.loads(Path(json_path).read_text())
    ppm = np.asarray(d["ppm_axis"], dtype=float)
    inten = np.asarray(d["intensity"], dtype=float)
    label = f"$^{{29}}$Si MAS · $\\delta_{{iso}}$={np.mean(d['delta_iso']):.1f} ppm"
    plot_nmr_spectrum(ppm, inten, path=out_png, label=label)
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
