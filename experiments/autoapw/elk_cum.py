# ruff: noqa: E402, I001, F401, B007, E501  # scratch probe script (hardcoded paths / sys.path insert)
"""Elk rhomt l=2 (m=0) cumulative on-site-moment buildup vs r/R, to compare radial shape vs gw.
Usage: python elk_cum.py STATE.OUT "B:2,N:2" ias
Prints fractional cumulative int_0^r rho_20/r' dr' at r/R fractions + the absolute moment.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/wladerer/github/gradwave/.claude/worktrees/lightcation-efg-diag/experiments/autoapw")
from elk_onsite import load  # noqa: E402


def main():
    path, spec_s, ias = sys.argv[1], sys.argv[2], int(sys.argv[3])
    spec = [(t.split(":")[0], int(t.split(":")[1])) for t in spec_s.split(",")]
    labels = []
    for lab, nat in spec:
        labels += [lab] * nat
    d = load(path)
    sp = d["ias_sp"][ias]
    nr = d["nrmt"][sp]
    rr = d["rsp"][sp][:nr]
    R = rr[-1]
    rho20 = d["rho"][ias][:nr, 6]  # real l=2 m=0 harmonic (idx 4+2)
    integ = rho20 / rr
    # cumulative trapezoid on the (non-uniform) mesh
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(rr))])
    tot = cum[-1]
    print(f"# Elk {path} ias={ias} {labels[ias]} R={R:.4f}b nr={nr} moment_int={tot:+.5e}",
          flush=True)
    for frac in (0.25, 0.5, 0.7, 0.85, 0.95, 1.0):
        idx = min(np.searchsorted(rr, frac * R), nr - 1)
        print(f"   r/R={frac:.2f} (r={rr[idx]:.4f}b): cum={100*cum[idx]/tot:.1f}% of total",
              flush=True)


if __name__ == "__main__":
    main()
