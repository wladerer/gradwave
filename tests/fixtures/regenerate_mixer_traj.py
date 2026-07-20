"""Regenerate tests/fixtures/golden/mixer_traj.pt, the golden fixture for
tests/unit/test_mixer_trajectories.py.

The fixture is a dict with three keys:

  pairs      list of 10 (rho_in, rho_out) tuples, each a complex128 G-space
             density vector (npw = 2757) recorded from a real Si SCF run with
             seed 3, alpha 0.6, history 5. These are the *inputs* to the mixers
             and are the irreducible experimental record -- a refactor of the
             mixer classes must keep reproducing the same outputs from them.
  metric_w   float64 (npw,) metric weights for JohnsonMixer's metric variant,
             also recorded from that SCF's density grid.
  outputs    dict name -> list of 10 output vectors, one per mixer configuration;
             output[name][i] is that mixer's step(rho_in_i, rho_out_i).

Only `outputs` is algorithm-dependent: it is what an intentional change to a
mixer must deliberately move. This script recomputes `outputs` from the
recorded `pairs`/`metric_w` using the current mixer implementations, exactly as
test_mixer_trajectories._replay constructs them (g2 = linspace(0, 8, npw),
alpha=0.6, history=5, check_g0=False; torch.manual_seed(3) before each replay).

`pairs`/`metric_w` are carried through unchanged: they are recorded SCF data,
not something the mixer code produces, so they cannot be re-derived here. To
re-record them (only needed if the input trajectory itself must change), run a
Si SCF and capture (rho_in, rho_out) at each iteration together with the
density-grid metric weights, then feed them in via --pairs-from.

Usage (needs the package importable, e.g. PYTHONPATH=src):
    python tests/fixtures/regenerate_mixer_traj.py            # rewrite the golden file
    python tests/fixtures/regenerate_mixer_traj.py -o /tmp/x.pt   # write elsewhere
"""

import argparse
from pathlib import Path

import torch

from gradwave.scf.mixing import BroydenMixer, JohnsonMixer, PulayMixer

GOLDEN = Path(__file__).parent / "golden" / "mixer_traj.pt"

# name -> (class, extra kwargs).  Mirrors test_mixer_trajectories exactly.
# metric_w is filled in per-run from the fixture's recorded weights.
CONFIGS = {
    "pulay": (PulayMixer, {}),
    "pulay_kerker": (PulayMixer, {"kerker": True}),
    "broyden": (BroydenMixer, {}),
    "johnson": (JohnsonMixer, {}),
    "johnson_metric": (JohnsonMixer, {"metric_w": None}),  # None -> recorded metric_w
}


def replay(cls, pairs, **kw):
    """Feed the recorded pairs through one freshly built mixer, as the test does."""
    n = pairs[0][0].shape[0]
    torch.manual_seed(3)
    m = cls(torch.linspace(0, 8, n, dtype=torch.float64),
            alpha=0.6, history=5, check_g0=False, **kw)
    return [m.step(rin, rout) for rin, rout in pairs]


def regenerate(pairs, metric_w):
    outputs = {}
    for name, (cls, kw) in CONFIGS.items():
        kw = dict(kw)
        if "metric_w" in kw and kw["metric_w"] is None:
            kw["metric_w"] = metric_w
        outputs[name] = replay(cls, pairs, **kw)
    return {"pairs": pairs, "outputs": outputs, "metric_w": metric_w}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=GOLDEN,
                    help="output .pt path (default: the golden fixture)")
    ap.add_argument("--pairs-from", type=Path, default=GOLDEN,
                    help="fixture supplying recorded pairs/metric_w (default: golden)")
    args = ap.parse_args()

    src = torch.load(args.pairs_from, weights_only=False)
    fixture = regenerate(src["pairs"], src["metric_w"])
    torch.save(fixture, args.out)
    print(f"wrote {args.out} ({len(fixture['pairs'])} pairs, "
          f"{len(fixture['outputs'])} mixer configs)")


if __name__ == "__main__":
    main()
