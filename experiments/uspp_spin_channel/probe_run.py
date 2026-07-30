"""Characterize the Ni PAW magnetization-channel stagnation.

Runs one Ni PAW case with the mag-channel probe installed and dumps, per outer
iteration, the total and magnetization residual L2 norms and their |G|-shell
fractions. Optionally with spin_precond on, to see how the Stoner
preconditioner reshapes the same trajectory. Emits JSON lines.

  uv run python -m experiments.uspp_spin_channel.probe_run --start-mag 0.3 --precond off
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from experiments.uspp_spin_channel import probe
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY, pseudo


def build(a=3.52, ecut=50, kmesh=(4, 4, 4)):
    paw = parse_upf_paw(pseudo("Ni.pbe-spn-kjpaw_psl.1.0.0.UPF"))
    c = a / 2.0
    cell = np.array([[0.0, c, c], [c, 0.0, c], [c, c, 0.0]])
    return setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=ecut * RY,
                      kmesh=kmesh, ecutrho=400 * RY, nbands=18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-mag", type=float, default=0.3)
    ap.add_argument("--precond", default="off", choices=["off", "on"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=120)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    system = build()
    probe.install(system)
    try:
        res = scf_uspp(system, SpinPBE(), nspin=2, start_mag=[args.start_mag],
                       smearing="gaussian", width=0.1, etol=1e-6, rhotol=1e-5,
                       mixing_alpha=0.3, max_iter=args.max_iter, verbose=False,
                       spin_precond=(args.precond == "on"))
    finally:
        probe.restore()

    hist = res.history
    for i, rec in enumerate(probe.LOG):
        m = rec["mag_shell_frac"]
        t = rec["tot_shell_frac"]
        out = {
            "it": rec["it"],
            "res": float(hist[i]["res"]) if i < len(hist) else None,
            "tot_norm": rec["tot_norm"],
            "mag_norm": rec["mag_norm"],
            "mag_low2": round(sum(m[:2]), 4),
            "mag_hi": round(sum(m[6:]), 4),
            "tot_low2": round(sum(t[:2]), 4),
        }
        print(json.dumps(out), flush=True)
    print(json.dumps({"summary": True, "precond": args.precond,
                      "start_mag": args.start_mag,
                      "n_iter": int(res.n_iter),
                      "converged": bool(res.converged),
                      "F_eV": float(res.energies.free_energy),
                      "mag_total": float(res.mag_total)}), flush=True)


if __name__ == "__main__":
    main()
