"""Prototype: schedule parity on the paired Ni scalar system.

The collinear path converges PD_Ni fcc in 13 iterations (quadratic diago
schedule, johnson). The spinor path on the SAME system floors at dm ~5e-4
(linear schedule, tol_eff = 0.03 r ~ 1.5e-5, an eigensolve looser than
rhotol). If the floor is eigensolve noise, quadratic + tight Davidson should
recover the collinear behavior on the spinor path.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from experiments.noncollinear_convergence import systems as sysmod
from experiments.noncollinear_convergence.run_matrix import run_nc, z

torch.set_num_threads(8)
outdir = Path("experiments/noncollinear_convergence/results") / "asus"
outdir.mkdir(parents=True, exist_ok=True)
CASES = {
    # quadratic schedule alone (the one-line production candidate)
    "socfree_quad": dict(mag_diago_schedule="quadratic"),
    # quadratic + tight base + no backoff churn: the clean discriminator
    "socfree_quad_tight": dict(mag_diago_schedule="quadratic", diago_tol=1e-11,
                               adaptive=False),
    # johnson + quadratic: full parity with the collinear nspin=2 default
    "socfree_johnson_quad": dict(mag_mixer="johnson",
                                 mag_diago_schedule="quadratic"),
}
for name, kw in CASES.items():
    row = run_nc(f"ni_{name}_s0.6_z", sysmod.ni_fcc(False), z(0.6),
                 outdir=outdir, **kw)
    print(json.dumps(row), flush=True)
    with open(outdir / "summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
print("DONE_PROTO", flush=True)
