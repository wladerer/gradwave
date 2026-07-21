"""Flagship: gradient-train a range-separated hybrid (α, ω) against a joint,
multi-material band-gap loss, by backprop through the periodic hybrid SCF, and
test transferability on a held-out material.

Each optimizer step re-converges the hybrid SCF for every training material at
the current (α, ω) (the frozen-orbital gap gradient is only meaningful at
self-consistency), forms the differentiable gap (benchmarks/hybrid_design/gap.py),
sums the squared-error loss over materials, and takes one backward pass for
d(loss)/d(α, ω). The gaps are wired to a full-BZ-consistent Γ cell at a loose
cutoff — the numbers demonstrate the machinery, not converged physics.

The targets are the gaps at a ground-truth (α*, ω*), so a perfect joint fit
exists: recovering (α*, ω*) from a perturbed start over an over-determined set
(3 materials, 2 parameters) shows the exact gradient path works, and the held-out
material matching its target shows the trained two-parameter hybrid transfers.

    uv run python benchmarks/hybrid_design/train.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from gap import differentiable_hybrid_gap, vbm_cbm  # noqa: E402

from gradwave.postscf.exchange_multik import HybridExchangeParams  # noqa: E402
from gradwave.postscf.hybrid import hybrid_scf  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import setup_system  # noqa: E402

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
MODE = "short_range"
_upf = {}


def upf(name):
    if name not in _upf:
        _upf[name] = parse_upf(f"{PSE}/{name}")
    return _upf[name]


def _cell(a, frac):
    cell = 0.5 * a * FCC
    return cell, np.asarray(frac) @ cell


def crystal(name, a, frac, soa, pseudos, ecut, nb, mix=0.7):
    cell, pos = _cell(a, frac)
    return dict(cell=cell, pos=pos, soa=soa, upfs=[upf(p) for p in pseudos],
                ecut=ecut, nb=nb, name=name, mix=mix)


_DIA = [[0, 0, 0], [0.25, 0.25, 0.25]]
_RS = [[0, 0, 0], [0.5, 0.5, 0.5]]
MATERIALS = {
    "Si":  crystal("Si", 5.43, _DIA, [0, 0], ["Si_ONCV_PBE-1.2.upf"], 18, 8),
    # C needs a smaller mixing and more iterations (hybrid diamond is stiff)
    "C":   crystal("C", 3.567, _DIA, [0, 0], ["C_ONCV_PBE-1.2.upf"], 30, 8, mix=0.3),
    "MgO": crystal("MgO", 4.21, _RS, [0, 1],
                   ["Mg_ONCV_PBE-1.2.upf", "O_ONCV_PBE-1.2.upf"], 40, 10),
    "AlAs": crystal("AlAs", 5.66, _DIA, [0, 1],
                    ["Al_ONCV_PBE-1.2.upf", "As_ONCV_PBE-1.2.upf"], 25, 10),
}
TRAIN = ["Si", "C", "MgO"]
HELDOUT = "AlAs"
ALPHA_STAR, OMEGA_STAR = 0.25, 0.20


def build(m):
    return setup_system(m["cell"], m["pos"], m["soa"], m["upfs"],
                        ecut=m["ecut"] * RY, kmesh=(1, 1, 1), nbands=m["nb"])


def _hybrid(m, alpha, omega, start, mix, rhotol, max_iter):
    return hybrid_scf(build(m), alpha=float(alpha), mode=MODE, omega=float(omega),
                      smearing="none", etol=1e-8, rhotol=rhotol, max_iter=max_iter,
                      mixing_alpha=mix, start_from=start, verbose=False)


def converge_and_gap(m, alpha, omega, start=None, strict=False):
    """Converge the hybrid SCF at (α, ω), warm-started from `start`. A gap only
    needs a well-converged density, not machine-tight, so training runs at
    rhotol 1e-6; if the stiff hybrid diamond still misses, retry once cold with
    half the mixing before falling back to the last iterate (training tolerates a
    near-converged gap). `strict` re-asserts convergence for the final readout."""
    r = _hybrid(m, alpha, omega, start, m["mix"], 1e-6, 150)
    if not r.converged:
        r = _hybrid(m, alpha, omega, None, m["mix"] * 0.5, 1e-6, 300)
    if strict:
        assert r.converged, f"{m['name']} hybrid SCF did not converge"
    return r


def gap_value(m, alpha, omega):
    (_, _, ev), (_, _, ec) = vbm_cbm(converge_and_gap(m, alpha, omega, strict=True))
    return ec - ev


def main():
    t0 = time.time()
    print(f"targets: gaps at (α*, ω*) = ({ALPHA_STAR}, {OMEGA_STAR})")
    targets, target_res = {}, {}
    for k in MATERIALS:
        r = converge_and_gap(MATERIALS[k], ALPHA_STAR, OMEGA_STAR, strict=True)
        (_, _, ev), (_, _, ec) = vbm_cbm(r)
        targets[k], target_res[k] = ec - ev, r
        tag = "train" if k in TRAIN else "held-out"
        print(f"  {k:4s} gap* = {targets[k]:7.3f} eV  ({tag})")

    params = HybridExchangeParams(alpha=0.12, omega=0.35, mode=MODE)
    opt = torch.optim.Adam(params.parameters(), lr=0.05)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.985)
    print(f"\nstart (α, ω) = ({float(params.alpha):.3f}, {float(params.omega):.3f}); "
          f"training on {TRAIN}")
    # seed the warm cache with the converged target densities so even step 0
    # warm-starts (a cold hybrid at the off-target start grinds to max_iter)
    hist, warm = [], {k: target_res[k] for k in TRAIN}
    best = (1e30, 0.12, 0.35)
    for step in range(55):
        ts = time.time()
        opt.zero_grad()
        loss = torch.zeros(())
        for k in TRAIN:
            res = converge_and_gap(MATERIALS[k], float(params.alpha),
                                   float(params.omega), start=warm.get(k))
            warm[k] = res
            g = differentiable_hybrid_gap(res, params)
            loss = loss + (g - targets[k]) ** 2
        loss.backward()
        opt.step()
        sched.step()
        a, o, L = float(params.alpha), float(params.omega), float(loss)
        best = min(best, (L, a, o))
        hist.append(dict(step=step, alpha=a, omega=o, loss=L))
        print(f"  step {step:2d}  α={a:.4f}  ω={o:.4f}  loss={L:.3e}  "
              f"({time.time()-ts:.0f}s)")
        if L < 1e-4:
            break

    _, a, o = best  # the best-fit hybrid over the trajectory (Adam oscillates)
    print(f"\nbest-fit (α, ω) = ({a:.4f}, {o:.4f})  target ({ALPHA_STAR}, {OMEGA_STAR})"
          f"  [loss {best[0]:.3e}]")
    print("transferability — held-out gap at the trained hybrid:")
    for k in [HELDOUT]:
        g = gap_value(MATERIALS[k], a, o)
        print(f"  {k}: gap = {g:.4f} eV  target {targets[k]:.4f}  "
              f"(err {abs(g-targets[k])*1e3:.2f} meV)")
    print("per-material fit (train):")
    for k in TRAIN:
        g = gap_value(MATERIALS[k], a, o)
        print(f"  {k}: gap {g:.4f}  target {targets[k]:.4f}  "
              f"(err {abs(g-targets[k])*1e3:.2f} meV)")
    print(f"\n{time.time()-t0:.0f}s total")
    (SP / "train.json").write_text(json.dumps(
        dict(alpha_star=ALPHA_STAR, omega_star=OMEGA_STAR, targets=targets,
             recovered=[a, o], history=hist), indent=1))


if __name__ == "__main__":
    main()
