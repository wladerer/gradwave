"""Learn a constrained exchange enhancement factor against EXPERIMENTAL solid
lattice constants — the functional-learning pipeline (Stage A: PBE-form κ,μ).

Ground truth is experiment (zero-point-corrected lattice constants), NOT another
functional. PBE and PBEsol are baselines we *report against*, not targets. The
learnable object is gradwave's `LearnableX` — PBE-form exchange with trainable
(κ, μ), which enforces the UEG limit F(0)=1 and the Lieb-Oxford bound F<1+κ BY
CONSTRUCTION, so training can only produce "weird PBE", never unphysical garbage.

Loss (pressure-at-experimental-volume, avoids differentiating an argmin): the
functional should have its equilibrium AT V_exp, i.e. P(V_exp; θ) = −dE/dV = 0.

    L(θ) = Σ_train  P(V_exp,i; θ)²

with P by a central finite difference in volume, and dP/dθ from `energy_param_grads`
(dE/dθ FREE at SCF convergence by variational stationarity — no response solve).
Held-out elements the fit never sees are the generalization test.

Prototype: reduced cutoffs for a fast loop; experimental V0 values are provisional
(replace with the Hao 2012 / Schimka 2011 ZP-corrected reference set for a rigorous
run). Serial here; the per-element EOS spokes are SeedPool-parallelizable.

    uv run python benchmarks/functional_learning/train_fx.py --train Al,Si,Pd,Ag --test Ge,Au,Pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks" / "delta_gauge"))
from lattices import geometry, natoms  # noqa: E402

from gradwave.core.xc.learnable import (  # noqa: E402
    PBE_KAPPA,
    PBE_MU,
    LearnableX,
    energy_param_grads,
)
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402

_PDIR = _ROOT / "benchmarks" / "delta_gauge" / "pseudos"

# element -> (structure, valence, reduced ecut [Ry], kmesh, smear, width [Ry],
#            experimental V0 [Å³/atom], ZP-corrected 0 K — PROVISIONAL values).
# Non-magnetic only for the prototype (no Fe/Ni; Cu excluded — defective pseudo).
_EL = {
    "Al": ("fcc", 3, 30, 12, "gaussian", 0.01, 16.24),
    "Si": ("diamond", 4, 24, 6, "none", 0.0, 19.85),
    "Ge": ("diamond", 4, 24, 6, "none", 0.0, 22.57),
    "Pd": ("fcc", 18, 48, 12, "gaussian", 0.01, 14.56),
    "Ag": ("fcc", 19, 48, 12, "gaussian", 0.01, 16.76),
    "Au": ("fcc", 11, 48, 12, "gaussian", 0.01, 16.76),
    "Pt": ("fcc", 10, 48, 12, "gaussian", 0.01, 14.98),
}
_UPF = {}


def _upf(el):
    if el not in _UPF:
        _UPF[el] = parse_upf(_PDIR / f"{el}.upf")
    return _UPF[el]


def _energy_and_grads(el, v0_per_atom, xc):
    """Converged E/atom [eV] and dE/dθ per atom at volume v0_per_atom [Å³/atom]."""
    struct, _z, ecut, k, smear, width, _ = _EL[el]
    cell, pos, _ = geometry(struct, el, v0_per_atom)
    nat = natoms(struct)
    system = setup_system(cell, pos, [0] * nat, [_upf(el)], ecut=ecut * RY,
                          kmesh=(k, k, k))
    res = scf(system, xc, smearing=smear, width=width * RY, etol=1e-9,
              rhotol=1e-8, diago_tol=1e-10, verbose=False)
    assert res.converged, f"{el} @ {v0_per_atom:.2f} did not converge"
    g = energy_param_grads(res, xc)
    e_atom = float(res.energies.total) / nat
    return e_atom, {n: (v / nat) for n, v in g.items()}


def _pressure_and_grad(el, xc, delta=0.01):
    """P(V_exp; θ) [eV/Å³] = −dE/dV and dP/dθ, by central FD in volume."""
    v = _EL[el][6]
    ep, gp = _energy_and_grads(el, v * (1 + delta), xc)
    em, gm = _energy_and_grads(el, v * (1 - delta), xc)
    dvol = 2.0 * delta * v
    p = -(ep - em) / dvol
    dp = {n: -(gp[n] - gm[n]) / dvol for n in gp}
    return p, dp


def _v0_of(el, xc, delta=0.03):
    """Equilibrium V0 [Å³/atom] of xc for element el, from a 3-point EOS min."""
    v = _EL[el][6]
    vs = np.array([v * (1 - delta), v, v * (1 + delta)])
    es = np.array([_energy_and_grads(el, vv, xc)[0] for vv in vs])
    # parabola vertex
    a, b, _c = np.polyfit(vs, es, 2)
    return -b / (2 * a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Al,Si,Pd,Ag")
    ap.add_argument("--test", default="Ge,Au,Pt")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fit", default="mu", choices=["mu", "kappa_mu"])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    train = [e.strip() for e in a.train.split(",") if e.strip()]
    test = [e.strip() for e in a.test.split(",") if e.strip()]

    xc = LearnableX(kappa=PBE_KAPPA, mu=PBE_MU)
    params = [xc.raw_mu] if a.fit == "mu" else [xc.raw_kappa, xc.raw_mu]
    opt = torch.optim.Adam(params, lr=a.lr)

    def report(tag):
        errs = {e: 100.0 * (_v0_of(e, xc) / _EL[e][6] - 1.0) for e in train + test}
        tr = np.mean([abs(errs[e]) for e in train])
        te = np.mean([abs(errs[e]) for e in test])
        print(f"  {tag}: κ={float(xc.kappa):.4f} μ={float(xc.mu):.4f} | "
              f"V0-err%% train MAE={tr:.3f} test MAE={te:.3f} | "
              + " ".join(f"{e}:{errs[e]:+.2f}" for e in train + test), flush=True)

    print(f"# functional learning: fit {a.fit} on train={train}, held-out test={test}", flush=True)
    print(f"# PBE reference: κ={PBE_KAPPA:.4f} μ={PBE_MU:.4f} (PBEsol μ≈0.1235)", flush=True)
    report("PBE (step 0)")
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        loss = 0.0
        gk = torch.zeros(())
        gm = torch.zeros(())
        for el in train:
            p, dp = _pressure_and_grad(el, xc)
            loss += p * p
            gk = gk + 2 * p * dp["raw_kappa"]
            gm = gm + 2 * p * dp["raw_mu"]
        if xc.raw_kappa in params:
            xc.raw_kappa.grad = gk.detach()
        xc.raw_mu.grad = gm.detach()
        opt.step()
        if step % 5 == 0 or step == a.steps:
            print(f"[step {step}] loss(ΣP²)={float(loss):.4e}", flush=True)
            report(f"step {step}")
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
