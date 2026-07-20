"""Train the PBE0 exchange mixing alpha through the self-consistent hybrid.

A hybrid mixes a fraction alpha of exact (Fock) exchange into the functional,

    E_xc = (1 - alpha) E_x^PBE + alpha E_x^Fock + E_c^PBE.

``hybrid_scf`` solves that fixed point (the Fock operator acts in every Davidson
step, ACE-compressed). At convergence the density is stationary, so the total
energy's derivative in alpha is the *explicit* one -- E_x^Fock - E_x^PBE on the
frozen converged orbitals -- and ``differentiable_hybrid_energy`` returns a scalar
that equals ``res.energies.total`` and carries exactly that gradient. That is the
same free-derivative-at-convergence argument the learnable-XC slot uses
(examples/train_xc_paw.py), now on the exchange-mixing parameter.

This script is the recovery sanity check: pick a target alpha*, record the
converged hybrid total energy E*, then start from a perturbed alpha and descend
(E(alpha) - E*)^2 back onto alpha*. Each step re-converges the hybrid SCF at the
current alpha (the stationary gradient is only exact at self-consistency) and
takes one backward pass for dE/dalpha. Si at Gamma, a loose cutoff -- the numbers
are illustrative of the machinery, not converged physics.

    uv run python examples/hybrid_train.py

Measured (CPU): recovers alpha* = 0.25 from a 0.10 start (final 0.251) in ~45
Adam steps, ~0.6 s per SCF, overshooting once before settling. Writes
examples/hybrid_train.json with the per-step (alpha, loss) history.
"""
import json
import time

import numpy as np
import torch

from gradwave.postscf.exchange_multik import HybridExchangeParams
from gradwave.postscf.hybrid import differentiable_hybrid_energy, hybrid_scf
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"

# Si in the diamond structure, one loose-cutoff Gamma cell reused every step.
a = 5.43
cell = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
pos = np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
upf = parse_upf(f"{PSE}/Si_ONCV_PBE-1.2.upf")


def system():
    return setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY, kmesh=(1, 1, 1),
                        nbands=8)


def converge(alpha, mode="full"):
    """One hybrid SCF at a fixed alpha (the SCF runs under no_grad)."""
    res = hybrid_scf(system(), alpha=float(alpha), mode=mode, smearing="none",
                     etol=1e-9, rhotol=1e-8, max_iter=80, verbose=False)
    assert res.converged, "hybrid SCF did not converge"
    return res


ALPHA_TARGET = 0.25
ALPHA_START = 0.10
N_STEPS = 45

# Reference: the converged PBE0 total energy at the target mixing.
e_target = float(converge(ALPHA_TARGET).energies.total)
print(f"target alpha* = {ALPHA_TARGET}  ->  E* = {e_target:+.6f} eV", flush=True)

# Train alpha from a perturbed start back onto alpha*.
params = HybridExchangeParams(alpha=ALPHA_START, mode="full")
opt = torch.optim.Adam(params.parameters(), lr=0.05)

history = []
t0 = time.time()
for step in range(N_STEPS):
    res = converge(params.alpha.detach())        # SCF at the current alpha (no_grad)
    e = differentiable_hybrid_energy(res, params)  # equals res total, grad in alpha
    loss = (e - e_target) ** 2
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    alpha = float(params.alpha.detach())
    history.append({"step": step, "alpha": alpha, "loss": float(loss.detach())})
    print(f"  step {step:2d}  alpha = {alpha:.4f}  loss = {float(loss.detach()):.3e}",
          flush=True)
    if float(loss.detach()) < 1e-4:
        break

alpha_final = float(params.alpha.detach())
print(f"recovered alpha = {alpha_final:.4f} (target {ALPHA_TARGET})  "
      f"in {len(history)} steps, {time.time() - t0:.0f}s", flush=True)

out = {
    "alpha_target": ALPHA_TARGET,
    "alpha_start": ALPHA_START,
    "alpha_final": alpha_final,
    "e_target_eV": e_target,
    "history": history,
}
with open("examples/hybrid_train.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote examples/hybrid_train.json", flush=True)
