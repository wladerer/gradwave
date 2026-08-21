"""Empirical Levels 2 & 3 of the EFG-gradcheck diagnostic: does autograd reach V_zz?

Cheap TiO2 config (correctness of the value is irrelevant here — we only need a structurally
valid ``info['efg']`` and to interrogate its autograd lineage). Demonstrates, at runtime, the
code-reading conclusion: the FLAPW SCF + EFG are numpy/scipy, so ``V_zz`` is a plain Python
float with no torch graph, and ``torch.autograd.grad(V_zz, u_tensor)`` cannot differentiate it.
"""
from __future__ import annotations

import numpy as np
import torch

from gradwave.flapw import crystal_scf_multi

A = 8.68083
C0 = 5.59096
RADII = {"Ti": 1.098, "O": 0.824}
CFG = dict(ecut=120.0, lmax=2, iters=4, tol=1e-3, kmesh=(1, 1, 1), smearing=0.0,
           efg=True, fullpot=False, kworkers=1)


def atoms_for(u):
    return [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
            ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
            ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]


def main():
    print("=== LEVEL 3: autograd lineage of V_zz through crystal_scf_multi ===", flush=True)
    _b, info = crystal_scf_multi([A, A, C0], atoms_for(0.3048), RADII, **CFG)
    vzz = info["efg"]["a0"]["V_zz"]
    print(f"type(V_zz) = {type(vzz).__name__};  value = {vzz:+.4f}", flush=True)
    print(f"isinstance(V_zz, torch.Tensor) = {isinstance(vzz, torch.Tensor)}", flush=True)
    print(f"has grad_fn = {getattr(vzz, 'grad_fn', None) is not None}", flush=True)
    tensor = info["efg"]["a0"]["tensor"]
    print(f"type(efg tensor) = {type(tensor).__name__} (numpy => no graph)", flush=True)

    print("\n=== LEVEL 2/3: torch.autograd.grad(V_zz, u) — the refinement one-liner ===",
          flush=True)
    u = torch.tensor(0.3048, dtype=torch.float64, requires_grad=True)
    # Build atoms from the tensor u; crystal_scf_multi immediately does np.asarray(frac) @ A,
    # so the tensor is detached at the very first line of _multi_setup.
    atoms = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
             ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
             ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]
    _b2, info2 = crystal_scf_multi([A, A, C0], atoms, RADII, **CFG)
    vzz2 = info2["efg"]["a0"]["V_zz"]
    try:
        vzz_t = torch.as_tensor(vzz2)
        (g,) = torch.autograd.grad(vzz_t, u)
        print(f"UNEXPECTED: got gradient {g}", flush=True)
    except (RuntimeError, TypeError) as e:
        print(f"torch.autograd.grad FAILED as expected:\n  {type(e).__name__}: {e}", flush=True)
    print(f"\nu.grad after a full SCF+EFG = {u.grad}  (None => u never entered any graph)",
          flush=True)
    print(f"u still leaf & requires_grad = {u.is_leaf and u.requires_grad}; "
          f"but np.asarray(u) severs it: {type(np.asarray([float(u)])).__name__}", flush=True)


if __name__ == "__main__":
    main()
