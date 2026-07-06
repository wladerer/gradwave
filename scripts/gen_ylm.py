"""Dev-time verifier for core/ylm.py against sympy's real spherical harmonics.

Run manually (sympy is a dev dependency):  uv run python scripts/gen_ylm.py

Checks our hard-coded Cartesian forms against sympy's Znm (real spherical
harmonics) evaluated at random directions, for every (l, m) channel up to
l = 3, allowing only a global ±1 sign per channel (signs cancel in all
physical contractions; see core/ylm.py docstring). Exits nonzero on failure.
"""

import sys

import numpy as np
import sympy as sp
import torch

from gradwave.core.ylm import ylm_all

# our ordering: (l, m) = (0,0),(1,0),(1,1),(1,-1),(2,0),(2,1),(2,-1),...
ORDER = [(0, 0)]
for l in range(1, 4):
    ORDER.append((l, 0))
    for m in range(1, l + 1):
        ORDER += [(l, m), (l, -m)]

theta_s, phi_s = sp.symbols("theta phi", real=True)
rng = np.random.default_rng(11)
pts = rng.normal(size=(200, 3))
pts /= np.linalg.norm(pts, axis=1, keepdims=True)
theta = np.arccos(np.clip(pts[:, 2], -1, 1))
phi = np.arctan2(pts[:, 1], pts[:, 0])

ours = ylm_all(3, torch.as_tensor(pts, dtype=torch.float64)).numpy()

fail = False
for idx, (l, m) in enumerate(ORDER):
    z = sp.Znm(l, m, theta_s, phi_s)
    f = sp.lambdify((theta_s, phi_s), sp.re(z.expand(func=True)), "numpy")
    ref = np.asarray(f(theta, phi), dtype=np.float64)
    v = ours[:, idx]
    err_plus = np.abs(v - ref).max()
    err_minus = np.abs(v + ref).max()
    err = min(err_plus, err_minus)
    sign = "+" if err_plus <= err_minus else "-"
    status = "ok" if err < 1e-12 else "FAIL"
    if err >= 1e-12:
        fail = True
    print(f"(l={l}, m={m:+d})  sign {sign}  max|Δ| = {err:.2e}  {status}")

sys.exit(1 if fail else 0)
