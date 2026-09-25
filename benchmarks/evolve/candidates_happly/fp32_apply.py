"""Run the whole apply in complex64 and upcast the result. Non-bit-exact (~1e-6)
-- valid only for Davidson *expansion*-vector applies (re-certified in fp64
before any band is declared converged), so it is admissible only at the loose
"expansion" gate, never the exact one. Demonstrates the tolerance oracle
admitting a genuinely approximate candidate while keeping its error visible."""

import torch


def apply(H, c):
    return H.apply(c.to(torch.complex64)).to(c.dtype)
