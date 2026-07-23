"""Numeric precision and device policy.

All physics runs in float64 / complex128. Single precision is not supported:
SCF convergence to 1e-8 eV and autograd gradchecks both require it.
"""

import torch

RDTYPE = torch.float64
CDTYPE = torch.complex128

# Loose-tolerance ("draft") precision for the early Davidson iterations on GPU.
# fp32 GEMMs/FFTs run several× faster on GeForce; the SCF's own adaptive
# diago-tolerance schedule re-polishes in fp64 once the residual tightens, so
# the converged answer is bit-compatible with the all-fp64 path.
CDTYPE_LOW = torch.complex64
RDTYPE_LOW = torch.float32


def real_of(cdtype: torch.dtype) -> torch.dtype:
    """Real dtype paired with a complex dtype (complex64→float32, else float64)."""
    return torch.float32 if cdtype == torch.complex64 else torch.float64
