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


def as_real(x, device=None):
    """Tensor in the real working dtype (accepts array-likes)."""
    return torch.as_tensor(x, dtype=RDTYPE, device=device)


def as_complex(x, device=None):
    """Tensor in the complex working dtype (accepts array-likes)."""
    return torch.as_tensor(x, dtype=CDTYPE, device=device)


def resolve_device(spec: str | torch.device | None) -> torch.device:
    """Resolve a user device spec ('cpu' | 'cuda' | 'cuda:N' | None) to a torch.device."""
    if spec is None:
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device: cuda requested but torch.cuda.is_available() is False")
    return dev
