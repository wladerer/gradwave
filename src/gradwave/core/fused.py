"""Optionally torch.compile'd elementwise kernels for the memory-bound glue
between FFTs in the Hamiltonian applies.

The spinor potential mix (h_u = ψ_u·v_uu + ψ_d·v_ud, h_d = ψ_u·v_ud* + ψ_d·v_dd)
is four complex multiplies and two adds over grid-sized tensors — pure memory
traffic. As separate torch ops each intermediate makes a full HBM round trip;
fusing them into one Inductor kernel keeps them in registers, which is exactly
the win on a bandwidth-limited GPU.

torch.compile is used only on CUDA, compiled lazily on first call, and falls
back to eager permanently if the backend raises — so CPU tests, unsupported
torch builds, and the GRADWAVE_COMPILE=0 opt-out are all unaffected.
"""

from __future__ import annotations

import os

import torch

_ENABLED = os.environ.get("GRADWAVE_COMPILE", "1") != "0"
_cache: dict = {}


def set_compile(enabled: bool) -> None:
    """Toggle Inductor fusion at runtime (benchmarks compare on/off). Off also
    means eager everywhere; on restores CUDA-only compilation."""
    global _ENABLED
    _ENABLED = bool(enabled)


def _spin_mix_full(psi_u, psi_d, v_uu, v_dd, v_ud):
    h_u = psi_u * v_uu + psi_d * v_ud
    h_d = psi_u * v_ud.conj() + psi_d * v_dd
    return h_u, h_d


def _spin_mix_diag(psi_u, psi_d, v_uu, v_dd):
    return psi_u * v_uu, psi_d * v_dd


def _dispatch(name, fn, args):
    """Run `fn` through its cached compiled kernel on CUDA, falling back to
    eager on the first backend error (and permanently thereafter)."""
    if not _ENABLED or not getattr(args[0], "is_cuda", False):
        return fn(*args)
    entry = _cache.get(name)
    if entry is None:
        try:
            compiled = torch.compile(fn, dynamic=True)
        except Exception:
            compiled = fn
        entry = {"fn": compiled, "eager": fn, "ok": compiled is not fn}
        _cache[name] = entry
    if entry["ok"]:
        try:
            return entry["fn"](*args)
        except Exception:
            entry["ok"] = False  # Inductor/Triton failed → eager from now on
    return entry["eager"](*args)


def spin_mix(psi_u, psi_d, v_uu, v_dd, v_ud):
    return _dispatch("spin_mix_full", _spin_mix_full,
                     (psi_u, psi_d, v_uu, v_dd, v_ud))


def spin_mix_diag(psi_u, psi_d, v_uu, v_dd):
    return _dispatch("spin_mix_diag", _spin_mix_diag, (psi_u, psi_d, v_uu, v_dd))
