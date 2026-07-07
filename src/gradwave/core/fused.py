"""Optionally torch.compile'd elementwise kernels for the memory-bound glue
between FFTs in the Hamiltonian applies.

The spinor potential mix (h_u = ψ_u·v_uu + ψ_d·v_ud, h_d = ψ_u·v_ud* + ψ_d·v_dd)
is four complex multiplies and two adds over grid-sized tensors — pure memory
traffic. As separate torch ops each intermediate makes a full HBM round trip;
fusing them into one kernel keeps them in registers, the win on a
bandwidth-limited GPU.

CRUCIAL: TorchInductor does NOT codegen complex operators — compiling the
complex form is *slower* than eager (it warns as much). The fusion only
materializes when the arithmetic is written on REAL components, which Inductor
fuses into a single kernel. Measured on an RTX 3050 at 80³/6 bands: eager
complex 2725 µs, compiled complex 3977 µs, compiled real-decomposition
1611 µs (1.69× over eager complex). So these kernels do the real/imag algebra
by hand and let compile fuse it.

torch.compile is used only on CUDA, compiled lazily on first call, and falls
back to eager permanently if the backend raises (e.g. Triton can't find
libcuda) — so CPU tests, unsupported builds, and the GRADWAVE_COMPILE=0 opt-out
are unaffected. On NixOS, Triton needs TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
(no /sbin/ldconfig); without it the eager fallback silently takes over.
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


def _spin_mix_full(pu, pd, v_uu, v_dd, vud_re, vud_im):
    """h_u = pu·v_uu + pd·v_ud, h_d = pu·v_ud* + pd·v_dd — real-component form."""
    pur, pui = pu.real, pu.imag
    pdr, pdi = pd.real, pd.imag
    hur = pur * v_uu + pdr * vud_re - pdi * vud_im
    hui = pui * v_uu + pdr * vud_im + pdi * vud_re
    hdr = pur * vud_re + pui * vud_im + pdr * v_dd
    hdi = -pur * vud_im + pui * vud_re + pdi * v_dd
    return torch.complex(hur, hui), torch.complex(hdr, hdi)


def _spin_mix_diag(pu, pd, v_uu, v_dd):
    """B⃗ = 0: h_u = pu·v_uu, h_d = pd·v_dd (v real) — real-component form."""
    return (torch.complex(pu.real * v_uu, pu.imag * v_uu),
            torch.complex(pd.real * v_dd, pd.imag * v_dd))


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


def spin_mix(psi_u, psi_d, v_uu, v_dd, vud_re, vud_im):
    return _dispatch("spin_mix_full", _spin_mix_full,
                     (psi_u, psi_d, v_uu, v_dd, vud_re, vud_im))


def spin_mix_diag(psi_u, psi_d, v_uu, v_dd):
    return _dispatch("spin_mix_diag", _spin_mix_diag, (psi_u, psi_d, v_uu, v_dd))
