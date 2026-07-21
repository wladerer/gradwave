"""XC functional interface (Layer A).

An XCFunctional maps grid densities to an XC energy density; the potential
v_xc = δE_xc/δρ is obtained by autograd — one differentiable implementation
serves as (a) the potential generator inside SCF, (b) the twice-differentiable
energy term for forces/Hessians, and (c) the trainable object for functional
learning (parameters are ordinary nn.Module parameters).

Internally functionals convert ρ [e/Å³] to atomic units, evaluate the
standard Hartree-a.u. expressions, and return e_xc [eV/Å³]:

    E_xc = (Ω/N) Σ_j e_xc(r_j)

GGA functionals receive σ = |∇ρ|² computed spectrally by the caller
(density.py) INSIDE the autograd graph, so autograd's v_xc automatically
contains the −∇·(∂e/∂∇ρ) term with spectral accuracy.

NaN discipline: no torch.where over expressions that are NaN in the dead
branch (NaN·0 = NaN in backward). Densities are floored via clamp before
fractional powers/logs; test densities sit far above the floor.
"""

from __future__ import annotations

import contextlib
import threading

import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV

RHO_FLOOR_AU = 1e-14  # a.u.; well below any physical grid density

# Thread-local switch that forces eager energy_density even when a functional
# has compile enabled. torch.compile with aot_autograd does not support double
# backward, and the f_xc kernel (dielectric/Newton/Stoner/learned-U) is exactly
# a double backward through E_xc, so those call sites wrap themselves in
# xc_eager() to stay on the eager path. The failure would otherwise raise inside
# the caller's second grad(), past any try/except here.
_EAGER_TLS = threading.local()


def _force_eager() -> bool:
    return getattr(_EAGER_TLS, "on", False)


@contextlib.contextmanager
def xc_eager():
    """Force any compile-enabled XC functional onto its eager path inside this
    block. Required around f_xc HVPs, since compiled code cannot double-backward."""
    prev = getattr(_EAGER_TLS, "on", False)
    _EAGER_TLS.on = True
    try:
        yield
    finally:
        _EAGER_TLS.on = prev


class CompilableXC:
    """Mixin, opt-in torch.compile of energy_density with an eager fallback.

    The XC transcendental chain is real-valued and fuses well under Inductor,
    unlike the complex FFT-bound Hamiltonian apply where two earlier attempts
    measured no gain. Compilation is lazy and
    per-instance. Any compile or runtime error, a missing host toolchain on a
    stock checkout or a Triton libcuda gap, latches the eager path so nothing
    downstream breaks. Route every hot caller through eval_energy_density so the
    flag reaches the PAW one-center loop, not just energy().

    Scope is first order, the compiled forward and its single backward v_xc. This
    PyTorch's aot_autograd cannot double-backward through compiled code, and the
    f_xc kernel is a double backward through E_xc, so those response and HVP call
    sites wrap their xc.energy() in xc_eager() to force this path back to eager.
    """

    _xc_compile_on: bool = False
    _xc_compile_dead: bool = False
    _xc_compile_dynamic: bool = True
    _xc_compiled = None
    _xc_compile_kwargs = None
    _xc_compile_error = None

    def enable_compile(self, dynamic: bool = True, **kwargs):
        """Turn on the compiled energy_density path. dynamic=True avoids a
        recompile per grid size, which otherwise dominates short runs."""
        self._xc_compile_on = True
        self._xc_compile_dynamic = dynamic
        self._xc_compile_kwargs = kwargs
        self._xc_compiled = None
        self._xc_compile_dead = False
        self._xc_compile_error = None
        return self

    def disable_compile(self):
        self._xc_compile_on = False
        self._xc_compiled = None
        return self

    def eval_energy_density(self, *args):
        """energy_density through the compiled callable when enabled, else eager.

        The compiled path serves the forward and its single backward (v_xc), which
        torch.compile handles correctly, including the GGA case where sigma is a
        function of rho in the outer graph. Double backward (f_xc) is unsupported,
        so those callers force this back to eager with xc_eager(). The eager
        fallback re-runs energy_density itself, so a genuine bug in the functional
        still raises. Only compile-layer failures are swallowed, and they latch
        _xc_compile_dead with the message on _xc_compile_error.
        """
        if not self._xc_compile_on or self._xc_compile_dead or _force_eager():
            return self.energy_density(*args)
        if self._xc_compiled is None:
            try:
                self._xc_compiled = torch.compile(
                    self.energy_density,
                    dynamic=self._xc_compile_dynamic,
                    **(self._xc_compile_kwargs or {}),
                )
            except Exception as exc:  # toolchain absent, degrade to eager
                self._latch_eager(exc)
                return self.energy_density(*args)
        try:
            return self._xc_compiled(*args)
        except Exception as exc:  # first-call compile failure (toolchain gap)
            self._latch_eager(exc)
            return self.energy_density(*args)

    def _latch_eager(self, exc: Exception) -> None:
        """Permanently fall back to the eager path and warn once. The latch runs
        only on the first failure (subsequent calls short-circuit on
        _xc_compile_dead), so the warning fires exactly once and makes the
        resulting performance cliff visible instead of silent."""
        import warnings

        self._xc_compile_dead = True
        self._xc_compile_error = repr(exc)
        warnings.warn(
            f"XC compile disabled, falling back to the eager energy_density path "
            f"for the rest of this run: {self._xc_compile_error}",
            stacklevel=2,
        )


class XCFunctional(CompilableXC, torch.nn.Module):
    """Base class. Subclasses implement energy_density()."""

    needs_gradient: bool = False  # True for GGAs
    needs_tau: bool = False  # True for meta-GGAs (depend on the kinetic-energy density τ)

    def energy_density(
        self,
        rho: torch.Tensor,
        sigma: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """e_xc [eV/Å³] pointwise. rho [e/Å³]; sigma = |∇ρ|² [e²/Å⁸] for GGAs;
        tau = ½Σf|∇ψ|² [e/Å⁵] for meta-GGAs.

        τ is an independent orbital field, not a functional of ρ on the grid, so
        (unlike σ) it does not ride the ρ autograd graph: its potential
        v_τ = ∂e/∂τ acts as the generalized-KS operator −½∇·(v_τ∇ψ), wired into
        H separately (see core.metagga). Non-meta functionals ignore the argument.
        """
        raise NotImplementedError

    def energy(
        self,
        rho: torch.Tensor,
        volume: float,
        sigma: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """E_xc [eV] = (Ω/N)Σ e_xc."""
        e = self.eval_energy_density(rho, sigma, tau)
        return e.sum() * (volume / e.numel())


def to_au(rho: torch.Tensor) -> torch.Tensor:
    """ρ [e/Å³] → ρ [e/bohr³], floored for NaN-safe powers/logs."""
    return torch.clamp(rho * BOHR_ANG**3, min=RHO_FLOOR_AU)


def eps_to_ev_density(rho_ang: torch.Tensor, eps_au: torch.Tensor) -> torch.Tensor:
    """ε [Ha/electron] → e_xc [eV/Å³] = ρ[e/Å³]·ε[eV]."""
    return rho_ang * eps_au * HARTREE_EV
