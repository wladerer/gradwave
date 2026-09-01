"""Surface energy from a slab-thickness sweep (Fiorentini-Methfessel).

Pure post-processing on (thickness, slab energy) points — no SCF dependency, so
it is trivially unit-testable in the same spirit as ``postscf.eos``. The energies
come from an SCF on a sequence of slabs of increasing thickness (``N`` layers or
formula units), all sharing the same in-plane surface area ``A`` and the same two
exposed faces.

The naive route, γ = (E_slab − N·E_bulk) / (2A), needs a separately converged
bulk reference energy E_bulk; the small difference of two large, differently
converged numbers diverges linearly with N (the well-known bulk-subtraction
problem). Fiorentini & Methfessel (J. Phys.: Condens. Matter 8, 6525, 1996) avoid
it by reading E_bulk **from the slope** of the slab sweep itself,

    E_slab(N) = 2 γ A + N · E_bulk,                              [eV]

a straight line in N whose slope is the per-layer bulk energy and whose intercept
is 2γA. Both surfaces of a symmetric slab are counted by the factor 2; for a slab
with two inequivalent faces the fit returns the *average* of the two surface
energies (their sum enters the intercept), which is all a single-face-free sweep
can resolve.

Base units follow the package: energy in eV, area in Å², so γ is eV/Å²
internally and converted to J/m² via ``EV_A2_TO_JM2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gradwave.constants import EV_A2_TO_JM2


@dataclass(frozen=True)
class SurfaceEnergyFit:
    """A Fiorentini-Methfessel surface-energy fit of a slab-thickness sweep.

    ``gamma`` is the (average, if the two faces differ) surface energy of one
    face; the factor of two surfaces has already been divided out. ``e_bulk`` is
    the per-layer bulk energy read off the slope, a free by-product of the method.
    """

    gamma: float               # eV/Å² — surface energy per unit area, one face
    e_bulk: float              # eV — bulk energy per layer/unit (the fit slope)
    area: float                # Å² — in-plane surface area used
    intercept: float           # eV — 2·γ·A
    rms_residual_eV: float     # RMS fit residual over the input points

    @property
    def gamma_Jm2(self) -> float:
        """Surface energy in J/m² (the conventional experimental unit)."""
        return self.gamma * EV_A2_TO_JM2

    @property
    def gamma_mJm2(self) -> float:
        """Surface energy in mJ/m²."""
        return self.gamma * EV_A2_TO_JM2 * 1e3


def surface_energy_fm(
    n_layers: np.ndarray | list[int] | list[float],
    e_slab: np.ndarray | list[float],
    area: float,
    *,
    n_surfaces: int = 2,
) -> SurfaceEnergyFit:
    """Fiorentini-Methfessel surface energy γ from a slab-thickness sweep.

    Fits ``E_slab(N) = intercept + N·E_bulk`` by least squares over the sweep and
    returns γ = intercept / (n_surfaces · area). ``n_layers`` counts the repeat
    unit (atomic layers or formula units) that ``E_bulk`` is *per*; it need not
    start at 1 or be contiguous, only be linear in the same unit as the energies.

    ``area`` is the in-plane surface area of one face [Å²] (= |a₁ × a₂| of the
    slab cell's two periodic surface vectors). ``n_surfaces`` is 2 for the usual
    slab exposing two equivalent faces; pass 1 only for a genuinely one-sided
    construction (e.g. a fixed/passivated bottom).

    At least two thicknesses are required; three or more let ``rms_residual_eV``
    diagnose non-linearity (quantum-size oscillations, unconverged thin slabs).
    """
    n = np.asarray(n_layers, dtype=float)
    e = np.asarray(e_slab, dtype=float)
    if n.shape != e.shape or n.ndim != 1:
        raise ValueError("n_layers and e_slab must be 1D arrays of equal length")
    if n.size < 2:
        raise ValueError("need at least two slab thicknesses to fit a slope")
    if np.ptp(n) == 0:
        raise ValueError("n_layers must span at least two distinct thicknesses")
    if area <= 0:
        raise ValueError("area must be positive")
    if n_surfaces < 1:
        raise ValueError("n_surfaces must be >= 1")

    # least-squares line E = intercept + slope·N
    slope, intercept = np.polyfit(n, e, 1)
    gamma = intercept / (n_surfaces * area)
    resid = e - (intercept + slope * n)
    rms = float(np.sqrt(np.mean(resid**2)))
    return SurfaceEnergyFit(
        gamma=float(gamma),
        e_bulk=float(slope),
        area=float(area),
        intercept=float(intercept),
        rms_residual_eV=rms,
    )


def surface_energy_subtraction(
    e_slab: float, n_layers: float, e_bulk: float, area: float,
    *, n_surfaces: int = 2,
) -> float:
    """Direct surface energy γ = (E_slab − N·E_bulk) / (n_surfaces·area) [eV/Å²].

    The textbook definition for a *single* slab given an independently converged
    per-layer bulk energy ``e_bulk``. Exposed mainly as a cross-check against
    ``surface_energy_fm`` (which instead reads ``e_bulk`` from the sweep slope);
    prefer the slope method when a full thickness sweep is available, since this
    form inherits the bulk-subtraction divergence for large ``n_layers``.
    """
    if area <= 0:
        raise ValueError("area must be positive")
    return (e_slab - n_layers * e_bulk) / (n_surfaces * area)
