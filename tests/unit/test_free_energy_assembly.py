"""Free-energy / σ→0-extrapolation assembly: the sign of F = E − σS and the
½ in E₀ = (E + F)/2.

These are exactly the invariants api/summary.py consumes (the `"e0"` field is
``0.5*(e.total + e.free_energy)``, and `free_energy_per_atom` rides
`e.free_energy`). A silent E↔F swap or a dropped/duplicated ½ would still
produce plausible-looking numbers, so pin the algebra directly against an
independently-built entropy term. Light: no SCF, just the smearing kernel and
the EnergyBreakdown dataclass.
"""

import torch

from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.occupations import FermiDirac, Gaussian


def _t(x):
    return torch.tensor(float(x), dtype=torch.float64)


def _entropy_sum(smearing, width, xs, weights):
    """S = Σ 2·w·s(x) ≥ 0, the generalized entropy the SCF layer folds into
    the −σS term (occupations.Smearing.entropy contract)."""
    x = torch.tensor(xs, dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)
    return float((2.0 * w * smearing.entropy(x)).sum())


def _breakdown(entropy_term):
    """A fixed, arbitrary set of KS energy components plus a smearing term.
    Component values are irrelevant to the algebra — only that total() sums
    them and free_energy/e0 combine total() with the smearing term."""
    return EnergyBreakdown(
        kinetic=_t(3.0), hartree=_t(1.25), xc=_t(-2.5), local=_t(-1.75),
        nonlocal_=_t(0.5), ewald=_t(-4.0), smearing=entropy_term)


def test_free_energy_sign_and_e0_factor():
    """F = E − σS (entropy LOWERS the free energy) and E₀ = (E + F)/2."""
    width = 0.1
    # occupations straddling the Fermi level → a genuinely positive entropy
    S = _entropy_sum(Gaussian(), width,
                     xs=[-2.0, -0.5, 0.0, 0.5, 2.0],
                     weights=[0.2, 0.2, 0.2, 0.2, 0.2])
    assert S > 0.0                                  # partial occupations ⇒ S>0
    entropy_term = _t(-width * S)                   # −σS, the SCF's stored field

    e = _breakdown(entropy_term)
    E, F = e.total, e.free_energy

    # sign: F = E − σS, strictly below E for S>0 (a swap would give E + σS > E)
    assert torch.isclose(F, E - width * S, atol=1e-12, rtol=0.0)
    assert float(F) < float(E)

    # the ½ extrapolation, both as the property and as summary.py's expression
    assert torch.isclose(e.e0, 0.5 * (E + F), atol=1e-12, rtol=0.0)
    assert torch.isclose(e.e0, E + 0.5 * entropy_term, atol=1e-12, rtol=0.0)
    # E₀ sits exactly halfway between E and F
    assert torch.isclose(e.e0 - F, E - e.e0, atol=1e-12, rtol=0.0)


def test_free_energy_zero_smearing_collapses():
    """With no entropy (fixed occupations) F = E = E₀ — no phantom ½ term."""
    e = _breakdown(_t(0.0))
    assert torch.isclose(e.free_energy, e.total, atol=1e-14, rtol=0.0)
    assert torch.isclose(e.e0, e.total, atol=1e-14, rtol=0.0)


def test_free_energy_matches_fermi_dirac_entropy():
    """Same algebra with the Fermi-Dirac −[f ln f + (1−f) ln(1−f)] entropy, so
    the sign/factor pin is not specific to one smearing kernel."""
    width = 0.05
    S = _entropy_sum(FermiDirac(), width,
                     xs=[-1.0, -0.3, 0.0, 0.3, 1.0],
                     weights=[0.1, 0.3, 0.2, 0.3, 0.1])
    assert S > 0.0
    e = _breakdown(_t(-width * S))
    assert torch.isclose(e.free_energy, e.total - width * S, atol=1e-12, rtol=0.0)
    assert float(e.free_energy) < float(e.total)
    assert torch.isclose(e.e0, 0.5 * (e.total + e.free_energy), atol=1e-12, rtol=0.0)
