"""Property-based contracts for the ion–ion Ewald sum.

``tests/unit/test_ewald.py`` pins the Ewald energy against hardcoded reference
cells (NaCl/CsCl Madelung constants, a triclinic η-flatness case, a QE-convention
NiO point). Those anchor the *absolute* constant. The tests here are the
complementary half: exact *invariances* that must hold for **any** cell, atom
count, geometry, and charge assignment, checked over Hypothesis-generated
configurations rather than one hand-picked geometry.

Each is a metamorphic relation — a transform of the input whose effect on the
output is known exactly — so no reference value is needed:

- translation of all atoms            → energy unchanged (E depends only on τ_a−τ_b)
- Σ_a F_a = 0                          → the force sum rule (translation ⇒ ΣF=0)
- relabelling atoms                    → energy unchanged (permutation invariance)
- shifting one atom by a lattice vec   → energy unchanged (lattice periodicity)
- scaling every charge by λ            → energy scales by λ² (bilinear in charge)
- changing the Ewald split η           → energy unchanged (converged sum)

The lattice-vector-shift and η-independence cases in particular exercise the
real-space image padding in ``_max_pair_offset`` (see ewald.py): a shifted atom
inflates the raw pair offset, and small η inflates rcut, both of which must leave
the converged energy fixed. All are fast (pure tensor sums, no SCF), so they sit
in the fast tier.
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from gradwave.core.energies.ewald import ewald_energy

# Minimum interatomic separation (Å) the strategy will admit. Two *distinct*
# point charges at the same site is a genuine physical singularity (E → ∞), not
# a code defect, so configurations below this floor are rejected rather than
# asserted on — the same sane-geometry assumption the example-based tests make.
_MIN_SEP = 0.7


def _rel(a: float, b: float) -> float:
    """Relative difference, floored so near-zero magnitudes stay well-defined."""
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def _min_image_distance(pos: np.ndarray, cell: np.ndarray) -> float:
    """Smallest distance between any two distinct atoms, over the 27 nearest
    periodic images — enough to catch a near-coincidence brought in by wrap."""
    na = pos.shape[0]
    if na < 2:
        return np.inf
    shells = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    images = shells @ cell
    best = np.inf
    for a in range(na):
        for b in range(a + 1, na):
            d = pos[a] - pos[b] + images
            best = min(best, float(np.linalg.norm(d, axis=1).min()))
    return best


@st.composite
def configs(draw, min_atoms: int = 1):
    """A random neutral-or-charged crystal: (cell, positions, charges).

    The cell is lower-triangular — three lengths in [4, 7] Å on the diagonal and
    bounded shear below it — which guarantees a strictly positive volume and a
    well-conditioned lattice, so the Ewald image lists stay small enough for the
    fast tier regardless of what Hypothesis draws. Positions are placed by random
    fractional coordinates (kept in-cell); charges are nonzero (|Z| ∈ [0.5, 4])
    with a random sign so the energy magnitude never collapses to zero and the
    relative tolerances stay meaningful. Charge neutrality is deliberately not
    imposed — the neutralizing background term makes every property below hold
    for non-neutral cells too.
    """
    l1, l2, l3 = (draw(st.floats(4.0, 7.0)) for _ in range(3))
    s21, s31, s32 = (draw(st.floats(-1.0, 1.0)) for _ in range(3))
    cell = np.array([[l1, 0.0, 0.0], [s21, l2, 0.0], [s31, s32, l3]], dtype=np.float64)

    na = draw(st.integers(min_atoms, 4))
    frac = np.array(
        [[draw(st.floats(0.0, 1.0)) for _ in range(3)] for _ in range(na)],
        dtype=np.float64,
    )
    pos = frac @ cell
    # reject unphysical near-coincident atoms (see _MIN_SEP)
    assume(_min_image_distance(pos, cell) > _MIN_SEP)

    charges = [
        draw(st.floats(0.5, 4.0)) * draw(st.sampled_from((-1.0, 1.0))) for _ in range(na)
    ]
    return {
        "cell": cell,
        "pos": torch.tensor(pos, dtype=torch.float64),
        "charges": torch.tensor(charges, dtype=torch.float64),
    }


@settings(max_examples=50, deadline=None)
@given(cfg=configs(), shift=st.lists(st.floats(-3.0, 3.0), min_size=3, max_size=3))
def test_translation_invariance(cfg, shift):
    """A rigid translation of every atom leaves the Ewald energy unchanged."""
    s = torch.tensor(shift, dtype=torch.float64)
    e0 = ewald_energy(cfg["pos"], cfg["charges"], cfg["cell"]).item()
    e1 = ewald_energy(cfg["pos"] + s, cfg["charges"], cfg["cell"]).item()
    assert _rel(e0, e1) < 1e-8


@settings(max_examples=50, deadline=None)
@given(cfg=configs())
def test_force_sum_rule(cfg):
    """Σ_a F_a = 0: translation invariance ⇒ the analytic forces sum to zero."""
    pos = cfg["pos"].clone().requires_grad_(True)
    e = ewald_energy(pos, cfg["charges"], cfg["cell"])
    (grad,) = torch.autograd.grad(e, pos)  # ∂E/∂τ = −F
    net = float(grad.sum(dim=0).abs().max())
    assert net < 1e-8 * float(grad.abs().max()) + 1e-9


@settings(max_examples=50, deadline=None)
@given(cfg=configs(min_atoms=2), data=st.data())
def test_permutation_invariance(cfg, data):
    """Relabelling the atoms (positions and charges together) is a no-op."""
    na = cfg["pos"].shape[0]
    perm = list(data.draw(st.permutations(range(na))))
    e0 = ewald_energy(cfg["pos"], cfg["charges"], cfg["cell"]).item()
    e1 = ewald_energy(cfg["pos"][perm], cfg["charges"][perm], cfg["cell"]).item()
    assert _rel(e0, e1) < 1e-9


@settings(max_examples=50, deadline=None)
@given(
    cfg=configs(min_atoms=2),
    n=st.lists(st.integers(-2, 2), min_size=3, max_size=3),
)
def test_lattice_vector_shift_invariance(cfg, n):
    """Moving one atom by a lattice vector R = n·cell leaves the energy fixed.

    This is the invariance the ``_max_pair_offset`` padding exists to preserve:
    the shifted atom pushes the raw pairwise offset well outside the cell, so an
    unpadded real-space sphere would silently drop near pairs and leak an error.
    """
    r = torch.tensor(np.asarray(n, dtype=np.float64) @ cfg["cell"], dtype=torch.float64)
    shifted = cfg["pos"].clone()
    shifted[0] = shifted[0] + r
    e0 = ewald_energy(cfg["pos"], cfg["charges"], cfg["cell"]).item()
    e1 = ewald_energy(shifted, cfg["charges"], cfg["cell"]).item()
    assert _rel(e0, e1) < 1e-7


@settings(max_examples=50, deadline=None)
@given(
    cfg=configs(),
    lam=st.floats(-3.0, 3.0).filter(lambda x: abs(x) > 0.1),
)
def test_energy_is_quadratic_in_charge(cfg, lam):
    """E(λZ) = λ² E(Z): every term is bilinear/quadratic in the charges.

    Subsumes the sign-flip invariance E(−Z) = E(Z) at λ = −1.
    """
    e0 = ewald_energy(cfg["pos"], cfg["charges"], cfg["cell"]).item()
    e1 = ewald_energy(cfg["pos"], lam * cfg["charges"], cfg["cell"]).item()
    assert _rel(e1, lam * lam * e0) < 1e-8


@settings(max_examples=40, deadline=None)
@given(
    cfg=configs(),
    eta1=st.floats(0.3, 1.2),
    eta2=st.floats(0.3, 1.2),
)
def test_eta_independence(cfg, eta1, eta2):
    """A converged Ewald sum is independent of the real/reciprocal split η."""
    e1 = ewald_energy(cfg["pos"], cfg["charges"], cfg["cell"], eta=eta1).item()
    e2 = ewald_energy(cfg["pos"], cfg["charges"], cfg["cell"], eta=eta2).item()
    assert _rel(e1, e2) < 1e-6
