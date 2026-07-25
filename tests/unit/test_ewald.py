import numpy as np
import torch

from gradwave.constants import E2
from gradwave.core.energies.ewald import ewald_energy

MADELUNG_NACL = 1.7475645946331822  # per ion pair, ref. nearest-neighbor distance
MADELUNG_CSCL = 1.7626747730990846


def test_nacl_madelung():
    a = 5.64
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])  # fcc primitive
    pos = torch.tensor([[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0]], dtype=torch.float64)
    q = torch.tensor([1.0, -1.0], dtype=torch.float64)
    e = ewald_energy(pos, q, cell)
    ref = -MADELUNG_NACL * E2 / (a / 2)
    assert abs(e.item() - ref) / abs(ref) < 1e-8


def test_cscl_madelung():
    a = 4.11
    cell = a * np.eye(3)
    pos = torch.tensor([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]], dtype=torch.float64)
    q = torch.tensor([1.0, -1.0], dtype=torch.float64)
    e = ewald_energy(pos, q, cell)
    d = a * np.sqrt(3) / 2
    ref = -MADELUNG_CSCL * E2 / d
    assert abs(e.item() - ref) / abs(ref) < 1e-8


def test_eta_independence():
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = torch.tensor([[0.0, 0.0, 0.0], [a / 4 + 0.05, a / 4, a / 4 - 0.1]], dtype=torch.float64)
    q = torch.tensor([4.0, 4.0], dtype=torch.float64)
    e1 = ewald_energy(pos, q, cell, eta=0.3)
    e2 = ewald_energy(pos, q, cell, eta=0.9)
    assert abs(e1.item() - e2.item()) < 1e-9 * abs(e1.item())


def test_eta_independence_low_symmetry():
    """A converged Ewald sum is η-independent for ANY cell. Low-symmetry
    (large, anisotropic) cells are the regression that a cubic-only η test
    masks: the auto-η is small, rcut = _ACC/√η rivals the intra-cell
    separations, and an unpadded |R|≤rcut real-space cutoff drops near pairs
    whose |d| = |τ_a − τ_b + R| < rcut — leaking an η-dependent error. Target:
    η-flat to well under 0.1 meV/atom across η ∈ [0.2, 2.0]."""
    cell = np.array([[9.2, 0.0, 0.0], [1.3, 6.1, 0.0], [0.8, 1.1, 4.7]])  # triclinic
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [3.1, 2.2, 1.0], [6.0, 4.5, 3.3], [1.5, 5.0, 2.0]],
        dtype=torch.float64,
    )
    q = torch.tensor([2.0, -1.0, 2.0, -3.0], dtype=torch.float64)
    na = pos.shape[0]
    etas = (0.2, 0.3, 0.5, 0.9, 1.5, 2.0)
    vals = [ewald_energy(pos, q, cell, eta=eta).item() / na for eta in etas]
    assert max(vals) - min(vals) < 1e-4  # eV/atom, i.e. < 0.1 meV/atom


def test_nio_mineral_ewald_vs_qe_convention():
    """Non-cubic (rhombohedral AFM-primitive) NiO ion–ion Ewald, formal
    charges Ni²⁺/O²⁻, anchored to an independent QE-convention point-charge
    Ewald (pymatgen's EwaldSummation, acc_factor=16). Guards the low-symmetry
    fix against silent regression and confirms the absolute constant, not just
    η-flatness."""
    cell = np.array(
        [
            [1.4743176388, 0.8511976856, 4.815101245],
            [-1.4743176388, 0.8511976856, 4.815101245],
            [0.0, -1.7023953712, 4.815101245],
        ]
    )
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 7.2226518676],
            [0.0, 0.0, 10.8339778013],
            [0.0, 0.0, 3.6113259338],
        ],
        dtype=torch.float64,
    )
    q = torch.tensor([2.0, 2.0, -2.0, -2.0], dtype=torch.float64)  # Ni, Ni, O, O
    ref = -96.55370972206813  # eV, pymatgen EwaldSummation (independent, QE convention)
    e = ewald_energy(pos, q, cell)
    assert abs(e.item() - ref) < 4e-4  # < 0.1 meV/atom over 4 atoms


def test_translation_invariance_and_force_sum_rule():
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    base = torch.tensor([[0.0, 0.0, 0.0], [a / 4 + 0.07, a / 4 - 0.03, a / 4]], dtype=torch.float64)
    q = torch.tensor([4.0, 4.0], dtype=torch.float64)
    shift = torch.tensor([0.31, -0.12, 0.55], dtype=torch.float64)
    e1 = ewald_energy(base, q, cell)
    e2 = ewald_energy(base + shift, q, cell)
    assert abs(e1.item() - e2.item()) < 1e-10 * abs(e1.item())

    pos = base.clone().requires_grad_(True)
    e = ewald_energy(pos, q, cell)
    (grad,) = torch.autograd.grad(e, pos)
    assert torch.abs(grad.sum(dim=0)).max() < 1e-9  # Σ_a F_a = 0


def test_forces_match_finite_differences():
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    base = torch.tensor([[0.0, 0.0, 0.0], [a / 4 + 0.07, a / 4 - 0.03, a / 4]], dtype=torch.float64)
    q = torch.tensor([4.0, 4.0], dtype=torch.float64)

    pos = base.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(ewald_energy(pos, q, cell), pos)

    h = 1e-5
    for comp in range(3):
        dp = torch.zeros_like(base)
        dp[1, comp] = h
        ep = ewald_energy(base + dp, q, cell).item()
        em = ewald_energy(base - dp, q, cell).item()
        fd = (ep - em) / (2 * h)
        assert abs(fd - grad[1, comp].item()) < 1e-7
