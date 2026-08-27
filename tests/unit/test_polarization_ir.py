"""Unit tests for Berry-phase polarization, Born charges and IR (no SCF).

These exercise the pure linear-algebra / bookkeeping pieces:

- gauge invariance of the string Berry phase under per-k U(N) band rotations,
- Miller-index matching (incl. the end-of-string G-shift),
- the ionic reduced polarization and its lattice-translation quantum,
- Γ-mode IR intensities of a diatomic spring toy (acoustic dark, optic bright),
- the Lorentzian spectrum.

The physics that needs real wavefunctions (centrosymmetric P = 0, Z* of MgO,
the end-to-end IR spectrum) is validated on asus — see the PR description.
"""

import numpy as np
import torch

from gradwave.postscf.born import born_charges_autograd
from gradwave.postscf.ir import gamma_modes, ir_intensities, lorentzian_spectrum
from gradwave.postscf.phonons_supercell import build_supercell
from gradwave.postscf.polarization import (
    _match_indices,
    reduced_ionic_polarization,
    string_berry_phase,
)

RTOL = 1e-9


def _random_string(nk: int, nocc: int, npw: int, seed: int = 0):
    """A closed ring of random normalized occupied coefficients on a shared
    Miller set (so the wrap link with e_dir = 0 is a plain closed loop)."""
    g = torch.Generator().manual_seed(seed)
    miller = torch.arange(npw, dtype=torch.int64)[:, None] * torch.tensor([[1, 0, 0]])
    coeffs, millers = [], []
    for _ in range(nk):
        c = torch.randn(nocc, npw, generator=g, dtype=torch.float64) + 1j * torch.randn(
            nocc, npw, generator=g, dtype=torch.float64
        )
        # orthonormalize the occupied block (a proper Bloch subspace)
        q, _ = torch.linalg.qr(c.transpose(0, 1))
        coeffs.append(q.transpose(0, 1).contiguous())
        millers.append(miller.clone())
    return coeffs, millers


def _random_unitary(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64) + 1j * torch.randn(
        n, n, generator=g, dtype=torch.float64
    )
    q, r = torch.linalg.qr(a)
    # fix the phase so q is Haar-uniform-ish (not needed here, any unitary works)
    ph = torch.diagonal(r) / torch.diagonal(r).abs()
    return q * ph.conj()


def test_string_berry_phase_gauge_invariant():
    coeffs, millers = _random_string(nk=6, nocc=3, npw=12, seed=1)
    e_dir = (0, 0, 0)  # closed ring on a shared basis
    phi0 = string_berry_phase(coeffs, millers, e_dir)
    # rotate the occupied bands independently at every k-point
    rot = [
        (_random_unitary(3, 10 + k) @ coeffs[k]) for k in range(len(coeffs))
    ]
    phi1 = string_berry_phase(rot, millers, e_dir)
    # a Berry phase is gauge invariant only modulo 2*pi
    two_pi = 2.0 * torch.pi
    diff = torch.remainder(phi1 - phi0 + torch.pi, two_pi) - torch.pi
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-9)


def test_match_indices_gshift():
    mill_a = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    mill_b = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    # gshift 0: matches (0,1)->(0,1)
    ia, ib = _match_indices(mill_a, mill_b, (0, 0, 0))
    got = {
        (int(mill_a[i, 0]), int(mill_b[j, 0]))
        for i, j in zip(ia.tolist(), ib.tolist(), strict=True)
    }
    assert got == {(0, 0), (1, 1)}
    # gshift = (-1,0,0): mill_a[i] == mill_b[j] + (-1,0,0); mill_b+(-1) = {-1,0,-2}
    ia, ib = _match_indices(mill_a, mill_b, (-1, 0, 0))
    pairs = {
        (int(mill_a[i, 0]), int(mill_b[j, 0]))
        for i, j in zip(ia.tolist(), ib.tolist(), strict=True)
    }
    # mill_b + (-1,0,0) = {-1, 0, -2}; intersect with mill_a {0,1,2} => only 0 (from b=1)
    assert pairs == {(0, 1)}


def test_ionic_polarization_translation_quantum():
    cell = torch.diag(torch.tensor([4.0, 4.5, 5.0], dtype=torch.float64))
    pos = torch.tensor([[0.3, 0.7, 0.1], [1.9, 2.2, 3.0]], dtype=torch.float64)
    charges = torch.tensor([2.0, 6.0], dtype=torch.float64)  # neutral-ish toy
    p0 = reduced_ionic_polarization(cell, pos, charges)
    # translate ALL atoms by the lattice vector a_0
    pos_t = pos + cell[0]
    p1 = reduced_ionic_polarization(cell, pos_t, charges)
    dp = p1 - p0
    # p_ion shifts by sum(Z) along direction 0, 0 elsewhere
    expected = torch.tensor([float(charges.sum()), 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(dp, expected, atol=1e-10)


def _diatomic_toy():
    """A 2-atom cubic 'molecule' with a diagonal spring: acoustic + optic modes."""
    a = 4.0
    cell = np.eye(3) * a
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ cell
    species = [0, 1]
    scmap = build_supercell(cell, pos, species, (1, 1, 1))
    masses = np.array([24.305, 15.999])  # Mg, O amu
    # spring constant k [eV/Å²]; Φ_home (2,3,2,3): each atom self +k, cross -k
    k = 5.0
    phi = np.zeros((2, 3, 2, 3))
    for i in range(3):
        phi[0, i, 0, i] = k
        phi[1, i, 1, i] = k
        phi[0, i, 1, i] = -k
        phi[1, i, 0, i] = -k
    return phi, scmap, masses


def test_gamma_modes_diatomic_acoustic_and_optic():
    phi, scmap, masses = _diatomic_toy()
    freqs, vecs = gamma_modes(phi, scmap, masses)
    assert freqs.shape == (6,)
    # three near-zero acoustic modes, three degenerate optic modes > 0
    assert np.all(np.abs(freqs[:3]) < 1.0)
    assert np.all(freqs[3:] > 50.0)
    assert np.allclose(freqs[3], freqs[4]) and np.allclose(freqs[4], freqs[5])


def test_ir_intensities_optic_bright_acoustic_dark():
    phi, scmap, masses = _diatomic_toy()
    # opposite Born charges (Mg +2, O -2), isotropic
    born = np.zeros((2, 3, 3))
    born[0] = 2.0 * np.eye(3)
    born[1] = -2.0 * np.eye(3)
    ir = ir_intensities(phi, scmap, masses, born)
    inten = ir["intensity"]
    # acoustic (first 3) carry no net dipole; optic (last 3) are IR bright
    assert np.all(inten[:3] < 1e-8)
    assert np.all(inten[3:] > 1e-3)
    assert np.all(ir["ir_active"][3:])
    assert not np.any(ir["ir_active"][:3])


def test_lorentzian_spectrum_peaks_at_modes():
    freqs = np.array([-10.0, 0.0, 400.0, 400.0])
    inten = np.array([1.0, 1.0, 2.0, 2.0])
    grid, spec = lorentzian_spectrum(freqs, inten, width=5.0, npoints=2000)
    # imaginary/acoustic dropped; a peak near 400
    peak = grid[np.argmax(spec)]
    assert abs(peak - 400.0) < 3.0
    assert spec.max() > 0


def test_born_autograd_linear_polarization():
    # P/e(pos) = C @ pos_flat with a known Jacobian => Z* = V * dP/du recovers C.
    torch.manual_seed(0)
    na, vol = 2, 40.0
    cmat = torch.randn(3, na * 3, dtype=torch.float64)

    def pol_fn(pos: torch.Tensor) -> torch.Tensor:
        return cmat @ pos.reshape(-1)

    pos0 = torch.randn(na, 3, dtype=torch.float64)
    z = born_charges_autograd(pol_fn, pos0, vol)
    # z[kappa,a,b] = V * dP_a/du_{kappa,b} = V * cmat[a, 3*kappa+b]
    expected = vol * cmat.reshape(3, na, 3).permute(1, 0, 2)
    assert torch.allclose(z, expected, atol=1e-9)
