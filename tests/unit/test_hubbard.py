"""DFT+U core: PP_PSWFC parsing, atomic-orbital projectors, occupation matrix.

The decisive structural test is the projector self-overlap on an isolated atom
at Γ: ⟨φ_m|φ_m'⟩ must be diagonal (Ylm m-orthogonality) with equal diagonal
entries (the (−i)^l phase, 4π/√Ω factor, and radial form factor are shared
across m). The diagonal value < 1 is the finite-cutoff capture of the orbital
— QE's `atomic` projection uses the orbitals un-renormalized, same as here.
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.hubbard import (
    HubbardManifold,
    build_hubbard_projectors,
    hubbard_dmatrix,
    hubbard_energy,
    hubbard_projectors,
    manifold_radial,
    occupation_matrices,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from tests.helpers import RY, system_device

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
def test_pswfc_parsed_and_normalized():
    se = parse_upf(FIX / "pseudos" / "PD_Se_FR.upf")
    assert len(se.pswfc) == 5
    d = se.hubbard_orbitals(2)
    assert len(d) == 2 and {o.j for o in d} == {2.5, 1.5}  # j-split 3D
    for o in se.pswfc:
        norm = float(np.sum(o.rchi**2 * se.rab))  # ∫(r·R)² dr in Å
        assert abs(norm - 1.0) < 1e-4
    # (2j+1)-combined scalar radial is renormalized
    rchi = manifold_radial(se, 2)
    assert abs(float(np.sum(rchi**2 * se.rab)) - 1.0) < 1e-10


def test_projector_overlap_diagonal_isolated_atom():
    torch.set_num_threads(4)
    se = parse_upf(FIX / "pseudos" / "PD_Se_FR.upf")
    system = setup_system(9.0 * np.eye(3), np.zeros((1, 3)), [0], [se],
                          ecut=40 * RY, kmesh=(1, 1, 1))
    hub = build_hubbard_projectors(system, [HubbardManifold(0, l=2, u=5.0, j=0.8)])
    assert hub.nproj == 5 and hub.n_sites == 1
    q = hubbard_projectors(hub, system.positions)[0]  # (5, npw) at Γ
    s = (q.conj() @ q.T).real
    diag = torch.diag(s)
    off = s - torch.diag(diag)
    assert float(off.abs().max()) < 1e-9  # Ylm m-orthogonality (exact)
    # diagonal entries equal up to FFT-grid anisotropy (the discrete G-sphere
    # is not perfectly isotropic; shrinks with denser grids — same effect as
    # the σ-state irrep splitting)
    assert float(diag.max() - diag.min()) < 2e-3  # rotational consistency
    assert 0.9 < float(diag.mean()) < 1.0  # finite-cutoff capture


def test_occupation_matrix_and_energy_algebra():
    """Synthetic ⟨φ|ψ⟩ → n Hermitian; E_U and D match the Dudarev formulas."""
    torch.manual_seed(0)
    # one site, d manifold (dim 5), 3 k-points, 4 bands
    se = parse_upf(FIX / "pseudos" / "PD_Se_FR.upf")
    system = setup_system(9.0 * np.eye(3), np.zeros((1, 3)), [0], [se],
                          ecut=25 * RY, kmesh=(1, 1, 1))
    U, J = 5.0, 0.8
    hub = build_hubbard_projectors(system, [HubbardManifold(0, l=2, u=U, j=J)])
    q = hubbard_projectors(hub, system.positions)
    dev = system_device(system)
    npw = system.batch.npw_max
    nk, nb = 1, 6
    coeffs = torch.randn(nk, nb, npw, dtype=torch.complex128).to(dev)
    coeffs = coeffs * system.batch.mask[:, None, :]
    occ = torch.rand(nk, nb, dtype=torch.float64).to(dev)
    kw = torch.ones(nk, dtype=torch.float64).to(dev)

    mats = occupation_matrices(q, coeffs, occ, kw, hub.sites)
    n = mats[0]
    assert n.shape == (5, 5)
    assert torch.allclose(n, n.conj().T, atol=1e-12)  # Hermitian
    # trace equals Σ_kv w f |⟨φ|ψ⟩|² summed over the manifold
    becp = torch.einsum("kpg,kbg->kbp", q.conj(), coeffs)
    tr_ref = float((kw[:, None] * occ * (becp.abs() ** 2).sum(-1)).sum())
    assert abs(float(torch.trace(n).real) - tr_ref) < 1e-10

    # Dudarev E_U = (U−J)/2 Tr[n(1−n)]
    e = hubbard_energy(mats, hub.sites)
    e_ref = 0.5 * (U - J) * float((torch.trace(n) - torch.trace(n @ n)).real)
    assert abs(float(e) - e_ref) < 1e-10

    # D = (U−J)(½I − n), Hermitian
    d = hubbard_dmatrix(mats, hub.sites, hub.nproj, dev)
    eye = torch.eye(5, dtype=torch.complex128, device=dev)
    assert torch.allclose(d, (U - J) * (0.5 * eye - n), atol=1e-12)
    assert torch.allclose(d, d.conj().T, atol=1e-12)


def test_hubbard_force_gradcheck():
    """E_U's position derivative (the +U force) via autograd vs finite diff.
    This is the projector-phase differentiability that hubbard_force relies on;
    tested here on frozen synthetic orbitals so no SCF is needed."""
    torch.manual_seed(1)
    se = parse_upf(FIX / "pseudos" / "PD_Se_FR.upf")
    # two atoms so a phase gradient is nonzero
    system = setup_system(9.0 * np.eye(3), np.array([[0.0, 0, 0], [2.3, 0.4, 0.0]]),
                          [0, 0], [se], ecut=20 * RY, kmesh=(1, 1, 1))
    hub = build_hubbard_projectors(system, [HubbardManifold(0, l=2, u=5.0, j=0.5)])
    dev = system_device(system)
    npw = system.batch.npw_max
    coeffs = torch.randn(1, 4, npw, dtype=torch.complex128).to(dev) * system.batch.mask[:, None, :]
    occ = torch.rand(1, 4, dtype=torch.float64).to(dev)
    kw = torch.ones(1, dtype=torch.float64).to(dev)

    def e_u(pos):
        q = hubbard_projectors(hub, pos)
        mats = occupation_matrices(q, coeffs, occ, kw, hub.sites)
        return hubbard_energy(mats, hub.sites)

    pos = system.positions.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(e_u, (pos,), atol=1e-6, rtol=1e-4)
