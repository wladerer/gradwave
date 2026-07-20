"""Hybrid (PBE0-form) SCF lifted from Γ to a full-BZ k-mesh.

The per-k ACE operator sums each k's exchange over the whole BZ through the
co-density momentum q = k−k′ and the range-separated kernel. Three layers of
gate:

- **Operator reduction.** At one k-point the multi-k Fock operator is exactly
  ``exchange.exchange_operator_direct`` (the Γ build); its energy trace matches
  ``multik_exchange_energy`` on a mesh, for the full and screened kernels.
- **SCF reduction.** At α = 0 the k-mesh hybrid SCF is exactly a PBE SCF.
- **k-mesh PBE0.** At α > 0 it converges, the Fock energy term equals
  α·(2/nspin)·``multik_exchange_energy`` on the converged orbitals (the operator
  and energy stay derivative-consistent), and the gap opens relative to PBE.
"""

import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.postscf import exchange
from gradwave.postscf import exchange_multik as xk
from gradwave.postscf.hybrid import hybrid_scf
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo, si_fcc


def _system(kmesh):
    cell, pos = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    return setup_system(cell, pos, [0, 0], [upf], ecut=14 * RY, kmesh=kmesh,
                        use_symmetry=False, time_reversal=False, nbands=8)


def _op_trace_energy(psi_per_k, w_per_k, kweights, vol):
    """½ Σ_k w_k Σ_t ⟨ψ̂_{tk}|W_{tk}⟩ with the (Ω/N) inner product."""
    n_r = psi_per_k[0].shape[1]
    e = 0.0
    for ik, (p, w) in enumerate(zip(psi_per_k, w_per_k, strict=True)):
        e += float(kweights[ik]) * 0.5 * (vol / n_r) * float((p.conj() * w).sum().real)
    return e


@pytest.fixture(scope="module")
def si_gamma():
    s = _system((1, 1, 1))
    r = scf(s, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    assert r.converged
    return s, r


@pytest.fixture(scope="module")
def si_mesh():
    s = _system((2, 1, 1))
    r = scf(s, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    assert r.converged
    return s, r


@pytest.fixture(scope="module")
def pbe_mesh():
    res = scf(_system((2, 1, 1)), PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    assert res.converged
    return res


def test_multik_operator_reduces_to_gamma_direct(si_gamma):
    system, res = si_gamma
    shape, g2, vol = system.grid.shape, system.grid.g2, system.grid.volume
    psi, kc, kw = xk.physical_periodic_orbitals(res, system)
    occ = res.occupations[0] > 1e-6
    psi_g = exchange.physical_orbitals(res.coeffs[0][occ], system.spheres[0].flat_idx,
                                       shape, vol)
    w_direct = exchange.exchange_operator_direct(psi_g, psi_g, shape, g2)
    w_mk = xk.multik_exchange_operator(psi, kc, kw, system.grid.g_cart, vol, mode="full")[0]
    assert float((w_mk - w_direct).abs().max()) < 1e-12


@pytest.mark.parametrize("mode,omega", [("full", None), ("short_range", 0.3)])
def test_operator_trace_matches_multik_energy(si_mesh, mode, omega):
    system, res = si_mesh
    gc, vol = system.grid.g_cart, system.grid.volume
    psi, kc, kw = xk.physical_periodic_orbitals(res, system)
    u = [p * (vol ** 0.5) for p in psi]  # bare periodic parts for the energy fn
    om = torch.tensor(omega, dtype=torch.float64) if omega is not None else None
    w = xk.multik_exchange_operator(psi, kc, kw, gc, vol, mode=mode, omega=om)
    e_trace = _op_trace_energy(psi, w, kw, vol)
    e_energy = float(xk.multik_exchange_energy(u, kc, kw, gc, vol, mode=mode, omega=om))
    assert abs(e_trace - e_energy) < 1e-9


def test_kmesh_alpha_zero_reduces_to_pbe(pbe_mesh):
    res_h = hybrid_scf(_system((2, 1, 1)), alpha=0.0, smearing="none", etol=1e-9,
                       rhotol=1e-8, verbose=False)
    assert res_h.converged
    assert float(res_h.energies.fock) == 0.0
    assert abs(float(res_h.energies.free_energy)
               - float(pbe_mesh.energies.free_energy)) < 1e-8


def test_kmesh_pbe0_converges_and_fock_consistent(pbe_mesh):
    res = hybrid_scf(_system((2, 1, 1)), alpha=0.25, mode="full", smearing="none",
                     etol=1e-9, rhotol=1e-8, verbose=False, max_iter=100)
    assert res.converged
    assert float(res.energies.fock) < 0
    # fock term ≡ α·(2/nspin)·E_x^multik on the converged orbitals
    u, kc, kw = xk.occupied_periodic_orbitals(res, res.system)
    gc, vol = res.system.grid.g_cart, res.system.grid.volume
    e_mk = float(xk.multik_exchange_energy(u, kc, kw, gc, vol, mode="full"))
    assert abs(float(res.energies.fock) - 0.25 * 2.0 * e_mk) < 1e-9
    # PBE0 opens the gap at the first k-point (4 occupied bands) relative to PBE
    gap_pbe = float(pbe_mesh.eigenvalues[0][4] - pbe_mesh.eigenvalues[0][3])
    gap_h = float(res.eigenvalues[0][4] - res.eigenvalues[0][3])
    assert gap_h > gap_pbe + 0.1
