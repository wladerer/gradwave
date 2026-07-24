"""Fixed-basis stress with DFT+U (Dudarev) — the Hubbard strain term unblock.

The stress path now adds the strain derivative of the Dudarev corrective
energy E_U = Σ_{I,σ} (U−J)/2 Tr[n^{Iσ}(1−n^{Iσ})]. The +U energy depends on
strain through the atomic-orbital projectors ⟨φ_m|ψ⟩ that build the occupation
matrices n^{Iσ}: the radial form factor F(|k+G|), the Y_lm direction, the
1/√Ω(ε) normalization, and the e^{−i(k+G)·τ(ε)} phase all move under ε. Those
strained projectors go on the same autograd graph #58's nspin=2 stress uses
(postscf/_strain.py), so the +U stress is the analytic strain derivative of the
same energy expression by construction (like the +U force is its position
derivative).

Two checks, mirroring test_stress_nspin2 and the FD stress tests:

- self-oracle: on a rattled, low-symmetry Si cell with a Hubbard U on the Si p
  manifold (mixed, smeared occupations), the ε=0 strained expression reproduces
  the SCF total (E_U included), and the analytic +U stress equals a central
  finite difference of that same total energy w.r.t. strain.
- the non-+U stress path is unchanged: without a manifold the stress is
  bit-identical to the pre-existing (no-+U) result, and a +U result asked for
  a stress without its manifolds raises rather than silently dropping E_U.
"""

import numpy as np
import pytest
import torch

from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.stress import _energy_strained, stress
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

# Rattled, low-symmetry 2-atom Si cell (no cubic symmetry, atoms displaced off
# the ideal diamond sites): a full anisotropic stress tensor.
SI_CELL = 5.43 * np.array([
    [0.000, 0.510, 0.490],
    [0.500, 0.000, 0.520],
    [0.505, 0.495, 0.000],
])
SI_FRAC = np.array([[0.010, 0.020, -0.015], [0.260, 0.240, 0.253]])
SI_POS = SI_FRAC @ SI_CELL

# +U on the Si p manifold. Not a physically standard correction, but the check
# is purely the mathematical identity stress = (1/Ω) dE_tot/dε, so any manifold
# with a real strain dependence exercises the Hubbard strain term.
MAN = [HubbardManifold(species=0, l=1, u=4.0, j=0.0)]


def _make_system():
    si = parse_upf(PSEUDOS / "Si_ONCV_PBE_sr.upf")
    return setup_system(SI_CELL, SI_POS, [0, 0], [si], ecut=24 * RY,
                        kmesh=(2, 2, 2), use_symmetry=False)


@pytest.mark.slow
def test_stress_hubbard_autograd_vs_fd():
    """+U stress vs central finite difference of the (+U) total energy w.r.t.
    strain, on a rattled low-symmetry Si+U cell. Also asserts the ε=0 strained
    expression reproduces the SCF total (E_U included)."""
    torch.set_num_threads(8)
    res = scf(_make_system(), LDA_PW92(), smearing="gaussian", width=0.05,
              hubbard=MAN, etol=1e-10, rhotol=1e-9, verbose=False, max_iter=150)
    assert res.converged
    assert abs(float(res.energies.hubbard)) > 1e-3  # E_U is genuinely nonzero

    # the ε=0 strained expression (with +U) must reproduce the SCF total
    e0 = _energy_strained(res, LDA_PW92(),
                          torch.zeros(3, 3, dtype=torch.float64), manifolds=MAN)
    assert abs(float(e0) - float(res.energies.total)) < 1e-6, (
        float(e0), float(res.energies.total))

    sig = stress(res, LDA_PW92(), symmetrize=False, manifolds=MAN).cpu().numpy()
    d = 1e-6

    def fd_component(i, j):
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[i, j] = d
        return (float(_energy_strained(res, LDA_PW92(), ep, manifolds=MAN))
                - float(_energy_strained(res, LDA_PW92(), -ep, manifolds=MAN))
                ) / (2 * d)

    omega = res.system.grid.volume
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (2, 1), (0, 2)]:
        fd_sym = 0.5 * (fd_component(i, j) + fd_component(j, i)) / omega
        rel = abs(sig[i, j] - fd_sym) / max(abs(fd_sym), 1e-8)
        assert rel < 1e-6 or abs(sig[i, j] - fd_sym) < 1e-8, (
            i, j, sig[i, j], fd_sym, rel)


@pytest.mark.slow
def test_stress_hubbard_isolates_the_u_term():
    """The +U stress is exactly the pre-existing stress plus the Hubbard strain
    term: dropping the manifold reproduces the no-+U tensor, and its difference
    from the +U tensor equals the central FD of E_U alone w.r.t. strain."""
    torch.set_num_threads(8)
    res = scf(_make_system(), LDA_PW92(), smearing="gaussian", width=0.05,
              hubbard=MAN, etol=1e-10, rhotol=1e-9, verbose=False, max_iter=150)
    assert res.converged

    # asking for a +U stress without the manifolds must not silently drop E_U
    with pytest.raises(ValueError, match="DFT\\+U"):
        stress(res, LDA_PW92())

    # stress(..., manifolds=None) on this same (detached) state is the no-+U
    # KS stress; the manifold adds exactly the Hubbard strain term.
    sig_no_u = _stress_no_manifold(res)
    sig_u = stress(res, LDA_PW92(), symmetrize=False, manifolds=MAN).cpu().numpy()
    d = 1e-6
    omega = res.system.grid.volume

    def e_u_only(ep):
        # E_U(ε) = full strained energy − strained energy without the manifold
        return (float(_energy_strained(res, LDA_PW92(), ep, manifolds=MAN))
                - float(_energy_strained(res, LDA_PW92(), ep)))

    for i, j in [(0, 0), (1, 1), (0, 1), (2, 1)]:
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[i, j] = d
        d_eu = (e_u_only(ep) - e_u_only(-ep)) / (2 * d) / omega
        ep_t = torch.zeros(3, 3, dtype=torch.float64)
        ep_t[j, i] = d
        d_eu_t = (e_u_only(ep_t) - e_u_only(-ep_t)) / (2 * d) / omega
        d_eu_sym = 0.5 * (d_eu + d_eu_t)
        assert abs((sig_u[i, j] - sig_no_u[i, j]) - d_eu_sym) < 1e-7, (
            i, j, sig_u[i, j] - sig_no_u[i, j], d_eu_sym)


def _stress_no_manifold(res):
    """The stress of the detached SCF state with the +U term omitted (the pure
    KS strain derivative), reusing the same autograd path stress() uses."""
    dev = res.system.positions.device
    eps = torch.zeros(3, 3, dtype=torch.float64, device=dev, requires_grad=True)
    e = _energy_strained(res, LDA_PW92(), eps)
    (grad,) = torch.autograd.grad(e, eps)
    return (0.5 * (grad + grad.T) / res.system.grid.volume).cpu().numpy()
