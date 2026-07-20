"""Off-stationarity E↔H consistency gates (Tier-0 verification).

The KS Hamiltonian is BY DEFINITION the Wirtinger derivative of the
total-energy functional: H ψ_nk = ∂E/∂(w_k f_nk ψ*_nk), with
v_eff = δ(E_H + E_xc + E_loc)/δρ. An inconsistency between the assembled E
and the H the SCF iterates is invisible at a self-consistent stationary
point (first-order errors vanish there) and can be invisible against QE
too when both codes share the error — the PAW ddd lesson (see
docs/verification.md). At a RANDOM non-stationary state nothing cancels.

These tests build the total energy exactly as scf.loop does (density_b,
becp_b, total_energy / the spin assembly), differentiate it with autograd
at random orbital coefficients, and demand the gradient equal the SCF's
own BatchedHamiltonian applied to those coefficients:

    grad_c E == 2 · w_k · f_nk · (H c)_nk      (torch grad = 2·∂E/∂c̄)

to near machine precision. A companion test checks the closed-form
potential expressions (Hartree, local) against autograd of the matching
energy expressions — two independent implementations of the same
functional derivative.

The identities hold at ANY cutoff/grid (they are properties of the
discretized functional, not of converged physics), so tiny, fast systems
are enough. Geometries are rattled/low-symmetry on purpose: symmetric
fixtures let error terms cancel by symmetry (the O₂-vs-Si lesson).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.batch import BatchedHamiltonian, becp_b, density_b, projectors_b
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.kinetic import kinetic_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.energies.total import total_energy
from gradwave.core.fftbox import r_to_g
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.common import spin_sigmas
from gradwave.scf.loop import (
    _stack_dij,
    effective_potentials,
    local_potential_r,
    setup_system,
)
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.fixture(autouse=True)
def _limit_threads():
    torch.set_num_threads(4)


def _si2_rattled(ecut_ry=12.0, kmesh=(2, 1, 1)):
    a = 5.43
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    # rattled off the diamond site — P1, no cancellations by symmetry
    positions = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])
    si = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    return setup_system(lattice, positions, [0, 0], [si],
                        ecut=ecut_ry * RY, kmesh=kmesh)


def _ni_box(ecut_ry=15.0):
    """One NLCC atom (PD_Ni carries a core density) in a box, off-center."""
    ni = parse_upf(FIX / "pseudos" / "PD_Ni_PBE.upf")
    return setup_system(8.0 * np.eye(3), np.array([[0.7, 0.4, 0.2]]), [0], [ni],
                        ecut=ecut_ry * RY, kmesh=(1, 1, 1))


def _random_coeffs(system, nb, seed):
    """Normalized random bands, padded (nk, nb, npw_max) — NOT an eigenstate."""
    bk = system.batch
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(bk.nk, nb, bk.npw_max, generator=gen, dtype=RDTYPE) \
        + 1j * torch.randn(bk.nk, nb, bk.npw_max, generator=gen, dtype=RDTYPE)
    # smooth envelope so the density is well scaled, then normalize per band
    c = c.to(CDTYPE) / (1.0 + bk.t)[:, None, :]
    c = c * bk.mask[:, None, :]
    return c / torch.linalg.norm(c, dim=-1, keepdim=True)


def _occ(system, nb, values):
    """Fixed fractional occupations, deliberately non-uniform across bands/k."""
    nk = system.batch.nk
    occ = torch.tensor(values, dtype=RDTYPE)[None, :].repeat(nk, 1)
    if nk > 1:  # vary across k too
        occ[1:] = occ[1:].flip(dims=(1,))
    return occ


def _eh_gap_nspin1(system, xc, nb, occ_values, seed=0, hubbard=None):
    """max |grad E − 2 w f Hc| / max |2 w f Hc| over the masked region."""
    grid, bk, spheres = system.grid, system.batch, system.spheres
    nk = bk.nk
    kw = system.kweights
    occ = _occ(system, nb, occ_values)
    projs = projectors_b(bk, system.positions)
    dij = _stack_dij(system)
    hub = hub_q = None
    if hubbard is not None:
        from gradwave.core.hubbard import (
            build_hubbard_projectors,
            hubbard_projectors,
        )
        hub = build_hubbard_projectors(system, hubbard)
        hub_q = hubbard_projectors(hub, system.positions)

    c = _random_coeffs(system, nb, seed).requires_grad_(True)
    trimmed = [c[ik, :, : int(bk.npw[ik])] for ik in range(nk)]
    rho = density_b(c, occ, kw, bk, grid.shape, grid.volume)
    eb = total_energy(
        coeffs_per_k=trimmed, occ=occ, kweights=kw, spheres=spheres, grid=grid,
        rho=rho, positions=system.positions, charges=system.charges,
        species_index=system.species_index, vloc_tables=system.vloc_tables,
        becp_per_k=[becp_b(projs, c)[ik] for ik in range(nk)], dij_full=dij,
        xc=xc, rho_core=system.rho_core,
    )
    e_tot = eb.total
    n_half = None
    if hub is not None:
        from gradwave.core.hubbard import hubbard_energy, occupation_matrices
        # mirror of scf.loop's nspin=1 +U bookkeeping: [0,2] occupations
        # split into two equal spin channels, E_U doubled
        n_half = occupation_matrices(hub_q, c, 0.5 * occ, kw, hub.sites)
        e_tot = e_tot + 2.0 * hubbard_energy(n_half, hub.sites)
    (g,) = torch.autograd.grad(e_tot, c)

    with torch.no_grad():
        hub_dij = None
        if hub is not None:
            from gradwave.core.hubbard import hubbard_dmatrix
            hub_dij = hubbard_dmatrix([m.detach() for m in n_half], hub.sites,
                                      hub.nproj, c.device).conj()
        veff = effective_potentials(system, xc, [rho.detach()],
                                    local_potential_r(system))[0]
        h = BatchedHamiltonian(bk, grid.shape, veff, projs,
                               hub_q=hub_q, hub_dij=hub_dij)
        expected = 2.0 * kw[:, None, None] * occ[:, :, None] * h.apply(c.detach())
    mask = bk.mask[:, None, :]
    gap = ((g - expected) * mask).abs().max()
    return float(gap / expected.abs().max())


def test_grad_energy_equals_hamiltonian_hubbard():
    """+U: Dudarev V_U = (U−J)(½−n) applied by the SCF vs autograd of E_U
    through the occupation matrices — including the nspin=1 half-occupation
    channel-splitting bookkeeping, which is exactly where a factor slip
    would hide."""
    from gradwave.core.hubbard import HubbardManifold
    from gradwave.core.xc.pbe import PBE
    system = _ni_box()
    gap = _eh_gap_nspin1(system, PBE(), nb=8,
                         occ_values=[2.0] * 5 + [1.5, 0.8, 0.3],
                         hubbard=[HubbardManifold(0, l=2, u=5.0, j=0.8)])
    assert gap < 1e-10


def test_grad_energy_equals_hamiltonian_lda():
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    system = _si2_rattled()
    assert _eh_gap_nspin1(system, LDA_PW92(), nb=5,
                          occ_values=[2.0, 2.0, 2.0, 1.2, 0.4]) < 1e-10


def test_grad_energy_equals_hamiltonian_pbe():
    from gradwave.core.xc.pbe import PBE
    system = _si2_rattled()
    assert _eh_gap_nspin1(system, PBE(), nb=5,
                          occ_values=[2.0, 2.0, 2.0, 1.2, 0.4]) < 1e-10


def test_grad_energy_equals_hamiltonian_nlcc():
    """NLCC: ρ_core shifts the XC argument in E and in v_xc — consistently."""
    from gradwave.core.xc.pbe import PBE
    system = _ni_box()
    assert system.rho_core is not None
    assert _eh_gap_nspin1(system, PBE(), nb=8,
                          occ_values=[2.0] * 5 + [1.5, 0.8, 0.3]) < 1e-10


def test_grad_energy_equals_hamiltonian_spin():
    """nspin=2: per-channel identity with the loop's spin energy assembly."""
    from gradwave.core.xc.spin import SpinPBE
    xc = SpinPBE()
    system = _si2_rattled()
    grid, bk, spheres = system.grid, system.batch, system.spheres
    nk, nb = bk.nk, 5
    kw = system.kweights
    occ_s = [_occ(system, nb, [1.0, 1.0, 1.0, 0.6, 0.2]),
             _occ(system, nb, [1.0, 1.0, 0.7, 0.3, 0.1])]
    projs = projectors_b(bk, system.positions)
    dij = _stack_dij(system)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)

    cs = [_random_coeffs(system, nb, seed).requires_grad_(True) for seed in (1, 2)]
    rho_s = [density_b(cs[sp], occ_s[sp], kw, bk, grid.shape, grid.volume)
             for sp in range(2)]
    rho_tot = rho_s[0] + rho_s[1]
    rho_g = r_to_g(rho_tot.to(CDTYPE))
    # mirror of scf.loop's nspin=2 energy assembly
    e_kin = sum(
        kinetic_energy([cs[sp][ik, :, : int(bk.npw[ik])] for ik in range(nk)],
                       occ_s[sp], kw, spheres) for sp in range(2))
    s_uu, s_dd, s_tt = spin_sigmas(rho_s[0], rho_s[1], xc, grid.g_cart)
    e_xc = xc.energy(rho_s[0], rho_s[1], grid.volume, s_uu, s_dd, s_tt)
    e_h = hartree_energy(rho_g, grid.g2, grid.volume)
    e_loc = local_energy(rho_g, vloc_g, grid.volume)
    e_nl = sum(
        nonlocal_energy([becp_b(projs, cs[sp])[ik] for ik in range(nk)],
                        dij, occ_s[sp], kw) for sp in range(2))
    e = e_kin + e_xc + e_h + e_loc + e_nl
    grads = torch.autograd.grad(e, cs)

    with torch.no_grad():
        veff_s = effective_potentials(system, xc,
                                      [r.detach() for r in rho_s],
                                      local_potential_r(system, vloc_g))
        mask = bk.mask[:, None, :]
        for sp in range(2):
            h = BatchedHamiltonian(bk, grid.shape, veff_s[sp], projs)
            expected = (2.0 * kw[:, None, None] * occ_s[sp][:, :, None]
                        * h.apply(cs[sp].detach()))
            gap = ((grads[sp] - expected) * mask).abs().max()
            assert float(gap / expected.abs().max()) < 1e-10, f"spin {sp}"


def _eh_gap_uspp(upf_name, nb, occ_values, ecut_ry=16.0, seed=0, nspin=1):
    """USPP/PAW gate: grad E == 2 w f (H c) with H's screened D built from
    the SAME state (v_eff from ρ[c] incl. augmentation, ddd from becsum[c]).
    This is the exact regime of the original ddd bug: the one-center term
    enters E as e1c_t(becsum) and enters H as ddd = ∂e1c/∂becsum — the gate
    fails unless they are derivatives of each other through the full
    ρ_aug/Q̃/phase chain. No S term appears: at unconstrained coefficients
    the energy's gradient IS Hc (εSc arises only from the orthonormality
    constraint at stationarity).

    nspin=2 runs the whole chain per channel (occ_values then is a pair of
    band-occupation lists): per-spin ρ_aug and becsum, the spin-XC σ chain
    on (ρ↑+½ρ_core, ρ↓+½ρ_core), per-channel ∫v_eff_σ Q screening, and for
    PAW the spin ddd — energy_and_ddd([ρ↑_ij, ρ↓_ij]) must be the exact
    channel-derivative of e1c_t at the same becsum pair."""
    from gradwave.constants import HBAR2_2M
    from gradwave.core.density import sigma_from_rho
    from gradwave.core.fftbox import g_to_r
    from gradwave.core.hamiltonian import becp, projectors
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.paw_onsite import OneCenter
    from gradwave.scf.uspp import setup_uspp
    from gradwave.scf.uspp_loop import _HkS, uspp_potentials_dscr

    if nspin == 1:
        from gradwave.core.xc.pbe import PBE
        xc = PBE()
        occ_values = [occ_values]
    else:
        from gradwave.core.xc.spin import SpinPBE
        xc = SpinPBE()

    a = 5.43
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])  # rattled, P1
    paw = parse_upf_paw(FIX / "pseudos" / upf_name)
    system = setup_uspp(lattice, pos, [0, 0], [paw], ecut=ecut_ry * RY,
                        kmesh=(2, 1, 1))
    grid, spheres = system.grid, system.spheres
    nk, vol, shape = len(spheres), grid.volume, grid.shape
    kw = system.kweights
    occ_s = []
    for sp in range(nspin):
        occ = torch.tensor(occ_values[sp], dtype=RDTYPE)[None, :].repeat(nk, 1)
        occ[1:] = occ[1:].flip(dims=(1,))
        occ_s.append(occ)

    projs = [projectors(pd, system.positions) for pd in system.proj_data]
    phase_arg = system.g_sphere @ system.positions.T
    phase_pos = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    onec = ([OneCenter(p, xc) for p in system.paws]
            if any(p.is_paw for p in system.paws) else None)
    vloc_g = local_potential_g(system.positions,
                               torch.tensor(system.species_of_atom),
                               system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real

    cs_s = []
    for sp in range(nspin):
        cs = []
        for ik, sph in enumerate(spheres):
            gen = torch.Generator().manual_seed(seed + 31 * ik + 977 * sp)
            c = (torch.randn(nb, sph.npw, generator=gen, dtype=torch.float64)
                 + 1j * torch.randn(nb, sph.npw, generator=gen,
                                    dtype=torch.float64))
            c = c.to(CDTYPE) / (1.0 + HBAR2_2M * sph.kpg2)
            c = c / torch.linalg.norm(c, dim=-1, keepdim=True)
            cs.append(c.requires_grad_(True))
        cs_s.append(cs)

    # ---- E(c), mirroring _scf_iteration's density/energy assembly ----
    rho_s, rho_ij_s, becps_s = [], [], []
    for sp in range(nspin):
        rho_sm = torch.zeros(shape, dtype=RDTYPE)
        becps = []
        rho_ij = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE)
                  for (s0, s1) in system.atom_slices]
        for ik, sph in enumerate(spheres):
            psi = g_to_r(cs_s[sp][ik], sph.flat_idx, shape)
            w = kw[ik] * occ_s[sp][ik]
            rho_sm = rho_sm + torch.einsum("b,bxyz->xyz", w, psi.abs() ** 2) / vol
            b = becp(projs[ik], cs_s[sp][ik])
            becps.append(b)
            for ia, (s0, s1) in enumerate(system.atom_slices):
                ba = b[:, s0:s1]
                rho_ij[ia] = rho_ij[ia] + torch.einsum(
                    "b,bi,bj->ij", w.to(CDTYPE), ba.conj(), ba)
        rho_ij = [0.5 * (m + m.conj().T) for m in rho_ij]
        aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE)
        for ia, spc in enumerate(system.species_of_atom):
            aug_sph = aug_sph + phase_pos[:, ia].conj() * torch.einsum(
                "ij,ijg->g", rho_ij[ia], system.aug[spc].q_g)
        aug_box = torch.zeros(grid.n_points, dtype=CDTYPE)
        aug_box[system.sphere_idx] = aug_sph / vol
        rho_aug = torch.fft.ifftn(aug_box.reshape(shape) * grid.n_points,
                                  dim=(-3, -2, -1)).real
        rho_s.append(rho_sm + rho_aug)
        rho_ij_s.append(rho_ij)
        becps_s.append(becps)
    rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]

    rho_g = r_to_g(rho_tot.to(CDTYPE))
    core = system.rho_core
    if nspin == 1:
        rho_xc = rho_tot if core is None else rho_tot + core
        sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho_xc, vol, sigma)
    else:
        c2 = 0.0 if core is None else 0.5 * core
        r_u, r_d = rho_s[0] + c2, rho_s[1] + c2
        s_uu, s_dd, s_tt = spin_sigmas(r_u, r_d, xc, grid.g_cart)
        e_xc = xc.energy(r_u, r_d, vol, s_uu, s_dd, s_tt)
    e = (e_xc + hartree_energy(rho_g, grid.g2, vol)
         + local_energy(rho_g, vloc_g, vol)
         + sum(kinetic_energy(cs_s[sp], occ_s[sp], kw, spheres)
               + nonlocal_energy(becps_s[sp], system.proj_data[0].dij_full,
                                 occ_s[sp], kw) for sp in range(nspin)))
    if onec is not None:
        for ia, spc in enumerate(system.species_of_atom):
            e = e + onec[spc].e1c_t([rho_ij_s[sp][ia].real
                                     for sp in range(nspin)])
    grads = torch.autograd.grad(e, [c for cs in cs_s for c in cs])

    # ---- H side: the SCF's own potentials + screened D at the same state ----
    with torch.no_grad():
        veff_s, dscr_s, _ = uspp_potentials_dscr(
            system, xc, [r.detach() for r in rho_s],
            [[m.detach() for m in ch] for ch in rho_ij_s],
            vloc_r, phase_pos, onec)
        gap = 0.0
        for sp in range(nspin):
            for ik, sph in enumerate(spheres):
                hs = _HkS(sph, shape, veff_s[sp], system.proj_data[ik],
                          projs[ik], dscr_s[sp], system.q_full)
                expected = (2.0 * kw[ik] * occ_s[sp][ik, :, None]
                            * hs.h(cs_s[sp][ik].detach()))
                gap = max(gap, float((grads[sp * nk + ik] - expected).abs().max()
                                     / expected.abs().max()))
    return gap


def test_grad_energy_equals_hamiltonian_uspp():
    """Bare USPP (rrkjus): gates the Q̃ augmentation-density chain and the
    ∫v_eff Q screening of D against autograd of the assembled energy."""
    assert _eh_gap_uspp("Si.pbe-n-rrkjus_psl.1.0.0.UPF", nb=5,
                        occ_values=[2.0, 2.0, 2.0, 1.2, 0.4]) < 1e-10


def test_grad_energy_equals_hamiltonian_paw():
    """PAW (kjpaw): additionally gates ddd == ∂E_1c/∂becsum through the full
    orbital chain — the term class of the original ddd bug."""
    assert _eh_gap_uspp("Si.pbe-n-kjpaw_psl.1.0.0.UPF", nb=5,
                        occ_values=[2.0, 2.0, 2.0, 1.2, 0.4]) < 1e-10


def test_grad_energy_equals_hamiltonian_uspp_spin():
    """nspin=2 USPP: per-channel ρ_aug/becsum and the ∫v_eff_σ Q screening,
    with the spin-XC σ chain coupling the channels through v_eff."""
    assert _eh_gap_uspp("Si.pbe-n-rrkjus_psl.1.0.0.UPF", nb=5, nspin=2,
                        occ_values=([1.0, 1.0, 1.0, 0.6, 0.2],
                                    [1.0, 1.0, 0.7, 0.3, 0.1])) < 1e-10


def test_grad_energy_equals_hamiltonian_paw_spin():
    """nspin=2 PAW: adds the spin one-center — [ddd↑, ddd↓] from
    energy_and_ddd([ρ↑_ij, ρ↓_ij]) must be the exact per-channel derivative
    of e1c_t at the same becsum pair (the ddd bug's term class, now with the
    channels coupled through the one-center spin XC)."""
    assert _eh_gap_uspp("Si.pbe-n-kjpaw_psl.1.0.0.UPF", nb=5, nspin=2,
                        occ_values=([1.0, 1.0, 1.0, 0.6, 0.2],
                                    [1.0, 1.0, 0.7, 0.3, 0.1])) < 1e-10


def test_grad_energy_equals_hamiltonian_soc_spinor():
    """SOC spinors: grad E == 2 w f (H c) on the doubled plane-wave axis.
    Gates the whole noncollinear chain at first order — the Pauli
    decomposition (m_x/m_y factors, the ⟨↑|V̂|↓⟩ = Bx − iBy sign), the
    exchange-field apply B⃗·σ⃗ against the m⃗-chain of the energy, and the
    j-resolved nonlocal (E_NL = Σ w f b†D_so b vs H's q/dij_so contraction,
    incl. the spinor-projector conjugation conventions). Spinor bands carry
    one electron each (occ ≤ 1)."""
    from gradwave.core.batch import g_to_r_b
    from gradwave.core.spinor_proj import build_so_projectors
    from gradwave.core.xc.noncollinear import (
        NoncollinearXC,
        energy_with_grid,
        vxc_and_bxc,
    )
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.scf.noncollinear import SpinorHamiltonian

    a = 5.653
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.52, 1.36, 1.47]])  # rattled, P1
    ga = parse_upf(FIX / "pseudos" / "Ga_ONCV_PBE_FR-1.0.upf")
    ars = parse_upf(FIX / "pseudos" / "As_ONCV_PBE_FR-1.1.upf")
    system = setup_system(lattice, pos, [0, 1], [ga, ars], ecut=14 * RY,
                          kmesh=(2, 1, 1), time_reversal=False)
    assert system.is_fr
    xc = NoncollinearXC(SpinPBE())
    grid, bk = system.grid, system.batch
    nk, vol, shape, m_pw = bk.nk, grid.volume, grid.shape, bk.npw_max
    kw = system.kweights
    nb = 8
    occ = _occ(system, nb, [1.0, 1.0, 0.9, 0.7, 0.55, 0.35, 0.2, 0.1])

    projs = projectors_b(bk, system.positions)
    q_so, dij_so = build_so_projectors(bk, system)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, vol)
    mask2 = torch.cat([bk.mask, bk.mask], dim=-1)

    gen = torch.Generator().manual_seed(5)
    c = (torch.randn(nk, nb, 2 * m_pw, generator=gen, dtype=RDTYPE)
         + 1j * torch.randn(nk, nb, 2 * m_pw, generator=gen, dtype=RDTYPE))
    t2 = torch.cat([bk.t, bk.t], dim=-1)
    c = c.to(CDTYPE) / (1.0 + t2)[:, None, :] * mask2[:, None, :]
    c = (c / torch.linalg.norm(c, dim=-1, keepdim=True)).requires_grad_(True)

    # ---- E(c): the Pauli-decomposed density + scf_noncollinear's assembly ----
    f = kw[:, None] * occ
    pu = g_to_r_b(c[..., :m_pw], bk, shape)
    pd = g_to_r_b(c[..., m_pw:], bk, shape)
    uu = torch.einsum("kb,kbxyz->xyz", f, pu.real**2 + pu.imag**2)
    dd = torch.einsum("kb,kbxyz->xyz", f, pd.real**2 + pd.imag**2)
    ud = torch.einsum("kb,kbxyz->xyz", f.to(CDTYPE), pu.conj() * pd)
    rho = (uu + dd) / vol
    m_vec = torch.stack([2.0 * ud.real, 2.0 * ud.imag, uu - dd]) / vol

    rho_g = r_to_g(rho.to(CDTYPE))
    b_so = torch.einsum("kpg,kbg->kbp", q_so.conj(), c)
    e = (torch.einsum("kb,kbg,kg->", f, c.real**2 + c.imag**2, t2)
         + hartree_energy(rho_g, grid.g2, vol)
         + energy_with_grid(xc, rho, m_vec, grid, rho_core=system.rho_core)
         + local_energy(rho_g, vloc_g, vol)
         + nonlocal_energy([b_so[ik] for ik in range(nk)], dij_so, occ, kw))
    (g,) = torch.autograd.grad(e, c)

    # ---- H side: v·1 + B⃗_xc·σ⃗ + j-resolved nonlocal at the same state ----
    with torch.no_grad():
        v_h = (torch.fft.ifftn(hartree_potential_g(rho_g.detach(), grid.g2),
                               dim=(-3, -2, -1)) * grid.n_points).real
        v_xc, b_xc, _ = vxc_and_bxc(xc, rho.detach(), m_vec.detach(), grid,
                                    rho_core=system.rho_core)
        v_r = v_h + v_xc + local_potential_r(system, vloc_g)
        h = SpinorHamiltonian(bk, shape, v_r, b_xc, projs, q=q_so,
                              dij_so=dij_so)
        expected = 2.0 * kw[:, None, None] * occ[:, :, None] * h.apply(c.detach())
    gap = ((g - expected) * mask2[:, None, :]).abs().max()
    assert float(gap / expected.abs().max()) < 1e-10


def test_potentials_equal_autograd_of_energies():
    """Closed-form v_H, v_loc vs autograd of E_H, E_loc — two independent
    implementations of the same functional derivative must agree."""
    system = _si2_rattled(kmesh=(1, 1, 1))
    grid = system.grid
    occ = _occ(system, 4, [2.0, 2.0, 1.5, 0.5])
    rho0 = density_b(_random_coeffs(system, 4, 3), occ, system.kweights,
                     system.batch, grid.shape, grid.volume)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)

    rho = rho0.detach().requires_grad_(True)
    e = (hartree_energy(r_to_g(rho.to(CDTYPE)), grid.g2, grid.volume)
         + local_energy(r_to_g(rho.to(CDTYPE)), vloc_g, grid.volume))
    (g,) = torch.autograd.grad(e, rho)
    v_from_e = g * (grid.n_points / grid.volume)

    v_h = (torch.fft.ifftn(hartree_potential_g(r_to_g(rho0.to(CDTYPE)), grid.g2),
                           dim=(-3, -2, -1)) * grid.n_points).real
    v_closed = v_h + local_potential_r(system, vloc_g)
    scale = max(float(v_closed.abs().max()), 1.0)
    assert float((v_from_e - v_closed).abs().max()) / scale < 1e-10
