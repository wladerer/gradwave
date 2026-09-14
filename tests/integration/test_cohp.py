"""COHP validation across the collinear and spinor projection paths.

There is no Quantum ESPRESSO COHP to compare against (QE carries no COHP), so
the check is the internal sum rule the band-limited projected Hamiltonian obeys
by construction: summing COHP over EVERY atom pair including the on-site blocks
and integrating to E_F reproduces the band-structure energy sum_n f_n eps_n, up
to the plane-wave spilling. Alongside it, the physical sign is fixed — a bound
dimer (O2, Bi2) must give a bonding (negative) ICOHP on its one bond.
"""

import dataclasses

import numpy as np
import pytest
import torch

from gradwave.postscf import cohp
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY, si_fcc

FIX = PSEUDOS


def _band_energy_step(res, system, g_spin):
    """sum_n f_n eps_n with a step occupation at E_F, matching the step
    occupation COHP._sumrule_icohp integrates with."""
    eig = res.eigenvalues.cpu().numpy()
    kw = system.kweights.cpu().numpy()
    # collapse a possible spin axis (nspin, nk, nb) -> (nk*..., nb) contribution
    if eig.ndim == 3:
        step = (eig < float(res.fermi)).astype(float) * g_spin
        return float((kw[None, :, None] * step * eig).sum())
    step = (eig < float(res.fermi)).astype(float) * g_spin
    return float((kw[:, None] * step * eig).sum())


def test_cohp_image_reconstruction():
    """Per-image (per-bond) decomposition is exact: summing the single-image COHP
    over the Born-von-Karman image shell reconstructs the sublattice COHP that
    `_accumulate` returns, and at Gamma the R=0 image equals the sublattice. Pure
    linear algebra on synthetic Hermitian H~(k) and projections, so it isolates the
    reciprocal-space Fourier identity (and its phase signs) with no SCF."""
    rng = np.random.default_rng(0)
    nk, nspin, g_spin = 2, 1, 2.0
    kpts = [np.array([0.0, 0, 0]), np.array([0.5, 0, 0])]   # MP n=2 along x
    kw = [0.5, 0.5]
    atom_of = np.array([0, 0, 1, 1])                        # 2 orbitals per atom
    nb, nproj = 3, 4

    def rc(*s):
        return torch.tensor(rng.standard_normal(s) + 1j * rng.standard_normal(s),
                            dtype=torch.complex128)
    proj = [rc(nb, nproj) for _ in range(nk)]
    htil = [(lambda A: 0.5 * (A + A.conj().T))(rc(nproj, nproj)) for _ in range(nk)]
    eigs = [torch.tensor(np.sort(rng.standard_normal(nb))) for _ in range(nk)]
    pair_list, fermi = [(0, 1, 2.0)], 0.0

    _, raw0, ic0, _ = cohp._accumulate(proj, htil, eigs, kw, atom_of, pair_list,
                                       g_spin, fermi)
    # sum single-image weights over the BvK cells R in {(0,0,0),(1,0,0)}
    raw_sum = [np.zeros(nb) for _ in range(nk)]
    ic_sum = 0.0
    for R in (np.array([0, 0, 0]), np.array([1, 0, 0])):
        raw_i, ic_i = cohp._accumulate_images(proj, htil, eigs, kw, kpts, atom_of,
                                              pair_list, {(0, 1): R}, nspin, nk,
                                              g_spin, fermi)
        for b in range(nk):
            raw_sum[b] += raw_i[(0, 1)][b]
        ic_sum += ic_i[(0, 1)]
    assert max(np.abs(raw_sum[b] - raw0[(0, 1)][b]).max() for b in range(nk)) < 1e-10
    assert abs(ic_sum - ic0[(0, 1)]) < 1e-10

    # Gamma (single k): the R=0 image is identical to the sublattice, exactly
    _, _, ic_g, _ = cohp._accumulate([proj[0]], [htil[0]], [eigs[0]], [1.0],
                                     atom_of, pair_list, g_spin, fermi)
    _, ici = cohp._accumulate_images([proj[0]], [htil[0]], [eigs[0]], [1.0],
                                     [kpts[0]], atom_of, pair_list,
                                     {(0, 1): np.array([0, 0, 0])}, 1, 1, g_spin, fermi)
    assert abs(ici[(0, 1)] - ic_g[(0, 1)]) < 1e-12


@pytest.fixture(scope="module")
def o2_gamma():
    """Converged Γ-only O2 PBE SCF shared by the collinear COHP tests.

    The three tests below build the byte-identical 7 Å box at ecut=40 Ry and run
    the same SCF, so one converged state serves all three (module scope) instead
    of re-solving it per test. Consumers only READ `res`, so this is a pure
    setup-sharing win — the per-test oracle assertions are unchanged.
    """
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    upf = parse_upf(f"{FIX}/PD_O_PBE.upf")
    L, d = 7.0, 1.21
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    system = setup_system(cell, pos, [0, 0], [upf], ecut=40 * RY, kmesh=(1, 1, 1))
    res = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-7,
              rhotol=1e-6, verbose=False, kerker=True)
    assert res.converged
    return res, system


@pytest.mark.standard
def test_cohp_collinear_o2_sum_rule(o2_gamma):
    """Norm-conserving O2 (nspin=1): the single O-O bond is bonding (ICOHP < 0),
    and the all-pairs COHP integrated to E_F reproduces the band energy up to the
    spilling."""
    res, system = o2_gamma

    c = cohp.cohp(res, width=0.2)
    assert c.kind == "collinear"
    assert 0.0 < c.spilling < 1.0
    # charge spilling is over the occupied manifold only: the valence states are
    # well described by the atomic orbitals, so it is smaller than total spilling
    assert 0.0 < c.charge_spilling < c.spilling
    # exactly the one nearest-neighbour pair is picked within rcut
    assert [p[:2] for p in c.pairs] == [(0, 1)]
    # the O-O bond is bonding: negative ICOHP
    assert c.pair_icohp["1-2"] < 0.0
    # total matches the per-pair sum, and the broadened curve is finite
    assert abs(c.total_icohp - sum(c.pair_icohp.values())) < 1e-9
    assert np.isfinite(c.total).all()

    # sum rule: all pairs incl on-site integrate to the band energy up to spilling
    band_e = _band_energy_step(res, system, g_spin=2.0)
    ratio = c._sumrule_icohp / band_e
    assert 0.90 < ratio <= 1.001, (c._sumrule_icohp, band_e)


@pytest.mark.standard
def test_cohp_summed_spins_false_is_exact_half(o2_gamma):
    """summed_spins=False reports the LOBSTER ISPIN=1 per-spin convention, which
    for a nonmagnetic nspin=1 run is EXACTLY half the physical two-electron value
    gradwave returns by default (the spin degeneracy g=2 -> 1). The relation must
    hold on every reported channel — per-pair and total ICOHP and the broadened
    COHP curves — not just the scalar bond number."""
    res, _ = o2_gamma
    full = cohp.cohp(res, width=0.2)
    half = cohp.cohp(res, width=0.2, summed_spins=False)

    assert half.pair_icohp.keys() == full.pair_icohp.keys()
    for lab in full.pair_icohp:
        assert half.pair_icohp[lab] == pytest.approx(0.5 * full.pair_icohp[lab], rel=1e-10)
        np.testing.assert_allclose(half.pair_cohp[lab], 0.5 * full.pair_cohp[lab], rtol=1e-10)
    assert half.total_icohp == pytest.approx(0.5 * full.total_icohp, rel=1e-10)
    np.testing.assert_allclose(half.total, 0.5 * full.total, rtol=1e-10)
    # the physical sign is unchanged by the convention
    assert half.pair_icohp["1-2"] < 0.0


@pytest.mark.slow
def test_cohp_soc_bi2():
    """Fully-relativistic Bi2: the j-resolved (SOC) COHP and the scalar-charge
    noncollinear COHP both give a bonding bond on the same spinor states, and both
    satisfy the band-energy sum rule."""
    torch.set_num_threads(8)
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.scf.noncollinear import scf_noncollinear
    bi = parse_upf(f"{FIX}/PD_Bi_FR.upf")
    L, d = 9.0, 2.7
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    system = setup_system(cell, pos, [0, 0], [bi], ecut=30 * RY, kmesh=(1, 1, 1),
                          nbands=24, time_reversal=False)
    res = scf_noncollinear(system, NoncollinearXC(SpinPBE()),
                           mag_vec_init=[[0, 0, 0.1], [0, 0, 0.1]], width=0.2,
                           smearing="gaussian", etol=1e-6, rhotol=1e-5,
                           verbose=False)
    assert res.converged

    cs = cohp.cohp_soc(res, width=0.3)
    cn = cohp.cohp_noncollinear(res, width=0.3)
    assert cs.kind == "soc" and cn.kind == "noncollinear"
    assert [p[:2] for p in cs.pairs] == [(0, 1)]
    # bonding on both projection bases
    assert cs.pair_icohp["1-2"] < 0.0
    assert cn.pair_icohp["1-2"] < 0.0
    # the two AO spans differ (spinor |l,j,mj> vs scalar l x spin), so the bond
    # strengths agree only to ~10%
    assert abs(cs.pair_icohp["1-2"] - cn.pair_icohp["1-2"]) < 0.1 * abs(cn.pair_icohp["1-2"])

    band_e = _band_energy_step(res, system, g_spin=1.0)
    assert 0.95 < cs._sumrule_icohp / band_e <= 1.001
    assert 0.95 < cn._sumrule_icohp / band_e <= 1.001
    assert 0.0 < cs.spilling < 1.0
    assert 0.0 < cs.charge_spilling < 1.0
    assert 0.0 < cn.charge_spilling < 1.0


@pytest.mark.standard
def test_cohp_resolve_images_and_iao_o2(o2_gamma):
    """On Gamma-only O2 the per-bond (resolve_images) COHP equals the sublattice
    COHP exactly (R=0), and the IAO basis spans the occupied manifold far better
    than the pseudo-atomic basis (much smaller charge spilling / RMSp) while still
    giving a bonding bond and satisfying the band-energy sum rule."""
    res, system = o2_gamma

    base = cohp.cohp(res, width=0.2)
    # --- per-image bond resolution: R=0 at Gamma, so one bond == the sublattice ---
    img = cohp.cohp(res, width=0.2, resolve_images=True)
    assert img.bond_images == {"1-2": (0, 0, 0)}
    assert abs(img.pair_icohp["1-2"] - base.pair_icohp["1-2"]) < 1e-9
    assert img.pair_icohp["1-2"] < 0.0

    # --- IAO basis: spans the occupied space, so spilling collapses toward zero ---
    iao = cohp.cohp(res, width=0.2, basis="iao")
    assert iao.basis == "iao" and base.basis == "pswfc"
    assert base.rmsp == pytest.approx(np.sqrt(base.spilling), rel=1e-6)
    # occupied-manifold spilling is tiny with IAOs, and well below the pswfc basis
    assert iao.charge_spilling < 1e-3
    assert iao.charge_spilling < 0.1 * base.charge_spilling
    # still a bonding bond, and the all-pairs sum rule still holds
    assert iao.pair_icohp["1-2"] < 0.0
    band_e = _band_energy_step(res, system, g_spin=2.0)
    assert 0.90 < iao._sumrule_icohp / band_e <= 1.001

    # differentiable RMSp objective agrees with the reported scalar (all bands),
    # and on the occupied manifold IAO drives the residual far below the pswfc basis
    assert float(cohp.projection_rmsp(res)) == pytest.approx(base.rmsp, rel=1e-5)
    assert (float(cohp.projection_rmsp(res, basis="iao", occupied_only=True))
            < float(cohp.projection_rmsp(res, occupied_only=True)))


@pytest.mark.standard
def test_cohp_k_band_resolved():
    """The k- and band-resolved COHP reconstructs the broadened curve and the
    ICOHP exactly. O2 on a 2-k mesh gives two blocks, so the per-(k, band) weights
    and the k-weighted reconstruction are genuinely exercised."""
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    upf = parse_upf(f"{FIX}/PD_O_PBE.upf")
    L, d = 7.0, 1.21
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    # reconstruction identity (recon == pair_cohp) is exact at any ecut
    system = setup_system(cell, pos, [0, 0], [upf], ecut=14 * RY, kmesh=(2, 1, 1))
    res = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-7,
              rhotol=1e-6, verbose=False, kerker=True)
    assert res.converged

    width = 0.2
    c = cohp.cohp(res, width=width)
    nblocks = c.band_cohp["1-2"].shape[0]
    assert nblocks == c.nspin * c.nk and nblocks >= 2
    assert c.band_energies.shape == c.band_cohp["1-2"].shape
    assert c.block_kpts.shape == (nblocks, 3)
    assert c.block_kweights.shape == (nblocks,)

    # the broadened pair curve is exactly the k-weighted sum of the per-block curves
    recon = np.zeros_like(c.energy_eV)
    for b in range(nblocks):
        recon += c.cohp_at_k("1-2", b, width=width)[1]
    assert np.abs(recon - c.pair_cohp["1-2"]).max() < 1e-9

    # ICOHP is the occupied, k-weighted sum of the per-eigenstate weights
    occ = c.band_energies < c.fermi_eV
    recon_icohp = float((c.band_cohp["1-2"] * c.block_kweights[:, None])[occ].sum())
    assert abs(recon_icohp - c.pair_icohp["1-2"]) < 1e-9
    # reshape helper round-trips the block layout
    assert c.bands_reshaped("1-2").shape == (c.nspin, c.nk, c.band_energies.shape[1])


@pytest.mark.standard
def test_cohp_explicit_pairs_and_rcut(o2_gamma):
    """Pair selection: an explicit `pairs` list overrides rcut, and a tight rcut
    that excludes the bond yields no pairs."""
    res, _ = o2_gamma
    # explicit pair wins regardless of order
    c = cohp.cohp(res, pairs=[(1, 0)], width=0.2)
    assert [p[:2] for p in c.pairs] == [(0, 1)]
    # rcut below the bond length selects nothing
    c0 = cohp.cohp(res, rcut=1.0, width=0.2)
    assert c0.pairs == []
    assert c0.pair_icohp == {}


@pytest.fixture(scope="module")
def diamond_c():
    """Converged diamond-carbon PBE SCF (PseudoDojo std, has PP_PSWFC) shared by
    the reference-leak / cross-route COHP hardening tests below. nbands=24 sits
    well past the diamond nbands-convergence sweep's knee (see the cohp.py module
    docstring): the eigenvalue route is within ~3% of the operator route there,
    so this is a fair, non-flaky point to regression-test the bracket at.

    ecut is 30 Ry, not the plane-wave-converged 45: the cross-route ratio and
    reference-leak this fixture feeds are flat in ecut (1.024 at 30 vs 1.025 at
    45, leak unchanged), so the lower cutoff keeps the ~167 s CI fixture cost
    down without moving the calibrated bracket."""
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    upf = parse_upf(f"{FIX}/PD_C_PBE_std.upf")
    cell, pos = si_fcc(a=3.567)
    system = setup_system(cell, pos, [0, 0], [upf], ecut=30 * RY, kmesh=(2, 2, 2),
                          nbands=24, use_symmetry=False)
    res = scf(system, PBE(), smearing="gaussian", width=0.05, etol=1e-8,
              rhotol=1e-7, verbose=False, kerker=True)
    assert res.converged
    return res, system


@pytest.mark.standard
def test_cohp_operator_vs_eigenvalue_cross_route(diamond_c):
    """Operator route vs. eigenvalue route on the SAME periodic solid (diamond
    C), the comparison test_cohp_soc_bi2 was missing: both a numeric bracket
    (not just a sign check) and a direct measurement of the reference-energy
    leak the module docstring warns about, on the SAME converged SCF.

    The bracket (0.7-1.4x) and the leak ceiling (0.5 eV/eV) are set from the
    diamond nbands-convergence sweep in the module docstring (nbands=24 gives
    ICOHP_operator/ICOHP_eigenvalue = 1.03x and a shift-slope of 0.008 eV/eV);
    both bounds carry a wide safety margin so this catches a REGRESSION (the
    ratio drifting materially further from 1, or the leak growing back toward
    the diamond sweep's near-nocc values of a few eV/eV) without being flaky.
    """
    res, system = diamond_c

    c_op = cohp.cohp(res, pairs=[(0, 1)], method="operator", width=0.2)
    c_eig = cohp.cohp(res, pairs=[(0, 1)], method="eigenvalue", width=0.2)
    icohp_op, icohp_eig = c_op.pair_icohp["1-2"], c_eig.pair_icohp["1-2"]

    # both routes agree on the physical sign (bonding) ...
    assert icohp_op < 0.0
    assert icohp_eig < 0.0
    # ... and, at a generous band count, on the magnitude to within a bracket
    # far tighter than the raw operator/eigenvalue overshoot/undershoot (~2x
    # each per the module docstring) because nbands=24 is well past the knee.
    ratio = icohp_op / icohp_eig
    assert 0.7 < ratio < 1.4, (icohp_op, icohp_eig, ratio)

    # reference-energy leak, measured directly (module docstring's diagnostic):
    # shifting eig and fermi by the same delta must leave the operator-route
    # ICOHP exactly invariant (H~ = <phi|H^|phi> never touches eig) ...
    DELTA = 1.0
    res_shift = dataclasses.replace(
        res, eigenvalues=res.eigenvalues + DELTA, fermi=res.fermi + DELTA)
    c_op_shift = cohp.cohp(res_shift, pairs=[(0, 1)], method="operator", width=0.2)
    assert abs(c_op_shift.pair_icohp["1-2"] - icohp_op) < 1e-6

    # ... while the eigenvalue route leaks, but only slightly at this band count
    c_eig_shift = cohp.cohp(res_shift, pairs=[(0, 1)], method="eigenvalue",
                            width=0.2)
    slope = c_eig_shift.pair_icohp["1-2"] - icohp_eig  # /DELTA=1.0
    assert abs(slope) < 0.5, slope


@pytest.mark.standard
def test_cohp_symmetry_scheme_consistency():
    """use_symmetry=True (IBZ + weights) vs. False (full/TR-reduced mesh) must
    give the SAME operator-route ICOHP on the same system to numerical noise --
    diamond C on a 3x3x3 mesh, which (unlike a pure time-reversal reduction)
    actually engages the cubic point group, so this exercises AO-projector
    handling at k-points that are symmetry- but not literally mesh-identical
    between the two runs. A mismatch here would mean the AO projectors (shared
    by both the operator and eigenvalue COHP routes) are not being evaluated
    correctly at symmetry-reduced k-points -- a bug independent of, and more
    serious than, the eigenvalue route's band-completeness leak, since it would
    also corrupt the already-"validated" diamond/GaAs operator-route numbers.
    (The eigenvalue route is NOT asserted here: it is measurably scheme-
    sensitive even at generous nbands -- see the module docstring -- which is a
    separate, expected symptom of its incomplete-basis H~, not a rotation bug.)
    """
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    upf = parse_upf(f"{FIX}/PD_C_PBE_std.upf")
    cell, pos = si_fcc(a=3.567)

    icohp_by_scheme = {}
    for use_sym in (False, True):
        system = setup_system(cell, pos, [0, 0], [upf], ecut=45 * RY,
                              kmesh=(3, 3, 3), nbands=16, use_symmetry=use_sym)
        res = scf(system, PBE(), smearing="gaussian", width=0.05, etol=1e-8,
                  rhotol=1e-7, verbose=False, kerker=True)
        assert res.converged
        c_op = cohp.cohp(res, pairs=[(0, 1)], method="operator", width=0.2)
        icohp_by_scheme[use_sym] = c_op.pair_icohp["1-2"]

    assert abs(icohp_by_scheme[True] - icohp_by_scheme[False]) < 1e-6, icohp_by_scheme


@pytest.mark.standard
def test_cohp_resolve_images_diamond_bond_degeneracy(diamond_c):
    """The concrete physics evidence behind the cohp.py module docstring's
    "gap 1 (bond resolution) is closed by resolve_images" claim: diamond's
    4 nearest-neighbour bonds are related by the tetrahedral site symmetry, so
    an EXACT per-bond decomposition must give sublattice/4 for every one of
    them. Verified two ways here: (1) the public resolve_images=True API gives
    ~1/4 of the resolve_images=False sublattice number, and (2) scanning every
    integer lattice shift R that puts atom 1 at the nearest-neighbour distance
    from atom 0 finds exactly 4 such R, and _accumulate_images gives the
    IDENTICAL per-bond ICOHP for all 4 (not merely close -- exact, since the
    site symmetry is exact), summing back to the sublattice number."""
    res, system = diamond_c

    sub = cohp.cohp(res, pairs=[(0, 1)], method="operator", width=0.2,
                    resolve_images=False)
    per_bond = cohp.cohp(res, pairs=[(0, 1)], method="operator", width=0.2,
                        resolve_images=True)
    assert per_bond.bond_images == {"1-2": (0, 0, 0)}
    ratio = per_bond.pair_icohp["1-2"] / sub.pair_icohp["1-2"]
    assert 0.24 < ratio < 0.26, (per_bond.pair_icohp, sub.pair_icohp, ratio)

    # cross-check via every symmetry-equivalent image directly, not just the
    # one _nearest_image_R happens to pick by naive fractional-coordinate
    # rounding -- rebuild the per-k projections/AO Hamiltonian once (the same
    # objects cohp.cohp() builds internally) and reuse them for every R.
    cell = np.asarray(system.grid.cell, dtype=float)
    pos = system.positions.detach().cpu().numpy()
    d0 = pos[0] - pos[1]
    dists: dict[float, list[np.ndarray]] = {}
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                R = np.array([n1, n2, n3])
                dist = round(float(np.linalg.norm(d0 - R @ cell)), 6)
                if dist > 1e-6:
                    dists.setdefault(dist, []).append(R)
    nn_dist = min(dists)
    nn_Rs = dists[nn_dist]
    assert len(nn_Rs) == 4, "diamond's tetrahedral coordination"

    system_, nspin, eig, coeffs, fermi, device, _ = cohp._unpack_result(res)
    cols = cohp._atomic_columns(system_)
    atom_of = np.array([c.atom for c in cols])
    kw = system_.kweights.to(device)
    from gradwave.core.hamiltonian import HamiltonianK, projectors
    proj_per_k, htilde_per_k, eig_per_k, kw_flat, kpts = [], [], [], [], []
    for ik, sph in enumerate(system_.spheres):
        c = coeffs[0][ik].to(device)
        e = eig[0, ik].to(device)
        q = cohp._ao_projectors_k(system_, sph, cols, device)
        overlap = torch.einsum("ig,jg->ij", q.conj(), q)
        becp = torch.einsum("bg,pg->bp", c, q.conj())
        proj = cohp._lowdin_project(becp, overlap)
        ois = cohp.o_inv_sqrt(overlap)
        pd = system_.proj_data[ik]
        p = projectors(pd, system_.positions).to(device)
        h = HamiltonianK(sph, system_.grid.shape, res.v_eff, pd, p)
        htilde = cohp._htilde_operator(q, ois, h.apply)
        proj_per_k.append(proj)
        htilde_per_k.append(htilde)
        eig_per_k.append(e)
        kw_flat.append(float(kw[ik]))
        kpts.append(np.asarray(sph.k_frac, dtype=float))

    pair_list = [(0, 1, nn_dist)]
    per_R = []
    for R in nn_Rs:
        _, ic = cohp._accumulate_images(
            proj_per_k, htilde_per_k, eig_per_k, kw_flat, kpts, atom_of,
            pair_list, {(0, 1): R}, nspin, len(system_.spheres), 2.0, res.fermi)
        per_R.append(ic[(0, 1)])
    assert max(per_R) - min(per_R) < 1e-6, per_R
    assert abs(sum(per_R) - sub.pair_icohp["1-2"]) < 0.5, (sum(per_R), sub.pair_icohp)


@pytest.mark.standard
def test_cohp_iao_does_not_reduce_bond_overshoot_diamond(diamond_c):
    """Guards a real, measured (NOT hoped-for) result: IAO spans the occupied
    manifold exactly (charge_spilling collapses toward zero) but does NOT
    shrink the per-bond COHP magnitude relative to the plain pswfc basis on
    diamond -- it fixes occupied-space completeness, a different property from
    the real-space extent that drives the inter-atomic H~ overlap, so it does
    not close the basis-diffuseness gap docs/plans/cohp-contracted-basis.md
    originally hypothesized it would. This pins the measurement (module
    docstring, QUANTITATIVE STATUS #2) so a future change to _iao_projectors_k
    that silently reintroduces that claim gets caught."""
    res, system = diamond_c

    base = cohp.cohp(res, pairs=[(0, 1)], method="operator", width=0.2,
                     resolve_images=True, basis="pswfc")
    iao = cohp.cohp(res, pairs=[(0, 1)], method="operator", width=0.2,
                    resolve_images=True, basis="iao")

    # IAO closes occupied-space completeness far better than pswfc ...
    assert iao.charge_spilling < 1e-3
    assert iao.charge_spilling < 0.1 * base.charge_spilling
    # ... but does NOT shrink (and here is marginally larger than) the pswfc
    # per-bond magnitude -- the basis-diffuseness gap is still open.
    assert abs(iao.pair_icohp["1-2"]) >= 0.9 * abs(base.pair_icohp["1-2"])


@pytest.mark.standard
def test_cohp_contracted_basis_moves_toward_lobster_direction(diamond_c):
    """A first, honest measurement of pseudo.sto_basis's fitted contracted
    STO basis (docs/plans/cohp-contracted-basis.md's deferred step 4) on the
    SAME diamond cell IAO was shown not to fix. Unlike IAO (which fixes
    occupied-space completeness, not real-space extent, and measurably lands
    on the WRONG side -- see test_cohp_iao_does_not_reduce_bond_overshoot_diamond
    above), the contracted basis targets extent directly by fitting at the
    free-atom level with an explicit compactness regularizer
    (pseudo.sto_basis.fit_contracted_sto's tail_reg). Measured: -20.68 eV/bond
    vs. pswfc's -20.96 eV/bond and IAO's -21.25 eV/bond -- the RIGHT
    direction (toward LOBSTER's -9.64 eV/bond), at the conservative default
    tail_reg=1e-3/n_primitives=3. This is NOT a claim of having closed the
    gap (it closes ~1.3% of the ~2.1x overshoot) -- it pins that the sign of
    the effect is correct, unlike IAO, and that the machinery runs
    end-to-end through the real operator route on a real PAW-free NC
    system. Closing the gap further is a tail_reg/n_primitives calibration
    question with no in-tree LOBSTER oracle to calibrate against yet."""
    res, system = diamond_c

    base = cohp.cohp(res, pairs=[(0, 1)], method="operator", width=0.2,
                     resolve_images=True, basis="pswfc")
    con = cohp.cohp(res, pairs=[(0, 1)], method="operator", width=0.2,
                    resolve_images=True, basis="contracted")

    assert con.charge_spilling < 0.05  # still representing the occupied states reasonably
    # the actual point: strictly SMALLER magnitude than pswfc, moving toward
    # LOBSTER -- the direction IAO gets wrong.
    assert abs(con.pair_icohp["1-2"]) < abs(base.pair_icohp["1-2"])


@pytest.fixture(scope="module")
def diamond_c_paw():
    """Converged diamond-carbon PBE PAW SCF (C.pbe-n-kjpaw_psl, which carries both
    PP_PSWFC chi and PP_FULL_WFC AE/PS partial waves), for the PAW AE-reconstruction
    COHP tests. a set so the C-C bond is LOBSTER's 1.547 A; full (unreduced) 2x2x2
    mesh so resolve_images can pick out the single nearest-neighbour bond."""
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp
    paw = parse_upf_paw(f"{FIX}/C.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 4 * 1.547 / np.sqrt(3)
    cell, pos = si_fcc(a=a)
    system = setup_uspp(cell, pos, [0, 0], [paw], ecut=60 * RY, kmesh=(2, 2, 2),
                        use_symmetry=False, nbands=24)
    res = scf_uspp(system, PBE(), nspin=1, smearing="gaussian", width=0.05,
                   etol=1e-8, rhotol=1e-7, verbose=False)
    assert res.converged
    return res, system


@pytest.mark.slow
def test_cohp_paw_ae_reconstruction_brackets_lobster(diamond_c_paw):
    """PAW all-electron reconstruction of the COHP projection on diamond, against
    the per-bond LOBSTER oracle -9.586 eV (per-spin ISPIN=1 convention). MEASURED
    (asus, ecut 60 Ry, 2x2x2 unreduced): the smooth pseudo projection, the AE
    reconstruction (candidate A, `paw_reconstruct=True`) and the S-metric
    augmentation (candidate B, `paw_reconstruct='smetric'`) come out at -9.43,
    -8.69 and -10.96 eV respectively -- the two candidates BRACKET the oracle, A
    below and B above, and neither cleanly lands on it (see cohp() docstring).

    This pins that measured picture rather than a hoped-for closure: it asserts
    the bracket (A magnitude < smooth < B magnitude, and signed A > oracle > B)
    and that the augmentation actually fires and stays bonding. It does NOT claim
    the residual is closed -- it is bracketed, and LOBSTER's own number is a
    basis-dependent approximation, not the exact AE value candidate A targets."""
    res, _ = diamond_c_paw
    ORACLE = -9.586  # LOBSTER per-bond C-C ICOHP, per-spin

    def bond(**kw):
        # paw_reconstruct is an EIGENVALUE-route feature (it augments the AO
        # becp); the operator route (now the default when the USPPResult carries
        # v_eff) ignores it. Pin method="eigenvalue" so this test exercises the
        # augmentation modes it is about.
        c = cohp.cohp(res, pairs=[(0, 1)], resolve_images=True,
                      summed_spins=False, method="eigenvalue", **kw)
        return c.pair_icohp["1-2"]

    smooth = bond(paw_reconstruct=False)
    ae = bond(paw_reconstruct=True)       # candidate A (default)
    smetric = bond(paw_reconstruct="smetric")  # candidate B

    # all three are bonding
    assert smooth < 0 and ae < 0 and smetric < 0
    # the augmentation actually changes the number (not a silent no-op)
    assert abs(ae - smooth) > 0.1
    assert abs(smetric - smooth) > 0.1
    # candidate A reduces the magnitude; candidate B increases it (measured bracket)
    assert abs(ae) < abs(smooth) < abs(smetric)
    # and the two candidates bracket the LOBSTER oracle (A below, B above)
    assert ae > ORACLE > smetric
    # each candidate lands within a sane factor of the oracle (regression band)
    assert 0.8 < ae / ORACLE < 1.0
    assert 1.05 < smetric / ORACLE < 1.25


@pytest.mark.slow
def test_cohp_paw_reconstruct_summed_spins_half(diamond_c_paw):
    """The summed_spins per-spin convention composes with the PAW reconstruction:
    on the PAW USPPResult too, summed_spins=False is exactly half the physical
    two-electron value, for both the smooth and the AE-reconstructed projection."""
    res, _ = diamond_c_paw
    for recon in (False, True):
        full = cohp.cohp(res, pairs=[(0, 1)], resolve_images=True, method="eigenvalue",
                         summed_spins=True, paw_reconstruct=recon)
        half = cohp.cohp(res, pairs=[(0, 1)], resolve_images=True, method="eigenvalue",
                         summed_spins=False, paw_reconstruct=recon)
        assert half.pair_icohp["1-2"] == pytest.approx(
            0.5 * full.pair_icohp["1-2"], rel=1e-9)


@pytest.mark.slow
def test_cohp_paw_operator_route(diamond_c_paw):
    """PAW COHP operator route H~ = O_S^{-1/2}<chi|H_PAW|chi>O_S^{-1/2}.

    The eigenvalue route H~ = P†diag(eps)P carries the plane-wave energy zero,
    so its per-bond ICOHP shifts when the eigenvalue reference is shifted, and
    its all-pairs sum rule overcounts the band energy. The operator route
    applies the converged PAW Hamiltonian directly, so it is EXACTLY invariant
    under a rigid eigenvalue/Fermi shift (the strong correctness check) and its
    charge spilling is positive (the S-metric overlap is consistent).

    MEASURED (asus, C.pbe-n-kjpaw_psl, ecut 60 Ry, 2x2x2 unreduced): the
    operator route is bit-exactly reference-invariant (|delta ICOHP| = 0 under a
    +1 eV shift) where the eigenvalue route leaks ~2e-2 eV; charge_spill is
    +0.0038 (positive); per-bond ICOHP -10.19 eV (summed_spins=True), i.e.
    ratio_x2 = 2.13 vs the LOBSTER oracle -9.586 (per-spin) -- the residual is
    the contracted-STO basis-definition difference, not a reference leak."""
    res, _ = diamond_c_paw
    assert res.v_eff is not None and res.dscr is not None  # SCF retained them

    op = cohp.cohp(res, pairs=[(0, 1)], resolve_images=True, method="operator",
                   summed_spins=False)
    assert op.method == "operator"          # the operator route actually fired
    assert op.pair_icohp["1-2"] < 0         # bonding
    assert op.charge_spilling > 0           # S-metric overlap is consistent

    # reference invariance: a rigid +1 eV shift leaves the operator ICOHP exact
    base = op.pair_icohp["1-2"]
    eig_save, fermi_save = res.eigenvalues.clone(), res.fermi
    res.eigenvalues = res.eigenvalues + 1.0
    res.fermi = res.fermi + 1.0
    try:
        shifted = cohp.cohp(res, pairs=[(0, 1)], resolve_images=True,
                            method="operator", summed_spins=False).pair_icohp["1-2"]
        leaky = cohp.cohp(res, pairs=[(0, 1)], resolve_images=True,
                          method="eigenvalue", summed_spins=False).pair_icohp["1-2"]
    finally:
        res.eigenvalues, res.fermi = eig_save, fermi_save
    assert shifted == pytest.approx(base, abs=1e-6)     # operator: invariant
    base_eig = cohp.cohp(res, pairs=[(0, 1)], resolve_images=True,
                         method="eigenvalue", summed_spins=False).pair_icohp["1-2"]
    assert abs(leaky - base_eig) > 1e-4                 # eigenvalue: leaks
