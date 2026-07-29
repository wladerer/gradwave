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
    system = setup_system(cell, pos, [0, 0], [upf], ecut=28 * RY, kmesh=(2, 1, 1))
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
    so this is a fair, non-flaky point to regression-test the bracket at."""
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    upf = parse_upf(f"{FIX}/PD_C_PBE_std.upf")
    cell, pos = si_fcc(a=3.567)
    system = setup_system(cell, pos, [0, 0], [upf], ecut=45 * RY, kmesh=(2, 2, 2),
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
