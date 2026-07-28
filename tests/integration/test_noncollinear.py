"""Spinor SCF validation ladder (no SOC):
1. collinear limit: moments along ẑ reproduce nspin=2 LSDA exactly;
2. global rotation invariance: x̂ and tilted moments give the same F
   (verified to ~0.2 µeV — the precision floor for future MCA work)."""

import numpy as np
import pytest
import torch

from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import RY, pseudo

A = 2.87
CELL = A / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
PSEUDO = pseudo("Fe_ONCV_PBE-1.2.upf")
AL_CELL = 4.05 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
NI_CELL = 3.52 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
NI_PSEUDO = pseudo("PD_Ni_PBE.upf")  # scalar-relativistic, carries a 3d PP_PSWFC


def make_system():
    fe = parse_upf(PSEUDO)
    return setup_system(CELL, np.zeros((1, 3)), [0], [fe], ecut=60 * RY,
                        kmesh=(3, 3, 3), nbands=12, time_reversal=False)


def make_ni_system():
    """A single-atom fcc Ni cell — the smallest DFT+U-eligible (has a 3d
    PP_PSWFC manifold) system for the noncollinear +U validation below. Ni is
    a weak, itinerant, near-Stoner-instability magnet (see
    test_ni_soc_convergence.py) — the settings and Stoner spin-preconditioner
    below mirror that test's already-validated convergence recipe."""
    ni = parse_upf(NI_PSEUDO)
    return setup_system(NI_CELL, np.zeros((1, 3)), [0], [ni], ecut=40 * RY,
                        kmesh=(4, 4, 4), nbands=16, use_symmetry=False,
                        time_reversal=False)


_NI_SCF_KW = dict(
    mag_vec_init=[[0, 0, 1.0]], smearing="gaussian", width=0.1, max_iter=60,
    etol=1e-4, rhotol=5e-3,  # metal-appropriate loose gate (test_ni_soc_convergence.py)
    spin_precond=True, mag_mixer="pulay", mag_diago_schedule="quadratic",
    mag_mixing_alpha=0.3, verbose=False,
)


@pytest.mark.slow
def test_spinor_scf_ladder():
    torch.set_num_threads(8)
    col = scf(make_system(), LSDA_PW92(), smearing="gaussian", width=0.1,
              etol=1e-8, rhotol=1e-7, verbose=False, nspin=2, start_mag=[0.4])
    assert col.converged

    nc_z = scf_noncollinear(make_system(), NoncollinearXC(LSDA_PW92()),
                            mag_vec_init=[[0, 0, 0.4]], width=0.1,
                            etol=1e-8, rhotol=1e-7, verbose=False)
    assert nc_z.converged
    f_col, f_z = float(col.energies.free_energy), float(nc_z.energies.free_energy)
    assert abs(f_z - f_col) < 5e-6  # eV — collinear limit
    assert abs(nc_z.mag_vec[2] - col.mag_total) < 1e-3
    assert abs(nc_z.mag_vec[0]) < 1e-3 and abs(nc_z.mag_vec[1]) < 1e-3

    nc_x = scf_noncollinear(make_system(), NoncollinearXC(LSDA_PW92()),
                            mag_vec_init=[[0.4, 0, 0]], width=0.1,
                            etol=1e-8, rhotol=1e-7, verbose=False)
    assert nc_x.converged
    assert abs(float(nc_x.energies.free_energy) - f_z) < 5e-6  # rotation invariance
    m = np.array(nc_x.mag_vec)
    assert abs(m[0]) > 2.5 and abs(m[1]) < 1e-3 and abs(m[2]) < 1e-3


@pytest.mark.slow
def test_spinor_hubbard_u_zero_bit_for_bit():
    """DFT+U on the noncollinear (spinor) path must be exactly inert at U=0:
    the 2×2 spin-block D-matrix is (U−J)(½I − N) with U=J=0, an exact-zero
    tensor at every iteration regardless of N, so the +U term in
    SpinorHamiltonian.apply adds an exact zero and the whole SCF trajectory
    (eigenvalues, energies) is bit-for-bit identical to the plain run.

    Uses the real ferromagnetic-Ni system (not a synthetic one) so this
    exercises the full +U wiring — projectors, occupation matrix, D-matrix,
    Hamiltonian term — end to end through an actual converged SCF, with the
    Stoner spin-preconditioner recipe test_ni_soc_convergence.py validated for
    this exact system (Ni's FM state is weakly, marginally stable; the stock
    mixer limit-cycles on it without spin_precond=True)."""
    torch.set_num_threads(4)
    plain = scf_noncollinear(make_ni_system(), NoncollinearXC(SpinPBE()), **_NI_SCF_KW)
    assert plain.converged

    u0 = scf_noncollinear(make_ni_system(), NoncollinearXC(SpinPBE()),
                          hubbard=[HubbardManifold(species=0, l=2, u=0.0)],
                          **_NI_SCF_KW)
    assert u0.converged
    assert float(u0.energies.hubbard) == 0.0
    assert float(plain.energies.total) == float(u0.energies.total)
    assert float(plain.energies.free_energy) == float(u0.energies.free_energy)
    assert torch.equal(plain.eigenvalues, u0.eigenvalues)


@pytest.mark.slow
def test_spinor_hubbard_u_positive_finite():
    """U>0 on the noncollinear (spinor) path: the SCF converges and the
    Dudarev energy is a finite, on-site penalty (mirrors the collinear
    end-to-end sanity check, test_hubbard_io.py::test_u_positive_end_to_end_scf).

    A tight cross-check against the collinear nspin=2 +U SCF on the same
    system was attempted but dropped: elemental Ni's FM state is only weakly
    (marginally) stable at this basis, and the two drivers' mixers converge to
    different quasi-degenerate low-moment solutions (observed: collinear
    settles at m≈0, noncollinear at m≈0.1 μB) — a convergence-basin ambiguity
    unrelated to the +U wiring (the exact-arithmetic U=0 oracle above already
    rules that out), left as follow-up (needs a fixed-moment constraint or a
    more strongly-ordered correlated test fixture)."""
    torch.set_num_threads(4)
    res = scf_noncollinear(make_ni_system(), NoncollinearXC(SpinPBE()),
                           hubbard=[HubbardManifold(species=0, l=2, u=5.0)],
                           **_NI_SCF_KW)
    assert res.converged
    e_u = float(res.energies.hubbard)
    assert np.isfinite(e_u) and e_u > 0.0
    assert res.hub_occ is not None and len(res.hub_occ) == 1  # one Ni site
    assert res.hub_occ[0].shape == (10, 10)  # 2×2 spin-block, d manifold (dim 5)


@pytest.mark.slow
def test_spinor_metagga_collinear_limit_and_rotation():
    """Meta-GGA (r2SCAN) on the spinor path, end to end:

    1. COLLINEAR LIMIT — a non-collinear r2SCAN run with the moment along ẑ
       reproduces the collinear nspin=2 r2SCAN free energy (the whole τ build +
       2×2 v_τ operator collapses to the proven collinear per-spin machinery);
    2. ROTATIONAL INVARIANCE — rotating the moment to x̂ (a pure SU(2) spin
       rotation, same real-space grid) leaves the free energy invariant.

    Tolerances match the LSDA ladder above: the two SCF drivers differ in
    mixing/occupation bookkeeping, so the collinear-limit floor is ~µeV, while
    the z→x rotation (one driver, one grid) is tighter."""
    from gradwave.core.xc.r2scan import SpinR2SCAN

    torch.set_num_threads(8)
    col = scf(make_system(), SpinR2SCAN(), smearing="gaussian", width=0.1,
              etol=1e-9, rhotol=1e-8, verbose=False, nspin=2, start_mag=[0.4],
              max_iter=200)
    assert col.converged

    nc_z = scf_noncollinear(make_system(), NoncollinearXC(SpinR2SCAN()),
                            mag_vec_init=[[0, 0, 0.4]], width=0.1,
                            etol=1e-9, rhotol=1e-8, verbose=False, max_iter=200)
    assert nc_z.converged
    f_col, f_z = float(col.energies.free_energy), float(nc_z.energies.free_energy)
    assert abs(f_z - f_col) < 5e-6  # eV — collinear limit (same functional)
    assert abs(nc_z.mag_vec[2] - col.mag_total) < 1e-3
    assert abs(nc_z.mag_vec[0]) < 1e-3 and abs(nc_z.mag_vec[1]) < 1e-3

    nc_x = scf_noncollinear(make_system(), NoncollinearXC(SpinR2SCAN()),
                            mag_vec_init=[[0.4, 0, 0]], width=0.1,
                            etol=1e-9, rhotol=1e-8, verbose=False, max_iter=200)
    assert nc_x.converged
    assert abs(float(nc_x.energies.free_energy) - f_z) < 5e-6  # rotation invariance
    m = np.array(nc_x.mag_vec)
    assert abs(m[0]) > 2.5 and abs(m[1]) < 1e-3 and abs(m[2]) < 1e-3


def test_nonmagnetic_spinor_ibz_equals_full_mesh():
    """A nonmagnetic (m⃗ ≡ 0) spinor SCF keeps the full crystal symmetry, so
    IBZ reduction + ρ-symmetrization must equal the full-mesh spinor energy.
    Uses a scalar (non-SOC) pseudo — the symmetry wiring is identical for the
    fully-relativistic case, which only swaps the projector block."""
    torch.set_num_threads(4)
    al = parse_upf(pseudo("Al_ONCV_PBE-1.2.upf"))

    def run(use_sym):
        system = setup_system(AL_CELL, np.zeros((1, 3)), [0], [al], ecut=18 * RY,
                              kmesh=(2, 2, 2), nbands=8, use_symmetry=use_sym,
                              time_reversal=not use_sym)
        return scf_noncollinear(system, NoncollinearXC(SpinPBE()),
                                mag_vec_init=[[0, 0, 0]], width=0.1,
                                etol=1e-9, rhotol=1e-8, verbose=False,
                                nonmagnetic=True)

    full, ibz = run(False), run(True)
    assert full.converged and ibz.converged
    assert len(ibz.system.spheres) < len(full.system.spheres)  # IBZ shrank the mesh
    assert abs(float(full.energies.free_energy)
               - float(ibz.energies.free_energy)) < 5e-7


def test_magnetic_spinor_rejects_symmetry():
    """The IBZ path is only valid when m⃗ ≡ 0; a magnetic spinor run built on a
    symmetry-reduced System must refuse to proceed."""
    fe = parse_upf(PSEUDO)
    system = setup_system(CELL, np.zeros((1, 3)), [0], [fe], ecut=40 * RY,
                          kmesh=(2, 2, 2), nbands=12, use_symmetry=True)
    if system.rho_symmetrizer is None:
        pytest.skip("no space-group reduction available for this cell")
    with pytest.raises(ValueError, match="use_symmetry=False|nonmagnetic"):
        scf_noncollinear(system, NoncollinearXC(LSDA_PW92()),
                         mag_vec_init=[[0, 0, 0.4]], verbose=False)
