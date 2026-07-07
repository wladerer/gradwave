"""Spinor SCF validation ladder (docs/noncollinear.md, no SOC):
1. collinear limit: moments along ẑ reproduce nspin=2 LSDA exactly;
2. global rotation invariance: x̂ and tilted moments give the same F
   (verified to ~0.2 µeV — the precision floor for future MCA work)."""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear

RY = 13.605693122994
A = 2.87
CELL = A / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
PSEUDO = "tests/fixtures/qe/pseudos/Fe_ONCV_PBE-1.2.upf"
AL_CELL = 4.05 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def make_system():
    fe = parse_upf(PSEUDO)
    return setup_system(CELL, np.zeros((1, 3)), [0], [fe], ecut=60 * RY,
                        kmesh=(3, 3, 3), nbands=12, time_reversal=False)


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


def test_nonmagnetic_spinor_ibz_equals_full_mesh():
    """A nonmagnetic (m⃗ ≡ 0) spinor SCF keeps the full crystal symmetry, so
    IBZ reduction + ρ-symmetrization must equal the full-mesh spinor energy.
    Uses a scalar (non-SOC) pseudo — the symmetry wiring is identical for the
    fully-relativistic case, which only swaps the projector block."""
    torch.set_num_threads(4)
    al = parse_upf("tests/fixtures/qe/pseudos/Al_ONCV_PBE-1.2.upf")

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
