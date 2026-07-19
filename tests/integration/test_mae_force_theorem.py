"""Force-theorem MAE evaluator (postscf/mae.py): exactness gates.

Two rungs:

1. No SOC -> exact rotation invariance. With scalar-relativistic pseudos the
   spin rotation is an exact symmetry of H (nothing in the Hamiltonian is
   locked to the lattice spin frame), so the frozen-potential band sum must be
   IDENTICAL for every magnetization direction to solver precision. This
   gates the whole pipeline - the rigid rotation of (m, B_xc), the SU(2)
   seed, the one-shot solve, and the band sum - with the anisotropy switched
   off by construction. The reference direction must also reproduce the SCF
   spectrum, pinning the frozen potential against the converged one.

2. SOC -> force theorem tracks self-consistency. On L1_0 FePt (fully
   relativistic Fe+Pt, small full mesh) the force-theorem band-energy
   difference must reproduce the two-SCF total-energy difference to the
   second-order accuracy the theorem promises. The mesh is far from
   k-converged, so the number is not the physical MAE - both routes share the
   mesh, and the gate is their mutual agreement, not the literature value.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.mae import force_theorem_mae
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"

SQ2 = 1.0 / np.sqrt(2.0)


def _o2_system(L=6.0, d=1.21):
    o = parse_upf(f"{PSE}/O_ONCV_PBE-1.2.upf")
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    return setup_system(cell, pos, [0, 0], [o, o], ecut=30 * RY, kmesh=(1, 1, 1),
                        nbands=8, time_reversal=False)


@pytest.mark.slow
def test_no_soc_band_sum_is_rotation_invariant():
    torch.set_num_threads(8)
    system = _o2_system()
    xc = NoncollinearXC(LSDA_PW92())
    res = scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 0.5], [0, 0, 0.5]],
                           smearing="gaussian", width=0.1, etol=1e-9,
                           rhotol=1e-8, max_iter=200, verbose=False)
    assert res.converged

    dirs = [[0, 0, 1.0], [1.0, 0, 0], [0, SQ2, SQ2], [0, 0, -1.0]]
    ft = force_theorem_mae(res, xc, dirs, verbose=False)

    # scalar-relativistic: the rotation is exact, so zero anisotropy
    assert float(ft.mae.abs().max()) < 1e-6, \
        f"no-SOC anisotropy {float(ft.mae.abs().max()):.2e} eV"
    # the reference direction reproduces the converged SCF spectrum
    d_eig = float((ft.eigenvalues[0] - res.eigenvalues).abs().max())
    assert d_eig < 1e-4, f"ref-direction spectrum off by {d_eig:.2e} eV"


def _fept_scf(axis, kmesh=(2, 2, 2)):
    fe = parse_upf(f"{PSE}/Fe_ONCV_PBE_FR-1.0.upf")
    pt = parse_upf(f"{PSE}/Pt_ONCV_PBE_FR-1.0.upf")
    a, c = 2.723, 3.712
    cell = np.diag([a, a, c])
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    ax = np.array(axis, float)
    init = [(3.0 * ax).tolist(), (0.4 * ax).tolist()]
    system = setup_system(cell, pos, [0, 1], [fe, pt], ecut=30 * RY, kmesh=kmesh,
                          nbands=30, use_symmetry=False, time_reversal=False)
    res = scf_noncollinear(system, NoncollinearXC(LSDA_PW92()),
                           mag_vec_init=init, smearing="gaussian", width=0.1,
                           etol=1e-9, rhotol=1e-7, max_iter=150,
                           mixing_alpha=0.3, mixing_history=12, verbose=False)
    assert res.converged
    return res


@pytest.mark.slow
def test_soc_force_theorem_tracks_self_consistent_mae():
    torch.set_num_threads(8)
    xc = NoncollinearXC(LSDA_PW92())
    res001 = _fept_scf([0, 0, 1.0])
    res100 = _fept_scf([1.0, 0, 0])
    d_scf = float(res100.energies.free_energy) - float(res001.energies.free_energy)

    ft = force_theorem_mae(res001, xc, [[0, 0, 1.0], [1.0, 0, 0]], verbose=False)
    d_ft = float(ft.mae[1])

    # the reference direction reproduces the converged SCF spectrum
    d_eig = float((ft.eigenvalues[0] - res001.eigenvalues).abs().max())
    assert d_eig < 1e-4, f"ref-direction spectrum off by {d_eig:.2e} eV"

    # second-order agreement: same sign, magnitude within the force-theorem
    # band (30% + a small absolute floor for the near-degenerate case)
    assert d_ft * d_scf > 0 or abs(d_scf) < 5e-5, \
        f"FT {d_ft * 1e3:+.4f} vs SCF {d_scf * 1e3:+.4f} meV: opposite sign"
    assert abs(d_ft - d_scf) < 0.3 * abs(d_scf) + 5e-5, \
        f"FT {d_ft * 1e3:+.4f} vs SCF {d_scf * 1e3:+.4f} meV"
