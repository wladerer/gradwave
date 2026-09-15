"""Force-theorem MAE evaluator (postscf/mae.py): exactness gates.

Two rungs:

1. No SOC -> exact rotation invariance. With scalar-relativistic pseudos the
   spin rotation is an exact symmetry of H (nothing in the Hamiltonian is
   locked to the lattice spin frame), so the frozen-potential band sum must be
   IDENTICAL for every magnetization direction to solver precision. This
   gates the whole pipeline with the anisotropy switched off by construction:
   the rigid rotation of (m, B_xc), the SU(2) seed, the one-shot solve, and
   the band sum. The reference direction must also reproduce the SCF
   spectrum, pinning the frozen potential against the converged one.

2. SOC -> force theorem tracks self-consistency. On L1_0 FePt (fully
   relativistic Fe+Pt, small full mesh) the force-theorem band-energy
   difference must reproduce the two-SCF total-energy difference to the
   second-order accuracy the theorem predicts. The mesh is far from
   k-converged, so the number is not the physical MAE. Both routes share the
   mesh, and the gate checks that they agree with each other rather than
   with the literature value.

3. Per-direction magnetic-IBZ fold -> full-mesh band sums. With magmoms=
   each one-shot solve folds into its own direction's Shubnikov IBZ; the
   folded band free energies must reproduce the full-mesh ones. The fold is
   exact for the collinear part of the frozen magnetization. The SOC-induced
   transverse textures in m(r) set the residual this gate bounds.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.mae import (
    _fband_strained,
    _frozen_oneshot,
    force_theorem_mae,
    mae_strained,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import PSEUDOS, RY, fept_l10

PSE = PSEUDOS
SQ2 = 1.0 / np.sqrt(2.0)


def _occupied_spectrum_delta(ft_eigs, scf_eigs, fermi, width):
    """Max |Δε| between the one-shot and SCF spectra over the bands that enter
    F_band (occupied + smearing tail).

    The frozen-potential reference solve reproduces the converged SCF exactly
    band by band — except the single topmost band of the Davidson block, a
    virtual tens of eV above E_F. With no state above it that band is never
    converged to its eigenvalue by EITHER solve: the residual meets ``tol`` but
    the Rayleigh-Ritz value is ambiguous, so the SCF's final solve and the
    force-theorem solve land on different Ritz mixtures (~0.1 eV apart on the
    L1_0 FePt mesh). It carries zero occupation and no anisotropy, so it must
    not gate the reproduction. Compare the bands below E_F + 10·width, which
    the force theorem reproduces to solver precision (~1e-8 eV)."""
    occ = scf_eigs < fermi + 10.0 * width
    return float((ft_eigs - scf_eigs)[occ].abs().max())


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
    d_eig = _occupied_spectrum_delta(ft.eigenvalues[0], res.eigenvalues,
                                     res.fermi, 0.1)
    assert d_eig < 1e-4, f"ref-direction spectrum off by {d_eig:.2e} eV"


def _fept_scf(axis, kmesh=(2, 2, 2)):
    fe = parse_upf(f"{PSE}/Fe_ONCV_PBE_FR-1.0.upf")
    pt = parse_upf(f"{PSE}/Pt_ONCV_PBE_FR-1.0.upf")
    cell, pos = fept_l10()
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
    d_eig = _occupied_spectrum_delta(ft.eigenvalues[0], res001.eigenvalues,
                                     res001.fermi, 0.1)
    assert d_eig < 1e-4, f"ref-direction spectrum off by {d_eig:.2e} eV"

    # second-order agreement: same sign, magnitude within the force-theorem
    # band (30% + a small absolute floor for the near-degenerate case)
    assert d_ft * d_scf > 0 or abs(d_scf) < 5e-5, \
        f"FT {d_ft * 1e3:+.4f} vs SCF {d_scf * 1e3:+.4f} meV: opposite sign"
    assert abs(d_ft - d_scf) < 0.3 * abs(d_scf) + 5e-5, \
        f"FT {d_ft * 1e3:+.4f} vs SCF {d_scf * 1e3:+.4f} meV"


@pytest.mark.slow
def test_folded_directions_match_full_mesh():
    torch.set_num_threads(8)
    xc = NoncollinearXC(LSDA_PW92())
    res = _fept_scf([0, 0, 1.0])

    # two directions whose Shubnikov groups fold the (2,2,2) mesh (8 -> 6)
    # and two whose groups leave every point in its own orbit (8 -> 8)
    dirs = [[0, 0, 1.0], [1.0, 0, 0], [SQ2, SQ2, 0], [SQ2, 0, SQ2]]
    full = force_theorem_mae(res, xc, dirs, verbose=False)
    fold = force_theorem_mae(res, xc, dirs, verbose=False,
                             magmoms=[[0, 0, 3.0], [0, 0, 0.4]])

    assert full.nk == [8, 8, 8, 8]
    assert fold.nk == [6, 8, 6, 8], f"folds {fold.nk}"

    # the fold is exact for the frozen fields (measured residual ~4e-12 eV);
    # the gate leaves room for the reference SCF's convergence-level
    # symmetry breaking of rho, nothing more
    d_f = (fold.band_free_energies - full.band_free_energies).abs().max()
    assert float(d_f) < 1e-6, f"folded vs full F_band off by {float(d_f):.2e} eV"
    d_mae = (fold.mae - full.mae).abs().max()
    assert float(d_mae) < 1e-6, f"folded vs full MAE off by {float(d_mae):.2e} eV"


@pytest.mark.slow
def test_mae_strain_gradient_matches_fd():
    """The differentiable force-theorem MAE-strain gradient (postscf.mae.
    mae_strained / mae_strain_gradient) is validated two ways on L1_0 FePt:

    1. per-direction F_band(ε=0) reproduces force_theorem_mae's band energies
       (the band-energy potential-expectation assembly is correct), and
    2. the analytic dMAE/dη for a volume-conserving tetragonal strain
       ε = η·diag(-1,-1,2) matches a central finite difference of the same
       frozen-orbital functional (autograd is the correct Hellmann-Feynman
       gradient). Coarse mesh — this checks the gradient identity, not a
       converged physical MAE."""
    torch.set_num_threads(8)
    xc = NoncollinearXC(LSDA_PW92())
    res = _fept_scf([0, 0, 1.0])
    ref = torch.as_tensor(res.mag_vec / np.linalg.norm(res.mag_vec),
                          dtype=torch.float64)
    prep_h = _frozen_oneshot(res, xc, (1, 0, 0), ref, smearing="gaussian",
                             width=0.1, diago_tol=1e-10)
    prep_e = _frozen_oneshot(res, xc, (0, 0, 1), ref, smearing="gaussian",
                             width=0.1, diago_tol=1e-10)

    # (1) F_band assembly == the established force theorem, per direction
    ft = force_theorem_mae(res, xc, [[0, 0, 1.0], [1.0, 0, 0]], verbose=False)
    eps0 = torch.zeros(3, 3, dtype=torch.float64)
    fb_h = float(_fband_strained(res, xc, prep_h, eps0))
    fb_e = float(_fband_strained(res, xc, prep_e, eps0))
    assert abs(fb_h - float(ft.band_free_energies[1])) < 1e-4
    assert abs(fb_e - float(ft.band_free_energies[0])) < 1e-4

    # (2) autograd dMAE/dη == central FD (same frozen orbitals)
    tetra = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 2.0]],
                         dtype=torch.float64)
    eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(mae_strained(res, xc, prep_h, prep_e, eps), eps)
    g = 0.5 * (g + g.T)
    d_ag = float(-g[0, 0] - g[1, 1] + 2 * g[2, 2])

    def _e(eta):
        return float(mae_strained(res, xc, prep_h, prep_e, eta * tetra))
    d_fd = (_e(1e-4) - _e(-1e-4)) / 2e-4
    rel = abs(d_ag - d_fd) / (abs(d_fd) + 1e-12)
    assert rel < 1e-4, f"autograd {d_ag:.6f} vs FD {d_fd:.6f} eV, rel {rel:.2e}"
