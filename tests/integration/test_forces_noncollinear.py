"""Hellmann–Feynman forces on the noncollinear/SOC spinor path (Stage 1 of the
NC/SOC forces gate, docs/gate_inventory.md) + the noncollinear DFT+U force
(Stage 2, ``hubbard_force_noncollinear``).

``postscf.forces.forces()`` previously had NO noncollinear/SOC branch at all —
a spinor ``NCResult`` was simply unsupported. ``_forces_noncollinear`` adds it,
generalizing the same three position-dependent terms (Ewald / E_loc structure
factor / E_NL projector phase) the collinear path already uses, onto the
spinor Hamiltonian the way ``postscf.stress``'s ``is_fr`` branch already
generalized stress. Two checks mirror this codebase's reduction-to-known-limit
+ finite-difference discipline (``test_forces_nspin2.py``, ``test_stress_soc.
py``):

1. COLLINEAR LIMIT (non-fr branch): a scalar-relativistic (no SOC) pseudo run
   through ``scf_noncollinear`` must give the SAME force as the plain
   collinear nspin=2 SCF on the same displaced geometry — exercises the
   "scalar projectors acting on the up/down spinor halves separately" branch
   of ``_forces_noncollinear``.
2. FINITE DIFFERENCE on an actual SOC (``is_fr``) system: analytic
   ``_forces_noncollinear`` forces vs. a central finite difference of the
   noncollinear free energy on fully-relativistic GaAs — genuinely exercises
   the j-resolved spinor nonlocal projector-phase force term.

Stage 2 (+U): the same finite-displacement discipline does NOT carry over
directly to ``hubbard_force_noncollinear`` in isolation — the Hellmann–Feynman
theorem only guarantees that the TOTAL energy is stationary in the orbitals at
self-consistency, not that the E_U TERM is separately stationary, so a finite
difference of E_U alone from independently re-converged SCFs picks up an
orbital-relaxation contribution the frozen-orbital force does not (confirmed
numerically while developing this test: FD of E_U alone disagreed with the
analytic force by an O(1) factor, while a synthetic gradcheck of the exact
same closed-form E_U(pos) — see ``tests/unit/test_hubbard.py::
test_hubbard_force_noncollinear_gradcheck`` — matched to gradcheck precision).
That gradcheck (mirroring the ONLY existing collinear +U force oracle,
``test_hubbard_force_gradcheck`` — there is no full-SCF-FD collinear +U force
test in this repo either) plus the exact U=0 zero
(``test_hubbard_force_noncollinear_u_zero_is_exact_zero``) are the Stage-2
self-oracles. ``test_hubbard_force_noncollinear_real_scf_sanity`` below adds a
genuine converged +U spinor SCF (not synthetic orbitals) as an end-to-end
sanity check, reusing the only NLCC-free-OR-NOT distinction that matters here:
every Hubbard-eligible (has PP_PSWFC) fully-relativistic pseudo in the fixture
set also carries an NLCC core charge, so ``forces()`` (Stage 1) itself cannot
run on this system yet (NLCC force is a documented, separate follow-up gate,
see the module docstring of postscf/forces.py); this test therefore checks
``hubbard_force_noncollinear`` alone, which has no NLCC dependence.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.postscf.forces import forces, hubbard_force_noncollinear
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"

pytestmark = pytest.mark.slow  # full SCF(s); not a fast-gate test

# Displaced Si cell (same geometry as test_forces_vs_qe / test_forces_nspin2).
A = 5.43
SI_CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0.0, 0.0], [0.24, 0.26, 0.255]]) @ SI_CELL

# Sheared, off-symmetry GaAs cell (test_stress_soc.py's fixture — SOC lives in
# the As/Ga p projectors, so the j-resolved nonlocal force term is genuinely
# exercised).
_CELL0, _ = si_fcc(5.653)
_SHEAR = np.array([[1.00, 0.02, 0.015], [0.0, 0.995, 0.01], [0.0, 0.0, 1.01]])
GAAS_CELL = _CELL0 @ _SHEAR.T
GAAS_FRAC = np.array([[0.0, 0.0, 0.0], [0.26, 0.24, 0.25]])


def test_forces_noncollinear_reduces_to_collinear_limit():
    """A scalar-relativistic (no SOC) pseudo run through scf_noncollinear (with
    m⃗ pinned to zero) must reproduce the plain collinear nspin=2 force on the
    same displaced Si cell to SCF-convergence precision — the non-fr branch of
    _forces_noncollinear (plain KB projectors on the up/down spinor halves)."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")  # no NLCC

    sys_c = setup_system(SI_CELL, SI_POS, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))
    r1 = scf(sys_c, LSDA_PW92(), smearing="gaussian", width=0.02, nspin=2,
             start_mag=[0.0, 0.0], etol=1e-10, rhotol=1e-9, verbose=False)
    assert r1.converged
    f1 = forces(r1).cpu().numpy()

    sys_nc = setup_system(SI_CELL, SI_POS, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2),
                          use_symmetry=False, time_reversal=False)
    assert not sys_nc.is_fr
    xc = NoncollinearXC(LSDA_PW92())
    r2 = scf_noncollinear(sys_nc, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                          smearing="gaussian", width=0.02, etol=1e-10, rhotol=1e-9,
                          verbose=False, nonmagnetic=True)
    assert r2.converged
    f2 = forces(r2).cpu().numpy()
    assert np.abs(f2 - f1).max() < 1e-6, f"\ncollinear:\n{f1}\nnoncollinear:\n{f2}"


def test_forces_soc_matches_finite_difference():
    """Fully-relativistic (SOC) GaAs: the analytic spinor force matches a
    central finite difference of the noncollinear free energy — exercises the
    j-resolved spinor nonlocal projector-phase force term end to end."""
    torch.set_num_threads(8)
    ga = parse_upf(FIX / "pseudos" / "Ga_ONCV_PBE_FR-1.0.upf")  # no NLCC
    as_ = parse_upf(FIX / "pseudos" / "As_ONCV_PBE_FR-1.1.upf")  # no NLCC

    def run(pos, etol=1e-9, rhotol=1e-8, max_iter=300):
        system = setup_system(GAAS_CELL, pos, [0, 1], [ga, as_], ecut=20 * RY,
                              kmesh=(1, 1, 1), use_symmetry=False, time_reversal=False)
        xc = NoncollinearXC(SpinPBE())
        return scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                                smearing="gaussian", width=0.1, etol=etol,
                                rhotol=rhotol, max_iter=max_iter, verbose=False)

    pos0 = GAAS_FRAC @ GAAS_CELL
    res = run(pos0)
    assert res.converged and res.system.is_fr
    f = forces(res, remove_net=False).cpu().numpy()

    h = 5e-3  # cartesian displacement (Å), atom 2 along x
    e = []
    for sign in (+1, -1):
        pos = pos0.copy()
        pos[1, 0] += sign * h
        r = run(pos)
        assert r.converged
        e.append(float(r.energies.free_energy))
    fd = -(e[0] - e[1]) / (2 * h)
    assert abs(fd - float(f[1, 0])) < 1e-3, (fd, float(f[1, 0]))


def test_hubbard_force_noncollinear_real_scf_sanity():
    """Real (not synthetic) converged +U spinor SCF, end to end: a two-atom
    fully-relativistic Bi "molecule in a box" with U=3 eV on the 5d manifold.

    Bi_ONCV_PBE_fr.upf carries an NLCC core charge, so forces() (Stage 1) is
    not runnable on this system yet (a separate, documented follow-up gate —
    see the module docstring); this test therefore only exercises
    hubbard_force_noncollinear, which has no NLCC dependence. Checks: the U=0
    force is exactly zero (the additive +U term vanishes identically, the
    "Stage 2 forces reduce to Stage 1" oracle — see test_hubbard.py for why a
    finite-difference cross-check needs frozen, not re-converged, orbitals),
    and U>0 gives finite, nonzero forces with the expected antisymmetry (two
    equivalent atoms, net force zero)."""
    torch.set_num_threads(8)
    bi = parse_upf(FIX / "pseudos" / "Bi_ONCV_PBE_fr.upf")
    cell = 7.0 * np.eye(3)
    pos = np.array([[0.0, 0, 0], [1.5, 1.3, 1.4]])

    def run(manifolds):
        system = setup_system(cell, pos, [0, 0], [bi], ecut=20 * RY, kmesh=(1, 1, 1),
                              use_symmetry=False, time_reversal=False)
        xc = NoncollinearXC(SpinPBE())
        res = scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                               smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8,
                               max_iter=200, verbose=False, hubbard=manifolds)
        assert res.converged
        return res

    manifolds0 = [HubbardManifold(species=0, l=2, u=0.0)]
    res0 = run(manifolds0)
    assert float(res0.energies.hubbard) == 0.0
    f0 = hubbard_force_noncollinear(res0, manifolds0).cpu().numpy()
    assert np.abs(f0).max() == 0.0

    manifolds = [HubbardManifold(species=0, l=2, u=3.0)]
    res = run(manifolds)
    e_u = float(res.energies.hubbard)
    assert np.isfinite(e_u) and e_u > 0.0
    assert res.hub_occ is not None and len(res.hub_occ) == 2  # two Bi sites

    fu = hubbard_force_noncollinear(res, manifolds).cpu().numpy()
    assert np.isfinite(fu).all()
    assert np.abs(fu).max() > 1e-3  # genuinely nonzero
    # net-force sum rule (uniform-shift invariance); the floor here is the
    # SCF's own convergence noise (etol=1e-9/rhotol=1e-8), not exact zero —
    # observed residual ~1.7e-5 at these settings.
    assert np.abs(fu.sum(axis=0)).max() < 1e-3
