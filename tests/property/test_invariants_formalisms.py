"""Property invariants beyond cubic NC Si — formalism breadth.

``tests/property/test_scf_invariants.py`` and ``test_metamorphic.py`` exercise the
highest-value exact identities (force sum rule, translation invariance of the
energy, charge conservation) but ONLY on insulating, cubic, norm-conserving Si.
That leaves formalism-specific breaks invisible: a metal (fractional smeared
occupations, a Fermi level) and a PAW system (augmentation charge ρ_aug, one-
center terms) route through code paths Si-NC never touches. This module carries
the same identities onto:

  * a smeared **metal** (fcc/CsCl-like Al, gaussian smearing), and
  * a **PAW** system (Si kjpaw via the ultrasoft/PAW loop), and

adds the audit-flagged gap — **charge conservation AFTER density symmetrization on
a low-symmetry (P-1) cell**, where the space-group projector does real folding
work and must not leak the total charge.

Every check is an exact identity of the theory (or holds to a known aliasing
floor), so it stands at the small, loose, coarse settings used here — the tiny
cell's physical inaccuracy is irrelevant to the identity. A convention / phase /
normalization / occupation bug in a formalism-specific path breaks it by orders
of magnitude.
"""

import math

import numpy as np
import pytest
import torch

from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.forces import forces as compute_forces
from gradwave.pseudo.upf import parse_upf
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.symmetry import RhoSymmetrizer, find_spacegroup
from tests.helpers import PSEUDOS, RY, pseudo


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(8)


# --- metal: fcc/CsCl-like Al, smeared ----------------------------------------

_AL_A = 3.9  # Å, cubic 2-atom cell (short-ish but only needs to converge)
_AL_CELL = np.diag([_AL_A, _AL_A, _AL_A])
# body-centered second atom pushed off the special site along z so the two atoms
# carry genuine, opposite forces (a force sum rule with actual teeth) while the
# cell stays metallic and fast.
_AL_POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.55]]) @ _AL_CELL
_AL_SCF_KW = dict(smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8,
                  max_iter=150, verbose=False)


def _al_system():
    al = parse_upf(str(PSEUDOS / "Al_ONCV_PBE-1.2.upf"))
    return setup_system(_AL_CELL, _AL_POS, [0, 0], [al], ecut=20 * RY,
                        kmesh=(3, 3, 3), nbands=20)


@pytest.fixture(scope="module")
def _al_res():
    res = scf(_al_system(), LDA_PW92(), **_AL_SCF_KW)
    assert res.converged, "smeared Al metal SCF did not converge"
    return res


@pytest.mark.standard
def test_metal_force_sum_rule(_al_res):
    """Σ_a F_a = 0 on a smeared metal: translational invariance of the cell
    forbids a net force. The two Al atoms carry real, opposite forces (the cell
    is off-equilibrium), and their sum vanishes to SCF tolerance. Guards the
    fractional-occupation force assembly (occupations enter the nonlocal term)."""
    f = compute_forces(_al_res).numpy()
    assert np.all(np.isfinite(f))
    # the cell is genuinely off-equilibrium: individual forces are nonzero, so
    # the sum-rule is not vacuously satisfied
    assert np.abs(f).max() > 1e-2, f"expected nonzero Al forces, got {f}"
    assert np.abs(f.sum(axis=0)).max() < 1e-6, f"metal ΣF ≠ 0: {f.sum(axis=0)}"


@pytest.mark.standard
def test_metal_charge_conservation(_al_res):
    """∫ρ(r) dr = N_electrons for the converged metal density (fractional
    smeared occupations summed over the k-mesh). A lost k-weight or a smearing-
    occupation normalization bug breaks the electron count."""
    grid = _al_res.system.grid
    dvol = float(grid.volume) / float(grid.n_points)
    n_int = float(_al_res.rho.sum()) * dvol
    assert abs(n_int - _al_res.system.n_electrons) < 1e-6, (
        f"metal ∫ρ={n_int:.8f} != N_e={_al_res.system.n_electrons}")


@pytest.mark.standard
def test_metal_translation_invariance(_al_res):
    """Rigid grid-incommensurate shift of the metal leaves the free energy
    invariant up to the XC-quadrature (egg-box) floor. The Fermi level / smeared
    entropy term must ride along with the density unchanged."""
    t = np.array([0.5137, -0.2911, 0.4302])  # Å, grid-incommensurate
    res_shift = scf(setup_system(_AL_CELL, _AL_POS + t, [0, 0],
                                 [parse_upf(str(PSEUDOS / "Al_ONCV_PBE-1.2.upf"))],
                                 ecut=20 * RY, kmesh=(3, 3, 3), nbands=20),
                    LDA_PW92(), **_AL_SCF_KW)
    assert res_shift.converged
    f0 = float(_al_res.energies.free_energy)
    f1 = float(res_shift.energies.free_energy)
    # semicore Al at 20 Ry has a larger egg-box than valence Si; bound above it
    assert abs(f1 - f0) < 5e-4, f"metal ΔF under translation = {f1 - f0:.2e} eV"


# --- PAW: Si kjpaw ------------------------------------------------------------

_SI_PAW_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
_SI_PAW_POS = np.array([[0.0, 0, 0], [5.43 / 4] * 3])
_PAW_SCF_KW = dict(smearing="none", etol=1e-8, rhotol=1e-7, max_iter=40,
                   verbose=False)


def _si_paw_system(pos=None):
    paw = parse_upf_paw(PSEUDOS / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    return setup_uspp(_SI_PAW_CELL, _SI_PAW_POS if pos is None else pos,
                      [0, 0], [paw], ecut=20 * RY, kmesh=(2, 2, 2))


@pytest.fixture(scope="module")
def _si_paw_res():
    res = scf_uspp(_si_paw_system(), PBE(), **_PAW_SCF_KW)
    assert res["converged"], "PAW Si SCF did not converge"
    return res


@pytest.mark.standard
def test_paw_charge_conservation(_si_paw_res):
    """∫ρ(r) dr = N_electrons for a PAW density ρ = ρ_smooth + ρ_aug. This is the
    PAW-specific charge check: the augmentation charge Q_ij(G) must add exactly
    the pseudized-vs-all-electron charge deficit back so the total integrates to
    the electron count. The floor is the UPF's own PP_Q vs ∫q⁰_ij precision."""
    res = _si_paw_res
    grid = res["system"].grid
    n = float(res["rho"].sum()) * float(grid.volume) / float(grid.n_points)
    assert abs(n - res["system"].n_electrons) < 1e-5, (
        f"PAW ∫ρ={n:.8f} != N_e={res['system'].n_electrons}")


@pytest.mark.standard
def test_paw_translation_invariance(_si_paw_res):
    """Rigid translation of the PAW cell leaves the total energy invariant.

    Uses a GRID-COMMENSURATE shift (an integer number of FFT steps along a
    lattice vector), which is a pure relabeling of the grid, so the invariance is
    EXACT (machine precision) rather than sitting on the egg-box floor. This is
    the strong form of the identity: it exercises the position dependence of the
    augmentation phases e^{iG·τ}, the local-potential structure factor, and the
    one-center assembly together — a wrong G or a sign slip in any of those would
    NOT re-index cleanly and would break the exactness. (A grid-INcommensurate
    shift on this sharp augmentation charge only probes the XC-quadrature egg-box,
    which is ~1e-3 eV and non-monotone in ecut — a much weaker check; verified
    separately that it stays at that quadrature floor.)
    """
    grid = _si_paw_res["system"].grid
    n0 = grid.shape[0]
    step = _SI_PAW_CELL[0] / n0  # one FFT step along a₁ → exact relabeling
    res_shift = scf_uspp(_si_paw_system(pos=_SI_PAW_POS + step[None, :]),
                         PBE(), **_PAW_SCF_KW)
    assert res_shift["converged"]
    e0 = float(_si_paw_res["energies"].total)
    e1 = float(res_shift["energies"].total)
    assert abs(e1 - e0) < 1e-8, f"PAW ΔE under commensurate shift = {e1 - e0:.2e} eV"


# --- charge conservation after symmetrization on a low-symmetry cell ---------

def _p1_si():
    """A triclinic cell with two Si atoms at ±r about the origin: space group
    P-1 (only {E, inversion}). Genuinely low symmetry, but the density
    symmetrizer still does real folding work — exactly the audit-flagged regime."""
    cell = np.array([[3.10, 0.00, 0.00],
                     [0.70, 3.30, 0.00],
                     [0.50, 0.40, 3.60]])
    frac_r = np.array([0.11, 0.23, 0.31])
    frac = np.array([frac_r, -frac_r % 1.0])
    pos = frac @ cell
    return cell, frac, pos


@pytest.mark.standard
def test_charge_conservation_after_symmetrization_low_symmetry():
    """The density symmetrizer is a projector onto the space-group-invariant
    subspace; it MUST conserve the total charge (∫ρ = N_e) because the identity
    op and every rotation map G=0 → G=0. The audit flagged this on low-symmetry
    cells, where the projector is nontrivial (unlike a P1 cell, where it is the
    identity) yet the dens_mask drops the Nyquist shell — a wrong mask or a
    corrupted G=0 map would leak charge here while passing on cubic Si.

    Run with ``use_symmetry=False`` so the converged density is a GENERIC (un-
    presymmetrized) field, then symmetrize it explicitly and check the integral.
    """
    cell, frac, pos = _p1_si()
    si = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    sys = setup_system(cell, pos, [0, 0], [si], ecut=16 * RY,
                       kmesh=(2, 2, 2), use_symmetry=False)
    res = scf(sys, LDA_PW92(), smearing="none", etol=1e-8, rhotol=1e-7,
              max_iter=80, verbose=False)
    assert res.converged, "low-symmetry Si SCF did not converge"

    sg = find_spacegroup(cell, frac, [0, 0])
    # genuinely low symmetry: nontrivial group (so the projector does work) but
    # small — P-1 is order 2. (>1 gives the test teeth; the box closes under
    # inversion for any FFT dims, so no equal-dims box is needed.)
    assert 1 < sg.n_ops <= 4, f"expected a small nontrivial group, got {sg.n_ops}"

    grid = res.system.grid
    dvol = float(grid.volume) / float(grid.n_points)
    n_before = float(res.rho.sum()) * dvol
    assert abs(n_before - res.system.n_electrons) < 1e-6  # baseline conservation

    rsym = RhoSymmetrizer(grid.shape, sg, dens_mask=grid.dens_mask)
    rho_g = r_to_g(res.rho.detach().to(torch.complex128))
    rho_g_sym = rsym.apply(rho_g)
    rho_sym = g_to_r_box(rho_g_sym, real=True)
    n_after = float(rho_sym.sum()) * dvol

    assert math.isfinite(n_after)
    assert abs(n_after - res.system.n_electrons) < 1e-6, (
        f"symmetrization leaked charge: ∫ρ {n_before:.8f} → {n_after:.8f} "
        f"(N_e={res.system.n_electrons})")
