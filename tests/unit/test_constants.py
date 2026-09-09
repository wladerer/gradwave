"""Internal consistency of gradwave.constants — a cheap guard against a future
factor edit silently drifting one derived constant away from its definition.

constants.py is the single source of truth for unit conversions; every module
imports from it rather than carrying its own factor (the CLAUDE.md rule). A
typo in one literal, or a derived factor pinned to a stale number instead of
its identity, would corrupt energies/forces/stress everywhere at once. These
assert the documented identities to machine precision plus the literal CODATA
2018 values.
"""

import math

from gradwave import constants as c


def test_rydberg_is_half_hartree():
    assert c.RY_EV == c.HARTREE_EV / 2.0            # exact by identity


def test_hbar2_2m_identity():
    # ħ²/2mₑ = Ry·a₀²  (the plane-wave kinetic prefactor)
    assert c.HBAR2_2M == c.RY_EV * c.BOHR_ANG**2    # exact


def test_e2_identity():
    # e²/(4πε₀) = Ha·a₀  (the Coulomb prefactor)
    assert c.E2 == c.HARTREE_EV * c.BOHR_ANG        # exact


def test_kbar_is_ten_gpa():
    assert c.EV_A3_TO_KBAR == 10.0 * c.EV_A3_TO_GPA  # exact


def test_bohr_angstrom_roundtrip():
    # BOHR_ANG is Å per Bohr; going Å→Bohr→Å must return the identity
    ang = 3.14159
    bohr = ang / c.BOHR_ANG
    assert math.isclose(bohr * c.BOHR_ANG, ang, rel_tol=0.0, abs_tol=1e-15)


def test_literal_codata_values():
    # CODATA 2018 (scipy.constants), pinned so a typo in any literal is caught
    assert c.HARTREE_EV == 27.211386245988
    assert c.BOHR_ANG == 0.529177210903
    assert c.RY_EV == 13.605693122994
    assert c.KB_EV == 8.617333262e-5
    assert c.ALPHA_FS == 7.2973525693e-3
    assert c.EV_A3_TO_GPA == 160.2176634


def test_derived_values_are_physical():
    # loose sanity so the identities above can't be satisfied by two matching
    # wrong numbers: ħ²/2mₑ ≈ 3.81 eV·Å², e²/(4πε₀) ≈ 14.4 eV·Å
    assert math.isclose(c.HBAR2_2M, 3.80998, rel_tol=1e-4)
    assert math.isclose(c.E2, 14.39964, rel_tol=1e-4)


def test_fine_structure_magnetic_identity():
    # the documented collapse e²/(mc²) = 2·α²·(ħ²/2mₑ)/E2 [in Å], used by the
    # magnetic-response prefactors — pins ALPHA_FS into a real combination
    e2_over_mc2 = 2.0 * c.ALPHA_FS**2 * c.HBAR2_2M / c.E2
    # classical electron radius r_e ≈ 2.8179403e-5 Å
    assert math.isclose(e2_over_mc2, 2.8179403262e-5, rel_tol=1e-6)
