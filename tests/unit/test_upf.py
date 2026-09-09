import dataclasses
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson

from gradwave.constants import BOHR_ANG, E2, RY_EV
from gradwave.pseudo.atomic import core_density_of_q, rhoatom_of_q
from gradwave.pseudo.kb import beta_form_factors
from gradwave.pseudo.local import alpha_z, vloc_of_g
from gradwave.pseudo.radial import sbt
from gradwave.pseudo.upf import parse_upf
from gradwave.pseudo.upf_paw import parse_upf_paw

PSEUDO_DIR = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
FE_KJPAW = PSEUDO_DIR / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"


@pytest.fixture(scope="module")
def si():
    return parse_upf(PSEUDO_DIR / "Si_ONCV_PBE-1.2.upf")


def test_header_and_units(si):
    assert si.element == "Si"
    assert si.z_valence == 4.0
    assert si.l_max == 1
    assert si.n_proj == 4
    assert [b.l for b in si.betas] == [0, 0, 1, 1]
    # hand-checked values from the raw file, through the documented conversions
    assert np.isclose(si.dij[0, 0], 1.3605849050e01 * RY_EV, rtol=1e-12)
    assert np.isclose(si.r[1] - si.r[0], 0.01 * BOHR_ANG, rtol=1e-10)
    assert si.dij.shape == (4, 4)
    # dij is diagonal for ONCV Si (checked by eye in the file)
    off = si.dij - np.diag(np.diag(si.dij))
    assert np.abs(off).max() == 0.0


def test_read_root_recovers_junk_after_document_element(tmp_path):
    """Real-world PSlibrary UPF v2 files carry content outside the single
    <UPF>…</UPF> root — a generator banner before it or a log/stray element
    after </UPF>. The standard ElementTree.parse rejects the latter with
    "junk after document element". The reader must carve out the root span and
    still return a correct pseudopotential (regression for the FeS2/pyrite
    kjpaw benchmark)."""
    clean = parse_upf_paw(FE_KJPAW)  # baseline: fixture is well-formed

    raw = FE_KJPAW.read_text()
    trailing = "\n<PP_LOG>generation succeeded</PP_LOG>\ntrailing junk & <garbage\n"
    # sanity: content after </UPF> is exactly the failure mode we are fixing —
    # ElementTree rejects it with "junk after document element".
    with pytest.raises(ET.ParseError, match="junk after document element"):
        ET.fromstring(raw + trailing)

    # the reader must also survive a leading generator banner before <UPF>
    junked = "Generated using 'atomic' code by A. Dal Corso  v.6.3\n" + raw + trailing

    path = tmp_path / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"
    path.write_text(junked)
    d = parse_upf_paw(path)

    # right element / valence / angular momenta
    assert d.element == "Fe"
    assert d.z_valence == clean.z_valence
    assert d.l_max == clean.l_max
    assert [b.l for b in d.betas] == [b.l for b in clean.betas]
    assert d.n_proj == clean.n_proj and d.n_proj > 0
    # PAW one-center dataset present and intact
    assert d.is_paw
    assert d.paw_occ is not None and len(d.aewfc) > 0 and len(d.pswfc) > 0
    # radial grid length sane and unchanged by the carve-out
    assert len(d.r) == len(clean.r) and len(d.r) > 100
    assert np.array_equal(d.r, clean.r)
    assert np.array_equal(d.dij, clean.dij)


def test_read_root_rejects_truncated_root(tmp_path):
    """A file whose <UPF> root is not closed is a broken/truncated pseudo, not
    recoverable junk — the reader must still raise rather than silently accept
    partial data."""
    raw = FE_KJPAW.read_text()
    truncated = raw[: len(raw) // 2]  # no </UPF>
    path = tmp_path / "truncated.UPF"
    path.write_text(truncated)
    with pytest.raises(ET.ParseError):
        parse_upf_paw(path)


def test_al_parses():
    al = parse_upf(PSEUDO_DIR / "Al_ONCV_PBE-1.2.upf")
    assert al.element == "Al"
    # SG15 Al carries the 2s2p semicore: 11 valence electrons, not 3
    assert al.z_valence == 11.0


def test_rhoatom_normalization(si):
    # ρ̂(0) = ∫ 4πr²ρ dr ≈ Z_val. SG15 truncates the atomic density at the
    # mesh edge (r ≈ 3.2 Å), losing ~1% of the tail charge — this is why the
    # SAD guess rescales to the exact electron count downstream.
    zhat = rhoatom_of_q(si, np.array([0.0]))[0]
    assert abs(zhat - si.z_valence) < 0.05


def test_rhoatom_integrates_full_mesh(si):
    # rhoatom_of_q must integrate the WHOLE radial mesh, not the 10-bohr _msh
    # local-channel clip (the atomic density has no −Z/r tail, so clipping only
    # sheds charge and distorts the SAD guess shape under its G=0 rescale). Pin
    # the no-clip behaviour: the form factor equals the full-mesh SBT exactly.
    q = np.array([0.0, 0.3, 1.0])
    full = sbt(0, si.rhoatom, si.r, si.rab, q)
    np.testing.assert_allclose(rhoatom_of_q(si, q), full, rtol=0, atol=0)


def test_vloc_long_range_is_coulomb(si):
    # For small G, v(G) → −4π Z e²/G² (the erf split must reassemble the tail)
    g = np.array([0.02, 0.05])
    v = vloc_of_g(si, g)
    coulomb = -4.0 * np.pi * si.z_valence * E2 / g**2
    assert np.allclose(v, coulomb, rtol=2e-3)


def test_vloc_decays_at_large_g(si):
    v = vloc_of_g(si, np.array([2.0, 30.0]))
    assert abs(v[1]) < 1e-2 * abs(v[0])


def test_vloc_rejects_g0(si):
    with pytest.raises(ValueError):
        vloc_of_g(si, np.array([0.0]))
    assert np.isfinite(alpha_z(si))


def test_beta_form_factors_shapes_and_l_behavior(si):
    q = np.array([0.0, 0.5, 1.0, 4.0])
    F = beta_form_factors(si, q)
    assert F.shape == (4, 4)
    # l>0 projectors vanish at q=0 (j_l(0)=0 for l≥1); l=0 do not
    assert abs(F[2, 0]) < 1e-12 and abs(F[3, 0]) < 1e-12
    assert abs(F[0, 0]) > 1e-3


# --- KB projector radial normalization -------------------------------------
# The tests above pin the β form-factor SHAPES and the j_l(0)=0 behavior but
# never pin the projector AMPLITUDE, so a normalization error in the radial β
# (a dropped/extra scale factor, a wrong √-power in the BOHR conversion) would
# pass silently. These pin it two ways, both parse-only (no SCF):
#
#  1. The radial norm  N_i = ∫ β_i(r)² r² dr = ∫ (r·β_i)(r)² dr.  For a
#     norm-conserving ONCV pseudo the projectors are normalized to UNITY, so
#     this is a physically-correct value, not merely a regression constant:
#     N_i = 1 to ~3e-7 for every Si projector.
#  2. beta_form_factors(q) — the quantity the KB energy actually consumes —
#     at a few q, frozen to its verified value (rtol below the spline-vs-exact
#     ~1e-11 and well inside a 2× scaling).
#
# A uniform 2× scaling of β sends N_i → 4 and F → 2F, which both pins reject
# (proven by test_kb_norm_pin_bites_on_beta_rescale).

# beta_form_factors(si, [0.5, 1.0, 2.0]); verified on asus, matches the direct
# radial SBT (_beta_form_factors_exact) to 8e-12.
_SI_BETA_FF_Q = np.array([0.5, 1.0, 2.0])
_SI_BETA_FF_REF = np.array([
    [-0.212783990849919, -0.210076724585821, -0.198790778582415],
    [0.397616152765118, 0.370866137370578, 0.275315648881791],
    [0.018363415904703, 0.036514301761989, 0.071355897363715],
    [-0.076828137657762, -0.145526842460631, -0.226487568281246],
])


def _kb_radial_norms(upf):
    """∫ β_i(r)² r² dr per projector, integrated over the radial mesh up to the
    projector's own cutoff index (β is r·β on the mesh, so β²r² = (rβ)²)."""
    return np.array([
        simpson(b.rbeta[: b.cutoff_idx] ** 2, x=upf.r[: b.cutoff_idx])
        for b in upf.betas
    ])


def test_kb_projector_radial_norm_is_unity(si):
    """The four ONCV Si KB projectors are normalized: ∫ β² r² dr = 1. Pins the
    radial-β amplitude (the √BOHR conversion and mesh integration) to its
    norm-conserving ground truth."""
    norms = _kb_radial_norms(si)
    assert norms.shape == (4,)
    np.testing.assert_allclose(norms, 1.0, rtol=0, atol=1e-5)


def test_kb_beta_form_factor_values(si):
    """Pin beta_form_factors (the amplitude the KB energy consumes) at a few q
    to its verified value — the SHAPE tests above never pinned the magnitude."""
    F = beta_form_factors(si, _SI_BETA_FF_Q)
    np.testing.assert_allclose(F, _SI_BETA_FF_REF, rtol=1e-5, atol=1e-8)


def test_kb_norm_pin_bites_on_beta_rescale(si):
    """Guard the guard: a uniform 2× scaling of the radial β must break both the
    unity-norm pin (N → 4) and the form-factor pin (F → 2F). Proves the two
    tests above bite on a real normalization error rather than passing
    vacuously. Parse-only — builds a scaled UPF in memory, no SCF."""
    scaled_betas = tuple(
        dataclasses.replace(b, rbeta=2.0 * b.rbeta) for b in si.betas
    )
    bad = dataclasses.replace(si, betas=scaled_betas)

    # norm quadruples — the unity pin rejects it
    bad_norms = _kb_radial_norms(bad)
    np.testing.assert_allclose(bad_norms, 4.0, rtol=0, atol=1e-4)
    assert not np.allclose(bad_norms, 1.0, rtol=0, atol=1e-5)

    # form factors double — the value pin rejects it (fresh UPF id ⇒ no spline
    # cache collision with `si`)
    F_bad = beta_form_factors(bad, _SI_BETA_FF_Q)
    np.testing.assert_allclose(F_bad, 2.0 * _SI_BETA_FF_REF, rtol=1e-5, atol=1e-8)
    assert not np.allclose(F_bad, _SI_BETA_FF_REF, rtol=1e-5, atol=1e-8)


# --- NLCC partial-core charge regression pin --------------------------------
# The integrated NLCC core charge Q = ∫ 4π r² ρc(r) dr equals the l=0 core form
# factor at q=0 (core_density_of_q(upf, 0)). gradwave stores densities in e/Å³,
# so parse_upf/parse_upf_paw divide PP_NLCC by BOHR_ANG**3 on read. These
# ground-truth Q (electrons) pin that conversion and the NLCC parse against a
# future units/parse regression. Dropping the /BOHR_ANG**3 factor would shrink Q
# by ~1/BOHR_ANG**3 ≈ 6.75×, which the rtol pin rejects (see test below).
#
# Only the pseudos that actually ship a PP_NLCC block are pinned: among the NC
# ONCV fixtures only the scalar-/fully-relativistic Si (_sr/_fr) carry a core
# correction — the SG15 "-1.2" Cu/Al/Fe/Si and Fe_FR pseudos have
# core_correction="F" (no NLCC), so there is nothing to pin for those elements
# here. The PAW kjpaw pseudos pin the smooth ρ̃_core (PP_NLCC) transform.
# _sr/_fr Si Q verified against core_density_of_q: 0.72414 / 0.72425.
_NLCC_CASES = [
    pytest.param("Si_ONCV_PBE_sr.upf", parse_upf, 0.7241, id="Si_ONCV_sr(NC)"),
    pytest.param("Si_ONCV_PBE_fr.upf", parse_upf, 0.7242, id="Si_ONCV_fr(NC)"),
    pytest.param("Fe.pbe-spn-kjpaw_psl.1.0.0.UPF", parse_upf_paw, 0.4500, id="Fe_kjpaw(PAW)"),
    pytest.param("Si.pbe-n-kjpaw_psl.1.0.0.UPF", parse_upf_paw, 2.6047, id="Si_kjpaw(PAW)"),
    pytest.param("Cu.pbe-dn-kjpaw_psl.1.0.0.UPF", parse_upf_paw, 9.7716, id="Cu_kjpaw(PAW)"),
]


@pytest.mark.parametrize("name, parser, q_ref", _NLCC_CASES)
def test_nlcc_core_charge(name, parser, q_ref):
    """Pin the integrated NLCC partial-core charge to its ground truth."""
    path = PSEUDO_DIR / name
    if not path.exists():
        pytest.skip(f"NLCC fixture {name} not committed")
    upf = parser(path)
    assert upf.core_rho is not None, f"{name} unexpectedly has no NLCC core density"
    q = core_density_of_q(upf, np.array([0.0]))[0]
    assert np.isclose(q, q_ref, rtol=1e-3), f"{name}: Q={q:.6f} vs ref {q_ref}"


def test_nlcc_pin_bites_on_dropped_bohr_conversion():
    """Guard the guard: without the e/Å³ conversion (÷BOHR_ANG³ on read) Q would
    be ~6.75× too small, and the rtol=1e-3 pin must reject that. This proves the
    pin above actually bites rather than passing vacuously."""
    path = PSEUDO_DIR / "Si_ONCV_PBE_sr.upf"
    if not path.exists():
        pytest.skip("Si_ONCV_PBE_sr.upf not committed")
    upf = parse_upf(path)
    q_correct = core_density_of_q(upf, np.array([0.0]))[0]
    # emulate the pre-fix parse (density left in e/Bohr³): Q scales by BOHR_ANG³
    q_broken = q_correct * BOHR_ANG**3
    assert np.isclose(q_correct, 0.7241, rtol=1e-3)
    assert 6.0 < q_correct / q_broken < 7.5  # ≈ 1/BOHR_ANG³ ≈ 6.75×
    assert not np.isclose(q_broken, 0.7241, rtol=1e-3)
