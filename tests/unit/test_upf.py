import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from gradwave.constants import BOHR_ANG, E2, RY_EV
from gradwave.pseudo.atomic import rhoatom_of_q
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
