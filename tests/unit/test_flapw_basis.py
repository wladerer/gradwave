"""Unit tests for the EFG anion basis-recipe helpers (gradwave.flapw.basis)."""
import pytest

from gradwave.flapw.basis import efg_anion_basis, merge_basis


def test_efg_anion_basis_period2_recipe():
    """The default (period-2) recipe: l=0 2s LO moved to 2p + unconfined l=1 HELO."""
    spec = efg_anion_basis(["O"])
    assert spec["los"] == {"O": [(0, "2s"), (1, {"e": 90.0, "confine": False})]}
    assert spec["el_override"] == {"O": {0: "2p"}}


def test_efg_anion_basis_multiple_species_and_helo_energy():
    spec = efg_anion_basis(["O", "F"], helo_e=75.0)
    assert set(spec["los"]) == {"O", "F"}
    for sp in ("O", "F"):
        assert spec["los"][sp][1] == (1, {"e": 75.0, "confine": False})
        assert spec["el_override"][sp] == {0: "2p"}


def test_efg_anion_basis_per_species_helo_energy():
    """Opt-in structure-specific HELO energy: a {species: eV} mapping raises E₂ on the hard anion
    (rutile-class O) while leaving another anion at the robust 90 eV default (absent species fall
    back to 90). Measured recipe — experiments/autoapw/efg_eta_anion.md."""
    spec = efg_anion_basis(["O", "F"], helo_e={"O": 120.0})
    assert spec["los"]["O"][1] == (1, {"e": 120.0, "confine": False})   # hard site raised
    assert spec["los"]["F"][1] == (1, {"e": 90.0, "confine": False})    # absent -> 90 default


def test_efg_anion_basis_helo_only():
    """semicore=False drops the l=0 LO / el_override, leaving the decisive l=1 HELO."""
    spec = efg_anion_basis(["O"], semicore=False)
    assert spec["los"] == {"O": [(1, {"e": 90.0, "confine": False})]}
    assert "el_override" not in spec


def test_efg_anion_basis_custom_period():
    """A heavier anion (valence 3s/3p) via explicit labels."""
    spec = efg_anion_basis(["S"], s_label="3s", s_energy_label="3p")
    assert spec["los"]["S"][0] == (0, "3s")
    assert spec["el_override"]["S"] == {0: "3p"}


def test_efg_anion_basis_rejects_empty_and_duplicates():
    with pytest.raises(ValueError):
        efg_anion_basis([])
    with pytest.raises(ValueError):
        efg_anion_basis(["O", "O"])


def test_merge_basis_later_wins_per_species():
    recipe = efg_anion_basis(["O"])
    override = {"los": {"O": [(0, "2s")]}, "el_override": {"Ti": {2: "3d"}}}
    merged = merge_basis(recipe, override)
    assert merged["los"]["O"] == [(0, "2s")]                 # explicit override wins for O
    assert merged["el_override"]["O"] == {0: "2p"}           # untouched recipe entry survives
    assert merged["el_override"]["Ti"] == {2: "3d"}          # new species added


def test_merge_basis_skips_none():
    recipe = efg_anion_basis(["O"])
    assert merge_basis(None, recipe, None) == recipe


def test_flapwparams_efg_anion_basis_validation():
    """FlapwParams rejects an anion species with no muffin-tin radius / duplicates."""
    from gradwave.inputs.models import FlapwParams, InputError

    FlapwParams(radii={"O": 0.82, "Ti": 1.1}, efg_anion_basis=["O"])   # ok
    with pytest.raises(InputError):
        FlapwParams(radii={"Ti": 1.1}, efg_anion_basis=["O"])          # O has no radius
    with pytest.raises(InputError):
        FlapwParams(radii={"O": 0.82}, efg_anion_basis=["O", "O"])     # duplicate
