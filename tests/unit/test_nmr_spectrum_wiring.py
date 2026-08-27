"""Final-wiring unit tests: NMR spectrum + PW/PAW EFG in the shielding driver.

Fast tier — no SCF. Covers the new ``nmr.efg`` / ``nmr.spectrum`` schema (YAML
round-trip + validation), the site → ``NMRSite`` assembly (δ_iso required, the
CSA sign/η carry-over, quadrupolar C_Q merge), the EFG gating, and the full
``api.run`` shielding path with a faked PAW SCF driving the EFG block and the
spectrum synthesis end to end (and through JSON serialization).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gradwave import api
from gradwave.api.flapw import (
    _assemble_nmr_sites,
    _resolve_nu0_hz,
    _resolve_pw_efg,
    _spectrum_block,
    _spectrum_kind,
)
from gradwave.inputs import NmrParams, NmrSpectrumParams, load_input
from gradwave.inputs.models import InputError


# --------------------------------------------------------------------------
# schema / parse round-trip + validation
# --------------------------------------------------------------------------
def _si_shield_yaml(extra_nmr_lines: str) -> str:
    """α-Si-like shielding input; ``extra_nmr_lines`` are appended under ``nmr:``
    (each already indented to the 6-space nmr-child column)."""
    pseudos = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
    head = textwrap.dedent(f"""
    task: nmr
    structure:
      cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
      positions: {{frac: [[0, 0, 0], [0.25, 0.25, 0.25]]}}
      species: [Si, Si]
    pseudopotentials:
      dir: {pseudos}
      map: {{Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF}}
    ecut: 163.3
    kpoints: {{mesh: [2, 2, 2]}}
    nmr:
      task: shielding
      shielding_level: gipaw
      sigma_ref: {{Si: 337.4}}
    """)
    return head + extra_nmr_lines


def test_efg_and_spectrum_yaml_roundtrip(tmp_path):
    """nmr.efg and the full nmr.spectrum block survive the YAML parse."""
    extra = (
        "  efg: auto\n"
        "  spectrum:\n"
        "    enabled: true\n"
        "    mode: mas\n"
        "    spin_rate_hz: 10000.0\n"
        "    larmor_mhz: 79.5\n"
        "    broadening_ppm: 1.5\n"
        "    lineshape: gauss\n"
        "    n_orientations: 800\n"
        "    n_points: 1024\n"
    )
    p = tmp_path / "si.yaml"
    p.write_text(_si_shield_yaml(extra))
    inp = load_input(p)
    assert inp.nmr.efg == "auto"
    spec = inp.nmr.spectrum
    assert spec.enabled and spec.mode == "mas"
    assert spec.spin_rate_hz == 10000.0
    assert spec.larmor_mhz == 79.5
    assert spec.broadening_ppm == 1.5
    assert spec.n_orientations == 800 and spec.n_points == 1024


def test_spectrum_larmor_map_roundtrip(tmp_path):
    """larmor_mhz accepts a per-isotope/species map."""
    extra = (
        "  spectrum:\n"
        "    enabled: true\n"
        '    larmor_mhz: {Si: 79.5, "29Si": 79.49}\n'
    )
    p = tmp_path / "si.yaml"
    p.write_text(_si_shield_yaml(extra))
    inp = load_input(p)
    assert inp.nmr.spectrum.larmor_mhz == {"Si": 79.5, "29Si": 79.49}


def test_efg_bool_roundtrip(tmp_path):
    p = tmp_path / "si.yaml"
    p.write_text(_si_shield_yaml("  efg: true\n"))
    assert load_input(p).nmr.efg is True


def test_nmr_efg_validation():
    with pytest.raises(InputError, match="nmr.efg"):
        NmrParams(task="shielding", efg="sometimes")


def test_spectrum_param_validation():
    with pytest.raises(InputError, match="mode"):
        NmrSpectrumParams(mode="fast")
    with pytest.raises(InputError, match="lineshape"):
        NmrSpectrumParams(lineshape="voigt")
    with pytest.raises(InputError, match="spin_rate_hz"):
        NmrSpectrumParams(enabled=True, mode="mas", spin_rate_hz=0.0)
    with pytest.raises(InputError, match="n_orientations"):
        NmrSpectrumParams(n_orientations=0)


def test_spectrum_defaults_off():
    """A default NmrParams carries a disabled spectrum and efg off (byte-identical)."""
    n = NmrParams(task="shielding")
    assert n.efg is False
    assert n.spectrum.enabled is False


# --------------------------------------------------------------------------
# EFG gating
# --------------------------------------------------------------------------
def test_resolve_pw_efg_gating():
    assert _resolve_pw_efg("auto", is_paw=True) is True
    assert _resolve_pw_efg("auto", is_paw=False) is False
    assert _resolve_pw_efg(True, is_paw=True) is True
    assert _resolve_pw_efg(False, is_paw=True) is False
    with pytest.raises(ValueError, match="all-PAW ground state"):
        _resolve_pw_efg(True, is_paw=False)


# --------------------------------------------------------------------------
# site -> NMRSite assembly
# --------------------------------------------------------------------------
def _shield_block(delta=True):
    site0 = {"site": 0, "species": "Si", "sigma_iso_ppm": 300.0,
             "sigma_aniso_ppm": 30.0, "sigma_eta": 0.4}
    site1 = {"site": 1, "species": "Si", "sigma_iso_ppm": 305.0,
             "sigma_aniso_ppm": 12.0, "sigma_eta": 0.1}
    if delta:
        site0["delta_iso_ppm"] = 37.4    # 337.4 - 300.0
        site1["delta_iso_ppm"] = 32.4
    return {"observable": "shielding", "sites": [site0, site1]}


def test_assemble_nmr_sites_csa_mapping():
    """δ_iso passed through; reduced shift anisotropy = -(2/3)·full σ anisotropy; η carried."""
    sites = _assemble_nmr_sites(_shield_block())
    assert len(sites) == 2
    s0 = sites[0]
    assert s0.delta_iso == pytest.approx(37.4)
    assert s0.delta_aniso == pytest.approx(-(2.0 / 3.0) * 30.0)
    assert s0.eta_csa == pytest.approx(0.4)
    assert s0.spin == 0.5 and s0.c_q == 0.0
    assert s0.label == "Si0"


def test_assemble_skips_unreferenced_sites():
    assert _assemble_nmr_sites(_shield_block(delta=False)) == []


def test_assemble_merges_quadrupolar_efg():
    """A quadrupolar EFG entry (spin, C_Q) merges into the matching site."""
    block = _shield_block()
    block["sites"][0]["species"] = "O"
    block["efg"] = {"method": "paw_petrilli_blochl", "sites": [
        {"site": 0, "species": "O", "V_zz_eV_ang2": 20.0, "eta": 0.3,
         "isotope": "17O", "C_Q_MHz": -1.23, "abs_C_Q_MHz": 1.23,
         "nu_Q_MHz": 0.1, "spin": 2.5, "Q_barn": -0.02558}]}
    sites = _assemble_nmr_sites(block)
    s0 = sites[0]
    assert s0.spin == 2.5
    assert s0.c_q == pytest.approx(-1.23e6)   # MHz -> Hz
    assert s0.eta_q == pytest.approx(0.3)


# --------------------------------------------------------------------------
# kind selection + Larmor resolution
# --------------------------------------------------------------------------
def test_spectrum_kind_selection():
    half = _assemble_nmr_sites(_shield_block())
    assert _spectrum_kind("static", half) == "csa_static"
    assert _spectrum_kind("mas", half) == "mas"

    block = _shield_block()
    block["efg"] = {"sites": [
        {"site": 0, "species": "Si", "isotope": "27Al", "C_Q_MHz": 1.0,
         "abs_C_Q_MHz": 1.0, "nu_Q_MHz": 0.1, "spin": 2.5, "eta": 0.2,
         "Q_barn": 0.1, "V_zz_eV_ang2": 5.0},
        {"site": 1, "species": "Si", "isotope": "27Al", "C_Q_MHz": 1.0,
         "abs_C_Q_MHz": 1.0, "nu_Q_MHz": 0.1, "spin": 2.5, "eta": 0.2,
         "Q_barn": 0.1, "V_zz_eV_ang2": 5.0}]}
    quad = _assemble_nmr_sites(block)
    assert _spectrum_kind("static", quad) == "quad2_ct_static"
    assert _spectrum_kind("mas", quad) == "quad2_ct_mas"


def test_spectrum_kind_rejects_mixed_spins():
    block = _shield_block()
    block["efg"] = {"sites": [
        {"site": 0, "species": "Si", "isotope": "27Al", "spin": 2.5,
         "C_Q_MHz": 1.0, "abs_C_Q_MHz": 1.0, "nu_Q_MHz": 0.1, "eta": 0.2,
         "Q_barn": 0.1, "V_zz_eV_ang2": 5.0}]}
    sites = _assemble_nmr_sites(block)  # site0 spin 2.5, site1 spin 0.5
    with pytest.raises(ValueError, match="mixed nuclear spins"):
        _spectrum_kind("static", sites)


def test_resolve_nu0_hz():
    assert _resolve_nu0_hz(None, ["Si"], []) is None
    assert _resolve_nu0_hz(79.5, ["Si"], []) == pytest.approx(79.5e6)
    assert _resolve_nu0_hz({"29Si": 79.49, "Si": 79.5}, ["Si"], ["29Si"]) == pytest.approx(79.49e6)
    assert _resolve_nu0_hz({"Si": 79.5}, ["Si"], []) == pytest.approx(79.5e6)
    with pytest.raises(ValueError, match="no entry"):
        _resolve_nu0_hz({"O": 54.2}, ["Si"], [])


# --------------------------------------------------------------------------
# spectrum block
# --------------------------------------------------------------------------
def test_spectrum_block_requires_referencing():
    cfg = NmrSpectrumParams(enabled=True, mode="static")
    with pytest.raises(ValueError, match="referenced sites"):
        _spectrum_block(cfg, _shield_block(delta=False))


def test_spectrum_block_static_csa():
    cfg = NmrSpectrumParams(enabled=True, mode="static", broadening_ppm=1.0,
                            n_orientations=400, n_points=512)
    out = _spectrum_block(cfg, _shield_block())
    assert out["kind"] == "csa_static"
    assert out["nucleus"] == ["Si"]
    assert len(out["ppm_axis"]) == 512 == len(out["intensity"])
    assert out["n_sites"] == 2
    # unit-area spectrum has a finite peak within the axis range
    lo, hi = out["ppm_range"]
    assert lo <= out["peak_ppm"] <= hi


def test_spectrum_block_mas_needs_larmor():
    cfg = NmrSpectrumParams(enabled=True, mode="mas", spin_rate_hz=1e4)
    with pytest.raises(ValueError, match="larmor_mhz"):
        _spectrum_block(cfg, _shield_block())


# --------------------------------------------------------------------------
# api.run end to end (faked PAW SCF) — EFG block + spectrum + JSON serialize
# --------------------------------------------------------------------------
def test_api_run_shielding_efg_and_spectrum(tmp_path, monkeypatch):
    import torch

    from gradwave.scf.results import USPPResult

    def _fake_run_scf(inp, verbose=True, **kw):
        return object.__new__(USPPResult)

    def _fake_gipaw(res, ctx, paws, **kw):
        # one axial Si shielding tensor (iso 300, zz excess)
        t = torch.diag(torch.tensor([290.0, 290.0, 320.0], dtype=torch.float64))[None]
        z = torch.zeros(1, 3, 3, dtype=torch.float64)
        return {"total": t, "bare": t, "core": z, "dia_aug": z, "para_aug": z}

    def _fake_efg_paw(res, *, isotopes=None):
        return [{"element": "Si", "site": 0,
                 "V": torch.diag(torch.tensor([-2.0, -2.0, 4.0], dtype=torch.float64)),
                 "V_zz": 4.0, "eta": 0.0}]

    monkeypatch.setattr("gradwave.api.scf.run_scf", _fake_run_scf)
    monkeypatch.setattr("gradwave.api.system._species_upfs",
                        lambda inp: (["Si"], [object()], [0]))
    monkeypatch.setattr("gradwave.api.system._as_paws", lambda upfs: list(upfs))
    monkeypatch.setattr("gradwave.api._common.build_xc", lambda inp: None)
    monkeypatch.setattr(
        "gradwave.postscf.kgeometry_nmr.build_uspp_response_ctx",
        lambda res, xc: object())
    monkeypatch.setattr(
        "gradwave.postscf.kgeometry_nmr.sigma_shielding_gipaw", _fake_gipaw)
    monkeypatch.setattr("gradwave.postscf.efg_paw.efg_paw", _fake_efg_paw)

    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5.0, pbc=True)
    from gradwave.inputs import Input, KPointsParams

    inp = Input(
        atoms=atoms, pseudo_dir=Path("."), pseudo_map={"Si": "Si.paw.UPF"},
        ecut=400.0, task="nmr", kpoints=KPointsParams(mesh=(2, 2, 2)),
        nmr=NmrParams(task="shielding", sigma_ref={"Si": 337.0}, efg="auto",
                      spectrum=NmrSpectrumParams(
                          enabled=True, mode="mas", spin_rate_hz=1.0e4,
                          larmor_mhz=79.5, broadening_ppm=2.0,
                          n_orientations=200, n_points=512)),
        output_dir=tmp_path, verbose=False)
    nmr = api.run(inp, verbose=False)["nmr"]

    # EFG block present (auto -> on for PAW)
    assert nmr["efg"]["method"] == "paw_petrilli_blochl"
    assert nmr["efg"]["sites"][0]["V_zz_eV_ang2"] == 4.0
    # spectrum synthesized
    spec = nmr["spectrum"]
    assert spec["mode"] == "mas" and spec["kind"] == "mas"
    assert spec["larmor_mhz"] == pytest.approx(79.5)
    assert len(spec["ppm_axis"]) == 512
    # JSON was written by api.run (would raise on a stray tensor)
    assert (tmp_path / "nmr.json").exists()
    report = (tmp_path / "nmr.out").read_text()
    assert "simulated spectrum" in report
    assert "electric field gradient (plane-wave PAW)" in report
