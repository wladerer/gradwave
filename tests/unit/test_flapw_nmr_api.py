"""FLAPW / NMR Input schema + api.run dispatch (fast tier).

Covers the leaf schema (round-trip parse of the all-electron flapw + nmr blocks,
without plane-wave pseudopotentials) and the full api.run wiring for the EFG and
shielding observables with a FAKED SCF, so the dispatch/summary/report layer is
exercised without paying for an SCF. The real rutile-TiO2 EFG end-to-end run
(reproducing a V_zz/C_Q through the actual FLAPW SCF) is the standard-tier
tests/integration/test_flapw_nmr_api_e2e.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gradwave import api
from gradwave.constants import BOHR_ANG
from gradwave.inputs import FlapwParams, Input, KPointsParams, NmrParams, load_input
from gradwave.inputs.models import InputError


def _rutile() -> Atoms:
    a_bohr = np.array([8.68083, 8.68083, 5.59096])
    cell = np.diag(a_bohr * BOHR_ANG)
    u = 0.3048
    frac = [[0, 0, 0], [0.5, 0.5, 0.5], [u, u, 0], [1 - u, 1 - u, 0],
            [0.5 + u, 0.5 - u, 0.5], [0.5 - u, 0.5 + u, 0.5]]
    return Atoms("Ti2O4", scaled_positions=frac, cell=cell, pbc=True)


# --------------------------------------------------------------------------
# schema / parse
# --------------------------------------------------------------------------
def _write_efg_yaml(tmp_path: Path) -> Path:
    yml = textwrap.dedent("""
    task: nmr
    structure:
      cell: [[4.5937, 0, 0], [0, 4.5937, 0], [0, 0, 2.9587]]
      positions:
        frac:
          - [0.0, 0.0, 0.0]
          - [0.5, 0.5, 0.5]
          - [0.3048, 0.3048, 0.0]
          - [0.6952, 0.6952, 0.0]
          - [0.8048, 0.1952, 0.5]
          - [0.1952, 0.8048, 0.5]
      species: [Ti, Ti, O, O, O, O]
    kpoints:
      mesh: [1, 1, 1]
    flapw:
      radii: {Ti: 0.95, O: 0.80}
      ecut: 150.0
      lmax: 2
      smearing: 0.0
      los: {Ti: [[0, "3s"], [1, "3p"]]}
      el_override: {Ti: {1: "3d"}}
    nmr:
      task: efg
      isotopes: {Ti: "49Ti", O: "17O"}
    """)
    p = tmp_path / "tio2_efg.yaml"
    p.write_text(yml)
    return p


def test_efg_yaml_roundtrip_all_electron(tmp_path):
    """An all-electron EFG input parses with NO pseudopotentials / ecut."""
    inp = load_input(_write_efg_yaml(tmp_path))
    assert inp.task == "nmr"
    assert inp.nmr.task == "efg"
    assert inp.nmr.isotopes == {"Ti": "49Ti", "O": "17O"}
    assert inp.flapw.radii == {"Ti": 0.95, "O": 0.80}
    assert inp.flapw.ecut == 150.0
    assert inp.flapw.los == {"Ti": [[0, "3s"], [1, "3p"]]}
    assert inp.flapw.el_override == {"Ti": {1: "3d"}}
    # no PW pseudos were supplied → empty map, placeholder ecut
    assert inp.pseudo_map == {}


def test_flapw_task_yaml_roundtrip(tmp_path):
    yml = textwrap.dedent("""
    task: flapw
    structure:
      cell: [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
      positions: {frac: [[0, 0, 0]]}
      species: [Ne]
    flapw:
      radii: {Ne: 0.74}
      ecut: 120.0
    """)
    p = tmp_path / "ne.yaml"
    p.write_text(yml)
    inp = load_input(p)
    assert inp.task == "flapw"
    assert inp.flapw.radii == {"Ne": 0.74}


def test_shielding_yaml_requires_pseudopotentials(tmp_path):
    """The plane-wave shielding observable still needs pseudopotentials + ecut."""
    yml = textwrap.dedent("""
    task: nmr
    structure:
      cell: [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
      positions: {frac: [[0, 0, 0]]}
      species: [Si]
    nmr:
      task: shielding
    """)
    p = tmp_path / "sh.yaml"
    p.write_text(yml)
    with pytest.raises(InputError, match="pseudopotentials"):
        load_input(p)


def test_flapw_radii_coverage_validated(tmp_path):
    yml = textwrap.dedent("""
    task: flapw
    structure:
      cell: [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
      positions: {frac: [[0, 0, 0]]}
      species: [Ne]
    flapw:
      radii: {Ar: 0.9}
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yml)
    with pytest.raises(InputError, match="missing a muffin-tin radius"):
        load_input(p)


def test_nmr_and_flapw_param_validation():
    with pytest.raises(InputError, match="nmr.task"):
        NmrParams(task="bogus")
    with pytest.raises(InputError, match="radii"):
        FlapwParams(radii={"O": -1.0})
    with pytest.raises(InputError, match="flapw.ecut"):
        FlapwParams(ecut=-1.0)


# --------------------------------------------------------------------------
# api.run dispatch with a faked FLAPW SCF (exercises dispatch/summary/report)
# --------------------------------------------------------------------------
class _FakeRecorder:
    def summarize(self):
        return {"n_iter": 7, "r_v": 1e-4, "r_nsph": None, "d_span_eV": 1e-4}


def _fake_efg_info(symbols):
    efg = {}
    for i, sym in enumerate(symbols):
        vzz = 19.0 if sym == "O" else -13.0
        efg[f"a{i}"] = {
            "V_zz": vzz, "eta": 0.5, "V_zz_valence": vzz * 0.9,
            "eta_valence": 0.4, "sphere_charge": 6.0,
            "tensor": np.diag([-vzz / 2, -vzz / 2, vzz]),
        }
    return {"efg": efg, "recorder": _FakeRecorder(), "e_fermi": None, "nbands": 20}


def _fake_crystal_scf_multi(*args, efg=False, **kwargs):
    # args: (cell, atoms_list, radii, ...) — recover symbols from atoms_list
    atoms_list = args[1]
    symbols = [sym for _frac, sym in atoms_list]
    bands = {"span": 5.0, "ev": [0.0, 1.0, 2.0], "e_fermi": None}
    info = _fake_efg_info(symbols) if efg else {
        "recorder": _FakeRecorder(), "e_fermi": None, "nbands": 20}
    return bands, info


def test_api_run_efg_dispatch(tmp_path, monkeypatch):
    """api.run(task=nmr, efg) routes to run_nmr, builds the summary and writes
    nmr.json/.out — with a faked SCF so it stays fast."""
    monkeypatch.setattr("gradwave.flapw.crystal_scf_multi", _fake_crystal_scf_multi)
    inp = Input(
        atoms=_rutile(), pseudo_dir=Path("."), pseudo_map={}, ecut=1.0, task="nmr",
        kpoints=KPointsParams(mesh=(1, 1, 1)),
        flapw=FlapwParams(radii={"Ti": 0.95, "O": 0.80}, ecut=150.0),
        nmr=NmrParams(task="efg", isotopes={"Ti": "49Ti", "O": "17O"}),
        output_dir=tmp_path, verbose=False)
    summary = api.run(inp, verbose=False)
    nmr = summary["nmr"]
    assert nmr["observable"] == "efg"
    assert nmr["n_sites"] == 6
    o_site = next(s for s in nmr["sites"] if s["species"] == "O")
    assert o_site["V_zz_eV_ang2"] == 19.0
    assert o_site["isotope"] == "17O"
    # C_Q wiring: C_Q = 2.4180 * Q * V_zz
    assert o_site["C_Q_MHz"] == pytest.approx(2.4180 * o_site["Q_barn"] * 19.0)
    # files written and the report renders the EFG table
    assert (tmp_path / "nmr.json").exists()
    report = (tmp_path / "nmr.out").read_text()
    assert "electric field gradient" in report
    assert "C_Q" in report


def test_api_run_efg_auto_isotopes(tmp_path, monkeypatch):
    """isotopes=None auto-selects a tabulated isotope per element."""
    monkeypatch.setattr("gradwave.flapw.crystal_scf_multi", _fake_crystal_scf_multi)
    inp = Input(
        atoms=_rutile(), pseudo_dir=Path("."), pseudo_map={}, ecut=1.0, task="nmr",
        kpoints=KPointsParams(mesh=(1, 1, 1)),
        flapw=FlapwParams(radii={"Ti": 0.95, "O": 0.80}),
        nmr=NmrParams(task="efg", isotopes=None),
        output_dir=tmp_path, verbose=False)
    nmr = api.run(inp, verbose=False)["nmr"]
    o_site = next(s for s in nmr["sites"] if s["species"] == "O")
    assert o_site["isotope"] == "17O"  # first tabulated O isotope


def test_api_run_flapw_task(tmp_path, monkeypatch):
    monkeypatch.setattr("gradwave.flapw.crystal_scf_multi", _fake_crystal_scf_multi)
    inp = Input(
        atoms=_rutile(), pseudo_dir=Path("."), pseudo_map={}, ecut=1.0, task="flapw",
        kpoints=KPointsParams(mesh=(1, 1, 1)),
        flapw=FlapwParams(radii={"Ti": 0.95, "O": 0.80}, ecut=150.0),
        output_dir=tmp_path, verbose=False)
    summary = api.run(inp, verbose=False)
    assert summary["task"] == "flapw"
    assert summary["flapw"]["eigenvalues_eV"] == [0.0, 1.0, 2.0]
    assert (tmp_path / "flapw.out").exists()


def test_api_run_shielding_dispatch(tmp_path, monkeypatch):
    """api.run(task=nmr, shielding) routes to the plane-wave GIPAW path with a
    faked SCF + sigma_shielding_dq."""
    import torch

    from gradwave.scf.loop import SCFResult

    def _fake_run_scf(inp, verbose=True, **kw):
        # a bare SCFResult instance (satisfies run_nmr's isinstance guard); its
        # fields are never read because sigma_shielding_dq is faked below
        return object.__new__(SCFResult)

    def _fake_sigma(res, **kw):
        # two sites, an axial shielding tensor each (iso + a zz excess)
        t = torch.zeros(2, 3, 3, dtype=torch.float64)
        for i in range(2):
            t[i] = torch.diag(torch.tensor([100.0, 100.0, 130.0], dtype=torch.float64))
        return t

    monkeypatch.setattr("gradwave.api.scf.run_scf", _fake_run_scf)
    monkeypatch.setattr("gradwave.postscf.kgeometry_nmr.sigma_shielding_dq", _fake_sigma)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=np.eye(3) * 5.0,
                  pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path("."), pseudo_map={"H": "H.upf"}, ecut=400.0,
        task="nmr", kpoints=KPointsParams(mesh=(2, 2, 2)),
        nmr=NmrParams(task="shielding"), output_dir=tmp_path, verbose=False)
    nmr = api.run(inp, verbose=False)["nmr"]
    assert nmr["observable"] == "shielding"
    assert nmr["method"] == "bare_analytic_dq"
    assert nmr["shielding_level"] == "bare"
    assert nmr["n_sites"] == 2
    s = nmr["sites"][0]
    assert s["sigma_iso_ppm"] == pytest.approx((100 + 100 + 130) / 3)
    assert s["sigma_aniso_ppm"] == pytest.approx(130 - 100)  # σ_zz − (σxx+σyy)/2
    assert "delta_iso_ppm" not in s  # no reference given → no shift emitted
    assert (tmp_path / "nmr.out").exists()


def _fake_bare_scf(monkeypatch):
    """Fake an NC SCFResult + a two-site axial bare shielding tensor."""
    import torch

    from gradwave.scf.loop import SCFResult

    def _fake_run_scf(inp, verbose=True, **kw):
        return object.__new__(SCFResult)

    def _fake_sigma(res, **kw):
        t = torch.zeros(2, 3, 3, dtype=torch.float64)
        for i in range(2):
            t[i] = torch.diag(torch.tensor([100.0, 100.0, 130.0], dtype=torch.float64))
        return t

    monkeypatch.setattr("gradwave.api.scf.run_scf", _fake_run_scf)
    monkeypatch.setattr(
        "gradwave.postscf.kgeometry_nmr.sigma_shielding_dq", _fake_sigma)


def test_shielding_reference_shift(tmp_path, monkeypatch):
    """nmr.sigma_ref keyed by species emits δ_iso = σ_ref − σ_iso per matching
    site (and only for matching sites)."""
    _fake_bare_scf(monkeypatch)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=np.eye(3) * 5.0,
                  pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path("."), pseudo_map={"H": "H.upf"}, ecut=400.0,
        task="nmr", kpoints=KPointsParams(mesh=(2, 2, 2)),
        nmr=NmrParams(task="shielding", sigma_ref={"H": 30.0}),
        output_dir=tmp_path, verbose=False)
    nmr = api.run(inp, verbose=False)["nmr"]
    iso = (100 + 100 + 130) / 3
    assert nmr["sigma_ref_ppm"] == {"H": 30.0}
    for s in nmr["sites"]:
        assert s["delta_iso_ppm"] == pytest.approx(30.0 - iso)


def test_reference_sigma_iso_helper(tmp_path, monkeypatch):
    """reference_sigma_iso runs the shielding task and averages σ_iso over the
    named species' sites."""
    _fake_bare_scf(monkeypatch)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=np.eye(3) * 5.0,
                  pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path("."), pseudo_map={"H": "H.upf"}, ecut=400.0,
        task="nmr", kpoints=KPointsParams(mesh=(2, 2, 2)),
        nmr=NmrParams(task="shielding"), output_dir=tmp_path, verbose=False)
    assert api.reference_sigma_iso(inp, "H") == pytest.approx((100 + 100 + 130) / 3)
    with pytest.raises(ValueError, match="no 'Li' site"):
        api.reference_sigma_iso(inp, "Li")


def test_shielding_gipaw_dispatch_and_breakdown(tmp_path, monkeypatch):
    """A PAW ground state auto-selects the absolute GIPAW assembly; the per-term
    breakdown (bare/core/dia_aug/para_aug) and total-tensor CSA are reported, and
    the shift references off the total σ_iso."""
    import torch

    from gradwave.scf.results import USPPResult

    def _fake_run_scf(inp, verbose=True, **kw):
        return object.__new__(USPPResult)

    def _fake_gipaw(res, ctx, paws, **kw):
        eye = torch.eye(3, dtype=torch.float64)
        bare = 40.0 * eye.expand(1, 3, 3).clone()
        core = 100.0 * eye.expand(1, 3, 3).clone()
        dia = 10.0 * eye.expand(1, 3, 3).clone()
        para = torch.diag(torch.tensor([5.0, 5.0, 35.0], dtype=torch.float64))[None]
        return {"total": bare + core + dia + para, "bare": bare, "core": core,
                "dia_aug": dia, "para_aug": para}

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

    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5.0, pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path("."), pseudo_map={"Si": "Si.paw.UPF"},
        ecut=400.0, task="nmr", kpoints=KPointsParams(mesh=(2, 2, 2)),
        nmr=NmrParams(task="shielding", sigma_ref={"Si": 337.0}),
        output_dir=tmp_path, verbose=False)
    nmr = api.run(inp, verbose=False)["nmr"]
    assert nmr["method"] == "gipaw_absolute"
    assert nmr["shielding_level"] == "gipaw"
    s = nmr["sites"][0]
    assert s["sigma_bare_ppm"] == pytest.approx(40.0)
    assert s["sigma_core_ppm"] == pytest.approx(100.0)
    assert s["sigma_dia_aug_ppm"] == pytest.approx(10.0)
    assert s["sigma_para_aug_ppm"] == pytest.approx((5 + 5 + 35) / 3)
    # total σ_iso = 40 + 100 + 10 + 15 = 165
    assert s["sigma_iso_ppm"] == pytest.approx(165.0)
    assert s["delta_iso_ppm"] == pytest.approx(337.0 - 165.0)


def test_shielding_level_yaml_roundtrip(tmp_path):
    """shielding_level and sigma_ref survive the YAML parse (task-schema check
    only; no SCF)."""
    pseudo_dir = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
    yml = textwrap.dedent(f"""
    task: nmr
    structure:
      cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
      positions: {{frac: [[0, 0, 0], [0.25, 0.25, 0.25]]}}
      species: [Si, Si]
    pseudopotentials:
      dir: {pseudo_dir}
      map: {{Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF}}
    ecut: 163.3
    kpoints: {{mesh: [2, 2, 2]}}
    nmr:
      task: shielding
      shielding_level: gipaw
      sigma_ref: {{Si: 337.4}}
    """)
    p = tmp_path / "si_shield.yaml"
    p.write_text(yml)
    inp = load_input(p)
    assert inp.nmr.shielding_level == "gipaw"
    assert inp.nmr.sigma_ref == {"Si": 337.4}
    p2 = tmp_path / "bad_level.yaml"
    p2.write_text(yml.replace("shielding_level: gipaw", "shielding_level: full"))
    with pytest.raises(InputError, match="shielding_level"):
        load_input(p2)


def test_shielding_level_gipaw_on_nc_rejected(tmp_path, monkeypatch):
    """An explicit shielding_level='gipaw' with a norm-conserving ground state
    is a clear error, not a silent bare fallback."""
    _fake_bare_scf(monkeypatch)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=np.eye(3) * 5.0,
                  pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path("."), pseudo_map={"H": "H.upf"}, ecut=400.0,
        task="nmr", kpoints=KPointsParams(mesh=(2, 2, 2)),
        nmr=NmrParams(task="shielding", shielding_level="gipaw"),
        output_dir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="needs an all-PAW ground state"):
        api.run(inp, verbose=False)
