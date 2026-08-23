"""`gradwave run` CLI rendering for the FLAPW EFG / flapw tasks (fast tier).

Drives the CLI end-to-end on a tiny rutile-TiO2 EFG input with the FLAPW SCF
FAKED (monkeypatched ``crystal_scf_multi``, mirroring
tests/unit/test_flapw_nmr_api.py), so the parse → dispatch → per-site EFG render
path is exercised without paying for a muffin-tin SCF. The real SCF end-to-end is
the standard-tier tests/integration/test_flapw_nmr_api_e2e.py.
"""

from __future__ import annotations

import textwrap

import numpy as np

from gradwave import cli


class _FakeRecorder:
    def summarize(self):
        # d_span 1e-4 < FlapwParams.tol default 3e-3 → the run reports converged
        return {"n_iter": 7, "r_v": 1e-4, "r_nsph": None, "d_span_eV": 1e-4}


def _fake_crystal_scf_multi(*args, efg=False, **kwargs):
    atoms_list = args[1]
    symbols = [sym for _frac, sym in atoms_list]
    bands = {"span": 5.0, "ev": [0.0, 1.0, 2.0], "e_fermi": None}
    if not efg:
        return bands, {"recorder": _FakeRecorder(), "e_fermi": None, "nbands": 20}
    efg_d = {}
    for i, sym in enumerate(symbols):
        vzz = 19.0 if sym == "O" else -13.0
        efg_d[f"a{i}"] = {
            "V_zz": vzz, "eta": 0.5, "V_zz_valence": vzz * 0.9,
            "eta_valence": 0.4, "sphere_charge": 6.0,
            "tensor": np.diag([-vzz / 2, -vzz / 2, vzz]),
        }
    return bands, {"efg": efg_d, "recorder": _FakeRecorder(),
                   "e_fermi": None, "nbands": 20}


_EFG_YAML = textwrap.dedent("""
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
    nmr:
      task: efg
      isotopes: {Ti: "49Ti", O: "17O"}
""")


def test_cli_run_nmr_efg_renders_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("gradwave.flapw.crystal_scf_multi", _fake_crystal_scf_multi)
    yml = tmp_path / "tio2_efg.yaml"
    yml.write_text(_EFG_YAML)

    rc = cli.main(["run", str(yml), "-o", str(tmp_path / "out")])
    assert rc == 0

    out = capsys.readouterr().out
    # the startup summary uses the FLAPW-aware block, not the PW ecut/pseudos one
    assert "flapw ecut" in out
    assert "task        nmr  (efg)" in out
    # the per-site EFG render
    assert "electric field gradient, 6 sites" in out
    assert "V_zz" in out
    assert "C_Q" in out
    assert "49Ti" in out and "17O" in out
    # the machine-readable summary + report also land on disk
    assert (tmp_path / "out" / "nmr.json").exists()
    assert "electric field gradient" in (tmp_path / "out" / "nmr.out").read_text()


def test_cli_run_flapw_task_renders_convergence(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("gradwave.flapw.crystal_scf_multi", _fake_crystal_scf_multi)
    yml = tmp_path / "ne.yaml"
    yml.write_text(textwrap.dedent("""
        task: flapw
        structure:
          cell: [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
          positions: {frac: [[0, 0, 0]]}
          species: [Ne]
        flapw:
          radii: {Ne: 0.74}
          ecut: 120.0
    """))
    rc = cli.main(["run", str(yml), "-o", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "FLAPW SCF" in out
    assert "converged" in out
    assert (tmp_path / "out" / "flapw.json").exists()
