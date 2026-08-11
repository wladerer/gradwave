"""Layer-C IO: checkpoint round-trip/restart, output files, CLI, analysis.

Fast tests run on a canned summary (no SCF); the standard-tier tests run
real small SCFs — a PAW checkpoint restart and a CLI end-to-end on an
NC config.
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
SI_CELL, SI_POS = si_fcc()


def _canned_summary():
    rng = np.random.default_rng(5)
    eig = np.sort(rng.normal(size=(3, 8)), axis=1) * 3.0
    occ = np.zeros((3, 8))
    occ[:, :4] = 2.0
    return {
        "code": {"name": "gradwave", "version": "0.1.0",
                 "created": "2026-07-15T12:00:00"},
        "task": "scf",
        "structure": {"cell_ang": SI_CELL.tolist(),
                      "positions_ang": SI_POS.tolist(),
                      "species": ["Si", "Si"], "n_atoms": 2,
                      "volume_ang3": 40.03, "density_g_cm3": 2.330,
                      "spacegroup": "Fd-3m (227)", "pointgroup": "m-3m",
                      "n_symops": 48},
        "parameters": {"formalism": "uspp/paw", "xc": "pbe",
                       "ecut_eV": 204.0, "ecutrho_eV": 816.0,
                       "kmesh": [2, 2, 2], "nk": 3, "nk_total": 8,
                       "kweights": [0.25, 0.5, 0.25], "nspin": 1,
                       "smearing": "none", "width_eV": 0.1,
                       "symmetry": True,
                       "mixing": {"scheme": "johnson", "alpha": 0.7,
                                  "history": 12, "kerker": "auto",
                                  "kerker_used": True},
                       "pseudos": {"Si": "Si.kjpaw.UPF"}},
        "scf": {"converged": True, "n_iter": 3, "fermi_eV": None,
                "gap_eV": 0.6,
                "energies_eV": {"kinetic": 100.0, "hartree": 20.0,
                                "xc": -30.0, "local": -90.0,
                                "nonlocal": 10.0, "ewald": -250.0,
                                "smearing": 0.0, "hubbard": 0.0,
                                "onecenter": -5.0, "total": -245.0,
                                "free_energy": -245.0, "e0": -245.0},
                "trace": [
                    {"iter": 1, "free_energy_eV": -244.0, "dE_eV": None,
                     "drho": 1e-1},
                    {"iter": 2, "free_energy_eV": -244.9, "dE_eV": -0.9,
                     "drho": 1e-3},
                    {"iter": 3, "free_energy_eV": -245.0, "dE_eV": -0.1,
                     "drho": 1e-8},
                ],
                "free_energy_per_atom_eV": -122.5,
                "convergence": {"final_dE_eV": -1e-9, "final_drho": 5e-8,
                                "etol_eV": 1e-8, "rhotol": 1e-7,
                                "ratio_q": 0.30, "warm_started": False}},
        "error_estimate": {"available": True, "ecut_eV": 204.0,
                           "ecut_large_eV": 510.0, "denergy_eV": -0.03,
                           "denergy_meV_per_atom": -15.0,
                           "free_energy_extrapolated_eV": -245.03,
                           "drho_L1_per_electron": 1e-3, "int_drho": 0.0,
                           "note": "first-order estimate",
                           "numerical_energy_error": {
                               "total_eV": 0.03, "total_meV_per_atom": 15.0,
                               "terms_eV": {"ecut": 0.03},
                               "note": "leading-order sum of the reachable terms"}},
        "eigenvalues_eV": eig.tolist(),
        "occupations": occ.tolist(),
        "runtime_s": 12.3,
        "outputs": {"json": "scf.json", "report": "scf.out"},
    }


def test_human_report_from_summary():
    from gradwave.io.output import format_output

    text = format_output(_canned_summary())
    for token in ("gradwave 0.1.0", "── structure", "── parameters",
                  "── self-consistency", "converged in 3 iterations",
                  "free energy F", "-245.0000000000", "gap 0.6000 eV",
                  "── eigenvalues",
                  # enriched fields
                  "density 2.330 g/cm³", "point group m-3m",
                  "48 symmetry operations", "8 k (3 IBZ)",
                  "F / atom", "-122.5000000000", "numerical total",
                  # mixing + convergence diagnostics
                  "mixing", "johnson", "Kerker on (auto)",
                  "ratio q ≈ 0.30", "etol 1e-08"):
        assert token in text, token


def _canned_magnetism_summary():
    return {
        "code": {"name": "gradwave", "version": "0.1.0",
                 "created": "2026-07-17T12:00:00"},
        "task": "magnetism",
        "structure": {"cell_ang": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
                      "positions_ang": [[3, 3, 2.4], [3, 3, 3.6]],
                      "species": ["O", "O"]},
        "parameters": {"formalism": "nc", "xc": "lda", "ecut_eV": 408.0,
                       "ecutrho_eV": None, "kmesh": [1, 1, 1], "nk": 1,
                       "kweights": [1.0], "nspin": 1, "smearing": "gaussian",
                       "width_eV": 0.1, "symmetry": False,
                       "pseudos": {"O": "O.upf"}},
        "magnetism": {"ordering": "ferromagnetic", "total_moment_muB": 1.999,
                      "atomic_moments_muB": [1.0, 1.0],
                      "moment_vectors_muB": [[0, 0, 1.0], [0, 0, 1.0]],
                      "exchange_J_meV": {"1": 1434.0}, "dmi_meV": {"1": 0.0},
                      "curie_temperature_mfa_K": 11094},
        "runtime_s": 300.0,
        "outputs": {"json": "magnetism.json", "report": "magnetism.out"},
    }


def test_magnetism_report_from_summary():
    from gradwave.io.output import format_output

    text = format_output(_canned_magnetism_summary())
    for token in ("── magnetism", "ordering: ferromagnetic",
                  "atomic moments [μB]: 1.000, 1.000", "Heisenberg exchange",
                  "J_1 = +1434", "mean-field T_c"):
        assert token in text, token


def test_load_input_magnetism_block(tmp_path):
    from gradwave.inputs import load_input

    (tmp_path / "in.yaml").write_text(f"""
structure:
  cell: [[6, 0, 0], [0, 6, 0], [0, 0, 6]]
  positions: {{cart: [[3, 3, 2.4], [3, 3, 3.6]]}}
  species: [O, O]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{O: O_ONCV_PBE-1.2.upf}}
ecut: 408.17
task: magnetism
magnetism: {{exchange: false, lam: 6.0, ref_atom: 1}}
""")
    inp = load_input(tmp_path / "in.yaml")
    assert inp.task == "magnetism"
    assert inp.magnetism.exchange is False
    assert inp.magnetism.lam == 6.0 and inp.magnetism.ref_atom == 1


def test_analysis_frames_and_plots(tmp_path):
    # analysis pulls in pandas/matplotlib (the `analysis` optional extra); skip
    # rather than fail the fast gate when they are not installed.
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from gradwave.io import analysis

    s = _canned_summary()
    df = analysis.scf_frame(s)
    assert list(df["iter"]) == [1, 2, 3]
    assert df["dF_from_final_eV"].iloc[-1] == 0.0

    ev = analysis.eigenvalues_frame(s)
    assert len(ev) == 3 * 8
    assert set(ev.columns) >= {"spin", "k", "band", "energy_eV",
                               "occupation"}

    dos = analysis.dos_frame(s, width=0.2)
    # ∫DOS dE recovers the electron count (occupied states × g_spin ×
    # weights sum): 4 bands × 2 = 8 electrons of the 16 states total
    de = dos["energy_eV"].iloc[1] - dos["energy_eV"].iloc[0]
    total_states = float(dos["dos"].sum() * de)
    assert abs(total_states - 16.0) < 0.1

    analysis.plot_scf(s, path=tmp_path / "scf.png")
    analysis.plot_dos(s, path=tmp_path / "dos.png")
    assert (tmp_path / "scf.png").exists()
    assert (tmp_path / "dos.png").exists()


def _canned_eos() -> dict:
    """An eos summary built from a real BM3 fit of synthetic points, so the
    fitted-curve column reproduces the input energies at the fit tolerance."""
    from gradwave.postscf.eos import EV_A3_TO_GPA, birch_murnaghan, fit_bm3

    v0, b0, b0p, e0 = 20.0, 0.55, 4.2, -108.0  # eV/Å³ for b0
    scales = [0.94, 0.97, 1.0, 1.03, 1.06, 1.09]
    v = v0 * np.array(scales)
    e = birch_murnaghan(v, e0, v0, b0, b0p)
    fit = fit_bm3(v, e)
    return {
        "task": "eos",
        "eos": {"scales": scales, "energy_kind": "free_energy", "n_atoms": 2,
                "volumes_ang3_per_atom": v.tolist(),
                "energies_eV_per_atom": e.tolist(), "fft_grid": [24, 24, 24],
                "v0_ang3_per_atom": fit.v0, "b0_GPa": fit.b0_GPa,
                "b0_prime": fit.b0_prime, "e0_eV_per_atom": fit.e0,
                "rms_residual_eV_per_atom": fit.rms_residual_eV,
                "b0_eV_ang3": fit.b0, "ev_a3_to_gpa": EV_A3_TO_GPA,
                "all_converged": True},
    }


def _canned_elastic() -> dict:
    c = [[166.0, 64.0, 64.0, 0.0, 0.0, 0.0],
         [64.0, 166.0, 64.0, 0.0, 0.0, 0.0],
         [64.0, 64.0, 166.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 80.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 80.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 80.0]]
    return {
        "task": "elastic",
        "elastic": {"strain": 0.005, "mode": "clamped", "n_atoms": 2,
                    "formalism": "nc", "c_GPa": c,
                    "bulk_modulus_GPa": {"voigt": 98.0, "reuss": 98.0,
                                         "hill": 98.0},
                    "shear_modulus_GPa": {"voigt": 62.0, "reuss": 60.0,
                                          "hill": 61.0},
                    "young_modulus_GPa": 150.0, "poisson_ratio": 0.22,
                    "mechanically_stable": True, "residual_stress_GPa": 0.01,
                    "all_converged": True},
    }


def test_eos_elastic_frames_and_plots(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from gradwave.io import analysis

    eos = _canned_eos()
    ef = analysis.eos_frame(eos)
    assert list(ef.columns) == ["scale", "volume_ang3_per_atom",
                                "energy_eV_per_atom", "bm3_eV_per_atom"]
    assert len(ef) == 6
    assert ef.attrs["b0_GPa"] > 0
    # the fitted curve reproduces the (synthetic BM3) energies at the fit tol
    assert (ef["bm3_eV_per_atom"] - ef["energy_eV_per_atom"]).abs().max() < 1e-6

    elastic = _canned_elastic()
    cf = analysis.elastic_frame(elastic)
    assert cf.shape == (6, 6)
    assert list(cf.index) == [1, 2, 3, 4, 5, 6]
    assert list(cf.columns) == [1, 2, 3, 4, 5, 6]
    assert cf.loc[1, 1] == pytest.approx(166.0)
    assert cf.attrs["mechanically_stable"] is True
    assert cf.attrs["bulk_modulus_GPa"]["hill"] == pytest.approx(98.0)

    analysis.plot_eos(eos, path=tmp_path / "eos.png")
    analysis.plot_elastic(elastic, path=tmp_path / "elastic.png")
    assert (tmp_path / "eos.png").exists()
    assert (tmp_path / "elastic.png").exists()


def test_cli_plot_autodetects_eos_and_elastic(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from gradwave.cli import main

    (tmp_path / "eos.json").write_text(json.dumps(_canned_eos()))
    assert main(["plot", str(tmp_path / "eos.json")]) == 0
    assert (tmp_path / "eos.eos.png").exists()

    (tmp_path / "elastic.json").write_text(json.dumps(_canned_elastic()))
    assert main(["plot", str(tmp_path / "elastic.json")]) == 0
    assert (tmp_path / "elastic.elastic.png").exists()


def test_cli_restart_flag_overrides_yaml(tmp_path, monkeypatch):
    """`gradwave run -r PATH` sets Input.restart, overriding the YAML restart:
    key. Parse-only: api.run is stubbed so nothing computes."""
    import gradwave.api as api
    from gradwave.cli import main

    captured: dict = {}

    def fake_run(inp, verbose=True):
        captured["inp"] = inp
        return {}

    monkeypatch.setattr(api, "run", fake_run)
    (tmp_path / "in.yaml").write_text(f"""
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {15 * RY}
kpoints: {{mesh: [2, 2, 2]}}
restart: {tmp_path / "from_yaml.pt"}
""")
    cli_ck = tmp_path / "from_cli.pt"
    assert main([str(tmp_path / "in.yaml"), "-r", str(cli_ck), "-q"]) == 0
    assert captured["inp"].restart == cli_ck  # CLI beats the YAML restart:

    # without -r the YAML restart: is honored
    captured.clear()
    assert main([str(tmp_path / "in.yaml"), "-q"]) == 0
    assert captured["inp"].restart == tmp_path / "from_yaml.pt"


def test_cli_run_prints_banner_and_summary(tmp_path, monkeypatch, capsys):
    """A bare `gradwave run` prints the wave banner + input summary + a
    'preparing' notice before the (silent) build; `-q` suppresses all of it.
    Parse-only: api.run is stubbed so nothing computes."""
    import gradwave.api as api
    from gradwave.cli import main

    def fake_run(inp, verbose=True):
        return {}

    monkeypatch.setattr(api, "run", fake_run)
    (tmp_path / "in.yaml").write_text(f"""
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {15 * RY}
kpoints: {{mesh: [2, 2, 2]}}
""")
    assert main([str(tmp_path / "in.yaml")]) == 0
    out = capsys.readouterr().out
    assert "gradwave" in out                       # banner wordmark
    assert "differentiable plane-wave DFT" in out  # tagline
    assert "preparing system" in out               # the pre-build notice
    assert "Si2" in out                            # structure summary line

    # -q suppresses the banner/summary/preparing block entirely
    assert main([str(tmp_path / "in.yaml"), "-q"]) == 0
    out_q = capsys.readouterr().out
    assert "gradwave" not in out_q
    assert "preparing system" not in out_q


@pytest.mark.standard
def test_paw_checkpoint_roundtrip_and_restart(tmp_path):
    from gradwave.core.xc.pbe import PBE
    from gradwave.io.checkpoint import (
        as_start_from,
        load_checkpoint,
        save_checkpoint,
    )
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")

    def build():
        return setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=15 * RY,
                          kmesh=(2, 2, 2), ecutrho=60 * RY)

    res = scf_uspp(build(), PBE(), etol=1e-9, rhotol=1e-8, verbose=False,
                   max_iter=60)
    assert res["converged"]
    f_ref = float(res["energies"].free_energy)

    ck = tmp_path / "checkpoint.pt"
    save_checkpoint(res, ck)  # default: no wavefunctions
    payload = load_checkpoint(ck)
    assert payload["kind"] == "uspp"
    assert "coeffs" not in payload
    assert abs(payload["energies_eV"]["free_energy"] - f_ref) < 1e-10

    # wavefunctions on request, bit-identical
    ck_wf = tmp_path / "checkpoint_wf.pt"
    save_checkpoint(res, ck_wf, wavefunctions=True)
    wf = load_checkpoint(ck_wf)["coeffs"]
    assert torch.equal(wf[0], res["coeffs"][0].cpu())
    assert ck_wf.stat().st_size > 2 * ck.stat().st_size

    # restart: same free energy, far fewer iterations
    res2 = scf_uspp(build(), PBE(), etol=1e-9, rhotol=1e-8, verbose=False,
                    max_iter=60, start_from=as_start_from(payload))
    assert res2["converged"]
    assert abs(float(res2["energies"].free_energy) - f_ref) < 1e-6
    assert res2["n_iter"] < res["n_iter"]


@pytest.mark.slow
def test_uspp_noncollinear_checkpoint_roundtrip_and_restart(tmp_path):
    """Checkpoint round trip + restart for scf_uspp_noncollinear (USPP/PAW
    spinor): both halves of its state — the magnetization field m⃗ and the
    4-channel (n, mx, my, mz) becsum — must archive and restart together,
    mirroring test_paw_checkpoint_roundtrip_and_restart's collinear USPP
    pattern above."""
    from gradwave.core.xc.spin import LSDA_PW92
    from gradwave.io.checkpoint import (
        as_start_from,
        load_checkpoint,
        save_checkpoint,
    )
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import setup_uspp
    from gradwave.scf.uspp_noncollinear import scf_uspp_noncollinear

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")

    def build():
        return setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=20 * RY,
                          kmesh=(2, 2, 2), nbands=8, use_symmetry=False)

    mag_seed = [[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]]
    res = scf_uspp_noncollinear(build(), LSDA_PW92(), mag_seed,
                                smearing="gaussian", width=0.05,
                                etol=1e-8, rhotol=1e-7, verbose=False)
    assert res.converged
    f_ref = float(res.energies.free_energy)

    ck = tmp_path / "checkpoint.pt"
    save_checkpoint(res, ck)  # default: no wavefunctions
    payload = load_checkpoint(ck)
    assert payload["kind"] == "uspp_noncollinear"
    assert "coeffs" not in payload
    assert abs(payload["energies_eV"]["free_energy"] - f_ref) < 1e-10

    # the (ρ, m⃗, becsum) state round-trips bit-for-bit through save/load
    assert torch.equal(payload["rho"], res.rho.cpu())
    assert torch.equal(payload["m"], res.m.cpu())
    for c4 in range(4):
        for a in range(len(res.rho_ij_chan[c4])):
            assert torch.equal(payload["rho_ij_chan"][c4][a],
                               res.rho_ij_chan[c4][a].cpu())
    assert payload["mag_vec"] == [pytest.approx(x) for x in res.mag_vec]

    # restart: reproduces the same converged free energy, far fewer iterations
    res2 = scf_uspp_noncollinear(build(), LSDA_PW92(), mag_seed,
                                 smearing="gaussian", width=0.05,
                                 etol=1e-8, rhotol=1e-7, verbose=False,
                                 start_from=as_start_from(payload))
    assert res2.converged
    assert abs(float(res2.energies.free_energy) - f_ref) < 1e-6
    assert res2.n_iter < res.n_iter


@pytest.mark.standard
def test_cli_end_to_end_nc(tmp_path):
    from gradwave.cli import main

    (tmp_path / "input.yaml").write_text(f"""
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {15 * RY}
xc: pbe
kpoints: {{mesh: [2, 2, 2]}}
scf: {{etol: 1.0e-8, rhotol: 1.0e-7}}
""")
    out = tmp_path / "results"
    rc = main([str(tmp_path / "input.yaml"), "-o", str(out), "-q"])
    assert rc == 0

    summary = json.loads((out / "scf.json").read_text())
    assert summary["scf"]["converged"]
    assert summary["parameters"]["formalism"] == "nc"
    assert summary["scf"]["trace"], "per-iteration trace missing"
    assert summary["scf"]["gap_eV"] and 0.3 < summary["scf"]["gap_eV"] < 1.5

    report = (out / "scf.out").read_text()
    assert "self-consistency" in report and "free energy F" in report
    assert (out / "checkpoint.pt").exists()

    # the plot subcommand consumes the JSON it just wrote; it routes through
    # gradwave.io.analysis, which needs the `analysis` optional extra
    # (pandas/matplotlib). Guard just this block so the end-to-end SCF and the
    # restart coverage below still run on a core-only install.
    if all(importlib.util.find_spec(m) for m in ("pandas", "matplotlib")):
        rc = main(["plot", str(out / "scf.json"), "-o",
                   str(tmp_path / "conv.png")])
        assert rc == 0 and (tmp_path / "conv.png").exists()

    # NC restart from the checkpoint: same F, fewer iterations
    (tmp_path / "input2.yaml").write_text(
        (tmp_path / "input.yaml").read_text()
        + f"restart: {out / 'checkpoint.pt'}\n")
    out2 = tmp_path / "results2"
    assert main([str(tmp_path / "input2.yaml"), "-o", str(out2), "-q"]) == 0
    s2 = json.loads((out2 / "scf.json").read_text())
    assert s2["scf"]["n_iter"] < summary["scf"]["n_iter"]
    assert abs(s2["scf"]["energies_eV"]["free_energy"]
               - summary["scf"]["energies_eV"]["free_energy"]) < 1e-6

    # the same restart via the -r CLI flag (no restart: in the YAML): -r must
    # override the (absent) YAML key and warm-start identically.
    out3 = tmp_path / "results3"
    assert main([str(tmp_path / "input.yaml"), "-o", str(out3),
                 "-r", str(out / "checkpoint.pt"), "-q"]) == 0
    s3 = json.loads((out3 / "scf.json").read_text())
    assert s3["scf"]["n_iter"] < summary["scf"]["n_iter"]
    assert abs(s3["scf"]["energies_eV"]["free_energy"]
               - summary["scf"]["energies_eV"]["free_energy"]) < 1e-6


@pytest.mark.standard
def test_cli_noncollinear_restart_via_nc_mag_seed(tmp_path):
    """A restart pointing at a *noncollinear* (spinor) checkpoint warm-starts
    the spinor SCF instead of raising. The checkpoint kind is "noncollinear",
    so api.run_scf routes it through checkpoint.nc_mag_seed (which decomposes
    the stored m⃗ field back onto per-atom moments) rather than as_start_from
    (which rejects a noncollinear kind). Both the YAML restart: key and the -r
    CLI flag drive it."""
    from gradwave.cli import main

    def write(name: str, extra: str = "") -> None:
        (tmp_path / name).write_text(f"""
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {15 * RY}
xc: pbe
noncollinear: true
start_mag: {{Si: 0.5}}
smearing: {{type: gaussian, width: 0.1}}
kpoints: {{mesh: [2, 2, 2]}}
scf: {{etol: 1.0e-7, rhotol: 1.0e-6}}
{extra}""")

    torch.set_num_threads(8)
    write("input.yaml")
    out = tmp_path / "run1"
    assert main([str(tmp_path / "input.yaml"), "-o", str(out), "-q"]) == 0
    s1 = json.loads((out / "scf.json").read_text())
    assert s1["parameters"]["formalism"] == "noncollinear"
    ck = out / "checkpoint.pt"
    assert ck.exists()
    from gradwave.io.checkpoint import load_checkpoint
    assert load_checkpoint(ck)["kind"] == "noncollinear"
    f1 = s1["scf"]["energies_eV"]["free_energy"]

    # YAML restart: routes through nc_mag_seed (does not raise), same energy
    write("input2.yaml", extra=f"restart: {ck}\n")
    out2 = tmp_path / "run2"
    assert main([str(tmp_path / "input2.yaml"), "-o", str(out2), "-q"]) == 0
    s2 = json.loads((out2 / "scf.json").read_text())
    assert abs(s2["scf"]["energies_eV"]["free_energy"] - f1) < 1e-6

    # the -r CLI flag on the plain input drives the identical NC restart path
    out3 = tmp_path / "run3"
    assert main([str(tmp_path / "input.yaml"), "-o", str(out3),
                 "-r", str(ck), "-q"]) == 0
    s3 = json.loads((out3 / "scf.json").read_text())
    assert abs(s3["scf"]["energies_eV"]["free_energy"] - f1) < 1e-6


@pytest.mark.standard
def test_relax_writes_extxyz_trajectory(tmp_path):
    """A relax task writes relax.xyz next to the JSON, one frame per step with
    energy and forces, re-readable by ASE."""
    from ase.io import read as ase_read

    from gradwave.cli import main

    rng = np.random.default_rng(1)
    pos = (SI_POS + rng.normal(0, 0.05, SI_POS.shape)).tolist()
    (tmp_path / "relax.yaml").write_text(f"""
task: relax
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {pos}
  species: [Si, Si]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {15 * RY}
xc: lda
kpoints: {{mesh: [2, 2, 2]}}
relax: {{optimizer: fire, fmax: 0.02, max_steps: 3}}
""")
    out = tmp_path / "results"
    assert main([str(tmp_path / "relax.yaml"), "-o", str(out), "-q"]) == 0

    summary = json.loads((out / "relax.json").read_text())
    assert summary["outputs"]["trajectory"] == "relax.xyz"
    xyz = out / "relax.xyz"
    assert xyz.exists()

    frames = ase_read(str(xyz), index=":")
    assert len(frames) == len(summary["relax"]["trajectory"])
    # energy and forces survive the extxyz round trip and match the JSON trace
    e_json = summary["relax"]["trajectory"][-1]["energy_eV"]
    assert abs(frames[-1].get_potential_energy() - e_json) < 1e-6
    assert frames[-1].get_forces().shape == (2, 3)


@pytest.mark.standard
def test_cli_relax_streams_scf_trace_vasp_style(tmp_path, capsys):
    """A relax streams each ionic step's SCF trace under a `── ionic step N ──`
    header (VASP OSZICAR-style) by default; `-q` silences the stream. The header
    and its summary line carry the same step number."""
    from gradwave.cli import main

    (tmp_path / "relax.yaml").write_text(f"""
task: relax
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {15 * RY}
xc: lda
symmetry: false
kpoints: {{mesh: [2, 2, 2]}}
relax: {{optimizer: bfgs, cell: true, fmax: 0.05, max_steps: 1}}
""")
    out = tmp_path / "results"
    out.mkdir()
    # pre-seed a corrupt/leftover relax.xyz: the incremental writer must overwrite
    # it cleanly on the first frame without erroring
    (out / "relax.xyz").write_bytes(b"\x00 corrupt leftover, not an xyz \x00")
    assert main([str(tmp_path / "relax.yaml"), "-o", str(out)]) == 0
    shown = capsys.readouterr().out
    assert "── ionic step 1 ──" in shown       # per-step header
    assert "  SCF   1" in shown                 # nested electronic-step trace
    assert "ionic step   1 ·" in shown          # summary line, same number
    # timing + memory: per-iteration time on the SCF line, and the run footer
    assert re.search(r"\|drho\| = \S+ +\d+\.\d\ds", shown)  # per-SCF-iter seconds
    assert "wall" in shown and "peak RSS" in shown          # timing + memory footer

    # incremental trajectory: the corrupt pre-existing relax.xyz was cleanly
    # overwritten, each step saved with per-atom forces + (variable-cell) stress
    from ase.io import read as ase_read
    traj = ase_read(str(out / "relax.xyz"), index=":")
    assert len(traj) >= 1
    assert traj[0].get_forces().shape == (2, 3)
    assert traj[0].calc.results.get("stress") is not None

    # -q silences the whole electronic/ionic stream AND the footer
    assert main([str(tmp_path / "relax.yaml"), "-o", str(out), "-q"]) == 0
    quiet = capsys.readouterr().out
    assert "ionic step" not in quiet
    assert "SCF" not in quiet
    assert "peak RSS" not in quiet


@pytest.mark.standard
def test_cli_end_to_end_paw_with_restart(tmp_path):
    """YAML → USPP/PAW routing (formalism detected from the UPF), then a
    second run warm-started through the YAML restart: key."""
    from gradwave.cli import main

    def write_input(name, restart=None):
        extra = f"restart: {restart}\n" if restart else ""
        (tmp_path / name).write_text(f"""
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF}}
ecut: {12 * RY}
ecutrho: {48 * RY}
xc: pbe
kpoints: {{mesh: [2, 2, 2]}}
scf: {{etol: 1.0e-8, rhotol: 1.0e-7, mixing: {{scheme: johnson}}}}
{extra}""")

    write_input("input.yaml")
    out = tmp_path / "run1"
    assert main([str(tmp_path / "input.yaml"), "-o", str(out), "-q"]) == 0
    s1 = json.loads((out / "scf.json").read_text())
    assert s1["parameters"]["formalism"] == "uspp/paw"
    assert s1["parameters"]["ecutrho_eV"] == pytest.approx(48 * RY)
    assert s1["scf"]["energies_eV"]["onecenter"] != 0.0

    write_input("input2.yaml", restart=out / "checkpoint.pt")
    out2 = tmp_path / "run2"
    assert main([str(tmp_path / "input2.yaml"), "-o", str(out2), "-q"]) == 0
    s2 = json.loads((out2 / "scf.json").read_text())
    assert s2["scf"]["n_iter"] < s1["scf"]["n_iter"]
    assert abs(s2["scf"]["energies_eV"]["free_energy"]
               - s1["scf"]["energies_eV"]["free_energy"]) < 1e-6


@pytest.mark.slow
def test_variable_cell_relax_reduces_stress(tmp_path):
    """`relax.cell: true` runs a variable-cell (FrechetCellFilter) relaxation.
    A 3%-compressed Si diamond (atoms on symmetric sites, so only the cell has a
    force) relaxes back to a near-stress-free state; relax.json reports the moving
    cell, its volume, and the final stress."""
    from gradwave.cli import main

    strained = (0.97 * SI_CELL).tolist()  # 3% isotropic compression
    v_start = float(abs(np.linalg.det(np.array(strained))))
    (tmp_path / "vc.yaml").write_text(f"""
task: relax
structure:
  cell: {strained}
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]
pseudopotentials:
  dir: {FIX / "pseudos"}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {30 * RY}
xc: lda
kpoints: {{mesh: [4, 4, 4]}}
relax: {{optimizer: bfgs, fmax: 0.02, max_steps: 20, cell: true}}
""")
    out = tmp_path / "results"
    assert main([str(tmp_path / "vc.yaml"), "-o", str(out), "-q"]) == 0

    r = json.loads((out / "relax.json").read_text())["relax"]
    assert r["cell_relaxed"] is True
    assert r["converged"], f"variable-cell relax not converged in {r['n_steps']} steps"
    # converged => nearly stress-free, and the compressed cell expanded back
    assert r["max_stress_eV_ang3"] < 1.0e-3, r["max_stress_eV_ang3"]
    assert r["volume_ang3"] > v_start, (r["volume_ang3"], v_start)
    # the cell genuinely moves step to step (not an atoms-only run)
    cells = [f["cell_ang"] for f in r["trajectory"]]
    assert cells[0] != cells[-1]
