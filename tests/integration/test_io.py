"""Layer-C IO: checkpoint round-trip/restart, output files, CLI, analysis.

Fast tests run on a canned summary (no SCF); the standard-tier tests run
real small SCFs — a PAW checkpoint restart and a CLI end-to-end on an
NC config.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
A = 5.43
SI_CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0, 0], [A / 4] * 3])


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
                      "species": ["Si", "Si"]},
        "parameters": {"formalism": "uspp/paw", "xc": "pbe",
                       "ecut_eV": 204.0, "ecutrho_eV": 816.0,
                       "kmesh": [2, 2, 2], "nk": 3,
                       "kweights": [0.25, 0.5, 0.25], "nspin": 1,
                       "smearing": "none", "width_eV": 0.1,
                       "symmetry": True,
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
                ]},
        "eigenvalues_eV": eig.tolist(),
        "occupations": occ.tolist(),
        "runtime_s": 12.3,
        "outputs": {"json": "scf.json", "report": "scf.out"},
    }


def test_human_report_from_summary():
    from gradwave.output import format_output

    text = format_output(_canned_summary())
    for token in ("gradwave 0.1.0", "── structure", "── parameters",
                  "── self-consistency", "converged in 3 iterations",
                  "free energy F", "-245.0000000000", "gap 0.6000 eV",
                  "── eigenvalues"):
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
    from gradwave.output import format_output

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
    from gradwave import analysis

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


@pytest.mark.standard
def test_paw_checkpoint_roundtrip_and_restart(tmp_path):
    from gradwave.checkpoint import (
        as_start_from,
        load_checkpoint,
        save_checkpoint,
    )
    from gradwave.core.xc.pbe import PBE
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

    # the plot subcommand consumes the JSON it just wrote
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
