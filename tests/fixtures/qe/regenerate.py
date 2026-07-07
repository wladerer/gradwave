"""Regenerate QE reference data for all fixture directories.

Usage:  uv run python tests/fixtures/qe/regenerate.py [case ...]

Requires pw.x on PATH (NixOS: `nix shell nixpkgs#quantum-espresso`, or the
system profile). For each case directory containing pw.in, runs pw.x and
writes reference.json with full-precision values parsed from the XML output
(data-file-schema.xml, Hartree units) converted to eV/Å, plus the per-term
breakdown from stdout (Ry). Any change to a pw.in requires rerunning this
and committing pw.in + reference.json together.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HARTREE_EV = 27.211386245988
RY_EV = HARTREE_EV / 2.0

HERE = Path(__file__).parent


def qe_version() -> str:
    out = subprocess.run(
        ["pw.x", "-version"], capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    m = re.search(r"PWSCF\s+(v\.\S+)", out.stdout + out.stderr)
    return m.group(1) if m else "unknown"


def parse_stdout_terms(text: str) -> dict:
    terms = {}
    patterns = {
        "one_electron": r"one-electron contribution\s*=\s*([-\d.]+)\s*Ry",
        "hartree": r"hartree contribution\s*=\s*([-\d.]+)\s*Ry",
        "xc": r"xc contribution\s*=\s*([-\d.]+)\s*Ry",
        "ewald": r"ewald contribution\s*=\s*([-\d.]+)\s*Ry",
        "smearing": r"smearing contrib.*=\s*([-\d.]+)\s*Ry",
        "internal_energy": r"internal energy E=F\+TS\s*=\s*([-\d.]+)\s*Ry",
    }
    for key, pat in (("total_magnetization", r"total magnetization\s*=\s*([-\d.]+)"),
                     ("absolute_magnetization", r"absolute magnetization\s*=\s*([-\d.]+)")):
        hits = re.findall(pat, text)
        if hits:
            terms[key] = float(hits[-1])
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            terms[key + "_eV"] = float(m.group(1)) * RY_EV
    return terms


def parse_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    out = root.find("output")
    tot = out.find("total_energy")
    data = {
        "etot_eV": float(tot.find("etot").text) * HARTREE_EV,
        "ewald_eV": float(tot.find("ewald").text) * HARTREE_EV
        if tot.find("ewald") is not None
        else None,
    }
    bs = out.find("band_structure")
    ef = bs.find("fermi_energy")
    if ef is not None:
        data["fermi_eV"] = float(ef.text) * HARTREE_EV
    hom = bs.find("highestOccupiedLevel")
    if hom is not None:
        data["vbm_eV"] = float(hom.text) * HARTREE_EV
    fr = out.find("forces")
    if fr is not None:
        # QE XML forces are Ha/bohr
        vals = [float(x) for x in fr.text.split()]
        ha_bohr_to_ev_ang = HARTREE_EV / 0.529177210903
        data["forces_eV_ang"] = [
            [v * ha_bohr_to_ev_ang for v in vals[i : i + 3]] for i in range(0, len(vals), 3)
        ]
    kpts, eigs, occs = [], [], []
    for ks in bs.findall("ks_energies"):
        kpts.append([float(x) for x in ks.find("k_point").text.split()])
        eigs.append([float(x) * HARTREE_EV for x in ks.find("eigenvalues").text.split()])
        occs.append([float(x) for x in ks.find("occupations").text.split()])
    data["k_points_tpiba"] = kpts  # units of 2π/alat
    data["eigenvalues_eV"] = eigs
    data["occupations"] = occs
    alat = root.find("output/atomic_structure").attrib.get("alat")
    data["alat_bohr"] = float(alat) if alat else None
    return data


def run_case(case: Path) -> None:
    print(f"== {case.name}")
    txt = subprocess.run(
        ["pw.x", "-in", "pw.in"], cwd=case, capture_output=True, text=True
    )
    if "JOB DONE" not in txt.stdout:
        print(txt.stdout[-3000:])
        raise RuntimeError(f"{case.name}: pw.x failed")
    (case / "pw.out").write_text(txt.stdout)

    prefix = re.search(r"prefix\s*=\s*'([^']+)'", (case / "pw.in").read_text()).group(1)
    xml_path = case / "tmp" / f"{prefix}.save" / "data-file-schema.xml"
    data = {"qe_version": qe_version(), **parse_xml(xml_path), **parse_stdout_terms(txt.stdout)}
    m = re.search(r"Dense\s+grid:.*FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)",
                  txt.stdout)
    if m:
        data["fft_dims"] = [int(m.group(i)) for i in (1, 2, 3)]

    # optional second stage: non-SCF bands run reusing the same outdir
    if (case / "pw_bands.in").exists():
        btxt = subprocess.run(
            ["pw.x", "-in", "pw_bands.in"], cwd=case, capture_output=True, text=True
        )
        if "JOB DONE" not in btxt.stdout:
            print(btxt.stdout[-3000:])
            raise RuntimeError(f"{case.name}: bands pw.x failed")
        bdata = parse_xml(xml_path)
        data["bands"] = {
            "k_points_tpiba": bdata["k_points_tpiba"],
            "eigenvalues_eV": bdata["eigenvalues_eV"],
            "vbm_eV": bdata.get("vbm_eV"),
        }

    (case / "reference.json").write_text(json.dumps(data, indent=1))
    print(f"   etot = {data['etot_eV']:.8f} eV  -> reference.json")


def main():
    wanted = sys.argv[1:]
    cases = [d for d in sorted(HERE.iterdir()) if (d / "pw.in").exists()]
    if wanted:
        cases = [c for c in cases if c.name in wanted]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
