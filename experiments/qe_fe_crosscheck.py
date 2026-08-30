"""Quantum ESPRESSO cross-check of the bcc-Fe moment at matched settings.

The decisive test for the gradwave 6^3 -> 8^3 moment drift: run QE (pw.x) with
the SAME pseudopotential (Fe_ONCV_PBE-1.2.upf, read natively by QE), same
ecutwfc 60 Ry, same a = 2.87 A, at k = 6^3, 8^3, 10^3 with gaussian and
Methfessel-Paxton (mp1) smearing at degauss = 0.1 eV. If QE ALSO rises to ~2.40
at 8^3/gaussian, gradwave tracks QE and the drift is real (6^3 under-converged);
if QE stays ~2.22, gradwave diverges from QE at fine mesh (a real discrepancy).

Pure standard library so it runs under `nix shell nixpkgs#quantum-espresso`.

Run (asus):
    nix shell nixpkgs#quantum-espresso --command \
        python3 experiments/qe_fe_crosscheck.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PSE = (HERE.parent / "tests" / "fixtures" / "qe" / "pseudos").resolve()
RY_EV = 13.605693122994
DEGAUSS_RY = 0.1 / RY_EV  # 0.1 eV in Ry

MESHES = [6, 8, 10]
# QE smearing keyword -> label. mp1 = Methfessel-Paxton order 1 (QE default n=1).
SMEARINGS = {"gaussian": "gaussian", "mp1": "methfessel-paxton"}

PWIN = """&control
  calculation = 'scf'
  prefix = 'fe_x'
  pseudo_dir = '{pseudo_dir}'
  outdir = './tmp'
  verbosity = 'high'
/
&system
  ibrav = 0
  nat = 1
  ntyp = 1
  ecutwfc = 60.0
  occupations = 'smearing'
  smearing = '{smearing}'
  degauss = {degauss:.10f}
  nspin = 2
  starting_magnetization(1) = 0.4
  nbnd = 12
/
&electrons
  conv_thr = 1.0d-9
  mixing_beta = 0.3
/
ATOMIC_SPECIES
Fe 55.845 Fe_ONCV_PBE-1.2.upf
CELL_PARAMETERS angstrom
-1.435 1.435 1.435
1.435 -1.435 1.435
1.435 1.435 -1.435
ATOMIC_POSITIONS crystal
Fe 0.00 0.00 0.00
K_POINTS automatic
{k} {k} {k} 0 0 0
"""


def qe_version() -> str:
    out = subprocess.run(["pw.x", "-version"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL)
    m = re.search(r"PWSCF\s+(v\.\S+)", out.stdout + out.stderr)
    return m.group(1) if m else "unknown"


def _parse(text: str) -> dict:
    out: dict = {}
    for key, pat in (("m_tot", r"total magnetization\s*=\s*([-\d.]+)"),
                     ("m_abs", r"absolute magnetization\s*=\s*([-\d.]+)")):
        hits = re.findall(pat, text)
        if hits:
            out[key] = float(hits[-1])
    fe = re.findall(r"the Fermi energy is\s*([-\d.]+)\s*ev", text)
    if fe:
        out["fermi_eV"] = float(fe[-1])
    en = re.findall(r"!\s*total energy\s*=\s*([-\d.]+)\s*Ry", text)
    if en:
        out["etot_eV"] = float(en[-1]) * RY_EV
    out["converged"] = "convergence has been achieved" in text
    return out


def run_point(label: str, smearing: str, k: int) -> dict:
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "pw.in").write_text(
            PWIN.format(pseudo_dir=str(PSE), smearing=smearing,
                        degauss=DEGAUSS_RY, k=k))
        proc = subprocess.run(["pw.x", "-in", "pw.in"], cwd=dp,
                              capture_output=True, text=True,
                              stdin=subprocess.DEVNULL)
        res = _parse(proc.stdout)
    res.update({"smearing": label, "qe_smearing": smearing, "k": k})
    return res


def main() -> None:
    if shutil.which("pw.x") is None:
        raise SystemExit("pw.x not on PATH — run under "
                         "`nix shell nixpkgs#quantum-espresso`")
    ver = qe_version()
    print(f"QE {ver}; pseudo_dir={PSE}", flush=True)
    rows = []
    for k in MESHES:
        for label, sm in SMEARINGS.items():
            r = run_point(label, sm, k)
            rows.append(r)
            print(f"QE {label:8s} k={k}^3: m_tot={r.get('m_tot')} "
                  f"m_abs={r.get('m_abs')} Ef={r.get('fermi_eV')} "
                  f"conv={r.get('converged')}", flush=True)
    out = {"qe_version": ver, "degauss_Ry": DEGAUSS_RY,
           "degauss_eV": 0.1, "pseudo": "Fe_ONCV_PBE-1.2.upf",
           "ecutwfc_Ry": 60.0, "a_ang": 2.87, "rows": rows}
    path = HERE / "qe_fe_data.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
