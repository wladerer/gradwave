"""PW/PAW EFG cross-validation on alpha-Al2O3 (corundum) vs the in-repo FLAPW EFG + Elk 11.

Runs a plane-wave PAW ground state (kjpaw pseudos already in tests/fixtures/qe/pseudos) and the
new ``postscf.efg_paw`` reconstruction, then reports V_zz / eta / C_Q for the O (anion, cleanest
target) and Al (cation) sites. References (see experiments/autoapw/efg_status.md):

    site   metric        gradwave-FLAPW    Elk 11      experiment
    O      V_zz          +25.56 (on-site)  +27.08
    O      C_Q(17O)      2.11 MHz          2.19 MHz
    Al     C_Q(27Al)     2.485 MHz         2.19 MHz    ~2.38 MHz

Cation caveat (from the FLAPW campaign): Al.pbe-n-kjpaw freezes the 2p semicore, so the Al
antishielding is under-captured exactly as the FLAPW frozen-[Ne] Al run under-shoots — the O
anion is the honest headline. Structure = corundum_efg.py (ASE spacegroup 167). Run on asus:

    OMP_NUM_THREADS=8 uv run python experiments/autoapw/paw_efg_corundum.py 2>&1 | tee log
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.constants import BOHR_ANG
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.efg_paw import efg_paw
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

RY = 13.605693122994
FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "qe" / "pseudos"

# corundum primitive cell (Bohr) + fractional atoms, from corundum_efg.py
CELL_BOHR = np.array([[4.497737, 2.596770, 8.184592],
                      [-4.497737, 2.596770, 8.184592],
                      [-0.000000, -5.193539, 8.184592]])
FRAC = np.array([
    [0.352160, 0.352160, 0.352160], [0.147840, 0.147840, 0.147840],
    [0.647840, 0.647840, 0.647840], [0.852160, 0.852160, 0.852160],   # Al ×4
    [0.250000, 0.556240, 0.943760], [0.056240, 0.750000, 0.443760],
    [0.556240, 0.943760, 0.250000], [0.443760, 0.056240, 0.750000],
    [0.943760, 0.250000, 0.556240], [0.750000, 0.443760, 0.056240],   # O ×6
])
SPECIES = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0=Al, 1=O
ISOTOPES = {"O": "17O", "Al": "27Al"}


def main() -> int:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    ecut = float(os.environ.get("ECUT_RY", "60")) * RY
    ecutrho = float(os.environ.get("ECUTRHO_RY", "480")) * RY
    kk = int(os.environ.get("K", "2"))

    al = parse_upf_paw(FIX / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    o = parse_upf_paw(FIX / "O.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell = CELL_BOHR * BOHR_ANG                    # -> Å
    pos = FRAC @ cell                              # fractional -> Cartesian Å
    n_el = 4 * al.z_valence + 6 * o.z_valence
    print(f"# corundum PW/PAW EFG  ecut={ecut/RY:.0f}Ry ecutrho={ecutrho/RY:.0f}Ry k{kk}^3 "
          f"n_el={n_el:.0f} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    t0 = time.time()
    system = setup_uspp(cell, pos, SPECIES, [al, o], ecut=ecut, kmesh=(kk, kk, kk),
                        ecutrho=ecutrho, nbands=int(n_el // 2 + 8))
    r = scf_uspp(system, PBE(), etol=1e-10, rhotol=1e-9, verbose=False, max_iter=200)
    print(f"# SCF converged={r['converged']} n_iter={r['n_iter']} "
          f"E={float(r['energies'].free_energy):.6f} eV  ({time.time()-t0:.0f}s)", flush=True)

    sites = efg_paw(r, isotopes=ISOTOPES)
    for s in sites:
        vsm = _vzz(s["V_smooth"]); vio = _vzz(s["V_ion"]); vos = _vzz(s["V_onsite"])
        cq = s.get("C_Q", {})
        line = (f"{s['element']}{s['site']:<2d} V_zz={s['V_zz']:+8.3f} eta={s['eta']:.3f}  "
                f"[smooth {vsm:+7.2f} | ion {vio:+7.2f} | onsite {vos:+7.2f}]")
        if cq:
            line += f"  C_Q({cq['isotope']})={cq['abs_C_Q_MHz']:.3f} MHz"
        print(line, flush=True)

    # equivalent-site spread (all Al equivalent, all O equivalent by R-3c symmetry)
    for elem in ("Al", "O"):
        vz = [s["V_zz"] for s in sites if s["element"] == elem]
        et = [s["eta"] for s in sites if s["element"] == elem]
        print(f"# {elem}: |V_zz| mean={np.mean(np.abs(vz)):.3f} spread={np.ptp(np.abs(vz)):.3f}  "
              f"eta mean={np.mean(et):.3f} spread={np.ptp(et):.3f}", flush=True)
    return 0


def _vzz(tensor: torch.Tensor) -> float:
    w = np.linalg.eigvalsh(tensor.detach().cpu().numpy())
    return float(w[np.argsort(np.abs(w))[2]])


if __name__ == "__main__":
    raise SystemExit(main())
