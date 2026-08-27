"""Front A — PAW O-dataset completeness scan for the anion EFG asymmetry eta on corundum Al2O3.

The PAW on-site l=2 density (the term that carries eta) is bounded by the AE/PS partial-wave set
of the dataset. Hypothesis: a richer/different O PAW dataset closes the eta gap (corundum O
eta=0.48 on the PW/PAW path, PR #394, vs Elk 0.74).

Reality found while scoping: ``pseudo.upf_paw.parse_upf_paw`` only accepts psl-style kjpaw PAW
(q_with_l, nqf=0, scalar/none-relativistic); it rejects RRKJ-refit USPPs (nqf>0) and never sees
JTH XML-PAW. GBRV / plain-USPP datasets parse but carry is_paw=False and NO AE/PS partial waves,
so the Petrilli-Blochl on-site term (``EFGOnSite.from_paw``) cannot be built from them at all.
Every parseable O PAW generation (psl 1.0.0, psl 0.1, old GIPAW-era kjpaw) ships the SAME 4
projectors (2 per l-channel: 2s x2, 2p x2), so this scans dataset GENERATION choices (rc,
reference energies, partial-wave shapes), not projector COUNT.

Provide the O datasets as a comma list of absolute paths in O_UPFS. Al stays fixed (the anion is
the honest target; Al freezes its 2p semicore on both paths). Elk 11: corundum O eta 0.74,
C_Q(17O) 2.19 MHz, V_zz(on-site) +27.08.

Run on asus (fetch the UPFs first with efg_fetch_o_datasets.sh):
  OMP_NUM_THREADS=5 O_UPFS="/tmp/efg_pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF,..." \
    uv run python experiments/autoapw/efg_eta_paw_datasets.py 2>&1 | tee scanA.log
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

# corundum primitive cell (Bohr) + fractional atoms (from paw_efg_corundum.py / corundum_efg.py)
CELL_BOHR = np.array([[4.497737, 2.596770, 8.184592],
                      [-4.497737, 2.596770, 8.184592],
                      [-0.000000, -5.193539, 8.184592]])
FRAC = np.array([
    [0.352160, 0.352160, 0.352160], [0.147840, 0.147840, 0.147840],
    [0.647840, 0.647840, 0.647840], [0.852160, 0.852160, 0.852160],   # Al x4
    [0.250000, 0.556240, 0.943760], [0.056240, 0.750000, 0.443760],
    [0.556240, 0.943760, 0.250000], [0.443760, 0.056240, 0.750000],
    [0.943760, 0.250000, 0.556240], [0.750000, 0.443760, 0.056240],   # O x6
])
SPECIES = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0=Al, 1=O
ISOTOPES = {"O": "17O", "Al": "27Al"}
ELK = dict(O_eta=0.740, O_cq=2.19, O_vzz_onsite=27.08, Al_cq=2.19)


def _vzz(tensor: torch.Tensor) -> float:
    w = np.linalg.eigvalsh(tensor.detach().cpu().numpy())
    return float(w[np.argsort(np.abs(w))[2]])


def run_dataset(o_upf: str, ecut: float, ecutrho: float, kk: int) -> dict:
    al = parse_upf_paw(FIX / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    o = parse_upf_paw(o_upf)
    cell = CELL_BOHR * BOHR_ANG
    pos = FRAC @ cell
    n_el = 4 * al.z_valence + 6 * o.z_valence
    t0 = time.time()
    system = setup_uspp(cell, pos, SPECIES, [al, o], ecut=ecut, kmesh=(kk, kk, kk),
                        ecutrho=ecutrho, nbands=int(n_el // 2 + 8))
    r = scf_uspp(system, PBE(), etol=1e-10, rhotol=1e-9, verbose=False, max_iter=200)
    sites = efg_paw(r, isotopes=ISOTOPES)
    o_eta = [s["eta"] for s in sites if s["element"] == "O"]
    o_vzz = [s["V_zz"] for s in sites if s["element"] == "O"]
    o_onsite = [_vzz(s["V_onsite"]) for s in sites if s["element"] == "O"]
    o_cq = [s["C_Q"]["abs_C_Q_MHz"] for s in sites if s["element"] == "O" and "C_Q" in s]
    al_cq = [s["C_Q"]["abs_C_Q_MHz"] for s in sites if s["element"] == "Al" and "C_Q" in s]
    return dict(
        converged=bool(r["converged"]), n_iter=int(r["n_iter"]),
        o_eta=float(np.mean(o_eta)), o_eta_spread=float(np.ptp(o_eta)),
        o_vzz=float(np.mean(np.abs(o_vzz))), o_onsite=float(np.mean(o_onsite)),
        o_cq=float(np.mean(o_cq)) if o_cq else float("nan"),
        al_cq=float(np.mean(al_cq)) if al_cq else float("nan"),
        secs=time.time() - t0,
    )


def main() -> int:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "5")))
    ecut = float(os.environ.get("ECUT_RY", "60")) * RY
    ecutrho = float(os.environ.get("ECUTRHO_RY", "480")) * RY
    kk = int(os.environ.get("K", "2"))
    upfs = [u for u in os.environ.get("O_UPFS", "").split(",") if u]
    if not upfs:
        upfs = [str(FIX / "O.pbe-n-kjpaw_psl.1.0.0.UPF")]
    print(f"# FRONT A corundum PAW O-dataset scan  ecut={ecut/RY:.0f}Ry ecutrho={ecutrho/RY:.0f}Ry "
          f"k{kk}^3 OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    print(f"# Elk 11: O eta={ELK['O_eta']:.3f} C_Q(17O)={ELK['O_cq']} MHz "
          f"V_zz_onsite={ELK['O_vzz_onsite']}", flush=True)
    rows = []
    for u in upfs:
        name = Path(u).name
        try:
            d = run_dataset(u, ecut, ecutrho, kk)
            print(f"{name:38s} | O eta={d['o_eta']:.3f} (spread {d['o_eta_spread']:.3f}) "
                  f"|V_zz|={d['o_vzz']:6.3f} onsite={d['o_onsite']:+7.3f} "
                  f"C_Q(17O)={d['o_cq']:.3f} | Al C_Q={d['al_cq']:.3f} | "
                  f"conv={d['converged']} n_it={d['n_iter']} ({d['secs']:.0f}s)", flush=True)
            rows.append((name, d))
        except Exception as ex:  # report and continue the scan
            print(f"{name:38s} | FAILED: {type(ex).__name__}: {ex}", flush=True)
    print("# --- summary (dataset -> O eta, |V_zz|, C_Q(17O), Al C_Q) ---", flush=True)
    for name, d in rows:
        print(f"#  {name:38s}  eta={d['o_eta']:.3f}  |V_zz|={d['o_vzz']:6.3f}  "
              f"C_Q={d['o_cq']:.3f}  AlC_Q={d['al_cq']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
