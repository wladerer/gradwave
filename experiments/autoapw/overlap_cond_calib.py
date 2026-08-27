"""Calibrate the overlap-conditioning guard: cond(S(Gamma)) for a SOFT dataset
(Si PAW, works) vs the HARD MgO O dataset (diverges), no SCF needed."""
from __future__ import annotations

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf.kgeometry_nmr import _overlap_kbprojectors
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import setup_uspp


def cond_S(paws, cell, pos, species, ecut_ry, ecutrho_ry):
    system = setup_uspp(cell, pos, species, paws, ecut=ecut_ry * RY,
                        kmesh=(1, 1, 1), ecutrho=ecutrho_ry * RY, nbands=12)
    kb = _overlap_kbprojectors(system, system.spheres[0])
    npw = kb.g_cart.shape[0]
    p = kb.p(torch.zeros(3, dtype=RDTYPE))
    smat = torch.eye(npw, dtype=CDTYPE) + p.mT @ (kb.dij_full.to(CDTYPE) @ p.conj())
    ev = torch.linalg.eigvalsh(0.5 * (smat + smat.mH))
    return npw, float(ev.min()), float(ev.max())


if __name__ == "__main__":
    torch.set_num_threads(5)
    # Si PAW (soft, works: bare ~ -5.7 ppm, stable)
    si = parse_upf_paw("tests/fixtures/qe/pseudos/Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    cell_si = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    pos_si = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell_si
    for e in (12.0, 20.0, 30.0):
        npw, mn, mx = cond_S([si], cell_si, pos_si, [0, 0], e, 4 * e)
        print(f"Si PAW  ecut {e:5.1f}: npw={npw} min-eig={mn:.4e} max={mx:.3f} "
              f"cond={mx / mn:.1f}", flush=True)
    # MgO (hard O, diverges)
    mg = parse_upf_paw("Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF")
    o = parse_upf_paw("tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF")
    A = 4.21
    cell_mgo = 0.5 * A * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    pos_mgo = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) @ cell_mgo
    for e, er in ((40.0, 160.0), (60.0, 360.0)):
        npw, mn, mx = cond_S([mg, o], cell_mgo, pos_mgo, [0, 1], e, er)
        print(f"MgO     ecut {e:5.1f}: npw={npw} min-eig={mn:.4e} max={mx:.3f} "
              f"cond={mx / mn:.1f}", flush=True)
    print("CALIB_DONE", flush=True)
