"""Control: all-NC MgO (ONCV Mg + O) through the PLAIN norm-conserving analytic
route (`sigma_shielding_dq`, uspp=None) at two ecut rungs.

If the NC bare term is ecut-stable on MgO while the PAW route diverges, the
defect is in the S-metric/USPP machinery (or its missing augmentation term),
not in the analytic ∂/∂q assembly itself.
"""
from __future__ import annotations

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.scf.loop import scf, setup_system

MG = "tests/fixtures/qe/pseudos/Mg_ONCV_PBE-1.2.upf"
O = "tests/fixtures/qe/pseudos/O_ONCV_PBE-1.2.upf"
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL


def run(ecut_ry: float) -> list[float]:
    from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq
    from gradwave.pseudo.upf import parse_upf

    torch.set_num_threads(5)
    upfs = [parse_upf(MG), parse_upf(O)]
    system = setup_system(CELL, POS, [0, 1], upfs, ecut=ecut_ry * RY,
                          kmesh=(2, 2, 2), nbands=12, use_symmetry=False)
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=120)
    assert res.converged
    sig = sigma_shielding_dq(res, use_symmetry=False)
    iso = [float(torch.diagonal(sig[a]).mean()) for a in range(2)]
    print(f"NC ecut {ecut_ry} Ry: bare sigma_iso  Mg = {iso[0]:+.2f}  "
          f"O = {iso[1]:+.2f} ppm", flush=True)
    return iso


if __name__ == "__main__":
    for e in (40.0, 60.0, 80.0):
        run(e)
