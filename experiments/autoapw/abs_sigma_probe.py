"""De-risk probe for the absolute-sigma USPP scoping.

(1) NC bare-sigma reference on Si (2,2,2) + wall time.
(2) Prove sigma_shielding is mechanically NC-locked on a PAW SCF (it derefs
    system.batch / res.v_eff that USPPSystem/USPPResult do not carry).
(3) Print the PAW smooth+nonlocal continuity closure number (the load-bearing
    validation that the S-metric PAW delta-u is physically correct).
"""
import time
import numpy as np
import torch

torch.set_num_threads(2)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.loop import scf, setup_system
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.kgeometry_nmr import (
    sigma_shielding, build_uspp_response_ctx, uspp_smooth_continuity,
    velocity_perturbation_q,
)
from gradwave.postscf._response import insulator_window
from tests.helpers import RY, si_fcc, si_upf
from pathlib import Path

PSEUDOS = Path("tests/fixtures/qe/pseudos")

print("=" * 70)
print("(1) NC bare-sigma reference: Si (2,2,2), ecut=12Ry")
cell, pos = si_fcc()
system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                      kmesh=(2, 2, 2), nbands=8, use_symmetry=False,
                      fft_shape=(20, 20, 20))
res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
print("  converged:", res.converged)
t0 = time.time()
sig = sigma_shielding(res, cg_tol=1e-9)
dt = time.time() - t0
diag = torch.diagonal(sig, dim1=1, dim2=2)
print(f"  bare sigma_iso per site (ppm): {diag.mean(dim=1).tolist()}")
print(f"  sigma_shielding wall time: {dt:.1f} s")

print("=" * 70)
print("(2) PAW Si SCF + prove sigma_shielding NC-lock")
paw = parse_upf_paw(PSEUDOS / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
from gradwave.scf.uspp import scf_uspp, setup_uspp
psys = setup_uspp(cell, pos, [0, 0], [paw], ecut=12 * RY,
                  kmesh=(2, 1, 1), nbands=8, use_symmetry=False,
                  fft_shape=(20, 20, 20))
pres = scf_uspp(psys, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                verbose=False, max_iter=80)
print("  PAW converged:", pres["converged"])
try:
    sigma_shielding(pres, cg_tol=1e-9)
    print("  sigma_shielding on PAW: RAN (unexpected)")
except Exception as e:
    print(f"  sigma_shielding on PAW FAILS: {type(e).__name__}: {str(e)[:120]}")

print("=" * 70)
print("(3) PAW smooth+nonlocal continuity closure (S-metric delta-u check)")
ctx = build_uspp_response_ctx(pres, PBE())
q_frac = np.array([0.5, 0.0, 0.0])
e_pol = np.array([0.0, 1.0, 0.0])
sol = velocity_perturbation_q(pres, q_frac, cg_tol=1e-11, max_iter=800, uspp=ctx)
c = uspp_smooth_continuity(pres, ctx, sol, e_pol)
scale = abs(c["source"])
rel_kin = abs(c["q_j_kin"] - c["source"]) / scale
rel_full = abs(c["q_j_kin"] + c["q_j_nl"] - c["source"]) / scale
print(f"  source={c['source']:.6e}")
print(f"  q_j_kin={c['q_j_kin']:.6e}  q_j_nl={c['q_j_nl']:.6e}")
print(f"  rel_kin_only={rel_kin:.3e} (KB matters)  rel_full={rel_full:.3e} (closes)")

print("=" * 70)
print("(4) PAW becsum / on-site bridge availability for para_aug X_beta")
print("  rho_ij_atoms present:", "rho_ij_atoms" in pres,
      "becps present:", "becps" in pres)
for kk in pres:
    if "rho_ij" in kk or "becp" in kk:
        v = pres[kk]
        try:
            shp = [t.shape for t in v] if isinstance(v, list) else v.shape
        except Exception:
            shp = type(v)
        print(f"    {kk}: {shp}")
print("DONE")
