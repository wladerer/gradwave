"""Bi2Se3 band inversion at Gamma: scalar-relativistic vs SOC, with parities.

Rhombohedral primitive cell (R-3m, Se1 at the inversion center). The
topological signature: the parities of the gap-edge states at Gamma SWAP
when SOC is turned on (Zhang et al., Nat. Phys. 2009).
"""
import time

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.irreps import band_irreps
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear

torch.set_num_threads(8)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"

A_HEX, C_HEX = 4.138, 28.64
MU, NU = 0.399, 0.206
CELL = np.array([
    [A_HEX / 2, A_HEX / (2 * np.sqrt(3)), C_HEX / 3],
    [-A_HEX / 2, A_HEX / (2 * np.sqrt(3)), C_HEX / 3],
    [0.0, -A_HEX / np.sqrt(3), C_HEX / 3],
])
FRAC = np.array([
    [0.0, 0.0, 0.0],          # Se1 (inversion center)
    [NU, NU, NU], [-NU, -NU, -NU],    # Se2
    [MU, MU, MU], [-MU, -MU, -MU],    # Bi
])
SPECIES = [0, 0, 0, 1, 1]  # 0 = Se, 1 = Bi
ECUT = 45 * RY

timings = {}


def spinor_parity(system, coeffs_gamma, band):
    """<psi|P|psi> for a spinor state at Gamma (inversion about the origin)."""
    ig = [i for i, sp in enumerate(system.spheres)
          if np.abs(sp.k_frac).max() < 1e-9][0]
    sph = system.spheres[ig]
    miller = sph.miller.cpu().numpy()
    index = {tuple(m): i for i, m in enumerate(miller)}
    perm = np.array([index[tuple(-m)] for m in miller])
    npw = sph.npw
    m_pw = system.batch.npw_max
    c = coeffs_gamma[band].cpu().numpy()
    p = 0.0
    for off in (0, m_pw):
        comp = c[off:off + npw]
        p += np.vdot(comp, comp[perm]).real
    return p, ig


# ---------------- scalar-relativistic (no SOC) ----------------
print("=== no-SOC (SG15 SR pseudos) ===", flush=True)
se = parse_upf(f"{PSE}/Se_ONCV_PBE-1.1.upf")
bi = parse_upf(f"{PSE}/Bi_ONCV_PBE-1.0.upf")
t0 = time.time()
sys_sr = setup_system(CELL, FRAC @ CELL, SPECIES, [se, bi], ecut=ECUT,
                      kmesh=(2, 2, 2), nbands=30)
if DEV != "cpu":
    sys_sr = sys_sr.to(DEV)
timings["SR setup"] = time.time() - t0
print(f"  ne={sys_sr.n_electrons:.0f} nk={len(sys_sr.spheres)} "
      f"npw={sys_sr.spheres[0].npw} grid={sys_sr.grid.shape}", flush=True)
t0 = time.time()
r_sr = scf(sys_sr, PBE(), smearing="gaussian", width=0.05,
           etol=1e-7, rhotol=1e-6, verbose=False)
if DEV != "cpu":
    torch.cuda.synchronize()
timings["SR scf"] = time.time() - t0
timings["SR iters"] = r_sr.n_iter
print(f"  conv={r_sr.converged} iters={r_sr.n_iter}", flush=True)

t0 = time.time()
irr = band_irreps(r_sr, [0, 0, 0], nbands=28)
timings["SR irreps"] = time.time() - t0
nocc = int(sys_sr.n_electrons // 2)  # 24
e_cursor = 0
print("  Gamma clusters around the gap (band_irreps, parity from g/u):")
count = 0
for cl in irr.clusters:
    lo = count
    count += cl.dim
    if lo <= nocc + 1 and count >= nocc - 3:
        occ_tag = "VB" if count <= nocc else ("CB" if lo >= nocc else "GAP-EDGE?")
        print(f"    bands {lo}-{count-1}: {cl.label:>4s}  "
              f"E={np.mean(cl.energies)-r_sr.fermi:+.4f} eV  [{occ_tag}]", flush=True)

# ---------------- SOC (PseudoDojo FR, NLCC) ----------------
print("\n=== SOC (PseudoDojo FR pseudos, NLCC) ===", flush=True)
se_fr = parse_upf(f"{PSE}/PD_Se_FR.upf")
bi_fr = parse_upf(f"{PSE}/PD_Bi_FR.upf")
t0 = time.time()
sys_fr = setup_system(CELL, FRAC @ CELL, SPECIES, [se_fr, bi_fr], ecut=ECUT,
                      kmesh=(2, 2, 2), nbands=45, time_reversal=False)
if DEV != "cpu":
    sys_fr = sys_fr.to(DEV)
timings["FR setup"] = time.time() - t0
ne = int(sys_fr.n_electrons)
print(f"  ne={ne} nk={len(sys_fr.spheres)} npw={sys_fr.spheres[0].npw} "
      f"grid={sys_fr.grid.shape} NLCC={sys_fr.rho_core is not None}", flush=True)
t0 = time.time()
r_fr = scf_noncollinear(sys_fr, NoncollinearXC(SpinPBE()),
                        mag_vec_init=[[0, 0, 0]] * 5,
                        smearing="gaussian", width=0.05,
                        etol=1e-7, rhotol=1e-6, verbose=False,
                        nonmagnetic=True)
if DEV != "cpu":
    torch.cuda.synchronize()
timings["FR scf"] = time.time() - t0
timings["FR iters"] = r_fr.n_iter
print(f"  conv={r_fr.converged} iters={r_fr.n_iter}", flush=True)

t0 = time.time()
ig = [i for i, sp in enumerate(sys_fr.spheres)
      if np.abs(sp.k_frac).max() < 1e-9][0]
eg = r_fr.eigenvalues[ig].cpu().numpy()
print(f"  Gamma gap-edge states (occupied spinor states: {ne}):")
for band in range(ne - 4, ne + 4):
    par, _ = spinor_parity(sys_fr, r_fr.coeffs[ig], band)
    tag = "VB" if band < ne else "CB"
    print(f"    band {band}: E={eg[band]-r_fr.fermi:+.4f} eV  "
          f"parity={par:+.3f}  [{tag}]", flush=True)
timings["FR parity"] = time.time() - t0

print("\nTimings:")
for k, v in timings.items():
    print(f"  {k:12s} {v if isinstance(v, int) else round(v, 1)}")
