"""gradwave FLAPW EFG for rutile TiO2 — validate vs Elk (experiments/autoapw/TIO2_NMR.md).
Args: [ecut] [kx ky kz] [iters] [fp] [smear] [rO] [rTi] (see the argv parsing below)."""
import sys
import time

import numpy as np

from gradwave.flapw import crystal_scf_multi
from gradwave.flapw.nmr import quadrupolar_coupling


def evals(site):
    """The three EFG principal values (eV/Å^2), sorted by descending magnitude: V_zz, V_yy, V_xx."""
    w = np.linalg.eigvalsh(site["tensor"])
    return w[np.argsort(-np.abs(w))]

ecut = float(sys.argv[1]) if len(sys.argv) > 1 else 320.0
kmesh = tuple(int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (2, 2, 3)
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 40
fullpot = len(sys.argv) > 6 and sys.argv[6].lower() in ("fp", "fullpot", "1", "true")
smearing = float(sys.argv[7]) if len(sys.argv) > 7 else 0.15
r_o = float(sys.argv[8]) if len(sys.argv) > 8 else 0.80      # O muffin-tin radius (Å)
r_ti = float(sys.argv[9]) if len(sys.argv) > 9 else 0.95     # Ti muffin-tin radius (Å)

u = 0.3048
a_bohr = [8.68083, 8.68083, 5.59096]          # rutile a,a,c in Bohr
atoms = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
         ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
         ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]
radii = {"Ti": r_ti, "O": r_o}                # Ti-O min bond 1.946 Å; no MT overlap

t0 = time.time()
bands, info = crystal_scf_multi(a_bohr, atoms, radii, ecut=ecut, lmax=2, iters=iters,
                                kmesh=kmesh, smearing=smearing, efg=True, fullpot=fullpot)
dt = time.time() - t0
print(f"TiO2 ecut={ecut} kmesh={kmesh} iters={iters} fullpot={fullpot}: "
      f"{dt:.1f}s, e_fermi={info.get('e_fermi')}")
# Elk ref eigenvalues (eV/Å^2, |V_zz|>|V_yy|>|V_xx|): Ti [+19.34,-13.16,-6.18] eta 0.36;
#                                                     O  [-19.1, +16.6, +2.5] eta 0.74
ti = info["efg"]["a0"]
o = info["efg"]["a2"]
et, eo = evals(ti), evals(o)
cq_ti = quadrupolar_coupling(ti["V_zz"], ti["eta"], "49Ti")
cq_o = quadrupolar_coupling(o["V_zz"], o["eta"], "17O")
print(f"  Ti: evals=[{et[0]:+.2f},{et[1]:+.2f},{et[2]:+.2f}] eta={ti['eta']:.3f} "
      f"C_Q(49Ti)={cq_ti['abs_C_Q_MHz']:.2f}  | Elk [+19.34,-13.16,-6.18] 0.36 11.5")
print(f"  O : evals=[{eo[0]:+.2f},{eo[1]:+.2f},{eo[2]:+.2f}] eta={o['eta']:.3f} "
      f"C_Q(17O)={cq_o['abs_C_Q_MHz']:.3f}  | Elk [-19.10,+16.60,+2.50] 0.74 1.18")
