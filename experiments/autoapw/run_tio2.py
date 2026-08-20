"""gradwave FLAPW EFG for rutile TiO2 — validate vs Elk (experiments/autoapw/TIO2_NMR.md).
Args: [ecut] [kx ky kz] [iters] [fp] [smear] [rO] [rTi] [lmax] [fp_lmax] [lo].
Env: GW_KWORKERS = k-point process-pool size (divide OMP_NUM_THREADS to match)."""
import os
import sys
import time

import numpy as np

from gradwave.flapw import crystal_scf_multi
from gradwave.flapw.nmr import quadrupolar_coupling


def evals(site):
    """The three EFG principal values (eV/Å^2), sorted by descending magnitude: V_zz, V_yy, V_xx."""
    w = np.linalg.eigvalsh(site["tensor"])
    return w[np.argsort(-np.abs(w))]


def main():
    ecut = float(sys.argv[1]) if len(sys.argv) > 1 else 320.0
    kmesh = tuple(int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (2, 2, 3)
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    fullpot = len(sys.argv) > 6 and sys.argv[6].lower() in ("fp", "fullpot", "1", "true")
    smearing = float(sys.argv[7]) if len(sys.argv) > 7 else 0.15
    r_o = float(sys.argv[8]) if len(sys.argv) > 8 else 0.80      # O muffin-tin radius (Å)
    r_ti = float(sys.argv[9]) if len(sys.argv) > 9 else 0.95     # Ti muffin-tin radius (Å)
    lmax = int(sys.argv[10]) if len(sys.argv) > 10 else 2        # augmentation l cutoff
    fp_lmax = int(sys.argv[11]) if len(sys.argv) > 11 else 2     # non-spherical potential L cutoff
    use_lo = len(sys.argv) > 12 and sys.argv[12].lower() in ("lo", "1", "true")
    kworkers = int(os.environ.get("GW_KWORKERS", "1"))

    # LAPW+LO semicore treatment (Blaha/Schwarz-style): Ti 3s/3p as confined local orbitals,
    # the l=1 energy parameter moved to the valence; an extra O 2p-flexibility LO above the band.
    lo_kw = {}
    if use_lo:
        lo_kw = dict(
            los={"Ti": [(0, "3s"), (1, "3p")], "O": [(1, 5.0)]},
            core={"Ti": [(0, 1, 2), (0, 2, 2), (1, 1, 6)]},      # freeze only 1s, 2s, 2p
            val_e={"Ti": 12},                                    # 3s2 3p6 3d2 4s2
            el_override={"Ti": {1: "3d"}},                       # l=1 linearized in the valence
        )

    u = 0.3048
    a_bohr = [8.68083, 8.68083, 5.59096]          # rutile a,a,c in Bohr
    atoms = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
             ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
             ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]
    radii = {"Ti": r_ti, "O": r_o}                # Ti-O min bond 1.946 Å; no MT overlap

    t0 = time.time()
    bands, info = crystal_scf_multi(a_bohr, atoms, radii, ecut=ecut, lmax=lmax, iters=iters,
                                    kmesh=kmesh, smearing=smearing, efg=True, fullpot=fullpot,
                                    fullpot_lmax=fp_lmax, kworkers=kworkers, **lo_kw)
    dt = time.time() - t0
    print(f"TiO2 ecut={ecut} kmesh={kmesh} iters={iters} fullpot={fullpot} lmax={lmax} "
          f"fp_lmax={fp_lmax} lo={use_lo}: {dt:.1f}s, e_fermi={info.get('e_fermi')}")
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



if __name__ == "__main__":
    main()
