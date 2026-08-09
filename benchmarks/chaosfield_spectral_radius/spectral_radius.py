"""ChaosField kill test — does the async-critical screening eigenvalue exceed 1
and grow with cell size on a metal?

Async (chaotic-relaxation) convergence of the SCF fixed point needs the spectral
radius of the fixed-point Jacobian M = K_Hxc·chi0 to satisfy rho(|M|) < 1 WITHOUT
the nonlocal Kerker preconditioner (a real-space actor field cannot apply the
long-range Kerker term locally). So we measure, on a simple metal (Al fcc) at
growing supercell size:

  rho(M)      = dominant_screening_eigenvalue  (spectral radius, the async gate)
  lam_max^re  = max_real_screening_eigenvalue  (soft-mode/instability margin 1-lam)

Prediction (plan): rho(M) > 1 already, and grows with box length L (the small-G
charge-sloshing mode ~ 1/G_min ~ L). If so, an unpreconditioned async field
diverges -> ChaosField's blocker is real.

Si (insulator) is included as a contrast at one size.
"""
import time
import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.soft_mode import (
    dominant_screening_eigenvalue,
    max_real_screening_eigenvalue,
)

torch.set_num_threads(8)
torch.manual_seed(0)

PSD = "tests/fixtures/qe/pseudos"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
ECUT = 16 * RY

def fcc_metal_super(a, n):
    """n×n×n supercell of the 1-atom fcc primitive → n^3 atoms."""
    prim = a / 2 * FCC
    cell = prim * n
    pos = np.array([i * prim[0] + j * prim[1] + k * prim[2]
                    for i in range(n) for j in range(n) for k in range(n)])
    return cell, pos

def diamond_super(a, n):
    """n×n×n supercell of the 2-atom diamond primitive → 2 n^3 atoms."""
    prim = a / 2 * FCC
    basis = np.array([[0.0, 0, 0], [a / 4] * 3])
    cell = prim * n
    pos = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = i * prim[0] + j * prim[1] + k * prim[2]
                pos += [shift + b for b in basis]
    return cell, np.array(pos)

def run_one(tag, cell, pos, upf, kmesh, metal):
    na = len(pos)
    nel_val = int(round(na * upf.z_valence))  # real valence (Al ONCV = 11, Si = 4)
    nbands = int(nel_val * 0.55) + 16  # ~occ (nel/2) + empty buffer for smearing
    L = float(np.linalg.norm(cell, axis=1).max())  # longest box length (sets smallest |G|)
    system = setup_system(cell, pos, [0] * na, [upf], ecut=ECUT, kmesh=kmesh,
                          nbands=nbands)
    xc = PBE()
    t0 = time.time()
    if metal:
        res = scf(system, xc, smearing="fermi-dirac", width=0.2, etol=1e-9, rhotol=1e-8,
                  max_iter=200, verbose=False)
    else:
        res = scf(system, xc, smearing="none", etol=1e-10, rhotol=1e-9,
                  max_iter=200, verbose=False)
    tscf = time.time() - t0
    conv = bool(res.converged)
    t0 = time.time()
    dom = dominant_screening_eigenvalue(res, xc, n_iter=30, tol=1e-3,
                                        chi0_tol=1e-5, chi0_max_iter=60)
    lam = max_real_screening_eigenvalue(res, xc, n_iter=40, tol=1e-3,
                                        chi0_tol=1e-5, chi0_max_iter=60)
    teig = time.time() - t0
    print(f"[{tag}] na={na:3d} L={L:5.2f}Å kmesh={kmesh} conv={conv} "
          f"nband={nbands} | rho(M)=|{dom.eigenvalue:+.3f}| (res={dom.residual:.1e}) "
          f"lam_max^re={lam.eigenvalue:+.3f} margin(1-lam)={1 - lam.eigenvalue:+.3f} "
          f"| t_scf={tscf:.0f}s t_eig={teig:.0f}s", flush=True)
    return dict(tag=tag, na=na, L=L, conv=conv, rho=abs(dom.eigenvalue),
                rho_res=dom.residual, lam_max=lam.eigenvalue)

def main():
    al = parse_upf(f"{PSD}/Al_ONCV_PBE-1.2.upf")
    si = parse_upf(f"{PSD}/Si_ONCV_PBE-1.2.upf")
    print("=== ChaosField: async-critical spectral radius rho(M=K_Hxc·chi0) vs box size ===", flush=True)
    rows = []
    # Al metal supercells: L grows as n → the small-G charge mode should grow.
    for n, km in [(1, (4, 4, 4)), (2, (2, 2, 2))]:
        cell, pos = fcc_metal_super(4.05, n)
        rows.append(run_one(f"Al n{n}", cell, pos, al, km, metal=True))
    # Si insulator contrast (one size).
    cell, pos = diamond_super(5.43, 1)
    rows.append(run_one("Si n1", cell, pos, si, (4, 4, 4), metal=False))

    print("\n=== VERDICT ===", flush=True)
    al_rows = [r for r in rows if r["tag"].startswith("Al") and r["conv"]]
    if len(al_rows) >= 2:
        al_rows.sort(key=lambda r: r["L"])
        grew = al_rows[-1]["rho"] > al_rows[0]["rho"]
        # Async needs rho(|M|) < 1. The blocker is confirmed when rho crosses 1 and
        # grows with box size — the tiny-box point may sit below 1 (weak charge mode)
        # without being a counterexample; what matters is the largest physical cell.
        crosses = al_rows[-1]["rho"] > 1.0
        print(f"Al rho(M): " + " -> ".join(f"{r['rho']:.2f}@L{r['L']:.1f}" for r in al_rows))
        print(f"  rho(M) grows with box length L: {grew}")
        print(f"  rho(M) > 1 at the largest cell: {crosses}")
        print("  => ChaosField blocker CONFIRMED (unpreconditioned async diverges, worsens with size)"
              if (crosses and grew) else
              "  => ChaosField blocker NOT confirmed by this data (revisit)")

if __name__ == "__main__":
    main()
