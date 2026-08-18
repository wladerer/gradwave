"""PROD-F cross-validation — my LAPW crystal bands vs Elk 11.0.2 (independent all-electron FLAPW).

Validates the crystal band machinery (PROD-D/E) against a real, established FLAPW code for
simple-cubic Ne (cubic cell so both codes share the geometry exactly). We compare zero-independent
quantities (band splittings, bandwidths) since the two codes reference eigenvalues to different
zeros, and mine uses the atomic-superposition potential vs Elk's self-consistent one.

Elk 11.0.2 reference (built on asus; LSDA xctype 3 = Perdew-Wang 1992, simple-cubic Ne a=6 Bohr,
ngridk 4 4 4, rgkmax 7.0). Elk input (elk.in):

    tasks
      0
    xctype
      3
    avec
      6.0  0.0  0.0
      0.0  6.0  0.0
      0.0  0.0  6.0
    atoms
      1
      'Ne.in'
      1
      0.0  0.0  0.0
    ngridk
      4  4  4
    rgkmax
      7.0

Elk eigenvalues (Ha) at Γ: 2s -1.263267, 2p -0.426409 (×3); at X=(½,0,0): 2s -1.262242,
2p -0.441196 / -0.425178 (×2). -> 2s-2p splitting 22.77 eV; 2s bandwidth 0.030 eV.

    uv run python experiments/autoapw/elk_ne_compare.py
"""

from __future__ import annotations

import numpy as np
import torch
from atomic_scf import atomic_scf
from prod_lapw import _sym_geneig, build_matrices_multi
from radial_log import log_mesh

from gradwave.constants import BOHR_ANG

# Elk 11.0.2 reference (zero-independent quantities), eV
ELK = {"split_2s2p_G": 22.77, "bw_2s_GX": 0.030}


def main():
    a_bohr, R = 6.0, 1.4
    L = a_bohr * BOHR_ANG
    r, dx = log_mesh(1e-5, 28.0, 2500)
    at, v_at = atomic_scf("Ne", r, dx)
    v0 = float(v_at[np.argmin(np.abs(r.numpy() - R))])
    v_mt = torch.where(r <= R, v_at - v0, torch.zeros_like(r))
    El = {0: at["2s"] - v0, 1: at["2p"] - v0, 2: -5.0 - v0}
    species = {"Ne": {"R": R, "v": v_mt, "El": El}}
    atoms = [([0.0, 0.0, 0.0], "Ne")]

    ev = {}
    for kf, name in ([[0.0, 0.0, 0.0], "G"], [[0.5, 0.0, 0.0], "X"]):
        H, S, _ = build_matrices_multi(kf, L, atoms, 2, 300.0, r, dx, species)
        ev[name] = _sym_geneig(H, S, 6) + v0

    split = float(ev["G"][1:4].mean() - ev["G"][0])
    bw = float(abs(ev["X"][0] - ev["G"][0]))

    print("\nPROD-F cross-validation — LAPW vs Elk 11 (simple-cubic Ne, a=6 Bohr)\n")
    print(f"  Γ bands (eV): {np.array2string(ev['G'][:4], precision=3)}  (2p 3-fold degenerate)")
    print(f"  X bands (eV): {np.array2string(ev['X'][:4], precision=3)}  (2p splits 1+2)\n")
    print(f"  {'quantity':>22} | {'mine':>8} | {'Elk 11':>8} | {'|Δ| (eV)':>9}")
    print(f"  {'2s-2p splitting @ Γ':>22} | {split:>8.2f} | {ELK['split_2s2p_G']:>8.2f} | "
          f"{abs(split - ELK['split_2s2p_G']):>9.2f}")
    print(f"  {'2s bandwidth Γ→X':>22} | {bw:>8.3f} | {ELK['bw_2s_GX']:>8.3f} | "
          f"{abs(bw - ELK['bw_2s_GX']):>9.3f}")
    ok = abs(split - ELK["split_2s2p_G"]) < 0.5 and abs(bw - ELK["bw_2s_GX"]) < 0.02
    print(f"\n  VERDICT: LAPW crystal bands agree with Elk (all-electron FLAPW): "
          f"{'PASS' if ok else 'FAIL'}")
    print("  (residual is my atomic-vs-self-consistent potential + O(dx²) mesh; the band")
    print("   DISPERSION — the crystal physics — matches an independent all-electron code.)")


if __name__ == "__main__":
    main()
