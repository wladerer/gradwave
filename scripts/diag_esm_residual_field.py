"""Localize the ESM open_z residual vacuum field by component decomposition.

Builds a STRICTLY mirror-symmetric, neutral, non-polar Al(100) slab (even
layers, centered so rho(z) is symmetric about the box center), runs the open_z
SCF to convergence at two vacuum thicknesses Lz with a PINNED dz (identical grid
spacing across the two boxes), then dumps the in-plane-averaged z-profile of each
electrostatic component and reports its residual slope in the vacuum.

Components (all in-plane-averaged over a,b at each z):
  1. rho_tot(z) = rho_elec - gaussian_ion_density  (neutral total charge)
  2. v_H_ESM(z) = hartree_potential_esm(rho_tot)    (ESM-core open Hartree)
  3. v_loc(z)   = periodic reciprocal-space local pseudopotential
  4. v_eff(z)   = the converged SCF effective potential (the full thing)

For each: residual slope in the vacuum (meV/A), same-vs-opposite-sign on the two
opposite vacuum faces, and Lz-scaling.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.energies.esm import (
    _default_beta,
    esm_delta_potential,
    gaussian_ion_density,
    hartree_potential_esm,
)
from gradwave.core.energies.hartree import hartree_potential_r
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.grids import build_fft_grid, good_fft_size
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import local_potential_r, scf, setup_system

torch.set_num_threads(6)

RY = 13.605693122994  # eV per Ry


def pseudo(name):
    import os
    for root in ("tests/fixtures/qe/pseudos", "benchmarks/delta_gauge/pseudos"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def build_symmetric_al_slab(nz, dz, nlayers=4):
    """Mirror-symmetric Al(100) slab centered in a box of height Lz=nz*dz."""
    a0 = 4.05                      # Al lattice constant [A]
    a = a0 / np.sqrt(2.0)          # in-plane 1x1 (100) cell edge
    d = a0 / 2.0                   # (100) interlayer spacing
    Lz = nz * dz
    C = 0.5 * Lz                   # box center
    cell = np.diag([a, a, Lz])
    p0 = np.array([0.0, 0.0])
    p1 = np.array([0.5 * a, 0.5 * a])
    half = nlayers // 2
    zs_hi = [C + (k + 0.5) * d for k in range(half)]
    zs_lo = [C - (k + 0.5) * d for k in reversed(range(half))]
    zs = zs_lo + zs_hi
    xy = [p1 if (k % 2 == 1) else p0 for k in range(nlayers)]
    for k in range(nlayers):  # enforce palindrome in xy (mirror partner match)
        assert np.allclose(xy[k], xy[nlayers - 1 - k]), "xy not palindromic"
    pos = np.array([[xy[k][0], xy[k][1], zs[k]] for k in range(nlayers)])
    return cell, pos


def inplane_avg(field, open_axis=2):
    f = field.detach().cpu().numpy()
    axes = tuple(a for a in range(3) if a != open_axis)
    return f.mean(axis=axes)


def slope_meV_per_A(z, prof, mask):
    zz, pp = z[mask], prof[mask]
    A = np.vstack([zz, np.ones_like(zz)]).T
    m, _ = np.linalg.lstsq(A, pp, rcond=None)[0]
    return m * 1000.0  # eV/A -> meV/A


def analyze(name, z, prof, vac_lo, vac_hi, dz, Lz, margin=2.0):
    lo_mask = (z >= margin) & (z <= vac_lo)
    hi_mask = (z >= vac_hi) & (z <= Lz - margin)
    m_lo = slope_meV_per_A(z, prof, lo_mask) if lo_mask.sum() > 3 else float("nan")
    m_hi = slope_meV_per_A(z, prof, hi_mask) if hi_mask.sum() > 3 else float("nan")
    same = ("SAME-sign(uniform/G|=0)" if (m_lo * m_hi > 0)
            else "OPPOSITE-sign(dipole/symmetric)")
    print(f"  [{name:9s}] left = {m_lo:+8.3f}  right = {m_hi:+8.3f} meV/A "
          f"({lo_mask.sum()}/{hi_mask.sum()} pts) -> {same}")
    return m_lo, m_hi


def run_one(nz, dz, ecut_ry, nx, ny):
    cell, pos = build_symmetric_al_slab(nz, dz)
    Lz = nz * dz
    al = parse_upf(pseudo("Al_ONCV_PBE-1.2.upf"))
    sys_o = setup_system(cell, pos, [0] * len(pos), [al], ecut=ecut_ry * RY,
                         fft_shape=[nx, ny, nz])
    common = dict(smearing="gaussian", width=0.2, etol=1e-7, rhotol=1e-6,
                  max_iter=200, verbose=False)
    res = scf(sys_o, LDA_PW92(), boundary="open_z", **common)
    print(f"\n=== Lz={Lz:.3f} A  shape=({nx},{ny},{nz})  dz={dz:.4f} A  "
          f"converged={res.converged} niter={res.n_iter} ===")

    grid = sys_o.grid
    beta = _default_beta(grid, 2)
    rho_elec = res.rho
    rho_ion = gaussian_ion_density(sys_o.positions, sys_o.charges, grid, beta)
    rho_tot = rho_elec - rho_ion

    z = np.arange(nz) * dz
    area = float(grid.volume) / Lz
    rt_z = inplane_avg(rho_tot) * area          # linear density along z [e/A]
    re_z = inplane_avg(rho_elec) * area
    net_charge = rt_z.sum() * dz
    dipole = (z * rt_z).sum() * dz
    rt_rev = rt_z[::-1]
    sym_err = np.max(np.abs(rt_z - rt_rev)) / (np.max(np.abs(rt_z)) + 1e-30)
    print(f"  rho_tot: net charge = {net_charge:+.3e} e, dipole = "
          f"{dipole:+.3e} e*A, mirror-asym = {sym_err:.3e} (0=perfect), "
          f"beta={beta:.3f} A")

    thr = 1e-3 * np.max(np.abs(re_z))
    slab_idx = np.where(np.abs(re_z) > thr)[0]
    z_slab_lo, z_slab_hi = z[slab_idx[0]], z[slab_idx[-1]]
    vac_lo, vac_hi = z_slab_lo - 1.0, z_slab_hi + 1.0
    print(f"  slab z in [{z_slab_lo:.2f}, {z_slab_hi:.2f}] A; "
          f"left vac < {vac_lo:.2f}, right vac > {vac_hi:.2f}")

    v_h_esm = inplane_avg(hartree_potential_esm(rho_tot, grid.cell, 2))
    v_loc = inplane_avg(local_potential_r(sys_o))
    v_eff = inplane_avg(res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0])
    v_dv_esm = inplane_avg(esm_delta_potential(rho_tot, grid.cell, 2))
    v_h_per = inplane_avg(hartree_potential_r(rho_elec, grid.g2))

    print("  --- residual vacuum slopes ---")
    slopes = {}
    slopes["rho_tot"] = analyze("rho_tot", z, rt_z, vac_lo, vac_hi, dz, Lz)
    slopes["v_H_ESM(rho_tot)"] = analyze("v_H_ESM", z, v_h_esm, vac_lo, vac_hi, dz, Lz)
    slopes["v_loc(periodic)"] = analyze("v_loc", z, v_loc, vac_lo, vac_hi, dz, Lz)
    slopes["v_H_periodic(rho_e)"] = analyze("v_H_per", z, v_h_per, vac_lo, vac_hi, dz, Lz)
    slopes["dV_esm(open-per)"] = analyze("dV_esm", z, v_dv_esm, vac_lo, vac_hi, dz, Lz)
    slopes["v_eff(full)"] = analyze("v_eff", z, v_eff, vac_lo, vac_hi, dz, Lz)

    return {
        "Lz": Lz, "dz": dz, "net_charge": net_charge, "dipole": dipole,
        "sym_err": sym_err, "slopes": slopes,
        "profiles": {"z": z, "rho_tot": rt_z, "rho_elec": re_z,
                     "v_H_ESM": v_h_esm, "v_loc": v_loc, "v_H_per": v_h_per,
                     "dV_esm": v_dv_esm, "v_eff": v_eff},
    }


def main():
    ecut = 20.0
    dz = 0.15
    cell_ref, _ = build_symmetric_al_slab(120, dz)
    g = build_fft_grid(cell_ref, ecut * RY, equal_dims=[(0, 1)])
    nx, ny = int(g.shape[0]), int(g.shape[1])
    print(f"pinned in-plane shape ({nx},{ny}); dz={dz} A fixed across Lz")

    nz_list = sorted({good_fft_size(int(round(L / dz))) for L in (18.0, 24.0, 30.0)})
    print("nz values:", nz_list, "-> Lz:", [round(nz * dz, 2) for nz in nz_list])

    results = []
    for nz in nz_list:
        try:
            results.append(run_one(nz, dz, ecut, nx, ny))
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  !! nz={nz} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n\n################ Lz-SCALING SUMMARY (left/right meV/A) ############")
    comps = ["rho_tot", "v_H_ESM(rho_tot)", "v_loc(periodic)",
             "v_H_periodic(rho_e)", "dV_esm(open-per)", "v_eff(full)"]
    print(f"{'component':22s} " + " ".join(f"{'Lz=%.1f' % r['Lz']:>17s}" for r in results))
    for c in comps:
        row = [f"{r['slopes'][c][0]:+.2f}/{r['slopes'][c][1]:+.2f}" for r in results]
        print(f"{c:22s} " + " ".join(f"{x:>17s}" for x in row))

    np.savez("/tmp/esm_diag_profiles.npz",
             **{f"Lz{r['Lz']:.1f}_{k}": v
                for r in results for k, v in r["profiles"].items()})
    print("\nprofiles saved to /tmp/esm_diag_profiles.npz")


if __name__ == "__main__":
    main()
