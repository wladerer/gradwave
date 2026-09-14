"""Decisive ESM localization: does the vacuum field depend on the slab's
POSITION in the box? ESM open_z is translation-covariant in exact arithmetic, so
shifting a fixed neutral slab within the box must NOT change the vacuum field. If
it does, and the field is SAME-sign on both faces (a uniform field), the residual
is the G|=0 open-Coulomb channel in hartree_potential_esm / the ESM correction.

Runs the SAME mirror-symmetric neutral Al slab at three z-offsets in a fixed
Lz=24 box (pinned dz) and decomposes the in-plane-averaged vacuum profile into
rho_tot, v_H_ESM (ESM core), v_loc+v_H_periodic (periodic pair), dV_esm (the SCF
correction), and v_eff. Reports the SAME-vs-OPPOSITE-sign face slopes for each.
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
from gradwave.grids import build_fft_grid
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import local_potential_r, scf, setup_system

torch.set_num_threads(6)
RY = 13.605693122994


def pseudo(name):
    import os
    for root in ("tests/fixtures/qe/pseudos", "benchmarks/delta_gauge/pseudos"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def slab(nz, dz, shift, nlayers=4):
    a0 = 4.05
    a = a0 / np.sqrt(2.0)
    d = a0 / 2.0
    Lz = nz * dz
    C = 0.5 * Lz + shift
    half = nlayers // 2
    zs = [C - (k + 0.5) * d for k in reversed(range(half))] + \
         [C + (k + 0.5) * d for k in range(half)]
    p0, p1 = np.array([0.0, 0.0]), np.array([0.5 * a, 0.5 * a])
    xh = [p0 if k % 2 == 0 else p1 for k in range(half)]
    xy = xh + xh[::-1]
    pos = np.array([[xy[k][0], xy[k][1], zs[k]] for k in range(nlayers)])
    return np.diag([a, a, Lz]), pos


def ipa(f, ax=2):
    f = f.detach().cpu().numpy()
    return f.mean(axis=tuple(a for a in range(3) if a != ax))


def face_slopes(z, prof, zc_lo, zc_hi, Lz, w=1.5, margin=0.5):
    """Slope [meV/A] in the deep vacuum near each box edge, avoiding the slab."""
    lo = (z >= margin) & (z <= min(zc_lo - 1.0, margin + w))
    hi = (z >= max(zc_hi + 1.0, Lz - margin - w)) & (z <= Lz - margin)
    def s(m):
        if m.sum() < 3:
            return float("nan")
        A = np.vstack([z[m], np.ones(m.sum())]).T
        return np.linalg.lstsq(A, prof[m], rcond=None)[0][0] * 1000
    return s(lo), s(hi)


def run(nz, dz, shift, nx, ny):
    cell, pos = slab(nz, dz, shift)
    Lz = nz * dz
    al = parse_upf(pseudo("Al_ONCV_PBE-1.2.upf"))
    s = setup_system(cell, pos, [0] * len(pos), [al], ecut=20 * RY, fft_shape=[nx, ny, nz])
    res = scf(s, LDA_PW92(), boundary="open_z", smearing="gaussian", width=0.2,
              etol=1e-7, rhotol=1e-6, max_iter=200, verbose=False)
    grid = s.grid
    z = np.arange(nz) * dz
    beta = _default_beta(grid, 2)
    rho_e = res.rho
    rho_tot = rho_e - gaussian_ion_density(s.positions, s.charges, grid, beta)
    area = float(grid.volume) / Lz
    re_z = ipa(rho_e) * area
    thr = 1e-3 * np.max(np.abs(re_z))
    idx = np.where(np.abs(re_z) > thr)[0]
    zc_lo, zc_hi = z[idx[0]], z[idx[-1]]
    print(f"\n=== shift={shift:+.2f} A  slab z in [{zc_lo:.2f},{zc_hi:.2f}]  "
          f"conv={res.converged} nit={res.n_iter} ===")

    comps = {
        "rho_tot": ipa(rho_tot) * area,
        "v_H_ESM": ipa(hartree_potential_esm(rho_tot, grid.cell, 2)),
        "v_loc": ipa(local_potential_r(s)),
        "v_H_per": ipa(hartree_potential_r(rho_e, grid.g2)),
        "dV_esm": ipa(esm_delta_potential(rho_tot, grid.cell, 2)),
        "v_eff": ipa(res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]),
    }
    comps["vloc+vHper"] = comps["v_loc"] + comps["v_H_per"]
    out = {}
    for name in ("rho_tot", "v_H_ESM", "vloc+vHper", "dV_esm", "v_eff"):
        lo, hi = face_slopes(z, comps[name], zc_lo, zc_hi, Lz)
        tag = "SAME(uniform/G|=0)" if lo * hi > 0 else "OPP(dipole/sym)"
        print(f"  [{name:11s}] Lface={lo:+9.3f}  Rface={hi:+9.3f} meV/A  {tag}")
        out[name] = (lo, hi)
    return out


def main():
    dz = 0.15
    nz = 160  # Lz = 24
    cref, _ = slab(nz, dz, 0.0)
    g = build_fft_grid(cref, 20 * RY, equal_dims=[(0, 1)])
    nx, ny = int(g.shape[0]), int(g.shape[1])
    print(f"pinned ({nx},{ny},{nz}) Lz={nz*dz} dz={dz}")
    tab = {}
    for sh in (0.0, 2.0, 4.0):
        tab[sh] = run(nz, dz, sh, nx, ny)
    print("\n### position (in-box) dependence of vacuum face slopes [meV/A] ###")
    print(f"{'component':12s}" + "".join(f"  shift={s:+.1f}(L/R)" for s in tab))
    for c in ("rho_tot", "v_H_ESM", "vloc+vHper", "dV_esm", "v_eff"):
        print(f"{c:12s}" + "".join(f"  {tab[s][c][0]:+7.2f}/{tab[s][c][1]:+7.2f}" for s in tab))


if __name__ == "__main__":
    main()
