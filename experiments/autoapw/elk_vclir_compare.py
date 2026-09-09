# ruff: noqa: E402, E501
"""Stage 2a: direct gw v_hart vs Elk VCLIR interstitial-potential comparison.

Reads Elk's interstitial Coulomb potential (vclir) from STATE.OUT, samples both it and
gw's saved v_hart along the Ti->O_short (x) and Ti->O_long (y) axes, gauge-aligns by the
interstitial-mean, and prints the along-axis profiles + the l=0 surface value at each O.

Usage (asus): uv run python elk_vclir_compare.py <STATE.OUT> <gw_vhart_rXXX.npz> [ngspec e.g. auto]
"""
import struct
import sys

import numpy as np

from gradwave.constants import BOHR_ANG, HARTREE_EV

HA = HARTREE_EV


def records(fh):
    while True:
        head = fh.read(4)
        if len(head) < 4:
            return
        (n,) = struct.unpack("<i", head)
        data = fh.read(n)
        fh.read(4)
        yield data


def load_elk(path):
    with open(path, "rb") as fh:
        recs = list(records(fh))
    nspecies = struct.unpack("<i", recs[2])[0]
    lmmaxo = struct.unpack("<i", recs[3])[0]
    nrmtmax = struct.unpack("<i", recs[4])[0]
    natoms, nrmt = [], []
    base = 6
    for s in range(nspecies):
        natoms.append(struct.unpack("<i", recs[base + 5 * s])[0])
        nrmt.append(struct.unpack("<i", recs[base + 5 * s + 1])[0])
    natmtot = sum(natoms)
    # 11 scalar records start at base2; first is ngridg (integer(3))
    base2 = 6 + 5 * nspecies
    ngridg = np.frombuffer(recs[base2][:12], "<i4")
    ngtot = int(np.prod(ngridg))
    rho_idx = 17 + 5 * nspecies
    vcl_idx = rho_idx + 1
    nrf = lmmaxo * nrmtmax * natmtot
    vcl_rec = recs[vcl_idx]
    # muffin-tin part then interstitial (real-space, ngtot reals)
    tail = vcl_rec[nrf * 8:]
    have = len(tail) // 8
    vclir = np.frombuffer(tail[:ngtot * 8], "<f8")
    vclir_grid = vclir.reshape(tuple(ngridg), order="F")
    return dict(nspecies=nspecies, natoms=natoms, ngridg=tuple(int(x) for x in ngridg),
                ngtot=ngtot, tail_reals=have, vclir=vclir_grid)


def trilin(grid, frac):
    """Periodic trilinear interpolation of a real-space grid at fractional coords (0..1)."""
    n = np.array(grid.shape)
    x = (np.asarray(frac) % 1.0) * n
    i0 = np.floor(x).astype(int)
    f = x - i0
    i0 = i0 % n
    i1 = (i0 + 1) % n
    c = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2]))
                ix = i1[0] if dx else i0[0]
                iy = i1[1] if dy else i0[1]
                iz = i1[2] if dz else i0[2]
                c += w * grid[ix, iy, iz]
    return float(c)


def interstitial_mask_mean(grid, cell_ang, centers_R):
    """Mean of grid over points farther than R from every center (Cartesian, Ang)."""
    n = np.array(grid.shape)
    ax = [np.arange(n[i]) / n[i] for i in range(3)]
    F = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    cart = F @ cell_ang
    inside = np.zeros(len(cart), dtype=bool)
    for c, R in centers_R:
        d = cart - c
        d -= np.round(d @ np.linalg.inv(cell_ang)) @ cell_ang  # min image
        inside |= np.linalg.norm(d, axis=1) < R
    return float(grid.reshape(-1)[~inside].mean())


def main():
    state, npz = sys.argv[1], sys.argv[2]
    d = load_elk(state)
    g = np.load(npz)
    v_gw = g["v_hart"]           # eV, nfft grid, mean0 gauge
    cell = g["cell"]             # Ang (3,3)
    ti, o_s, o_l = g["ti"], g["o_short"], g["o_long"]
    R_ti, R_o = float(g["R_ti"]), float(g["R_o"])
    print(f"# Elk ngridg={d['ngridg']} ngtot={d['ngtot']} tail_reals={d['tail_reals']} "
          f"(match={d['tail_reals']>=d['ngtot']})")
    print(f"# gw nfft={v_gw.shape} cell(Ang)diag={np.diag(cell)}  R_ti={R_ti:.3f} R_o={R_o:.3f} Ang")

    ve = d["vclir"] * HA         # eV
    ainv = np.linalg.inv(cell)
    centers_R = [(ti, R_ti), (o_s, R_o), (o_l, R_o)]
    m_gw = interstitial_mask_mean(v_gw, cell, centers_R)
    m_el = interstitial_mask_mean(ve, cell, centers_R)
    print(f"# interstitial-mean gauge:  gw={m_gw:+.4f}  elk={m_el:+.4f} eV (subtracted below)")

    def samp(center_frac, grid, mean):
        return trilin(grid, center_frac) - mean

    # along-axis profiles: Ti -> O direction, in Bohr steps of true distance
    for name, oc in (("Ti->O_short(x)", o_s), ("Ti->O_long(y)", o_l)):
        axis = (oc - ti)
        L = np.linalg.norm(axis)
        u = axis / L
        print(f"\n## {name}  |Ti-O|={L/BOHR_ANG:.3f} Bohr")
        print(f"   {'d_fromTi(Bohr)':>14} {'gw(eV)':>10} {'elk(eV)':>10} {'gw-elk':>9}")
        for frac_d in np.linspace(0.25, 1.35, 12):
            pt = ti + u * (L * frac_d)
            fr = pt @ ainv
            vg = samp(fr, v_gw, m_gw)
            vl = samp(fr, ve, m_el)
            dist_from_ti = (L * frac_d) / BOHR_ANG
            inside = dist_from_ti < R_ti / BOHR_ANG or abs(L * frac_d - L) < R_o
            tag = " (in-MT)" if inside else ""
            print(f"   {dist_from_ti:>14.3f} {vg:>10.3f} {vl:>10.3f} {vg-vl:>9.3f}{tag}")

    # l=0 surface value at each O (mean over a sphere of radius R_o), gauge-aligned
    def surf_l0(center, R, grid, mean):
        th = np.linspace(0, np.pi, 24)
        ph = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        T, P = np.meshgrid(th, ph, indexing="ij")
        dirs = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
        w = (np.sin(T).reshape(-1))
        pts = center + R * dirs
        vals = np.array([trilin(grid, (p @ ainv)) for p in pts]) - mean
        return float(np.sum(vals * w) / np.sum(w))

    vs_gw, vl_gw = surf_l0(o_s, R_o, v_gw, m_gw), surf_l0(o_l, R_o, v_gw, m_gw)
    vs_el, vl_el = surf_l0(o_s, R_o, ve, m_el), surf_l0(o_l, R_o, ve, m_el)
    print(f"\n## l=0 surface value at O (gauge-aligned, eV):")
    print(f"   O_short:  gw={vs_gw:+.3f}  elk={vs_el:+.3f}")
    print(f"   O_long :  gw={vl_gw:+.3f}  elk={vl_el:+.3f}")
    print(f"   Δ(long-short):  gw={vl_gw-vs_gw:+.3f}  elk={vl_el-vs_el:+.3f}  (Elk-gw gap={(vl_el-vs_el)-(vl_gw-vs_gw):+.3f})")


if __name__ == "__main__":
    main()
