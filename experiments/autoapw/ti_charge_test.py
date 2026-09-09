# ruff: noqa: E402, E501
"""Attribute the residual masked-Weinert gap: if gw's Ti in-sphere charge is scaled to
Elk's (18.67 / 18.75 e), does masked ΔC0_ext jump to Elk's +2.90 / +4.79?

Usage: uv run python ti_charge_test.py /tmp/gw_state_rXXX.pkl <elk_Ti_q> <elk_dC0>"""
import copy
import math
import pickle
import sys

import numpy as np

from gradwave.flapw.core_levels import onsite_madelung_potentials
from gradwave.flapw.coulomb import cell_matrix, _min_image_dist
from gradwave.flapw.scf import _weinert_multi


def imask(rho_I, spheres, A, nfft):
    ainv = np.linalg.inv(cell_matrix(A))
    m = np.zeros((nfft, nfft, nfft), dtype=bool)
    for sp in spheres:
        m |= _min_image_dist(np.asarray(sp["tau"]) @ ainv, nfft, A) < sp["R"]
    return m


def dc0(rho_I, spheres, A, nfft, keys, short, long, mask=True):
    ri = np.where(imask(rho_I, spheres, A, nfft), 0.0, rho_I) if mask else rho_I
    _, _, _, vh, _ = _weinert_multi(ri, spheres, A, nfft)
    v = onsite_madelung_potentials(vh, spheres, keys, A)
    return v[long] - v[short], v


def q_of(sp):
    rr = np.asarray(sp["rr"], float)
    drw = rr * float(sp["dx"])
    return float(np.sum(4 * math.pi * np.asarray(sp["rho_sph"], float) * rr**2 * drw))


def main():
    st = pickle.load(open(sys.argv[1], "rb"))
    elk_ti = float(sys.argv[2])
    elk_dc0 = float(sys.argv[3])
    rho_I, spheres, A, nfft = st["rho_I"], st["spheres"], st["A"], st["nfft"]
    keys, short, long, ti_key = st["keys"], st["short"], st["long"], st["ti_key"]
    ti_i = keys.index(ti_key)
    gw_ti = q_of(spheres[ti_i])
    print(f"gw Ti q_sph={gw_ti:.4f}  elk Ti q_sph={elk_ti:.4f}  (Δ={gw_ti-elk_ti:+.4f} e)")

    d_orig, _ = dc0(rho_I, spheres, A, nfft, keys, short, long, mask=True)
    print(f"masked, gw Ti charge:        ΔC0_ext = {d_orig:+.4f}")

    sp2 = copy.deepcopy(spheres)
    sp2[ti_i]["rho_sph"] = np.asarray(sp2[ti_i]["rho_sph"], float) * (elk_ti / gw_ti)
    d_ti, _ = dc0(rho_I, sp2, A, nfft, keys, short, long, mask=True)
    print(f"masked, Ti scaled to Elk:    ΔC0_ext = {d_ti:+.4f}   [Elk={elk_dc0:+.3f}]")
    print(f"  Ti-charge contribution to ΔC0_ext = {d_ti-d_orig:+.4f} eV")


if __name__ == "__main__":
    main()
