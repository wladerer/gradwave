# ruff: noqa: E402, I001, F401, B007, E501  # scratch probe script (hardcoded paths / sys.path insert)
"""Generalized Elk STATE.OUT on-site EFG + in-sphere-charge decomposition.

Ports the validated ``elk_onsite_corundum.py`` reader (validated vs EFG.OUT on TiO2/corundum)
to an arbitrary cell. Record order follows ``writestate.f90``:
  0 version, 1 spinpol, 2 nspecies, 3 lmmaxo, 4 nrmtmax, 5 nrcmtmax,
  then per species 5 records (natoms, nrmt, rsp, nrcmt, rcmt),
  then 11 scalar records (ngridg..dlefe),
  then the rhomt+rhoir record, then the vclmt+vclir record.
So the rho record index = 6 + 5*nspecies + 11 = 17 + 5*nspecies; vcl = that + 1.

For each requested global atom index (ias) it reports:
  * on-site  = interior l=2 Poisson r^2-coeff of rhomt        (the on-site EFG term)
  * full     = l=2 r^2-coeff of vclmt near origin (validates vs EFG.OUT)
  * boundary = full - on-site
  * q_l0     = sqrt(4pi) * int rho_00 r^2 dr  = in-sphere l=0 charge (vs INFO.OUT MT charge)

Usage: python elk_onsite.py STATE.OUT "Li:3,N:1" [ias_list e.g. 0,1]
"""
import struct
import sys

import numpy as np
from scipy.special import sph_harm_y

from gradwave.flapw.efg import _tensor_from_v  # noqa: E402

HA_EV = 97.173654


def records(fh):
    while True:
        head = fh.read(4)
        if len(head) < 4:
            return
        (n,) = struct.unpack("<i", head)
        data = fh.read(n)
        fh.read(4)
        yield data


def load(path):
    with open(path, "rb") as fh:
        recs = list(records(fh))
    nspecies = struct.unpack("<i", recs[2])[0]
    lmmaxo = struct.unpack("<i", recs[3])[0]
    nrmtmax = struct.unpack("<i", recs[4])[0]
    natoms, nrmt, rsp = [], [], []
    base = 6
    for s in range(nspecies):
        natoms.append(struct.unpack("<i", recs[base + 5 * s])[0])
        nrmt.append(struct.unpack("<i", recs[base + 5 * s + 1])[0])
        rsp.append(np.frombuffer(recs[base + 5 * s + 2], "<f8"))
    natmtot = sum(natoms)
    ias_sp = []
    for s in range(nspecies):
        ias_sp += [s] * natoms[s]
    rho_idx = 17 + 5 * nspecies
    vcl_idx = rho_idx + 1

    def unpack_fmt(rec):
        nrf = lmmaxo * nrmtmax * natmtot
        arr = np.frombuffer(rec[: nrf * 8], "<f8")
        return arr.reshape(natmtot, nrmtmax, lmmaxo)

    return dict(nspecies=nspecies, lmmaxo=lmmaxo, natoms=natoms, nrmt=nrmt, rsp=rsp,
                ias_sp=ias_sp, natmtot=natmtot,
                rho=unpack_fmt(recs[rho_idx]), vcl=unpack_fmt(recs[vcl_idx]))


def real_to_complex_l2():
    th = np.linspace(0.1, np.pi - 0.1, 40)
    ph = np.linspace(0.0, 2 * np.pi, 41)[:-1]
    T, P = np.meshgrid(th, ph, indexing="ij")
    Y = {M: sph_harm_y(2, M, T, P) for M in range(-2, 3)}
    R = {0: Y[0].real}
    for m in (1, 2):
        R[-m] = np.sqrt(2.0) * Y[-m].imag
        R[m] = np.sqrt(2.0) * Y[m].real
    Ystack = np.stack([Y[M].ravel() for M in range(-2, 3)], axis=1)
    Tmat = np.zeros((5, 5), dtype=complex)
    for i, m in enumerate(range(-2, 3)):
        coef, *_ = np.linalg.lstsq(Ystack, R[m].ravel(), rcond=None)
        Tmat[i] = coef
    return Tmat


def rcoeffs_l2(fmt_ias, nr):
    return [fmt_ias[:nr, 4 + i] for i in range(5)]


def onsite_v(rcoef, rr, Tmat):
    mom = np.array([np.trapezoid(rc / rr, rr) for rc in rcoef])
    v_real = (4 * np.pi / 5.0) * mom
    return {M: complex(sum(v_real[i] * Tmat[i, M + 2] for i in range(5))) for M in range(-2, 3)}


def full_v(rcoef, rr, Tmat):
    sl = slice(1, 9)
    c_real = np.array([np.polyfit(rr[sl] ** 2, rc[sl], 1)[0] for rc in rcoef])
    return {M: complex(sum(c_real[i] * Tmat[i, M + 2] for i in range(5))) for M in range(-2, 3)}


def maxabs(t):
    w = np.linalg.eigvalsh(t)
    return float(w[np.argmax(np.abs(w))])


def q_l0(fmt_ias, rr, nr):
    rho00 = fmt_ias[:nr, 0]
    return float(np.sqrt(4 * np.pi) * np.trapezoid(rho00 * rr ** 2, rr))


def main():
    path = sys.argv[1]
    spec = [(t.split(":")[0], int(t.split(":")[1])) for t in sys.argv[2].split(",")]
    ias_list = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    d = load(path)
    Tmat = real_to_complex_l2()
    # label per global atom
    labels = []
    for lab, nat in spec:
        labels += [lab] * nat
    if ias_list is None:
        ias_list, off = [], 0
        for lab, nat in spec:
            ias_list.append(off)
            off += nat
    print(f"# {path}  nspecies={d['nspecies']} natoms={d['natoms']} "
          f"lmmaxo={d['lmmaxo']} natmtot={d['natmtot']}", flush=True)
    for ias in ias_list:
        sp = d["ias_sp"][ias]
        nr = d["nrmt"][sp]
        rr = d["rsp"][sp][:nr]
        rc_rho = rcoeffs_l2(d["rho"][ias], nr)
        rc_vcl = rcoeffs_l2(d["vcl"][ias], nr)
        vf = full_v(rc_vcl, rr, Tmat)
        tf, vzzf, etaf = _tensor_from_v([vf[0], vf[1], vf[2]])
        vo = onsite_v(rc_rho, rr, Tmat)
        to, vzzo, etao = _tensor_from_v([vo[0], vo[1], vo[2]])
        tb = tf - to
        q0 = q_l0(d["rho"][ias], rr, nr)
        print(f"ias={ias} {labels[ias]}: full Vzz={vzzf*HA_EV:+.3f} (eta {etaf:.3f})  "
              f"onsite Vzz={vzzo*HA_EV:+.3f} (eta {etao:.3f})  "
              f"boundary Vzz={maxabs(tb)*HA_EV:+.3f}  q_l0(in-sphere)={q0:.4f} e", flush=True)


if __name__ == "__main__":
    main()
