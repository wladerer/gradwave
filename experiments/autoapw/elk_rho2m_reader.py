"""Read Elk STATE.OUT rhomt/vclmt l=2 muffin-tin components (read-only, tiny numpy).

Layout confirmed against writestate.f90 + a header probe:
  rec27 = rfmt(density) + rhoir ;  rec28 = rfmt(vclmt) + vclir
rfmt is Fortran (lmmaxo, nrmtmax, natmtot), real spherical harmonics, index
j=l(l+1)+m+1 (Elk genrlmv). Species order Ti(2 atoms) then O(4): ias 0,1=Ti; 2..5=O.

Self-validation: reconstruct the O1/Ti1 EFG tensor from vclmt's l=2 r^2-coefficient
and match EFG.OUT (a.u.). Then report the ON-SITE density EFG (interior l=2 Poisson
of rhomt) — directly comparable to gradwave's muffin-tin valence tensor — and dump
rho_2M(r) for O and Ti so the ratio profile vs gradwave can be formed.
"""
import struct
import sys

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "/home/wladerer/github/gradwave/src")
from gradwave.flapw.efg import _tensor_from_v  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/wladerer/tio2_efg/STATE.OUT"
HA_PER_B2_TO_EV_A2 = 97.173654  # 1 Ha/bohr^2 -> eV/Ang^2
BOHR = 0.52917721067


def records(fh):
    while True:
        head = fh.read(4)
        if len(head) < 4:
            return
        (n,) = struct.unpack("<i", head)
        data = fh.read(n)
        fh.read(4)
        yield data


def load():
    with open(PATH, "rb") as fh:
        recs = list(records(fh))
    # header
    lmmaxo = struct.unpack("<i", recs[3])[0]
    nrmtmax = struct.unpack("<i", recs[4])[0]
    # species blocks: Ti rec6..10, O rec11..15
    natoms = [struct.unpack("<i", recs[6])[0], struct.unpack("<i", recs[11])[0]]
    nrmt = [struct.unpack("<i", recs[7])[0], struct.unpack("<i", recs[12])[0]]
    rsp = [np.frombuffer(recs[8], "<f8"), np.frombuffer(recs[13], "<f8")]
    natmtot = sum(natoms)
    # ias -> species index
    ias_sp = [0] * natoms[0] + [1] * natoms[1]

    def unpack_fmt(rec):
        nrf = lmmaxo * nrmtmax * natmtot
        arr = np.frombuffer(rec[: nrf * 8], "<f8")
        # Fortran (lmmaxo, nrmtmax, natmtot) -> C reshape reversed
        return arr.reshape(natmtot, nrmtmax, lmmaxo)  # [ias, ir, lm]

    rho = unpack_fmt(recs[27])
    vcl = unpack_fmt(recs[28])
    return dict(lmmaxo=lmmaxo, nrmt=nrmt, rsp=rsp, ias_sp=ias_sp,
                natmtot=natmtot, rho=rho, vcl=vcl)


# real (Elk genrlmv) -> complex Y transform for l=2, built numerically
def real_to_complex_l2():
    th = np.linspace(0.1, np.pi - 0.1, 40)
    ph = np.linspace(0.0, 2 * np.pi, 41)[:-1]
    T, P = np.meshgrid(th, ph, indexing="ij")
    Y = {M: sph_harm_y(2, M, T, P) for M in range(-2, 3)}
    # Elk real harmonics R_{2,m}
    R = {}
    R[0] = Y[0].real
    for m in (1, 2):
        R[-m] = np.sqrt(2.0) * Y[-m].imag   # rlm(m<0) = sqrt2 * Im(Y_{2,-m})
        R[m] = np.sqrt(2.0) * Y[m].real     # rlm(m>0) = sqrt2 * Re(Y_{2,+m})
    # Solve R_m = sum_M Tmat[m,M] Y_M  (least squares over grid)
    Ystack = np.stack([Y[M].ravel() for M in range(-2, 3)], axis=1)  # (npts,5)
    Tmat = np.zeros((5, 5), dtype=complex)  # rows m=-2..2, cols M=-2..2
    for i, m in enumerate(range(-2, 3)):
        coef, *_ = np.linalg.lstsq(Ystack, R[m].ravel(), rcond=None)
        Tmat[i] = coef
    return Tmat  # v_M = sum_m cR_m * Tmat[m,M]


def rcoeffs_l2(fmt_ias, nr):
    """Return the 5 real l=2 radial coefficient arrays (m=-2..2) on the valid mesh."""
    # 0-based lm index = l(l+1)+m ; l=2 -> 4,5,6,7,8 for m=-2..2
    return [fmt_ias[:nr, 4 + i] for i in range(5)]  # each (nr,)


def onsite_v_from_rho(rcoef, rr):
    """Interior l=2 Poisson r^2-coefficient v_M (a.u.) from real-harmonic density coeffs.
    v_M = (4pi/5) \int rho_2M(r)/r dr, converted real->complex."""
    Tmat = real_to_complex_l2()
    # complex density r^2 moment per real m: mom_m = \int rho^R_m / r dr
    mom = np.array([np.trapz(rc / rr, rr) for rc in rcoef])  # real, len5
    v_real = (4 * np.pi / 5.0) * mom       # a.u., real-harmonic v_m
    v_cplx = {M: complex(sum(v_real[i] * Tmat[i, M + 2] for i in range(5)))
              for M in range(-2, 3)}
    return v_cplx


def tensor_from_vcl(rcoef, rr):
    """EFG tensor (a.u.) directly from vclmt's l=2 r^2-coefficient near the origin.
    vcl_2M(r) ~ c_2M r^2; fit c from the first few points. Real->complex, then _tensor_from_v."""
    Tmat = real_to_complex_l2()
    # fit each real coeff to a r^2 near origin (skip r=0). Use pts 1..8
    sl = slice(1, 9)
    c_real = np.array([np.polyfit(rr[sl] ** 2, rc[sl], 1)[0] for rc in rcoef])
    v_cplx = {M: complex(sum(c_real[i] * Tmat[i, M + 2] for i in range(5)))
              for M in range(-2, 3)}
    return v_cplx


def main():
    d = load()
    Tmat = real_to_complex_l2()
    np.set_printoptions(precision=5, suppress=True)
    print("real->complex l=2 |T| row-sums:", np.abs(Tmat).sum(1))
    for label, ias in (("Ti1", 0), ("O1", 2)):
        sp = d["ias_sp"][ias]
        nr = d["nrmt"][sp]
        rr = d["rsp"][sp][:nr]
        rc_rho = rcoeffs_l2(d["rho"][ias], nr)
        rc_vcl = rcoeffs_l2(d["vcl"][ias], nr)
        # --- validation: EFG from vclmt vs EFG.OUT ---
        v_vcl = tensor_from_vcl(rc_vcl, rr)
        # _tensor_from_v expects list [v0,v1,v2] complex with v_-m=(-1)^m conj(v_m)
        vlist = [v_vcl[0], v_vcl[1], v_vcl[2]]
        tens, vzz, eta = _tensor_from_v(vlist)
        print(f"\n=== {label} ===")
        print(f"EFG from vclmt (a.u.): Vzz={vzz:.5f} eta={eta:.4f}")
        print("  tensor a.u.:\n", tens)
        # --- on-site density EFG ---
        v_on = onsite_v_from_rho(rc_rho, rr)
        t_on, vzz_on, eta_on = _tensor_from_v([v_on[0], v_on[1], v_on[2]])
        print(f"on-site rhomt EFG (a.u.): Vzz={vzz_on:.5f} eta={eta_on:.4f}  "
              f"-> eV/A^2 Vzz={vzz_on*HA_PER_B2_TO_EV_A2:.3f}")
        print("  on-site tensor eV/A^2:\n", t_on * HA_PER_B2_TO_EV_A2)
        # --- dump rho_2M(r) profile (real m=0 and m=+2 dominant) at sample radii ---
        rc0, rc2p = rc_rho[2], rc_rho[4]  # real m=0, m=+2
        print("  r(bohr)   rho^R_{2,0}     rho^R_{2,+2}")
        for frac in (0.2, 0.4, 0.6, 0.8, 0.95):
            idx = int(frac * (nr - 1))
            print(f"   {rr[idx]:.4f}   {rc0[idx]:+.6e}   {rc2p[idx]:+.6e}")
        # save full profile for ratio-vs-gradwave
        np.savez(f"/home/wladerer/elk_rho2m_{label}.npz", rr_bohr=rr,
                 rho_real_m=np.stack(rc_rho), vzz_onsite_ev=vzz_on * HA_PER_B2_TO_EV_A2)


if __name__ == "__main__":
    main()
