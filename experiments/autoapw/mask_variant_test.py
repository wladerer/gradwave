# ruff: noqa: E402, E501
"""Fast test on a pickled converged state: does masking rho_I to the interstitial
(Elk's variant B) fix ΔC0_ext without re-running the SCF? Also scans a finer Weinert grid.

Usage (asus): uv run python mask_variant_test.py /tmp/gw_state_rXXX.pkl <elk_dC0>"""
import pickle
import sys

import numpy as np

from gradwave.flapw.core_levels import onsite_madelung_potentials
from gradwave.flapw.coulomb import cell_matrix, _min_image_dist
from gradwave.flapw.scf import _weinert_multi


def inside_mask(rho_I, spheres, A, nfft):
    ainv = np.linalg.inv(cell_matrix(A))
    m = np.zeros((nfft, nfft, nfft), dtype=bool)
    for sp in spheres:
        cfrac = np.asarray(sp["tau"]) @ ainv
        m |= _min_image_dist(cfrac, nfft, A) < sp["R"]
    return m


def c0(v_hart, spheres, keys, A, short, long):
    v = onsite_madelung_potentials(v_hart, spheres, keys, A)
    return v[long] - v[short], v


def main():
    st = pickle.load(open(sys.argv[1], "rb"))
    elk = float(sys.argv[2]) if len(sys.argv) > 2 else None
    rho_I, spheres, A, nfft = st["rho_I"], st["spheres"], st["A"], st["nfft"]
    keys, short, long = st["keys"], st["short"], st["long"]

    # in-MT rho_I mass per sphere (the catastrophic-cancellation magnitude)
    m = inside_mask(rho_I, spheres, A, nfft)
    vol = float(abs(np.linalg.det(cell_matrix(A))))
    print(f"nfft={nfft}  rho_I total={rho_I.mean()*vol:.3f}  in-MT total={rho_I[m].sum()*vol/nfft**3:+.3f} e")

    # (A) current: unmasked
    _, _, _, vh_A, _ = _weinert_multi(rho_I, spheres, A, nfft)
    dA, _ = c0(vh_A, spheres, keys, A, short, long)
    # (B) masked to interstitial (variant B = Elk)
    rho_masked = np.where(m, 0.0, rho_I)
    _, _, _, vh_B, _ = _weinert_multi(rho_masked, spheres, A, nfft)
    dB, vB = c0(vh_B, spheres, keys, A, short, long)

    print(f"(A) UNMASKED  ΔC0_ext(long-short) = {dA:+.4f} eV")
    print(f"(B) MASKED    ΔC0_ext(long-short) = {dB:+.4f} eV" + (f"   [Elk={elk:+.3f}]" if elk else ""))
    print(f"    per-site C0_ext masked: " + "  ".join(f"{('Ti' if k==st['ti_key'] else k[:6])}={vB[k]:+.3f}" for k in keys))

    # grid-refinement scan (zero-pad rho_I in G-space -> finer real grid), masked & unmasked
    def upsample(rho, n, nf):
        g = np.fft.fftn(rho)
        gf = np.zeros((nf, nf, nf), dtype=complex)
        h = n // 2
        # copy the n^3 coefficients into the corners of the nf^3 box (fftfreq layout)
        idx = list(range(0, h + 1)) + list(range(nf - (n - h - 1), nf))
        src = list(range(0, h + 1)) + list(range(n - (n - h - 1), n))
        for i, si in zip(idx, src):
            for j, sj in zip(idx, src):
                gf[i, j][idx] = g[si, sj][src]
        return np.fft.ifftn(gf).real * (nf**3 / n**3)

    print("  grid scan (nfft: masked ΔC0_ext, unmasked ΔC0_ext):")
    for fac in (1, 1.5, 2, 3):
        nf = int(round(nfft * fac))
        nf += nf % 2
        rf = upsample(rho_I, nfft, nf) if nf != nfft else rho_I
        mf = inside_mask(rf, spheres, A, nf)
        _, _, _, vhu, _ = _weinert_multi(rf, spheres, A, nf)
        _, _, _, vhm, _ = _weinert_multi(np.where(mf, 0.0, rf), spheres, A, nf)
        du, _ = c0(vhu, spheres, keys, A, short, long)
        dm, _ = c0(vhm, spheres, keys, A, short, long)
        print(f"    nfft={nf:3d} (x{fac}): masked={dm:+.4f}  unmasked={du:+.4f}")


if __name__ == "__main__":
    main()
