"""H-B test: is the O EFG's lattice/boundary term under-resolved on the interstitial
FFT grid at the 0.024 A O|Ti near-touch?

The ONLY grid-dependent part of the EFG is the interstitial boundary (lattice) term
v_bc from _weinert_multi. Capture the converged rho_I + spheres from a 1-iter resume,
then rebuild v_grid/vbc_own at grid scale 1.0/1.5/2.0 (band-limited zero-pad upsample
of rho_I; pseudocharges regenerated analytically on the finer grid), and recompute the
full O/Ti EFG. Movement >=10% -> H-B live; flat -> H-B dead.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/home/wladerer/gw-efg/src")
sys.path.insert(0, "/home/wladerer/gw-efg/experiments/autoapw")
from _common import A_BOHR, ATOMS, RADII  # noqa: E402

import gradwave.flapw.scf as scfmod  # noqa: E402
from gradwave.constants import BOHR_ANG  # noqa: E402
from gradwave.flapw import crystal_scf_multi  # noqa: E402
from gradwave.flapw.coulomb import cell_matrix  # noqa: E402
from gradwave.flapw.efg import efg_tensor_full, interstitial_l2_boundary  # noqa: E402

VARIANTS = {"base": dict(lmax=3, los=None), "aug4": dict(lmax=4, los=None)}
TAG = os.environ.get("HB_TAG", "aug4")
STATE = f"/home/wladerer/tio2_states/{TAG}_k222.pkl"
cfg = dict(ecut=300.0, smearing=0.0, fullpot=True, fullpot_lmax=4, use_symmetry=True,
           subspace_reuse=False, kworkers=1, kerker=0.7, **VARIANTS[TAG])

cap = {}
_orig_w = scfmod._weinert_multi


def _wrap_w(rho_I, spheres, L, nfft):
    cap.update(rho_I=rho_I, spheres=spheres, L=L, nfft=nfft)
    return _orig_w(rho_I, spheres, L, nfft)


scfmod._weinert_multi = _wrap_w
state = pickle.load(open(STATE, "rb"))
crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                  v_start={"__full_state__": state}, kmesh=(2, 2, 2), **cfg)

rho_I, spheres, nfft0 = cap["rho_I"], cap["spheres"], cap["nfft"]
A = cell_matrix(np.asarray(A_BOHR) * BOHR_ANG)
print(f"base nfft={nfft0}  cell(A) diag~{np.diag(A)}")


def upsample(rho, n2):
    n = rho.shape[0]
    if n2 == n:
        return rho.astype(float).copy()
    F = np.fft.fftshift(np.fft.fftn(rho))
    p = (n2 - n) // 2
    Fp = np.pad(F, ((p, n2 - n - p),) * 3)
    return np.fft.ifftn(np.fft.ifftshift(Fp)).real * (n2 ** 3 / n ** 3)


def eigs(t):
    w = np.linalg.eigvalsh(t)
    return w[np.argsort(-np.abs(w))]


labels = {0: "Ti1", 2: "O1"}
for scale in (1.0, 1.5, 2.0):
    n2 = int(round(nfft0 * scale))
    n2 += (n2 - nfft0) % 2  # keep n2-nfft0 even
    rI2 = upsample(rho_I, n2)
    _, _, v_grid2, vbc_own2 = scfmod._weinert_multi(rI2, spheres, np.asarray(A_BOHR) * BOHR_ANG, n2)
    print(f"\n--- grid scale {scale} (nfft={n2}) ---  rho_I integral "
          f"{rI2.sum()*abs(np.linalg.det(A))/n2**3:.5f} (base "
          f"{rho_I.sum()*abs(np.linalg.det(A))/nfft0**3:.5f})")
    for i, sp in enumerate(spheres):
        if i not in labels:
            continue
        rr, dx, R = sp["rr"], sp["dx"], sp["R"]
        drw = rr * dx
        vbc = interstitial_l2_boundary(v_grid2, sp["tau"], R, A)
        own = vbc_own2[i] if vbc_own2 else {}
        vbc = {m: vbc[m] - own.get((2, m), 0.0) for m in range(-2, 3)}
        t, vzz, eta = efg_tensor_full(sp["rho_2m"], rr, drw, vbc, R)
        print(f"   {labels[i]}: eigs={eigs(t)}  Vzz={vzz:+.3f} eta={eta:.3f}")
