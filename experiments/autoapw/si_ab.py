"""A/B + certificate evidence for the OPT-IN shift-invert Lanczos secular solver
(``lapw.solve_geneig_shift_invert``) on REAL production TiO2 pencils.

Captures the actual (H, S) secular matrices of the FIRST k (= Γ under symmetry, where the rutile
multiplets live) of N consecutive ``_multi_iterate`` calls from a warm/cold state (same capture as
``pencil_bench``), then per pencil:

    dense    solve_geneig  (the production exact solve — parity reference + timing baseline)
    SI       solve_geneig_shift_invert warm-started from the previous pencil's eigenvalues (σ just
             below the window bottom), median-timed; reports max|Δε| over the occupied window, the
             two-shift inertia certificate outcome (engaged / fell-back), and the speedup.

Plus a deliberately mis-placed σ (mid-spectrum) asserting the certificate returns None (loud
fallback). Env: SI_ECUT (via cfg PI_ECUT), SI_N (iters, default 6), SI_COLD (1 → capture from a
cold MT staging iteration, for ecut values with no cached warm state).
"""
import os
import time

import numpy as np
from _common import A_BOHR, ATOMS, RADII, env_int
from perf_iter import cfg, load_state

import gradwave.flapw.scf as scf_mod
from gradwave.flapw.lapw import solve_geneig, solve_geneig_shift_invert
from gradwave.flapw.scf import _multi_init_state, _multi_iterate, _multi_setup


def capture_pencils(c, warm, n):
    """First (H, S, dense_evals) of the first k of each of ``n`` consecutive iterations."""
    # the capture wraps solve_geneig for the DENSE reference; the captured SCF must therefore run
    # all-dense (shift_invert=False) so every iteration's first-k pencil is grabbed — with the
    # "auto" default an above-crossover iteration would take the SI path and bypass the wrap.
    c = dict(c, shift_invert=False)
    ctx = _multi_setup(a_bohr=A_BOHR, atoms=ATOMS, radii=RADII, **c)
    st = _multi_init_state(ctx, warm)
    if warm is None:
        st.mt_phase = True
    grabbed = []
    real = scf_mod.solve_geneig
    seen = {"n": 0}

    def wrapper(hmat, smat, nbands, with_vecs=False, tol=1e-8):
        out = real(hmat, smat, nbands, with_vecs=True, tol=tol)
        if seen["n"] == 0:
            grabbed.append((hmat.copy(), smat.copy(), out[0].copy()))
        seen["n"] += 1
        return out if with_vecs else out[0]

    scf_mod.solve_geneig = wrapper
    try:
        for it in range(n):
            seen["n"] = 0
            _multi_iterate(ctx, st, it, iters=200, tol=0.0)
    finally:
        scf_mod.solve_geneig = real
    return grabbed, ctx.nbands


def _timeit(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _sigma(evp):
    return float(evp[0]) - max(1.0, 0.02 * float(evp[-1] - evp[0]))


def main():
    c = cfg()
    c.pop("verbose", None)
    c["kworkers"] = 1
    n = env_int("SI", "N", 6)
    cold = env_int("SI", "COLD", 0)
    if env_int("SI", "D5", 0):
        # the aug6 D5 basis: the second parity gate at a richer in-sphere basis (lmax=6), warm-
        # started from a saved aug6 state (cold aug6 diverges — NaN in the radial solve). Same
        # geometry/dim ~737, but the l≤6 augmentation + fullpot l=4 Hamiltonian and Γ multiplets.
        import pickle
        c["lmax"] = 6
        d5_state = os.path.expanduser(os.environ.get(
            "SI_D5_STATE", "~/tio2_states/d5_aug6_noLO_k222.pkl"))
        with open(d5_state, "rb") as f:
            warm = {"__full_state__": pickle.load(f)}
    else:
        warm = None if cold else {"__full_state__": load_state(c)}
    pencils, nbands = capture_pencils(c, warm, n)
    dim = pencils[0][0].shape[0]
    print(f"# si_ab ecut={c['ecut']} lmax={c['lmax']} fp={c['fullpot_lmax']} k={c['kmesh'][0]} "
          f"d5={env_int('SI', 'D5', 0)} cold={cold} dim={dim} nbands={nbands} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    worst = 0.0
    engaged = 0
    ratios = []
    prev_ev = None
    for i, (h, s, ev_dense) in enumerate(pencils):
        ref = ev_dense[:nbands]
        sigma = _sigma(prev_ev if prev_ev is not None else ev_dense)
        out = solve_geneig_shift_invert(h, s, nbands, sigma)
        if out is None:
            print(f"it={i} dim={h.shape[0]} SI=fell-back (certificate declined at σ={sigma:.3f})",
                  flush=True)
        else:
            ev = np.sort(out[0])
            d = float(np.abs(ev - ref).max())
            worst = max(worst, d)
            engaged += 1
            t_dense = _timeit(lambda h=h, s=s: solve_geneig(h, s, nbands, with_vecs=True))
            t_si = _timeit(lambda h=h, s=s, sg=sigma: solve_geneig_shift_invert(h, s, nbands, sg))
            ratios.append(t_dense / t_si)
            print(f"it={i} dim={h.shape[0]} max|Δε|={d:.2e} eV  dense {t_dense*1e3:.1f}ms "
                  f"SI {t_si*1e3:.1f}ms  x{t_dense/t_si:.2f}", flush=True)
        prev_ev = ev_dense
    # mis-placed σ (far ABOVE the occupied window → the deep bands are far from σ and get missed)
    # must be caught by the inertia certificate and DECLINED (None → the SCF's loud dense fallback)
    h0, s0, ev0 = pencils[0]
    bad_sigma = float(ev0[-1]) + 50.0
    bad = solve_geneig_shift_invert(h0, s0, nbands, sigma=bad_sigma)
    print(f"MISPLACED-σ certificate (σ={bad_sigma:.1f}, window top {ev0[nbands-1]:.1f}): "
          f"{'DECLINED (None)' if bad is None else 'ENGAGED (BAD!)'}", flush=True)
    print(f"SUMMARY dim={dim} engaged={engaged}/{len(pencils)} worst|Δε|={worst:.2e} eV "
          f"median_speedup={np.median(ratios) if ratios else float('nan'):.2f}x "
          f"({'PARITY-OK' if worst < 1e-6 else 'PARITY-FAIL'})", flush=True)


if __name__ == "__main__":
    main()
