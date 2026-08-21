"""Crossover-dim calibration for the auto-default shift-invert secular solve.

Sweeps a ladder of plane-wave cutoffs on the production rutile TiO2 cell to produce REAL secular
pencils across a range of dimensions, and at each dim times the two production paths on the FIRST
k-point (= Γ under symmetry, where the rutile multiplets live):

    dense    solve_geneig                (full eigh(S) + MRRR subset — the exact default)
    SI       solve_geneig_shift_invert   (LDL^H factor of H−σS reused as ARPACK OPinv + the
                                          two-shift Sylvester-inertia completeness certificate)

σ is warmed from the previous captured iteration's eigenvalues (the production placement, just
below the occupied window). Both are median-timed over several reps; parity is checked over the
occupied+pad window. The output is a dim → speedup table whose break-even fixes the crossover dim
D* baked into ``scf._SHIFT_INVERT_CROSSOVER_DIM`` (the "auto" policy engages SI only above it).

This measures the PRODUCTION SI path (certificate included), so the crossover it reports is the
honest one for the auto default — higher than a bare ``eigsh`` crossover, because the certificate
adds one extra inertia factorization per solve.

Env: SC_ECUTS (comma list of eV cutoffs; default a ladder spanning ~dim 200..2500),
     SC_LMAX (aug lmax, default 3 = production), SC_N (capture iterations, default 3),
     SC_REPS (timing reps, default 5).
"""
import os
import time

import numpy as np
from _common import A_BOHR, ATOMS, RADII

import gradwave.flapw.scf as scf_mod
from gradwave.flapw.lapw import solve_geneig, solve_geneig_shift_invert
from gradwave.flapw.scf import _multi_init_state, _multi_iterate, _multi_setup


def capture_pencils(ecut, lmax, n):
    """First (H, S, dense_evals) of the first k of each of ``n`` consecutive cold MT iterations."""
    cfg = dict(ecut=ecut, lmax=lmax, kmesh=(2, 2, 2), smearing=0.0, fullpot=False,
               use_symmetry=True, kworkers=1, subspace_reuse=False, shift_invert=False)
    ctx = _multi_setup(a_bohr=A_BOHR, atoms=ATOMS, radii=RADII, **cfg)
    st = _multi_init_state(ctx, None)
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


def _timeit(fn, reps):
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
    ecuts = [float(x) for x in os.environ.get(
        "SC_ECUTS", "60,100,150,210,290,390,510,660").split(",")]
    lmax = int(os.environ.get("SC_LMAX", "3"))
    n = int(os.environ.get("SC_N", "3"))
    reps = int(os.environ.get("SC_REPS", "5"))
    print(f"# si_crossover lmax={lmax} n={n} reps={reps} OMP={os.environ.get('OMP_NUM_THREADS')}",
          flush=True)
    print(f"{'ecut':>6} {'dim':>6} {'nbands':>6} {'dense_ms':>9} {'SI_ms':>9} {'speedup':>8} "
          f"{'engaged':>8} {'max|dEv|':>10}", flush=True)
    rows = []
    for ecut in ecuts:
        pencils, nbands = capture_pencils(ecut, lmax, n)
        dim = pencils[0][0].shape[0]
        # use the LAST captured pencil (warmest σ), σ from the previous iteration's eigenvalues
        h, s, ev_dense = pencils[-1]
        ref = ev_dense[:nbands]
        prev_ev = pencils[-2][2] if len(pencils) > 1 else ev_dense
        sigma = _sigma(prev_ev)
        out = solve_geneig_shift_invert(h, s, nbands, sigma)
        t_dense = _timeit(lambda h=h, s=s, nb=nbands: solve_geneig(h, s, nb, with_vecs=True), reps)
        if out is None:
            print(f"{ecut:>6.0f} {dim:>6} {nbands:>6} {t_dense*1e3:>9.1f} {'--':>9} "
                  f"{'--':>8} {'fellback':>8} {'--':>10}", flush=True)
            rows.append((dim, None))
            continue
        ev = np.sort(out[0])
        d = float(np.abs(ev - ref).max())
        t_si = _timeit(
            lambda h=h, s=s, sg=sigma, nb=nbands: solve_geneig_shift_invert(h, s, nb, sg), reps)
        spd = t_dense / t_si
        print(f"{ecut:>6.0f} {dim:>6} {nbands:>6} {t_dense*1e3:>9.1f} {t_si*1e3:>9.1f} "
              f"{spd:>8.2f} {'yes':>8} {d:>10.2e}", flush=True)
        rows.append((dim, spd))
    # crossover: smallest dim with speedup >= 1.0 (break-even) and >= 1.3 (solid-win margin)
    be = next((dim for dim, spd in rows if spd is not None and spd >= 1.0), None)
    solid = next((dim for dim, spd in rows if spd is not None and spd >= 1.3), None)
    print(f"CROSSOVER break-even(dim @ spd>=1.0)={be}  solid-win(dim @ spd>=1.3)={solid}",
          flush=True)


if __name__ == "__main__":
    main()
