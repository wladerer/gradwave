"""Dense-vs-iterative generalized-eigensolver crossover on REAL production pencils.

Captures the actual (H, S) secular matrices of two consecutive `_multi_iterate` calls on the
TiO2 config (by wrapping ``scf.solve_geneig``), then times, on iteration t's pencil:

    dense    solve_geneig (the production exact solve: full eigh(S) + MRRR subset)
    aug      solve_geneig_subspace_aug warm-started from iteration t-1's eigenvectors
    lobpcg   scipy.sparse.linalg.lobpcg on the pencil (B=S), X = t-1 eigenvectors
    eigsh    shift-invert Lanczos (dense LU of H - sigma*S as the operator)

plus each method's eigenvalue error vs dense over the occupied+pad window. The roadmap
question this answers: does an iterative pencil solve beat dense eigh at production sizes
(npw ~ a few hundred to ~2k)? Run at PB_ECUT=300 (production) and larger to find a crossover.

Env: PB_ECUT (300), PB_COLD=1 (capture from a cold MT iteration instead of the cached warm
fullpot state — the only option for ecut values with no cached state), PI_* as perf_iter.
"""
import os
import time

import numpy as np
from _common import A_BOHR, ATOMS, RADII, env_float, env_int
from perf_iter import cfg, load_state

import gradwave.flapw.scf as scf_mod
from gradwave.flapw.lapw import solve_geneig, solve_geneig_subspace_aug
from gradwave.flapw.scf import _multi_init_state, _multi_iterate, _multi_setup


def capture_two_iterations(c, warm):
    """(H1,S1,C1), (H2,S2) for the FIRST k of two consecutive iterations."""
    ctx = _multi_setup(a_bohr=A_BOHR, atoms=ATOMS, radii=RADII, **c)
    st = _multi_init_state(ctx, warm)
    if warm is None:
        st.mt_phase = True                              # plain cold MT capture
    grabbed = []                                        # one (H, S, ev, vecs) per iteration
    real = scf_mod.solve_geneig
    state = {"seen": 0}

    def wrapper(H, S, nbands, with_vecs=False, tol=1e-8):
        out = real(H, S, nbands, with_vecs=True, tol=tol)
        if state["seen"] == 0:                          # first k of this iteration only
            grabbed.append((H.copy(), S.copy(), out[0].copy(), out[1].copy()))
        state["seen"] += 1
        return out if with_vecs else out[0]

    scf_mod.solve_geneig = wrapper
    try:
        for it in range(2):
            state["seen"] = 0
            _multi_iterate(ctx, st, it, iters=200, tol=0.0)
    finally:
        scf_mod.solve_geneig = real
    return grabbed[0], grabbed[1], ctx.nbands


def timeit(fn, n=5):
    fn()                                                # warmup
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    c = cfg()
    c["ecut"] = env_float("PB", "ECUT", c["ecut"])
    c["kworkers"] = 1
    cold = env_int("PB", "COLD", 0)
    warm = None if cold else {"__full_state__": load_state(c)}
    (h1, s1, _e1, c1), (h2, s2, e2, _c2), nbands = capture_two_iterations(c, warm)
    dim = h2.shape[0]
    nb_solve = c1.shape[1]
    print(f"# pencil_bench ecut={c['ecut']} cold={cold} dim={dim} nbands={nbands} "
          f"nb_solve={nb_solve} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    ref = e2[:nbands]

    t_dense = timeit(lambda: solve_geneig(h2, s2, nb_solve, with_vecs=True))
    print(f"dense    {t_dense * 1e3:9.1f} ms   (reference)")

    ev_a, _v, resid = solve_geneig_subspace_aug(h2, s2, c1, nb_solve)
    t_aug = timeit(lambda: solve_geneig_subspace_aug(h2, s2, c1, nb_solve))
    print(f"aug-RR   {t_aug * 1e3:9.1f} ms   x{t_dense / t_aug:5.1f}  "
          f"max|dEv|={np.abs(ev_a[:nbands] - ref).max():.2e} eV  resid={resid:.1e}")

    from scipy.sparse.linalg import lobpcg
    x0 = np.ascontiguousarray(c1)
    try:
        def run_lobpcg():
            return lobpcg(h2, x0.copy(), B=s2, largest=False, tol=1e-6, maxiter=200)
        w, _ = run_lobpcg()
        t_lob = timeit(run_lobpcg, n=3)
        print(f"lobpcg   {t_lob * 1e3:9.1f} ms   x{t_dense / t_lob:5.1f}  "
              f"max|dEv|={np.abs(np.sort(w)[:nbands] - ref).max():.2e} eV")
    except Exception as exc:                            # ill-conditioned S can break LOBPCG
        print(f"lobpcg   FAILED: {type(exc).__name__}: {exc}")

    from scipy.sparse.linalg import eigsh
    sigma = float(e2[0]) - 1.0
    try:
        def run_eigsh():
            w2, _ = eigsh(h2, k=nb_solve, M=s2, sigma=sigma, which="LM", maxiter=500)
            return w2
        w2 = run_eigsh()
        t_si = timeit(run_eigsh, n=3)
        print(f"eigsh-SI {t_si * 1e3:9.1f} ms   x{t_dense / t_si:5.1f}  "
              f"max|dEv|={np.abs(np.sort(w2)[:nbands] - ref).max():.2e} eV")
    except Exception as exc:
        print(f"eigsh-SI FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
