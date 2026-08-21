"""Newton-F determinism gate for the auto-default shift-invert secular solve.

Constraint: shift-invert (iterative, matches dense only to solver tol) must NEVER be the solver
inside Newton's finite-difference F. ``newton_polish`` FORCES ``shift_invert=False`` in F, so the
caller's auto/True policy has zero effect there and F stays bit-deterministic.

This proves it on the BCT-O test config: newton_polish is run with the SAME warm state twice,
once with ``scf_kwargs`` carrying ``shift_invert="auto"`` and once ``shift_invert=False``. If the
force works, the two runs are bit-identical — same ``f_evals`` and the same ``residual_norm`` to
the last bit (F never saw the iterative solver). A mismatch means shift-invert leaked into F.
"""
import numpy as np

from gradwave.flapw import crystal_scf_multi, newton_polish

BCT_A = [5.4, 5.4, 6.6]
ATOMS = [((0.0, 0.0, 0.0), "O"), ((0.5, 0.5, 0.5), "O")]
RADII = {"O": 0.75}
CFG = dict(ecut=100.0, lmax=2, kmesh=(1, 1, 1), smearing=0.0, efg=False, fullpot=True,
           fullpot_lmax=2, use_symmetry=True, kworkers=1, subspace_reuse=False)


def polish(mode):
    _, info = crystal_scf_multi(BCT_A, ATOMS, RADII, iters=22, tol=0.0,
                                shift_invert=mode, **CFG)
    _st, ninfo = newton_polish(BCT_A, ATOMS, RADII, info["state"],
                               scf_kwargs=dict(CFG, shift_invert=mode),
                               maxiter=3, inner_maxiter=8, f_tol=1e-6)
    return ninfo


def main():
    a = polish("auto")
    b = polish(False)
    print(f"auto : f_evals={a['f_evals']} residual_norm={a['residual_norm']!r} "
          f"rounds={a['rounds']} converged={a['converged']}", flush=True)
    print(f"False: f_evals={b['f_evals']} residual_norm={b['residual_norm']!r} "
          f"rounds={b['rounds']} converged={b['converged']}", flush=True)
    same_fe = a["f_evals"] == b["f_evals"]
    same_rn = float(np.abs(a["residual_norm"] - b["residual_norm"])) == 0.0
    ok = same_fe and same_rn
    print(f"NEWTON-DET f_evals_equal={same_fe} residual_bit_equal={same_rn} "
          f"{'PASS (F is exact/deterministic)' if ok else 'FAIL (SI leaked into F)'}", flush=True)


if __name__ == "__main__":
    main()
