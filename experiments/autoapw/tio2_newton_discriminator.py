"""Discriminate WHY newton_polish(k333) diverges from A/B-round states where job 128 converged.

Round 1+2 of the basis A/B: Newton k333 diverged from r_nsph 4.9e-3 AND 7.3e-4 states,
while job 128 converged from 2.2e-4. Candidate explanations: (a) the k222 state's
INTERSTITIAL channel (r_v — the measured 94%-interstitial unstable mode) is hot even
though r_nsph looks gated; (b) the k333 Newton basin is simply tighter than 1e-3;
(c) the surface-phase memo (e42729e/65cc440, the only physics-touching delta since
job 128) broke the F map for Newton. Three tests from the saved base_k222.pkl state:

  A. newton_polish at k222 (the state's own mesh). The newton_probe converged k222 from
     r_v 2.6, so divergence here == the machinery is broken (c).
  B. newton_polish at k333 (reproduce the failure, now logging both residual channels).
  C. B again with the memo disabled (budget 0 -> always compute inline). Divergence in
     B but convergence in C == the memo is implicated despite the 2.7e-15 validation.

Also prints r_v AND r_nsph of a 1-iteration resume from the state, so we finally see
the interstitial channel the A/B runner never printed.
"""
import os
import pickle
import sys
import time

from _common import A_BOHR, ATOMS, RADII

import gradwave.flapw.efg as efg_mod
from gradwave.flapw import crystal_scf_multi, newton_polish

CFG = dict(ecut=300.0, lmax=3, smearing=0.0, fullpot=True, fullpot_lmax=4,
           use_symmetry=True, subspace_reuse=False, kworkers=3, kerker=0.7, los=None)


def channels(state, kmesh, tag):
    _, i = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=False,
                             v_start={"__full_state__": state}, kmesh=kmesh, **CFG)
    it = i["recorder"].iters[-1]
    print(f"{tag}: 1-it resume r_v={it['r_v']:.3e} r_nsph={it.get('r_nsph', float('nan')):.3e}",
          flush=True)


def polish(state, kmesh, tag):
    t0 = time.time()
    st, ni = newton_polish(A_BOHR, ATOMS, RADII, state,
                           scf_kwargs=dict(CFG, kmesh=kmesh),
                           maxiter=4, inner_maxiter=12, f_tol=1e-6)
    print(f"{tag} ({time.time()-t0:.0f}s): {ni}", flush=True)
    return st, ni


def main():
    path = os.environ.get("ND_STATE", os.path.expanduser("~/tio2_states/base_k222.pkl"))
    with open(path, "rb") as f:
        state = pickle.load(f)
    print(f"state: {path}", flush=True)
    channels(state, (2, 2, 2), "k222")
    channels(state, (3, 3, 3), "k333")

    polish(state, (2, 2, 2), "A: newton k222")
    polish(state, (3, 3, 3), "B: newton k333 (memo on)")

    efg_mod._PHASE_CACHE.clear()
    efg_mod._PHASE_CACHE_BUDGET = 0
    polish(state, (3, 3, 3), "C: newton k333 (memo OFF)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
