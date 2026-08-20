"""Decisive no-SCF test of the forensics verdict (elk_efg_forensics.md): is gradwave's
lattice/boundary term sign-INVERTED vs the true external field?

Elk's raw Cartesian O1 tensor at (u,u,0) has T_xy = -0.1838 a.u. = -17.86 eV/A^2
(EFG.OUT, same electron-potential convention as ours). The eigenvalue listings are
sign/permutation-ambiguous under near-degeneracy, but T_xy on this site is symmetry-pinned:
  gradwave T_xy > 0  -> boundary-term inversion CONFIRMED (beta = -2.16 story)
  gradwave T_xy < 0  -> inversion hypothesis KILLED
One EFG evaluation from a saved gated state; no SCF.
"""
import os
import pickle
import sys

import numpy as np
from _common import A_BOHR, ATOMS, RADII

from gradwave.flapw import crystal_scf_multi

CFG = dict(ecut=300.0, lmax=3, smearing=0.0, fullpot=True, fullpot_lmax=4,
           use_symmetry=True, subspace_reuse=False, kworkers=3, kerker=0.7, los=None)


def main():
    path = os.path.expanduser(os.environ.get("TS_STATE", "~/tio2_states/base_k222.pkl"))
    with open(path, "rb") as f:
        state = pickle.load(f)
    print(f"state: {path}", flush=True)
    _, i = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                             v_start={"__full_state__": state}, kmesh=(2, 2, 2), **CFG)
    np.set_printoptions(precision=3, suppress=True)
    for key, name in (("a0", "Ti1"), ("a2", "O1"), ("a3", "O2")):
        t = i["efg"][key]["tensor"]
        print(f"{name} Cartesian tensor (eV/A^2):\n{t}", flush=True)
    txy = i["efg"]["a2"]["tensor"][0, 1]
    print(f"O1 T_xy = {txy:+.3f} eV/A^2   (Elk O1: -17.86)", flush=True)
    print("VERDICT: " + ("INVERSION CONFIRMED (T_xy sign opposite Elk)" if txy > 0
                         else "inversion hypothesis KILLED (T_xy sign matches Elk)"),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
