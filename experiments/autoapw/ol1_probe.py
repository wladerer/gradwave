"""Fast (seconds) pre-probe for the l=1 (O 2p) local orbital conditioning.

Resumes a converged state (default the #344 O-2s-LO state ~/tio2_states/aug4olo_k222.pkl),
rebuilds the O sphere channels exactly as _multi_iterate does, then for an E2 ladder calls
_build_lo(l=1, ...) and reports resid_frac (near-linear-dependence with valence span{u,udot}),
whether #344's conditioning would orthogonalize it (resid_frac < lo_cond_tol), and whether the
degeneracy guard fires. No SCF — this only tells us which ladder points are usable and how the
conditioning classifies them, so we don't burn SCF slots on configs that instantly raise.

Env: STATE=path (warm state), CONDTOL=0.18, ECUT=300 LMAX=4.
"""
import os
import pickle

import numpy as np
from _common import A_BOHR, ATOMS, RADII

from gradwave.flapw import scf as S

CONDTOL = float(os.environ.get("CONDTOL", str(S._LO_COND_TOL)))
ECUT = float(os.environ.get("ECUT", "300"))
LMAX = int(os.environ.get("LMAX", "4"))
STATE = os.path.expanduser(os.environ.get("STATE", "~/tio2_states/aug4olo_k222.pkl"))

# ladder of E2 (absolute atomic energies, eV). O 2p = -11.72, O 2s = -30.92.
LADDER = [-8.0, -11.72, -14.0, -16.0, -18.0, -20.0, -24.0, -28.0]


def main():
    with open(STATE, "rb") as f:
        state = pickle.load(f)
    ctx = S._multi_setup(A_BOHR, ATOMS, RADII, ecut=ECUT, lmax=LMAX, fullpot=True,
                         fullpot_lmax=4, use_symmetry=True, kworkers=1, subspace_reuse=False,
                         los={"O": [(0, "2s")]}, lo_cond_tol=CONDTOL)
    st = S._multi_init_state(ctx, {"__full_state__": state})
    r, dx, r_np = ctx.r, ctx.dx, ctx.r_np
    # find an O key
    okey = next(k for k, s in zip(ctx.keys, ctx.syms, strict=True) if s == "O")
    R = ctx.R_by_key[okey]
    v0 = float(st.v_by_key[okey].numpy()[np.argmin(np.abs(r_np - R))])
    import torch
    vmt = torch.where(r <= R, st.v_by_key[okey] - v0, torch.zeros_like(r))
    nl = S._VALENCE_NL["O"]
    El = {lang: ctx.at_by_sym["O"].get(nl.get(lang, ""), -5.0) - v0
          for lang in range(max(LMAX, 2) + 1)}
    from gradwave.flapw.lapw import radial_channels_all
    chan = radial_channels_all(LMAX, El, r, dx, vmt, R)
    ch1 = chan[1]
    u, ud = ch1["u_in"], ch1["ud_in"]
    print(f"# ol1_probe STATE={STATE} v0(O)={v0:.3f} eV  valence l=1 E={El[1]+v0:.3f} eV "
          f"(2p={ctx.at_by_sym['O']['2p']:.3f})  cond_tol={CONDTOL}", flush=True)
    print(f"{'E2_abs':>8} {'e2-v0':>9} {'resid_frac':>11} {'orthog?':>8}  status")
    for e2abs in LADDER:
        e2 = e2abs - v0
        try:
            lo = S._build_lo(1, e2, ch1, u, ud, r, dx, vmt, R, cond_tol=CONDTOL)
            print(f"{e2abs:8.2f} {e2:9.2f} {lo['resid_frac']:11.4f} "
                  f"{str(lo['orthogonalized']):>8}  H_pp={lo['H_pp']:.3f} "
                  f"S_pu={lo['S_pu']:.3f} a={lo['a']:.2f} b={lo['b']:.2f} cn={lo['cn']:.2f}")
        except Exception as e:
            print(f"{e2abs:8.2f} {e2:9.2f} {'--':>11} {'--':>8}  RAISED {type(e).__name__}: "
                  f"{str(e)[:60]}")


if __name__ == "__main__":
    main()
