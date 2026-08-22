"""Fast (seconds) HELO conditioning pre-probe for the l=1 channel on CORUNDUM (Al AND O).

The prior l=1 study (oxygen_l1_efg_diagnosis) only ever swept E2 in the LOW/negative range
(-8..-28 eV) where the confined LO is ~86% inside the valence p-span (resid_frac ~0.14). A HELO
lives at HIGH POSITIVE energy: u(E_high) is a scattering-like radial with an extra node, genuinely
distinct from span{u(E1), u_dot(E1)}. This probe rebuilds the Al and O l=1 spheres from the
converged corundum state and reports resid_frac over a HIGH-energy ladder — the fast test of
whether a high-E l=1 LO is a genuine new radial DOF (resid_frac -> ~1) before any SCF is spent.

No SCF. Env: STATE=~/efg_multimat/corundum.pkl  ECUT=300 LMAX=4 CONDTOL=0.18
"""
import os
import pickle

import numpy as np
import torch

from gradwave.flapw import atom as _atom
from gradwave.flapw import scf as _scf
from gradwave.flapw.lapw import radial_channels_all

# ---- corundum cell + runtime Al injection (mirrors corundum_efg.py) ----
_atom.CONFIG["Al"] = (13.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 1)])
_scf._CORE["Al"] = [(0, 1, 2), (0, 2, 2), (1, 1, 6)]
_scf._VAL_E["Al"] = 3
_scf._N_VAL_BANDS["Al"] = 2
_scf._VALENCE_NL["Al"] = {0: "3s", 1: "3p"}
_al_ha = {"1s": -55.5, "2s": -3.93, "2p": -2.56, "3s": -0.286, "3p": -0.10}
_atom.NIST_LDA_EV["Al"] = {k: v * 27.211386 for k, v in _al_ha.items()}

CELL_BOHR = np.array([[4.497737, 2.596770, 8.184592],
                      [-4.497737, 2.596770, 8.184592],
                      [-0.000000, -5.193539, 8.184592]])
ATOMS = [((0.352160, 0.352160, 0.352160), "Al"), ((0.147840, 0.147840, 0.147840), "Al"),
         ((0.647840, 0.647840, 0.647840), "Al"), ((0.852160, 0.852160, 0.852160), "Al"),
         ((0.250000, 0.556240, 0.943760), "O"), ((0.056240, 0.750000, 0.443760), "O"),
         ((0.556240, 0.943760, 0.250000), "O"), ((0.443760, 0.056240, 0.750000), "O"),
         ((0.943760, 0.250000, 0.556240), "O"), ((0.750000, 0.443760, 0.056240), "O")]
RADII = {"Al": 0.97, "O": 0.824}

CONDTOL = float(os.environ.get("CONDTOL", str(_scf._LO_COND_TOL)))
ECUT = float(os.environ.get("ECUT", "300"))
LMAX = int(os.environ.get("LMAX", "4"))
STATE = os.path.expanduser(os.environ.get("STATE", "~/efg_multimat/corundum.pkl"))

# HIGH-energy ladder (absolute atomic energy, eV). A few negatives for the redundant baseline,
# then the untested high-positive HELO regime.
LADDER = [-6.0, 0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 90.0]


def main():
    with open(STATE, "rb") as f:
        state = pickle.load(f)
    ctx = _scf._multi_setup(CELL_BOHR, ATOMS, RADII, ecut=ECUT, lmax=LMAX, fullpot=True,
                            fullpot_lmax=4, use_symmetry=True, kworkers=1, subspace_reuse=False,
                            los={"O": [(0, "2s")]}, el_override={"O": {0: "2p"}},
                            lo_cond_tol=CONDTOL)
    st = _scf._multi_init_state(ctx, {"__full_state__": state})
    r, dx, r_np = ctx.r, ctx.dx, ctx.r_np
    for sym in ("Al", "O"):
        key = next(k for k, s in zip(ctx.keys, ctx.syms, strict=True) if s == sym)
        R = ctx.R_by_key[key]
        v0 = float(st.v_by_key[key].numpy()[np.argmin(np.abs(r_np - R))])
        vmt = torch.where(r <= R, st.v_by_key[key] - v0, torch.zeros_like(r))
        nl = _scf._VALENCE_NL[sym]
        El = {lang: ctx.at_by_sym[sym].get(nl.get(lang, ""), -5.0) - v0
              for lang in range(max(LMAX, 2) + 1)}
        chan = radial_channels_all(LMAX, El, r, dx, vmt, R)
        ch1 = chan[1]
        u, ud = ch1["u_in"], ch1["ud_in"]
        e1 = El[1] + v0
        print(f"\n# {sym}: R={R:.4f} bohr  v0={v0:.3f} eV  valence l=1 E1={e1:.3f} eV", flush=True)
        print(f"{'E2_abs':>8} {'conf':>5} {'resid_frac':>11} {'orthog?':>8}  status")
        for confine in (True, False):
            for e2abs in LADDER:
                e2 = e2abs - v0
                try:
                    lo = _scf._build_lo(1, e2, ch1, u, ud, r, dx, vmt, R,
                                        cond_tol=CONDTOL, confine=confine)
                    print(f"{e2abs:8.2f} {str(confine):>5} {lo['resid_frac']:11.4f} "
                          f"{str(lo['orthogonalized']):>8}  H_pp={lo['H_pp']:.3f} "
                          f"S_pu={lo['S_pu']:+.3f} a={lo['a']:.2f} b={lo['b']:.2f} "
                          f"cn={lo['cn']:.2f}")
                except Exception as e:  # noqa: BLE001
                    print(f"{e2abs:8.2f} {str(confine):>5} {'--':>11} {'--':>8}  RAISED "
                          f"{type(e).__name__}: {str(e)[:50]}")


if __name__ == "__main__":
    main()
