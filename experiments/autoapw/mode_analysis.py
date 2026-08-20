"""Empirical mode analysis of the FLAPW SCF error dynamics (DMD on the state trajectory).

The recorder tells us WHICH scalar (r_nsph) converges last, but not which physical modes are
slow or how slow. This measures it directly: iterate the PLAIN damped map F (one crystal_scf_multi
iteration chained through the full state; chaining resets the Anderson history each step, which is
exactly what we want — the bare map's Jacobian, not Anderson's accelerated dynamics), collect the
step differences d_k = x_{k+1} − x_k in the loop's own convergence metric, and do dynamic mode
decomposition: the least-squares one-step operator restricted to span{d_k} has eigenvalues = the
dominant contraction rates ρ_i of F′ at the fixed point, and eigenvectors that can be attributed
to physical channels (per-atom spherical / per-(l,m) aspherical / interstitial).

Output per mode: |ρ|, implied iterations-per-decade (ln10/−ln|ρ|), and the channel breakdown.
That tells us WHAT to precondition (e.g. one aspherical channel at ρ→0.95 ⇒ per-channel beta or a
rank-k Broyden seed), or whether Newton is the only fix (many clustered slow modes).

Env: MA_ECUT/MA_LMAX/MA_FPLMAX/MA_K/MA_KWORKERS/MA_WARM (as newton_probe), MA_STEPS (default 18).
"""
import sys
import time

import numpy as np
from _common import A_BOHR, ATOMS, RADII, env_int, fullpot_cfg

from gradwave.flapw import crystal_scf_multi
from gradwave.flapw.newton import StateLayout as Layout

CFG = fullpot_cfg("MA")


def run(iters, v_start=None):
    return crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=iters, tol=0.0,
                             v_start=v_start, **CFG)


def main():
    n0 = env_int("MA", "WARM", 24)
    m = env_int("MA", "STEPS", 18)
    print(f"config: {CFG}\nwarm={n0} steps={m}", flush=True)

    t0 = time.time()
    _, i0 = run(n0)
    st = i0["state"]
    lay = Layout(st)
    spans = lay.segments(st)
    print(f"warmup {n0} it ({time.time()-t0:.0f}s); dof={lay.flatten(st).size}", flush=True)

    # plain damped-map trajectory (fresh Anderson history each step = bare F)
    xs = [lay.flatten(st)]
    t0 = time.time()
    for j in range(m):
        _, ii = run(1, v_start={"__full_state__": lay.unflatten(xs[-1], st)})
        xs.append(lay.flatten(ii["state"]))
        if j % 5 == 4:
            d = np.linalg.norm(xs[-1] - xs[-2])
            print(f"  step {j+1}/{m} |d|={d:.3e} ({time.time()-t0:.0f}s)", flush=True)
    D = np.stack([xs[j + 1] - xs[j] for j in range(m)], axis=1)  # (dof, m)

    # DMD in span{d_k}: A_hat = Q^H D2 pinv(R1) with D1 = Q R1
    D1, D2 = D[:, :-1], D[:, 1:]
    Q, R1 = np.linalg.qr(D1)
    A_hat = Q.T @ D2 @ np.linalg.pinv(R1)
    lam, W = np.linalg.eig(A_hat)
    order = np.argsort(-np.abs(lam))
    print(f"\ncontraction spectrum (top {min(8, lam.size)} of {lam.size}):", flush=True)
    for i in order[:8]:
        rho = abs(lam[i])
        mode = np.real(Q @ W[:, i])
        mode /= np.linalg.norm(mode) + 1e-300
        frac = sorted(((lab, float(np.sum(mode[a:b] ** 2))) for lab, a, b in spans),
                      key=lambda t: -t[1])
        top = "  ".join(f"{lab}={f:.2f}" for lab, f in frac[:3] if f > 0.02)
        per_dec = np.log(10.0) / max(-np.log(rho), 1e-12) if rho < 1 else float("inf")
        print(f"  |rho|={rho:6.3f}  it/decade={per_dec:6.1f}  {top}", flush=True)

    n_slow = int(np.sum(np.abs(lam) > 0.5))
    print(f"\nmodes with |rho|>0.5: {n_slow} (Anderson m=5 can kill ~5; "
          f"more than that clustered near |rho|max favors Newton)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
