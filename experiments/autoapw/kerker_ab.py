"""A/B the interstitial Kerker screen on the hard TiO2 config (DMD-motivated).

Shared cold warmup -> three 40-iteration continuations from the SAME full state:
plain Anderson (the measured r_v=0.46 staller), kerker=0.7, kerker=1.5 [Å⁻¹].
Reports the r_v / r_nsph trajectory tail of each. Success = the screened runs
break the plateau (r_v decaying where plain stalls); newton_polish remains the
closer for machine-precision, this is about making Anderson itself robust.

Env: KA_ECUT/KA_LMAX/KA_FPLMAX/KA_K/KA_KWORKERS/KA_WARM/KA_ITERS override the config.
"""
import os
import sys
import time

from newton_probe import A_BOHR, ATOMS, RADII

from gradwave.flapw import crystal_scf_multi

CFG = dict(ecut=float(os.environ.get("KA_ECUT", "300")),
           lmax=int(os.environ.get("KA_LMAX", "3")),
           kmesh=(int(os.environ.get("KA_K", "2")),) * 3,
           smearing=0.0, efg=False, fullpot=True,
           fullpot_lmax=int(os.environ.get("KA_FPLMAX", "4")), use_symmetry=True,
           kworkers=int(os.environ.get("KA_KWORKERS", "4")), subspace_reuse=False)


def tail(info, n=6):
    tr = info["recorder"].to_trace_dict()["columns"]
    rv, rn = tr["r_v"], tr["r_nsph"]
    def fmt(v):
        return "  ".join("-" if x is None else f"{x:.2e}" for x in v[-n:])
    return f"r_v: {fmt(rv)}\n    r_nsph: {fmt(rn)}"


def main():
    n0 = int(os.environ.get("KA_WARM", "24"))
    it = int(os.environ.get("KA_ITERS", "40"))
    print(f"config: {CFG}", flush=True)
    t0 = time.time()
    _, i0 = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=n0, tol=0.0, **CFG)
    st0 = i0["state"]
    print(f"warmup {n0} it ({time.time()-t0:.0f}s)", flush=True)
    for k0 in (None, 0.7, 1.5):
        t0 = time.time()
        _, ii = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=it, tol=0.0, kerker=k0,
                                  v_start={"__full_state__": st0}, **CFG)
        print(f"kerker={k0} ({time.time()-t0:.0f}s)\n    {tail(ii)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
