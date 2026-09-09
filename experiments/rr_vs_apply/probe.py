#!/usr/bin/env python
"""PROBE: RR-subspace GEMM vs true H-apply wall split in davidson_batched.

Question: is deep-band locking (RR-subspace deflation of converged deep
semicore bands) worth building for large-N slab SCF? It only pays if the
lockable RR-GEMM work (Gram s = v.conj()@hv.mT, eigh(s), Ritz reconstruction
x=u·v / hx=u·hv) is a large share of wall. A prior heuristic profiler buried
that work UNDIFFERENTIATED inside its ~47% "h-apply matmul" bucket.

This reproduces the moonshot's representative heavy-semicore slab
(Cu(100) 2x2x3 = 12 atoms, Cu_ONCV_PBE-1.2 19e semicore, ~150 bands,
kmesh (2,2,1), ecut 40 Ry, nspin 1) and runs one SCF to convergence with
davidson instrumented (GRADWAVE_RR_PROFILE=1) to time apply / gram / eigh /
ritz / ortho separately, and captures per-band Davidson history to measure the
by-round-2 lockable fraction.

Run on asus, OMP_NUM_THREADS=8:
    GRADWAVE_RR_PROFILE=1 OMP_NUM_THREADS=8 uv run python \
        experiments/rr_vs_apply/probe.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GRADWAVE_RR_PROFILE", "1")
os.environ.setdefault("GRADWAVE_EIGENSOLVER", "davidson")  # not CheFSI

import torch  # noqa: E402

from gradwave.constants import RY_EV as RY  # noqa: E402
from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402
import gradwave.solvers.davidson as dav  # noqa: E402
import gradwave.solvers.registry as reg  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

FIX = Path(__file__).resolve().parents[2] / "tests/fixtures/qe/pseudos"


def build_slab():
    from ase.build import fcc100

    slab = fcc100("Cu", size=(2, 2, 3), a=3.61, vacuum=None, orthogonal=True)
    slab.center(vacuum=6.0, axis=2)
    cell = np.asarray(slab.cell.array, float)
    pos = np.asarray(slab.get_positions(), float)
    return cell, pos, len(slab)


def main():
    cell, pos, natoms = build_slab()
    upf = parse_upf(FIX / "Cu_ONCV_PBE-1.2.upf")
    zval = float(upf.z_valence)
    nelec = zval * natoms
    nbands = 150
    print(f"natoms={natoms} zval={zval} nelec={nelec} nbands={nbands}", flush=True)

    sys_ = setup_system(
        cell, pos, [0] * natoms, [upf], ecut=40.0 * RY,
        kmesh=(2, 2, 1), nbands=nbands, use_symmetry=True,
    )
    nk = len(sys_.kpoints.weights) if hasattr(sys_, "kpoints") else "?"
    print(f"nk(IBZ)={nk}", flush=True)

    # Capture per-solve Davidson history by wrapping davidson_batched to inject a
    # fresh history_out list each call (the SCF adapter doesn't pass one).
    histories: list[list] = []
    diag_tols: list[float] = []
    _orig = dav.davidson_batched

    def _wrapped(*a, **kw):
        h: list = []
        kw["history_out"] = h
        diag_tols.append(float(kw.get("tol", a[4] if len(a) > 4 else float("nan"))))
        r = _orig(*a, **kw)
        histories.append(h)
        return r

    dav.davidson_batched = _wrapped
    try:
        dav._rrp_reset()
        t0 = time.perf_counter()
        result = scf(
            sys_, PBE(), smearing="gaussian", width=0.15,
            max_iter=80, etol=1e-6, rhotol=1e-5,
            eigensolver="davidson", verbose=True,
        )
        wall = time.perf_counter() - t0
    finally:
        dav.davidson_batched = _orig

    prof = dict(dav._RR_PROF)
    energy = getattr(result, "energy", getattr(result, "e_total", None))
    print("\n==== TIMING (summed over all SCF steps) ====", flush=True)
    print(f"total scf wall        : {wall:9.3f} s", flush=True)
    apply = prof.get("apply", 0.0)
    gram = prof.get("gram", 0.0)
    eigh = prof.get("eigh", 0.0)
    ritz = prof.get("ritz", 0.0)
    ortho = prof.get("ortho", 0.0)
    rr = gram + eigh + ritz
    dav_total = apply + rr + ortho
    for k, v in [("apply (true H·ψ)", apply), ("gram (v.conj@hv)", gram),
                 ("eigh", eigh), ("ritz (u·v, u·hv)", ritz),
                 ("ortho/restart QR", ortho)]:
        print(f"  {k:22s}: {v:9.3f} s  ({100*v/wall:5.1f}% wall)", flush=True)
    print(f"  {'RR-GEMM (gram+eigh+ritz)':22s}: {rr:9.3f} s  "
          f"({100*rr/wall:5.1f}% wall)", flush=True)
    print(f"  {'davidson-internal total':22s}: {dav_total:9.3f} s  "
          f"({100*dav_total/wall:5.1f}% wall)", flush=True)
    print(f"\nRR-GEMM share of wall               : {100*rr/wall:5.1f}%", flush=True)
    print(f"RR-GEMM share of davidson-internal  : {100*rr/dav_total:5.1f}%",
          flush=True)
    print(f"RR-GEMM as fraction of apply bucket  : {100*rr/apply:5.1f}%",
          flush=True)

    # Lockable fraction: within each Davidson solve, fraction of bands with
    # residual < tol by round 2 (deep Cu 3s/3p/3d semicore should converge fast).
    def round2_frac(h, tol):
        rounds = {it: rn for (it, rn, _eig) in h}
        rn = rounds.get(2)
        if rn is None:  # solve converged in <2 rounds -> everything lockable
            rn = rounds.get(max(rounds)) if rounds else None
        if rn is None:
            return None
        return float((rn < tol).float().mean())

    fracs = []
    for h, tol in zip(histories, diag_tols):
        f = round2_frac(h, tol)
        if f is not None:
            fracs.append(f)
    print("\n==== LOCKABLE FRACTION (bands with res<tol by round 2) ====",
          flush=True)
    print(f"n_scf_steps={len(histories)} diag_tol≈{diag_tols[-1] if diag_tols else '?'}",
          flush=True)
    if fracs:
        print(f"  step 1 (cold)      : {fracs[0]:.3f}", flush=True)
        mid = len(fracs) // 2
        print(f"  step {mid+1} (mid)      : {fracs[mid]:.3f}", flush=True)
        print(f"  step {len(fracs)} (last)     : {fracs[-1]:.3f}", flush=True)
        print(f"  mean over steps    : {np.mean(fracs):.3f}", flush=True)
        print(f"  min over steps     : {np.min(fracs):.3f}", flush=True)

    lock = float(np.mean(fracs)) if fracs else 0.0
    rr_share = rr / wall
    # Ceiling: locking removes converged bands from RR GEMMs (Gram/ritz scale
    # ~(nb_active/nb)^2 for gram, ~linear for ritz reconstruction output width).
    # Optimistic upper bound: eliminate the lockable fraction of RR-GEMM work.
    ceiling_savings = lock * rr_share
    ceiling_speedup = 1.0 / (1.0 - ceiling_savings) if ceiling_savings < 1 else float("inf")
    print("\n==== DEEP-BAND-LOCKING CEILING ====", flush=True)
    print(f"lockable_fraction × RR-GEMM-share = {lock:.3f} × {rr_share:.3f} "
          f"= {ceiling_savings:.4f} of wall", flush=True)
    print(f"optimistic ceiling speedup ≈ {ceiling_speedup:.3f}x", flush=True)
    print(f"\nenergy={energy}", flush=True)

    out = dict(
        config=dict(system="Cu(100) 2x2x3", natoms=natoms, zval=zval,
                    nelec=nelec, nbands=nbands, ecut_ry=40.0,
                    kmesh="(2,2,1)", nk=str(nk), nspin=1, threads=torch.get_num_threads()),
        wall_s=wall, prof=prof, rr_gemm_s=rr, rr_share_wall=rr_share,
        rr_share_dav=rr / dav_total, rr_over_apply=rr / apply,
        lockable_round2=dict(per_step=fracs, mean=lock),
        ceiling_savings=ceiling_savings, ceiling_speedup=ceiling_speedup,
        energy=float(energy) if energy is not None else None,
        n_scf_steps=len(histories), diag_tol=diag_tols[-1] if diag_tols else None,
    )
    outp = Path(__file__).resolve().parent / "result.json"
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
