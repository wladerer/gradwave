#!/usr/bin/env python
"""Tensor-train (QTT) rank of a converged slab density along the surface normal.

Backlog item ``tt_rank_slab`` (docs/h100_backlog.md #4). Decides the QTT/wavelet
surface-compression bet: a slab has a wide, nearly-constant vacuum region along
the surface normal, which *should* make the density low-rank in that direction
compared with a bulk density. This probe measures it.

It provides a self-contained SVD-based TT-SVD estimator (Oseledets' sequential
algorithm at a relative Frobenius tolerance -- no external TT library needed) and
a QTT wrapper that folds a length-2^L axis into L binary modes. Given a converged
density it reports the QTT rank profile of the planar-averaged density n(z) along
the normal, and (with a bulk reference) the slab-vs-bulk rank ratio.

The estimator is validated on SYNTHETIC tensors with a known rank -- a rank-1
outer product (all TT ranks 1), a smooth vacuum-like profile (low QTT rank), and
white noise (near-maximal rank) -- so the correctness check needs no SCF.

Density input: a gradwave checkpoint (``io.checkpoint.save_checkpoint`` -> the
3-D real-space total density ``rho`` on the FFT box) or a raw ``.npy``/``.npz``
array. The real converged Pt/Au slab is supplied by the H100 (the cell that OOMs
the RTX 3050); this driver is analysis-only.

Usage:
  # synthetic self-test of the rank estimator (no SCF, laptop-safe)
  uv run python experiments/surface_efficiency/tt_rank.py --synthetic --out tt.json

  # slab vs bulk on real converged densities (H100 supplies the checkpoints)
  uv run python experiments/surface_efficiency/tt_rank.py \
      --slab-ckpt slab.pt --bulk-ckpt bulk.pt --normal-axis 2 --epsilon 1e-8 \
      --out tt.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# TT-SVD: sequential SVD sweep at a relative Frobenius tolerance (Oseledets).
# --------------------------------------------------------------------------- #
def tt_svd(tensor, epsilon=1e-8, max_rank=None):
    """TT-SVD decomposition of an n-D array.

    Returns (cores, ranks) where ``cores[k]`` has shape (r_k, n_k, r_{k+1}) with
    r_0 = r_d = 1, and ``ranks`` is the list of interior TT ranks [r_1..r_{d-1}].
    The truncation follows the standard delta = epsilon/sqrt(d-1) * ||A||_F
    per-mode split, giving a global relative Frobenius error <= epsilon.
    """
    a = np.asarray(tensor, dtype=np.float64)
    shape = a.shape
    d = a.ndim
    if d == 1:
        return [a.reshape(1, shape[0], 1)], []
    norm = np.linalg.norm(a)
    delta = (epsilon / np.sqrt(d - 1)) * norm if norm > 0 else 0.0
    cores = []
    ranks = []
    r_prev = 1
    c = a.reshape(shape[0], -1)  # unfold: leave mode 0 as rows
    for k in range(d - 1):
        c = c.reshape(r_prev * shape[k], -1)
        u, s, vt = np.linalg.svd(c, full_matrices=False)
        r = _truncation_rank(s, delta)
        if max_rank is not None:
            r = min(r, max_rank)
        r = max(r, 1)
        cores.append(u[:, :r].reshape(r_prev, shape[k], r))
        ranks.append(r)
        c = (np.diag(s[:r]) @ vt[:r, :])
        r_prev = r
    cores.append(c.reshape(r_prev, shape[-1], 1))
    return cores, ranks


def _truncation_rank(s, delta):
    """Largest r such that sum_{i>r} s_i^2 <= delta^2 (tail energy bound)."""
    if delta <= 0:
        return int(np.sum(s > 1e-300))
    tail = np.cumsum(s[::-1] ** 2)[::-1]  # tail[i] = sum_{j>=i} s_j^2
    # keep the smallest r with tail energy beyond r within delta^2
    keep = np.searchsorted(-tail, -delta ** 2, side="right")
    return int(max(keep, 1))


def tt_reconstruct(cores):
    """Contract TT cores back to a full tensor (for error checks/tests)."""
    out = cores[0]
    for core in cores[1:]:
        out = np.tensordot(out, core, axes=([out.ndim - 1], [0]))
    return out.reshape([c.shape[1] for c in cores])


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def qtt_ranks(vec, epsilon=1e-8, pad="edge"):
    """QTT ranks of a 1-D signal: fold length to 2^L, TT-SVD the binary modes.

    Non-power-of-2 lengths are padded up to the next power of two (edge-padded by
    default so a vacuum tail stays flat). Returns (ranks, info).
    """
    v = np.asarray(vec, dtype=np.float64).ravel()
    n = v.size
    n2 = _next_pow2(n)
    if n2 != n:
        v = np.pad(v, (0, n2 - n), mode=pad)
    levels = int(np.log2(n2))
    folded = v.reshape([2] * levels)
    _, ranks = tt_svd(folded, epsilon=epsilon)
    info = {"orig_len": int(n), "padded_len": int(n2), "levels": levels,
            "max_rank": int(max(ranks)) if ranks else 1,
            "mean_rank": float(np.mean(ranks)) if ranks else 1.0,
            "sum_rank": int(np.sum(ranks)) if ranks else 0}
    return ranks, info


# --------------------------------------------------------------------------- #
# Density loading + normal-direction reduction.
# --------------------------------------------------------------------------- #
def load_density(path):
    """Load a 3-D real-space density from a gradwave checkpoint or .npy/.npz."""
    path = Path(path)
    if path.suffix in (".npy",):
        return np.asarray(np.load(path), dtype=np.float64)
    if path.suffix in (".npz",):
        z = np.load(path)
        key = "rho" if "rho" in z else next(iter(z.keys()))
        return np.asarray(z[key], dtype=np.float64)
    # gradwave checkpoint
    from gradwave.io.checkpoint import load_checkpoint
    payload = load_checkpoint(path)
    rho = payload["rho"]
    rho = rho.numpy() if hasattr(rho, "numpy") else np.asarray(rho)
    return np.asarray(rho, dtype=np.float64)


def planar_average(rho, normal_axis=2):
    """n(z): average the 3-D density over the two in-plane axes."""
    rho = np.asarray(rho, dtype=np.float64)
    other = tuple(ax for ax in range(rho.ndim) if ax != normal_axis)
    return rho.mean(axis=other)


def density_qtt_report(rho, normal_axis=2, epsilon=1e-8):
    nz = planar_average(rho, normal_axis)
    ranks, info = qtt_ranks(nz, epsilon=epsilon)
    info["ranks"] = [int(r) for r in ranks]
    info["normal_axis"] = normal_axis
    info["epsilon"] = epsilon
    info["grid_shape"] = list(np.asarray(rho).shape)
    return info


# --------------------------------------------------------------------------- #
# Synthetic validation (no SCF).
# --------------------------------------------------------------------------- #
def synthetic_selftest(epsilon=1e-8, seed=0):
    rng = np.random.default_rng(seed)
    checks = {}

    # 1) rank-1 outer product: every interior TT rank must be 1.
    u = rng.standard_normal(8)
    v = rng.standard_normal(8)
    w = rng.standard_normal(8)
    rank1 = u[:, None, None] * v[None, :, None] * w[None, None, :]
    cores, ranks = tt_svd(rank1, epsilon=1e-10)
    rec = tt_reconstruct(cores)
    checks["rank1_outer"] = {
        "ranks": [int(r) for r in ranks], "max_rank": int(max(ranks)),
        "recon_rel_err": float(np.linalg.norm(rec - rank1) / np.linalg.norm(rank1)),
        "pass": max(ranks) == 1}

    # 2) smooth vacuum-like n(z): a slab bump + long flat vacuum -> low QTT rank.
    L = 256
    z = np.linspace(0, 1, L)
    slab = np.exp(-((z - 0.5) ** 2) / (2 * 0.05 ** 2))  # bump in the middle
    vac = slab.copy()
    vac[: L // 4] = slab[: L // 4] * 0 + 1e-6           # flat vacuum tails
    vac[3 * L // 4:] = 1e-6
    r_smooth, i_smooth = qtt_ranks(vac, epsilon=epsilon)
    checks["vacuum_like_nz"] = {**i_smooth, "ranks": [int(r) for r in r_smooth]}

    # 3) white noise of the same length: near-maximal QTT rank.
    noise = rng.standard_normal(L)
    r_noise, i_noise = qtt_ranks(noise, epsilon=epsilon)
    checks["white_noise_nz"] = {**i_noise, "ranks": [int(r) for r in r_noise]}

    # the discriminating assertion: a vacuum-like profile compresses, noise does not
    checks["smooth_below_noise"] = bool(i_smooth["max_rank"] < i_noise["max_rank"])
    checks["reconstruct_exact_lowrank"] = bool(
        checks["rank1_outer"]["recon_rel_err"] < 1e-9)
    return checks


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_synthetic(cfg):
    checks = synthetic_selftest(epsilon=cfg.epsilon, seed=cfg.seed)
    out = {"kind": "tt_rank_slab", "mode": "synthetic", "epsilon": cfg.epsilon,
           "checks": checks}
    Path(cfg.out).write_text(json.dumps(out, indent=2))
    print(f"[synthetic] rank-1 outer max_rank={checks['rank1_outer']['max_rank']} "
          f"(err {checks['rank1_outer']['recon_rel_err']:.1e})", flush=True)
    print(f"[synthetic] vacuum-like n(z) max QTT rank={checks['vacuum_like_nz']['max_rank']}  "
          f"white-noise max QTT rank={checks['white_noise_nz']['max_rank']}  "
          f"smooth<noise={checks['smooth_below_noise']}", flush=True)
    print(f"wrote {cfg.out}", flush=True)
    return out


def run_measure(cfg):
    out = {"kind": "tt_rank_slab", "mode": "measure",
           "epsilon": cfg.epsilon, "normal_axis": cfg.normal_axis}
    slab = density_qtt_report(load_density(cfg.slab_ckpt), cfg.normal_axis, cfg.epsilon)
    out["slab"] = {"source": str(cfg.slab_ckpt), **slab}
    print(f"[slab] max QTT rank(normal)={slab['max_rank']} mean={slab['mean_rank']:.2f} "
          f"grid={slab['grid_shape']}", flush=True)
    if cfg.bulk_ckpt:
        bulk = density_qtt_report(load_density(cfg.bulk_ckpt), cfg.normal_axis, cfg.epsilon)
        out["bulk"] = {"source": str(cfg.bulk_ckpt), **bulk}
        out["slab_over_bulk_max_rank"] = slab["max_rank"] / max(bulk["max_rank"], 1)
        out["slab_over_bulk_sum_rank"] = slab["sum_rank"] / max(bulk["sum_rank"], 1)
        print(f"[bulk] max QTT rank(normal)={bulk['max_rank']} "
              f"mean={bulk['mean_rank']:.2f}", flush=True)
        print(f"[decision] slab/bulk max-rank ratio = {out['slab_over_bulk_max_rank']:.3f} "
              "(<1 => slab normal is low-rank => GO on QTT compression)", flush=True)
    Path(cfg.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {cfg.out}", flush=True)
    return out


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true",
                   help="run the no-SCF estimator self-test and exit")
    p.add_argument("--slab-ckpt", default=None, help="converged slab density (ckpt/.npy/.npz)")
    p.add_argument("--bulk-ckpt", default=None, help="bulk reference density")
    p.add_argument("--normal-axis", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=1e-8, help="TT-SVD relative tolerance")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="tt_rank_slab.json")
    return p.parse_args(argv)


def main(argv=None):
    cfg = _parse_args(argv)
    if cfg.synthetic or not cfg.slab_ckpt:
        if not cfg.synthetic and not cfg.slab_ckpt:
            print("[note] no --slab-ckpt given; running --synthetic self-test", flush=True)
        run_synthetic(cfg)
    else:
        run_measure(cfg)


if __name__ == "__main__":
    main()
