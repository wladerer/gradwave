#!/usr/bin/env python
"""Stochastic-DFT trace-estimator variance probe (Baer-Neuhauser-Rabani).

Backlog item ``stochastic_dft_variance`` (docs/h100_backlog.md #3). A cheap
feasibility measurement, not a solver: it estimates a spectral quantity of a
Hamiltonian -- the trace of the Fermi operator ``Tr f(H)`` (the electron count)
and, optionally, its position-space diagonal (the density) -- by the stochastic
recipe at the heart of stochastic DFT:

    Tr f(H)  ~=  (1/N_chi) sum_i  chi_i^H  f(H)  chi_i ,   chi_i random +-1,

where ``f(H)`` is applied WITHOUT diagonalizing H, via a Chebyshev expansion of
the Fermi function ``f(E) = 1 / (1 + exp((E - mu) / w))``. The probe measures how
the estimator variance falls with N_chi (~ 1/N_chi, so the standard error ~
1/sqrt(N_chi)) and how the *relative* error shrinks with system size N
(self-averaging: relative std ~ 1/sqrt(N * N_chi)). Those two curves decide the
top large-N esoteric bet -- stochastic DFT only pays at hundreds+ electrons.

It reuses gradwave's existing H-apply (``core.batch.BatchedHamiltonian``,
assembled from a converged density exactly as ``postscf.dos.kpm_dos`` does) for
the real-material path; it does NOT reimplement the apply. The synthetic path
needs no SCF at all: a dense Hermitian operator with a known spectrum, so the
stochastic estimate has an exact deterministic reference ``sum_i f(eig_i)``.

Compute note: the large-N variance scan (hundreds of electrons, the regime where
self-averaging pays) is native-fp64 + big-vector H100 work. Locally, run only the
tiny ``--mode synthetic`` self-test.

Usage:
  # synthetic self-test: stochastic Tr f(H) -> deterministic as N_chi grows
  uv run python experiments/stochastic_dft/variance_probe.py \
      --mode synthetic --sizes 200 400 800 --nchi 1 2 4 8 16 32 --out var.json

  # real-material path (H100): variance from a converged small-cell Hamiltonian
  uv run python experiments/stochastic_dft/variance_probe.py \
      --mode scf --input si.yaml --nchi 1 2 4 8 16 32 --out var.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Chebyshev expansion of the Fermi function on a mapped spectral interval.
# --------------------------------------------------------------------------- #
def fermi(energy, mu, width):
    """f(E) = 1/(1+exp((E-mu)/w)); numerically-stable, w=0 -> step."""
    energy = np.asarray(energy, dtype=np.float64)
    if width <= 0:
        return (energy < mu).astype(np.float64)
    x = (energy - mu) / width
    out = np.empty_like(x)
    pos = x > 0
    out[pos] = np.exp(-x[pos]) / (1.0 + np.exp(-x[pos]))
    out[~pos] = 1.0 / (1.0 + np.exp(x[~pos]))
    return out


def chebyshev_fermi_coeffs(order, a, b, mu, width):
    """Chebyshev coefficients c_m of f(E) with E = a*x + b, x in [-1,1].

    c_m via the discrete cosine transform at (order+1) Chebyshev nodes. The
    reconstruction convention pairs with :func:`_cheb_fermi_apply`:
        f(H) ~= c_0 T_0(Ht) + sum_{m>=1} c_m T_m(Ht),   Ht = (H - b)/a.
    (c_0 already carries its 1/2 factor.)
    """
    m = order + 1
    k = np.arange(m)
    nodes_x = np.cos(np.pi * (k + 0.5) / m)          # Chebyshev nodes in [-1,1]
    fvals = fermi(a * nodes_x + b, mu, width)
    coeffs = np.empty(m, dtype=np.float64)
    for j in range(m):
        coeffs[j] = (2.0 / m) * np.sum(fvals * np.cos(np.pi * j * (k + 0.5) / m))
    coeffs[0] *= 0.5
    return coeffs


def _cheb_fermi_apply(matvec, x, coeffs, a, b):
    """Apply f(H) to a batch x (columns are vectors) via Chebyshev recurrence.

    matvec(V) returns H @ V for a (dim, nvec) batch (real or complex).
    """
    def hs(v):
        return (matvec(v) - b * v) / a

    t_prev = x
    t_cur = hs(x)
    out = coeffs[0] * t_prev + coeffs[1] * t_cur
    for cm in coeffs[2:]:
        t_next = 2.0 * hs(t_cur) - t_prev
        out = out + cm * t_next
        t_prev, t_cur = t_cur, t_next
    return out


# --------------------------------------------------------------------------- #
# Hutchinson trace estimator of Tr f(H).
# --------------------------------------------------------------------------- #
def stochastic_fermi_samples(matvec, dim, n_chi, coeffs, a, b, *,
                             rng=None, xp=np, dtype=float):
    """Per-random-vector samples s_i = chi_i^H f(H) chi_i (real).

    Mean over the samples estimates Tr f(H); the sample variance / n_chi is the
    variance of that mean. Random vectors are Rademacher (+-1), the standard
    stochastic-DFT choice (zero-mean, unit-variance, minimal estimator variance
    for real operators).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    signs = rng.integers(0, 2, size=(dim, n_chi)) * 2 - 1  # +-1
    chi = xp.asarray(signs, dtype=dtype)
    fchi = _cheb_fermi_apply(matvec, chi, coeffs, a, b)
    # s_i = <chi_i, f(H) chi_i>; real part guards tiny complex round-off.
    prod = (xp.conj(chi) * fchi).real if xp.iscomplexobj(fchi) else chi * fchi
    samples = prod.sum(axis=0)
    return np.asarray(samples, dtype=np.float64) if xp is np else samples.cpu().numpy()


def variance_curve(matvec, dim, coeffs, a, b, nchi_list, *,
                   n_repeat=64, seed=0, xp=np, dtype=float, ref_trace=None):
    """Variance / std-error of the Tr f(H) estimator vs N_chi.

    Draws a large pool of per-vector samples once, then for each N_chi forms
    n_repeat independent estimators (each an N_chi-average) and reports their
    empirical mean, std, and -- if a deterministic reference is given -- the mean
    absolute relative error. The 1/sqrt(N_chi) law shows up as std ~ N_chi^-1/2.
    """
    max_nchi = int(max(nchi_list))
    pool = max_nchi * n_repeat
    rng = np.random.default_rng(seed)
    samples = stochastic_fermi_samples(matvec, dim, pool, coeffs, a, b,
                                       rng=rng, xp=xp, dtype=dtype)
    rows = []
    for nchi in nchi_list:
        nchi = int(nchi)
        chunk = samples[: nchi * n_repeat].reshape(n_repeat, nchi)
        ests = chunk.mean(axis=1)
        row = {"n_chi": nchi, "est_mean": float(ests.mean()),
               "est_std": float(ests.std(ddof=1)) if n_repeat > 1 else 0.0}
        if ref_trace is not None:
            row["ref_trace"] = float(ref_trace)
            row["rel_err_mean"] = float(np.mean(np.abs(ests - ref_trace)) / abs(ref_trace))
            row["rel_std"] = row["est_std"] / abs(ref_trace)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Operator sources.
# --------------------------------------------------------------------------- #
def synthetic_operator(dim, *, gap=0.3, seed=0, metal=False):
    """A dense real-symmetric H with a known spectrum in ~[-1, 1].

    metal=False places a gap of width `gap` around 0 (insulator); metal=True
    fills the spectrum with no gap. Returns (matvec, eigvals, (a, b)) where
    (a, b) map the spectrum to Chebyshev [-1,1] with a small margin.
    """
    rng = np.random.default_rng(seed)
    half = dim // 2
    if metal:
        eig = np.sort(rng.uniform(-1.0, 1.0, size=dim))
    else:
        lo = rng.uniform(-1.0, -gap / 2, size=half)
        hi = rng.uniform(gap / 2, 1.0, size=dim - half)
        eig = np.sort(np.concatenate([lo, hi]))
    # random orthogonal basis so H is dense (not diagonal)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    h = (q * eig) @ q.T
    h = 0.5 * (h + h.T)
    lmin, lmax = float(eig.min()), float(eig.max())
    span = lmax - lmin
    a = (lmax - lmin) / 2 * 1.05 + 1e-9
    b = (lmax + lmin) / 2

    def matvec(x):
        return h @ x

    return matvec, eig, (a, b), (lmin - 0.025 * span, lmax + 0.025 * span)


def scf_operator(input_path, *, device="cpu"):
    """Build H-apply for a converged small cell via gradwave's own machinery.

    Runs the SCF from an input file, then assembles ``BatchedHamiltonian`` from
    the converged potential exactly as ``postscf.dos.kpm_dos`` does (Gamma /
    first k-point only for this probe). Returns
    (matvec, dim, (a, b), (lmin, lmax), n_electrons).
    """
    import torch

    from gradwave.api import run_scf
    from gradwave.core.batch import BatchedHamiltonian, projectors_b
    from gradwave.inputs import load_input
    from gradwave.postscf.dos import _spectral_bounds

    inp = load_input(input_path)
    res = run_scf(inp)
    system = res.system
    bk, grid = system.batch, system.grid
    assert bk is not None, "probe needs the batched-k geometry (system.batch)"
    nspin = getattr(res, "nspin", 1)
    veff = res.v_eff if nspin == 2 else res.v_eff[None]
    p_b = projectors_b(bk, system.positions)
    h = BatchedHamiltonian(bk, grid.shape, veff[0], p_b)
    lmin, lmax = _spectral_bounds(h, bk)
    a = (lmax - lmin) / 2 * 1.001
    b = (lmax + lmin) / 2
    mask = bk.mask

    # Act on the first k-point only; pack columns as the band axis.
    def matvec(x):
        # x: (npw_max, nvec) complex -> apply -> (npw_max, nvec)
        v = x.T[None, :, :] * mask[0][None, None, :]  # (1, nvec, npw_max)
        hv = h.apply(v.to(torch.complex128))
        return hv[0].T  # (npw_max, nvec)

    dim = int(mask[0].sum().item())
    n_el = float(system.n_electrons)
    return matvec, dim, mask, (a, b), (lmin, lmax), n_el, res


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_synthetic(cfg):
    order = cfg.order
    out = {"kind": "stochastic_dft_variance", "mode": "synthetic",
           "order": order, "width": cfg.width, "metal": cfg.metal, "sizes": []}
    for dim in cfg.sizes:
        matvec, eig, (a, b), _ = synthetic_operator(dim, gap=cfg.gap, seed=cfg.seed,
                                                    metal=cfg.metal)
        # half-filling: mu at the spectrum median (well inside the gap for the
        # insulator), so Tr f(H) ~= dim/2 electrons.
        mu = cfg.mu if cfg.mu is not None else float(np.median(eig))
        coeffs = chebyshev_fermi_coeffs(order, a, b, mu, cfg.width)
        ref = float(fermi(eig, mu, cfg.width).sum())
        rows = variance_curve(matvec, dim, coeffs, a, b, cfg.nchi,
                              n_repeat=cfg.repeat, seed=cfg.seed + 1, ref_trace=ref)
        # Chebyshev truncation accuracy vs the exact Fermi trace (deterministic).
        cheb_trace = float(sum(
            c * np.polynomial.chebyshev.Chebyshev.basis(m)((eig - b) / a).sum()
            for m, c in enumerate(coeffs)))
        out["sizes"].append({
            "dim": dim, "mu": mu, "ref_trace": ref,
            "cheb_trace": cheb_trace,
            "cheb_trace_rel_err": abs(cheb_trace - ref) / abs(ref),
            "curve": rows,
        })
        big = rows[-1]
        print(f"[synthetic] dim={dim:5d} ref={ref:.4f} cheb={cheb_trace:.4f} "
              f"(rel {abs(cheb_trace-ref)/abs(ref):.1e})  "
              f"N_chi={big['n_chi']}: rel_std={big['rel_std']:.3e}", flush=True)
    # self-averaging summary: relative std at the largest N_chi vs system size
    out["self_averaging"] = [
        {"dim": s["dim"], "rel_std_at_nchi_max": s["curve"][-1]["rel_std"]}
        for s in out["sizes"]]
    Path(cfg.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {cfg.out}", flush=True)
    return out


def run_scf_mode(cfg):
    import torch
    matvec, dim, mask, (a, b), (lmin, lmax), n_el, res = scf_operator(
        cfg.input, device=cfg.device)
    mu = cfg.mu if cfg.mu is not None else float(res.fermi)
    coeffs = chebyshev_fermi_coeffs(cfg.order, a, b, mu, cfg.width)
    # spin factor 2 for nspin=1 so Tr f(H) matches the electron count
    g_spin = 2.0 / getattr(res, "nspin", 1)
    rows = variance_curve(matvec, dim, coeffs, a, b, cfg.nchi,
                          n_repeat=cfg.repeat, seed=cfg.seed + 1,
                          xp=torch, dtype=torch.complex128,
                          ref_trace=n_el / g_spin)
    for r in rows:
        r["est_electrons"] = r["est_mean"] * g_spin
    out = {"kind": "stochastic_dft_variance", "mode": "scf", "input": str(cfg.input),
           "dim": dim, "n_electrons": n_el, "fermi": float(res.fermi),
           "lmin": lmin, "lmax": lmax, "order": cfg.order, "width": cfg.width,
           "g_spin": g_spin, "curve": rows}
    Path(cfg.out).write_text(json.dumps(out, indent=2))
    print(f"[scf] dim={dim} n_el={n_el} Tr f(H)*g -> {rows[-1]['est_electrons']:.3f} "
          f"(rel_std={rows[-1]['rel_std']:.2e} at N_chi={rows[-1]['n_chi']})", flush=True)
    print(f"wrote {cfg.out}", flush=True)
    return out


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="synthetic", choices=["synthetic", "scf"])
    p.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800],
                   help="synthetic Hilbert-space dimensions (system sizes)")
    p.add_argument("--nchi", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                   help="numbers of random vectors to sweep")
    p.add_argument("--order", type=int, default=400, help="Chebyshev order")
    p.add_argument("--width", type=float, default=0.05, help="Fermi width (kT), same units as H")
    p.add_argument("--mu", type=float, default=None,
                   help="chemical potential (default: median / fermi)")
    p.add_argument("--gap", type=float, default=0.3, help="synthetic insulator gap")
    p.add_argument("--metal", action="store_true", help="synthetic: gapless spectrum")
    p.add_argument("--repeat", type=int, default=64, help="independent estimators per N_chi")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input", default=None, help="SCF input file (mode=scf)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="stochastic_dft_variance.json")
    return p.parse_args(argv)


def main(argv=None):
    cfg = _parse_args(argv)
    if cfg.mode == "synthetic":
        run_synthetic(cfg)
    else:
        if not cfg.input:
            raise SystemExit("mode=scf requires --input <file>")
        run_scf_mode(cfg)


if __name__ == "__main__":
    main()
