"""Unit tests for the P0 H100-backlog drivers (docs/h100_backlog.md items 1/3/4).

All checks are tiny/synthetic and need no SCF: the geometry helpers and pseudo
resolver for the surface vacuum ladder, the Chebyshev-Fermi trace estimator's
convergence to the deterministic reference for the stochastic-DFT probe, and the
TT-SVD/QTT rank estimator on known-rank tensors for the TT-rank slab probe. The
heavy real-material runs are deferred to the H100 runner.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_EXP = Path(__file__).resolve().parents[2] / "experiments"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vl = _load(_EXP / "surface_efficiency" / "vacuum_ladder.py", "vl_driver")
sdft = _load(_EXP / "stochastic_dft" / "variance_probe.py", "sdft_driver")
ttr = _load(_EXP / "surface_efficiency" / "tt_rank.py", "ttr_driver")


# --------------------------------------------------------------------------- #
# 1. surface_vacuum_ladder: geometry + pseudo resolution (ASE only, no SCF)
# --------------------------------------------------------------------------- #
ase = pytest.importorskip("ase")


def test_clean_slab_vacuum_and_orthogonality():
    slab = vl.clean_slab(6.0, "Pt", layers=3)
    assert len(slab) == 12  # 2x2x3
    assert vl.actual_vacuum_per_face(slab) == pytest.approx(6.0, abs=1e-6)
    # ESM slab constraint: c must be perpendicular to the a,b plane
    assert vl.check_orthogonal(slab) < 1e-10


def test_thin_vacuum_shrinks_the_cell():
    thin = vl.clean_slab(6.0, "Pt")
    thick = vl.clean_slab(15.0, "Pt")
    assert vl.cell_c(thin) < vl.cell_c(thick)


def test_slab_with_co_adsorbate_indices():
    atoms, ads_idx = vl.slab_with_adsorbate(6.0, "ontop", "Pt", "CO")
    assert len(atoms) == 14  # 12 slab + C + O
    assert ads_idx == [12, 13]
    assert atoms.get_chemical_symbols()[-2:] == ["C", "O"]


def test_pseudo_resolution_finds_repo_upfs():
    # Pt from delta_gauge, C/O from the qe fixture dir -- both repo defaults.
    resolved = vl.resolve_pseudos(
        ["Pt", "C", "O"], vl._DEFAULT_PMAP, vl._pseudo_search_dirs(None))
    assert set(resolved) == {"Pt", "C", "O"}
    for p in resolved.values():
        assert Path(p).is_file()


# --------------------------------------------------------------------------- #
# 3. stochastic_dft_variance: Chebyshev-Fermi trace + Hutchinson variance
# --------------------------------------------------------------------------- #
def test_chebyshev_fermi_trace_matches_exact():
    """Deterministic Chebyshev trace of f(H) matches sum_i f(eig_i)."""
    matvec, eig, (a, b), _ = sdft.synthetic_operator(300, gap=0.3, seed=1)
    mu = float(np.median(eig))
    coeffs = sdft.chebyshev_fermi_coeffs(400, a, b, mu, width=0.05)
    ref = float(sdft.fermi(eig, mu, 0.05).sum())
    cheb = float(sum(
        c * np.polynomial.chebyshev.Chebyshev.basis(m)((eig - b) / a).sum()
        for m, c in enumerate(coeffs)))
    assert abs(cheb - ref) / abs(ref) < 1e-4
    # half-filled insulator: trace ~ dim/2 electrons
    assert ref == pytest.approx(150.0, abs=1.0)


def test_stochastic_trace_converges_to_deterministic():
    """Estimator error falls ~1/sqrt(N_chi) and lands on the exact trace."""
    dim = 400
    matvec, eig, (a, b), _ = sdft.synthetic_operator(dim, gap=0.3, seed=2)
    mu = float(np.median(eig))
    coeffs = sdft.chebyshev_fermi_coeffs(400, a, b, mu, width=0.05)
    ref = float(sdft.fermi(eig, mu, 0.05).sum())
    rows = sdft.variance_curve(matvec, dim, coeffs, a, b,
                               [1, 4, 16, 64], n_repeat=48, seed=3, ref_trace=ref)
    # monotone-ish decrease of the estimator std with N_chi
    stds = [r["est_std"] for r in rows]
    assert stds[0] > stds[-1] > 0
    # 1/sqrt(N_chi): 16x more vectors -> ~4x smaller std (allow a loose band)
    ratio = stds[0] / stds[-1]
    assert 2.5 < ratio < 12.0
    # the large-N_chi estimate sits on the deterministic reference
    assert abs(rows[-1]["est_mean"] - ref) / ref < 0.05


def test_self_averaging_relative_error_falls_with_size():
    """Relative error at fixed N_chi shrinks as the system grows."""
    rel = []
    for dim in (100, 400, 1600):
        matvec, eig, (a, b), _ = sdft.synthetic_operator(dim, gap=0.3, seed=dim)
        mu = float(np.median(eig))
        coeffs = sdft.chebyshev_fermi_coeffs(300, a, b, mu, width=0.05)
        ref = float(sdft.fermi(eig, mu, 0.05).sum())
        rows = sdft.variance_curve(matvec, dim, coeffs, a, b, [8],
                                   n_repeat=64, seed=1, ref_trace=ref)
        rel.append(rows[-1]["rel_std"])
    assert rel[0] > rel[-1]  # self-averaging


# --------------------------------------------------------------------------- #
# 4. tt_rank_slab: TT-SVD / QTT rank estimator on known-rank tensors
# --------------------------------------------------------------------------- #
def test_tt_svd_rank1_outer_product_is_rank1():
    rng = np.random.default_rng(0)
    u, v, w = (rng.standard_normal(6) for _ in range(3))
    t = u[:, None, None] * v[None, :, None] * w[None, None, :]
    cores, ranks = ttr.tt_svd(t, epsilon=1e-12)
    assert max(ranks) == 1
    rec = ttr.tt_reconstruct(cores)
    assert np.linalg.norm(rec - t) / np.linalg.norm(t) < 1e-10


def test_tt_svd_reconstruction_within_tolerance():
    rng = np.random.default_rng(1)
    t = rng.standard_normal((4, 5, 6, 3))
    cores, ranks = ttr.tt_svd(t, epsilon=1e-6)
    rec = ttr.tt_reconstruct(cores)
    assert np.linalg.norm(rec - t) / np.linalg.norm(t) <= 1e-6 + 1e-9


def test_qtt_smooth_below_noise():
    L = 256
    z = np.linspace(0, 1, L)
    smooth = np.exp(-((z - 0.5) ** 2) / (2 * 0.05 ** 2))
    noise = np.random.default_rng(2).standard_normal(L)
    r_s, i_s = ttr.qtt_ranks(smooth, epsilon=1e-8)
    r_n, i_n = ttr.qtt_ranks(noise, epsilon=1e-8)
    assert i_s["max_rank"] < i_n["max_rank"]


def test_synthetic_selftest_all_pass():
    checks = ttr.synthetic_selftest(epsilon=1e-8)
    assert checks["rank1_outer"]["pass"]
    assert checks["reconstruct_exact_lowrank"]
    assert checks["smooth_below_noise"]


def test_planar_average_shape():
    rho = np.random.default_rng(0).random((8, 10, 16))
    nz = ttr.planar_average(rho, normal_axis=2)
    assert nz.shape == (16,)
    assert nz == pytest.approx(rho.mean(axis=(0, 1)))
