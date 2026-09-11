"""Fast-tier unit tests for supercell band-structure unfolding fold-map and
residue-mask integer identities (no SCF). The physics controls (perfect Si
supercell + defect) live in tests/integration/test_unfold_si.py (standard tier).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from gradwave.postscf.unfold import (
    as_supercell_matrix,
    fold_map,
    residue_mask,
)


def test_as_supercell_matrix_diagonal_and_full():
    assert np.array_equal(as_supercell_matrix([2, 2, 2]), np.diag([2, 2, 2]))
    M = [[1, 1, 0], [0, 1, 0], [0, 0, 2]]
    assert np.array_equal(as_supercell_matrix(M), np.asarray(M))


@pytest.mark.parametrize("bad", [
    [2, 2],                       # wrong length
    [[1, 0], [0, 1]],             # wrong shape
    [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],  # non-integer
    [[1, 0, 0], [2, 0, 0], [0, 0, 1]],    # singular (det 0)
])
def test_as_supercell_matrix_rejects(bad):
    with pytest.raises(ValueError):
        as_supercell_matrix(bad)


def test_fold_map_dedup_and_offset_identity():
    # A primitive path along Γ→X→2X (in units of the primitive reciprocal) with
    # M = diag(2,2,2): k=(0.5,0,0) folds to K=(1,0,0)≡(0,0,0), same reduced image
    # as Γ. k=(0.25,0,0) folds to (0.5,0,0) — a distinct image. The dedup must
    # collapse the first and third onto one unique K.
    M = np.diag([2, 2, 2])
    kpts = np.array([
        [0.0, 0.0, 0.0],    # Γ  -> K=(0,0,0)
        [0.25, 0.0, 0.0],   #    -> K=(0.5,0,0)
        [0.5, 0.0, 0.0],    # X  -> K=(1,0,0) ≡ (0,0,0)
    ])
    fm = fold_map(M, kpts)
    # exact fold identity: raw = k @ Mᵀ = K_reduced + offset (integer)
    raw = kpts @ M.T
    assert np.allclose(fm.kpts_super + fm.offset, raw)
    assert np.all(fm.offset.astype(int) == np.rint(raw).astype(int))
    # Γ and X share one reduced image; the midpoint is distinct
    assert fm.image[0] == fm.image[2]
    assert fm.image[1] != fm.image[0]
    assert len(fm.unique_K) == 2  # dedup fired


def test_fold_map_reduces_into_half_open_bz():
    M = np.diag([3, 1, 1])
    kpts = np.array([[0.9, 0.0, 0.0]])  # raw K0 = 2.7 -> reduce to -0.3
    fm = fold_map(M, kpts)
    assert np.all(fm.kpts_super >= -0.5 - 1e-9)
    assert np.all(fm.kpts_super < 0.5 + 1e-9)


def test_residue_mask_2x1x1_gamma_and_zone_boundary():
    # M = diag(2,1,1): M⁻ᵀ = diag(0.5,1,1). A hand-checkable case.
    M = np.diag([2, 1, 1])
    Minv_T = np.linalg.inv(M.astype(float)).T
    miller = torch.tensor(
        [[g0, g1, 0] for g0 in range(-2, 3) for g1 in range(-1, 2)],
        dtype=torch.int64)

    # Γ (offset 0): (G)·M⁻ᵀ integer ⟺ first Miller index even.
    m0 = residue_mask(miller, np.array([0, 0, 0]), Minv_T).numpy()
    expect0 = (miller[:, 0].numpy() % 2 == 0)
    assert np.array_equal(m0, expect0)

    # Zone boundary k=(0.5,0,0): raw = (1,0,0), offset n=(1,0,0), reduced K=0.
    # (G - n)·M⁻ᵀ integer ⟺ first Miller index ODD — the wraparound offset flips
    # which coset carries the primitive character.
    fm = fold_map(M, np.array([[0.5, 0.0, 0.0]]))
    m1 = residue_mask(miller, fm.offset[0], Minv_T).numpy()
    expect1 = (miller[:, 0].numpy() % 2 == 1)
    assert np.array_equal(m1, expect1)
    # disjoint + covering: Γ and boundary cosets tile the sphere
    assert np.array_equal(m0 | m1, np.ones_like(m0))
    assert not np.any(m0 & m1)


def test_unitarity_weights_sum_to_one_over_residue_classes():
    # M = diag(2,2,2): the 8 primitive-k images folding to K=0 are the parity
    # cosets (i,j,k)∈{0,1}³. Their residue masks must partition the sphere, so
    # Σ_classes P = Σ_G |c|² = 1 for any normalized supercell eigenvector.
    rng = np.random.default_rng(0)
    M = np.diag([2, 2, 2])
    Minv_T = np.linalg.inv(M.astype(float)).T
    miller = torch.tensor(
        list(itertools.product(range(-2, 3), repeat=3)), dtype=torch.int64)
    npw = miller.shape[0]
    c = torch.tensor(rng.standard_normal(npw) + 1j * rng.standard_normal(npw))
    c = c / c.abs().pow(2).sum().sqrt()  # normalize Σ|c|² = 1

    masks = []
    total = 0.0
    for i, j, k in itertools.product(range(2), repeat=2 + 1):
        offset = np.array([i, j, k], dtype=np.int64)
        m = residue_mask(miller, offset, Minv_T)
        masks.append(m.numpy())
        total += float((c.abs() ** 2 * m).sum())
    assert total == pytest.approx(1.0, abs=1e-12)
    # exact partition: every G in exactly one coset
    stacked = np.stack(masks).astype(int).sum(axis=0)
    assert np.array_equal(stacked, np.ones(npw, dtype=int))
