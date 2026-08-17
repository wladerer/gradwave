"""Little group of q / star of q — group-theory identities (Phase 0 of the
symmetry-breaking star-unfold; see docs/design/little-group-star-unfold.md)."""

import numpy as np
import pytest
import torch

from gradwave.grids import build_fft_grid
from gradwave.symmetry import (
    QFieldSymmetrizer,
    RhoSymmetrizer,
    coupled_axis_groups,
    find_spacegroup,
    little_cogroup,
    little_group_ibz,
    reduce_mesh,
    star_of_q,
)

pytestmark = pytest.mark.standard

RY = 13.605693122994


def _grid_sg_cell(cell, frac, species, ecut_ry=20):
    sg = find_spacegroup(cell, frac, species)
    grid = build_fft_grid(cell, ecut_ry * RY, equal_dims=coupled_axis_groups(sg))
    return grid, sg, np.asarray(cell, float)


def _si():
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    frac = np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]])
    return _grid_sg_cell(cell, frac, [0, 0])


def _fe():
    a = 2.87
    cell = a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    return _grid_sg_cell(cell, np.zeros((1, 3)), [0], ecut_ry=30)


def _qsym(grid, q, sg, cell):
    return QFieldSymmetrizer(grid.shape, q, sg, cell, grid.g2, grid.dens_mask)


def _fcc_si():
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    frac = np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]])
    return cell, find_spacegroup(cell, frac, [0, 0])


def _bcc_fe():
    a = 2.87
    cell = a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    frac = np.zeros((1, 3))
    return cell, find_spacegroup(cell, frac, [0])


@pytest.mark.parametrize("builder", [_fcc_si, _bcc_fe])
@pytest.mark.parametrize("q", [
    [0.0, 0.0, 0.0],          # Gamma: full group
    [0.5, 0.0, 0.0],          # a zone-boundary point
    [0.25, 0.0, 0.0],         # a general line point
    [0.25, 0.25, 0.0],
    [0.5, 0.5, 0.5],
])
def test_orbit_stabilizer(builder, q):
    """|star(q)| * |little co-group(q)| == |point group|."""
    _cell, sg = builder()
    lg, g0 = little_cogroup(q, sg)
    qs, reps = star_of_q(q, sg)
    assert len(g0) == lg.n_ops
    assert len(qs) == len(reps)
    assert len(qs) * lg.n_ops == sg.n_ops, (len(qs), lg.n_ops, sg.n_ops)


def test_gamma_reduces_to_full_group():
    """q=Γ: little co-group is the full group, star is a single point, and the
    little-group IBZ equals the ordinary (TR-off) IBZ."""
    _cell, sg = _fcc_si()
    lg, g0 = little_cogroup([0, 0, 0], sg)
    assert lg.n_ops == sg.n_ops
    assert np.all(g0 == 0)
    qs, _ = star_of_q([0, 0, 0], sg)
    assert len(qs) == 1
    k_lg, w_lg = little_group_ibz((4, 4, 4), [0, 0, 0], sg, time_reversal=False)
    k_full, w_full = reduce_mesh((4, 4, 4), (0, 0, 0), sg, time_reversal=False)
    assert len(k_lg) == len(k_full)
    assert abs(w_lg.sum() - 1.0) < 1e-12


def test_umklapp_fixes_q():
    """Every little-co-group op maps q back to q modulo the reported umklapp."""
    _cell, sg = _fcc_si()
    q = np.array([0.5, 0.0, 0.0])
    lg, g0 = little_cogroup(q, sg)
    for w, g in zip(lg.rotations, g0, strict=True):
        w_inv_t = np.round(np.linalg.inv(w).T).astype(np.int64)
        assert np.allclose(w_inv_t @ q - q, g, atol=1e-9)


def test_star_reps_generate_star():
    """Each star member is W⁻ᵀ q of its reported rep op (mod 1)."""
    _cell, sg = _fcc_si()
    q = np.array([0.25, 0.0, 0.0])
    qs, reps = star_of_q(q, sg)
    for qi, r in zip(qs, reps, strict=True):
        w_inv_t = np.round(np.linalg.inv(sg.rotations[r]).T).astype(np.int64)
        d = (w_inv_t @ q - qi)
        assert np.max(np.abs(d - np.round(d))) < 1e-9


def test_little_group_ibz_ge_full_ibz():
    """The little-group IBZ is at least as large as the full-group IBZ (fewer
    ops fix q, so less reduction) and never exceeds the full mesh."""
    _cell, sg = _fcc_si()
    q = [0.25, 0.0, 0.0]
    k_lg, _ = little_group_ibz((4, 4, 4), q, sg, time_reversal=False)
    k_full, _ = reduce_mesh((4, 4, 4), (0, 0, 0), sg, time_reversal=False)
    assert len(k_full) <= len(k_lg) <= 4 ** 3


def test_qfield_gamma_matches_rho_symmetrizer():
    """At q=Γ the QFieldSymmetrizer is the same group-average projector as
    RhoSymmetrizer (full group, g0=0, and the q-shifted mask = density mask)."""
    grid, sg, cell = _si()
    qsym = _qsym(grid, [0.0, 0.0, 0.0], sg, cell)
    rsym = RhoSymmetrizer(grid.shape, sg, dens_mask=grid.dens_mask)
    torch.manual_seed(0)
    c = torch.randn(grid.shape, dtype=torch.complex128)
    rel = float((qsym.apply(c) - rsym.apply(c)).abs().max() / rsym.apply(c).abs().max())
    assert rel < 1e-12, rel


@pytest.mark.parametrize("q", [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
def test_qfield_idempotent_ordinary(q):
    """Ordinary small rep: the little-co-group average is a projector,
    P(P c) == P c. [0.5,0,0] exercises the hard features together — nonsymmorphic
    (glide) ops AND nonzero umklapps g0 — so its machine-precision idempotence is
    the decisive check that the q-shifted mask + gather construction is right."""
    grid, sg, cell = _si()
    qsym = _qsym(grid, q, sg, cell)
    torch.manual_seed(1)
    c = torch.randn(grid.shape, dtype=torch.complex128)
    pc = qsym.apply(c)
    rel = float((qsym.apply(pc) - pc).abs().max() / max(1e-30, float(pc.abs().max())))
    assert rel < 1e-10, rel


@pytest.mark.parametrize("q", [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]])
def test_qfield_symmorphic_zone_boundary_idempotent(q):
    """A symmorphic crystal (bcc Fe, Im-3m) has an ordinary small rep even at the
    zone boundary — with the q-shifted mask the projector is idempotent there,
    which the ordinary density mask (centred at G=0) would break for g0≠0."""
    grid, sg, cell = _fe()
    qsym = _qsym(grid, q, sg, cell)
    torch.manual_seed(3)
    c = torch.randn(grid.shape, dtype=torch.complex128)
    pc = qsym.apply(c)
    rel = float((qsym.apply(pc) - pc).abs().max() / max(1e-30, float(pc.abs().max())))
    assert rel < 1e-10, rel


def test_qfield_is_a_projector_not_identity():
    """P genuinely projects (changes a generic field) when the little group is
    non-trivial."""
    grid, sg, cell = _si()
    q = [0.25, 0.0, 0.0]
    assert little_cogroup(q, sg)[0].n_ops > 1
    qsym = _qsym(grid, q, sg, cell)
    torch.manual_seed(2)
    c = torch.randn(grid.shape, dtype=torch.complex128) * qsym.mask.reshape(grid.shape)
    changed = float((qsym.apply(c) - c).abs().max() / c.abs().max())
    assert changed > 1e-3, changed


def test_qfield_projective_rep_raises():
    """Non-symmorphic Si (diamond glide) has PROJECTIVE small reps at certain q
    (here [1/4,1/4,0], whose little group carries glide ops with a non-trivial
    factor system) — the plain little-co-group average is not a projector, so
    construction fails fast rather than fold with a wrong operator. The response
    fold must use the full mesh at such q (see the design note)."""
    grid, sg, cell = _si()
    with pytest.raises(NotImplementedError, match="projective"):
        _qsym(grid, [0.25, 0.25, 0.0], sg, cell)
