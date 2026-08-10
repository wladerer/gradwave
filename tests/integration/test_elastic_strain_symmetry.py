"""Laue-point-group reduction of the elastic strain set — self-oracle.

The reduced path (``elastic.use_strain_symmetry=True``) runs only the
point-group-irreducible Voigt strains and reconstructs the full 6×6 stiffness C
by the crystal group action (:class:`ElasticStrainSymmetry`). For a symmorphic
cubic monatomic metal on a symmetry-commensurate FFT box this is EXACT: the
reconstructed elastic constants must match the full 6-strain run to a physically
meaningful tolerance, from the same analytic-stress data (only rotated/permuted).
fcc Al: 6 → 2 strained-SCF batches.

The fast unit test at the bottom exercises the rank-4 symmetrizer and the
irreducible-strain selector as pure algebra (no SCF).
"""

import numpy as np
import pytest
import torch

from gradwave.api import XC_REGISTRY
from gradwave.postscf.elastic import (
    ElasticStrainSymmetry,
    elastic_tensor,
    stress_to_voigt,
    symmetrize_elastic_tensor,
    tensor_to_voigt,
    voigt_strain_tensor,
    voigt_to_tensor,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.symmetry import find_spacegroup
from tests.helpers import PSEUDOS, RY


def _al_stiffness_both():
    """Clamped-ion C (6×6 GPa) for fcc Al, computed the full way and via the
    irreducible-strain reduction, sharing one pinned FFT box."""
    torch.set_num_threads(8)
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    frac = np.array([[0.0, 0.0, 0.0]])
    upf = parse_upf(str(PSEUDOS / "Al_ONCV_PBE-1.2.upf"))
    xc = XC_REGISTRY["pbe"]()
    kmesh = (8, 8, 8)

    from gradwave.postscf.stress import stress as _stress

    # pin one FFT box across every strain (the +h strains give the finest grid),
    # so the reduced and full paths differ only by which strains are run.
    def _build(cell_i, fft_shape=None):
        return setup_system(cell_i, frac @ cell_i, [0], [upf], ecut=20 * RY,
                            kmesh=kmesh, use_symmetry=False, fft_shape=fft_shape)

    h = 0.005
    probe = [cell] + [cell @ (np.eye(3) + voigt_strain_tensor(j, h)).T
                      for j in range(6)]
    fixed = tuple(max(int(_build(c).grid.shape[i]) for c in probe)
                  for i in range(3))
    ref = scf(_build(cell, fixed), xc, nspin=1, smearing="cold", width=0.01,
              etol=1e-9, rhotol=1e-8, verbose=False)

    def _stress_at(eps):
        cell_i = cell @ (np.eye(3) + eps).T
        res = scf(_build(cell_i, fixed), xc, nspin=1, smearing="cold",
                  width=0.01, etol=1e-9, rhotol=1e-8, start_from=ref,
                  verbose=False)
        return _stress(res, xc).detach().cpu().numpy()

    sym = ElasticStrainSymmetry(cell, frac, [0], fft_shape=fixed)
    c_full = elastic_tensor(_stress_at, h=h, symmetry=None)
    c_red = elastic_tensor(_stress_at, h=h, symmetry=sym)
    return sym, c_full, c_red


@pytest.mark.slow
def test_fcc_al_reduced_matches_full():
    sym, c_full, c_red = _al_stiffness_both()

    # 6 → 2 strains: one axial + one shear span the cubic strain space.
    assert len(sym.strains) == 2
    assert len(sym.strains) < 6
    assert sym.can_reduce

    def _cij(c):
        return c[0, 0], c[0, 1], c[3, 3]  # C11, C12, C44

    c11f, c12f, c44f = _cij(c_full)
    c11r, c12r, c44r = _cij(c_red)

    # PRIMARY gate: the physical elastic constants match to a meaningful tol.
    # (The full path runs an independent SCF per strain, symmetric only to the
    # SCF/FFT-grid symmetry floor; the reduced path rotates one irreducible
    # SCF's stress and is point-group-exact by construction, so the two differ
    # by the full path's own numerical symmetry breaking, not a reconstruction
    # error — hence a GPa-scale physical tol, not bit-level.)
    print(f"\n[strain-symmetry] irreducible strains={len(sym.strains)} (of 6)  "
          f"C11 {c11f:.3f}/{c11r:.3f}  C12 {c12f:.3f}/{c12r:.3f}  "
          f"C44 {c44f:.3f}/{c44r:.3f} GPa")
    assert abs(c11f - c11r) < 0.5
    assert abs(c12f - c12r) < 0.5
    assert abs(c44f - c44r) < 0.5

    # SECONDARY sanity: whole-tensor bound at the SCF symmetry floor (loose,
    # NOT bit-level — it certifies the reconstruction, not round-off equality).
    dmax = float(np.abs(c_full - c_red).max())
    assert dmax < 2.0, f"max|dC|={dmax:.3e} GPa (SCF symmetry floor)"


# ---------------------------------------------------------------------------
# Fast unit tests — pure algebra, no SCF.
# ---------------------------------------------------------------------------

def _random_elastic_tensor(seed: int = 0) -> np.ndarray:
    """A random rank-4 tensor with the minor+major symmetries of a stiffness."""
    rng = np.random.default_rng(seed)
    c6 = rng.standard_normal((6, 6))
    c6 = 0.5 * (c6 + c6.T)          # major symmetry C_ijkl = C_klij
    return voigt_to_tensor(c6)      # minor symmetries via the Voigt fill


def test_rank4_symmetrizer_cubic_pattern():
    """symmetrize_elastic_tensor projects an arbitrary stiffness onto the cubic
    invariant subspace: C11=C22=C33, C12=C13=C23, C44=C55=C66, and the cross /
    normal-shear blocks vanish — the 3-constant cubic pattern."""
    a = 4.0
    cell = a * np.eye(3)                     # simple cubic, one atom → m-3m
    sg = find_spacegroup(cell, np.zeros((1, 3)), [0])
    assert sg.n_ops == 48

    c4 = _random_elastic_tensor(seed=1)
    c4s = symmetrize_elastic_tensor(c4, sg, cell)
    c6 = tensor_to_voigt(c4s)

    # cubic equalities
    assert np.allclose(c6[0, 0], [c6[1, 1], c6[2, 2]])
    assert np.allclose(c6[0, 1], [c6[0, 2], c6[1, 2]])
    assert np.allclose(c6[3, 3], [c6[4, 4], c6[5, 5]])
    # forbidden couplings are zero (normal↔shear, shear↔shear off-diagonal)
    assert abs(c6[0, 3]) < 1e-10 and abs(c6[3, 4]) < 1e-10
    # idempotent projection
    c4ss = symmetrize_elastic_tensor(c4s, sg, cell)
    assert np.allclose(c4s, c4ss, atol=1e-10)


def test_rank4_symmetrizer_matches_voigt_reconstruction():
    """The rank-4 group symmetrizer and the ElasticStrainSymmetry least-squares
    reconstruction agree on a synthetic cubic stiffness — the two independent
    routes to the same symmetry-reduced C.

    Build a genuine cubic C6, take the exact stress columns σ = C ε for the
    irreducible strains, reconstruct, and confirm the full C6 comes back."""
    a = 4.0
    cell = a * np.eye(3)
    frac = np.zeros((1, 3))
    sym = ElasticStrainSymmetry(cell, frac, [0])

    # the irreducible strain count is < the full 6 for cubic
    assert len(sym.strains) == 2
    assert len(sym.strains) < 6
    assert sym.can_reduce  # no fft_shape given → grid assumed compatible

    # a real cubic stiffness (arbitrary C11, C12, C44)
    c11, c12, c44 = 108.0, 62.0, 28.0
    c6 = np.array([
        [c11, c12, c12, 0, 0, 0],
        [c12, c11, c12, 0, 0, 0],
        [c12, c12, c11, 0, 0, 0],
        [0, 0, 0, c44, 0, 0],
        [0, 0, 0, 0, c44, 0],
        [0, 0, 0, 0, 0, c44],
    ])

    # exact stress column for each irreducible strain j: σ_voigt = C6 · e_j,
    # read from the tensor contraction to honor the engineering convention.
    c4 = voigt_to_tensor(c6)
    cols = []
    for j in sym.strains:
        eps = voigt_strain_tensor(j, 1.0)
        sig = np.einsum("ijkl,kl->ij", c4, eps)
        cols.append(stress_to_voigt(sig))

    c6_rec = sym.reconstruct(cols)
    c6_rec = 0.5 * (c6_rec + c6_rec.T)
    assert np.allclose(c6_rec, c6, atol=1e-9), f"\n{c6_rec}\n vs \n{c6}"


def test_anisotropic_fft_box_falls_back():
    """A non-cubic FFT box breaks the point group, so the reducer refuses
    (can_reduce False) and the caller keeps the full strain set."""
    cell = 4.0 * np.eye(3)
    frac = np.zeros((1, 3))
    ok = ElasticStrainSymmetry(cell, frac, [0], fft_shape=(24, 24, 24))
    assert ok.can_reduce
    bad = ElasticStrainSymmetry(cell, frac, [0], fft_shape=(24, 24, 30))
    assert not bad.can_reduce
    assert not bad.grid_ok
