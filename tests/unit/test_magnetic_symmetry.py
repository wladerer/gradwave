"""Magnetic space groups (Shubnikov): op classification, magnetic-IBZ k-fold,
and (ρ, m⃗) field symmetrization (gradwave.symmetry).

The axial-vector filter is cross-checked against spglib.get_magnetic_symmetry;
the grey-group (m⃗ = 0) limit must reproduce the paramagnetic reduce_mesh with
time reversal EXACTLY (same reps, same weights). Textbook op counts:

  L1_0 FePt (P4/mmm, 16 ops), m ∥ [001]: 8 unitary (C4h) + 8 anti-unitary
  L1_0 FePt,                   m ∥ [100]: 4 unitary (C2h) + 4 anti, 8 dropped
  bcc Fe   (Im-3m, 48 ops),    m ∥ [001]: 8 unitary (C4h) + 8 anti, 32 dropped
"""

import numpy as np
import spglib
import torch

from gradwave.symmetry import (
    MagneticSymmetrizer,
    find_spacegroup,
    magnetic_spacegroup,
    reduce_mesh,
    reduce_mesh_magnetic,
)


def _fept():
    a, c = 2.723, 3.712
    cell = np.diag([a, a, c])
    frac = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]])
    species = [0, 1]
    return cell, frac, species, find_spacegroup(cell, frac, species)


def _bcc_fe():
    a = 2.87
    cell = a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    frac = np.array([[0.0, 0, 0]])
    species = [0]
    return cell, frac, species, find_spacegroup(cell, frac, species)


def _spglib_counts(cell, frac, species, magmoms):
    ds = spglib.get_magnetic_symmetry(
        (cell, frac, np.asarray(species) + 1, np.asarray(magmoms, float)),
        symprec=1e-6,
    )
    tr = np.asarray(ds["time_reversals"], dtype=bool)
    # one representative translation per unique (rotation, TR) — mirrors
    # find_spacegroup's centering dedup so counts compare like-for-like
    seen = set()
    n_u = n_a = 0
    for w, t in zip(ds["rotations"], tr, strict=True):
        key = (w.tobytes(), bool(t))
        if key in seen:
            continue
        seen.add(key)
        n_a += int(t)
        n_u += int(not t)
    return n_u, n_a


def test_grey_group_reproduces_paramagnetic_fold():
    cell, frac, species, sg = _fept()
    mg = magnetic_spacegroup(sg, [[0.0, 0, 0], [0, 0, 0]], cell)
    assert mg.n_unitary == sg.n_ops and mg.n_anti == sg.n_ops
    for mesh in [(2, 2, 2), (3, 3, 2), (4, 4, 3)]:
        k_ref, w_ref = reduce_mesh(mesh, (0, 0, 0), sg, time_reversal=True)
        k_mag, w_mag = reduce_mesh_magnetic(mesh, (0, 0, 0), mg)
        assert np.array_equal(k_ref, k_mag)
        assert np.array_equal(w_ref, w_mag)


def test_fept_moment_along_c():
    cell, frac, species, sg = _fept()
    assert sg.n_ops == 16  # P4/mmm
    moms = [[0, 0, 3.0], [0, 0, 0.4]]
    mg = magnetic_spacegroup(sg, moms, cell)
    assert (mg.n_unitary, mg.n_anti) == (8, 8)  # C4h + 8·T, nothing dropped
    assert _spglib_counts(cell, frac, species, moms) == (8, 8)

    # every unitary op must fix the z axis as an axial vector
    a_t = cell.T
    for w in mg.unitary.rotations:
        s = a_t @ w @ np.linalg.inv(a_t)
        assert np.allclose(np.linalg.det(s) * s @ [0, 0, 1.0], [0, 0, 1.0], atol=1e-10)

    # inversion is unitary (axial vectors are inversion-even), so −W⁻ᵀ of every
    # anti-unitary op is already a unitary k-action: the magnetic IBZ matches
    # the PARAMAGNETIC+TR fold exactly. 144 → 30 k (4.8×) at (6,6,4).
    k_mag, w_mag = reduce_mesh_magnetic((6, 6, 4), (0, 0, 0), mg)
    k_ref, w_ref = reduce_mesh((6, 6, 4), (0, 0, 0), sg, time_reversal=True)
    assert abs(w_mag.sum() - 1.0) < 1e-12
    assert len(k_mag) == 30
    assert np.array_equal(k_mag, k_ref) and np.array_equal(w_mag, w_ref)


def test_fept_moment_in_plane():
    cell, frac, species, sg = _fept()
    moms = [[3.0, 0, 0], [0.4, 0, 0]]
    mg = magnetic_spacegroup(sg, moms, cell)
    assert (mg.n_unitary, mg.n_anti) == (4, 4)  # C2h + 4·T; C4z-class dropped
    assert _spglib_counts(cell, frac, species, moms) == (4, 4)
    k_mag, _ = reduce_mesh_magnetic((6, 6, 4), (0, 0, 0), mg)
    assert len(k_mag) == 48  # 144 → 48 (3×): the C4z-class is genuinely lost


def test_bcc_fe_moment_along_z():
    cell, frac, species, sg = _bcc_fe()
    assert sg.n_ops == 48  # Im-3m
    moms = [[0, 0, 2.2]]
    mg = magnetic_spacegroup(sg, moms, cell)
    assert (mg.n_unitary, mg.n_anti) == (8, 8)  # O_h -> C4h, 32 dropped
    assert _spglib_counts(cell, frac, species, moms) == (8, 8)
    # unitary and anti-unitary sets are disjoint once moments are nonzero
    keys_u = {w.tobytes() for w in mg.unitary.rotations}
    keys_a = {w.tobytes() for w in mg.anti_rotations}
    assert not keys_u & keys_a

    k_mag, w_mag = reduce_mesh_magnetic((4, 4, 4), (0, 0, 0), mg)
    assert len(k_mag) == 13  # 64 → 13 (4.9×)
    assert abs(w_mag.sum() - 1.0) < 1e-12


def _mag_symmetrizer(shape=(12, 12, 10)):
    """FePt m∥[001] symmetrizer on a small box with a safe sub-Nyquist mask."""
    cell, frac, species, sg = _fept()
    mg = magnetic_spacegroup(sg, [[0, 0, 3.0], [0, 0, 0.4]], cell)
    millers = np.stack(
        np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n).astype(int) for n in shape],
                    indexing="ij"),
        axis=-1,
    )
    mask = torch.as_tensor(
        (np.abs(millers) <= np.array([n // 3 for n in shape])).all(-1))
    return MagneticSymmetrizer(shape, mg, cell, dens_mask=mask), shape


def test_field_symmetrizer_idempotent_and_invariant():
    ms, shape = _mag_symmetrizer()
    gen = torch.Generator().manual_seed(7)
    rho = torch.randn(shape, dtype=torch.float64, generator=gen)
    m = torch.randn(3, *shape, dtype=torch.float64, generator=gen)
    rho_g = torch.fft.fftn(rho.to(torch.complex128), dim=(-3, -2, -1))
    m_g = torch.fft.fftn(m.to(torch.complex128), dim=(-3, -2, -1))

    rho_s, m_s = ms.apply(rho_g), ms.apply_m(m_g)
    assert torch.allclose(ms.apply(rho_s), rho_s, atol=1e-11)
    assert torch.allclose(ms.apply_m(m_s), m_s, atol=1e-11)

    # per-op invariance of the symmetrized m⃗: m_a(G) = ax_ab phase m_b(W^T G)
    rs = ms.rho_sym
    flat = m_s.reshape(3, -1)
    for iop in range(rs.idx.shape[0]):
        mapped = torch.einsum(
            "ab,bn->an", ms.axial[iop].to(flat.dtype),
            flat[:, rs.idx[iop]]) * rs.phase[iop]
        assert torch.allclose(mapped * rs.mask, flat * rs.mask, atol=1e-11)

    # symmetrized fields are real in r-space (the group maps G to -G partners
    # consistently; a real input must stay real)
    m_r = torch.fft.ifftn(m_s, dim=(-3, -2, -1))
    assert float(m_r.imag.abs().max()) < 1e-11


def test_field_symmetrizer_uniform_moments():
    ms, shape = _mag_symmetrizer()
    # uniform m⃗ ∥ z is the magnetic axis: preserved exactly.
    # uniform m⃗ ∥ x has no invariant component under C4h(z): averaged to zero.
    for direction, survives in (([0, 0, 1.0], True), ([1.0, 0, 0], False)):
        m = torch.zeros(3, *shape, dtype=torch.float64)
        for i, v in enumerate(direction):
            m[i] = v
        m_g = torch.fft.fftn(m.to(torch.complex128), dim=(-3, -2, -1))
        m_s = torch.fft.ifftn(ms.apply_m(m_g), dim=(-3, -2, -1)).real
        if survives:
            assert torch.allclose(m_s, m, atol=1e-11)
        else:
            assert float(m_s.abs().max()) < 1e-11


def test_magnetic_becsum_symmetrizer():
    """bct O-pair (I4/mmm after translation dedup, atom-swapping ops included),
    FM m∥z: symmetrized Pauli becsum channels must be idempotent, invariant
    under every single-op action, and keep mx=my=0 for a collinear input."""
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.paw_symmetry import MagneticBecsumSymmetrizer

    o = parse_upf_paw("tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF")
    a, c = 3.0, 4.1
    cell = np.diag([a, a, c])
    frac = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]])
    sg = find_spacegroup(cell, frac, [0, 0])
    mg = magnetic_spacegroup(sg, [[0, 0, 1.0], [0, 0, 1.0]], cell)
    assert (mg.n_unitary, mg.n_anti) == (8, 8)
    ms = MagneticBecsumSymmetrizer(mg, cell, [o], [0, 0], None)

    nm = sum(2 * b.l + 1 for b in o.betas)
    gen = torch.Generator().manual_seed(3)

    def rand_sym():
        x = torch.randn(nm, nm, dtype=torch.float64, generator=gen)
        return 0.5 * (x + x.T)

    chans = [[rand_sym() for _ in range(2)] for _ in range(4)]
    sym1 = ms.apply(chans)
    sym2 = ms.apply(sym1)
    for c1, c2 in zip(sym1, sym2, strict=True):
        for m1, m2 in zip(c1, c2, strict=True):
            assert torch.allclose(m1, m2, atol=1e-12)

    # single-op invariance: one term of apply (times N) must fix the output
    sgc, bec = ms._bec.sg, ms._bec
    for iop in range(sgc.n_ops):
        amap, ax = sgc.atom_map[iop], ms.axial[iop]
        for at in range(2):
            d = bec.d_full[iop][0].real  # real inputs, real D blocks
            src = [sym1[ch][int(amap[at])] for ch in range(4)]
            assert torch.allclose(d @ src[0] @ d.T, sym1[0][at], atol=1e-11)
            for i in range(3):
                mix = ax[i, 0] * src[1] + ax[i, 1] * src[2] + ax[i, 2] * src[3]
                assert torch.allclose(d @ mix @ d.T, sym1[i + 1][at], atol=1e-11)

    # collinear input (mx=my=0) stays collinear under a z-axis magnetic group
    zeros = [torch.zeros(nm, nm, dtype=torch.float64) for _ in range(2)]
    col = [[rand_sym(), rand_sym()], list(zeros), list(zeros),
           [rand_sym(), rand_sym()]]
    scol = ms.apply(col)
    assert float(scol[1][0].abs().max()) < 1e-13
    assert float(scol[2][1].abs().max()) < 1e-13
