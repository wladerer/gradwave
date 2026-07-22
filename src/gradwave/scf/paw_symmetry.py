"""becsum symmetrization for USPP/PAW under the space group (Layer B).

With IBZ k-sums the becsum ρ^a_ij is computed on the wedge only; restoring
the full-group value requires averaging over operations, which mixes m
components within each l and permutes atoms:

    ρ^a_ij ← (1/N) Σ_op D^{l_i}(op) ρ^{map(op,a)} D^{l_j}(op)ᵀ

with D^l(W) the real-spherical-harmonic rotation matrices of the CARTESIAN
rotation S = Aᵀ W A⁻ᵀ. D is built numerically from the identity
Y_lm(S⁻¹ r̂) = Σ_m' D^l_{m'm} Y_lm'(r̂), projected on a Gauss–Legendre ×
uniform-φ sphere grid (exact for band-limited integrands) — no Wigner
formula conventions to get wrong. QE's PAW_symmetrize does the same job.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.core.gaunt import ylm_np


def _sphere_quad(lmax: int):
    from scipy.special import roots_legendre

    n = lmax + 2
    z, wz = roots_legendre(n)
    nphi = 2 * lmax + 3
    phi = np.arange(nphi) * (2.0 * math.pi / nphi)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    st = np.sqrt(1.0 - zz**2)
    dirs = np.stack([st * np.cos(pp), st * np.sin(pp), zz], -1).reshape(-1, 3)
    w = (wz[:, None] * np.full(nphi, 2.0 * math.pi / nphi)).reshape(-1)
    return dirs, w


def ylm_rotation_matrices(sg, cell: np.ndarray, lmax: int) -> list:
    """Per-op block-diagonal D matrices, one (2l+1)² block per l ≤ lmax.

    Returns [ops][l] → (2l+1, 2l+1) torch.float64 with
    Y_lm(S⁻¹ r̂) = Σ_m' D_{m'm} Y_lm'(r̂).
    """
    a_t = np.asarray(cell, dtype=float).T
    dirs, w = _sphere_quad(2 * lmax)
    y0 = ylm_np(lmax, dirs)  # (npt, (lmax+1)²)
    out = []
    for w_mat in sg.rotations:
        s = a_t @ w_mat @ np.linalg.inv(a_t)  # Cartesian rotation
        y_rot = ylm_np(lmax, dirs @ np.linalg.inv(s).T)  # Y(S⁻¹ r̂)
        blocks = []
        for ell in range(lmax + 1):
            sl = slice(ell * ell, (ell + 1) ** 2)
            # D_{m'm} = ∫ Y_lm'(r̂) Y_lm(S⁻¹r̂) dΩ
            d = np.einsum("pi,pj,p->ij", y0[:, sl], y_rot[:, sl], w)
            blocks.append(torch.as_tensor(d, dtype=torch.float64))
        out.append(blocks)
    return out


class MagneticBecsumSymmetrizer:
    """Pauli-channel becsum symmetrization under a magnetic (Shubnikov) group.

    Spatial half: BecsumSymmetrizer's per-op D^l blocks on the COMBINED op
    list (unitary then anti-unitary). Channel half: the n channel is a scalar;
    (mx, my, mz) mix through the axial 3×3  s_T·det(S)·S  paired with the SAME
    op as the D-blocks and atom map (s_T = −1 on the anti-unitary set). The
    pairing is fixed empirically: this orientation makes the group average
    idempotent (a projector — see the unit test); the transposed axial breaks
    idempotency outright. Anti-unitary ops additionally complex-conjugate the
    on-site matrix (T is anti-unitary; the projectors are real, so K is
    elementwise conj — a no-op on the Hermitized real becsum the spinor loop
    keeps, retained for correctness).
    """

    def __init__(self, mg, cell, paws, species_of_atom, atom_slices):
        self._bec = BecsumSymmetrizer(mg.combined(), cell, paws,
                                      species_of_atom, atom_slices)
        a_t = np.asarray(cell, dtype=float).T
        a_t_inv = np.linalg.inv(a_t)
        ax = []
        for iop, w_mat in enumerate(self._bec.sg.rotations):
            s = a_t @ w_mat @ a_t_inv
            r_ax = np.linalg.det(s) * s  # axial (pseudo-vector) action
            if iop >= mg.n_unitary:
                r_ax = -r_ax  # s_T: time reversal flips m⃗
            ax.append(r_ax)
        self.axial = torch.as_tensor(np.stack(ax), dtype=torch.float64)
        self.n_unitary = mg.n_unitary

    def to(self, device) -> MagneticBecsumSymmetrizer:
        self._bec.to(device)
        self.axial = self.axial.to(device)
        return self

    def apply(self, chans_atoms: list) -> list:
        """[4][na] Pauli channels [n_ij, mx_ij, my_ij, mz_ij] → symmetrized
        (same nesting; accumulation runs in complex128, real inputs come back
        real — the symmetrized imaginary part of a Hermitian channel is the
        antisymmetric piece the spinor loop discards anyway)."""
        sgc, bec = self._bec.sg, self._bec
        na = len(chans_atoms[0])
        acc = [[torch.zeros(m.shape, dtype=torch.complex128, device=m.device)
                for m in ch] for ch in chans_atoms]
        for iop in range(sgc.n_ops):
            amap, ax = sgc.atom_map[iop], self.axial[iop]
            anti = iop >= self.n_unitary
            for a in range(na):
                d = bec.d_full[iop][bec.species_of_atom[a]]
                src = [chans_atoms[c][int(amap[a])].to(torch.complex128)
                       for c in range(4)]
                if anti:
                    src = [x.conj() for x in src]
                acc[0][a] = acc[0][a] + d @ src[0] @ d.T
                for i in range(3):
                    mix = ax[i, 0] * src[1] + ax[i, 1] * src[2] + ax[i, 2] * src[3]
                    acc[i + 1][a] = acc[i + 1][a] + d @ mix @ d.T
        return [
            [(m / sgc.n_ops if ch0.is_complex() else (m / sgc.n_ops).real.to(ch0.dtype))
             for m, ch0 in zip(ch, chans_atoms[c], strict=True)]
            for c, ch in enumerate(acc)
        ]


class BecsumSymmetrizer:
    """Precomputed per-op rotation blocks expanded to the projector columns."""

    def __init__(self, sg, cell, paws, species_of_atom, atom_slices):
        self.sg = sg
        self.atom_slices = atom_slices
        lmax = max(b.l for p in paws for b in p.betas)
        d_ops = ylm_rotation_matrices(sg, cell, lmax)
        # expand to the m-expanded projector basis per species: block-diag of
        # D^{l_i} over channels (channels don't mix — same radial function)
        self.d_full = []  # [op][species] → (nm, nm)
        for iop in range(sg.n_ops):
            per_sp = []
            for p in paws:
                blocks = [d_ops[iop][b.l] for b in p.betas]
                per_sp.append(torch.block_diag(*blocks).to(torch.complex128))
            self.d_full.append(per_sp)
        self.species_of_atom = list(species_of_atom)

    def to(self, device) -> BecsumSymmetrizer:
        self.d_full = [[d.to(device) for d in per_sp] for per_sp in self.d_full]
        return self

    def apply(self, rho_ij_atoms: list) -> list:
        """ρ^a ← (1/N) Σ_op D ρ^{map(op,a)} Dᵀ (one spin channel)."""
        n_ops = self.sg.n_ops
        out = [torch.zeros_like(m) for m in rho_ij_atoms]
        for iop in range(n_ops):
            amap = self.sg.atom_map[iop]
            for a in range(len(rho_ij_atoms)):
                d = self.d_full[iop][self.species_of_atom[a]]
                out[a] = out[a] + d @ rho_ij_atoms[int(amap[a])] @ d.T
        return [m / n_ops for m in out]
