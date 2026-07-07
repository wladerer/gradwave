"""k-batched plane-wave machinery (Layer A/B boundary).

Ragged per-k plane-wave counts are padded to npw_max with a mask; padded
slots carry zero coefficients and scatter into flat index 0 (adding zeros —
harmless). All heavy operations (FFTs, Hamiltonian applies, Rayleigh–Ritz)
then run as single batched tensor ops over (nk, nb, npw_max) — this is what
saturates BLAS/GPU instead of looping 36 small problems in Python.

Padded-slot invariants (everything relies on them):
  - coefficients: 0 in padded slots, always (enforced by `mask` multiplies)
  - kinetic t:    0 in padded slots (harmless in Teter: K(0) = 1, times r = 0)
  - flat_idx:     0 in padded slots (scatter adds 0 there; gather result is
                  discarded by the mask)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gradwave.constants import HBAR2_2M
from gradwave.dtypes import CDTYPE, RDTYPE


@dataclass
class BatchedK:
    """Padded per-k data for the batched SCF path."""

    npw: torch.Tensor  # (nk,) true plane-wave counts
    mask: torch.Tensor  # (nk, npw_max) bool
    flat_idx: torch.Tensor  # (nk, npw_max) int64, 0 in padding
    kpg: torch.Tensor  # (nk, npw_max, 3), 0 in padding
    t: torch.Tensor  # (nk, npw_max) kinetic (ħ²/2m)|k+G|², 0 in padding
    # projector data (empty first dim if no projectors)
    proj_phase_free: torch.Tensor  # (nk, nproj, npw_max) complex
    proj_atom_index: torch.Tensor  # (nproj,)
    dij_full: torch.Tensor  # (nproj, nproj)

    @property
    def nk(self) -> int:
        return int(self.npw.shape[0])

    @property
    def npw_max(self) -> int:
        return int(self.mask.shape[1])


def build_batched(spheres, proj_data, device=None) -> BatchedK:
    """Assemble padded batch tensors from per-k GSphere + ProjectorData lists."""
    nk = len(spheres)
    npw = torch.tensor([s.npw for s in spheres], dtype=torch.int64, device=device)
    m = int(npw.max())

    mask = torch.zeros(nk, m, dtype=torch.bool, device=device)
    flat_idx = torch.zeros(nk, m, dtype=torch.int64, device=device)
    kpg = torch.zeros(nk, m, 3, dtype=RDTYPE, device=device)
    t = torch.zeros(nk, m, dtype=RDTYPE, device=device)
    nproj = proj_data[0].f_ylm_phase_free.shape[0] if proj_data else 0
    pf = torch.zeros(nk, nproj, m, dtype=CDTYPE, device=device)

    for ik, (s, pd) in enumerate(zip(spheres, proj_data, strict=True)):
        n = s.npw
        mask[ik, :n] = True
        flat_idx[ik, :n] = s.flat_idx.to(device)
        kpg[ik, :n] = s.kpg.to(device)
        t[ik, :n] = HBAR2_2M * s.kpg2.to(device)
        if nproj:
            pf[ik, :, :n] = pd.f_ylm_phase_free.to(device)

    return BatchedK(
        npw=npw, mask=mask, flat_idx=flat_idx, kpg=kpg, t=t,
        proj_phase_free=pf,
        proj_atom_index=proj_data[0].atom_index.to(device) if nproj else
        torch.zeros(0, dtype=torch.int64, device=device),
        dij_full=proj_data[0].dij_full.to(device),
    )


def g_to_r_b(coeffs: torch.Tensor, bk: BatchedK, shape) -> torch.Tensor:
    """(nk, nb, npw_max) → (nk, nb, n1, n2, n3): f = Σ_G c e^{iGr}."""
    nk, nb, m = coeffs.shape
    n = shape[0] * shape[1] * shape[2]
    box = torch.zeros(nk, nb, n, dtype=coeffs.dtype, device=coeffs.device)
    idx = bk.flat_idx[:, None, :].expand(nk, nb, m)
    box = box.scatter_add(2, idx, coeffs)
    box = box.reshape(nk, nb, *shape)
    return torch.fft.ifftn(box, dim=(-3, -2, -1)) * n


def box_to_sphere_b(box: torch.Tensor, bk: BatchedK) -> torch.Tensor:
    """(nk, nb, n1, n2, n3) → coefficients (nk, nb, npw_max); masked."""
    nk, nb = box.shape[0], box.shape[1]
    n = box.shape[-3] * box.shape[-2] * box.shape[-1]
    coeff = torch.fft.fftn(box, dim=(-3, -2, -1)).reshape(nk, nb, n) / n
    idx = bk.flat_idx[:, None, :].expand(nk, nb, bk.npw_max)
    return coeff.gather(2, idx) * bk.mask[:, None, :]


def projectors_b(bk: BatchedK, positions: torch.Tensor) -> torch.Tensor:
    """Full projectors (nk, nproj, npw_max), differentiable in positions."""
    if bk.proj_phase_free.shape[1] == 0:
        return bk.proj_phase_free
    phase_arg = torch.einsum("kgi,ai->kga", bk.kpg, positions)  # (nk, npw, na)
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))
    return bk.proj_phase_free * phases[:, :, bk.proj_atom_index].permute(0, 2, 1)


class BatchedHamiltonian:
    """H apply for all k at once, fixed V_eff(r) and projectors (solver path).

    Uses a persistent scatter buffer with one extra "trash" slot: padded
    plane-wave slots write their zeros there instead of colliding with the
    true G=0 box entry (plain scatter assignment would otherwise be
    order-undefined). Non-sphere box entries are zeroed once at allocation
    and never written again. This is a no_grad fast path — the functional
    g_to_r_b/box_to_sphere_b remain the differentiable API.
    """

    def __init__(self, bk: BatchedK, shape, v_eff_r: torch.Tensor, p: torch.Tensor):
        self.bk = bk
        self.shape = shape
        self.n = shape[0] * shape[1] * shape[2]
        self.v_eff_r = v_eff_r
        self.p = p  # (nk, nproj, npw_max)
        # padded slots → trash index n (one past the box)
        self.idx_scatter = torch.where(
            bk.mask, bk.flat_idx, torch.full_like(bk.flat_idx, self.n)
        )
        self._box = None

    def _get_box(self, nk: int, nb: int, dtype, device):
        if (
            self._box is None
            or self._box.shape[0] != nk
            or self._box.shape[1] < nb
            or self._box.dtype != dtype
        ):
            self._box = torch.zeros(nk, nb, self.n + 1, dtype=dtype, device=device)
        return self._box[:, :nb]

    def _band_chunk(self, nk: int, device) -> int:
        """Bands per chunk so dense-box temporaries stay under ~380 MB on GPU
        (the apply chain holds ~4 such temporaries at once). CPU: no limit."""
        if device.type != "cuda":
            return 1_000_000
        return max(1, int(4e8 / (16 * self.n * max(nk, 1))))

    def apply(self, c: torch.Tensor) -> torch.Tensor:
        """(nk, nb, npw_max) → H c, mask preserved. Chunked over bands to
        bound peak memory on the dense grid (math identical)."""
        bk = self.bk
        nk, nb, m = c.shape
        out = bk.t[:, None, :] * c

        chunk = self._band_chunk(nk, c.device)
        for lo in range(0, nb, chunk):
            hi = min(lo + chunk, nb)
            cc = c[:, lo:hi]
            nbc = hi - lo
            box = self._get_box(nk, nbc, cc.dtype, cc.device)
            idx = self.idx_scatter[:, None, :].expand(nk, nbc, m)
            box.scatter_(2, idx, cc)
            psi = torch.fft.ifftn(box[..., : self.n].reshape(nk, nbc, *self.shape),
                                  dim=(-3, -2, -1))
            # fftn(ifftn(·)) is norm-neutral: the 1/N and ×N of the fftbox
            # conventions cancel, so no scaling factors here
            vg = torch.fft.fftn(psi * self.v_eff_r, dim=(-3, -2, -1)).reshape(nk, nbc, self.n)
            gath = bk.flat_idx[:, None, :].expand(nk, nbc, m)
            out[:, lo:hi] += vg.gather(2, gath)

        if self.p.shape[1]:
            b = torch.einsum("kpg,kbg->kbp", self.p.conj(), c)
            out = out + torch.einsum(
                "kbp,pq,kqg->kbg", b, bk.dij_full.to(c.dtype), self.p
            )
        return out * bk.mask[:, None, :]


def density_b(
    coeffs: torch.Tensor,  # (nk, nb, npw_max)
    occ: torch.Tensor,  # (nk, nb)
    kweights: torch.Tensor,  # (nk,)
    bk: BatchedK,
    shape,
    volume: float,
) -> torch.Tensor:
    """ρ(r) on the dense grid [e/Å³]. Band-chunked to bound dense-grid memory."""
    nk, nb, _ = coeffs.shape
    n = shape[0] * shape[1] * shape[2]
    if coeffs.device.type == "cuda":
        chunk = max(1, int(4e8 / (16 * n * max(nk, 1))))
    else:
        chunk = nb
    w = kweights[:, None] * occ
    rho = None
    for lo in range(0, nb, chunk):
        hi = min(lo + chunk, nb)
        psi = g_to_r_b(coeffs[:, lo:hi], bk, shape)
        contrib = torch.einsum(
            "kb,kbxyz->xyz", w[:, lo:hi].to(psi.real.dtype), psi.real**2 + psi.imag**2
        )
        rho = contrib if rho is None else rho + contrib
    return rho / volume


def becp_b(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """⟨p|ψ⟩ overlaps (nk, nb, nproj)."""
    return torch.einsum("kpg,kbg->kbp", p.conj(), c)
