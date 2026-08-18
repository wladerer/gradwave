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
from gradwave.core import opcount
from gradwave.core.fftbox import g_to_r_box
from gradwave.core.hamiltonian import ProjectorData
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import GSphere

# GPU dense-grid temporary budget [bytes]: bands are chunked so the ~4 dense-box
# temporaries the apply/density chain holds at once stay under this. Sizes a
# band chunk as budget / (elem_bytes · n_grid · nk). CPU paths do not chunk.
_GPU_DENSE_BUDGET_BYTES = 4e8

# Small-cell fast path for the local potential term V(r)·ψ(r). On the wavefunction
# G-sphere this term is EXACTLY the convolution out(G_i)=Σ_j V̂(G_i−G_j) c(G_j) =
# M @ c, with M[i,j]=V̂(G_i−G_j) (a Toeplitz/difference matrix; V̂=FFT of v_eff).
# One dense GEMM replaces the scatter → ifftn → ·v_eff → fftn → gather chain,
# deleting both FFTs and the irregular scatter/gather that dominate the small-cell
# apply. Bit-identical to the FFT path (it's an algebraic identity, not an
# approximation); measured ~1.5× whole-SCF on small metals. M is npw²·16 B, so the
# path is memory-gated: it activates only when the cached per-k matrix fits the
# budget below (nk·npw²·elem ≤ budget), which restricts it to the small cells where
# npw² beats the box FFT's N·logN. Disable globally by setting the flag False.
#
# The budget caps the cached matrix at nk·npw²·16 B (fp64 worst case); the
# difference-index table adds ~half that again, both held for the Hamiltonian's
# lifetime. The default (256 MiB) is deliberately conservative so the path never
# surprises a memory-tight GPU: it covers small cells and modest k-meshes (e.g.
# npw≈260 up to ~250 k-points). Raise it to extend coverage to finer meshes /
# larger npw when memory allows; set the flag False to disable entirely.
#
# CUDA is opt-in (default off). The whole-SCF win is measured and verified on CPU
# fp64 (dense GEMM beats the box FFT at small npw). On GPU the picture did NOT hold
# end-to-end on the tested hardware (RTX 3050): the isolated fp32 M@c is tensor-
# core-fast, but the non-apply fp64 FFTs (density build, Hartree, XC) dominate the
# SCF and Amdahl-dilute the gain, while fp64 GEMM is crippled on consumer GPUs so a
# pure-fp64 GPU SCF would REGRESS. A data-center GPU (real fp64 / saturation) may
# invert this — flip _TOEPLITZ_ON_CUDA to test there — but it stays off by default
# until validated so the path never silently slows a GPU run.
_TOEPLITZ_LOCAL_ENABLED = True
_TOEPLITZ_ON_CUDA = False  # opt-in: GPU whole-SCF win unproven; consumer fp64 regresses
_TOEPLITZ_M_BUDGET_BYTES = 1 << 28  # 256 MiB cap on the cached local-potential matrix

# Optional H-application instrumentation. BatchedHamiltonian.apply is the single
# chokepoint every batched Davidson round (NC and USPP/PAW) funnels its H|ψ⟩
# through, so tallying band·k applies here counts the eigensolver work a warm
# start saves. Disabled by default (one dict read per apply); a test enables it,
# resets, runs an SCF, then reads the accumulated count.
_HAPPLY_TALLY = {"on": False, "count": 0}


def reset_happly_tally() -> None:
    """Zero and enable the band·k H-application counter (see
    ``BatchedHamiltonian.apply``)."""
    _HAPPLY_TALLY["on"] = True
    _HAPPLY_TALLY["count"] = 0


def happly_tally() -> int:
    """Total band·k H-applications tallied since the last ``reset_happly_tally``."""
    return _HAPPLY_TALLY["count"]


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


def build_batched(
    spheres: list[GSphere], proj_data: list[ProjectorData], device: torch.device | None = None
) -> BatchedK:
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
        dij_full=proj_data[0].dij_full.to(device) if nproj else
        torch.zeros((0, 0), dtype=RDTYPE, device=device),
    )


def g_to_r_b(coeffs: torch.Tensor, bk: BatchedK, shape: tuple[int, int, int]) -> torch.Tensor:
    """(nk, nb, npw_max) → (nk, nb, n1, n2, n3): f = Σ_G c e^{iGr}."""
    nk, nb, m = coeffs.shape
    n = shape[0] * shape[1] * shape[2]
    box = torch.zeros(nk, nb, n, dtype=coeffs.dtype, device=coeffs.device)
    idx = bk.flat_idx[:, None, :].expand(nk, nb, m)
    box = box.scatter_add(2, idx, coeffs)
    box = box.reshape(nk, nb, *shape)
    return g_to_r_box(box)


def box_to_sphere_b(box: torch.Tensor, bk: BatchedK) -> torch.Tensor:
    """(nk, nb, n1, n2, n3) → coefficients (nk, nb, npw_max); masked."""
    nk, nb = box.shape[0], box.shape[1]
    n = box.shape[-3] * box.shape[-2] * box.shape[-1]
    opcount.bump("fft")
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

    def __init__(
        self,
        bk: BatchedK,
        shape: tuple[int, int, int],
        v_eff_r: torch.Tensor,
        p: torch.Tensor,
        hub_q: torch.Tensor | None = None,
        hub_dij: torch.Tensor | None = None,
        smooth: tuple[tuple[int, int, int], torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.bk = bk
        # dual grid (USPP/PAW): run the local-potential FFT on the smaller
        # smooth box. Exact for ⟨ψ|V|ψ⟩ (see uspp_setup). The kinetic and
        # nonlocal terms are sphere-based and untouched.
        if smooth is not None:
            shape, flat_idx, v_eff_r = smooth
        else:
            flat_idx = bk.flat_idx
        self.shape = shape
        self.n = shape[0] * shape[1] * shape[2]
        self.v_eff_r = v_eff_r
        self.gather_idx = flat_idx  # box → sphere for the local term
        self.p = p  # (nk, nproj, npw_max)
        # DFT+U: atomic-orbital projectors + density-dependent D-matrix, added
        # as a second nonlocal term (same becp contraction as KB).
        self.hub_q = hub_q  # (nk, nproj_U, npw_max)
        self.hub_dij = hub_dij  # (nproj_U, nproj_U) — already transposed for the apply
        # padded slots → trash index n (one past the box)
        self.idx_scatter = torch.where(
            bk.mask, flat_idx, torch.full_like(flat_idx, self.n)
        )
        self._box: torch.Tensor | None = None
        # Toeplitz small-cell fast path (see _TOEPLITZ_LOCAL_ENABLED). Eligible
        # only for the plain box (no USPP dual grid) and when the cached matrix
        # fits the budget. The difference-index table depends on geometry alone
        # (the Miller indices are fixed for the H's lifetime) so it is built once
        # here; M itself is built lazily per working precision and reused across
        # every apply (v_eff is fixed for this Hamiltonian).
        self._toep_idx: torch.Tensor | None = None
        self._toep_M_cache: dict[torch.dtype, torch.Tensor] = {}
        on_device = bk.mask.device.type == "cpu" or _TOEPLITZ_ON_CUDA
        if _TOEPLITZ_LOCAL_ENABLED and on_device and smooth is None:
            nk, npw = bk.mask.shape
            if nk * npw * npw * 16 <= _TOEPLITZ_M_BUDGET_BYTES:
                self._toep_idx = self._build_toeplitz_index()
        # cdtype → cast (t, v_eff, p, p_conj, dij)
        self._tab_cache: dict[
            torch.dtype, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        # cdtype → cast (hub_q, hub_q_conj, hub_dij)
        self._hub_cache: dict[torch.dtype, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _tables(
        self, cdtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cast tables to the working precision of the coefficients (cached).

        Feeding complex64 coefficients is not enough on its own: multiplying by
        an fp64 table promotes the result back to complex128. Precomputing
        matching-precision copies keeps the whole apply in fp32 when asked."""
        cached = self._tab_cache.get(cdtype)
        if cached is None:
            from gradwave.dtypes import real_of

            rdtype = real_of(cdtype)
            p = self.p.to(cdtype)
            cached = (
                self.bk.t.to(rdtype),
                self.v_eff_r.to(rdtype),
                p,
                # cached resolved conjugate: constant for the H's lifetime but
                # consumed every apply — materializing p.conj() per Davidson
                # round re-allocates the full projector table twice per round
                p.conj().resolve_conj(),
                self.bk.dij_full.to(cdtype),
            )
            self._tab_cache[cdtype] = cached
        return cached

    def _build_toeplitz_index(self) -> torch.Tensor:
        """(nk, npw, npw) box-flat index of G_i−G_j for the local-potential
        Toeplitz matrix. Recovers each sphere point's Miller triple from its
        box-flat index (gather_idx), takes the box-wrapped pairwise difference,
        and re-flattens. Padded slots reuse their raw box index; their M entries
        are harmless (the input is masked before the contraction and the output
        rows are masked after)."""
        shape = self.shape
        s1s2, s2 = shape[1] * shape[2], shape[2]
        flat = self.gather_idx.to(torch.long)  # (nk, npw) box → sphere
        g0 = flat // s1s2
        rem = flat % s1s2
        g = torch.stack([g0, rem // s2, rem % s2], dim=-1)  # (nk, npw, 3) Miller
        n = torch.tensor(shape, device=flat.device)
        diff = (g[:, :, None, :] - g[:, None, :, :]) % n  # (nk, npw, npw, 3)
        return diff[..., 0] * s1s2 + diff[..., 1] * s2 + diff[..., 2]

    def _toeplitz_M(self, cdtype: torch.dtype) -> torch.Tensor:
        """Local-potential matrix M[k,i,j]=V̂(G_i−G_j) at the working precision,
        cached for the H's lifetime (v_eff is fixed). V̂ is the FFT of v_eff on
        the dense box; indexing it by the geometry table gives M with no explicit
        difference loop."""
        M = self._toep_M_cache.get(cdtype)
        if M is None:
            from gradwave.core.fftbox import r_to_g

            assert self._toep_idx is not None
            vhat = r_to_g(self.v_eff_r.to(cdtype)).reshape(-1)
            M = vhat[self._toep_idx]
            self._toep_M_cache[cdtype] = M
        return M

    def _get_box(self, nk: int, nb: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (
            self._box is None
            or self._box.shape[0] != nk
            or self._box.shape[1] < nb
            or self._box.dtype != dtype
        ):
            self._box = torch.zeros(nk, nb, self.n + 1, dtype=dtype, device=device)
        return self._box[:, :nb]

    def _band_chunk(self, nk: int, device: torch.device, elem_bytes: int = 16) -> int:
        """Bands per chunk so dense-box temporaries stay under ~380 MB on GPU
        (the apply chain holds ~4 such temporaries at once). CPU: no limit.

        elem_bytes scales the budget by the coefficient precision: the fp32
        draft (complex64, 8 B) fits twice as many bands as fp64 (complex128,
        16 B), giving larger — and thus more efficient — batched FFTs."""
        if device.type != "cuda":
            return 1_000_000
        return max(1, int(_GPU_DENSE_BUDGET_BYTES / (elem_bytes * self.n * max(nk, 1))))

    def apply(self, c: torch.Tensor) -> torch.Tensor:
        """(nk, nb, npw_max) → H c, mask preserved. Chunked over bands to
        bound peak memory on the dense grid (math identical)."""
        bk = self.bk
        nk, nb, m = c.shape
        if _HAPPLY_TALLY["on"]:
            _HAPPLY_TALLY["count"] += nk * nb
        opcount.bump("hpsi", nk * nb)
        t_r, v_eff, p, p_conj, dij = self._tables(c.dtype)
        out = t_r[:, None, :] * c

        if self._toep_idx is not None:
            # Small-cell Toeplitz path: local V·ψ as one dense GEMM per k,
            # out(G_i) += Σ_j M[i,j] c(G_j). Mask the input so padded columns
            # (whose M entries are undefined) cannot pollute valid rows; the
            # output is masked again at the return, as in the FFT path.
            M = self._toeplitz_M(c.dtype)
            cm = c * bk.mask[:, None, :]
            out = out + torch.einsum("kij,kbj->kbi", M, cm)
        else:
            chunk = self._band_chunk(nk, c.device, c.element_size())
            for lo in range(0, nb, chunk):
                hi = min(lo + chunk, nb)
                cc = c[:, lo:hi]
                nbc = hi - lo
                box = self._get_box(nk, nbc, cc.dtype, cc.device)
                idx = self.idx_scatter[:, None, :].expand(nk, nbc, m)
                box.scatter_(2, idx, cc)
                opcount.bump("fft")
                psi = torch.fft.ifftn(box[..., : self.n].reshape(nk, nbc, *self.shape),
                                      dim=(-3, -2, -1))
                # fftn(ifftn(·)) is norm-neutral: the 1/N and ×N of the fftbox
                # conventions cancel, so no scaling factors here
                opcount.bump("fft")
                vg = torch.fft.fftn(psi * v_eff, dim=(-3, -2, -1)).reshape(nk, nbc, self.n)
                gath = self.gather_idx[:, None, :].expand(nk, nbc, m)
                out[:, lo:hi] += vg.gather(2, gath)

        if p.shape[1]:
            b = becp_b(p, c, p_conj)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b, dij, p)
        if self.hub_q is not None and self.hub_dij is not None:
            hq, hq_conj, hd = self._hub_tables(c.dtype)
            bh = torch.einsum("kpg,kbg->kbp", hq_conj, c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", bh, hd, hq)
        return out * bk.mask[:, None, :]

    def _hub_tables(self, cdtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._hub_cache.get(cdtype)
        if cached is None:
            # only called from apply()'s `self.hub_q is not None and self.hub_dij
            # is not None` branch, so both are guaranteed set here.
            assert self.hub_q is not None
            assert self.hub_dij is not None
            hq = self.hub_q.to(cdtype)
            cached = (hq, hq.conj().resolve_conj(), self.hub_dij.to(cdtype))
            self._hub_cache[cdtype] = cached
        return cached


def density_b(
    coeffs: torch.Tensor,  # (nk, nb, npw_max)
    occ: torch.Tensor,  # (nk, nb)
    kweights: torch.Tensor,  # (nk,)
    bk: BatchedK,
    shape: tuple[int, int, int],
    volume: float,
) -> torch.Tensor:
    """ρ(r) on the dense grid [e/Å³]. Band-chunked to bound dense-grid memory."""
    nk, nb, _ = coeffs.shape
    n = shape[0] * shape[1] * shape[2]
    if coeffs.device.type == "cuda":
        chunk = max(1, int(_GPU_DENSE_BUDGET_BYTES / (coeffs.element_size() * n * max(nk, 1))))
    else:
        chunk = nb
    w = kweights[:, None] * occ
    rho: torch.Tensor | None = None
    for lo in range(0, nb, chunk):
        hi = min(lo + chunk, nb)
        psi = g_to_r_b(coeffs[:, lo:hi], bk, shape)
        contrib = torch.einsum(
            "kb,kbxyz->xyz", w[:, lo:hi].to(psi.real.dtype), psi.real**2 + psi.imag**2
        )
        rho = contrib if rho is None else rho + contrib
    # nb >= 1 always (an SCF needs at least one band), so the loop runs at
    # least once and rho is always set here — same pattern as scf/loop.py's
    # own "runs >=1 iteration" narrowing (PR #182).
    assert rho is not None
    return rho / volume


def becp_b(p: torch.Tensor, c: torch.Tensor,
           p_conj: torch.Tensor | None = None) -> torch.Tensor:
    """⟨p|ψ⟩ overlaps (nk, nb, nproj). Pass a cached resolved conjugate via
    p_conj in per-round hot paths to skip re-materializing p.conj()."""
    pc = p.conj() if p_conj is None else p_conj
    return torch.einsum("kpg,kbg->kbp", pc, c)
