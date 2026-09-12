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

import os
import time
from collections.abc import Callable
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
# band chunk as budget / (elem_bytes · n_grid · nk). CPU paths do not chunk by
# default (opt in with GRADWAVE_CPU_DENSE_BUDGET — see _cpu_dense_budget_bytes).
# GRADWAVE_GPU_DENSE_BUDGET overrides (bytes, read per call); chunking is
# BIT-EXACT — only the batch tiling changes.
# [D-003] budget defaults & precedence rationale — docs/design/decision-records.md
_GPU_DENSE_BUDGET_BYTES = 4e8


def _gpu_dense_budget_bytes() -> float:
    """Dense-grid band-chunk budget in bytes: GRADWAVE_GPU_DENSE_BUDGET when set
    (must be > 0), else the 4e8 default. Lowering it shrinks the per-chunk FFT-box
    peak without changing any result (see `_dense_band_chunk`)."""
    raw = os.environ.get("GRADWAVE_GPU_DENSE_BUDGET")
    if raw is None:
        return _GPU_DENSE_BUDGET_BYTES
    budget = float(raw)
    if budget <= 0:
        raise ValueError(
            f"GRADWAVE_GPU_DENSE_BUDGET must be > 0 bytes, got {budget!r}")
    return budget


# Process-local CPU dense-box band-chunk budget set by the SCF's auto memory
# estimator (scf.loop._auto_cpu_dense_budget). A fallback BELOW the explicit
# GRADWAVE_CPU_DENSE_BUDGET env var: a hand-forced budget still wins, and None
# restores the historical unchunked CPU path.
# [D-003] why env > override > None — docs/design/decision-records.md
_CPU_DENSE_BUDGET_OVERRIDE: float | None = None


def set_cpu_dense_budget_override(budget: float | None) -> None:
    """Set (or clear, with ``None``) the process-local CPU dense-box budget the
    SCF auto estimator computes for a large cell. Consulted by
    ``_cpu_dense_budget_bytes`` only when ``GRADWAVE_CPU_DENSE_BUDGET`` is unset,
    so an explicit env var always wins. The SCF sets this in a try/finally so a
    chain of runs (relax/EOS) never leaks one cell's budget into the next."""
    global _CPU_DENSE_BUDGET_OVERRIDE
    if budget is not None and budget <= 0:
        raise ValueError(f"CPU dense budget override must be > 0, got {budget!r}")
    _CPU_DENSE_BUDGET_OVERRIDE = budget


def _cpu_dense_budget_bytes() -> float | None:
    """CPU dense-box band-chunk budget in bytes, or ``None`` for *no chunking*
    (the historical CPU behaviour — one batched FFT over all bands).

    Resolution order: the explicit ``GRADWAVE_CPU_DENSE_BUDGET`` env var (must be
    > 0) wins; else the process-local override the SCF auto estimator set for a
    large cell (``set_cpu_dense_budget_override``); else ``None``. Chunking the
    H-apply local term is bit-identical (no cross-band reduction); the density
    build's per-chunk partial sum reorders the band sum at the ~1e-16 ulp level.
    [D-003] slab sizing evidence — docs/design/decision-records.md"""
    raw = os.environ.get("GRADWAVE_CPU_DENSE_BUDGET")
    if raw is None:
        return _CPU_DENSE_BUDGET_OVERRIDE
    budget = float(raw)
    if budget <= 0:
        raise ValueError(
            f"GRADWAVE_CPU_DENSE_BUDGET must be > 0 bytes, got {budget!r}")
    return budget


def _dense_band_chunk(n_grid: int, nk: int, device: torch.device, elem_bytes: int) -> int:
    """Bands per chunk so dense-box temporaries stay under the memory budget (the
    apply/density chain holds ~4 such temporaries at once).

    GPU: always chunked to the `_gpu_dense_budget_bytes` budget. CPU: unchunked by
    default (returns a large sentinel → one batched FFT over all bands, historical
    behaviour), UNLESS `GRADWAVE_CPU_DENSE_BUDGET` is set, which turns on the same
    bit-exact chunking to fit a big slab's box on a memory-tight host.

    elem_bytes scales the budget by the coefficient precision: the fp32 draft
    (complex64, 8 B) fits twice as many bands as fp64 (complex128, 16 B),
    giving larger — and thus more efficient — batched FFTs."""
    if device.type == "cuda":
        budget: float | None = _gpu_dense_budget_bytes()
    else:
        budget = _cpu_dense_budget_bytes()
        if budget is None:
            return 1_000_000
    return max(1, int(budget / (elem_bytes * n_grid * max(nk, 1))))

# Small-cell fast path for the local potential term V(r)·ψ(r). On the wavefunction
# G-sphere this term is EXACTLY the convolution out(G_i)=Σ_j V̂(G_i−G_j) c(G_j) =
# M @ c, with M[i,j]=V̂(G_i−G_j) (a Toeplitz/difference matrix; V̂=FFT of v_eff).
# One dense GEMM replaces the scatter → ifftn → ·v_eff → fftn → gather chain.
# Bit-identical to the FFT path (an algebraic identity, not an approximation).
# M is npw²·16 B, so the path is memory-gated (nk·npw²·elem ≤ budget below).
#
# AUTO-GATED: whether it is USED is a one-time timed trial per (device, nk,
# npw_max, shape, dtype) signature, not a size threshold. ``GRADWAVE_TOEPLITZ``
# in {"auto", "on", "off"} (read once at import) forces the path for A/B
# benchmarks; "on" bypasses the trial.
# [D-001] measured verdict gate (14× small-npw win / 0.8× production-ecut loss)
# — docs/design/decision-records.md
_TOEPLITZ_MODE = os.environ.get("GRADWAVE_TOEPLITZ", "auto").strip().lower()
# [D-002] off on consumer GPUs (fp64 FFTs dominate; would regress) — see
# docs/design/decision-records.md; flip to validate on a data-center GPU.
_TOEPLITZ_ON_CUDA = False
# 256 MiB cap on the cached per-k matrix (+~half again for the index table);
# deliberately conservative — raise to extend coverage ([D-001]).
_TOEPLITZ_M_BUDGET_BYTES = 1 << 28
# measured verdicts: (device type, nk, npw_max, shape, dtype) → use Toeplitz?
_TOEP_VERDICT: dict[
    tuple[str, int, int, tuple[int, int, int], torch.dtype], bool
] = {}
# Adopt only when t_toep < margin·t_fft (≥30% faster): headroom covers the
# per-iteration M rebuild amortized over the iteration's applies ([D-001]).
_TOEP_TRIAL_MARGIN = 0.7

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
    # Toeplitz difference-index tables keyed by dense-grid shape, filled
    # lazily by BatchedHamiltonian._toeplitz_idx (geometry-only, shared
    # across the per-iteration Hamiltonian rebuilds of one SCF). Each entry
    # stores (flat_idx it was built from, table); the consumer revalidates
    # before trusting a hit — a foreign table is silently wrong physics.
    # [D-006] cache placement & revalidation — docs/design/decision-records.md
    toep_idx_cache: (
        dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None

    @property
    def nk(self) -> int:
        return int(self.npw.shape[0])

    @property
    def npw_max(self) -> int:
        return int(self.mask.shape[1])

    def reindex(self, idx: torch.Tensor) -> BatchedK:
        """This batch restricted/reordered to k rows ``idx`` (per-k fields only;
        npw_max, the atom index and the D-matrix are k-independent). The
        Toeplitz difference-index cache is dropped, NOT inherited: it belongs
        to the parent's flat_idx and would be wrong physics on the reindexed
        spheres ([D-006], docs/design/decision-records.md)."""
        import dataclasses

        return dataclasses.replace(
            self, npw=self.npw[idx], mask=self.mask[idx],
            flat_idx=self.flat_idx[idx], kpg=self.kpg[idx], t=self.t[idx],
            proj_phase_free=self.proj_phase_free[idx], toep_idx_cache=None)


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
    idx = bk.proj_atom_index
    if positions.requires_grad or torch.is_grad_enabled():
        # Differentiable path (forces rebuild this with grad): the batched gather
        # keeps a single clean autograd node. The SCF itself calls under
        # no_grad, so this branch runs only where the graph is needed.
        return bk.proj_phase_free * phases[:, :, idx].permute(0, 2, 1)
    # Memory-light no-grad path: materialize per k rather than gathering all k
    # at once — byte-identical (per-k independent, no cross-k reduction), one
    # k-slice transient instead of a third full projector-table copy.
    # [D-005] measured 3×0.73 GiB peak at Si-64/30 Ry — docs/design/decision-records.md
    out = torch.empty_like(bk.proj_phase_free)
    for k in range(bk.proj_phase_free.shape[0]):
        out[k] = bk.proj_phase_free[k] * phases[k][:, idx].permute(1, 0)
    return out


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
        # Toeplitz small-cell fast path (see _TOEPLITZ_MODE). Eligible only for
        # the plain box (no USPP dual grid) and when the cached matrix fits the
        # budget; whether it is USED is the per-geometry measured verdict
        # (`_use_toeplitz`). The difference-index table is geometry-only and
        # cached on the BatchedK across Hamiltonian rebuilds; M is built lazily
        # per working precision and reused across every apply of this H.
        self._toep_idx: torch.Tensor | None = None
        self._toep_M_cache: dict[torch.dtype, torch.Tensor] = {}
        on_device = bk.mask.device.type == "cpu" or _TOEPLITZ_ON_CUDA
        self._toep_eligible = False
        # mode captured at construction so a caller forcing "on" around the
        # ctor (e.g. the dfpt_q Sternheimer build) binds THIS H, apply-time
        # global flips notwithstanding
        self._toep_forced = _TOEPLITZ_MODE == "on"
        if _TOEPLITZ_MODE != "off" and on_device and smooth is None:
            nk, npw = bk.mask.shape
            if nk * npw * npw * 16 <= _TOEPLITZ_M_BUDGET_BYTES:
                self._toep_eligible = True
        # cdtype → cast (t, v_eff, p, dij). No resident p.conj() copy: becp_b
        # folds the conjugation into the BLAS matmul.
        # [D-004] why no conjugate table — docs/design/decision-records.md
        self._tab_cache: dict[
            torch.dtype, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        # cdtype → cast (hub_q, hub_q_conj, hub_dij)
        self._hub_cache: dict[torch.dtype, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _tables(
        self, cdtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cast tables to the working precision of the coefficients (cached).

        Feeding complex64 coefficients is not enough on its own: multiplying by
        an fp64 table promotes the result back to complex128. Precomputing
        matching-precision copies keeps the whole apply in fp32 when asked.

        Returns ``(t, v_eff, p, dij)``. There is no resident ``p.conj()`` copy:
        ``becp_b`` folds the conjugation into the BLAS matmul (see its docstring),
        so a materialized conjugate table is pure resident overhead."""
        cached = self._tab_cache.get(cdtype)
        if cached is None:
            from gradwave.dtypes import real_of

            rdtype = real_of(cdtype)
            cached = (
                self.bk.t.to(rdtype),
                self.v_eff_r.to(rdtype),
                self.p.to(cdtype),
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

    def _toeplitz_idx(self) -> torch.Tensor:
        """The (nk, npw, npw) difference-index table, cached on the BatchedK.

        Geometry-only (Miller indices are fixed for the whole SCF), so it is
        built at most once per BatchedK and shared across the per-iteration
        Hamiltonian rebuilds. A cache hit is trusted only if it was built from
        THIS operator's gather_idx (identity fast path, else an O(nk·npw)
        equality check) — a foreign table from a derived BatchedK is silently
        wrong physics. [D-006] history & the Sternheimer-NaN postmortem —
        docs/design/decision-records.md"""
        if self._toep_idx is None:
            if self.bk.toep_idx_cache is None:
                self.bk.toep_idx_cache = {}
            gather = self.gather_idx
            idx: torch.Tensor | None = None
            hit = self.bk.toep_idx_cache.get(self.shape)
            if hit is not None:
                src, cached = hit
                if src is gather or (
                    src.shape == gather.shape and bool(torch.equal(src, gather))
                ):
                    idx = cached
            if idx is None:
                idx = self._build_toeplitz_index()
                self.bk.toep_idx_cache[self.shape] = (gather, idx)
            self._toep_idx = idx
        return self._toep_idx

    def _toeplitz_M(self, cdtype: torch.dtype) -> torch.Tensor:
        """Local-potential matrix M[k,i,j]=V̂(G_i−G_j) at the working precision,
        cached for the H's lifetime (v_eff is fixed). V̂ is the FFT of v_eff on
        the dense box; indexing it by the geometry table gives M with no explicit
        difference loop."""
        M = self._toep_M_cache.get(cdtype)
        if M is None:
            from gradwave.core.fftbox import r_to_g

            vhat = r_to_g(self.v_eff_r.to(cdtype)).reshape(-1)
            M = vhat[self._toeplitz_idx()]
            self._toep_M_cache[cdtype] = M
        return M

    def _local_toep(self, c: torch.Tensor) -> torch.Tensor:
        """Local V·ψ as one dense GEMM per k: out(G_i) = Σ_j M[i,j] c(G_j).
        Mask the input so padded columns (whose M entries are undefined) cannot
        pollute valid rows; the caller masks the output as in the FFT path."""
        M = self._toeplitz_M(c.dtype)
        cm = c * self.bk.mask[:, None, :]
        return torch.einsum("kij,kbj->kbi", M, cm)

    def _local_fft_into(self, c: torch.Tensor, out: torch.Tensor,
                        v_eff: torch.Tensor, count: bool = True) -> None:
        """Local V·ψ via the dense-box FFT pair, added into `out` in place.
        Chunked over bands to bound peak memory on the dense grid."""
        nk, nb, m = c.shape
        chunk = self._band_chunk(nk, c.device, c.element_size())
        for lo in range(0, nb, chunk):
            hi = min(lo + chunk, nb)
            cc = c[:, lo:hi]
            nbc = hi - lo
            box = self._get_box(nk, nbc, cc.dtype, cc.device)
            idx = self.idx_scatter[:, None, :].expand(nk, nbc, m)
            box.scatter_(2, idx, cc)
            if count:
                opcount.bump("fft")
            psi = torch.fft.ifftn(box[..., : self.n].reshape(nk, nbc, *self.shape),
                                  dim=(-3, -2, -1))
            # fftn(ifftn(·)) is norm-neutral: the 1/N and ×N of the fftbox
            # conventions cancel, so no scaling factors here
            if count:
                opcount.bump("fft")
            vg = torch.fft.fftn(psi * v_eff, dim=(-3, -2, -1)).reshape(nk, nbc, self.n)
            gath = self.gather_idx[:, None, :].expand(nk, nbc, m)
            out[:, lo:hi] += vg.gather(2, gath)

    def _use_toeplitz(self, c: torch.Tensor, v_eff: torch.Tensor) -> bool:
        """The measured per-geometry gate for the Toeplitz local apply ([D-001],
        docs/design/decision-records.md).

        "on" forces it (within eligibility); "auto" runs a one-time trial per
        (device, nk, npw_max, shape, dtype) signature: both local-term paths
        are timed on the real block and Toeplitz is adopted only if
        t_toep < ``_TOEP_TRIAL_MARGIN`` · t_fft. On a losing verdict the
        trial's M and index table are freed immediately."""
        if not self._toep_eligible:
            return False
        if self._toep_forced or _TOEPLITZ_MODE == "on":
            return True
        nk, nb, m = c.shape
        key = (c.device.type, nk, m, self.shape, c.dtype)
        verdict = _TOEP_VERDICT.get(key)
        if verdict is None:

            def timed(f: Callable[[], object]) -> float:
                f()  # warm (JIT allocations, M/idx builds land here)
                t0 = time.perf_counter()
                f()
                return time.perf_counter() - t0

            scratch = torch.zeros_like(c)
            t_fft = timed(lambda: self._local_fft_into(c, scratch, v_eff, count=False))
            t_toep = timed(lambda: self._local_toep(c))
            verdict = t_toep < _TOEP_TRIAL_MARGIN * t_fft
            _TOEP_VERDICT[key] = verdict
            if not verdict:  # free the trial's M (+ shared idx) right away
                self._toep_M_cache.clear()
                self._toep_idx = None
                if self.bk.toep_idx_cache is not None:
                    self.bk.toep_idx_cache.pop(self.shape, None)
        return verdict

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
        """`_dense_band_chunk` on this operator's dense grid."""
        return _dense_band_chunk(self.n, nk, device, elem_bytes)

    def apply(self, c: torch.Tensor) -> torch.Tensor:
        """(nk, nb, npw_max) → H c, mask preserved. Chunked over bands to
        bound peak memory on the dense grid (math identical)."""
        bk = self.bk
        nk, nb, m = c.shape
        if _HAPPLY_TALLY["on"]:
            _HAPPLY_TALLY["count"] += nk * nb
        opcount.bump("hpsi", nk * nb)
        t_r, v_eff, p, dij = self._tables(c.dtype)
        out = t_r[:, None, :] * c

        if self._use_toeplitz(c, v_eff):
            # Small-cell Toeplitz path (measured verdict, see _use_toeplitz):
            # local V·ψ as one dense GEMM per k.
            out = out + self._local_toep(c)
        else:
            self._local_fft_into(c, out, v_eff)

        if p.shape[1]:
            b = becp_b(p, c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b, dij, p)
        if self.hub_q is not None and self.hub_dij is not None:
            hq, hq_conj, hd = self._hub_tables(c.dtype)
            bh = torch.einsum("kpg,kbg->kbp", hq_conj, c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", bh, hd, hq)
        return out * bk.mask[:, None, :]

    def apply_cols(self, c: torch.Tensor, kcol: torch.Tensor) -> torch.Tensor:
        """H applied to a compacted COLUMN batch: ``c`` (ncol, npw_max) with a
        per-column k index ``kcol`` (ncol,) into this operator's BatchedK.

        The active-set compaction path of ``postscf._response.cg_sternheimer``:
        once most (k, band) Sternheimer columns have converged, the survivors
        are gathered into one flat batch and applied here, so the per-iteration
        cost tracks the ACTIVE column count instead of the full (nk, nb) block.
        Math-identical to ``apply`` on the corresponding rows (same tables,
        same FFT local term — the Toeplitz path is not used here: gathering
        M[kcol] per apply would cost ncol·npw² memory traffic every iteration,
        which defeats the compaction).
        """
        if _HAPPLY_TALLY["on"]:
            _HAPPLY_TALLY["count"] += int(c.shape[0])
        opcount.bump("hpsi", int(c.shape[0]))
        t_r, v_eff, p, dij = self._tables(c.dtype)
        out = t_r[kcol] * c
        # local term: dense-box FFT pair with per-column scatter/gather
        ncol, m = c.shape
        chunk = self._band_chunk(1, c.device, c.element_size())
        for lo in range(0, ncol, chunk):
            hi = min(lo + chunk, ncol)
            cc = c[lo:hi]
            box = torch.zeros(hi - lo, self.n + 1, dtype=c.dtype, device=c.device)
            box.scatter_(1, self.idx_scatter[kcol[lo:hi]], cc)
            opcount.bump("fft")
            psi = torch.fft.ifftn(
                box[:, : self.n].reshape(hi - lo, *self.shape), dim=(-3, -2, -1))
            opcount.bump("fft")
            vg = torch.fft.fftn(psi * v_eff, dim=(-3, -2, -1)).reshape(hi - lo, self.n)
            out[lo:hi] += vg.gather(1, self.gather_idx[kcol[lo:hi]])
        if p.shape[1]:
            ps = p[kcol]  # (ncol, nproj, npw_max)
            # ⟨β|ψ⟩ per column with the conj folded into the matmul (see becp_b);
            # c is (ncol, npw_max) so unsqueeze to a length-1 band batch.
            b = torch.matmul(
                c.unsqueeze(1), ps.conj().transpose(-2, -1)
            ).squeeze(1)  # (ncol, nproj)
            out = out + torch.einsum("cp,pq,cqg->cg", b, dij, ps)
        if self.hub_q is not None and self.hub_dij is not None:
            hq, hq_conj, hd = self._hub_tables(c.dtype)
            bh = torch.einsum("cpg,cg->cp", hq_conj[kcol], c)
            out = out + torch.einsum("cp,pq,cqg->cg", bh, hd, hq[kcol])
        return out * self.bk.mask[kcol]

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
    chunk = _dense_band_chunk(n, nk, coeffs.device, coeffs.element_size())
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
    """⟨p|ψ⟩ overlaps (nk, nb, nproj) = Σ_g conj(p) c.

    Computed as a batched matmul against a conjugate-transpose VIEW of ``p``:
    ``matmul(c, p.conj().mT)`` — BLAS folds the conjugation (cgemm ConjTrans),
    so no conjugate table is ever materialized. ``p_conj`` is retained for
    call-site compatibility and IGNORED.
    [D-004] why the einsum + cached-conj form was replaced —
    docs/design/decision-records.md"""
    return torch.matmul(c, p.conj().transpose(-2, -1))
