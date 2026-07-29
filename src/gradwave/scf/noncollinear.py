"""Non-collinear spinor SCF (no spin-orbit yet).

Spinors ride the existing k-batched machinery as DOUBLED plane-wave vectors
c (nk, nb, 2·npw_max): [.., :npw] = up component, [.., npw:] = down. The
batched Davidson operates on ℂ^{2npw} unchanged (kinetic/mask/preconditioner
tensors are simply concatenated); only the Hamiltonian apply knows about
spin, mixing the components in real space through

    V̂(r) = [v_H + v_loc + v_xc]·𝟙 + B⃗_xc(r)·σ⃗

with (v_xc, B⃗_xc) from ONE autograd call on the locally-collinear XC.
The nonlocal (scalar-relativistic) projectors act on each component
independently; SOC will add the 2×2 j-resolved structure here.

Density matrix by Pauli decomposition: ρ = Σf(|ψ↑|²+|ψ↓|²),
m_z = Σf(|ψ↑|²−|ψ↓|²), m_x = 2Σf Re(ψ↑*ψ↓), m_y = 2Σf Im(ψ↑*ψ↓).
Mixing runs on the 4-vector (ρ, m⃗) with Kerker on the ρ block ONLY
(the collinear lesson: Kerker's G=0 zero must never pin magnetization).

Each spinor band holds ONE electron (Fermi degeneracy g = 1). Build the
System with time_reversal=False: TR flips m⃗, so the plain TR-reduced mesh
is only valid for collinear-limit checks. For real k-savings pass
setup_system(..., use_symmetry=True, magmoms=...): k then folds into the
MAGNETIC IBZ of the Shubnikov group (anti-unitary g·T ops act as −W⁻ᵀ) and
(ρ, m⃗) are re-symmetrized over the full magnetic group each iteration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from gradwave.core.batch import BatchedK, becp_b, projectors_b
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.hubbard import HubbardManifold
from gradwave.core.metagga import spinor_metagga_tau_operator, spinor_tau_matrix_b
from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.core.xc.noncollinear import (
    NoncollinearXC,
    local_frame_tau,
    tau_operator_fields,
    vtau_up_dn,
    vxc_and_bxc,
)
from gradwave.dtypes import CDTYPE, CDTYPE_LOW, RDTYPE, RDTYPE_LOW
from gradwave.grids import FFTGrid
from gradwave.scf.common import (
    MP_CROSSOVER,
    adaptive_diago_tol,
    convergence_gate,
    record_iteration,
    symmetrize_rho,
)
from gradwave.scf.guess import sad_density
from gradwave.scf.loop import System, _stack_dij
from gradwave.scf.mixing import PulayMixer
from gradwave.scf.moment_penalty import field_coeff
from gradwave.scf.spinor_common import (
    apply_local_spinor,
    pack_grid_channels,
    pauli_density_accumulate,
    spinor_band_chunk,
    spinor_kinetic_energy,
    spinor_potential_blocks,
    spinor_pw_seed,
    spinor_scalar_nonlocal_energy,
    unpack_grid_channels,
)
from gradwave.solvers.davidson import davidson_batched

logger = logging.getLogger(__name__)


class SpinorHamiltonian:
    """H apply on doubled vectors (nk, nb, 2·npw_max).

    Nonlocal: scalar-relativistic pseudos act per spin component with p;
    fully-relativistic pseudos use spinor projectors q on the DOUBLED axis
    (j-resolved SOC — see core/spinor_proj.py)."""

    def __init__(
        self, bk: BatchedK, shape: tuple[int, int, int], v_r: torch.Tensor,
        b_vec_r: torch.Tensor, p: torch.Tensor, q: torch.Tensor | None = None,
        dij_so: torch.Tensor | None = None,
        metagga_op: Callable[[torch.Tensor], torch.Tensor] | None = None,
        hub_q: torch.Tensor | None = None, hub_dij: torch.Tensor | None = None,
    ) -> None:
        self.bk = bk
        self.shape = shape
        self.p = p  # (nk, nproj, npw_max) scalar projectors
        self.q = q  # (nk, nproj_so, 2·npw_max) spinor projectors (FR)
        self.dij_so = dij_so
        # meta-GGA: a callable c → V_τ c (the 2×2 generalized-KS τ operator,
        # core.metagga.spinor_metagga_tau_operator with the current v_τ fields),
        # or None for LDA/GGA. Hermitian, so it adds straight into H·c.
        self.metagga_op = metagga_op
        # DFT+U: atomic-orbital projectors (spin-independent — the same
        # projector acts on both spinor components) + the 2×2 spin-block
        # D-matrix (core.hubbard.hubbard_dmatrix_noncollinear), shape
        # (2, nproj_U, 2, nproj_U). Orthogonal to the SOC/scalar-relativistic
        # KB nonlocal term above — added unconditionally when present, so the
        # spin-orbit (is_fr) path gets +U through the same apply.
        self.hub_q = hub_q  # (nk, nproj_U, npw_max)
        self.hub_dij = hub_dij  # (2, nproj_U, 2, nproj_U)
        self.m = bk.npw_max
        # Precompute the 2×2 potential blocks once (fixed per H): v_uu/v_dd
        # are ⟨↑|V̂|↑⟩/⟨↓|V̂|↓⟩ (real), v_ud is ⟨↑|V̂|↓⟩ (complex); nonmagnetic
        # runs (B⃗ ≡ 0) take a fast path that skips the spin-flip term.
        self.b_zero, self._v_uu, self._v_dd, self._v_ud = \
            spinor_potential_blocks(v_r, b_vec_r)
        self._cache: dict = {}
        self._hub_cache: dict = {}  # cdtype → cast (hub_q, hub_q_conj, hub_dij)

    def _tables(self, cdtype: torch.dtype) -> dict[str, torch.Tensor | None]:
        """Working-precision copies of the fixed tensors (cached per dtype)."""
        cached = self._cache.get(cdtype)
        if cached is None:
            from gradwave.dtypes import real_of

            rdtype = real_of(cdtype)
            p = self.p.to(cdtype)
            q = None if self.q is None else self.q.to(cdtype)
            cached = {
                "t": self.bk.t.to(rdtype),
                "v_uu": self._v_uu.to(rdtype),
                "v_dd": self._v_dd.to(rdtype),
                "v_ud": self._v_ud.to(cdtype),
                "p": p,
                # conjugates cached too: they are constant per H but consumed
                # in every band chunk of every Davidson round
                "p_conj": p.conj().resolve_conj(),
                "q": q,
                "q_conj": None if q is None else q.conj().resolve_conj(),
                "dij_so": None if self.dij_so is None else self.dij_so.to(cdtype),
                "dij": self.bk.dij_full.to(cdtype),
            }
            self._cache[cdtype] = cached
        return cached

    def _hub_tables(self, cdtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._hub_cache.get(cdtype)
        if cached is None:
            hq = self.hub_q.to(cdtype)
            cached = (hq, hq.conj().resolve_conj(), self.hub_dij.to(cdtype))
            self._hub_cache[cdtype] = cached
        return cached

    def _band_chunk(self, nk: int, device: torch.device, elem_bytes: int = 16) -> int:
        """Bands per chunk bounding the dense-grid temporaries — the shared
        spinor heuristic (scf/spinor_common.py)."""
        return spinor_band_chunk(self.shape, nk, device, elem_bytes)

    def apply(self, c: torch.Tensor) -> torch.Tensor:
        bk, m = self.bk, self.m
        tab = self._tables(c.dtype)
        t_r = tab["t"]
        cu, cd = c[..., :m], c[..., m:]
        out_u = t_r[:, None, :] * cu
        out_d = t_r[:, None, :] * cd

        nk, nb = c.shape[0], c.shape[1]
        chunk = self._band_chunk(nk, c.device, c.element_size())
        apply_local_spinor(out_u, out_d, cu, cd, bk, self.shape, chunk,
                           tab["v_uu"], tab["v_dd"], tab["v_ud"], self.b_zero)

        mask = bk.mask[:, None, :]
        out = torch.cat([out_u * mask, out_d * mask], dim=-1)

        # nonlocal, band-chunked like the FFT mix above: the unchunked einsums
        # materialize (nk, nb, 2·npw) temporaries — at 384 k and a 240-vector
        # Davidson block that is a >5 GB spike per temporary, which OOM-killed
        # the A100 FePt run through allocator fragmentation. In-place adds on
        # band slices bound the spike at the chunk size.
        if self.q is not None:  # spin-orbit (j-resolved) nonlocal
            q, qc, dso = tab["q"], tab["q_conj"], tab["dij_so"]
            mask2 = torch.cat([mask, mask], dim=-1)
            for lo in range(0, nb, chunk):
                hi = min(lo + chunk, nb)
                b = torch.einsum("kpg,kbg->kbp", qc, c[:, lo:hi])
                out[:, lo:hi] += torch.einsum("kbp,pq,kqg->kbg", b, dso, q) * mask2
        elif self.p.shape[1]:
            dij, p, pc = tab["dij"], tab["p"], tab["p_conj"]
            for lo in range(0, nb, chunk):
                hi = min(lo + chunk, nb)
                bu = torch.einsum("kpg,kbg->kbp", pc, cu[:, lo:hi])
                bd = torch.einsum("kpg,kbg->kbp", pc, cd[:, lo:hi])
                out[:, lo:hi, :m] += torch.einsum("kbp,pq,kqg->kbg", bu, dij, p) * mask
                out[:, lo:hi, m:] += torch.einsum("kbp,pq,kqg->kbg", bd, dij, p) * mask
        if self.hub_q is not None and self.hub_dij is not None:  # DFT+U (Dudarev)
            hq, hqc, hd = self._hub_tables(c.dtype)
            for lo in range(0, nb, chunk):
                hi = min(lo + chunk, nb)
                bu = torch.einsum("kpg,kbg->kbp", hqc, cu[:, lo:hi])
                bd = torch.einsum("kpg,kbg->kbp", hqc, cd[:, lo:hi])
                b = torch.stack([bu, bd], dim=2)  # (nk, nbc, 2, nproj_U)
                # proj[k,b,σ,m] = Σ_{σ'm'} D_{(σm),(σ'm')} b[k,b,σ',m'] — direct
                # matrix-vector contraction (no transpose; see hubbard_dmatrix_noncollinear)
                proj = torch.einsum("sptq,kbtq->kbsp", hd, b)
                out[:, lo:hi, :m] += torch.einsum(
                    "kbp,kpg->kbg", proj[:, :, 0], hq) * mask
                out[:, lo:hi, m:] += torch.einsum(
                    "kbp,kpg->kbg", proj[:, :, 1], hq) * mask
        if self.metagga_op is not None:  # meta-GGA −½∇·(M∇) generalized-KS term
            out = out + self.metagga_op(c)
        return out


@dataclass
class NCResult:
    converged: bool
    n_iter: int
    energies: EnergyBreakdown
    fermi: float
    mag_vec: tuple  # ∫ m⃗ dr [μB]
    mag_abs: float  # ∫ |m⃗| dr [μB]
    rho: torch.Tensor
    m: torch.Tensor  # (3, grid)
    eigenvalues: torch.Tensor  # (nk, nb)
    system: System
    history: list = field(default_factory=list)
    coeffs: torch.Tensor | None = None  # (nk, nb, 2·npw_max) spinor coefficients
    occupations: torch.Tensor | None = None  # (nk, nb) spinor occupations (g=1)
    formalism: str = "noncollinear"  # result-type tag shared by all four SCF drivers
    hub_occ: list | None = None  # DFT+U per-site 2×2 spin-block occupation matrices N^I


_MAG_MIXERS = {"pulay": "PulayMixer", "johnson": "JohnsonMixer",
               "broyden": "BroydenMixer"}


def _build_nc_mixer(
    g2_vec: torch.Tensor, ng: int, nonmagnetic: bool, mixing_alpha: float,
    mag_mixing_alpha: float, mixing_history: int,
    precond_op: Callable[[torch.Tensor], torch.Tensor] | None,
    m: torch.Tensor, device: torch.device, mag_mixer: str = "pulay",
) -> tuple[PulayMixer, torch.Tensor | None, torch.Tensor]:
    """Build the (ρ, m⃗) mixer. Nonmagnetic: single ρ block with Kerker and
    check_g0, and m is zeroed. Magnetic: 4 blocks with Kerker on the ρ block only
    (kerker_mask, check_g0=False) and a decoupled m⃗ step (base_step_scale) to hold
    the magnetic branch against moment collapse. ``mag_mixer`` selects the
    magnetic-path mixer class (pulay/johnson/broyden). Returns
    (mixer, base_step_scale, m)."""
    base_step_scale = None
    if nonmagnetic:
        m = torch.zeros_like(m)
        mixer = PulayMixer(g2_vec, alpha=mixing_alpha, history=mixing_history,
                           kerker=True, check_g0=True)
    else:
        import gradwave.scf.mixing as _mixmod
        kerker_mask = torch.cat([torch.ones(ng, dtype=torch.bool, device=device),
                                 torch.zeros(3 * ng, dtype=torch.bool, device=device)])
        ratio = mag_mixing_alpha / mixing_alpha if mixing_alpha > 0 else 1.0
        base_step_scale = torch.cat([
            torch.ones(ng, dtype=RDTYPE, device=device),
            torch.full((3 * ng,), float(ratio), dtype=RDTYPE, device=device)])
        cls = getattr(_mixmod, _MAG_MIXERS[mag_mixer])
        kw = dict(alpha=mixing_alpha, history=mixing_history, kerker=True,
                  check_g0=False, kerker_mask=kerker_mask,
                  step_scale=base_step_scale)
        mixer = cls(torch.cat([g2_vec] * 4), **kw)
    if precond_op is not None:
        # override constant Kerker on the density-total (charge) block; m⃗ blocks
        # keep their own step (base_step_scale) and are untouched by this operator.
        mixer.precond_op = precond_op
        mixer.precond_slice = slice(0, ng)
    return mixer, base_step_scale, m


def _solve_spinor_bands(
    bk: BatchedK, grid: FFTGrid, v_r: torch.Tensor, b_xc: torch.Tensor,
    projs_b: torch.Tensor, q_so: torch.Tensor | None, dij_so: torch.Tensor | None,
    coeffs: torch.Tensor, t2: torch.Tensor, mask2: torch.Tensor, tol_eff: float,
    mixed_precision: bool, mp_crossover: float,
    metagga_op: Callable[[torch.Tensor], torch.Tensor] | None = None,
    hub_q: torch.Tensor | None = None, hub_dij: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SpinorHamiltonian]:
    """Diagonalize the spinor Hamiltonian for one iteration at diagonalization
    tolerance tol_eff (optional fp32 draft with an fp64 spinor renorm over the
    doubled 2·npw axis so the electron count stays conserved through mixing).
    Returns (eigs, coeffs, h); h is reused for the band-chunk size in the Pauli
    density accumulation. metagga_op (meta-GGA) is the current v_τ operator.
    hub_q/hub_dij (DFT+U) are the atomic-orbital projectors and the current
    (lagged one iteration, like V_eff) 2×2 spin-block D-matrix, or None."""
    use_low = mixed_precision and tol_eff > mp_crossover
    cdtype = CDTYPE_LOW if use_low else CDTYPE
    t2_solve = t2.to(RDTYPE_LOW) if use_low else t2
    h = SpinorHamiltonian(bk, grid.shape, v_r, b_xc, projs_b, q=q_so, dij_so=dij_so,
                          metagga_op=metagga_op, hub_q=hub_q, hub_dij=hub_dij)
    dav = davidson_batched(h.apply, coeffs.to(cdtype), t2_solve, mask2, tol=tol_eff)
    eigs = dav.eigenvalues.to(RDTYPE)
    coeffs = dav.eigenvectors.to(CDTYPE)
    if use_low:
        # fp32 draft: renormalize spinors in fp64 so the electron count
        # (ρ at G=0) stays conserved through mixing (see collinear scf)
        coeffs = coeffs / torch.linalg.norm(
            coeffs, dim=-1, keepdim=True).clamp_min(1e-30)
    return eigs, coeffs, h


def _nc_adaptive_backoff(
    adaptive: bool, it: int, last_backoff: int, stall_window: int, adapt_mult: float,
    history: list[dict[str, int | float]], mixer: PulayMixer,
    base_step_scale: torch.Tensor | None, verbose: bool,
) -> tuple[float, int]:
    """When the residual stalls or bounces over a window (a limit cycle at a
    frustrated moment / SOC), halve the global mixing step multiplier and drop the
    DIIS history — MUTATES mixer (step_scale, reset) — so the pre-stall vectors
    stop fighting the recovery. Returns (adapt_mult, last_backoff)."""
    if not (adaptive and it - last_backoff >= stall_window
            and it > 2 * stall_window and adapt_mult > 0.1):
        return adapt_mult, last_backoff
    recent = min(h["res"] for h in history[-stall_window:])
    before = min(h["res"] for h in history[-2 * stall_window:-stall_window])
    if recent > 0.9 * before:
        adapt_mult = max(0.5 * adapt_mult, 0.1)
        mixer.step_scale = (adapt_mult if base_step_scale is None
                            else base_step_scale * adapt_mult)
        mixer.reset()
        last_backoff = it
        if verbose:
            print(f"  NC-SCF: residual stalled — mixing step x{adapt_mult:.2f}",
                  flush=True)
    return adapt_mult, last_backoff


def _nc_energy_breakdown(
    coeffs: torch.Tensor, occ: torch.Tensor, t2: torch.Tensor, entropy_term: torch.Tensor,
    rho_out: torch.Tensor, m_out: torch.Tensor, q_so: torch.Tensor | None,
    dij_so: torch.Tensor | None, projs_b: torch.Tensor, m_pw: int, vloc_g: torch.Tensor,
    e_ew: torch.Tensor, system: System, grid: FFTGrid, xc: NoncollinearXC, vol: float, nk: int,
    tau_up: torch.Tensor | None = None, tau_dn: torch.Tensor | None = None,
) -> EnergyBreakdown:
    """Per-iteration energy breakdown. The nonlocal term takes the SOC path
    (q_so becp) or the scalar-relativistic per-spin (up/down) path. tau_up/tau_dn
    (meta-GGA) are the local-frame per-spin τ from the current orbitals. Returns
    EnergyBreakdown."""
    rho_g_out = r_to_g(rho_out.to(CDTYPE))
    t_occ = (system.kweights[:, None] * occ).to(coeffs.real.dtype)
    e_kin = spinor_kinetic_energy(t_occ, coeffs, t2)
    e_h = hartree_energy(rho_g_out, grid.g2, vol)
    from gradwave.core.xc.noncollinear import energy_with_grid

    e_xc = energy_with_grid(xc, rho_out, m_out, grid, rho_core=system.rho_core,
                            tau_up=tau_up, tau_dn=tau_dn)
    e_loc = local_energy(rho_g_out, vloc_g, vol)
    if q_so is not None:
        b_so = torch.einsum("kpg,kbg->kbp", q_so.conj(), coeffs)
        e_nl = nonlocal_energy([b_so[ik] for ik in range(nk)], dij_so, occ,
                               system.kweights)
    else:
        bu = becp_b(projs_b, coeffs[..., :m_pw])
        bd = becp_b(projs_b, coeffs[..., m_pw:])
        dij = _stack_dij(system)
        e_nl = spinor_scalar_nonlocal_energy(bu, bd, dij, occ,
                                             system.kweights, nk)
    return EnergyBreakdown(kinetic=e_kin, hartree=e_h, xc=e_xc, local=e_loc,
                           nonlocal_=e_nl, ewald=e_ew, smearing=entropy_term)


def _nc_effective_potential(
    xc: NoncollinearXC, rho: torch.Tensor, m: torch.Tensor, grid: FFTGrid, system: System,
    vloc_r: torch.Tensor, nonmagnetic: bool, constrain_dirs: torch.Tensor | None,
    constrain_lambda: float, constrain_mode: str, constrain_target_mag: torch.Tensor | None,
    atom_weights: torch.Tensor | None, vol: float,
    tau_up: torch.Tensor | None = None, tau_dn: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-iteration effective potential v_r and exchange field b_xc. Nonmagnetic
    zeros b_xc; otherwise an optional Ma-Dudarev constraining field is ADDED to
    b_xc after (never before) the nonmagnetic zeroing. One vxc_and_bxc autograd
    call, unchanged. tau_up/tau_dn (meta-GGA) are held fixed in that call — the
    τ response is the separate 2×2 operator. Returns (v_r, b_xc)."""
    rho_g_box = r_to_g(rho.to(CDTYPE))
    v_h = g_to_r_box(hartree_potential_g(rho_g_box, grid.g2), real=True)
    v_xc, b_xc, _ = vxc_and_bxc(xc, rho, m, grid, rho_core=system.rho_core,
                                tau_up=tau_up, tau_dn=tau_dn)
    if nonmagnetic:
        b_xc = torch.zeros_like(b_xc)
    elif constrain_dirs is not None:
        # Constraining field B_c(r) = Σ_I (∂E_p/∂M_I) w_I(r) pins each atomic
        # moment M_I = ∫ w_I m⃗ dr toward its target ê_I. ∂E_p/∂M_I comes from
        # autograd on penalty_energy (gradwave.scf.moment_penalty), so the
        # direction-only "perp" and magnitude-robust "vector" penalties share
        # one definition. It adds to the exchange field b_xc (= δE/δm⃗).
        cf = vol / grid.n_points
        m_at = torch.einsum("axyz,ixyz->ai", atom_weights, m) * cf   # M_I (na,3)
        g = field_coeff(m_at, constrain_dirs, constrain_lambda,
                        constrain_mode, constrain_target_mag)
        b_xc = b_xc + torch.einsum("ai,axyz->ixyz", g, atom_weights)
    v_r = v_h + v_xc + vloc_r
    return v_r, b_xc


def _bootstrap_spinor_tau(
    xc: NoncollinearXC, coeffs: torch.Tensor, system: System, nbands: int, nk: int,
    bk: BatchedK, grid: FFTGrid, vol: float, m_pw: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Seed the KE-density-matrix (τ_0, τ⃗) from the initial spinors so iteration
    1 has a valid τ for the τ-dependent v_xc and operator (refined immediately
    from the diagonalized orbitals). (None, None) for non-meta-GGA — mirrors
    scf.loop._bootstrap_tau on the doubled spinor axis."""
    if not xc.needs_tau:
        return None, None
    nocc = max(int(round(system.n_electrons)), 1)  # spinor bands hold 1 e⁻ each
    occ0 = torch.zeros(nk, nbands, dtype=RDTYPE, device=device)
    occ0[:, :nocc] = 1.0
    return spinor_tau_matrix_b(coeffs, occ0, system.kweights, bk, grid.shape,
                               vol, m_pw)


def _nc_metagga_step(
    xc: NoncollinearXC, rho: torch.Tensor, m: torch.Tensor, grid: FFTGrid, system: System,
    tau_scalar: torch.Tensor | None, tau_vec: torch.Tensor | None, nonmagnetic: bool,
    bk: BatchedK, m_pw: int,
) -> (
    tuple[tuple[torch.Tensor, torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]
    | tuple[tuple[None, None], None]
):
    """Meta-GGA per-iteration τ machinery for the spinor loop, or (None,None),None
    for LDA/GGA. From the stored KE-density-matrix (τ_0, τ⃗) — built from the
    previous iteration's orbitals — and the current (ρ, m⃗):

      * project τ into the local frame → (τ_up, τ_dn), held fixed in the
        energy and v_xc/B⃗_xc autograd;
      * form v_τ↑, v_τ↓ = ∂E_xc/∂τ_± and rotate them back to the 2×2 operator
        fields (v_τ0, v_τ⃗); a nonmagnetic (m⃗ ≡ 0) run keeps only the scalar
        block v_τ0 (mirroring the b_xc zeroing).

    Returns ((τ_up, τ_dn), metagga_op)."""
    if not xc.needs_tau:
        return (None, None), None
    tau_up, tau_dn = local_frame_tau(m, tau_scalar, tau_vec, xc.m_eps)
    vtu, vtd = vtau_up_dn(xc, rho, m, grid, tau_up, tau_dn,
                          rho_core=system.rho_core)
    if nonmagnetic:
        v0 = 0.5 * (vtu + vtd)
        vvec = torch.zeros(3, *rho.shape, dtype=RDTYPE, device=rho.device)
    else:
        v0, vvec = tau_operator_fields(vtu, vtd, m, xc.m_eps)

    def op(c, _v0=v0, _vv=vvec):
        return spinor_metagga_tau_operator(c, _v0, _vv, bk, grid.shape, m_pw)

    return (tau_up, tau_dn), op


def _seed_nc_density(
    grid: FFTGrid, system: System, mag_vec_init: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial (ρ, m⃗): SAD total density plus atom-directed magnetization
    channels seeded from mag_vec_init (per-atom moment fraction·direction)."""
    rho = sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                      system.n_electrons)
    m_chan = [
        sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                    None, atom_scale=[float(mag_vec_init[a, i])
                                      for a in range(len(system.species_of_atom))])
        for i in range(3)
    ]
    return rho.to(device), torch.stack(m_chan).to(device)


def _unpack_mixed_fields(
    mixed: torch.Tensor, n_chan: int, ng: int, mask_flat: torch.Tensor, grid: FFTGrid,
    device: torch.device,
) -> list[torch.Tensor]:
    """Inverse-FFT the mixed (ρ, m⃗) vector back to per-channel real-space fields:
    [rho] when nonmagnetic (n_chan=1), else [rho, m_x, m_y, m_z]."""
    return unpack_grid_channels(mixed, n_chan, ng, mask_flat, grid.shape,
                                grid.n_points, device)


@torch.no_grad()
def scf_noncollinear(
    system: System,
    xc: NoncollinearXC,
    mag_vec_init: list[list[float]] | torch.Tensor,  # (na, 3) initial moment fraction·direction
    smearing: str = "gaussian",
    width: float = 0.1,
    max_iter: int = 120,
    etol: float = 1e-8,
    rhotol: float = 1e-7,
    mixing_alpha: float = 0.5,
    mixing_history: int = 8,
    mag_mixing_alpha: float | None = None,  # separate step for m⃗ (None → max(mixing_alpha,0.6))
    spin_precond: bool = False,  # opt-in Stoner precond on the (longitudinal) m⃗ channel
    mag_mixer: str = "pulay",  # magnetic-path mixer: pulay/johnson/broyden
    mag_diago_schedule: str = "linear",  # adaptive diago tol schedule for magnetic runs
    adaptive: bool = True,  # back off mixing on a stalled/oscillating residual
    diago_tol: float = 1e-9,
    verbose: bool = True,
    nonmagnetic: bool = False,  # pin m⃗ ≡ 0 (QE's domag=false): nonmagnetic + SOC
    mixed_precision: bool = False,  # opt-in fp32 draft (situational — see scf())
    constrain_dirs: torch.Tensor | None = None,  # (na,3) unit target directions ê_I
    constrain_lambda: float = 0.0,  # penalty strength λ [eV/μB²] (Ma-Dudarev)
    atom_weights: torch.Tensor | None = None,  # (na,*grid) Hirshfeld weights; needed to constrain
    constrain_mode: str = "perp",  # "perp" (direction only) or "vector" (+magnitude)
    constrain_target_mag: torch.Tensor | None = None,  # per-atom |M| target [μB], mode="vector"
    precond_op: Callable[[torch.Tensor], torch.Tensor] | None = None,  # r -> P.r on density block
    # (charge channel), overriding constant Kerker there — e.g. a fitted learned_precond filter
    mixer_hook: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,  # (it, vin, vout)
    hubbard: list[HubbardManifold] | None = None,  # noncollinear DFT+U
    # (Dudarev); the 2×2 spin-block generalization of the collinear occupation
    # matrix (core.hubbard.occupation_matrices_noncollinear). Shared with SOC
    # (is_fr): the +U term is orthogonal to the SOC nonlocal term, so a
    # fully-relativistic pseudo gets +U through the same SpinorHamiltonian apply.
) -> NCResult:
    # A plain RhoSymmetrizer (paramagnetic group) is only valid with m⃗ ≡ 0.
    # A MagneticSymmetrizer (setup_system(..., magmoms=...)) carries the
    # Shubnikov group of the moment configuration and symmetrizes m⃗ too, so
    # magnetic runs on the magnetic IBZ are allowed. The seeded mag_vec_init
    # must match the magmoms the group was built from — symmetrization
    # projects every iteration onto that magnetic symmetry.
    mag_sym_active = hasattr(system.rho_symmetrizer, "apply_m")
    if system.rho_symmetrizer is not None and not (nonmagnetic or mag_sym_active):
        raise ValueError(
            "noncollinear SCF with a nonzero m⃗ requires use_symmetry=False "
            "(time reversal and the space group act on m⃗) or a MAGNETIC "
            "symmetry system (setup_system(..., magmoms=...)); the nonmagnetic "
            "(m⃗ ≡ 0) case keeps the full crystal symmetry — pass nonmagnetic=True"
        )
    grid, bk = system.grid, system.batch
    vol, nk = grid.volume, len(system.spheres)
    device = system.positions.device
    mp_crossover = MP_CROSSOVER
    nbands = 2 * system.nbands  # spinor bands hold one electron each
    m_pw = bk.npw_max
    mag_vec_init = torch.as_tensor(mag_vec_init, dtype=RDTYPE)

    # initial (ρ, m⃗): SAD total + atom-directed magnetization channels
    rho, m = _seed_nc_density(grid, system, mag_vec_init, device)

    mask_flat = grid.dens_mask.reshape(-1)
    g2_vec = grid.g2.reshape(-1)[mask_flat]
    ng = int(mask_flat.sum())
    n_chan = 1 if nonmagnetic else 4
    # magnetization-aware mixing: ρ and m⃗ ride one packed vector but must NOT
    # share a single step. QE/VASP mix the spin channel separately from charge;
    # here m⃗ gets its own step through the mixer's per-component step_scale.
    # The failure mode is moment COLLAPSE: at a small mixing_alpha the charge is
    # under-relaxed and the magnetization, dragged toward the transient small
    # m_out before the exchange field self-consistifies, decays into the wrong
    # nonmagnetic basin (bcc O2: |M| → 0 at alpha=0.4 while alpha=0.7 keeps the
    # triplet). Decoupling the m⃗ step with a floor keeps the magnetization mixed
    # vigorously enough to hold the magnetic branch regardless of the charge
    # step; the adaptive backoff below is the counterweight against overshoot.
    if mag_mixing_alpha is None:
        mag_mixing_alpha = max(mixing_alpha, 0.6)
    mixer, base_step_scale, m = _build_nc_mixer(
        g2_vec, ng, nonmagnetic, mixing_alpha, mag_mixing_alpha, mixing_history,
        precond_op, m, device, mag_mixer=mag_mixer)
    # diago schedule: nonmagnetic keeps the spinor-family "linear" default; a
    # magnetic run may opt into the collinear "quadratic" schedule, which drives
    # the eigensolve tolerance down with the residual instead of flooring it.
    diago_schedule = "linear" if nonmagnetic else mag_diago_schedule

    projs_b = projectors_b(bk, system.positions)
    q_so = dij_so = None
    if system.is_fr:
        from gradwave.core.spinor_proj import build_so_projectors

        q_so, dij_so = build_so_projectors(bk, system)

    # DFT+U: frozen atomic-orbital projectors (positions fixed); the per-site
    # 2×2 spin-block occupation matrix N^I is recomputed from the fresh
    # spinors each iteration (like the density) and lags one step into V_U —
    # mirrors the collinear scf() bookkeeping (scf/loop.py).
    hub = hub_q = None
    n_hub_nc = None
    if hubbard:
        from gradwave.core.hubbard import build_hubbard_projectors, hubbard_projectors
        hub = build_hubbard_projectors(system, hubbard)
        hub_q = hubbard_projectors(hub, system.positions)
        n_hub_nc = [torch.zeros(2 * s["dim"], 2 * s["dim"], dtype=CDTYPE, device=device)
                    for s in hub.sites]

    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, vol)
    vloc_r = g_to_r_box(vloc_g, real=True)

    # E_ewald is constant across the loop (positions frozen) — build it once.
    e_ew = ewald_energy(system.positions, system.charges, grid.cell)

    # initial spinors: alternate up/down lowest plane waves
    coeffs = spinor_pw_seed(nk, nbands, m_pw, device)
    t2 = torch.cat([bk.t, bk.t], dim=-1)
    mask2 = torch.cat([bk.mask, bk.mask], dim=-1)

    # meta-GGA (needs_tau): the KE-density matrix (τ_0, τ⃗) is an orbital field,
    # NOT mixed — it is rebuilt each iteration from the current spinors and
    # projected into the local frame at point of use. Bootstrap it from the seed
    # so iteration 1 has a valid τ for the τ-dependent H (refined immediately).
    tau_scalar, tau_vec = _bootstrap_spinor_tau(
        xc, coeffs, system, nbands, nk, bk, grid, vol, m_pw, device)

    scheme = SCHEMES[smearing]
    e_free_prev, converged, history = None, False, []
    mu = 0.0
    # adaptive mixing-backoff state: a global step multiplier layered on top of
    # base_step_scale, cut when the residual stops falling (see the loop below).
    adapt_mult, last_backoff, stall_window = 1.0, 0, 6

    # nonmagnetic + SOC keeps the full crystal symmetry (m⃗ ≡ 0): reduce k to
    # the IBZ in setup_system and symmetrize ρ each step, exactly as the scalar
    # path does. With a MagneticSymmetrizer, m⃗ is symmetrized as well —
    # spatially like ρ but mixed by the per-op axial 3×3 (s_T·det(S)·S).
    def symmetrize(r_out):
        return symmetrize_rho(system.rho_symmetrizer, r_out, grid)

    def symmetrize_m(m_r):
        if not mag_sym_active or nonmagnetic:
            return m_r
        m_g = torch.stack([r_to_g(m_r[i].to(CDTYPE)) for i in range(3)])
        m_g = system.rho_symmetrizer.apply_m(m_g)
        return g_to_r_box(m_g, real=True)

    def vec_of(fields):
        return pack_grid_channels(fields, mask_flat)

    for it in range(1, max_iter + 1):
        t_it = time.perf_counter()
        # meta-GGA: local-frame τ_± (held fixed in v_xc/B_xc) and the 2×2 v_τ
        # operator, both from the stored (τ_0, τ⃗) and the current (ρ, m⃗).
        (tau_up, tau_dn), metagga_op = _nc_metagga_step(
            xc, rho, m, grid, system, tau_scalar, tau_vec, nonmagnetic, bk, m_pw)
        v_r, b_xc = _nc_effective_potential(
            xc, rho, m, grid, system, vloc_r, nonmagnetic, constrain_dirs,
            constrain_lambda, constrain_mode, constrain_target_mag, atom_weights,
            vol, tau_up=tau_up, tau_dn=tau_dn)

        # DFT+U: the 2×2 spin-block D-matrix from the PREVIOUS iteration's
        # occupation matrix (zero on the first iteration) — lags one step
        # into V_U exactly like v_r/b_xc lag one step into the density.
        hub_dij_nc = None
        if hub is not None:
            from gradwave.core.hubbard import hubbard_dmatrix_noncollinear
            hub_dij_nc = hubbard_dmatrix_noncollinear(
                n_hub_nc, hub.sites, hub.nproj, device)

        tol_eff = adaptive_diago_tol(it, history, diago_tol,
                                     system.n_electrons, schedule=diago_schedule)
        eigs, coeffs, h = _solve_spinor_bands(
            bk, grid, v_r, b_xc, projs_b, q_so, dij_so, coeffs, t2, mask2,
            tol_eff, mixed_precision, mp_crossover, metagga_op=metagga_op,
            hub_q=hub_q, hub_dij=hub_dij_nc)

        mu = float(find_fermi(eigs, system.kweights, scheme, width,
                              system.n_electrons, degeneracy=1.0))
        mu_t = torch.tensor(mu, dtype=RDTYPE, device=device)
        occ, s_ent = occupations_and_entropy(eigs, mu_t, scheme, width, degeneracy=1.0)
        entropy_term = -width * (system.kweights[:, None] * s_ent).sum()

        # DFT+U occupation matrices from the fresh spinors; E_U (Dudarev),
        # evaluated at self-consistency with this iteration's orbitals/occ
        # (like the rest of the energy breakdown below).
        e_hub = torch.zeros((), dtype=RDTYPE, device=device)
        if hub is not None:
            from gradwave.core.hubbard import (
                hubbard_energy,
                occupation_matrices_noncollinear,
            )
            n_hub_nc = occupation_matrices_noncollinear(
                hub_q, coeffs[..., :m_pw], coeffs[..., m_pw:], occ,
                system.kweights, hub.sites)
            e_hub = hubbard_energy(n_hub_nc, hub.sites)

        # Pauli-decomposed density matrix — the shared band-chunked, fused-FFT
        # accumulation (scf/spinor_common.py)
        nbc = h._band_chunk(nk, coeffs.device, coeffs.element_size())
        w_kb = system.kweights[:, None] * occ
        rho_out, m_out = pauli_density_accumulate(
            coeffs, w_kb, bk, grid.shape, m_pw, nbands, nbc, device)
        rho_out, m_out = rho_out / vol, m_out / vol
        rho_out = symmetrize(rho_out)  # no-op unless IBZ symmetry is active
        m_out = symmetrize_m(m_out)  # no-op unless MAGNETIC symmetry is active
        if nonmagnetic:
            # pin m⃗ ≡ 0 BEFORE E_xc so the pinned state's energy sees no
            # eigensolver noise in m_out (mirror the b_xc zeroing above)
            m_out = torch.zeros_like(m_out)

        # meta-GGA: rebuild the KE-density matrix (τ_0, τ⃗) from the NEW spinors
        # (consistent with rho_out/m_out) for the energy, and carry it to the
        # next iteration's H. τ rides the orbitals; it is never mixed.
        if xc.needs_tau:
            tau_scalar, tau_vec = spinor_tau_matrix_b(
                coeffs, occ, system.kweights, bk, grid.shape, vol, m_pw)
            tau_up_e, tau_dn_e = local_frame_tau(m_out, tau_scalar, tau_vec,
                                                 xc.m_eps)
        else:
            tau_up_e = tau_dn_e = None

        # energies
        energies = _nc_energy_breakdown(
            coeffs, occ, t2, entropy_term, rho_out, m_out, q_so, dij_so, projs_b,
            m_pw, vloc_g, e_ew, system, grid, xc, vol, nk,
            tau_up=tau_up_e, tau_dn=tau_dn_e)
        if hub is not None:
            energies.hubbard = e_hub
        e_free = float(energies.free_energy)

        if nonmagnetic:  # m_out already pinned to 0 above (before E_xc)
            vin, vout = vec_of([rho]), vec_of([rho_out])
        else:
            vin, vout = vec_of([rho, *m]), vec_of([rho_out, *m_out])
        res_norm = float(torch.linalg.norm(vout - vin)) * vol
        de = record_iteration(history, it, e_free, e_free_prev, res_norm, t_it)
        if verbose:
            mv = [float(m_out[i].mean()) * vol for i in range(3)]
            print(f"  NC-SCF {it:3d}  F = {e_free:+.8f}  dE = {de:.2e}  "
                  f"|dρ,m| = {res_norm:.2e}  m⃗ = ({mv[0]:+.3f},{mv[1]:+.3f},{mv[2]:+.3f})",
                  flush=True)

        if convergence_gate(de, res_norm, tol_eff, etol, rhotol, diago_tol):
            converged = True
            rho, m = rho_out, m_out
            break

        e_free_prev = e_free
        # Stoner preconditioner on the (longitudinal) magnetization channel:
        # near a FM solution the m ∥ moment-axis mode carries a Stoner-enhanced
        # gain that stalls history mixing (fcc Ni + SOC: the m_z residual
        # limit-cycles at ~1e-4 while ρ/m_x/m_y sit three orders lower). The
        # collinear cure (scf/spin_precond.py) applied to the non-collinear
        # moment channel: build the Newton model (I − χ₀^diag K_mm)⁻¹ from the
        # current spinor states + longitudinal kernel and apply it to each
        # Cartesian m-block (transverse blocks are in the operator's identity
        # regime near collinearity). Rebuilt each iteration like the collinear
        # path; None in the insulating limit.
        if spin_precond and not nonmagnetic:
            from gradwave.scf.spin_precond import build_stoner_precond_nc
            sp = build_stoner_precond_nc(
                system, coeffs, eigs, mu, scheme, width, rho_out, m_out, xc,
                m_pw)
            if sp is None:
                mixer.extra_precond = None
            else:
                def _spin_pc(rvec, _sp=sp, _ng=ng):
                    out = rvec.clone()
                    for c in (1, 2, 3):  # m_x, m_y, m_z channels
                        out[c * _ng:(c + 1) * _ng] = _sp.apply(
                            rvec[c * _ng:(c + 1) * _ng])
                    return out
                mixer.extra_precond = _spin_pc
        # adaptive fallback against a stalled/oscillating residual (halve the
        # global mixing step and drop the DIIS history) — see _nc_adaptive_backoff
        adapt_mult, last_backoff = _nc_adaptive_backoff(
            adaptive, it, last_backoff, stall_window, adapt_mult, history, mixer,
            base_step_scale, verbose)
        if mixer_hook is not None:
            mixer_hook(it, vin, vout)
        mixed = mixer.step(vin, vout)
        fields = _unpack_mixed_fields(mixed, n_chan, ng, mask_flat, grid, device)
        rho = fields[0]
        if not nonmagnetic:
            m = torch.stack(fields[1:])

    if not converged:
        logger.warning(
            "NC-SCF did NOT converge in %d iterations: F=%+.8f eV, |drho|=%.3e",
            it, e_free, res_norm)
    m_int = [float(m[i].mean()) * vol for i in range(3)]
    m_norm = torch.sqrt((m**2).sum(dim=0))
    return NCResult(
        converged=converged, n_iter=it, energies=energies, fermi=mu,
        mag_vec=tuple(m_int), mag_abs=float(m_norm.mean()) * vol,
        rho=rho, m=m, eigenvalues=eigs, system=system, history=history,
        coeffs=coeffs, occupations=occ, hub_occ=n_hub_nc,
    )


def band_structure_nc(res: NCResult, xc: NoncollinearXC, kpts_frac: np.ndarray,
                      nbands: int | None = None, diago_tol: float = 1e-8,
                      chunk: int = 4, verbose: bool = False,
                      mixed_precision: bool = False) -> np.ndarray:
    """Spinor band energies along arbitrary k (SOC-aware): rebuilds the
    converged potential from (ρ, m⃗) and solves per path chunk."""
    from gradwave.core.batch import build_batched
    from gradwave.core.hamiltonian import build_projector_data
    from gradwave.core.spinor_proj import build_so_projectors
    from gradwave.grids import build_gsphere
    from gradwave.pseudo.kb import beta_form_factors
    from gradwave.solvers.davidson import davidson_batched_ms

    system = res.system
    grid = system.grid
    device = res.rho.device
    nbands = nbands or 2 * system.nbands
    kpts = np.asarray(kpts_frac, dtype=float)
    nonmagnetic = float(res.m.abs().max()) < 1e-12

    # meta-GGA (needs_tau): rebuild the KE-density-matrix (τ_0, τ⃗) from the
    # CONVERGED spinor orbitals + occupations at the SCF's own k-mesh
    # (system.batch — NCResult now carries both res.coeffs and res.occupations,
    # PR #103's occupations field originally added for the SOC stress), project
    # to the local-frame per-spin τ_±, and form the fixed real-space v_τ
    # operator fields (v_τ0, v_τ⃗) exactly as the SCF's _nc_metagga_step does.
    # These fields are k-independent (like v_r/b_xc below); only the operator
    # itself (spinor_metagga_tau_operator) needs the PATH k-point's own bk.
    tau_up = tau_dn = None
    if getattr(xc, "needs_tau", False):
        bk0 = system.batch
        tau_scalar, tau_vec = spinor_tau_matrix_b(
            res.coeffs, res.occupations, system.kweights, bk0, grid.shape,
            grid.volume, bk0.npw_max)
        tau_up, tau_dn = local_frame_tau(res.m, tau_scalar, tau_vec, xc.m_eps)

    # rebuild converged V, B (meta-GGA: v_xc is evaluated at the converged τ_±,
    # held fixed exactly as the SCF's _nc_effective_potential does)
    rho_g_box = r_to_g(res.rho.to(CDTYPE))
    v_h = g_to_r_box(hartree_potential_g(rho_g_box, grid.g2), real=True)
    v_xc, b_xc, _ = vxc_and_bxc(xc, res.rho, res.m, grid, rho_core=system.rho_core,
                                tau_up=tau_up, tau_dn=tau_dn)
    if nonmagnetic:
        b_xc = torch.zeros_like(b_xc)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)
    vloc_r = g_to_r_box(vloc_g, real=True)
    v_r = v_h + v_xc + vloc_r

    # meta-GGA v_τ operator fields (v_τ0, v_τ⃗), fixed at the converged state —
    # None for LDA/GGA, in which case metagga_op stays None per path chunk.
    v0 = vvec = None
    if tau_up is not None:
        vtu, vtd = vtau_up_dn(xc, res.rho, res.m, grid, tau_up, tau_dn,
                              rho_core=system.rho_core)
        if nonmagnetic:
            v0 = 0.5 * (vtu + vtd)
            vvec = torch.zeros(3, *res.rho.shape, dtype=RDTYPE, device=device)
        else:
            v0, vvec = tau_operator_fields(vtu, vtd, res.m, xc.m_eps)

    eigs = np.empty((len(kpts), nbands))
    for lo in range(0, len(kpts), chunk):
        hi = min(lo + chunk, len(kpts))
        spheres = [build_gsphere(grid, system.ecut, k, device=device)
                   for k in kpts[lo:hi]]
        npw_max = max(sp.npw for sp in spheres)
        so_tabs = [torch.zeros(hi - lo, u.n_proj, npw_max, dtype=RDTYPE, device=device)
                   for u in system.upfs]
        pd_list = []
        for ic, sph in enumerate(spheres):
            import numpy as _np

            q = _np.sqrt(sph.kpg2.cpu().numpy())
            for sp_i, u in enumerate(system.upfs):
                so_tabs[sp_i][ic, :, : sph.npw] = torch.as_tensor(
                    beta_form_factors(u, q), dtype=RDTYPE, device=device)
            pd_list.append(build_projector_data(
                sph, system.species_of_atom,
                [t[ic, :0] for t in so_tabs], [[] for _ in system.upfs],
                [torch.as_tensor(u.dij, dtype=RDTYPE, device=device)
                 for u in system.upfs], grid.volume))
        bk = build_batched(spheres, pd_list, device=device)
        q_so, dij_so = build_so_projectors(bk, system, so_tables=so_tabs)
        metagga_op = None
        if v0 is not None:
            def metagga_op(c, _v0=v0, _vv=vvec, _bk=bk, _mpw=bk.npw_max):
                return spinor_metagga_tau_operator(c, _v0, _vv, _bk, grid.shape, _mpw)
        h = SpinorHamiltonian(bk, grid.shape, v_r, torch.zeros_like(b_xc)
                              if nonmagnetic else b_xc,
                              projectors_b(bk, system.positions),
                              q=q_so, dij_so=dij_so, metagga_op=metagga_op)
        c0 = torch.zeros(hi - lo, nbands, 2 * bk.npw_max, dtype=CDTYPE, device=device)
        for b_i in range(nbands):
            c0[:, b_i, (b_i // 2) + (b_i % 2) * bk.npw_max] = 1.0
        t2 = torch.cat([bk.t, bk.t], dim=-1)
        mask2 = torch.cat([bk.mask, bk.mask], dim=-1)
        out = davidson_batched_ms(h.apply, c0, t2, mask2, tol=diago_tol,
                                  max_iter=100, mixed_precision=mixed_precision)
        eigs[lo:hi] = out.eigenvalues.cpu().numpy()
        if verbose:
            print(f"  nc-band chunk {lo}-{hi - 1}/{len(kpts) - 1} "
                  f"res={float(out.residual_norms.max()):.1e}", flush=True)
    return eigs
