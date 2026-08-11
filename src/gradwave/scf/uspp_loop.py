"""Ultrasoft/PAW plane-wave SCF — stage 1: augmentation + S-operator (Layer B).

The ultrasoft generalized eigenproblem H|ψ⟩ = ε S|ψ⟩ with

    S = 1 + Σ_a Σ_ij q_ij |β^a_i⟩⟨β^a_j|,   q_ij = ∫ Q_ij(r⃗) d³r
    ρ(r) = ρ_smooth(r) + ρ_aug(r),
    ρ_aug(G) = (1/Ω) Σ_a e^{−iG·τ_a} Σ_ij ρ^a_ij Q̃_ij(G)
    ρ^a_ij = Σ_nk w_k f_nk ⟨ψ|β^a_i⟩⟨β^a_j|ψ⟩
    D^scr_ij = D_ij + ∫ v_eff(r) Q_ij(r⃗−τ_a) d³r        (rebuilt each iteration)

with the augmentation form factors from the UPF's per-L radial functions:

    Q̃_(i,mi),(j,mj)(G) = 4π Σ_{LM} (−i)^L c^{LM}_{limi,ljmj} Y_LM(Ĝ) ∫ q^L_ij(r) j_L(Gr) dr

(c = real Gaunt coefficients, core/gaunt.py). The plane-wave energy terms are
assembled exactly as in the NC loop but with ρ = ρ_s + ρ_aug and the BARE
D_ij in E_NL; the one-center PAW corrections (stage 2, postscf/paw_onsite.py)
add per-atom radial-grid terms on top and are required to match QE's total.

Deliberately per-k and unbatched — correctness first; the batched fast path
can absorb it later.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.batch import BatchedK
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.fftbox import box_to_sphere, g_to_r, g_to_r_box, r_to_g
from gradwave.core.hamiltonian import ProjectorData, becp, projectors
from gradwave.core.hubbard import HubbardManifold
from gradwave.core.occupations import (
    SCHEMES,
)
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import CDTYPE, CDTYPE_LOW, RDTYPE
from gradwave.grids import FFTGrid, GSphere
from gradwave.scf.common import (
    MP_CROSSOVER,
    adaptive_diago_tol,
    assemble_pw_energies,
    convergence_gate,
    hubbard_u_ramp_scale,
    mix_hubbard_occ,
    record_iteration,
    shared_fermi_occupations,
    spin_xc_energy,
    symmetrize_rho,
    validate_hubbard_conv,
    warm_start_densities,
)
from gradwave.scf.guess import sad_density
from gradwave.scf.layout import MixLayout
from gradwave.scf.loop import (
    effective_potentials,
    resolve_atom_moments,
    vtau_potential,
    vtau_spin_potential,
)
from gradwave.scf.mixing import BroydenMixer, JohnsonMixer, PulayMixer
from gradwave.scf.options import SCFOptions
from gradwave.scf.paw_onsite import OneCenter
from gradwave.scf.results import USPPResult
from gradwave.scf.uspp_hubbard import USPPHubbardData
from gradwave.scf.uspp_setup import USPPSystem
from gradwave.solvers.precond import teter

if TYPE_CHECKING:
    # Annotation-only, mirroring scf/loop.py's own TYPE_CHECKING-only
    # DistKContext import: scf_uspp() takes an optional distributed context,
    # but gradwave.distributed itself never imports this module at runtime, so
    # importing it eagerly here would be a needless top-level dependency on
    # torch.distributed for every non-distributed run.
    from gradwave.distributed import DistKContext
    from gradwave.scf.recorder import SCFRecorder

logger = logging.getLogger(__name__)

# A warm-start source for scf_uspp(): either a converged USPPResult, or the
# plainer dict/SimpleNamespace-like checkpoint view checkpoint.as_start_from()
# hands back (mirrors scf/loop.py's own _StartFrom for the NC driver). The
# `dict[str, USPPSystem | int]` shape a previous, mechanically-inferred
# annotation carried here was never actually constructed anywhere in this
# codebase (every real dict-shaped caller uses the SimpleNamespace/Tensor/list
# shape below; every other caller passes a real USPPResult, e.g.
# tests/integration/test_uspp_warmstart.py, or opts out via `cast(Any, ...)`
# at the call site in calculator.py) -- dropped in favor of the member that's
# actually exercised.
_USPPStartFrom = (
    dict[str, "SimpleNamespace | int | torch.Tensor | list[torch.Tensor] | None"]
    | USPPResult
    | None
)


class _HkS:
    """H and S applies at one k for fixed v_eff and screened D."""

    def __init__(
        self,
        sphere: GSphere,
        shape: tuple[int, int, int],
        v_eff_r: torch.Tensor,
        pd: ProjectorData,
        p: torch.Tensor,
        dscr: torch.Tensor,
        q_full: torch.Tensor,
        hub_sphi: torch.Tensor | None = None,
        hub_d: torch.Tensor | None = None,
    ) -> None:
        self.sphere, self.shape = sphere, shape
        self.v_eff_r = v_eff_r
        self.p = p
        self.dscr = dscr.to(CDTYPE)
        self.q = q_full.to(CDTYPE)
        self.t = HBAR2_2M * sphere.kpg2
        # DFT+U: S-dressed atomic-orbital projectors + Dudarev D (apply
        # convention wants D^T = conj(D) for Hermitian D, like the NC path)
        self.hub_sphi = hub_sphi
        self.hub_d = hub_d

    def h(self, c: torch.Tensor) -> torch.Tensor:
        out = self.t * c
        psi = g_to_r(c, self.sphere.flat_idx, self.shape)
        out = out + box_to_sphere(r_to_g(psi * self.v_eff_r), self.sphere.flat_idx)
        b = becp(self.p, c)
        out = out + (b @ self.dscr) @ self.p
        if self.hub_sphi is not None:
            # both call sites (_solve_bands_uspp's batched and per-k paths)
            # only ever pass hub_sphi/hub_d together, gated by the same
            # `hub is not None` check.
            assert self.hub_d is not None
            bh = becp(self.hub_sphi, c)
            out = out + (bh @ self.hub_d) @ self.hub_sphi
        return out

    def s(self, c: torch.Tensor) -> torch.Tensor:
        b = becp(self.p, c)
        return c + (b @ self.q) @ self.p


def davidson_gen(
    hs: _HkS,
    x0: torch.Tensor,
    nbands: int,
    tol: float,
    max_iter: int = 60,
    max_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block Davidson for H x = ε S x. x0 (nb0 ≥ nbands, npw).

    HV/SV are cached — H and S act only on the block of new directions each
    iteration; restarts contract them through the Ritz rotation.
    """
    npw = x0.shape[1]
    max_dim = max_dim or min(npw, max(4 * nbands, nbands + 24))

    def ortho_block(d, v_prev):
        """Orthonormalize d against v_prev and internally (two GS passes)."""
        for _ in range(2):
            if v_prev is not None:
                d = d - (d @ v_prev.conj().T) @ v_prev
            q, _ = torch.linalg.qr(d.T, mode="reduced")
            d = q.T.contiguous()
        return d

    v_sub = ortho_block(x0, None)
    hv, sv = hs.h(v_sub), hs.s(v_sub)
    # Bound before the loop purely so ty can see eps/x as always real Tensors
    # after it (max_iter is always >= 1 in practice, same as
    # solvers/davidson.py's identical precedent); overwritten on the loop's
    # first pass every real invocation. hx/sx are always reassigned earlier in
    # each iteration than any use, so they don't need this treatment.
    eps = torch.zeros(nbands, dtype=RDTYPE, device=x0.device)
    x = v_sub[:nbands]
    hx = sx = None
    for _ in range(max_iter):
        h_sub = v_sub.conj() @ hv.T
        s_sub = v_sub.conj() @ sv.T
        h_sub = 0.5 * (h_sub + h_sub.conj().T)
        s_sub = 0.5 * (s_sub + s_sub.conj().T)
        ell = torch.linalg.cholesky(s_sub)
        a = torch.linalg.solve_triangular(ell, h_sub, upper=False)  # L⁻¹H
        # (L⁻¹H)L⁻† = (L⁻¹ (L⁻¹H)†)† — solve with L again, then dagger
        a = torch.linalg.solve_triangular(ell, a.conj().T, upper=False).conj().T
        w, u = torch.linalg.eigh(0.5 * (a + a.conj().T))
        u = torch.linalg.solve_triangular(ell.conj().T, u, upper=True)
        eps = w[:nbands].real
        u_r = u[:, :nbands].T.to(CDTYPE)
        x, hx, sx = u_r @ v_sub, u_r @ hv, u_r @ sv
        r = hx - eps[:, None].to(CDTYPE) * sx
        rnorm = torch.linalg.norm(r, dim=1)
        if float(rnorm.max()) < tol:
            return eps, x
        active = rnorm > tol
        d = teter(r[active], hs.t, eps[active])
        if v_sub.shape[0] + int(active.sum()) > max_dim:
            # restart from the Ritz vectors, RE-ORTHONORMALIZED in the
            # standard metric (they are S-orthonormal; using them as-is lets
            # the basis drift toward linear dependence and the ill-conditioned
            # overlap Cholesky then produces spurious below-minimum states).
            # HV/SV rotate with the same triangular transform: x = RᵀQᵀ ⇒
            # basis Qᵀ = R⁻ᵀ x.
            qq, rr = torch.linalg.qr(x.T, mode="reduced")
            v_sub = qq.T.contiguous()
            rt = rr.T  # x = RᵀQᵀ (plain transpose), so Qᵀ = (Rᵀ)⁻¹x
            hv = torch.linalg.solve_triangular(rt, hx, upper=False)
            sv = torch.linalg.solve_triangular(rt, sx, upper=False)
        d = ortho_block(d, v_sub)
        v_sub = torch.cat([v_sub, d], dim=0)
        hv = torch.cat([hv, hs.h(d)], dim=0)
        sv = torch.cat([sv, hs.s(d)], dim=0)
    return eps, x


@dataclass
class _IterOps:
    """Frozen per-run operators and tables for `_scf_iteration` — everything
    the one-iteration SCF map needs beyond the state it maps. Built once by
    `_build_iter_ops`; newton.py and the mixer rig construct it directly to
    evaluate the raw map without the driver."""

    system: USPPSystem
    xc: XCFunctional | SpinXC
    nspin: int
    smearing: str
    width: float
    batched: bool
    projs: list[torch.Tensor]
    bk: BatchedK | None
    p_b: torch.Tensor | None
    hub: USPPHubbardData | None
    vloc_g: torch.Tensor
    vloc_r: torch.Tensor
    e_ewald: torch.Tensor  # constant E_ewald (frozen positions), built once
    phase_pos: torch.Tensor
    is_paw: bool
    onec: dict[int, OneCenter] | None  # per-species OneCenter ({sp: oc}; list also indexes)
    grid: FFTGrid
    vol: float
    dev: torch.device
    shape: tuple[int, int, int]
    mask_flat: torch.Tensor
    g_spin: int
    nk: int
    nb: int
    mixed_precision: bool = False
    dist_ctx: "DistKContext | None" = None  # k-point-sharded distributed SCF (see
    # gradwave.distributed): `system` is THIS RANK's local k-shard; None (default)
    # runs the ordinary single-process path, byte-for-byte unchanged.
    kweights_global: torch.Tensor | None = None  # full-mesh kweights, gathered once
    # by _build_iter_ops -- always a real Tensor by the time _scf_iteration reads it
    # (equal to system.kweights itself when dist_ctx is None); Optional only so this
    # trailing dataclass field can default like its neighbors.
    hub_occ_mix: float = 1.0  # DFT+U occupation-matrix damping β in (0,1] (1.0 = raw one-step lag)
    hub_u_ramp_iters: int = 0  # DFT+U linear U-ramp length in iterations (0 = off)
    boundary: str = "periodic"  # electrostatic BC (periodic | open_z | open_z_metal);
    # the ESM modes add the correction to v_eff and the energy (core/energies/esm)
    esm_bias: float = 0.0  # applied capacitor bias [V] for boundary="open_z_metal"


_MP_CROSSOVER = MP_CROSSOVER  # diago tol above this runs the fp32 draft solves


def _resolve_start_mag(
    start_mag: list[float] | None, species_of_atom: list[int], n_species: int
) -> list[float]:
    """Per-ATOM moment fractions from start_mag, mirroring loop._seed_density:
    accept one entry per atom (AFM/ferrimagnetic seeds) or one per species
    (broadcast to that species' atoms); raise on a length matching neither."""
    return resolve_atom_moments(start_mag, species_of_atom, n_species, default=0.0)


def _species_atoms(system: USPPSystem) -> dict[int, list[int]]:
    """Atoms grouped by species, for batching per-atom augmentation
    contractions into one einsum per species."""
    groups: dict[int, list[int]] = {}
    for a, sp in enumerate(system.species_of_atom):
        groups.setdefault(sp, []).append(a)
    return groups


def aug_dmat_batched(
    system: USPPSystem, fields_g: torch.Tensor, phase_pos: torch.Tensor
) -> torch.Tensor:
    """Block-diagonal ∫ w_c(r) Q_ij(r−τ_a) d³r for C grid fields at once — the
    augmentation screening of D.

    ``fields_g`` (C, nG) is the masked-G coefficients of the C real fields
    (``r_to_g(w).reshape(-1)[dens_mask]``); ``phase_pos`` is e^{+iG·τ} on the
    density sphere (nG, na). Returns (C, nproj, nproj) [q_full dtype]: the
    Hermitized augmentation D per channel — callers add dij_full and the PAW
    one-center ddd. Species-batched (one einsum per species over channel×atom),
    and pure einsum, so it stays autograd-safe for the force/stress graph and
    gives block placement identical to the per-atom form (placed by atom index).
    """
    out = torch.zeros(
        (fields_g.shape[0], *system.q_full.shape), dtype=system.q_full.dtype, device=fields_g.device
    )
    for sp, atoms in _species_atoms(system).items():
        contr = torch.einsum(
            "ijg,cg,ga->caij", system.aug[sp].q_g.conj(), fields_g, phase_pos[:, atoms]
        )
        herm = (0.5 * (contr + contr.conj().transpose(-2, -1))).real
        for i, a in enumerate(atoms):
            s0, s1 = system.atom_slices[a]
            out[:, s0:s1, s0:s1] = herm[:, i]
    return out


def uspp_potentials_dscr(
    system: USPPSystem,
    xc: XCFunctional | SpinXC,
    rho_s: list[torch.Tensor],
    rho_ij_s: list[list[torch.Tensor]],
    vloc_r: torch.Tensor,
    phase_pos: torch.Tensor,
    onec: list[OneCenter] | dict[int, OneCenter] | None,
    boundary: str = "periodic",
    esm_bias: float = 0.0,
    tau_s: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """(veff_s, dscr_s, e_onec) from the per-channel FULL densities (smooth +
    aug) and per-atom becsums — THE assembly the USPP/PAW SCF iterates with.
    A standalone function (not inlined in `_scf_iteration`) so the
    off-stationarity E↔H consistency gate can test the exact potential and
    screened D the solver applies
    (tests/unit/test_energy_hamiltonian_consistency.py).

    rho_ij_s: [spin][atom] becsum matrices (the mixer-side becsum in the SCF;
    the same-state becsum in the gate). onec: per-species OneCenter list for
    PAW, None for bare USPP.

    tau_s (meta-GGA only): the LAGGING smooth kinetic-energy density τ̃ — a
    grid for nspin=1, [τ̃↑, τ̃↓] for nspin=2 — passed to the local v_xc as a
    held-fixed constant so it carries the multiplicative ∂e/∂ρ|_τ term. The
    τ-response operator −½∇·(v_τ∇ψ) is NOT built here; `_scf_iteration` builds
    it from the same τ̃ and composes it onto the H-apply (see `_uspp_vtau_fields`)."""
    grid = system.grid
    dev = system.positions.device
    nspin = len(rho_s)
    # v_eff = v_H + v_xc(+½ core fold) + v_loc — the identical per-spin assembly
    # the norm-conserving loop iterates (loop.effective_potentials); USPP passes
    # the FULL (smooth + aug) densities and its own vloc_r. The smooth τ̃ rides
    # the density argument for the meta-GGA v_xc; there is no τ compensation
    # charge (the meta-GGA kernel is short-ranged, so n̂ appears only in ρ).
    # effective_potentials takes a bare grid for nspin=1, a [τ↑, τ↓] list for
    # nspin=2 (matching its ρ argument shape). boundary/esm_bias thread the ESM
    # open-boundary electrostatics through the same assembly.
    tau_arg = None if tau_s is None else (tau_s[0] if nspin == 1 else tau_s)
    veff_s = effective_potentials(system, xc, rho_s, vloc_r, tau=tau_arg,
                                  boundary=boundary, esm_bias=esm_bias)

    # screened D per spin/atom: D_ij + Σ_G ṽ_σ(G) e^{iGτ} Q̃_ij(G)* —
    # batched over the atoms of each species (one einsum per species, one
    # q_g.conj() per species, instead of a Python loop of small kernels
    # re-materializing the conjugate per atom)
    mask_flat = grid.dens_mask.reshape(-1)
    fields_g = torch.stack(
        [r_to_g(veff_s[isp].to(CDTYPE)).reshape(-1)[mask_flat] for isp in range(nspin)]
    )
    d_aug = aug_dmat_batched(system, fields_g, phase_pos)
    dscr_s = [d_aug[isp] + system.proj_data[0].dij_full for isp in range(nspin)]
    e_onec = torch.zeros((), dtype=RDTYPE, device=dev)
    if onec is not None:
        dscr_s = [d.clone() for d in dscr_s]
        # one-center per species: ONE batched energy_and_ddd over that
        # species' atoms (a single autograd graph on the table device)
        # instead of a per-atom Python loop with per-atom device syncs
        for sp, atoms in _species_atoms(system).items():
            bec_b = [torch.stack([rho_ij_s[isp][a] for a in atoms]) for isp in range(nspin)]
            e_at, ddd = onec[sp].energy_and_ddd_batch(bec_b[0] if nspin == 1 else bec_b)
            e_onec = e_onec + e_at.sum().to(dev)
            ddd_s = [ddd] if nspin == 1 else ddd
            for isp in range(nspin):
                dd = ddd_s[isp].to(dev)
                for i, a in enumerate(atoms):
                    s0, s1 = system.atom_slices[a]
                    dscr_s[isp][s0:s1, s0:s1] += dd[i]
    return veff_s, dscr_s, e_onec


def _bootstrap_tau_uspp(
    xc: XCFunctional | SpinXC,
    rho_s: list[torch.Tensor],
    nspin: int,
) -> list[torch.Tensor] | None:
    """Orbital-free Thomas-Fermi seed τ̃ = C_F ρ^{5/3} for iteration 1 (and after
    a solver-blowup reseed), where the USPP cold start has no orbitals to build
    the real τ̃ from — the USPP analogue of `scf.loop._bootstrap_tau`. r2SCAN
    hard-requires a τ, so a valid positive seed is needed before the first solve;
    it is refined to the true ½Σf|∇ψ̃|² immediately after. None for GGA/LDA."""
    if not xc.needs_tau:
        return None
    if nspin == 1:
        cf = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0)
        return [cf * rho_s[0].clamp_min(0.0) ** (5.0 / 3.0)]
    # spin-polarized per-channel TF constant (6π² per spin)
    cf = 0.3 * (6.0 * math.pi**2) ** (2.0 / 3.0)
    return [cf * rho_s[isp].clamp_min(0.0) ** (5.0 / 3.0) for isp in range(nspin)]


def _uspp_vtau_fields(
    ops: "_IterOps",
    rho_s: list[torch.Tensor],
    tau_s: list[torch.Tensor] | None,
) -> list[torch.Tensor] | None:
    """Per-spin scaled v_τ = ∂e_xc/∂τ̃ grids for the smooth generalized-KS
    operator −½∇·(v_τ∇ψ̃), or None when the functional is not τ-dependent or τ̃
    is not yet seeded (iteration 1 of a cold start). Mirrors
    `scf.loop._build_metagga_apply`'s v_τ extraction but on the USPP FULL
    (smooth + aug) density: v_τ is autograded from the SAME density the smooth
    v_xc sees, plus the NLCC core fold."""
    xc, system, grid, nspin = ops.xc, ops.system, ops.grid, ops.nspin
    if not xc.needs_tau or tau_s is None:
        return None
    core = system.rho_core
    if nspin == 1:
        assert isinstance(xc, XCFunctional)
        rho_tot = rho_s[0]
        rho_for_xc = rho_tot if core is None else rho_tot + core
        return [vtau_potential(xc, rho_for_xc, tau_s[0], grid)]
    assert isinstance(xc, SpinXC)
    cu2 = None if core is None else 0.5 * core
    r_u = rho_s[0] if cu2 is None else rho_s[0] + cu2
    r_d = rho_s[1] if cu2 is None else rho_s[1] + cu2
    vu, vd = vtau_spin_potential(xc, r_u, r_d, tau_s[0], tau_s[1], grid)
    return [vu, vd]


def _build_iter_ops(
    system: USPPSystem,
    xc: XCFunctional | SpinXC,
    *,
    nspin: int = 1,
    smearing: str = "none",
    width: float = 0.1,
    batched: bool = True,
    hubbard: list[HubbardManifold] | None = None,
    mixed_precision: bool = False,
    dist_ctx: "DistKContext | None" = None,
    hub_occ_mix: float = 1.0,
    hub_u_ramp_iters: int = 0,
    boundary: str = "periodic",
    esm_bias: float = 0.0,
) -> _IterOps:
    grid = system.grid
    vol = grid.volume
    dev = system.positions.device
    projs = [projectors(pd, system.positions) for pd in system.proj_data]
    # Static across the run (the mesh doesn't change), so gathered once here
    # rather than every iteration alongside the eigenvalues (mirrors
    # scf/loop.py's own `kweights_global`).
    kweights_global = system.kweights
    if dist_ctx is not None:
        from gradwave.distributed import gather_cat

        kweights_global = gather_cat(system.kweights, dist_ctx)
    bk = p_b = None
    if batched or hubbard:
        from gradwave.core.batch import build_batched, projectors_b

        bk = build_batched(system.spheres, system.proj_data, device=dev)
        p_b = projectors_b(bk, system.positions)
    hub = None
    if hubbard:
        from gradwave.scf.uspp_hubbard import build_uspp_hubbard

        # bk/p_b are built above whenever `batched or hubbard`; hubbard is
        # truthy here so that condition already held.
        assert p_b is not None
        hub = build_uspp_hubbard(system, hubbard, bk, p_b)
    vloc_g = local_potential_g(
        system.positions,
        torch.tensor(system.species_of_atom, device=dev),
        system.vloc_tables,
        grid.g_cart,
        vol,
    )
    vloc_r = g_to_r_box(vloc_g, real=True)
    # E_ewald depends only on the (frozen) positions — build once, reuse each step.
    e_ewald = ewald_energy(system.positions, system.charges, grid.cell)
    phase_arg = system.g_sphere @ system.positions.T  # (nGm, na)
    phase_pos = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    is_paw = any(p.is_paw for p in system.paws)
    onec = None
    if is_paw:
        from gradwave.scf.paw_onsite import onecenters

        # tables on the SCF device, cached on the system so forces/stress/
        # response reuse them instead of reconstructing per call
        onec = onecenters(system, xc, device=dev)
    return _IterOps(
        system=system,
        xc=xc,
        nspin=nspin,
        smearing=smearing,
        width=width,
        batched=batched,
        projs=projs,
        bk=bk,
        p_b=p_b,
        hub=hub,
        vloc_g=vloc_g,
        vloc_r=vloc_r,
        e_ewald=e_ewald,
        phase_pos=phase_pos,
        is_paw=is_paw,
        onec=onec,
        grid=grid,
        vol=vol,
        dev=dev,
        shape=grid.shape,
        mask_flat=grid.dens_mask.reshape(-1),
        g_spin=2 if nspin == 1 else 1,
        nk=len(system.spheres),
        nb=system.nbands,
        mixed_precision=mixed_precision,
        dist_ctx=dist_ctx,
        kweights_global=kweights_global,
        hub_occ_mix=hub_occ_mix,
        hub_u_ramp_iters=hub_u_ramp_iters,
        boundary=boundary,
        esm_bias=esm_bias,
    )


def _assemble_iter_energies(
    ops: _IterOps,
    coeffs: list[list[torch.Tensor]],
    occ_s: list[torch.Tensor],
    becps_s: list[list[torch.Tensor]],
    rho_out_s: list[torch.Tensor],
    rho_tot_out: torch.Tensor,
    entropy_term: torch.Tensor,
    e_hub: torch.Tensor,
    e_onec: torch.Tensor,
    tau_s: list[torch.Tensor] | None = None,
) -> EnergyBreakdown:
    """Total-energy breakdown for one SCF-map evaluation: charge-conservation
    check, XC energy (nspin 1/2), and assemble_pw_energies. Pure function of the
    already-computed densities/occupations.

    tau_s (meta-GGA only): the smooth τ̃ [τ̃] / [τ̃↑, τ̃↓] consistent with these
    orbitals, so the smooth E_xc^PW[ρ̃+n̂, τ̃] is evaluated at the same τ̃ the
    Hamiltonian's next step will lag on."""
    system, xc, nspin = ops.system, ops.xc, ops.nspin
    grid, vol, vloc_g = ops.grid, ops.vol, ops.vloc_g
    hub, e_ewald = ops.hub, ops.e_ewald
    core = system.rho_core

    n_tot = float(rho_tot_out.sum()) * vol / grid.n_points
    if abs(n_tot - system.n_electrons) >= 1e-5:
        raise ValueError(f"charge not conserved: {n_tot:.8f} vs {system.n_electrons}")

    rho_g_out = r_to_g(rho_tot_out.to(CDTYPE))
    from gradwave.core.density import sigma_from_rho

    if nspin == 1:
        # nspin=1 is always dispatched with a collinear XCFunctional (mirrors
        # scf/loop.py's own nspin-based xc dispatch).
        assert isinstance(xc, XCFunctional)
        rho_xc_out = rho_tot_out if core is None else rho_tot_out + core
        sigma = sigma_from_rho(rho_xc_out, grid.g_cart) if xc.needs_gradient else None
        tau_xc = tau_s[0] if (xc.needs_tau and tau_s is not None) else None
        e_xc = xc.energy(rho_xc_out, vol, sigma, tau_xc)
    else:
        e_xc = spin_xc_energy(xc, rho_out_s, core, vol, grid.g_cart, tau_s=tau_s)
    energies = assemble_pw_energies(
        coeffs,
        occ_s,
        system.kweights,
        system.spheres,
        grid,
        vol,
        rho_g_out,
        e_xc,
        vloc_g,
        becps_s,
        system.proj_data[0].dij_full,
        system.positions,
        system.charges,
        entropy_term,
        nspin,
        e_hub=float(e_hub) if hub is not None else 0.0,
        e_onec=e_onec,
        e_ewald=e_ewald,
    )
    from gradwave.core.energies.esm import esm_energy, esm_mode_of

    _esm_mode = esm_mode_of(ops.boundary)
    if _esm_mode is not None:
        # ΔE = open-minus-periodic electrostatic correction on the FULL (smooth +
        # aug) total density — same term the NC loop adds; detached breakdown.
        energies.esm = esm_energy(rho_tot_out, system.positions, system.charges,
                                  grid, mode=_esm_mode, bias=ops.esm_bias)
    return energies


def _hubbard_occ_update(
    ops: _IterOps,
    hub: USPPHubbardData | None,
    coeffs: list[list[torch.Tensor]],
    coeffs_b: list[torch.Tensor | None],
    occ_s: list[torch.Tensor],
    n_hub_s: list[list[torch.Tensor]] | None,
    u_scale: float = 1.0,
) -> tuple[None, torch.Tensor] | tuple[list[list[torch.Tensor]], torch.Tensor]:
    """Refresh the DFT+U per-spin occupation matrices from the fresh orbitals and
    return (n_hub_s, e_hub). No Hubbard manifold → (n_hub_s, 0). The
    _padded_coeffs closure captures coeffs_b by default argument (per-k pads the
    trimmed orbitals back to npw_max).

    ``n_hub_s`` on entry is the PREVIOUS iteration's matrix; it is the damping
    target for the ``ops.hub_occ_mix`` (β) mix that produces the carry-forward
    ``n = (1-β)·n_prev + β·n_new``. ``e_hub`` stays at the FRESH ``n_new`` and is
    scaled by ``u_scale`` for the U-ramp (mirrors the D-matrix scaling in
    ``_solve_bands_uspp``). β=1.0 AND u_scale=1.0 reproduce today's numbers
    bit-for-bit — the NC driver's ``_hubbard_occ_update`` mirrors this exactly.

    Under a distributed (dist_ctx) run, occupation_matrices' k-sum
    (core/hubbard.py's ``einsum("kb,kbp,kbq->pq", ...)``) is only a sum over
    THIS RANK's k-shard; the per-site matrices are all_reduce-summed across
    ranks (a LINEAR operation, safe before the nonlinear Dudarev trace) before
    Tr[n(1−n)] is evaluated, mirroring the density/becsum reduction in
    _build_output_density below — hubbard_occ_and_energy's own e_hub (computed
    from the un-reduced, rank-local matrices) is discarded and recomputed from
    the globally-reduced ones via hubbard_energy directly."""
    system, nspin, batched = ops.system, ops.nspin, ops.batched
    bk, dev, nk, nb = ops.bk, ops.dev, ops.nk, ops.nb
    e_hub = torch.zeros((), dtype=RDTYPE, device=dev)
    if hub is None:
        # n_hub_s is set together with hub at scf_uspp's setup (`n_hub_s =
        # None; if hubbard: n_hub_s = [...]`), so it's None here too.
        assert n_hub_s is None
        return n_hub_s, e_hub
    from gradwave.core.hubbard import hubbard_energy, hubbard_occ_and_energy, occupation_matrices

    # _build_iter_ops only ever builds bk/p_b when `batched or hubbard` was
    # true; hub is not None here means hubbard was, so bk is real regardless
    # of this call's own `batched` value.
    assert bk is not None

    def _padded_coeffs(isp, _cb=coeffs_b):
        if batched:
            return _cb[isp]
        cp = torch.zeros(nk, nb, bk.npw_max, dtype=CDTYPE, device=dev)
        for ik, sph in enumerate(system.spheres):
            cp[ik, :, : sph.npw] = coeffs[isp][ik]
        return cp

    n_hub_new, e_hub = hubbard_occ_and_energy(
        lambda isp, w: occupation_matrices(
            hub.sphi, _padded_coeffs(isp), w, system.kweights, hub.sites
        ),
        occ_s,
        hub.sites,
        nspin,
    )
    if ops.dist_ctx is not None:
        from gradwave.distributed import all_reduce_

        # occupation_matrices (core/hubbard.py) returns n_full[s0:s1, s0:s1]
        # PER SITE -- a non-contiguous view into the shared (nproj, nproj)
        # n_full whenever more than one site shares that tensor (two Si
        # atoms of the same manifold here: dim < nproj). Gloo's all_reduce
        # silently reduces the WRONG bytes on a non-contiguous view (no
        # exception -- see the sibling NC+U distributed fix, PR #196, which
        # hardens all_reduce_ itself against this for the shared primitive);
        # until that lands, force a contiguous copy at this call site so this
        # driver's own DFT+U reduction is correct regardless of merge order.
        n_hub_new = [[all_reduce_(m.contiguous(), ops.dist_ctx) for m in ch] for ch in n_hub_new]
        e_hub = (
            hubbard_energy(n_hub_new[0], hub.sites) + hubbard_energy(n_hub_new[1], hub.sites)
            if nspin == 2
            else 2.0 * hubbard_energy(n_hub_new[0], hub.sites)
        )
    # U-ramp: E_U is linear in U_eff, so scaling the full-U energy by u_scale
    # matches the ramped-U D-matrix built in _solve_bands_uspp this iteration.
    if u_scale != 1.0:
        e_hub = e_hub * u_scale
    # occupation-matrix damping for the NEXT iteration's V_U (energy stays fresh)
    n_hub_new = mix_hubbard_occ(n_hub_s, n_hub_new, ops.hub_occ_mix)
    return n_hub_new, e_hub


def _build_output_density(
    ops: _IterOps,
    coeffs: list[list[torch.Tensor]],
    coeffs_b: list[torch.Tensor | None],
    occ_s: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Smooth densities + per-spin becsum (Hermitized, becsum-symmetrized) +
    augmentation charge for one SCF-map evaluation. Returns (rho_out_s, rho_ij_s,
    becps_s). Accumulation order and conj placement are load-bearing at the 1e-8
    identity floor the tests assert — keep them byte-for-byte."""
    system, nspin, batched = ops.system, ops.nspin, ops.batched
    bk, shape, vol, dev = ops.bk, ops.shape, ops.vol, ops.dev
    nk, nb, p_b, projs = ops.nk, ops.nb, ops.p_b, ops.projs
    phase_pos, grid = ops.phase_pos, ops.grid
    sp_atoms = _species_atoms(system)
    rho_out_s, becps_s = [], []
    rho_ij_s = [
        [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev) for (s0, s1) in system.atom_slices]
        for _ in range(nspin)
    ]
    for isp in range(nspin):
        if batched:
            # k-batched: one band-chunked batched FFT stack for the density,
            # one becp einsum over all k, and the becsum contracted with k
            # folded into the band axis — instead of nk small FFTs plus
            # nk·na tiny einsums per iteration
            from gradwave.core.batch import becp_b, density_b

            # _build_iter_ops always builds bk/p_b when ops.batched (the `if
            # batched or hubbard:` guard there), and _solve_bands_uspp always
            # runs before this and fills coeffs_b[isp] with a real Tensor in
            # its own `if batched:` branch (the in-place mutation its
            # docstring calls out) -- so all three are real here.
            assert bk is not None and p_b is not None
            x_b = coeffs_b[isp]
            assert x_b is not None
            rho_sp = density_b(x_b, occ_s[isp], system.kweights, bk, shape, vol)
            b_all = becp_b(p_b, x_b)
            becps = [b_all[ik] for ik in range(nk)]
            w_all = (system.kweights[:, None] * occ_s[isp]).reshape(-1).to(CDTYPE)
            b_flat = b_all.reshape(nk * nb, -1)
            for a, (s0, s1) in enumerate(system.atom_slices):
                ba = b_flat[:, s0:s1]
                rho_ij_s[isp][a] = torch.einsum("b,bi,bj->ij", w_all, ba.conj(), ba)
        else:
            rho_sp = torch.zeros(shape, dtype=RDTYPE, device=dev)
            becps = []
            for ik, sph in enumerate(system.spheres):
                c = coeffs[isp][ik]
                psi_r = g_to_r(c, sph.flat_idx, shape)
                w = system.kweights[ik] * occ_s[isp][ik]
                rho_sp = rho_sp + torch.einsum("b,bxyz->xyz", w, (psi_r.abs() ** 2)) / vol
                b = becp(projs[ik], c)
                becps.append(b)
                for a, (s0, s1) in enumerate(system.atom_slices):
                    ba = b[:, s0:s1]
                    rho_ij_s[isp][a] = rho_ij_s[isp][a] + torch.einsum(
                        "b,bi,bj->ij", w.to(CDTYPE), ba.conj(), ba
                    )
        if ops.dist_ctx is not None:
            # rho_sp/rho_ij_s[isp] above are each a k-weighted sum over ONLY
            # this rank's k-shard (batched density_b's own reduction, or the
            # per-k loop's running accumulation); all_reduce SUM completes the
            # sum over the full mesh before Hermitizing/symmetrizing below —
            # the same density/becsum reduction point as scf/loop.py's
            # NC driver, just with becsum (complex, per-atom) added alongside
            # the real density grid.
            from gradwave.distributed import all_reduce_

            rho_sp = all_reduce_(rho_sp, ops.dist_ctx)
            rho_ij_s[isp] = [all_reduce_(m, ops.dist_ctx) for m in rho_ij_s[isp]]
        rho_ij_s[isp] = [0.5 * (m + m.conj().T) for m in rho_ij_s[isp]]
        if system.becsum_sym is not None:
            # the collinear driver only ever carries the per-channel
            # BecsumSymmetrizer; the Pauli-channel MagneticBecsumSymmetrizer
            # is the noncollinear (spinor) driver's business.
            from gradwave.scf.paw_symmetry import BecsumSymmetrizer
            assert isinstance(system.becsum_sym, BecsumSymmetrizer)
            rho_ij_s[isp] = system.becsum_sym.apply(rho_ij_s[isp])
        becps_s.append(becps)

        # augmentation charge: all atoms of a species in one einsum
        aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE, device=dev)
        for sp, atoms in sp_atoms.items():
            bec_sp = torch.stack([rho_ij_s[isp][a] for a in atoms])
            aug_sph = aug_sph + torch.einsum(
                "aij,ijg,ga->g", bec_sp, system.aug[sp].q_g, phase_pos[:, atoms].conj()
            )
        aug_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
        aug_box[system.sphere_idx] = aug_sph / vol
        rho_aug = g_to_r_box(aug_box.reshape(shape), real=True)
        rho_out_sp = rho_sp + rho_aug
        rho_out_sp = symmetrize_rho(system.rho_symmetrizer, rho_out_sp, grid)
        rho_out_s.append(rho_out_sp)
    return rho_out_s, rho_ij_s, becps_s


def _solve_bands_uspp(
    ops: _IterOps,
    veff_s: list[torch.Tensor],
    dscr_s: list[torch.Tensor],
    n_hub_s: list[list[torch.Tensor]] | None,
    coeffs: list[list[torch.Tensor | None]],
    coeffs_b: list[torch.Tensor | None],
    tol_eff: float,
    seed_salt: int,
    u_scale: float = 1.0,
    metagga_v_s: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Generalized eigensolve H x = ε S x per spin (batched or per-k). Warm-starts
    from and MUTATES coeffs/coeffs_b IN PLACE — the frozen warm-start contract
    newton.py relies on. The S-normalization is fp64 always (even under mixed
    precision), and x_b is cast to CDTYPE before it. Returns eigs_s.

    ``u_scale`` (default 1.0) linearly scales the Hubbard V_U D-matrix for the
    U-ramp; V_U is exactly linear in U_eff (see hubbard_u_ramp_scale)."""
    system, nspin, batched = ops.system, ops.nspin, ops.batched
    bk, shape, dev = ops.bk, ops.shape, ops.dev
    nk, nb, p_b, projs, hub = ops.nk, ops.nb, ops.p_b, ops.projs, ops.hub
    # cold-start seeds below key on the GLOBAL k-index so a distributed run
    # (system is THIS RANK's local k-shard, ik loops 0..nk_local) reproduces
    # the EXACT SAME per-k random draw a single-process run would use for
    # that same global k-point -- keying on the local loop index alone
    # collides different ranks' k-points onto the same seed (e.g. every
    # rank's local ik=0 would seed identically) and desyncs the two runs'
    # Davidson trajectories from the very first cold-started k-point.
    k_off = ops.dist_ctx.k_start if ops.dist_ctx is not None else 0
    eigs_s = []
    for isp in range(nspin):
        hub_d = None
        if hub is not None:
            from gradwave.core.hubbard import hubbard_dmatrix

            # n_hub_s is set together with hub at scf_uspp's setup and only
            # ever refreshed (never reset to None) while hub stays set.
            assert n_hub_s is not None
            dij_hub = hubbard_dmatrix(n_hub_s[isp], hub.sites, hub.nproj, dev)
            if u_scale != 1.0:  # U-ramp: V_U is linear in U_eff
                dij_hub = dij_hub * u_scale
            hub_d = dij_hub.conj()  # apply wants D^T
        if batched:
            from gradwave.core.batch import becp_b
            from gradwave.scf.uspp_batch import BatchedHS, davidson_gen_batched

            # _build_iter_ops always builds bk/p_b when ops.batched (the `if
            # batched or hubbard:` guard there).
            assert bk is not None and p_b is not None
            smooth = None
            if system.smooth_shape is not None:
                # filter v_eff onto the smooth box (dense G-coeffs restricted to
                # the smooth sphere by Miller), for the dual-grid H-apply
                vg = r_to_g(veff_s[isp].to(CDTYPE)).reshape(-1)[system.smooth2dense]
                v_s = g_to_r_box(vg.reshape(system.smooth_shape), real=True)
                smooth = (system.smooth_shape, system.smooth_flat_idx, v_s)
            hs_b = BatchedHS(
                bk,
                shape,
                veff_s[isp],
                p_b,
                dscr_s[isp],
                system.q_full,
                hub_sphi=hub.sphi if hub else None,
                hub_d=hub_d,
                smooth=smooth,
                metagga_v=None if metagga_v_s is None else metagga_v_s[isp],
            )
            cb_isp = coeffs_b[isp]
            if cb_isp is None:
                # per-k CPU seeds (identical to the per-k path), padded
                x0 = torch.zeros(nk, nb + 4, bk.npw_max, dtype=CDTYPE, device=dev)
                for ik, sph in enumerate(system.spheres):
                    gen = torch.Generator().manual_seed(1234 + ik + k_off + 7777 * isp + seed_salt)
                    xk = torch.randn(
                        nb + 4, sph.npw, generator=gen, dtype=torch.float64
                    ) + 1j * torch.randn(nb + 4, sph.npw, generator=gen, dtype=torch.float64)
                    xk = xk.to(dev) * torch.exp(-0.5 * HBAR2_2M * sph.kpg2 / system.ecut * 4.0)
                    x0[ik, :, : sph.npw] = xk.to(CDTYPE)
            else:
                x0 = cb_isp
            # fp32 draft while the diago tolerance is loose; the subspace
            # reduction inside stays fp64 and the S-normalization below is
            # fp64 always, so the fp64 finish is bit-identical physics
            use_low = ops.mixed_precision and tol_eff > _MP_CROSSOVER
            eig_b, x_b = davidson_gen_batched(
                hs_b, x0.to(CDTYPE_LOW) if use_low else x0, nb, tol=tol_eff
            )
            x_b = x_b.to(CDTYPE)
            b_all = becp_b(p_b, x_b)
            snorm = (x_b.abs() ** 2).sum(dim=-1) + torch.einsum(
                "kbi,ij,kbj->kb", b_all.conj(), system.q_full.to(CDTYPE), b_all
            ).real
            x_b = x_b / torch.sqrt(snorm)[..., None]
            coeffs_b[isp] = x_b
            for ik, sph in enumerate(system.spheres):
                coeffs[isp][ik] = x_b[ik, :, : sph.npw]
            eigs_s.append(eig_b)
            continue
        if metagga_v_s is not None:
            # scf_uspp gates needs_tau to the batched solver up front; this is
            # the belt-and-braces guard for a direct per-k caller.
            raise NotImplementedError(
                "meta-GGA on the USPP/PAW path requires the batched solver "
                "(batched=True); the per-k τ operator is not wired"
            )
        eigs_l = []
        for ik, sph in enumerate(system.spheres):
            hs = _HkS(
                sph,
                shape,
                veff_s[isp],
                system.proj_data[ik],
                projs[ik],
                dscr_s[isp],
                system.q_full,
                hub_sphi=(hub.sphi[ik, :, : sph.npw] if hub else None),
                hub_d=hub_d,
            )
            c_isp_ik = coeffs[isp][ik]
            if c_isp_ik is None:
                # seed on CPU (device-independent determinism), then move
                gen = torch.Generator().manual_seed(1234 + ik + k_off + 7777 * isp + seed_salt)
                x0 = torch.randn(
                    nb + 4, sph.npw, generator=gen, dtype=torch.float64
                ) + 1j * torch.randn(nb + 4, sph.npw, generator=gen, dtype=torch.float64)
                x0 = x0.to(dev) * torch.exp(-0.5 * HBAR2_2M * sph.kpg2 / system.ecut * 4.0)
                x0 = x0.to(CDTYPE)
            else:
                x0 = c_isp_ik
            e_k, c_k = davidson_gen(hs, x0, nb, tol=tol_eff)
            b = becp(projs[ik], c_k)
            snorm = (c_k.abs() ** 2).sum(dim=1).real + torch.einsum(
                "bi,ij,bj->b", b.conj(), system.q_full.to(CDTYPE), b
            ).real
            c_k = c_k / torch.sqrt(snorm)[:, None]
            coeffs[isp][ik] = c_k
            eigs_l.append(e_k)
        eigs_s.append(torch.stack(eigs_l))
    return eigs_s


class _IterStep(TypedDict):
    """One `_scf_iteration` evaluation's result -- each key its own real type
    (not the flattened value-union a plain `dict[str, ...]` return would give
    every key), mirroring noncollinear.py's `_WorkingTables` TypedDict."""

    eigs_s: list[torch.Tensor]
    occ_s: list[torch.Tensor]
    mu: float
    n_hub_s: list[list[torch.Tensor]] | None
    rho_out_s: list[torch.Tensor]
    rho_ij_s: list[list[torch.Tensor]]
    becps_s: list[list[torch.Tensor]]
    energies: EnergyBreakdown
    # Fresh smooth kinetic-energy density τ̃ built from THIS iteration's
    # orbitals — the driver feeds it back as the next iteration's lagging τ_s
    # (the τ analogue of the density/becsum warm state). None for GGA/LDA.
    tau_out_s: list[torch.Tensor] | None
    # Full-mesh (gathered) eigenvalues/occupations under a distributed
    # (dist_ctx) run -- identical to eigs_s/occ_s (same object) when
    # ops.dist_ctx is None. scf_uspp substitutes these into the final
    # USPPResult so it looks like an ordinary full-mesh run; eigs_s/occ_s
    # above stay THIS RANK's local shard, the shape every other per-k
    # collective point in this file (becps_s, coeffs) already keeps local.
    eigs_global_s: list[torch.Tensor]
    occ_global_s: list[torch.Tensor]


@torch.no_grad()
def _scf_iteration(
    ops: _IterOps,
    rho_s: list[torch.Tensor],
    rho_ij_mix: list[list[torch.Tensor]],
    coeffs: list[list[torch.Tensor | None]],
    coeffs_b: list[torch.Tensor | None],
    n_hub_s: list[list[torch.Tensor]] | None,
    tol_eff: float,
    seed_salt: int,
    it: int | None = None,
    tau_s: list[torch.Tensor] | None = None,
) -> _IterStep:
    """ONE evaluation of the SCF map at (rho_s, rho_ij_mix): potentials →
    screened D (+ one-center ddd from the MIXER-side becsum) → generalized
    Davidson (warm-started via coeffs/coeffs_b, mutated in place) →
    shared-Fermi occupations (+U matrices) → fresh densities/becsum →
    energy assembly. No mixing, no convergence judgment, no rescue —
    those belong to the driver (or to newton/the rig, which call this
    directly).

    ``it`` (the 1-based SCF iteration) drives the DFT+U U-ramp; when None (the
    newton/rig direct callers) or ``ops.hub_u_ramp_iters<=0`` the ramp factor is
    1.0 (full U).

    ``tau_s`` (meta-GGA only) is the LAGGING smooth τ̃ from the previous
    iteration's orbitals — it dresses the smooth v_xc (multiplicative ∂e/∂ρ|_τ)
    and the generalized-KS operator −½∇·(v_τ∇ψ̃) in H. The FRESH τ̃ built from
    this iteration's orbitals is used for the energy and returned as
    ``tau_out_s`` for the driver to feed back next step (τ is a functional of
    the orbitals, never mixed — same lag as the density/becsum/Hubbard state)."""
    system, xc, nspin = ops.system, ops.xc, ops.nspin
    smearing, width, hub = ops.smearing, ops.width, ops.hub
    vloc_r, phase_pos = ops.vloc_r, ops.phase_pos
    is_paw, onec, dev = ops.is_paw, ops.onec, ops.dev
    # DFT+U U-ramp factor for this iteration; 1.0 (no ramp) when it is None.
    u_scale = 1.0 if it is None else hubbard_u_ramp_scale(it, ops.hub_u_ramp_iters)
    veff_s, dscr_s, e_onec = uspp_potentials_dscr(
        system, xc, rho_s, rho_ij_mix, vloc_r, phase_pos, onec if is_paw else None,
        boundary=ops.boundary, esm_bias=ops.esm_bias, tau_s=tau_s,
    )
    # meta-GGA generalized-KS τ operator for the smooth orbitals, from the same
    # lagging τ̃ that dressed v_xc; None for GGA/LDA or the cold-start iter 1.
    metagga_v_s = _uspp_vtau_fields(ops, rho_s, tau_s)

    eigs_s = _solve_bands_uspp(
        ops, veff_s, dscr_s, n_hub_s, coeffs, coeffs_b, tol_eff, seed_salt, u_scale,
        metagga_v_s=metagga_v_s,
    )

    dist_ctx = ops.dist_ctx
    if dist_ctx is not None:
        from gradwave.distributed import gather_cat

        # A metal's Fermi level depends on eigenvalues from EVERY k-point,
        # not just this rank's shard: gather them into the global array
        # before the Fermi search (mirrors scf/loop.py's NC driver), then
        # keep only this rank's slice of the resulting occupations for the
        # local density/becsum/energy calls below.
        eigs_global_s = [gather_cat(e, dist_ctx) for e in eigs_s]
    else:
        eigs_global_s = eigs_s

    kweights_g = ops.kweights_global
    assert kweights_g is not None  # always set by _build_iter_ops
    occ_global_s, mu, entropy_term = shared_fermi_occupations(
        eigs_global_s, kweights_g, smearing, width, system.n_electrons, nspin, dev
    )
    occ_s = (
        [occ_global_s[sp][dist_ctx.k_start : dist_ctx.k_end] for sp in range(nspin)]
        if dist_ctx is not None
        else occ_global_s
    )

    # _solve_bands_uspp mutates coeffs[isp][ik] in place for every (isp, ik)
    # (both its batched and per-k branches loop over the full nk/nspin range
    # unconditionally), so every element is now a real Tensor -- narrower than
    # the cold-start-compatible `Tensor | None` these downstream helpers'
    # coeffs parameter carries for the seed/warm-start path.
    coeffs_full = cast("list[list[torch.Tensor]]", coeffs)

    # DFT+U: fresh S-metric occupation matrices + Dudarev E_U (lags one
    # step into V_U like the NC path; nspin=1 splits [0,2] occupations
    # into two equal channels)
    n_hub_s, e_hub = _hubbard_occ_update(ops, hub, coeffs_full, coeffs_b, occ_s, n_hub_s, u_scale)

    rho_out_s, rho_ij_s, becps_s = _build_output_density(ops, coeffs_full, coeffs_b, occ_s)
    rho_tot_out = rho_out_s[0] if nspin == 1 else rho_out_s[0] + rho_out_s[1]

    # Fresh smooth τ̃ from this iteration's orbitals (same occupations that made
    # rho_out), for a consistent meta-GGA energy and as the lagging τ_s the
    # driver feeds back next step. all_reduce completes the k-sum under dist_ctx,
    # mirroring the density/becsum reduction in _build_output_density.
    tau_out_s: list[torch.Tensor] | None = None
    if xc.needs_tau:
        from gradwave.core.metagga import tau_b

        # coeffs_b[isp] is the padded (nk, nb, npw_max) block the batched solver
        # filled in place; needs_tau is gated to the batched solver in scf_uspp,
        # which builds bk (the `if batched or hubbard` guard in _build_iter_ops).
        bk = ops.bk
        assert bk is not None
        tau_out_s = []
        for isp in range(nspin):
            cb = coeffs_b[isp]
            assert cb is not None
            tau_out_s.append(
                tau_b(cb, occ_s[isp], system.kweights, bk, ops.shape, ops.vol)
            )
        if dist_ctx is not None:
            from gradwave.distributed import all_reduce_

            tau_out_s = [all_reduce_(t, dist_ctx) for t in tau_out_s]

    energies = _assemble_iter_energies(
        ops, coeffs_full, occ_s, becps_s, rho_out_s, rho_tot_out, entropy_term, e_hub,
        e_onec, tau_out_s,
    )
    if dist_ctx is not None:
        from gradwave.distributed import all_reduce_

        # Kinetic and nonlocal (projector) energy are sums over k, computed
        # above from this rank's local shard only. Every other term
        # (Hartree, XC, local pseudopotential, Ewald, entropy, hubbard,
        # onecenter) is a function of the ALREADY-global density/becsum/
        # occupation-matrix, so it comes out identical on every rank without
        # further communication (see _hubbard_occ_update/_build_output_density).
        energies.kinetic = all_reduce_(energies.kinetic, dist_ctx)
        energies.nonlocal_ = all_reduce_(energies.nonlocal_, dist_ctx)
    return _IterStep(
        eigs_s=eigs_s,
        occ_s=occ_s,
        mu=mu,
        n_hub_s=n_hub_s,
        rho_out_s=rho_out_s,
        rho_ij_s=rho_ij_s,
        becps_s=becps_s,
        energies=energies,
        tau_out_s=tau_out_s,
        eigs_global_s=eigs_global_s,
        occ_global_s=occ_global_s,
    )


def _resolve_uspp_mixing_scheme(mixing_scheme: str | None, nspin: int) -> str:
    """PAW/USPP mixing-scheme default. None → johnson for both nspin==1 and
    nspin==2; an explicit scheme always wins.

    The composite (density, becsum) mixing vector carries the on-site
    augmentation-charge mode, whose response is stiff even in a gapped
    insulator, so Kerker-preconditioned Johnson beats plain Pulay across
    non-magnetic PAW insulators AND metals (measured, same fixed point:
    Si 18→12, Cu 19→13, Pt 21→12, 8-atom Si 20→13).

    nspin==2 was on pulay while the bcc Fe johnson blowup (29→93) stood, on the
    theory that johnson discarded the becsum step-damping a robust ferromagnet
    leans on. That blowup no longer reproduces after the #199 Stoner
    spin_precond change, and a re-benchmark (experiments/uspp_mixing_default)
    puts johnson ahead on every magnetic PAW case at the same fixed point:
    bcc Fe 30→16, fcc Ni across three seeds 19/25/27→13/15/18, and a
    two-sublattice AFM Fe 58→31. So the whole USPP/PAW path now defaults to
    johnson, matching the norm-conserving path's magnetic default rather than
    mirror-inverting it."""
    if mixing_scheme is not None:
        return mixing_scheme
    return "johnson"


def _build_mixer(
    scheme: str,
    g2_full: torch.Tensor,
    *,
    alpha,
    history,
    kerker,
    kerker_mask,
    step_scale,
    metric_w,
    w0,
    adapt_ids,
) -> PulayMixer | BroydenMixer | JohnsonMixer:
    """Construct the charge mixer for the requested scheme over the composite
    (density + becsum) vector. Kept out of scf_uspp so the scheme dispatch is
    one place."""
    if scheme not in ("pulay", "broyden", "johnson"):
        raise ValueError("mixing_scheme must be 'pulay', 'broyden', or 'johnson'")
    if scheme == "broyden":
        return BroydenMixer(
            g2_full,
            alpha=alpha,
            history=history,
            kerker=kerker,
            kerker_mask=kerker_mask,
            check_g0=False,
            step_scale=step_scale,
        )
    if scheme == "johnson":
        return JohnsonMixer(
            g2_full,
            alpha=alpha,
            history=history,
            kerker=kerker,
            kerker_mask=kerker_mask,
            check_g0=False,
            step_scale=step_scale,
            metric_w=metric_w,
            w0=w0,
        )
    return PulayMixer(
        g2_full,
        alpha=alpha,
        history=history,
        kerker=kerker,
        kerker_mask=kerker_mask,
        check_g0=False,
        step_scale=step_scale,
        adapt_blocks=adapt_ids,
    )


def _seed_becsum(
    system: USPPSystem,
    nspin: int,
    start_from: _USPPStartFrom,
    # _seed_scf_density (the sole caller's source) returns spin_frac as either
    # ALL-real (nspin=2, `[up, dn]`) or ALL-None (nspin=1, `[None]`) -- never a
    # per-element mix -- so this matches that shape (`list[X] | list[None]`,
    # not the per-element-Optional `list[X | None]`, which `list`'s
    # invariance would make a distinct, non-assignable type from either).
    spin_frac: list[list[float]] | list[None],
    dev: torch.device,
) -> list[list[torch.Tensor]]:
    """Per-spin becsum seed: from a warm-start state, else the reference atomic
    PAW occupations (spin-split by start_mag); zeros for bare USPP without
    PP_OCCUPATIONS."""
    rho_ij_s: list[list[torch.Tensor]] = [[] for _ in range(nspin)]
    if start_from is not None:
        # rho_ij_atoms is a flat list[Tensor] per atom for nspin=1, else a
        # list[Tensor] per spin -- _USPPStartFrom's dict variant (and
        # _DictBridge's untyped getattr passthrough for a USPPResult) can't
        # express that per-nspin shape difference (a single static type can't
        # be both `list[Tensor]` and `list[list[Tensor]]` and still let `m in
        # src` type as Tensor in both branches), so this one is cast to Any
        # rather than forcing a union that would just push the ambiguity onto
        # `m.detach()` below.
        prev_bec = cast(Any, start_from["rho_ij_atoms"])
        for isp in range(nspin):
            src = prev_bec if nspin == 1 else prev_bec[isp]
            rho_ij_s[isp] = [m.detach().to(device=dev, dtype=CDTYPE).clone() for m in src]
        return rho_ij_s
    for a, sp in enumerate(system.species_of_atom):
        paw = system.paws[sp]
        nm = sum(2 * b.l + 1 for b in paw.betas)
        for isp in range(nspin):
            m0 = torch.zeros(nm, nm, dtype=CDTYPE, device=dev)
            if paw.paw_occ is not None:
                if nspin == 1:
                    frac = 0.5
                else:
                    # spin_frac is [None] only for nspin=1 (the branch above);
                    # nspin=2 always seeds it with two real per-atom lists
                    # (_seed_scf_density's `[[up...], [dn...]]` return).
                    sf_isp = spin_frac[isp]
                    assert sf_isp is not None
                    frac = sf_isp[a]
                col = 0
                for i, b in enumerate(paw.betas):
                    for _m in range(2 * b.l + 1):
                        m0[col, col] = (
                            paw.paw_occ[i] / (2 * b.l + 1) * (2.0 * frac if nspin == 1 else frac)
                        )
                        col += 1
            rho_ij_s[isp].append(m0)
    return rho_ij_s


def _seed_orbitals_uspp(
    system: USPPSystem,
    nspin: int,
    nk: int,
    nb: int,
    batched: bool,
    start_from: _USPPStartFrom,
    dev: torch.device,
) -> tuple[list[list[torch.Tensor | None]], list[torch.Tensor | None]]:
    """Seed the generalized Davidson from a previous result's eigenvectors — the
    ionic-step analogue of the density/becsum warm start — instead of the random
    atomic-scale start. Returns ``(coeffs, coeffs_b)``: per-spin, per-k blocks
    ``[(nb, npw_k)]`` and, for the batched solver, a padded ``(nk, nb, npw_max)``
    block; both None-filled (cold start) when ``start_from`` carries no orbitals.

    Reuse requires the previous orbitals to live on THIS system's G-spheres: same
    plane-wave count per k and matching fractional k (``scf_uspp`` already pins
    start_from to the same FFT grid via the density seed, so a same-grid ionic
    move qualifies directly; the calculator remaps orbitals onto the new spheres
    for grid-shape changes before passing them here). Anything else — a k-set
    reshaped by a symmetry-group change, a band-count change — bails to the cold
    start, keeping the density warm start intact."""
    coeffs: list[list[torch.Tensor | None]] = [[None] * nk for _ in range(nspin)]
    coeffs_b: list[torch.Tensor | None] = [None] * nspin
    prev_c = None
    if start_from is not None:
        # _USPPStartFrom's dict variant types "coeffs" as (at most) a flat
        # list[Tensor], but the nspin=2 shape is really a list-of-lists
        # [spin][k] (same _StartFrom-shaped debt as scf/loop.py's own
        # _seed_orbitals -- cast past it there too, same reasoning).
        prev_c = cast(
            "list[Any] | None",
            start_from.get("coeffs")
            if isinstance(start_from, dict)
            else getattr(start_from, "coeffs", None),
        )
    if prev_c is None:
        return coeffs, coeffs_b
    prev_sys = (
        start_from.get("system")
        if isinstance(start_from, dict)
        else getattr(start_from, "system", None)
    )
    prev_sph = getattr(prev_sys, "spheres", None)
    chans = [prev_c] if nspin == 1 else list(prev_c)
    compat = len(chans) == nspin and all(
        ch is not None
        and len(ch) == nk
        and all(
            ch[ik] is not None
            and ch[ik].shape[0] >= nb
            and ch[ik].shape[1] == system.spheres[ik].npw
            and (
                prev_sph is None
                or np.allclose(
                    np.asarray(prev_sph[ik].k_frac, dtype=float),
                    np.asarray(system.spheres[ik].k_frac, dtype=float),
                )
            )
            for ik in range(nk)
        )
        for ch in chans
    )
    if not compat:
        return coeffs, coeffs_b
    npw_max = max(s.npw for s in system.spheres)
    for isp, ch in enumerate(chans):
        for ik in range(nk):
            coeffs[isp][ik] = ch[ik][:nb].detach().to(device=dev, dtype=CDTYPE).clone()
        if batched:
            xb = torch.zeros(nk, nb, npw_max, dtype=CDTYPE, device=dev)
            for ik in range(nk):
                ck = coeffs[isp][ik]
                assert ck is not None  # just assigned two lines above
                xb[ik, :, : system.spheres[ik].npw] = ck
            coeffs_b[isp] = xb
    return coeffs, coeffs_b


def _seed_scf_density(
    system: USPPSystem,
    grid: FFTGrid,
    vol: float,
    dev: torch.device,
    nspin: int,
    start_from: _USPPStartFrom,
    start_mag: list[float] | None,
) -> tuple[list[torch.Tensor], list[list[float]]] | tuple[list[torch.Tensor], list[None]]:
    """Seed (rho_s, spin_frac): warm-start rescale from a prior result, or a SAD
    density (nspin=1), or per-atom spin-split SAD (nspin=2). spin_frac carries the
    per-atom up/down fractions (or [None]) used later to seed becsum."""
    if start_from is not None:
        # shared grid/nspin validation + volume-ratio rescale (electron count
        # exactly conserved on the new cell) — common.warm_start_densities
        rho_s = warm_start_densities(start_from, nspin, grid, vol, dev)
        if nspin == 1:
            return rho_s, [None]
        mags = _resolve_start_mag(start_mag, system.species_of_atom, len(system.paws))
        return rho_s, [[(1.0 + m) / 2.0 for m in mags], [(1.0 - m) / 2.0 for m in mags]]
    if nspin == 1:
        return [
            sad_density(
                grid, system.positions, system.species_of_atom, system.paws, system.n_electrons
            )
        ], [None]
    # per-ATOM moment fractions (AFM/ferrimagnetic seeds pass one entry per atom;
    # a per-species list is broadcast). sad_density's atom_scale seeds each atom
    # directly — identical to the old species_scale path when the moments are
    # uniform within a species.
    mags = _resolve_start_mag(start_mag, system.species_of_atom, len(system.paws))
    up = [(1.0 + m) / 2.0 for m in mags]
    dn = [(1.0 - m) / 2.0 for m in mags]
    n_up = sum(float(system.charges[a]) * up[a] for a in range(len(system.species_of_atom)))
    rho_s = [
        sad_density(
            grid, system.positions, system.species_of_atom, system.paws, n_up, atom_scale=up
        ),
        sad_density(
            grid,
            system.positions,
            system.species_of_atom,
            system.paws,
            system.n_electrons - n_up,
            atom_scale=dn,
        ),
    ]
    return rho_s, [up, dn]


@torch.no_grad()
def scf_uspp(
    system: USPPSystem,
    xc: XCFunctional | SpinXC,
    *,
    nspin: int = 1,
    start_mag: list[float] | None = None,
    smearing: str = "none",
    width: float = 0.1,
    max_iter: int = 60,
    etol: float = 1e-8,
    rhotol: float = 1e-7,
    diago_tol: float = 1e-9,
    mixing_alpha: float = 0.7,
    mixing_history: int | None = None,
    trust_factor: float = 20.0,
    batched: bool = True,
    hubbard: list[HubbardManifold] | None = None,
    hub_occ_mix: float = 1.0,  # DFT+U occupation-matrix damping β in (0,1] (1.0 = raw one-step
    # lag, bit-for-bit). Mirrors scf.loop.scf; a +U knob tied to `hubbard`, not part of SCFOptions
    hub_u_ramp_iters: int = 0,  # DFT+U linear U-ramp length (0 = off); convergence is blocked until
    # the ramp completes so the reported final energy is at full U. Mirrors scf.loop.scf
    start_from: _USPPStartFrom = None,
    criterion: str = "drho",
    energy_metric: bool = False,  # opt-in energy-metric gate (converge on
    # 1/2<r|K_Hxc|r> < entol); overrides `criterion`. See scf.loop.scf / docs.
    entol: float = 1e-6,  # eV, energy-error threshold for energy_metric
    rho_safety: float = 1e-2,
    adapt_step: bool = False,
    mixing_scheme: str | None = None,
    mixing_kerker: bool | None = None,
    mixing_metric: str = "plain",
    spin_precond: bool = False,
    mixed_precision: bool = False,
    precond: str = "kerker",
    opts: SCFOptions | None = None,
    verbose: bool = True,
    dist_ctx: "DistKContext | None" = None,  # k-point-sharded distributed SCF (see
    # gradwave.distributed): `system` is THIS RANK's local k-shard (built by
    # gradwave.distributed.shard_uspp_system); None (default) runs the ordinary
    # single-process path, byte-for-byte unchanged. DFT+U (`hubbard`) IS
    # supported under dist_ctx (unlike the NC driver's dist_ctx, which rejects
    # it) -- see _hubbard_occ_update's all_reduce over the occupation matrices.
    recorder: "SCFRecorder | None" = None,  # per-iteration flight recorder (scf.recorder);
    # None (default) builds a fresh cheap-path recorder. NOT part of SCFOptions — an
    # internal diagnostics object, so it is exempt from the opts-vs-flat-kwarg guard below.
    boundary: str = "periodic",  # periodic | open_z | open_z_metal — open-boundary
    # (ESM) electrostatics (slab, c ⊥ a,b); adds the differentiable correction on the
    # FULL (smooth + augmentation) density to v_eff and the total energy.
    esm_bias: float = 0.0,  # applied capacitor bias [V] for boundary="open_z_metal"
) -> USPPResult:
    """USPP/PAW SCF. nspin=2 takes a SpinXC functional and start_mag (list,
    in [-1, 1]) with one entry per species OR one per atom (the latter for
    AFM/ferrimagnetic seeds; a length matching neither raises); mixing then
    runs in the (total, magnetization) basis with Kerker on the total for
    smeared systems.
    batched=True solves all k in one padded generalized-Davidson block
    (identical eigenpairs; batched=False is the reference per-k path).
    hubbard: list[HubbardManifold] — Dudarev DFT+U with S-metric occupation
    matrices (QE U_projection_type='atomic' convention for USPP).
    start_from: a previous scf_uspp result on the SAME FFT grid and spin
    count — seeds (ρ, becsum) from its converged state instead of SAD +
    atomic occupations. The density is rescaled by the volume ratio so the
    electron count is conserved. This is the right start for scans (EOS
    volumes, displacements): adjacent points are small perturbations, so
    the warm start both cuts iterations and keeps trajectory-dependent
    branches (FM vs NM) from flipping between points.
    criterion: "drho" (default) demands both etol and rhotol; "energy"
    converges on a settled 3-iteration free-energy tail (< etol) with only
    the loose rho_safety residual bound — the honest criterion for smeared
    metals, whose residual floors at occupation noise (O(res²) energy
    error) while F is long converged.
    adapt_step: OPT-IN per-block adaptive damping. Blocks whose residual
    grows across iterations get their damped step cut by the observed
    gain, with a plateau-triggered global halving on top. On FM Ni at the
    default mixing_alpha this prevents the silent collapse to the NM
    branch (m and F land at the validated values), but it does NOT reach
    tight convergence — the monotone multipliers over-react to transient
    startup growth (measured head-to-head: static alpha=0.3 converges to
    |dρ| 2e-3 where adaptive stalls at 2e-2 with the ρ-block floored).
    Use it as a stabilizer for exploratory runs at unknown damping; for
    production FM metals keep hand-set mixing_alpha (0.3 for Ni).
    mixing_scheme: None (default) resolves to "johnson" for both nspin==1
    and nspin==2 (Kerker-preconditioned Broyden; the augmentation-charge
    mode in the composite (density, becsum) vector makes it beat pulay
    across PAW insulators and metals alike, and it also wins on every
    magnetic PAW case at the same fixed point now that the bcc Fe blowup no
    longer reproduces, see experiments/uspp_mixing_default).
    Override with "pulay", "broyden", or "johnson"; broyden is limited-memory
    Broyden-II, whose sequential secant updates keep directional gain
    estimates that Pulay's residual-span extrapolation loses (the QE
    default scheme; candidate for hand-set damping on FM metals).
    spin_precond: Stoner preconditioner on the magnetization channel
    (smeared nspin=2 only; scf/spin_precond.py) — the physics-informed
    treatment of the Stoner-expansive mode, applied to residuals before
    damping/mixing.
    mixed_precision: fp32 draft in the batched generalized Davidson while
    the adaptive diago tolerance is above 1e-5; the subspace reduction and
    S-normalization stay fp64 and the final iterations re-polish in fp64,
    so converged results are unchanged. Batched path only (the per-k
    reference path ignores it). The payoff is on consumer GPUs (fp64 at
    1/64 of fp32 throughput); CPU gains are modest.
    opts: an SCFOptions object (scf/options.py) — the readable form of
    all of the above. When given, the flat config kwargs must be left at
    their defaults; supplying both opts and a non-default flat kwarg raises
    ValueError (rather than silently overwriting it)."""
    mixing_w0, bec_step_scale = 0.01, None  # MixerOptions defaults
    if opts is not None:
        # opts is the readable form of every flat config kwarg; passing both is
        # ambiguous, so reject any flat kwarg left non-default alongside opts
        # rather than silently overwriting it.
        _flat_defaults = {
            "smearing": "none",
            "width": 0.1,
            "max_iter": 60,
            "etol": 1e-8,
            "rhotol": 1e-7,
            "diago_tol": 1e-9,
            "mixing_alpha": 0.7,
            "mixing_history": None,
            "trust_factor": 20.0,
            "batched": True,
            "criterion": "drho",
            "energy_metric": False,
            "entol": 1e-6,
            "rho_safety": 1e-2,
            "adapt_step": False,
            "mixing_scheme": None,
            "mixing_kerker": None,
            "mixing_metric": "plain",
            "spin_precond": False,
            "mixed_precision": False,
            "precond": "kerker",
            "verbose": True,
        }
        _supplied = {
            k
            for k, v in (
                ("smearing", smearing),
                ("width", width),
                ("max_iter", max_iter),
                ("etol", etol),
                ("rhotol", rhotol),
                ("diago_tol", diago_tol),
                ("mixing_alpha", mixing_alpha),
                ("mixing_history", mixing_history),
                ("trust_factor", trust_factor),
                ("batched", batched),
                ("criterion", criterion),
                ("energy_metric", energy_metric),
                ("entol", entol),
                ("rho_safety", rho_safety),
                ("adapt_step", adapt_step),
                ("mixing_scheme", mixing_scheme),
                ("mixing_kerker", mixing_kerker),
                ("mixing_metric", mixing_metric),
                ("spin_precond", spin_precond),
                ("mixed_precision", mixed_precision),
                ("precond", precond),
                ("verbose", verbose),
            )
            if v != _flat_defaults[k]
        }
        if _supplied:
            raise ValueError(
                "scf_uspp: configure through `opts` OR the flat keyword "
                "arguments, not both (conflicting flat kwargs: "
                f"{sorted(_supplied)})"
            )
        smearing, width = opts.smearing, opts.width
        max_iter, etol, rhotol = opts.max_iter, opts.etol, opts.rhotol
        diago_tol, criterion = opts.diago_tol, opts.criterion
        energy_metric, entol = opts.energy_metric, opts.entol
        rho_safety, batched, verbose = opts.rho_safety, opts.batched, opts.verbose
        mixed_precision = opts.mixed_precision
        mx = opts.mixer
        mixing_alpha, mixing_history = mx.alpha, mx.history
        mixing_scheme, mixing_kerker = mx.scheme, mx.kerker
        mixing_metric, trust_factor = mx.metric, mx.trust_factor
        adapt_step, spin_precond = mx.adapt_step, mx.spin_precond
        mixing_w0, bec_step_scale = mx.w0, mx.bec_step_scale
        precond = mx.precond
    mixing_scheme = _resolve_uspp_mixing_scheme(mixing_scheme, nspin)
    if criterion not in ("drho", "energy"):
        raise ValueError("criterion must be 'drho' or 'energy'")
    if mixing_metric not in ("plain", "coulomb"):
        raise ValueError("mixing_metric must be 'plain' or 'coulomb'")
    if hubbard:
        validate_hubbard_conv(hub_occ_mix, hub_u_ramp_iters)
    if hasattr(system.rho_symmetrizer, "apply_m"):
        raise ValueError(
            "system was built with magnetic symmetry (magmoms=...) — "
            "only scf_uspp_noncollinear consumes it (anti-unitary ops "
            "would mis-fold collinear spin channels); rebuild without "
            "magmoms"
        )
    # IBZ symmetry needs no special handling under dist_ctx: rho_symmetrizer
    # and becsum_sym are k-set-independent and _build_output_density applies
    # them AFTER the cross-rank all_reduce, so every rank symmetrizes the
    # identical global density/becsum (see gradwave.distributed's docstring).
    if dist_ctx is not None and start_from is not None:
        # A relax/EOS warm start hands the previous, reassembled FULL-mesh
        # result here; `system` is only this rank's k-shard, so slice the per-k
        # orbital seed down to [k_start, k_end) or _seed_orbitals_uspp's k-count
        # check silently cold-starts every orbital (see shard_start_from). The
        # per-atom becsum (rho_ij_atoms) it also carries stays global.
        from gradwave.distributed import shard_start_from

        start_from = shard_start_from(start_from, dist_ctx)
    grid = system.grid
    vol = grid.volume
    dev = system.positions.device
    nk = len(system.spheres)

    rho_s, spin_frac = _seed_scf_density(system, grid, vol, dev, nspin, start_from, start_mag)

    ops = _build_iter_ops(
        system,
        xc,
        nspin=nspin,
        smearing=smearing,
        width=width,
        batched=batched,
        hubbard=hubbard,
        mixed_precision=mixed_precision,
        dist_ctx=dist_ctx,
        hub_occ_mix=hub_occ_mix,
        hub_u_ramp_iters=hub_u_ramp_iters,
        boundary=boundary,
        esm_bias=esm_bias,
    )
    if xc.needs_tau:
        # meta-GGA on the USPP/PAW path: the smooth τ̃ enters H exactly as on
        # the NC path, but the AE/PS τ correction lives in the PAW one-center
        # spheres. Bare ultrasoft has no one-center sphere to carry it (QE
        # likewise supports meta-GGA only with PAW or norm-conserving), and the
        # τ generalized-KS operator is wired only into the batched solver.
        if not ops.is_paw:
            raise NotImplementedError(
                "meta-GGA requires PAW; bare ultrasoft has no one-center τ "
                "augmentation (use a PAW dataset or a norm-conserving pseudo)"
            )
        if not batched:
            raise NotImplementedError(
                "meta-GGA on the USPP/PAW path requires the batched solver "
                "(batched=True)"
            )
    hub = ops.hub
    n_hub_s = None
    if hub is not None:
        n_hub_s = [
            [torch.zeros(s["dim"], s["dim"], dtype=CDTYPE, device=dev) for s in hub.sites]
            for _ in range(nspin)
        ]

    # Mixing vector = [ρ channels on the density sphere, flattened becsum per
    # spin]. Mixing becsum TOGETHER with ρ (QE keeps it inside rho%mix the
    # same way) is essential for metals: the D-feedback loops (∫v_eff Q and
    # the one-center ddd) must see a becsum coherent with the mixed density —
    # a fresh or independently-damped becsum gives a gain>1 charge
    # oscillation for semicore-metal PAW (fcc Ni diverges with ×9/iteration).
    # Kerker damps the ρ-TOTAL block only (becsum is localized; the
    # magnetization channel must keep its G=0 free for ↑↓ transfer).
    # MixLayout owns the composite-vector structure: packing, Kerker mask
    # (ρ-total block only — becsum is localized and the magnetization
    # channel keeps its G=0 free for ↑↓ transfer), the becsum step scale
    # (the on-site becsum↔ddd feedback is the stiffest direction), and the
    # adaptive-damping block ids
    if bec_step_scale is None:
        # Johnson handles the on-site becsum↔ddd mode without extra
        # damping (FM Ni 27→16 it); the 0.4 stays for pulay/broyden
        bec_step_scale = 1.0 if mixing_scheme == "johnson" else 0.4
    layout = MixLayout(grid, nspin, system.atom_slices, device=dev, bec_step_scale=bec_step_scale)
    g2_mix, ng, nbec = layout.g2_sphere, layout.ng, layout.nbec
    g2_full, kerker_mask = layout.g2_full, layout.kerker_mask
    step_scale = layout.step_scale
    adapt_ids = layout.block_ids if adapt_step else None
    use_kerker = (smearing != "none") if mixing_kerker is None else bool(mixing_kerker)
    if mixing_history is None:
        # per-scheme defaults, measured on FM Ni (see JohnsonMixer)
        mixing_history = 12 if mixing_scheme == "johnson" else 8
    metric_w = None
    if mixing_metric == "coulomb":
        # QE rho_ddot: Coulomb-metric inner products (long-range emphasis).
        # G=0 is EXCLUDED (zero weight — 1/G² there poisons the Gram
        # matrix); becsum components get unit weight
        wg = torch.where(g2_mix > 1e-12, 1.0 / g2_mix.clamp_min(1e-12), torch.zeros_like(g2_mix))
        metric_w = torch.cat([wg] * nspin + [torch.ones(nbec, device=dev)] * nspin)
    mixer = _build_mixer(
        mixing_scheme,
        g2_full,
        alpha=mixing_alpha,
        history=mixing_history,
        kerker=use_kerker,
        kerker_mask=kerker_mask,
        step_scale=step_scale,
        metric_w=metric_w,
        w0=mixing_w0,
        adapt_ids=adapt_ids,
    )
    if precond not in ("kerker", "local_tf"):
        raise ValueError("precond must be 'kerker' or 'local_tf'")
    tf_precond = None
    if precond == "local_tf":
        # position-dependent TF screening on the ρ-total block of the composite
        # mixing vector (the first `ng` entries; becsum and the magnetization
        # channel keep plain damping, matching the kerker_mask). set_density()
        # is called with the current smooth density each iteration.
        from gradwave.scf.local_tf import LocalTFPrecond

        tf_precond = LocalTFPrecond(grid.g2, grid.shape, layout.mask, q0_max=mixer.q0)
        mixer.precond_op = tf_precond
        mixer.precond_slice = slice(0, ng)
    # warm-start the eigensolver from a previous result's orbitals when they fit
    # this grid's G-spheres (same-grid ionic move, or calculator-remapped for a
    # grid-shape change); else a cold random start. coeffs/coeffs_b are then
    # mutated in place by _scf_iteration each cycle.
    coeffs, coeffs_b = _seed_orbitals_uspp(system, nspin, nk, ops.nb, batched, start_from, dev)
    e_free_prev, history, converged = None, [], False
    # meta-GGA lagging smooth τ̃ (None ⇒ GGA/LDA). Iteration 1 uses a Thomas-Fermi
    # seed (no orbitals yet); from iter 2 on it is the true ½Σf|∇ψ̃|² rebuilt from
    # each iteration's fresh orbitals, never mixed, consistent at the fixed point.
    tau_state: list[torch.Tensor] | None = _bootstrap_tau_uspp(xc, rho_s, nspin)
    rescue_count, seed_salt = 0, 0  # solver-blowup rescue state (task #55)
    last_reset_it = -10  # trust-region reset cooldown (task #55)
    occ_s = eigs_s = mu = None
    energies: EnergyBreakdown | None = None
    # Bound before the loop purely so ty can see these names as always
    # defined after it (the loop runs `for it in range(1, max_iter + 1)`, and
    # max_iter is always >= 1 in practice); overwritten on the loop's first
    # pass every real invocation.
    it = 0
    e_free = 0.0
    res_norm = float("nan")
    rho_out_s: list[torch.Tensor] = []
    becps_s: list[list[torch.Tensor]] = []
    # Full-mesh (gathered) eigenvalues/occupations under dist_ctx -- tracked
    # separately from the LOCAL eigs_s/occ_s above (same "runs >=1 iteration"
    # binding as it/e_free/etc.) and substituted into the final USPPResult
    # below so it looks like an ordinary full-mesh run; equal to eigs_s/occ_s
    # (same object) whenever dist_ctx is None.
    eigs_global_s: list[torch.Tensor] = []
    occ_global_s: list[torch.Tensor] = []

    # PAW one-center machinery; becsum seeded from the reference atomic
    # occupations (spin-split by start_mag; zeros for bare USPP where the UPF
    # carries no PP_OCCUPATIONS). rho_ij_mix is the MIXER-side becsum used
    # for the one-center ddd; rho_ij_s holds each iteration's fresh becsum.
    is_paw, onec = ops.is_paw, ops.onec  # reuse the OneCenter list built in _build_iter_ops
    rho_ij_s = _seed_becsum(system, nspin, start_from, spin_frac, dev)
    rho_ij_mix = [[m.clone() for m in ch] for ch in rho_ij_s]

    # flight recorder (scf.recorder): cheap per-iteration diagnostics, always
    # collected in memory, detached and off the autograd graph. Defaults on.
    if recorder is None:
        from gradwave.scf.recorder import SCFRecorder

        seed_mom = (
            float((rho_s[0] - rho_s[1]).abs().sum()) * vol / grid.n_points if nspin == 2 else 0.0
        )
        recorder = SCFRecorder(grid.g2, nspin=nspin, seed_moment=seed_mom)
    recorder.set_mixing(
        scheme=mixing_scheme, alpha=mixing_alpha, kerker=bool(use_kerker), history=mixing_history
    )

    for it in range(1, max_iter + 1):
        t_it = time.perf_counter()
        # quadratic schedule (common.adaptive_diago_tol). SAD starts don't
        # deserve a tight first solve; warm starts do (their density is
        # already near a fixed point — scan/rig callers control precision
        # through diago_tol)
        tol_eff = adaptive_diago_tol(
            it,
            history,
            diago_tol,
            system.n_electrons,
            schedule="quadratic",
            first_tol=1e-3 if start_from is None else diago_tol,
        )

        step = _scf_iteration(
            ops, rho_s, rho_ij_mix, coeffs, coeffs_b, n_hub_s, tol_eff, seed_salt, it,
            tau_s=tau_state,
        )
        tau_state = step["tau_out_s"]  # lag the fresh τ̃ into the next step's H
        eigs_s, occ_s, mu = step["eigs_s"], step["occ_s"], step["mu"]
        eigs_global_s, occ_global_s = step["eigs_global_s"], step["occ_global_s"]
        n_hub_s = step["n_hub_s"]
        rho_out_s, rho_ij_s = step["rho_out_s"], step["rho_ij_s"]
        becps_s, energies = step["becps_s"], step["energies"]
        e_free = float(energies.free_energy)

        to_mix = layout.pack

        # solver-blowup rescue: a warm-started Davidson can deterministically
        # return states ~10² eV up from a CONVERGED-quality density (F falls
        # off a cliff, |Δρ| freezes, mixer resets change nothing — task #55
        # fingerprint). Detect the jump, throw away the poisoned warm starts,
        # and re-solve from salted fresh seeds WITHOUT feeding the garbage
        # into the mixer.
        if (
            e_free_prev is not None
            and history
            and abs(e_free - e_free_prev) > 5.0
            and history[-1]["res"] < 1e-2
            and rescue_count < 2
        ):
            rescue_count += 1
            seed_salt = 104729 * rescue_count
            logger.debug(
                "USPP iter %d solver blowup: dE=%.3e eV from residual %.1e "
                "(rescue %d/2), discarding warm starts and reseeding with "
                "salt=%d",
                it,
                abs(e_free - e_free_prev),
                history[-1]["res"],
                rescue_count,
                seed_salt,
            )
            coeffs_b = [None] * nspin
            # the lagging τ̃ was built from the poisoned orbitals; fall back to the
            # TF seed (r2SCAN needs a valid τ) then rebuild from clean orbitals
            tau_state = _bootstrap_tau_uspp(xc, rho_s, nspin)
            for isp in range(nspin):
                coeffs[isp] = [None] * nk
            if verbose:
                print(
                    f"  USPP {it:3d}  [solver blowup: dE = "
                    f"{abs(e_free - e_free_prev):.1f} eV from residual "
                    f"{history[-1]['res']:.1e} — reseeding eigensolver]"
                )
            continue

        rho_in_vec = to_mix(rho_s, rho_ij_mix)
        rho_out_vec = to_mix(rho_out_s, rho_ij_s)
        res_norm = (
            float(torch.linalg.norm(rho_out_vec[: ng * nspin] - rho_in_vec[: ng * nspin])) * vol
        )
        de = record_iteration(history, it, e_free, e_free_prev, res_norm, t_it)
        # flight recorder: cheap detached per-iteration metrics. The total
        # real-space residual is ρ_out − ρ_in; history[-1]["t"] is this
        # iteration's own measured wall time (no extra sync).
        rho_in_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
        rho_out_tot = rho_out_s[0] if nspin == 1 else rho_out_s[0] + rho_out_s[1]
        # energy-metric gate (opt-in): the residual's second-order energy error
        # 1/2<r|K_Hxc|r>, per-channel. Computed only when selected (one f_xc HVP
        # per iteration); the default path pays nothing. r is the SMOOTH-density
        # residual (the quantity res_norm gates on), read back from the mixing
        # vectors so the output augmentation does not contaminate it; the
        # augmentation/one-center PAW correction to K_Hxc is on-site and higher
        # order, omitted here, so this is the dominant Hartree+XC term.
        e_metric = e_metric_charge = e_metric_mag = None
        if energy_metric:
            from gradwave.postscf._response import kernel_energy_error

            sm_in, _ = layout.unpack(rho_in_vec)
            sm_out, _ = layout.unpack(rho_out_vec)
            r_s = [sm_out[sp] - sm_in[sp] for sp in range(nspin)]
            e_metric, e_metric_charge, e_metric_mag = kernel_energy_error(
                grid, xc, r_s, sm_in, system.rho_core, nspin
            )
        recorder.record(
            it=it,
            free_energy=e_free,
            dE=de,
            res_norm=res_norm,
            t_iter=float(history[-1]["t"]),
            drho_r=rho_out_tot - rho_in_tot,
            eigs=list(eigs_s),
            fermi=mu,
            entropy=0.0,
            mag_abs=(float((rho_out_s[0] - rho_out_s[1]).abs().sum()) * vol / grid.n_points)
            if nspin == 2
            else None,
            e_metric=e_metric,
            e_metric_charge=e_metric_charge,
            e_metric_mag=e_metric_mag,
        )
        if verbose:
            mag = ""
            if nspin == 2:
                m = float((rho_out_s[0] - rho_out_s[1]).sum()) * vol / grid.n_points
                mag = f"  m = {m:+.4f} muB"
            print(
                f"  USPP {it:3d}  F = {e_free:+.10f} eV  "
                f"dE = {(0.0 if de == float('inf') else de):.3e}  "
                f"|drho| = {res_norm:.3e}{mag}"
            )
        if energy_metric:
            # energy-metric gate (overrides `criterion`): the residual's exact
            # second-order energy error < entol, with etol and the stale-solve
            # guard kept (see scf.common.convergence_gate).
            done = convergence_gate(de, res_norm, tol_eff, etol, rhotol, diago_tol,
                                    energy_error=e_metric, entol=entol)
        elif criterion == "energy":
            # QE-style energy criterion: the free energy is variational, its
            # error is O(residual²), and for smeared metals the residual
            # floors at occupation noise long after F has settled. Require a
            # settled 3-iteration tail (a single small dE can be a sloshing
            # coincidence) plus a loose residual safety that excludes frozen
            # limit cycles without demanding the unreachable plateau floor.
            tail = [h["free_energy"] for h in history[-3:]]
            done = (
                len(tail) == 3
                and max(tail) - min(tail) < etol
                and res_norm < rho_safety
                and tol_eff <= diago_tol * 1.01
            )
        else:
            done = convergence_gate(de, res_norm, tol_eff, etol, rhotol, diago_tol)
        # Block convergence until the U-ramp reaches full U, so the reported
        # final energy is never at a partial, ramped U_eff.
        if hub_u_ramp_iters > 0 and it < hub_u_ramp_iters:
            done = False
        if done:
            converged = True
            rho_s = rho_out_s
            if is_paw:  # report the one-center energy at the FINAL fresh becsum
                # onec is set together with is_paw in _build_iter_ops.
                assert onec is not None
                e_onec = torch.zeros((), dtype=RDTYPE, device=dev)
                for sp, atoms in _species_atoms(system).items():
                    fresh = [
                        onec[sp]._to_real_t(torch.stack([rho_ij_s[isp][a] for a in atoms]))
                        for isp in range(nspin)
                    ]
                    e_onec = e_onec + onec[sp].e1c_t(fresh).sum().to(dev)
                energies.onecenter = e_onec
            break
        e_free_prev = e_free
        # trust region: a residual jump means the DIIS history is lying about
        # the curvature (typical of the wild first USPP/PAW iterations from
        # the SAD + atomic-becsum start); discard it and restart from a plain
        # damped step rather than extrapolating into nonsense.
        # WINDOWED baseline + cooldown (task #55): with an all-time best, one
        # touch of the eigensolver noise floor poisons the criterion forever —
        # every subsequent iteration resets the mixer, DIIS never re-learns,
        # and the SCF degenerates into pure damped iteration, which DIVERGES
        # geometrically (×2.5/iter observed) for gain>1 sloshing modes and
        # ends 148 eV up in a frozen limit cycle. Recent-best forgets old
        # floors; the cooldown stops a reset that didn't help from re-firing
        # before DIIS has curvature again.
        best_res = min(h["res"] for h in history[-10:])
        if it > 1 and res_norm > trust_factor * best_res and it - last_reset_it >= 5:
            logger.debug(
                "USPP iter %d trust-region mixer reset: residual %.2e > %gx windowed best %.2e",
                it,
                res_norm,
                trust_factor,
                best_res,
            )
            mixer.reset()
            last_reset_it = it
            if verbose:
                print(
                    f"  USPP {it:3d}  [mixer reset: residual jumped "
                    f"{res_norm:.2e} > {trust_factor:g}x best {best_res:.2e}]"
                )
        if spin_precond and nspin == 2 and smearing != "none":
            # Stoner preconditioner on the m-channel (arXiv:2606.26693):
            # rebuilt each iteration from the current orbitals; neutralizes
            # the Stoner-expansive magnetization mode that plain damping
            # amplifies and history mixing cannot hold
            from gradwave.scf.spin_precond import build_stoner_precond

            sp = build_stoner_precond(
                system,
                coeffs,
                eigs_s,
                mu,
                SCHEMES[smearing],
                width,
                rho_out_s[0] + rho_out_s[1],
                rho_out_s[0] - rho_out_s[1],
                xc,
                dist_ctx=dist_ctx,
            )
            if sp is None:
                mixer.extra_precond = None
            else:

                def _spin_pc(rvec, _sp=sp):
                    out = rvec.clone()
                    out[ng : 2 * ng] = _sp.apply(rvec[ng : 2 * ng])
                    return out

                mixer.extra_precond = _spin_pc
        if tf_precond is not None:
            tf_precond.set_density(rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1])
        mixed = mixer.step(rho_in_vec, rho_out_vec)
        rho_s, bec_mixed = layout.unpack(mixed)
        # Hermitize the mixed becsum (linear combinations preserve it up
        # to roundoff) for the next iteration's one-center ddd
        for isp in range(nspin):
            for a in range(len(system.atom_slices)):
                m = bec_mixed[isp][a]
                rho_ij_mix[isp][a] = 0.5 * (m + m.conj().T)

    # Band-count guard (collinear nspin=2). default_nbands sizes the per-channel
    # band count from the PARAMAGNETIC ceil(N/2) with no moment dependence, so a
    # sizable spin moment can leave the majority channel with more occupied
    # states than bands — the Fermi solver then cannot place all majority
    # electrons and the electron count / free energy come out wrong. Occupation
    # left in the highest band is the direct symptom (occ_s is per-state in
    # [0,1] on the nspin=2 path); warn to raise nbands. Cheap: one reduction.
    if nspin == 2 and occ_s is not None:
        top_occ = max(float(o[:, -1].max()) for o in occ_s)
        if top_occ > 1e-3:
            logger.warning(
                "collinear nspin=2: the highest band (of %d) carries occupation "
                "%.3g — the majority spin channel is not fully accommodated, so "
                "the converged electron count / free energy may be wrong. "
                "default_nbands sizes bands from the paramagnetic ceil(N/2) with "
                "no moment dependence; pass an explicit larger nbands covering "
                "ceil((N+|M|)/2), e.g. setup_uspp(..., nbands=...).",
                system.nbands, top_occ,
            )

    if not converged:
        logger.warning(
            "USPP/PAW SCF did NOT converge in %d iterations: F=%+.10f eV, |drho|=%.3e",
            it,
            e_free,
            res_norm,
        )
    rho_final = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    # Return the becsum that PAIRS with the returned density. At convergence
    # rho_s was set to the fresh map output (rho_out_s), so the fresh output
    # becsum (rho_ij_s) is the consistent partner. On non-convergence rho_s is
    # the mixer's output (the next-iteration input), so the mixed becsum
    # (rho_ij_mix) is the match — it also tracks the energies (the one-center
    # ddd is built from rho_ij_mix) and avoids a poisoned solver-blowup output.
    # Consumers that read ρ and becsum as one state (paw_forces/paw_stress via
    # _normalize_spin) then see a self-consistent pair regardless of convergence.
    rho_ij_final = rho_ij_s if converged else rho_ij_mix
    # Explicit keyword args (each already None-default on USPPResult) instead
    # of building a heterogeneous dict and splatting it with **extra: a plain
    # `dict[str, ...]` can't carry a different value type per key, so **extra
    # forced every one of these fields to accept the union of ALL of them
    # (hub_occ/hub_sites/rho_spin/mag_total/mag_abs all showing up as
    # "expected `int | float | None`, found `None | list[list[Tensor]] |
    # ...`" and the like) -- passing None directly for the not-applicable
    # case is behaviorally identical (same field default).
    hub_occ = n_hub_s if hub is not None else None
    hub_sites = hub.sites if hub is not None else None
    rho_spin = rho_s if nspin == 2 else None
    mag_total = (
        float((rho_s[0] - rho_s[1]).sum()) * vol / grid.n_points if nspin == 2 else None
    )
    mag_abs = (
        float((rho_s[0] - rho_s[1]).abs().sum()) * vol / grid.n_points if nspin == 2 else None
    )
    # The loop above always runs (max_iter >= 1 in practice) so these are real
    # values from the last iteration by the time we get here, never the
    # pre-loop placeholders.
    assert energies is not None and eigs_s is not None and occ_s is not None
    # _solve_bands_uspp mutates every coeffs[isp][ik] in place unconditionally
    # (same reasoning as _scf_iteration's own coeffs_full cast above), so by
    # the final return every element is a real Tensor, not the seed-compatible
    # `Tensor | None` the warm-start-carrying `coeffs` variable is typed for.
    coeffs_final = cast("list[list[torch.Tensor]]", coeffs)
    if dist_ctx is not None:
        from gradwave.distributed import gather_list_cat

        # Reassemble a normal, full-mesh USPPResult: gather this rank's local
        # per-k coefficients/⟨β|ψ⟩ into the global per-k lists, and substitute
        # the GLOBAL eigenvalues/occupations (already gathered inside the last
        # _scf_iteration call) and the ORIGINAL unsharded system — a caller
        # sees the same shapes/content a single-process run on the full mesh
        # would have produced (rho/rho_ij_atoms/hub_occ are already global,
        # see _build_output_density/_hubbard_occ_update's own all_reduce).
        coeffs_final = [gather_list_cat(coeffs_final[isp], dist_ctx) for isp in range(nspin)]
        becps_s = [gather_list_cat(becps_s[isp], dist_ctx) for isp in range(nspin)]
        eigs_s = eigs_global_s
        occ_s = occ_global_s
        # DistKContext.full_system is System | USPPSystem (shared with the NC
        # driver's dist_ctx) -- this rank's dist_ctx was built by
        # shard_uspp_system, so it's always a USPPSystem here.
        system = cast("USPPSystem", dist_ctx.full_system)
    return USPPResult(
        converged=converged,
        n_iter=len(history),
        energies=energies,
        eigenvalues=eigs_s[0] if nspin == 1 else torch.stack(eigs_s),
        occupations=occ_s[0] if nspin == 1 else torch.stack(occ_s),
        coeffs=coeffs_final[0] if nspin == 1 else coeffs_final,
        rho=rho_final,
        rho_ij_atoms=rho_ij_final[0] if nspin == 1 else rho_ij_final,
        becps=becps_s[0] if nspin == 1 else becps_s,
        history=history,
        fermi=mu,
        system=system,
        nspin=nspin,
        smearing=smearing,
        width=width,
        mixer_mult=mixer.block_mult,
        rho_out_spin=rho_out_s,  # RAW map output (pre-mixing) — rig/diagnostics
        hub_occ=hub_occ,
        hub_sites=hub_sites,
        rho_spin=rho_spin,
        mag_total=mag_total,
        mag_abs=mag_abs,
        recorder=recorder,
        boundary=boundary,
        esm_bias=esm_bias,
    )
