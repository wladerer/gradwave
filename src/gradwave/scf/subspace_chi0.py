"""Cheap low-rank subspace χ₀ + Woodbury dielectric preconditioner (M1).

The exact-dielectric "teacher" (``experiments/chi0_precond_crux``: apply_chi0 →
apply_k_hxc → anderson_solve) cuts the outer SCF iteration count on a stiff
magnet (bcc-Fe FM) by ~2.2-2.6× but costs ~7.6× more FFTs per step, because
each preconditioner apply runs a full conduction-space Sternheimer solve of χ₀.

This module builds the CHEAP student. The shipped metallic response
(``scf/implicit.py::_chi0_channel_metal``) already decomposes Adler-Wiser χ₀
into three physically disjoint pieces summed over the computed band window:

  1. window → above-window transitions, a Sternheimer solve (the FFT-heavy part);
  2. the in-window band-pair sum Σ_nm β_nm ⟨ψ_m|w|ψ_n⟩ ψ_n*ψ_m with the
     divided-difference weight β_nm = (f_n−f_m)/(ε_n−ε_m) — pure GEMM over the
     Ritz codensities the eigensolver already returns each SCF step;
  3. the rank-one δμ Fermi-level-shift term −δμ Σ_n f'_n |ψ_n|² that conserves
     electron number.

The **subspace χ₀** keeps pieces (2)+(3) and DROPS the Sternheimer piece (1).
Both retained pieces are real, symmetric, and low rank. Writing each as a sum of
real rank-one dyads (an off-diagonal pair {n,m} splits into the real and
imaginary codensity parts p_nm = Re(ψ_n*ψ_m), q_nm = Im(ψ_n*ψ_m); the diagonal
is |ψ_n|²; δμ is the Fermi-window field Q_σ = Σ_kn w_k f'_n |ψ_n|²):

    χ₀_sub = Σ_p c_p |φ_p⟩⟨φ_p|,   φ_p real.

The outer SCF Newton step preconditions the density residual with
ε_ρ⁻¹ = (1 − χ₀·K_Hxc)⁻¹ (density-space Jacobian J_ρ = χ₀·K_Hxc, matching the
teacher). With the low-rank χ₀_sub,

    M_ρ = χ₀_sub·K_Hxc = Σ_p c_p |φ_p⟩⟨K_Hxc φ_p|  =  U · diag(c) · W†,

so (1 − M_ρ)⁻¹ is EXACT by Woodbury: 1 + U (diag(1/c) − W†U)⁻¹ W† with an
n_col × n_col dense LU — the same identity ``StonerSpinPrecond`` uses for its
m-channel Stoner diagonal, here generalised to the full coupled charge+spin
window (the columns live in the stacked (up,dn) density space; K_Hxc supplies
the charge/spin coupling). ZERO FFTs per apply: the codensities and their
K_Hxc images are built ONCE at freeze; the per-step apply is one dense solve.

Frozen at a converged reference exactly like the teacher, so the ONLY thing that
differs between them is χ₀ fidelity (full Sternheimer vs in-window sum) — the
apples-to-apples measurement the M1 crux asks for.
"""

from __future__ import annotations

import torch

from gradwave.core.fftbox import g_to_r, r_to_g
from gradwave.core.occupations import SCHEMES
from gradwave.dtypes import CDTYPE
from gradwave.postscf._response import (
    divided_difference_weights,
    occupation_derivative,
)
from gradwave.scf.implicit import apply_k_hxc
from gradwave.scf.loop import SCFResult


# --------------------------------------------------------------------------
# The generalised Woodbury apply (StonerSpinPrecond, coupled charge+spin).
# --------------------------------------------------------------------------
class WoodburyPrecond:
    """(1 − M_ρ)⁻¹ on the stacked density residual, M_ρ = U diag(c) W†.

    ``u_g`` / ``w_g``: (n_col, nvec) codensity output columns φ̂_p and their
    K_Hxc images (Ŵ_p = (K_Hxc φ_p)^), on the density sphere, stacked over spin
    (nvec = ng for nspin=1, 2·ng in the (up,dn) basis for nspin=2). ``cvals``:
    (n_col,) real dyad weights c_p. The physical pairing ⟨a,b⟩ = Ω Σ_G â* b̂
    carries the cell volume Ω into diag(1/c) (as in ``StonerSpinPrecond``).

    ``acts_on = "grid"`` so the SCF hands over the whole grid block; for nspin=2
    the residual arrives in the mixer's (total, magnetization) basis and is
    converted to (up,dn) — and back — purely in G-space (a linear combination,
    NO FFT), keeping the apply FFT-free.
    """

    acts_on = "grid"

    def __init__(self, u_g: torch.Tensor, w_g: torch.Tensor,
                 cvals: torch.Tensor, volume: float, nspin: int, ng: int,
                 g0_idx: int) -> None:
        self.nspin = nspin
        self.ng = ng
        self.volume = volume
        self.g0_idx = g0_idx  # position of G=0 in the density sphere
        self._u = u_g
        self._w = w_g
        self.cvals = cvals
        self.n_col = int(u_g.shape[0])
        d_inv = torch.diag(1.0 / (volume * cvals.to(CDTYPE)))
        a = d_inv - torch.einsum("ag,bg->ab", w_g.conj(), u_g)
        self._a_lu = torch.linalg.lu_factor(a)
        # telemetry
        self.n_calls = 0

    def _apply_updn(self, r: torch.Tensor) -> torch.Tensor:
        """(1 − M_ρ)⁻¹ r on the stacked (up,dn) sphere vector."""
        proj = torch.einsum("ag,g->a", self._w.conj(), r)
        sol = torch.linalg.lu_solve(*self._a_lu, proj[:, None])[:, 0]
        return r + torch.einsum("ag,a->g", self._u, sol)

    def __call__(self, r_grid: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        if self.nspin == 1:
            out = self._apply_updn(r_grid.to(CDTYPE))
            out[self.g0_idx] = 0.0  # conserve N_e: total-channel G=0 stays pinned
            return out.to(r_grid.dtype)
        ng = self.ng
        tot, mag = r_grid[:ng], r_grid[ng:2 * ng]
        updn = torch.cat([(tot + mag) / 2.0, (tot - mag) / 2.0]).to(CDTYPE)
        out = self._apply_updn(updn)
        up, dn = out[:ng], out[ng:2 * ng]
        res = torch.cat([up + dn, up - dn]).to(r_grid.dtype)
        # conserve total electron number: pin the total-channel G=0. The
        # magnetization G=0 (total moment) is free — it is not a fixed count.
        res[self.g0_idx] = 0.0
        return res


# --------------------------------------------------------------------------
# Build the low-rank columns from a converged reference's Ritz window.
# --------------------------------------------------------------------------
def _channel_dyads(res: SCFResult, isp: int, mu: float, scheme, width: float,
                   f_full: float, fp_cut: float, pair_cut: float):
    """Real rank-one dyads (φ real-space field, weight c) of χ₀_sub for one spin
    channel. Pieces (2) in-window band pairs + (3) δμ Fermi-shift of
    ``_chi0_channel_metal`` — the Sternheimer piece (1) is intentionally omitted.

    φ fields are per-channel real grids; the caller embeds them in the stacked
    (up,dn) space and applies K_Hxc for the projection columns."""
    system = res.system
    grid = system.grid
    ng_shape = grid.shape
    vol = float(grid.volume)
    # Two 1/Ω factors: one is the physical χ₀ measure (the /volume in
    # _chi0_channel_metal), the other converts the density-sphere physical
    # projection ⟨φ_p, ·⟩ = Ω Σ_G φ̂* (·) used by the Woodbury apply into the
    # bare wavefunction-sphere G-sum V = (1/Ω)∫ that the matrix-free response
    # uses. Validated bit-for-bit against apply_chi0_subspace.
    v2 = vol * vol

    dyads: list[tuple[torch.Tensor, float]] = []
    # δμ accumulators (per channel): Q_σ = Σ_kn w_k f'_n |ψ_n|², den = Σ w_k f'_n
    q_field = torch.zeros(ng_shape, dtype=torch.float64, device=grid.g2.device)
    den = 0.0

    for ik, sph in enumerate(system.spheres):
        if res.nspin == 1:
            c_all, eps, occ = res.coeffs[ik], res.eigenvalues[ik], res.occupations[ik]
        else:
            c_all = res.coeffs[isp][ik]
            eps = res.eigenvalues[isp][ik]
            occ = res.occupations[isp][ik]
        kw = float(system.kweights[ik])
        occ_d = occupation_derivative(eps, mu, scheme, width, f_full)
        beta = divided_difference_weights(eps, occ, occ_d)  # (nb, nb), symmetric
        psi_r = g_to_r(c_all, sph.flat_idx, ng_shape)  # (nb, *grid) complex
        nb = c_all.shape[0]

        # δμ field: Σ_n f'_n |ψ_n|²  (∫|ψ_n|² = Ω → codensity |ψ|² integrates Ω)
        dens_n = (psi_r.conj() * psi_r).real  # (nb, *grid)
        q_field += kw * torch.einsum("n,n...->...", occ_d, dens_n)
        den += kw * float(occ_d.sum())

        # in-window band pairs. Diagonal: c = w_k β_nn |ψ_n|²; off-diag {n,m}
        # (n>m): two real dyads 2·w_k β_nm on p=Re(ψ_n*ψ_m), q=Im(ψ_n*ψ_m).
        for n in range(nb):
            bnn = float(beta[n, n])
            if abs(kw * bnn) > pair_cut:
                dyads.append((dens_n[n], kw * bnn / v2))
            for m in range(n):
                bnm = float(beta[n, m])
                if abs(kw * bnm) <= pair_cut:
                    continue
                gnm = psi_r[n].conj() * psi_r[m]  # ψ_n* ψ_m
                w = 2.0 * kw * bnm / v2
                dyads.append((gnm.real.contiguous(), w))
                dyads.append((gnm.imag.contiguous(), w))

    if abs(den) > 1e-30:
        dyads.append((q_field, -1.0 / (v2 * den)))
    return dyads


def build_woodbury_subspace(res: SCFResult, xc, *, fp_cut: float = 1e-8,
                            pair_cut: float = 1e-6, max_cols: int = 512):
    """Assemble the frozen low-rank Woodbury preconditioner from a converged
    reference ``res`` (a metal / smeared FSM; the in-window Adler-Wiser sum
    needs partial occupations to carry weight).

    Cost: one ``apply_k_hxc`` per retained column (a handful of FFTs), done ONCE.
    Returns a :class:`WoodburyPrecond` (zero FFTs per subsequent apply), or
    ``None`` when no column carries Fermi-surface weight (insulating limit — the
    operator is the identity and the SCF is better off with plain Kerker/TF)."""
    nspin = getattr(res, "nspin", 1)
    grid = res.system.grid
    mask = grid.dens_mask.reshape(-1)
    ng = int(mask.sum())
    vol = float(grid.volume)
    scheme = SCHEMES[res.smearing]
    if nspin == 1:
        mu_s = (float(res.fermi),)
        f_full = 2.0
    else:
        mu_s = getattr(res, "fermi_spin", None) or (float(res.fermi),) * 2
        f_full = 1.0

    # collect (channel, φ real field, weight) dyads
    cols: list[tuple[int, torch.Tensor, float]] = []
    for isp in range(nspin):
        for phi, c in _channel_dyads(res, isp, float(mu_s[isp]), scheme,
                                     res.width, f_full, fp_cut, pair_cut):
            cols.append((isp, phi, c))
    if not cols:
        return None
    # cost control: keep the largest-|weight| columns
    cols.sort(key=lambda t: -abs(t[2]))
    cols = cols[:max_cols]

    n_col = len(cols)
    nvec = ng * nspin
    u_g = torch.zeros(n_col, nvec, dtype=CDTYPE, device=grid.g2.device)
    w_g = torch.zeros(n_col, nvec, dtype=CDTYPE, device=grid.g2.device)
    cvals = torch.tensor([c for (_, _, c) in cols], dtype=torch.float64,
                         device=grid.g2.device)
    for i, (isp, phi, _c) in enumerate(cols):
        # output column φ_p, embedded in its own spin channel
        phi_g = r_to_g(phi.to(CDTYPE)).reshape(-1)[mask]
        if nspin == 1:
            u_g[i] = phi_g
            khxc = apply_k_hxc(res, xc, phi)
            w_g[i] = r_to_g(khxc.to(CDTYPE)).reshape(-1)[mask]
        else:
            u_g[i, isp * ng:(isp + 1) * ng] = phi_g
            # K_Hxc couples spin: embed φ in channel isp, apply, keep both blocks
            stacked = torch.zeros(2, *grid.shape, dtype=torch.float64,
                                  device=grid.g2.device)
            stacked[isp] = phi
            khxc = apply_k_hxc(res, xc, stacked)  # (2, *grid), coupled
            w_g[i, :ng] = r_to_g(khxc[0].to(CDTYPE)).reshape(-1)[mask]
            w_g[i, ng:] = r_to_g(khxc[1].to(CDTYPE)).reshape(-1)[mask]

    g0_idx = int((grid.g2.reshape(-1)[mask] == 0).nonzero().flatten()[0])
    return WoodburyPrecond(u_g, w_g, cvals, vol, nspin, ng, g0_idx)


# --------------------------------------------------------------------------
# Matrix-free reference (validation): χ₀_sub w = pieces (2)+(3), no Sternheimer.
# --------------------------------------------------------------------------
@torch.no_grad()
def apply_chi0_subspace(res: SCFResult, w_r: torch.Tensor) -> torch.Tensor:
    """δρ = χ₀_sub w — the in-window + δμ Adler-Wiser response, matrix-free.

    A faithful copy of ``scf/implicit.py::_chi0_channel_metal`` with the
    ``_sternheimer_above`` (window → virtual) block removed. Used ONLY to
    validate the low-rank factorisation in :class:`WoodburyPrecond` (their
    implied M_ρ = χ₀_sub·K_Hxc must agree to round-off). nspin=1: grid field in
    and out; nspin=2: stacked (2, *grid)."""
    nspin = getattr(res, "nspin", 1)
    scheme = SCHEMES[res.smearing]
    if nspin == 1:
        return _chi0_sub_channel(res, 0, w_r, 2.0, float(res.fermi), scheme,
                                 res.width)
    mu_s = getattr(res, "fermi_spin", None) or (float(res.fermi),) * 2
    return torch.stack([
        _chi0_sub_channel(res, isp, w_r[isp], 1.0, float(mu_s[isp]), scheme,
                          res.width) for isp in range(nspin)])


def _chi0_sub_channel(res, isp, w_r, f_full, mu, scheme, width):
    system = res.system
    grid = system.grid
    ng_shape = grid.shape
    from gradwave.core.fftbox import box_to_sphere
    from gradwave.dtypes import RDTYPE

    dr = torch.zeros(ng_shape, dtype=RDTYPE, device=grid.g2.device)
    stash = []
    num = 0.0
    den = 0.0
    for ik, sph in enumerate(system.spheres):
        if nspin_of(res) == 1:
            c_all, eps, occ = res.coeffs[ik], res.eigenvalues[ik], res.occupations[ik]
        else:
            c_all = res.coeffs[isp][ik]
            eps = res.eigenvalues[isp][ik]
            occ = res.occupations[isp][ik]
        occ_d = occupation_derivative(eps, mu, scheme, width, f_full)
        psi_r = g_to_r(c_all, sph.flat_idx, ng_shape)
        wpsi_sph = box_to_sphere(r_to_g(psi_r * w_r), sph.flat_idx)
        v_mat = c_all.conj() @ wpsi_sph.transpose(-1, -2)
        vnn = torch.diagonal(v_mat).real
        kw = float(system.kweights[ik])
        num += kw * float((occ_d * vnn).sum())
        den += kw * float(occ_d.sum())
        stash.append((c_all, eps, occ, occ_d, psi_r, v_mat, kw))
    dmu = num / den if abs(den) > 1e-30 else 0.0

    for c_all, eps, occ, occ_d, psi_r, v_mat, kw in stash:
        pr = psi_r.reshape(c_all.shape[0], -1)
        beta = divided_difference_weights(eps, occ, occ_d)
        a_mat = beta * v_mat.transpose(-1, -2)
        inwin = torch.einsum("nm,ng,mg->g", a_mat, pr.conj(), pr).real
        occ_resp = torch.einsum("n,ng->g", occ_d, (pr.conj() * pr).real)
        dr += (kw * (inwin - dmu * occ_resp)).reshape(ng_shape)
    return dr / grid.volume


def nspin_of(res) -> int:
    return getattr(res, "nspin", 1)
