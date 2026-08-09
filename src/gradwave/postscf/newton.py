"""Newton-Krylov SCF finisher (task #65).

Every mixer approximates the inverse dielectric operator (1 − Kχ₀)⁻¹ from
history or model susceptibilities. The differentiable machinery owns the
EXACT independent-particle response χ̃ (postscf/uspp_implicit: generalized
Sternheimer with the Fermi-surface occupation channel) and the exact
Hartree-XC + one-center kernels as autograd HVPs — so the true Newton step

    (I − χ̃K) δ = r,    r = F(x) − x

is computable, and the outer iteration converges quadratically near the
fixed point. This is a FINISHER: each Newton step costs one raw SCF
iteration (the residual) plus an Anderson-accelerated inner solve whose
every iteration is a full Sternheimer batch — far more than a mixed step,
worth it only to land 1e-10 from a 1e-3..1e-5 start in 2-3 steps, or to
polish states for derivative work (the adjoint assumes a tightly
converged fixed point).

Coverage follows the response machinery: nspin 1 or 2 (the composite mixing
vector doubles to the per-spin (δρ↑, δρ↓, δbec↑, δbec↓), and χ̃/K_Hxc reuse
the spin-resolved operators already in uspp_implicit._ConvergedUSPP). DFT+U is
supported: the composite vector gains the per-channel Hubbard occupation-matrix
block δn (packed as (Re, Im) pairs, mirroring the USPP adjoint's join/split),
the raw map carries the mixer-side n through _scf_iteration into V_U, and the
Newton Jacobian folds in the Dudarev kernel K_U = −(U−J) via
_ConvergedUSPP.{k_hub, apply_chi0}'s hub channel. nspin=2 follows the USPP/PAW
formalism itself — smeared occupations
(a shared Fermi level per iteration, magnetic insulators via a small width),
so the Fermi-surface χ̃ channel carries the response.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import torch

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import RDTYPE
from gradwave.postscf.uspp_implicit import _check_supported, _ConvergedUSPP
from gradwave.scf.results import USPPResult
from gradwave.scf.uspp_loop import _build_iter_ops, _scf_iteration


def _pack(w_sp: list[torch.Tensor], mats_sp: list[list[torch.Tensor]],
          hub_sp: list[list[torch.Tensor]] | None = None) -> torch.Tensor:
    """Composite residual/state vector: per-spin grid fields first, then the
    per-spin, per-atom becsum blocks, then (with +U) the per-channel, per-site
    Hubbard occupation-matrix blocks. The layout matches the USPP adjoint's
    ``join`` (uspp_implicit.uspp_density_loss_param_grads): the hub matrices are
    complex Hermitian, so each is appended as a (Re, Im) pair. ``w_sp`` and
    ``mats_sp`` are per-spin lists; ``hub_sp`` is a per-channel list."""
    parts = ([w.reshape(-1) for w in w_sp]
             + [m.reshape(-1) for mats in mats_sp for m in mats])
    if hub_sp is not None:
        for ch in hub_sp:
            for m in ch:
                parts.append(m.real.reshape(-1))
                parts.append(m.imag.reshape(-1))
    return torch.cat(parts)


def _unpack(
    v: torch.Tensor, shape: tuple[int, int, int], n_pts: int, nbec: list[int],
    nspin: int, hub_dims: list[int] | None = None, nsh: int = 0
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]],
           list[list[torch.Tensor]] | None]:
    """Inverse of :func:`_pack`: flat v → (per-spin grid fields, per-spin
    per-atom becsum matrices, per-channel per-site Hubbard occupation matrices
    or None). ``nsh`` (number of hub channels) and ``hub_dims`` (per-site
    manifold dimensions) select the hub block; ``nsh == 0`` skips it."""
    w_sp, off = [], 0
    for _ in range(nspin):
        w_sp.append(v[off:off + n_pts].reshape(shape))
        off += n_pts
    mats_sp = []
    for _ in range(nspin):
        mats = []
        for n in nbec:
            mats.append(v[off:off + n * n].reshape(n, n))
            off += n * n
        mats_sp.append(mats)
    hub_sp: list[list[torch.Tensor]] | None = None
    if nsh:
        assert hub_dims is not None
        hub_sp = []
        for _ in range(nsh):
            ch = []
            for d in hub_dims:
                re = v[off:off + d * d].reshape(d, d)
                im = v[off + d * d:off + 2 * d * d].reshape(d, d)
                ch.append(torch.complex(re, im))
                off += 2 * d * d
            hub_sp.append(ch)
    return w_sp, mats_sp, hub_sp


def newton_polish(res: USPPResult, xc: XCFunctional | SpinXC, *, tol: float = 1e-10,
                  max_newton: int = 5,
                  inner_tol: float = 1e-8, max_inner: int = 60,
                  cg_tol: float = 1e-9, cg_max_iter: int = 200,
                  beta: float = 0.3, history: int = 8,
                  diago_tol: float = 1e-11, verbose: bool = False) -> USPPResult:
    """Polish a near-converged scf_uspp result to `tol` in the density
    residual by exact-Jacobian Newton steps. Returns an updated USPPResult
    (fresh orbitals/energies from the final residual evaluation, plus a
    `newton` list of per-step residual norms).

    nspin=1 or 2: for spin the composite residual/state vector is the per-spin
    (δρ↑, δρ↓, δbec↑, δbec↓), and the exact Jacobian reuses the spin-resolved
    χ̃ / K_Hxc / one-center operators already in _ConvergedUSPP (χ̃ block-
    diagonal over spin except the shared-Fermi δμ; K_Hxc cross-spin)."""
    _check_supported(res)
    if getattr(xc, "needs_tau", False):
        raise NotImplementedError(
            "USPP Newton polish does not support meta-GGA (needs_tau): the raw "
            "map here evaluates H without the τ generalized-KS operator"
        )
    nspin = res.nspin
    system = res.system
    grid = system.grid
    shape, n_pts = tuple(grid.shape), grid.n_points
    nbec = [s1 - s0 for (s0, s1) in system.atom_slices]
    smearing = res.smearing
    width = res.width

    # DFT+U: reconstruct the manifold list from the result's hub_sites (same
    # species-resolved mapping _ConvergedUSPP uses), so the raw map carries the
    # mixer-side occupation n through _scf_iteration into V_U and the Newton
    # Jacobian gains the hub channel. hub_occ is the [channel][site] occupation
    # (nsh = 1 for nspin=1's half-occupancy channel, 2 for nspin=2).
    has_hub = res.hub_occ is not None
    manifolds = None
    nsh = 0
    hub_dims: list[int] = []
    nhub_s: list[list[torch.Tensor]] | None = None
    if has_hub:
        from gradwave.core.hubbard import HubbardManifold

        assert res.hub_sites is not None
        man: dict[int, HubbardManifold] = {}
        for st in res.hub_sites:
            sp = system.species_of_atom[int(st["atom"])]
            man[sp] = HubbardManifold(species=sp, l=int(st["l"]),
                                      u=float(st["u"]), j=float(st["j"]))
        manifolds = list(man.values())
        nhub_s = [[m.detach().clone() for m in ch] for ch in res.hub_occ]
        nsh = len(nhub_s)
        hub_dims = [int(m.shape[0]) for m in nhub_s[0]]

    # the raw map is evaluated through _scf_iteration directly (stage 3):
    # operators built once, orbital warm starts carried across Newton steps.
    # hub_occ_mix stays at its raw one-step-lag default (1.0) so the map is the
    # exact fixed point the Newton correction targets.
    ops = _build_iter_ops(system, xc, nspin=nspin, smearing=smearing,
                          width=width, batched=True, hubbard=manifolds)
    coeffs = [[None] * ops.nk for _ in range(nspin)]
    coeffs_b = [None] * nspin

    # per-spin state (nspin=1: a length-1 list; nspin=2: [↑, ↓])
    if nspin == 1:
        rho_s = [res.rho.detach().clone()]
    else:
        # scf_uspp always sets rho_spin when nspin=2 (see results.py).
        assert res.rho_spin is not None
        rho_s = [r.detach().clone() for r in res.rho_spin]
    if nspin == 1:
        rho_ij_flat = cast("list[torch.Tensor]", res.rho_ij_atoms)
        bec_s = [[m.detach().clone() for m in rho_ij_flat]]
    else:
        rho_ij_nested = cast("list[list[torch.Tensor]]", res.rho_ij_atoms)
        bec_s = [[m.detach().clone() for m in ch] for ch in rho_ij_nested]
    hist_out = []
    it_out = None
    best = float("inf")
    stalls = 0
    with torch.no_grad():
        for step in range(1, max_newton + 1):
            it_out = _scf_iteration(ops, rho_s, bec_s, coeffs, coeffs_b,
                                    nhub_s, diago_tol, 0)
            r_rho_s = [(f - r).to(RDTYPE) for f, r in
                       zip(it_out["rho_out_s"], rho_s, strict=True)]
            r_bec_s = [[(a - b).real.to(RDTYPE) for a, b in
                        zip(fbec, bec, strict=True)]
                       for fbec, bec in
                       zip(it_out["rho_ij_s"], bec_s, strict=True)]
            # +U: the occupation-matrix residual n_out(x) − n_in per channel
            # (complex Hermitian; the raw map's hub_occ_mix=1.0 makes n_out the
            # fresh matrix, so this is the true fixed-point residual).
            r_hub_s = None
            if has_hub:
                assert nhub_s is not None and it_out["n_hub_s"] is not None
                r_hub_s = [[(a - b) for a, b in zip(fh, nh, strict=True)]
                           for fh, nh in
                           zip(it_out["n_hub_s"], nhub_s, strict=True)]
            r_vec = _pack(r_rho_s, r_bec_s, r_hub_s)
            rn = float(torch.linalg.norm(r_vec))
            hist_out.append(rn)
            if verbose:
                print(f"  newton {step}: |F(x)-x| = {rn:.3e}")
            if rn < tol:
                break
            # noise floor of the residual EVALUATION (eigensolver noise in
            # the raw map): quadratic steps stop improving — stop honestly
            # at the achievable precision instead of thrashing
            if rn > 0.5 * best:
                stalls += 1
                if stalls >= 2:
                    if verbose:
                        print(f"  newton: residual floored at {best:.2e}")
                    break
            else:
                stalls = 0
            best = min(best, rn)

            # Jacobian frozen at the CURRENT x: the iteration's orbitals
            # diagonalize H[x], so they define χ̃ at x exactly. The frozen-state
            # dict follows the shape _ConvergedUSPP / _window_uspp expect per
            # nspin (flat per-k for nspin=1, per-spin, per-k for nspin=2).
            rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
            jac_res: dict[str, Any] = dict(
                system=system, nspin=nspin, smearing=smearing, width=width,
                fermi=it_out["mu"], rho=rho_tot,
                coeffs=(coeffs[0] if nspin == 1 else coeffs),
                eigenvalues=(it_out["eigs_s"][0] if nspin == 1
                             else it_out["eigs_s"]),
                occupations=(it_out["occ_s"][0] if nspin == 1
                             else it_out["occ_s"]),
                rho_ij_atoms=(bec_s[0] if nspin == 1 else bec_s))
            if nspin == 2:
                jac_res["rho_spin"] = rho_s
            if has_hub:
                # freeze χ̃'s +U channel (S-dressed projectors + V_U from the
                # current occupation) at x — _ConvergedUSPP rebuilds it from
                # these two keys exactly as the adjoint does.
                jac_res["hub_occ"] = nhub_s
                jac_res["hub_sites"] = res.hub_sites
            cs = _ConvergedUSPP(jac_res, xc)
            dpsi_warm = [[torch.zeros_like(c[:ns]) for c, ns in
                          zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
                         for isp in range(nspin)]

            # inner solve: δ = r + χ̃(K δ), Anderson-accelerated — the
            # same fixed-point shape as the adjoint, on the forward side
            d = r_vec.clone()
            prev_d = prev_g = None
            hist_dd, hist_dg = [], []
            # Bound before the loop purely so ty can see `gn` as always
            # assigned in the `else` clause below (max_inner is always >= 1
            # in practice, same "runs >=1 iteration" pattern as
            # scf/loop.py/uspp_loop.py) -- overwritten every real iteration.
            gn = float("inf")
            for _it in range(1, max_inner + 1):
                d_rho_s, d_bec_s, d_hub_s = _unpack(
                    d, shape, n_pts, nbec, nspin, hub_dims, nsh)
                v_sp = cs.k_hxc_grid(d_rho_s)
                d_ddd = cs.hvp_onecenter([[m.to(torch.complex128)
                                           for m in ch] for ch in d_bec_s])
                # +U: K_U δn = −(U−J)·herm(δn) is the D-matrix perturbation the
                # hub channel of χ̃ consumes; its δn response packs back in.
                d_hubd = None
                if nsh:
                    assert d_hub_s is not None
                    d_hubd = cs.k_hub(d_hub_s)
                chi_rho, chi_bec, chi_hub = cs.apply_chi0(
                    v_sp, d_ddd, dpsi_warm, cg_tol, cg_max_iter,
                    d_hub_sp=d_hubd)
                g_vec = r_vec + _pack(
                    [cr.to(RDTYPE) for cr in chi_rho],
                    [[m.real.to(RDTYPE) for m in ch] for ch in chi_bec],
                    chi_hub)
                g_res = g_vec - d
                gn = float(torch.linalg.norm(g_res)) / max(
                    1.0, float(torch.linalg.norm(d)))
                if gn < inner_tol:
                    d = g_vec
                    break
                if prev_g is not None:
                    # prev_d/prev_g are always set together (initialized
                    # together, reassigned together below).
                    assert prev_d is not None
                    hist_dd.append(d - prev_d)
                    hist_dg.append(g_res - prev_g)
                    if len(hist_dg) > history:
                        hist_dd.pop(0)
                        hist_dg.pop(0)
                prev_d, prev_g = d, g_res
                if hist_dg:
                    dg = torch.stack(hist_dg, dim=1)
                    dd = torch.stack(hist_dd, dim=1)
                    gam = torch.linalg.lstsq(dg, g_res[:, None]).solution[:, 0]
                    d = d + beta * g_res - (dd + beta * dg) @ gam
                else:
                    d = d + beta * g_res
            else:
                raise RuntimeError(
                    f"Newton inner solve stalled ({gn:.2e} after "
                    f"{max_inner} iterations)")

            d_rho_s, d_bec_s, d_hub_s = _unpack(
                d, shape, n_pts, nbec, nspin, hub_dims, nsh)
            rho_s = [r + dr for r, dr in zip(rho_s, d_rho_s, strict=True)]
            bec_s = [[b + m.to(b.dtype) for b, m in zip(bec, dbec, strict=True)]
                     for bec, dbec in zip(bec_s, d_bec_s, strict=True)]
            if has_hub:
                assert nhub_s is not None and d_hub_s is not None
                nhub_s = [[n + dh.to(n.dtype) for n, dh in
                           zip(nch, dch, strict=True)]
                          for nch, dch in zip(nhub_s, d_hub_s, strict=True)]

    # The outer loop above always runs (max_newton >= 1 in practice), so
    # it_out is a real _IterStep from the last iteration here, never the
    # pre-loop placeholder (same "runs >=1 iteration" pattern as
    # scf/loop.py/uspp_loop.py).
    assert it_out is not None
    rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    out: dict[str, Any] = dict(
        coeffs=(coeffs[0] if nspin == 1 else coeffs),
        eigenvalues=(it_out["eigs_s"][0] if nspin == 1 else it_out["eigs_s"]),
        occupations=(it_out["occ_s"][0] if nspin == 1 else it_out["occ_s"]),
        becps=(it_out["becps_s"][0] if nspin == 1 else it_out["becps_s"]),
        fermi=it_out["mu"], energies=it_out["energies"],
        rho_out_spin=it_out["rho_out_s"], rho=rho_tot,
        rho_ij_atoms=(bec_s[0] if nspin == 1 else bec_s),
        newton=hist_out, converged=min(hist_out) < tol)
    if has_hub:
        out["hub_occ"] = nhub_s
    if nspin == 2:
        dvol = grid.volume / n_pts
        dmag = rho_s[0] - rho_s[1]
        out["rho_spin"] = rho_s
        out["mag_total"] = float(dmag.sum() * dvol)
        out["mag_abs"] = float(dmag.abs().sum() * dvol)
    return replace(res, **out)
