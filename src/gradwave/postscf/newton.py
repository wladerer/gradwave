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
the spin-resolved operators already in uspp_implicit._ConvergedUSPP), no +U
(the +U raw-map still lacks the occupation block; the adjoint's +U response
exists). nspin=2 follows the USPP/PAW formalism itself — smeared occupations
(a shared Fermi level per iteration, magnetic insulators via a small width),
so the Fermi-surface χ̃ channel carries the response.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import RDTYPE
from gradwave.postscf.uspp_implicit import _check_supported, _ConvergedUSPP
from gradwave.scf.results import USPPResult
from gradwave.scf.uspp_loop import _build_iter_ops, _scf_iteration


def _pack(w_sp: list[torch.Tensor], mats_sp: list[list[torch.Tensor]]) -> torch.Tensor:
    """Composite residual/state vector: per-spin grid fields first, then the
    per-spin, per-atom becsum blocks (same layout the USPP adjoint's join uses,
    minus the hub block). ``w_sp`` and ``mats_sp`` are per-spin lists."""
    return torch.cat([w.reshape(-1) for w in w_sp]
                     + [m.reshape(-1) for mats in mats_sp for m in mats])


def _unpack(
    v: torch.Tensor, shape: tuple[int, int, int], n_pts: int, nbec: list[int], nspin: int
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """Inverse of :func:`_pack`: flat v → (per-spin grid fields, per-spin
    per-atom becsum matrices)."""
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
    return w_sp, mats_sp


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
    if res.hub_occ is not None:
        raise NotImplementedError("newton_polish: +U raw-map plumbing not "
                                  "implemented (the adjoint's +U response "
                                  "exists; the finisher's packed vector "
                                  "lacks the occupation block)")
    nspin = res.nspin
    system = res.system
    grid = system.grid
    shape, n_pts = tuple(grid.shape), grid.n_points
    nbec = [s1 - s0 for (s0, s1) in system.atom_slices]
    smearing = res.smearing
    width = res.width
    # the raw map is evaluated through _scf_iteration directly (stage 3):
    # operators built once, orbital warm starts carried across Newton steps
    ops = _build_iter_ops(system, xc, nspin=nspin, smearing=smearing,
                          width=width, batched=True)
    coeffs = [[None] * ops.nk for _ in range(nspin)]
    coeffs_b = [None] * nspin

    # per-spin state (nspin=1: a length-1 list; nspin=2: [↑, ↓])
    rho_s = ([res.rho.detach().clone()] if nspin == 1
             else [r.detach().clone() for r in res.rho_spin])
    bec_s = ([[m.detach().clone() for m in res.rho_ij_atoms]] if nspin == 1
             else [[m.detach().clone() for m in ch] for ch in res.rho_ij_atoms])
    hist_out = []
    it_out = None
    best = float("inf")
    stalls = 0
    with torch.no_grad():
        for step in range(1, max_newton + 1):
            it_out = _scf_iteration(ops, rho_s, bec_s, coeffs, coeffs_b,
                                    None, diago_tol, 0)
            r_rho_s = [(f - r).to(RDTYPE) for f, r in
                       zip(it_out["rho_out_s"], rho_s, strict=True)]
            r_bec_s = [[(a - b).real.to(RDTYPE) for a, b in
                        zip(fbec, bec, strict=True)]
                       for fbec, bec in
                       zip(it_out["rho_ij_s"], bec_s, strict=True)]
            r_vec = _pack(r_rho_s, r_bec_s)
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
            jac_res = dict(
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
            cs = _ConvergedUSPP(jac_res, xc)
            dpsi_warm = [[torch.zeros_like(c[:ns]) for c, ns in
                          zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
                         for isp in range(nspin)]

            # inner solve: δ = r + χ̃(K δ), Anderson-accelerated — the
            # same fixed-point shape as the adjoint, on the forward side
            d = r_vec.clone()
            prev_d = prev_g = None
            hist_dd, hist_dg = [], []
            for _it in range(1, max_inner + 1):
                d_rho_s, d_bec_s = _unpack(d, shape, n_pts, nbec, nspin)
                v_sp = cs.k_hxc_grid(d_rho_s)
                d_ddd = cs.hvp_onecenter([[m.to(torch.complex128)
                                           for m in ch] for ch in d_bec_s])
                chi_rho, chi_bec, _ = cs.apply_chi0(
                    v_sp, d_ddd, dpsi_warm, cg_tol, cg_max_iter)
                g_vec = r_vec + _pack(
                    [cr.to(RDTYPE) for cr in chi_rho],
                    [[m.real.to(RDTYPE) for m in ch] for ch in chi_bec])
                g_res = g_vec - d
                gn = float(torch.linalg.norm(g_res)) / max(
                    1.0, float(torch.linalg.norm(d)))
                if gn < inner_tol:
                    d = g_vec
                    break
                if prev_g is not None:
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

            d_rho_s, d_bec_s = _unpack(d, shape, n_pts, nbec, nspin)
            rho_s = [r + dr for r, dr in zip(rho_s, d_rho_s, strict=True)]
            bec_s = [[b + m.to(b.dtype) for b, m in zip(bec, dbec, strict=True)]
                     for bec, dbec in zip(bec_s, d_bec_s, strict=True)]

    rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    out = dict(
        coeffs=(coeffs[0] if nspin == 1 else coeffs),
        eigenvalues=(it_out["eigs_s"][0] if nspin == 1 else it_out["eigs_s"]),
        occupations=(it_out["occ_s"][0] if nspin == 1 else it_out["occ_s"]),
        becps=(it_out["becps_s"][0] if nspin == 1 else it_out["becps_s"]),
        fermi=it_out["mu"], energies=it_out["energies"],
        rho_out_spin=it_out["rho_out_s"], rho=rho_tot,
        rho_ij_atoms=(bec_s[0] if nspin == 1 else bec_s),
        newton=hist_out, converged=min(hist_out) < tol)
    if nspin == 2:
        dvol = grid.volume / n_pts
        dmag = rho_s[0] - rho_s[1]
        out["rho_spin"] = rho_s
        out["mag_total"] = float(dmag.sum() * dvol)
        out["mag_abs"] = float(dmag.abs().sum() * dvol)
    return replace(res, **out)
