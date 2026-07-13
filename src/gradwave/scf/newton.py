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

Coverage follows the response machinery: nspin=1, no +U (the spin kernel
HVP is the same missing piece as the spin adjoint).
"""

from __future__ import annotations

import torch

from gradwave.dtypes import RDTYPE
from gradwave.postscf.uspp_implicit import _check_supported, _ConvergedUSPP
from gradwave.scf.uspp import scf_uspp


def _pack(w_r, mats):
    return torch.cat([w_r.reshape(-1)]
                     + [m.reshape(-1) for m in mats])


def _unpack(v, shape, n_pts, nbec):
    w_r = v[:n_pts].reshape(shape)
    mats, off = [], n_pts
    for n in nbec:
        mats.append(v[off:off + n * n].reshape(n, n))
        off += n * n
    return w_r, mats


def newton_polish(res: dict, xc, *, tol: float = 1e-10, max_newton: int = 5,
                  inner_tol: float = 1e-8, max_inner: int = 60,
                  cg_tol: float = 1e-9, cg_max_iter: int = 200,
                  beta: float = 0.3, history: int = 8,
                  diago_tol: float = 1e-11, verbose: bool = False) -> dict:
    """Polish a near-converged scf_uspp result to `tol` in the density
    residual by exact-Jacobian Newton steps. Returns an updated result
    dict (fresh orbitals/energies from the final residual evaluation,
    plus a "newton" list of per-step residual norms)."""
    _check_supported(res)
    system = res["system"]
    grid = system.grid
    shape, n_pts = tuple(grid.shape), grid.n_points
    nbec = [s1 - s0 for (s0, s1) in system.atom_slices]
    kw = dict(nspin=1, smearing=res.get("smearing", "none"),
              width=res.get("width", 0.1))

    rho = res["rho"].detach().clone()
    bec = [m.detach().clone() for m in res["rho_ij_atoms"]]
    hist_out = []
    r1 = None
    best = float("inf")
    stalls = 0
    with torch.no_grad():
        for step in range(1, max_newton + 1):
            state = dict(system=system, nspin=1, rho=rho,
                         rho_ij_atoms=bec)
            r1 = scf_uspp(system, xc, max_iter=1, start_from=state,
                          diago_tol=diago_tol, etol=1e-300, rhotol=1e-300,
                          verbose=False, **kw)
            f_rho = r1["rho_out_spin"][0]
            f_bec = r1["rho_ij_atoms"]
            r_rho = (f_rho - rho).to(RDTYPE)
            r_bec = [(a - b).real.to(RDTYPE) for a, b in
                     zip(f_bec, bec, strict=True)]
            r_vec = _pack(r_rho, r_bec)
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

            # Jacobian frozen at the CURRENT x: the 1-iteration call's
            # orbitals diagonalize H[x], so they define χ̃ at x exactly
            jac_res = dict(r1)
            jac_res["rho"] = rho
            jac_res["rho_ij_atoms"] = bec
            cs = _ConvergedUSPP(jac_res, xc)
            dpsi_warm = [torch.zeros_like(c[:ns]) for c, ns in
                         zip(cs.c_win, cs.n_solve, strict=True)]

            # inner solve: δ = r + χ̃(K δ), Anderson-accelerated — the
            # same fixed-point shape as the adjoint, on the forward side
            d = r_vec.clone()
            prev_d = prev_g = None
            hist_dd, hist_dg = [], []
            for _it in range(1, max_inner + 1):
                d_rho, d_bec = _unpack(d, shape, n_pts, nbec)
                v_r = cs.k_hxc_grid(d_rho)
                d_ddd = cs.hvp_onecenter([m.to(torch.complex128)
                                          for m in d_bec])
                chi_rho, chi_bec = cs.apply_chi0(
                    v_r, d_ddd, dpsi_warm, cg_tol, cg_max_iter)
                g_vec = r_vec + _pack(chi_rho.to(RDTYPE),
                                      [m.real.to(RDTYPE) for m in chi_bec])
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

            d_rho, d_bec = _unpack(d, shape, n_pts, nbec)
            rho = rho + d_rho
            bec = [b + m.to(b.dtype) for b, m in zip(bec, d_bec, strict=True)]
    out = dict(r1)
    out["rho"] = rho
    out["rho_ij_atoms"] = bec
    out["newton"] = hist_out
    out["converged"] = min(hist_out) < tol
    return out
