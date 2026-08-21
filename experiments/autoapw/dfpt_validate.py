"""Phase-2 FLAPW-DFPT validation: the augmented-basis chi_0 density response.

Restores the base rutile fixed point (dfpt_base.py checkpoint) CHEAPLY (one iterate at the
converged potentials — no fragile re-convergence), then validates the chi_0 machinery in
:mod:`gradwave.flapw.dfpt` against finite differences of the SAME frozen-potential secular
assembly:

  GATE A (chi_0 OPERATOR, fixed geometry): perturb the aspherical potential v_nsph by eps*dv
  (dS = 0 exactly — a potential change leaves the overlap untouched), assemble the sum-over-
  states wavefunction response, contract to drho_2M, and compare to a one-step finite
  difference of the density at the perturbed potential. Isolates the wavefunction-response +
  density-matrix calculus with no moving-boundary terms.

  GATE B (bare structural drho/du, frozen v_eff): displace the atoms (u -> u +/- delta) at
  FROZEN potentials (rigidly-moving muffin tins: species/v_nsph/warp unchanged), take dH/dS by
  finite difference of the secular assembly and the amplitude geometry term analytically, run
  the same chi_0 response, and compare drho_2M/du to a central finite difference of the density.
  Then bare dV_zz/du = (dV_zz/drho)(drho/du) + Phase-1 explicit boundary term (0 for Ti by
  symmetry) — the UNSCREENED gradient (no K_Hxc feedback), reported against the FD reference.

Run on asus (use_symmetry=False => full k-mesh, so the per-k response sum is the full-BZ
density directly, no star-unfolding of the response needed).
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from gradwave.flapw import dfpt
from gradwave.flapw.efg import (
    _valence_v,
    interstitial_l2_boundary,
    sphere_density_multipoles_bands,
)
from gradwave.flapw.scf import (
    _bands_amps,
    _lapw_multi_k,
    _multi_init_state,
    _multi_iterate,
    _multi_setup,
    _nsph_radial_integrals,
    _us_ext,
    _ylm_star,
)

PKL = sys.argv[1] if len(sys.argv) > 1 else "dfpt_base_state.pkl"
DELTA = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-4
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4     # Gate-A potential FD step

with open(PKL, "rb") as fh:
    BASE = pickle.load(fh)
CFG = BASE["config"]
A, C0, U0 = CFG["A"], CFG["C0"], CFG["U0"]
RADII, ECUT, LMAX, FP_LMAX, KK = CFG["RADII"], CFG["ECUT"], CFG["LMAX"], CFG["FP_LMAX"], CFG["KK"]


def atoms_for(u):
    return [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
            ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
            ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]


def dfrac_du():
    # d(fractional coords)/du for each atom (preserves P4_2/mnm)
    return np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]],
                    dtype=float)


def build_ctx_state(u):
    """ctx + a populated post-iterate state at geometry u, frozen at the base potentials."""
    ctx = _multi_setup(a_bohr=[A, A, C0], atoms=atoms_for(u), radii=RADII, ecut=ECUT, lmax=LMAX,
                       kmesh=(KK, KK, KK), smearing=0.0, fullpot=True, fullpot_lmax=FP_LMAX,
                       use_symmetry=False, kworkers=1, subspace_reuse=False,
                       shift_invert=False, verbose=False)
    st = _multi_init_state(ctx, {"__full_state__": BASE["state"]})
    _multi_iterate(ctx, st, 0, iters=1, tol=1e-12)     # populate kdata/occ/vmt/El/v_nsph/v_hart
    return ctx, st


def solve_inputs(ctx, st):
    """Reconstruct the exact (species, chan, v_nsph, nsph_int, warp) the iterate solved with,
    so H/S can be rebuilt at perturbed geometry/potential with return_HS."""
    from gradwave.flapw.lapw import radial_channels_all
    r, dx, r_np = ctx.r, ctx.dx, ctx.r_np
    species = {}
    for k in ctx.keys:
        R = ctx.R_by_key[k]
        v0 = float(st.v_by_key[k].numpy()[np.argmin(np.abs(r_np - R))])
        v0off = (v0 - st.v_i0_prev) if st.v_i0_prev is not None else 0.0
        species[k] = {"R": R, "v": st.vmt_by_key[k], "El": st.El_by_key[k], "v0off": v0off}
    chan = {k: radial_channels_all(ctx.lmax, sp["El"], r, dx, sp["v"], sp["R"])
            for k, sp in species.items()}
    nsph_int = _nsph_radial_integrals(st.v_nsph, species, ctx.lmax, r, dx, chan=chan)
    return species, chan, st.v_nsph, nsph_int, st.warp_state


def solve_k(ctx, species, chan, v_nsph, nsph_int, warp, atoms_cart, kf, nb_solve, want_HS):
    return _lapw_multi_k(kf, ctx.A, atoms_cart, species, ctx.lmax, ctx.ecut, ctx.r, ctx.dx,
                         nb_solve, v_nsph=v_nsph, chan=chan, lodat=None, nsph_int=nsph_int,
                         c_prev=None, warp=warp, shift_invert=False, return_HS=want_HS)


def rho2m_from_solve(ctx, res, occ, us_by_key, acart, nb_solve):
    """Full-mesh-consistent per-atom rho_2m for ONE k from a solve result (occupied bands)."""
    lset2 = [(0, 0)] + [(2, m) for m in range(-2, 3)]
    _ev, C = res[0], res[1]
    ks, abl_all, vol = res[3], res[4], res[5]
    npw = len(res[2])
    ylm_by_l = {lang: _ylm_star(lang, ks) for lang in range(ctx.lmax + 1)}
    out = {}
    for ai, k in enumerate(ctx.keys):
        phase = np.exp(1j * (ks @ np.asarray(acart[ai][0])))
        cp = C[:npw, :nb_solve] * phase[:, None]
        amps = _bands_amps(cp, ks, abl_all[ai], ctx.lmax, vol, ylm_by_l=ylm_by_l)
        out[k] = sphere_density_multipoles_bands(amps, list(occ), us_by_key[k],
                                                 ctx.rr_by_key[k], ctx.lmax, lset2)
    return out


def report(tag, an, fd):
    an = np.asarray(an)
    fd = np.asarray(fd)
    denom = max(np.abs(fd).max(), 1e-30)
    relmax = np.abs(an - fd).max() / denom
    print(f"  [{tag}] max|analytic-FD|/max|FD| = {relmax:.3e}  "
          f"(|FD|max={np.abs(fd).max():.3e})", flush=True)
    return relmax


def main():
    t0 = time.time()
    print(f"CONFIG ecut={ECUT} lmax={LMAX} fp_lmax={FP_LMAX} k={KK} delta={DELTA} eps={EPS}",
          flush=True)
    ctx, st = build_ctx_state(U0)
    species, chan, v_nsph, nsph_int, warp = solve_inputs(ctx, st)
    nb_solve = st.nb_solve
    nbands = ctx.nbands
    kfracs, kw = ctx.kfracs, ctx.kw
    acart = ctx.acart
    us_by_key = {k: _us_ext(st.El_by_key[k], ctx.lmax, ctx.r, ctx.dx, st.vmt_by_key[k],
                            ctx.R_by_key[k], None) for k in ctx.keys}
    lset2 = [(0, 0)] + [(2, m) for m in range(-2, 3)]

    # base full-basis solves (return_HS) per k
    base_res = [solve_k(ctx, species, chan, v_nsph, nsph_int, warp, acart, kf, nb_solve, True)
                for kf in kfracs]
    rank0 = base_res[0][1].shape[1]
    occ_full = np.zeros(rank0)
    occ_full[:nbands] = 2.0
    # sanity: reproduce base rho_2m / V_zz from these solves
    rho2m_base = {k: {lm: np.zeros(ctx.rr_by_key[k].shape, dtype=complex) for lm in lset2}
                  for k in ctx.keys}
    for res, w in zip(base_res, kw, strict=False):
        r2 = rho2m_from_solve(ctx, res, occ_full[:nb_solve], us_by_key, acart, nb_solve)
        for k in ctx.keys:
            for lm in lset2:
                rho2m_base[k][lm] += w * r2[k][lm]
    for k, lbl in (("a0", "Ti"), ("a2", "O")):
        rr = ctx.rr_by_key[k]
        drw = rr * ctx.dx
        R = ctx.R_by_key[k]
        vval = _valence_v(rho2m_base[k], rr, drw)
        vbc = interstitial_l2_boundary(st.v_hart, acart[ctx.keys.index(k)][0], R, ctx.A)
        vfull = {m: vval[m] + vbc[m] / R**2 for m in range(-2, 3)}
        _, vzz = dfpt.dvzz_du_jvp(vfull, {0: 0j, 1: 0j, 2: 0j})
        ref = BASE["efg"][lbl]["V_zz"]
        print(f"  base {lbl}: V_zz(reconstructed)={vzz:+.4f}  (ref {ref:+.4f})", flush=True)

    # ---- GATE A: chi_0 operator (fixed geometry, potential perturbation dv on v_nsph) ----
    print("=== GATE A: chi_0 operator (potential perturbation, dS=0) ===", flush=True)
    rng = np.random.default_rng(0)
    dv_nsph = {k: {lm: (rng.standard_normal(v.shape) * 0.1 * np.abs(v).max())
                   for lm, v in d.items()} for k, d in v_nsph.items()}
    v_nsph_p = {k: {lm: v_nsph[k][lm] + EPS * dv_nsph[k][lm] for lm in d}
                for k, d in v_nsph.items()}
    nsph_int_p = _nsph_radial_integrals(v_nsph_p, species, ctx.lmax, ctx.r, ctx.dx, chan=chan)

    drho_an = {k: {lm: np.zeros(ctx.rr_by_key[k].shape, dtype=complex) for lm in lset2}
               for k in ctx.keys}
    drho_fd = {k: {lm: np.zeros(ctx.rr_by_key[k].shape, dtype=complex) for lm in lset2}
               for k in ctx.keys}
    for res, kf, w in zip(base_res, kfracs, kw, strict=False):
        ev, C, mill, ks, abl_all, vol = res[0], res[1], res[2], res[3], res[4], res[5]
        Hm, S = res[6], res[7]
        res_p = solve_k(ctx, species, chan, v_nsph_p, nsph_int_p, warp, acart, kf, nb_solve, True)
        dH = (res_p[6] - Hm) / EPS
        dS = np.zeros_like(S)
        dC, occ_idx = dfpt.eig_response_occupied(ev, C, dH, dS, occ_full)
        npw = len(mill)
        for ai, k in enumerate(ctx.keys):
            tau = np.asarray(acart[ai][0])
            amps, damps = dfpt.sphere_amp_response_k(C[:npw], dC[:npw], occ_idx, ks,
                                                     abl_all[ai], ctx.lmax, vol, tau, None)
            occ_occ = occ_full[occ_idx]
            dd = dfpt.dmats_response(amps, damps, occ_occ, ctx.lmax)
            dr = dfpt.multipoles_from_dmats(dd, us_by_key[k], ctx.rr_by_key[k], ctx.lmax, lset2)
            for lm in lset2:
                drho_an[k][lm] += w * dr[lm]
        r2p = rho2m_from_solve(ctx, res_p, occ_full[:nb_solve], us_by_key, acart, nb_solve)
        r2b = rho2m_from_solve(ctx, res, occ_full[:nb_solve], us_by_key, acart, nb_solve)
        for k in ctx.keys:
            for lm in lset2:
                drho_fd[k][lm] += w * (r2p[k][lm] - r2b[k][lm]) / EPS
    for k, lbl in (("a0", "Ti"), ("a2", "O")):
        an = np.concatenate([drho_an[k][(2, m)] for m in range(-2, 3)])
        fd = np.concatenate([drho_fd[k][(2, m)] for m in range(-2, 3)])
        report(f"GateA {lbl} drho_2M", an, fd)

    gate_b(ctx, st, species, chan, v_nsph, nsph_int, base_res, occ_full, us_by_key,
           acart, kfracs, kw, nb_solve, nbands, rho2m_base)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)


def warp_for(v_grid, theta_i, nfft):
    """Rebuild the interstitial warp (FFT of (v_grid - <v_grid>_ThetaI) * ThetaI) from a FROZEN
    grid and a (possibly moved) interstitial indicator — the same zero-mean convention as
    scf._restore_full_state. Under a displacement at frozen v_eff, Theta_I moves with the spheres,
    so this recomputes the warp at the perturbed geometry from the unchanged grid values."""
    ti = theta_i.reshape(-1)
    vi_mean = float(v_grid.reshape(-1)[ti].mean())
    u = np.where(theta_i, v_grid - vi_mean, 0.0)
    return (np.fft.fftn(u) / nfft**3, nfft)


def theta_i_for(u):
    """Interstitial indicator Theta_I at geometry u (cheap: setup only, no SCF)."""
    ctxp = _multi_setup(a_bohr=[A, A, C0], atoms=atoms_for(u), radii=RADII, ecut=ECUT, lmax=LMAX,
                        kmesh=(KK, KK, KK), smearing=0.0, fullpot=True, fullpot_lmax=FP_LMAX,
                        use_symmetry=False, kworkers=1, subspace_reuse=False, shift_invert=False)
    return ctxp.theta_i


def gate_b(ctx, st, species, chan, v_nsph, nsph_int, base_res, occ_full, us_by_key,
           acart, kfracs, kw, nb_solve, nbands, rho2m_base):
    """GATE B: bare structural drho_2M/du (frozen v_eff) via chi_0, vs central FD; then the
    UNSCREENED dV_zz/du = (dV_zz/drho)(drho/du) + Phase-1 boundary term."""
    print("=== GATE B: bare structural drho_2M/du (frozen v_eff) ===", flush=True)
    from gradwave.flapw.coulomb import cell_matrix
    A_ang = cell_matrix([A, A, C0]) if not hasattr(ctx.A, "shape") else ctx.A
    d = DELTA
    v_grid = st.v_grid_prev
    nfft = ctx.nfft
    dfrac = dfrac_du()
    dtau_du = dfrac @ A_ang                      # (natom, 3) Angstrom/u
    acart_p = [(np.asarray(f) @ A_ang, key)
               for (f, _s), key in zip(atoms_for(U0 + d), ctx.keys, strict=True)]
    acart_m = [(np.asarray(f) @ A_ang, key)
               for (f, _s), key in zip(atoms_for(U0 - d), ctx.keys, strict=True)]
    warp_p = warp_for(v_grid, theta_i_for(U0 + d), nfft)
    warp_m = warp_for(v_grid, theta_i_for(U0 - d), nfft)

    lset2 = [(0, 0)] + [(2, m) for m in range(-2, 3)]
    drho_an = {k: {lm: np.zeros(ctx.rr_by_key[k].shape, dtype=complex) for lm in lset2}
               for k in ctx.keys}
    drho_fd = {k: {lm: np.zeros(ctx.rr_by_key[k].shape, dtype=complex) for lm in lset2}
               for k in ctx.keys}
    for res, kf, w in zip(base_res, kfracs, kw, strict=False):
        ev, C, mill, ks, abl_all, vol = res[0], res[1], res[2], res[3], res[4], res[5]
        rp = solve_k(ctx, species, chan, v_nsph, nsph_int, warp_p, acart_p, kf, nb_solve, True)
        rm = solve_k(ctx, species, chan, v_nsph, nsph_int, warp_m, acart_m, kf, nb_solve, True)
        dH = (rp[6] - rm[6]) / (2 * d)
        dS = (rp[7] - rm[7]) / (2 * d)
        dC, occ_idx = dfpt.eig_response_occupied(ev, C, dH, dS, occ_full)
        npw = len(mill)
        for ai, k in enumerate(ctx.keys):
            tau = np.asarray(acart[ai][0])
            amps, damps = dfpt.sphere_amp_response_k(C[:npw], dC[:npw], occ_idx, ks,
                                                     abl_all[ai], ctx.lmax, vol, tau,
                                                     dtau_du[ai])
            dd = dfpt.dmats_response(amps, damps, occ_full[occ_idx], ctx.lmax)
            dr = dfpt.multipoles_from_dmats(dd, us_by_key[k], ctx.rr_by_key[k], ctx.lmax, lset2)
            for lm in lset2:
                drho_an[k][lm] += w * dr[lm]
        r2p = rho2m_from_solve(ctx, rp, occ_full[:nb_solve], us_by_key, acart_p, nb_solve)
        r2m = rho2m_from_solve(ctx, rm, occ_full[:nb_solve], us_by_key, acart_m, nb_solve)
        for k in ctx.keys:
            for lm in lset2:
                drho_fd[k][lm] += w * (r2p[k][lm] - r2m[k][lm]) / (2 * d)
    for k, lbl in (("a0", "Ti"), ("a2", "O")):
        an = np.concatenate([drho_an[k][(2, m)] for m in range(-2, 3)])
        fd = np.concatenate([drho_fd[k][(2, m)] for m in range(-2, 3)])
        report(f"GateB {lbl} drho_2M/du", an, fd)

    # ---- bare dV_zz/du = valence (chi_0) + boundary (Phase-1 explicit, frozen v_hart) ----
    print("=== bare dV_zz/du (unscreened) vs FD reference ===", flush=True)
    ref = {"a0": 2510.0, "a2": 1485.0}
    for k, lbl in (("a0", "Ti"), ("a2", "O")):
        rr = ctx.rr_by_key[k]
        drw = rr * ctx.dx
        R = ctx.R_by_key[k]
        ai = ctx.keys.index(k)
        tau = np.asarray(acart[ai][0])
        dtau = dtau_du[ai]
        # base coefficients (valence + boundary)
        vval = _valence_v(rho2m_base[k], rr, drw)
        vbc = interstitial_l2_boundary(st.v_hart, tau, R, ctx.A)
        vbase = {m: vval[m] + vbc[m] / R**2 for m in range(-2, 3)}
        # chi_0 valence tangent
        dvval = dfpt.valence_v_coeffs(drho_an[k], rr, drw)
        # Phase-1 explicit boundary tangent (frozen v_hart): central FD of the surface projection
        bcp = interstitial_l2_boundary(st.v_hart, tau + d * dtau, R, ctx.A)
        bcm = interstitial_l2_boundary(st.v_hart, tau - d * dtau, R, ctx.A)
        dvbc = {m: (bcp[m] - bcm[m]) / (2 * d) / R**2 for m in range(-2, 3)}
        dv_chi0 = {i: dvval[i] for i in range(3)}                        # v_m, m=0,1,2
        dv_expl = {i: dvbc[i] for i in range(3)}
        dv_tot = {i: dv_chi0[i] + dv_expl[i] for i in range(3)}
        dvzz_chi0, _ = dfpt.dvzz_du_jvp(vbase, dv_chi0)
        dvzz_expl, _ = dfpt.dvzz_du_jvp(vbase, dv_expl)
        dvzz_bare, vzz0 = dfpt.dvzz_du_jvp(vbase, dv_tot)
        print(f"    [{lbl}] |dvval|={ {m: round(abs(dvval[m]),2) for m in range(-2,3)} } "
              f"|dvbc/R2|={ {m: round(abs(dvbc[m]),2) for m in range(-2,3)} }", flush=True)
        print(f"  {lbl}: V_zz={vzz0:+.3f}  dV_zz/du bare={dvzz_bare:+.1f} "
              f"(chi0={dvzz_chi0:+.1f} explicit={dvzz_expl:+.1f})  "
              f"FD_ref(screened)={ref[k]:+.0f}  bare/ref={dvzz_bare/ref[k]:+.2f}", flush=True)


if __name__ == "__main__":
    main()
