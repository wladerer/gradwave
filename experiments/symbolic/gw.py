"""G0W0 (one-shot GW) with a Godby-Needs plasmon-pole model — the Si direct gap.

The full-distance consumer: on top of the RPA χ₀/W machinery (rpa_lfe.py) it builds
the screened Coulomb W = ε⁻¹v over the q-mesh, the self-energy Σ = iGW split into
bare exchange Σ_x + screened correlation Σ_c (PPA), and solves the quasiparticle
equation for the band edges at Γ:

    ε^QP_n = ε^DFT_n + Z_n [ ⟨n|Σ_x + Σ_c(ε^DFT_n)|n⟩ − ⟨n|v_xc|n⟩ ],
    Z_n = 1 / (1 − ∂Σ_c/∂ω|_{ε^DFT_n}).

Godby-Needs PPA: ε⁻¹_{GG'}(q,ω) − δ ≈ Ω²/(ω²−ω̃²), the two params (ω̃, Ω²) fixed from
ε⁻¹ at ω=0 and ω=iE₀. Mesh-closed Γ-centered q-grid; q=0 skipped in the interaction
sum with the standard spherical-average exchange-divergence correction (under-screens
the head → gap opening improves with mesh; a prototype, not a converged number).

Run:  uv run python experiments/symbolic/gw.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.api import build_system, run_scf
from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.energies.hartree import hartree_potential_g
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.fftbox import g_to_r, g_to_r_box, r_to_g
from gradwave.grids import build_gsphere, reciprocal_cell
from gradwave.inputs import load_input

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rpa_absorption import build_hk  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)
HA = HARTREE_EV
E0_AU = 1.0  # Godby-Needs imaginary reference frequency (1 Hartree)
ETA_AU = 0.05 / HA


def mesh_and_maps(n):
    """Γ-centered n³ mesh (fractional, folded), plus add/sub index maps (mod 1)."""
    vals = (np.arange(n)) / n
    vals = np.where(vals > 0.5, vals - 1.0, vals)
    pts = np.array([[a, b, c] for a in vals for b in vals for c in vals])
    key = {tuple(np.round((p % 1.0), 6)): i for i, p in enumerate(pts)}

    def idx(p):
        return key[tuple(np.round((p % 1.0), 6))]

    sub = np.array([[idx(pts[a] - pts[b]) for b in range(len(pts))]
                    for a in range(len(pts))])
    return pts, sub


def gw_gset(cell, n_gw, box_shape):
    """Smallest n_gw reciprocal G (incl 0) by |G|; Cartesian + box flat index."""
    b = reciprocal_cell(cell)
    ms = np.array([[i, j, k] for i in range(-3, 4) for j in range(-3, 4)
                   for k in range(-3, 4)])
    gc = ms @ b
    order = np.argsort(np.sum(gc ** 2, axis=1))[:n_gw]
    miller, gcart = ms[order], gc[order]
    n1, n2, n3 = box_shape
    flat = ((miller[:, 0] % n1) * (n2 * n3) + (miller[:, 1] % n2) * n3
            + (miller[:, 2] % n3))
    return miller, gcart, flat


def main():
    n = 6               # q-mesh (denser — convergence check vs the 4³ run)
    nb = 64             # bands kept for the χ₀ / Σ sums
    n_gw = 51           # screening G-vectors
    yaml_path = str(HERE / "si_abs.yaml")

    inp = load_input(yaml_path)
    system = build_system(inp)
    t0 = time.perf_counter()
    res = run_scf(inp, system=system, verbose=False)
    v_eff = (res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0])
    grid = system.grid
    Omega = grid.volume / BOHR_ANG ** 3            # bohr³
    nelec = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                          for i in range(len(system.species_of_atom)))))
    nocc = nelec // 2
    shape = grid.shape
    print(f"[SCF {time.perf_counter()-t0:.0f}s] nocc={nocc} mesh={n}³ nb={nb} n_gw={n_gw}")

    # v_xc(r) = v_eff − v_H − v_loc  (eV, real space)
    rho_g = r_to_g(res.rho.to(torch.complex128))
    v_h = g_to_r_box(hartree_potential_g(rho_g, grid.g2), real=True)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)
    v_loc = g_to_r_box(vloc_g, real=True)
    v_xc = (v_eff - v_h - v_loc).cpu().numpy()      # (n1,n2,n3) eV

    pts, sub = mesh_and_maps(n)
    nk = len(pts)
    gmiller, gcart, gflat = gw_gset(grid.cell, n_gw, shape)
    gflat_t = torch.as_tensor(gflat)

    # precompute eigenpairs + real-space u_n(r) + coeffs/kpg (for the q→0 head)
    print("diagonalizing mesh + building u(r)...")
    EV, U, VC, KPG = [], [], [], []
    for p in pts:
        H, sph = build_hk(system, v_eff, p)
        ev, V = np.linalg.eigh(H)
        EV.append(ev[:nb])
        c = torch.as_tensor(V[:, :nb].T, dtype=torch.complex128)   # (nb, npw)
        U.append(g_to_r(c, sph.flat_idx, shape).reshape(nb, -1))    # (nb, Nbox)
        VC.append(V[:, :nb])                                        # (npw, nb) coeffs
        KPG.append(sph.kpg.cpu().numpy() * BOHR_ANG)                # (npw,3) bohr⁻¹
    EV = np.array(EV)                                # (nk, nb) eV
    mu = 0.5 * (EV[:, nocc - 1].max() + EV[:, nocc].min())          # rough Fermi level

    # q=0 microcell radius (bohr⁻¹) and momentum matrix ⟨a|(k+G)_x|b⟩ (a.u.)
    Vq = (2 * np.pi) ** 3 / (Omega * nk)
    qc = (3 * Vq / (4 * np.pi)) ** (1.0 / 3.0)
    v0_avg = (4 * np.pi) ** 2 * qc / Vq

    def momentum(ik, bra, ket, alpha=0):
        Vk = VC[ik]
        return np.einsum("ga,g,gb->ab", np.conj(Vk[:, bra]), KPG[ik][:, alpha],
                         Vk[:, ket], optimize=True)

    def Bmat(uk, ukq):
        """B_{ab}(G) = [r_to_g(conj(u_ak)·u_bkq)](G) at the GW G-set → (na,nb_,nG)."""
        prod = torch.conj(uk)[:, None, :] * ukq[None, :, :]         # (na,nb_,Nbox)
        Mg = r_to_g(prod.reshape(uk.shape[0], ukq.shape[0], *shape))
        return Mg.reshape(uk.shape[0], ukq.shape[0], -1)[..., gflat_t].numpy()

    # ---- W(q): χ₀ at ω=0 and iE₀ → ε → ε⁻¹ → PPA (ω̃, Ω²) for each q≠0 ----
    print("building W(q) + plasmon-pole...")
    t1 = time.perf_counter()
    qg2 = np.empty((nk, n_gw))
    for iq, q in enumerate(pts):
        qg2[iq] = np.sum((q @ reciprocal_cell(grid.cell) + gcart) ** 2, axis=1) \
            * BOHR_ANG ** 2                          # |q+G|² in bohr⁻²
    vG = np.where(qg2 > 1e-8, 4.0 * np.pi / np.maximum(qg2, 1e-12), 0.0)  # 4π/|q+G|²
    iq0 = int(np.argmin(np.abs(pts).sum(axis=1)))    # Γ / q=0 index
    vG[iq0, 0] = 4.0 * np.pi / qc ** 2               # head Coulomb (spherical microcell)

    wtil = np.zeros((nk, n_gw, n_gw))
    Omega2 = np.zeros((nk, n_gw, n_gw))
    eps_static_head = np.zeros(nk)
    for iq in range(nk):
        is_q0 = iq == iq0
        chi0 = np.zeros((n_gw, n_gw), dtype=complex)
        chiE = np.zeros((n_gw, n_gw), dtype=complex)
        for ik in range(nk):
            ikq = sub[ik, iq]                        # k − q  (mesh-closed)
            B = Bmat(U[ik][:nb], U[ikq][:nocc])      # (nb, nocc, nG): c@k, v@(k−q)
            Ec = EV[ik][:nb] / HA
            Ev = EV[ikq][:nocc] / HA
            dE = Ec[:, None] - Ev[None, :]           # (nb, nocc)
            occ_c = EV[ik][:nb] < mu
            good = (dE > 1e-5) & (~occ_c[:, None])   # c empty, v occ, positive gap
            if is_q0:                                # q→0 head: G=0 ← k·p momentum
                p = momentum(ik, np.arange(nb), np.arange(nocc), 0)  # (nb,nocc)
                B[:, :, 0] = qc * p / np.where(np.abs(dE) > 1e-6, dE, 1.0)
            Bf = B[good]                             # (npair, nG)
            d = dE[good]
            f0 = -2.0 / d
            fE = -2.0 * d / (E0_AU ** 2 + d ** 2)
            chi0 += np.einsum("p,pg,ph->gh", f0, Bf, Bf.conj(), optimize=True)
            chiE += np.einsum("p,pg,ph->gh", fE, Bf, Bf.conj(), optimize=True)
        if is_q0:                                    # decouple head from body (the
            chi0[0, 1:] = chi0[1:, 0] = 0.0          # head-body wing sign is subtle;
            chiE[0, 1:] = chiE[1:, 0] = 0.0          # head-only LFE avoids overshoot)
        pref = 2.0 / Omega / nk
        chi0 *= pref
        chiE *= pref
        I = np.eye(n_gw)
        einv0 = np.linalg.inv(I - vG[iq][:, None] * chi0)
        einvE = np.linalg.inv(I - vG[iq][:, None] * chiE)
        eps_static_head[iq] = (1.0 / einv0[0, 0]).real
        D0 = einv0 - I
        DE = einvE - I
        with np.errstate(divide="ignore", invalid="ignore"):
            r = DE / D0
            w2 = E0_AU ** 2 * r / (1.0 - r)          # ω̃²
        ok = np.isfinite(w2) & (w2.real > 1e-6) & (np.abs(D0) > 1e-8)
        wt = np.sqrt(np.where(ok, w2.real, 1.0))
        wtil[iq] = wt
        Omega2[iq] = np.where(ok, -(D0.real) * wt ** 2, 0.0)
    print(f"  W built in {time.perf_counter()-t1:.0f}s;  "
          f"macroscopic ε(q→0 head) ≈ {eps_static_head[iq0]:.1f} (Si ε∞≈11.9);  "
          f"head plasmon ω̃(q→0) = {wtil[iq0,0,0]*HA:.1f} eV (Si ~16.7)")

    # ---- self-energy at Γ for VBM (nocc−1) and CBM (nocc) ----
    iG = iq0                                         # Γ index
    states = {"VBM": nocc - 1, "CBM": nocc}

    print("self-energy at Γ...")
    out = {}
    for name, n_st in states.items():
        Edft = EV[iG][n_st] / HA
        sx = 0.0
        sc = np.zeros(2, dtype=complex)              # Σ_c at E and E+δ (for Z)
        dEz = 0.02
        for iq in range(nk):
            ikq = sub[iG, iq]                        # Γ − q
            B = Bmat(U[iG][n_st:n_st + 1], U[ikq][:nb])[0]   # (nb, nG): n@Γ, m@(Γ−q)
            Em = EV[ikq][:nb] / HA
            fm = (EV[ikq][:nb] < mu).astype(float)
            if iq == iq0:                            # q→0 head: G=0 ← k·p (skip m=n)
                p = momentum(iG, [n_st], np.arange(nb), 0)[0]       # (nb,)
                den = Edft - Em
                B[:, 0] = np.where(np.abs(den) > 1e-4, qc * p / den, 0.0)
            # exchange: −Σ_{v occ} Σ_G v(q+G)|B|²/(Ω N_q)
            for v in np.where(fm > 0.5)[0]:
                sx -= np.sum(vG[iq] * np.abs(B[v]) ** 2) / Omega / nk
            # correlation (PPA): Σ_m Σ_GG' B(G)B*(G') Ω²/(2ω̃) v(q+G')/(Ω N_q) × poles
            wt = wtil[iq]
            Om2 = Omega2[iq]
            amp = Om2 / (2.0 * wt) * vG[iq][None, :] / Omega / nk    # (G,G')
            for j, Eev in enumerate((Edft, Edft + dEz)):
                res_pole = (1 - fm)[:, None, None] / (
                    Eev - Em[:, None, None] - wt[None] + 1j * ETA_AU) \
                    + fm[:, None, None] / (
                    Eev - Em[:, None, None] + wt[None] - 1j * ETA_AU)
                BB = B[:, :, None] * B.conj()[:, None, :]            # (m,G,G')
                sc[j] += np.sum(BB * amp[None] * res_pole)
        sx += (-v0_avg / Omega / nk) if EV[iG][n_st] < mu else 0.0   # q=0 self-exch
        vxc_nn = float(np.mean(np.abs(U[iG][n_st].cpu().numpy()) ** 2
                               * v_xc.reshape(-1)))                  # ⟨n|v_xc|n⟩ eV
        sx_eV = sx.real * HA
        sc_eV = sc[0].real * HA
        dsc = (sc[1].real - sc[0].real) / dEz                        # ∂Σ_c/∂ω (a.u./a.u.)
        Z = 1.0 / (1.0 - dsc)
        qp = EV[iG][n_st] + Z * (sx_eV + sc_eV - vxc_nn)
        out[name] = dict(edft=EV[iG][n_st], sx=sx_eV, sc=sc_eV, vxc=vxc_nn, Z=Z, qp=qp)
        print(f"  {name}: ε_DFT={EV[iG][n_st]:+.2f}  Σx={sx_eV:+.2f}  Σc={sc_eV:+.2f}  "
              f"vxc={vxc_nn:+.2f}  Z={Z:.2f}  →  ε_QP={qp:+.2f} eV")

    gdft = out["CBM"]["edft"] - out["VBM"]["edft"]
    gqp = out["CBM"]["qp"] - out["VBM"]["qp"]
    print(f"\nSi direct gap at Γ:  DFT(PBE) = {gdft:.2f} eV   →   G0W0 = {gqp:.2f} eV")
    print(f"(PBE Γ direct ~2.5–2.6, G0W0 ~3.2–3.4 eV expt; this mesh is coarse)")
    np.savez(HERE / "gw_si.npz", **{k: np.array(list(v.values()))
                                    for k, v in out.items()},
             gap_dft=gdft, gap_qp=gqp)


if __name__ == "__main__":
    main()
