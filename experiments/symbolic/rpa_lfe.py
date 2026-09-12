"""RPA dielectric WITH local-field effects, via the Dyson screening ε=1−vχ₀.

Upgrades rpa_absorption.py (independent-particle head only) to the full microscopic
dielectric matrix and its inversion — the local-field effects (LFE):

  χ₀_{GG'}(q,ω) = (2/Ω)(1/N_k) Σ_k Σ_{v∈occ,c∈unocc} M_vc(G) M*_vc(G')
                   · [1/(ω−Δ+iη) − 1/(ω+Δ+iη)],   Δ = ε_{c,k+q} − ε_{v,k}
  M_vc(G) = ⟨u_vk| e^{-iGr} |u_{c,k+q}⟩ = [r_to_g(conj(u_vk)·u_{c,k+q})](G)
  ε_{GG'}(ω) = δ_{GG'} − v_G(q) χ₀_{GG'}(ω),   v_G(q) = 4π/|q+G|²   (a.u.)
  ε_M^{no-LFE}(ω) = ε_{00}(ω)          (head only)
  ε_M^{LFE}(ω)    = 1 / [ε(ω)⁻¹]_{00}   (invert the full matrix — the Dyson screening)

Done at a small finite q along a Cartesian axis (the optical limit q→0) so every
matrix element is an ordinary plane-wave overlap — no q→0 head/wing bookkeeping.
Full Γ-centered k-mesh (χ₀ is a BZ sum). Kernel is Hartree-only = RPA.

Run:  uv run python experiments/symbolic/rpa_lfe.py [input.yaml]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.api import build_system, run_scf
from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.fftbox import g_to_r, r_to_g
from gradwave.grids import build_gsphere, reciprocal_cell
from gradwave.inputs import load_input

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rpa_absorption import build_hk, kramers_kronig  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def full_mesh(n):
    """Γ-centered n×n×n k-mesh, folded to (−½,½]."""
    r = (np.arange(n) + 0.0) / n
    r = np.where(r > 0.5, r - 1.0, r)
    return np.array([[a, b, c] for a in r for b in r for c in r])


def lfe_gvectors(cell, q_frac, n_lfe, box_shape):
    """Smallest n_lfe reciprocal-lattice G (incl. 0) by |q+G|, + their box indices."""
    b = reciprocal_cell(cell)                       # rows b_i [Å⁻¹], 2π included
    q_cart = q_frac @ b
    ms = np.array([[i, j, k] for i in range(-3, 4) for j in range(-3, 4)
                   for k in range(-3, 4)])
    gcart = ms @ b
    order = np.argsort(np.sum((q_cart + gcart) ** 2, axis=1))
    sel = order[:n_lfe]
    miller = ms[sel]
    qg2_ang = np.sum((q_cart + gcart[sel]) ** 2, axis=1)      # Å⁻²
    n1, n2, n3 = box_shape
    flat = ((miller[:, 0] % n1) * (n2 * n3) + (miller[:, 1] % n2) * n3
            + (miller[:, 2] % n3))
    return miller, qg2_ang, flat


def u_of_r(coeffs, flat_idx, shape):
    """Cell-periodic u_n(r) on the FFT box for a band block (nb, npw) complex."""
    c = torch.as_tensor(coeffs, dtype=torch.complex128)
    return g_to_r(c, torch.as_tensor(flat_idx), shape).numpy()  # (nb, n1,n2,n3)


def chi0(system, v_eff, nocc, ncond, kmesh, q_frac, omega, eta, lfe_flat):
    """Accumulate χ₀_{GG'}(ω) over the mesh (a.u., prefactor applied at the end)."""
    grid = system.grid
    shape = grid.shape
    nlfe = len(lfe_flat)
    chi = np.zeros((len(omega), nlfe, nlfe), dtype=complex)
    w = omega / HARTREE_EV
    eta_au = eta / HARTREE_EV
    for kf in kmesh:
        Hk, sph_k = build_hk(system, v_eff, kf)
        Hq, sph_q = build_hk(system, v_eff, kf + q_frac)
        ev_k, V_k = np.linalg.eigh(Hk)
        ev_q, V_q = np.linalg.eigh(Hq)
        uv = u_of_r(V_k[:, :nocc].T, sph_k.flat_idx.cpu().numpy(), shape)   # (nocc,box)
        nc = min(ncond, V_q.shape[1] - nocc)
        uc = u_of_r(V_q[:, nocc:nocc + nc].T, sph_q.flat_idx.cpu().numpy(), shape)
        Ev = ev_k[:nocc] / HARTREE_EV
        Ec = ev_q[nocc:nocc + nc] / HARTREE_EV
        # M_vc(G) via FFT of conj(u_v)·u_c, gathered at the LFE G-vectors
        M = np.empty((nocc, nc, nlfe), dtype=complex)
        uc_t = torch.as_tensor(uc)
        for v in range(nocc):
            prod = torch.conj(torch.as_tensor(uv[v]))[None] * uc_t          # (nc,box)
            Mg = r_to_g(prod).reshape(nc, -1).numpy()                       # (nc, Nbox)
            M[v] = Mg[:, lfe_flat]
        Mf = M.reshape(nocc * nc, nlfe)
        dE = (Ec[None, :] - Ev[:, None]).ravel()                           # (nvc,)
        good = dE > 1e-5
        Mf, dE = Mf[good], dE[good]
        denom = 1.0 / (w[:, None] - dE[None, :] + 1j * eta_au) \
            - 1.0 / (w[:, None] + dE[None, :] + 1j * eta_au)               # (nω, nvc)
        chi += np.einsum("wv,vg,vh->wgh", denom, Mf, Mf.conj(), optimize=True)
    chi *= (2.0 / (grid.volume / BOHR_ANG ** 3)) / len(kmesh)              # g_s=2, /Ω, /Nk
    return chi


def run(yaml_path, nmesh=6, q0=0.02, ncond=24, n_lfe=27, eta=0.15,
        omega_max=16.0, nomega=320):
    inp = load_input(yaml_path)
    system = build_system(inp)
    t0 = time.perf_counter()
    res = run_scf(inp, system=system, verbose=False)
    v_eff = res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]
    nelec = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                          for i in range(len(system.species_of_atom)))))
    nocc = nelec // 2
    grid = system.grid
    q_frac = np.array([q0, 0.0, 0.0])
    _, qg2_ang, lfe_flat = lfe_gvectors(grid.cell, q_frac, n_lfe, grid.shape)
    vG = 4.0 * np.pi / (qg2_ang * BOHR_ANG ** 2)                # a.u., 4π/|q+G|²
    omega = np.linspace(0.05, omega_max, nomega)
    kmesh = full_mesh(nmesh)
    print(f"[SCF {time.perf_counter()-t0:.0f}s] nocc={nocc} mesh={nmesh}³={len(kmesh)}k "
          f"n_lfe={n_lfe} |q|={q0}")

    t1 = time.perf_counter()
    chi = chi0(system, v_eff, nocc, ncond, kmesh, q_frac, omega, eta, lfe_flat)
    print(f"χ₀ built in {time.perf_counter()-t1:.0f}s")

    eps_head = 1.0 - vG[0] * chi[:, 0, 0]                       # no local fields
    I = np.eye(len(vG))
    eps_M = np.empty(len(omega), dtype=complex)
    for w in range(len(omega)):
        eps = I - (vG[:, None] * chi[w])                       # ε_{GG'} = δ − v_G χ₀
        eps_M[w] = 1.0 / np.linalg.inv(eps)[0, 0]              # macroscopic w/ LFE
    return dict(omega=omega, eps_noLFE=eps_head, eps_LFE=eps_M,
                nocc=nocc, nk=len(kmesh))


def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "si_abs.yaml")
    out = run(yaml_path)
    om = out["omega"]
    # SAVE the computed spectra FIRST — χ₀ is expensive, never discard it on a
    # post-processing hiccup.
    np.savez(HERE / "si_lfe.npz", omega=om,
             eps2_noLFE=out["eps_noLFE"].imag, eps2_LFE=out["eps_LFE"].imag,
             eps1_noLFE=out["eps_noLFE"].real, eps1_LFE=out["eps_LFE"].real)

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # NumPy 2.x rename
    for tag, e in (("no-LFE (head only)", out["eps_noLFE"]),
                   ("with LFE (Dyson)", out["eps_LFE"])):
        e1_0 = 1.0 + (2 / np.pi) * trapz(e.imag[1:] / om[1:], om[1:])
        pk = om[np.argmax(e.imag)]
        print(f"  {tag:22s}: ε₁(0)≈{e1_0:.2f}  ε₂ peak {pk:.2f} eV  "
              f"max ε₂ {e.imag.max():.1f}")
    print("(LFE should lower ε₁(0) and the peak height toward experiment: Si ε₁(0)=11.9)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(om, out["eps_noLFE"].imag, label="ε₂  no local fields (IP/RPA)")
        ax.plot(om, out["eps_LFE"].imag, label="ε₂  with local fields (RPA)")
        ax.set_xlabel("ω (eV)"); ax.set_ylabel("ε₂(ω)")
        ax.set_title("Si — local-field effects on the absorption"); ax.legend()
        fig.tight_layout(); fig.savefig(HERE / "si_lfe.png", dpi=120)
        print(f"saved {HERE/'si_lfe.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
