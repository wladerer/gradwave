"""Independent-particle / RPA (no local fields) optical absorption ε(ω).

A CONSUMER for the full-spectrum block-diagonalization: at each IBZ k it needs the
eigenvalues AND eigenvectors of H(k) for all valence + many conduction bands, then
forms momentum matrix elements and sums interband transitions (Adler–Wiser, q→0):

  ε₂(ω) = (4π²/Ω) g_s Σ_k w_k Σ_{v∈occ,c∈unocc} |p_cv|²/Δ_cv² · L_η(ω − Δ_cv)

p_cv = ⟨ψ_ck| (k+G) |ψ_vk⟩ (velocity/momentum gauge; the nonlocal-pseudopotential
commutator [V_nl, r] is OMITTED — exact for the local part, a standard first
approximation). ε₁ from Kramers–Kronig; refractive n, κ; absorption α(ω)=2ωκ/c.

Two eigensolver backends select how H(k) is diagonalized:
  'dense' : numpy eigh (full spectrum)
  'sym'   : symbasis.SymBasis  (symmetry-blocked full spectrum; exact at high-sym k)

Run:  uv run python experiments/symbolic/rpa_absorption.py <input.yaml>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.api import build_system, run_scf
from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.hamiltonian import HamiltonianK, projectors
from gradwave.grids import build_gsphere
from gradwave.inputs import load_input
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.postscf.irreps import little_group
from gradwave.symmetry import find_spacegroup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_walltime import symmetrize  # noqa: E402
from symbasis import SymBasis  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)
C_AU = 137.035999  # speed of light in atomic units


def ibz_kpoints(system, res):
    """(k_frac list, normalized weights) for the SCF IBZ mesh."""
    kfracs = np.array([np.asarray(s.k_frac, float) for s in system.spheres])
    for attr in ("kweights", "k_weights", "weights"):
        w = getattr(system, attr, None)
        if w is None:
            w = getattr(res, attr, None)
        if w is not None:
            w = np.asarray(w, float)
            break
    else:
        w = np.ones(len(kfracs))
    return kfracs, w / w.sum()


def build_hk(system, v_eff, kfrac):
    grid = system.grid
    sph = build_gsphere(grid, system.ecut, np.asarray(kfrac, float))
    beta_ls, dij_sp = species_projector_tables(system.upfs, None)
    pd = projector_data_at_k(sph, system.species_of_atom, system.upfs,
                             beta_ls, dij_sp, grid.volume, None)
    p = projectors(pd, system.positions)
    h = HamiltonianK(sph, grid.shape, v_eff, pd, p)
    npw = sph.npw
    H = h.apply(torch.eye(npw, dtype=torch.complex128)).transpose(0, 1).numpy()
    return 0.5 * (H + H.conj().T), sph


def eps2_at_k(evals_eV, V, kpg_ang, nocc, ncond, omega_eV, eta_eV, omega_max):
    """Interband ε₂ contribution at one k (velocity gauge), a.u. internally."""
    E = evals_eV / HARTREE_EV
    kpg_au = kpg_ang * BOHR_ANG                       # Å⁻¹ → bohr⁻¹
    nb = min(nocc + ncond, V.shape[1])
    Vb = V[:, :nb]
    # momentum matrix P[a][n,m] = ⟨n|(k+G)_a|m⟩
    P = np.stack([Vb.conj().T @ (kpg_au[:, a:a + 1] * Vb) for a in range(3)])
    Ev, Ec = E[:nocc], E[nocc:nb]
    dE = Ec[None, :] - Ev[:, None]                    # (nocc, ncond)
    p2 = (np.abs(P[:, nocc:nb, :nocc]) ** 2).sum(0).T / 3.0   # (nocc,ncond) isotropic
    mask = (dE > 1e-4) & (dE < omega_max / HARTREE_EV + 0.3)
    dEs, oscs = dE[mask], (p2[mask] / dE[mask] ** 2)
    w = omega_eV / HARTREE_EV
    eta = eta_eV / HARTREE_EV
    lor = (eta / np.pi) / ((w[:, None] - dEs[None, :]) ** 2 + eta ** 2)
    return lor @ oscs                                 # (nω,)


def kramers_kronig(omega_eV, eps2):
    """ε₁(ω) = 1 + (2/π) P∫ ω' ε₂/(ω'²−ω²) dω'  (discrete, principal value)."""
    w = omega_eV
    dw = w[1] - w[0]
    eps1 = np.ones_like(w)
    for i in range(len(w)):
        d = w ** 2 - w[i] ** 2
        d[i] = np.inf
        eps1[i] += (2 / np.pi) * np.sum(w * eps2 / d) * dw
    return eps1


def run(yaml_path, backend="dense", ncond=40, eta=0.15,
        omega_max=16.0, nomega=400):
    inp = load_input(yaml_path)
    system = build_system(inp)
    t0 = time.perf_counter()
    res = run_scf(inp, system=system, verbose=False)
    v_eff = res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]
    t_scf = time.perf_counter() - t0

    nelec = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                          for i in range(len(system.species_of_atom)))))
    nocc = nelec // 2
    kfracs, weights = ibz_kpoints(system, res)
    grid = system.grid
    sg = find_spacegroup(grid.cell,
                         system.positions.cpu().numpy() @ np.linalg.inv(grid.cell),
                         system.species_of_atom)
    omega = np.linspace(0.05, omega_max, nomega)

    print(f"[SCF {t_scf:.0f}s]  nelec={nelec} nocc={nocc}  IBZ k-points={len(kfracs)}  "
          f"backend={backend}")

    eps2 = np.zeros_like(omega)
    t_diag = 0.0
    sb_cache = {}
    for kf, wk in zip(kfracs, weights):
        H, sph = build_hk(system, v_eff, kf)
        pref = (4 * np.pi ** 2 / (grid.volume / BOHR_ANG ** 3)) * 2.0 * wk  # g_s=2
        td = time.perf_counter()
        if backend == "sym":
            ops = little_group(kf, sg, grid.cell)
            Ws = [np.asarray(op["W"], int) for op in ops]
            Hs = symmetrize(H, sph.miller.cpu().numpy(), ops, kf)
            key = round(float(np.abs(kf).sum()), 6)
            sb = sb_cache.get(sph.npw) or SymBasis(sph.miller.cpu().numpy(), ops, Ws, kf)
            ev, V = sb.block_eig(Hs)
        else:
            ev, V = np.linalg.eigh(H)
        t_diag += time.perf_counter() - td
        kpg = sph.kpg.cpu().numpy()
        eps2 += pref * eps2_at_k(ev, V, kpg, nocc, ncond, omega, eta, omega_max)

    eps1 = kramers_kronig(omega, eps2)
    absmod = np.sqrt(eps1 ** 2 + eps2 ** 2)
    kappa = np.sqrt(np.maximum((absmod - eps1) / 2, 0))
    alpha = 2 * (omega / HARTREE_EV) * kappa / C_AU        # 1/bohr
    alpha_cm = alpha / (BOHR_ANG * 1e-8)                   # 1/cm
    return dict(omega=omega, eps1=eps1, eps2=eps2, alpha_cm=alpha_cm,
                eps1_0=float(eps1[0]), t_diag=t_diag, nocc=nocc,
                nk=len(kfracs), system=system, res=res, sg=sg, v_eff=v_eff)


def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "si_abs.yaml")
    out = run(yaml_path, backend="dense")
    print(f"static ε₁(0) ≈ {out['eps1_0']:.2f}  (Si expt ~11.9; PBE overestimates)")
    peak = out["omega"][np.argmax(out["eps2"])]
    print(f"ε₂ peak at {peak:.2f} eV, max ε₂ = {out['eps2'].max():.1f}; "
          f"diag time {out['t_diag']:.1f}s over {out['nk']} k (dense)")

    # cross-check: symmetry backend reproduces the dense spectrum (at Γ + full run)
    outs = run(yaml_path, backend="sym")
    err = float(np.abs(out["eps2"] - outs["eps2"]).max())
    print(f"\nsym backend: ε₂ matches dense to {err:.2e}; "
          f"diag time {outs['t_diag']:.1f}s (sym) vs {out['t_diag']:.1f}s (dense)")

    # save spectrum + plot
    np.savez(HERE / "si_absorption.npz", omega=out["omega"], eps1=out["eps1"],
             eps2=out["eps2"], alpha_cm=out["alpha_cm"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(out["omega"], out["eps1"], label="ε₁")
        ax[0].plot(out["omega"], out["eps2"], label="ε₂")
        ax[0].axhline(0, lw=0.5, color="k"); ax[0].set_xlabel("ω (eV)")
        ax[0].set_ylabel("ε(ω)"); ax[0].legend(); ax[0].set_title("Si dielectric (IP/RPA)")
        ax[1].plot(out["omega"], out["alpha_cm"] / 1e6)
        ax[1].set_xlabel("ω (eV)"); ax[1].set_ylabel("α (10⁶ cm⁻¹)")
        ax[1].set_title("Si absorption")
        fig.tight_layout(); fig.savefig(HERE / "si_absorption.png", dpi=110)
        print(f"saved {HERE/'si_absorption.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
