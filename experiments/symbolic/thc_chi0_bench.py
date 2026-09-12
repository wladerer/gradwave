"""Does THC/ISDF speed up the GW χ₀ construction? Walltime vs size.

χ₀ construction is the GW bottleneck. Two ways to build χ₀(q,ω) at fixed q:

  G-basis (what gw.py does):
    M_vc(G) = FFT(conj u_v · u_c)(G)      → N_pair FFTs of the box
    χ₀_{GG'} = Σ_vc freq·M(G)M*(G')       → O(N_pair · N_G²)

  THC/ISDF (point basis):
    ζ, {r_μ} from ISDF (one-time)         → N_orb FFTs, no per-pair FFT
    Θ_vc(μ) = φ_v*(r_μ) φ_c(r_μ)          → point values, NO FFT
    P_{μν}  = Σ_vc freq·Θ(μ)Θ*(ν)         → O(N_pair · N_μ²)
    (RPA then works in the N_μ×N_μ point basis with the ζ-Coulomb W_μν)

THC wins on BOTH axes at scale: N_orb FFTs vs N_pair FFTs, and N_μ² vs N_G² in the
contraction — with N_μ ~ few×N_orb independent of the plane-wave cutoff N_G. We
sweep N_G (screening cutoff) at fixed bands and time each build.

Run:  uv run python experiments/symbolic/thc_chi0_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.api import build_system, run_scf
from gradwave.core.fftbox import r_to_g
from gradwave.grids import build_gsphere
from gradwave.inputs import load_input
from gradwave.postscf.isdf import build_isdf, orbitals_on_grid, select_interpolation_points

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rpa_absorption import build_hk  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def timeit(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    nb = 40                    # occ + unocc bands (pairs = nocc·(nb−nocc))
    n_mu = 140                 # ISDF rank (≈ few × N_orb; from thc_demo)
    inp = load_input(str(HERE / "si_abs.yaml"))
    system = build_system(inp)
    res = run_scf(inp, system=system, verbose=False)
    v_eff = res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]
    grid = system.grid
    shape = grid.shape
    nocc = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                         for i in range(len(system.species_of_atom))))) // 2
    npair = nocc * (nb - nocc)

    # orbitals at Γ (coeffs + real-space); one q for the χ₀ timing
    H, sph = build_hk(system, v_eff, np.zeros(3))
    ev, V = np.linalg.eigh(H)
    coeffs = torch.as_tensor(V[:, :nb].T, dtype=torch.complex128)
    phi_r = orbitals_on_grid(coeffs, sph.flat_idx, grid.shape)          # (nb, N_r)
    u = phi_r                                                           # same convention
    E = ev[:nb] / 27.211386
    dE = (E[nocc:nb, None] - E[None, :nocc]).ravel()                    # (npair,) c−v
    freq = -2.0 / dE                                                    # static χ₀ factor
    freq_t = torch.as_tensor(freq)

    # ---- THC one-time factorization (amortized over all q, ω) ----
    t_isdf = time.perf_counter()
    pts = select_interpolation_points(phi_r, n_mu)
    zeta = build_isdf(phi_r, pts)                                       # (N_r, n_mu)
    phi_mu = phi_r[:, pts]                                              # (nb, n_mu)
    t_isdf = time.perf_counter() - t_isdf
    n_mu_eff = len(pts)

    # THC pair-point amplitudes Θ_vc(μ) = φ_v*(r_μ) φ_c(r_μ)   (no FFT)
    Theta = (phi_mu.conj()[:nocc][:, None, :]
             * phi_mu[nocc:nb][None, :, :]).reshape(npair, n_mu_eff)

    def build_THC():
        # P_{μν}(0) = Σ_vc freq · Θ(μ) Θ*(ν)   — the point-basis polarizability
        return torch.einsum("p,pm,pn->mn", freq_t, Theta, Theta.conj())

    print(f"nocc={nocc} nb={nb} pairs={npair} | ISDF rank N_μ={n_mu_eff} "
          f"(one-time {t_isdf*1e3:.0f} ms) | box {shape}")
    print(f"{'N_G':>6} {'G-basis χ₀':>12} {'THC χ₀':>10} {'speedup':>8} "
          f"{'G FFTs':>7} {'THC FFTs':>9}")

    for ecut_scr in (150, 250, 400, 600, 900):
        sph_g = build_gsphere(grid, ecut_scr, np.zeros(3))
        gflat = sph_g.flat_idx
        n_g = sph_g.npw

        def build_Gbasis():
            # M_vc(G) via FFT of conj(u_v)·u_c, then χ₀ = Σ freq M M†
            M = torch.empty((npair, n_g), dtype=torch.complex128)
            idx = 0
            for v in range(nocc):
                prod = (u.conj()[v][None] * u[nocc:nb]).reshape(nb - nocc, *shape)
                Mg = r_to_g(prod).reshape(nb - nocc, -1)[:, gflat]
                M[idx:idx + (nb - nocc)] = Mg
                idx += nb - nocc
            return torch.einsum("p,pg,ph->gh", freq_t, M, M.conj())

        # cheap correctness tie: static macroscopic-ish trace agreement is not
        # meaningful across bases; we only benchmark the build cost here.
        tG = timeit(build_Gbasis, reps=2)
        tT = timeit(build_THC, reps=3)
        print(f"{n_g:6d} {tG*1e3:10.1f}m {tT*1e3:8.1f}m {tG/tT:7.1f}× "
              f"{npair:7d} {nb:9d}")

    print(f"\nG-basis: N_pair={npair} box-FFTs + O(N_pair·N_G²) contraction.")
    print(f"THC: {nb} orbital-FFTs (one-time) + O(N_pair·N_μ²), N_μ={n_mu_eff} "
          f"independent of N_G.")
    print("→ THC decouples the polarizability rank from the plane-wave cutoff: the")
    print("  win grows as (N_G/N_μ)² and it also removes the per-pair FFTs.")


if __name__ == "__main__":
    main()
