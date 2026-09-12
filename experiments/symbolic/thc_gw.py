"""THC-GW step 1: the THC-factored χ₀, validated vs the direct build (GPU).

The GW screened Coulomb W = ε⁻¹v is built from χ₀. THC/ISDF factorizes the pair
densities so χ₀ never touches the O(N_pair·N_G²) form:

    ρ_vc(r) ≈ Σ_μ ζ_μ(r) Θ_vc(μ),   Θ_vc(μ) = φ_v*(r_μ) φ_c(r_μ)
    M_vc(G) = Σ_μ ζ_G(μ) Θ_vc(μ)                          (ζ_G = FFT(ζ_μ))
    χ₀_{GG'} = Σ_vc M(G)M*(G') freq = ζ_G · P · ζ_G†,   P_{μν} = Σ_vc Θ(μ)Θ*(ν) freq

so the polarizability collapses onto the N_μ×N_μ point matrix P (independent of the
plane-wave cutoff), and RPA/GW then runs in the μ-basis with the ζ-Coulomb W_μν.

This builds χ₀(q=0) both ways at Γ, checks the THC form converges to the direct one
as the ISDF rank grows, and times both — on the asus GPU (where the ISDF belongs).

Run (on asus):  uv run python experiments/symbolic/thc_gw.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def timeit(fn, reps=3):
    fn(); sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    nb = 48
    ecut_scr = 600
    inp = load_input(str(HERE / "si_abs.yaml"))
    system = build_system(inp)
    res = run_scf(inp, system=system, verbose=False)
    v_eff = res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]
    grid = system.grid
    shape = grid.shape
    nocc = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                         for i in range(len(system.species_of_atom))))) // 2
    npair = nocc * (nb - nocc)

    # orbitals on the GPU grid at Γ
    H, sph = build_hk(system, v_eff, np.zeros(3))
    ev, V = np.linalg.eigh(H)
    coeffs = torch.as_tensor(V[:, :nb].T, dtype=torch.complex128, device=DEV)
    flat_idx = sph.flat_idx.to(DEV)
    phi_r = orbitals_on_grid(coeffs, flat_idx, shape)                  # (nb, N_r) GPU
    E = torch.as_tensor(ev[:nb] / 27.211386, device=DEV)
    dE = (E[nocc:nb, None] - E[None, :nocc]).reshape(-1)               # (npair,)
    freq = (-2.0 / dE)                                                  # static χ₀

    # screening G-set (exclude G=0 → body/wing of χ₀(q=0))
    sph_g = build_gsphere(grid, ecut_scr, np.zeros(3))
    gflat = sph_g.flat_idx.to(DEV)
    n_g = sph_g.npw
    u = phi_r

    def chi0_direct():
        M = torch.empty((npair, n_g), dtype=torch.complex128, device=DEV)
        i = 0
        for v in range(nocc):
            prod = (u.conj()[v][None] * u[nocc:nb]).reshape(nb - nocc, *shape)
            M[i:i + (nb - nocc)] = r_to_g(prod).reshape(nb - nocc, -1)[:, gflat]
            i += nb - nocc
        return torch.einsum("p,pg,ph->gh", freq, M, M.conj())

    print(f"device={DEV} | nocc={nocc} nb={nb} pairs={npair} n_G={n_g} box={shape}")
    chi_ref = chi0_direct()
    ref_norm = chi_ref.norm().item()
    t_dir = timeit(chi0_direct, reps=3)

    print(f"\n{'N_μ':>5} {'N_μ/N_orb':>9} {'χ₀ rel err':>11} {'ISDF(1×)':>9} "
          f"{'THC χ₀':>9} {'direct χ₀':>10} {'speedup':>8}")
    for n_mu in (150, 250, 400, 600):
        t0 = time.perf_counter()
        pts = select_interpolation_points(phi_r, n_mu, sketch=64)  # cap 6GB GPU
        zeta = build_isdf(phi_r, pts)                                  # (N_r, n_mu)
        sync(); t_isdf = time.perf_counter() - t0
        n_eff = len(pts)
        zeta_G = r_to_g(zeta.transpose(0, 1).reshape(n_eff, *shape).to(torch.complex128)
                        ).reshape(n_eff, -1)[:, gflat].transpose(0, 1)   # (n_G, n_μ)
        phi_mu = phi_r[:, pts]
        Theta = (phi_mu.conj()[:nocc][:, None, :]
                 * phi_mu[nocc:nb][None, :, :]).reshape(npair, n_eff)

        def chi0_thc():
            P = torch.einsum("p,pm,pn->mn", freq, Theta, Theta.conj())
            return zeta_G @ P @ zeta_G.conj().transpose(0, 1)

        chi_thc = chi0_thc()
        err = (chi_thc - chi_ref).norm().item() / ref_norm
        t_thc = timeit(chi0_thc, reps=3)
        print(f"{n_eff:5d} {n_eff/nb:9.1f} {err:11.2e} {t_isdf*1e3:7.0f}m "
              f"{t_thc*1e3:8.2f}m {t_dir*1e3:9.2f}m {t_dir/t_thc:7.1f}×")

    print(f"\nTHC χ₀ = ζ_G·P·ζ_G† converges to the direct χ₀ as the ISDF rank grows;")
    print(f"its build is independent of N_G (P is N_μ×N_μ). ISDF is one-time per k")
    print(f"(amortized over all ω, and — next — over q via the multi-k pair extension).")
    print(f"Next THC-GW steps: (1) multi-k/q pairs (k,k+q), (2) ε=1−W_μν P, (3) Σ_c.")


if __name__ == "__main__":
    main()
