"""Tensor hypercontraction (ISDF) on the GW pair densities — Si at Γ.

gradwave's postscf/isdf.py IS tensor hypercontraction (Lu & Ying 2015): it
factorizes every orbital pair product

    ρ_ij(r) = ψ_i*(r) ψ_j(r)  ≈  Σ_μ ζ_μ(r) ψ_i*(r_μ) ψ_j(r_μ)

onto O(N) interpolation points {r_μ} chosen by pivoted QR. The GW/RPA χ₀ is built
from exactly these pair densities (occupied × unoccupied), so THC collapses the
O(N_occ·N_unocc) pair index onto N_μ ~ O(N) points — the scaling win for GW.

This runs gradwave's own ISDF on Si occ+unocc orbitals at Γ and reports the rank
N_μ and the pair-density reconstruction accuracy vs the exact pairs — i.e. that THC
reproduces the χ₀ ingredient at low rank.

Run:  uv run python experiments/symbolic/thc_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.api import build_system, run_scf
from gradwave.inputs import load_input
from gradwave.postscf.isdf import build_isdf, orbitals_on_grid, select_interpolation_points

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rpa_absorption import build_hk  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def main():
    nb = 28                    # occ + unocc orbitals to factorize
    inp = load_input(str(HERE / "si_abs.yaml"))
    system = build_system(inp)
    t0 = time.perf_counter()
    res = run_scf(inp, system=system, verbose=False)
    v_eff = res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]
    nocc = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                         for i in range(len(system.species_of_atom))))) // 2
    grid = system.grid
    print(f"[SCF {time.perf_counter()-t0:.0f}s] nocc={nocc}, factorizing {nb} orbitals at Γ")

    # orbitals on the real-space grid at Γ
    H, sph = build_hk(system, v_eff, np.zeros(3))
    _, V = np.linalg.eigh(H)
    coeffs = torch.as_tensor(V[:, :nb].T, dtype=torch.complex128)      # (nb, npw)
    phi_r = orbitals_on_grid(coeffs, sph.flat_idx, grid.shape)         # (nb, N_r)
    Nr = phi_r.shape[1]
    occ, unocc = range(nocc), range(nocc, nb)
    npair = nocc * (nb - nocc)

    # exact occ×unocc pair densities (the χ₀ ingredient)
    rho_exact = (phi_r.conj()[list(occ)][:, None, :]
                 * phi_r[list(unocc)][None, :, :])                     # (nocc, nunocc, N_r)
    norm_ex = rho_exact.reshape(-1, Nr).norm(dim=1)

    print(f"\n{'N_μ':>5} {'N_μ/N_orb':>9} {'pair fit err':>13} {'compression':>12}")
    for n_mu in (40, 70, 110, 150):
        pts = select_interpolation_points(phi_r, n_mu)
        zeta = build_isdf(phi_r, pts)                                  # (N_r, n_mu) real
        phi_mu = phi_r[:, pts]                                         # (nb, n_mu)
        # THC reconstruction: ρ_vc(r) ≈ Σ_μ ζ_μ(r) φ_v*(r_μ) φ_c(r_μ)
        coef = (phi_mu.conj()[list(occ)][:, None, :]
                * phi_mu[list(unocc)][None, :, :])                     # (nocc,nunocc,n_mu)
        rho_thc = torch.einsum("rm,vcm->vcr", zeta.to(coeffs.dtype), coef)
        err = ((rho_thc - rho_exact).reshape(-1, Nr).norm(dim=1)
               / norm_ex).mean().item()
        print(f"{len(pts):5d} {len(pts)/nb:9.1f} {err:13.2e} "
              f"{npair/len(pts):11.1f}×")

    print(f"\nInterpretation: the χ₀ pair index (N_occ·N_unocc = {npair} pairs) collapses")
    print(f"onto N_μ ~ few×N_orb interpolation points — the THC/ISDF scaling reduction")
    print(f"for the RPA/GW polarizability. gradwave's ISDF is Γ-only + occupied-wired")
    print(f"today; GW needs the occ×unocc pair space (shown here) + the multi-k/q")
    print(f"momentum-transfer extension (flagged 'future work' in isdf.py).")


if __name__ == "__main__":
    main()
