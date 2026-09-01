"""Independent-particle / RPA optical dielectric function ε(ω) and absorption.

For a converged insulating SCF this evaluates the frequency-dependent macroscopic
dielectric function at the independent-particle (RPA-without-local-fields) level and
the derived optical constants. The interband (Adler–Wiser, q→0) imaginary part is

    ε₂(ω) = (16π² e²/Ω) Σ_k w_k Σ_{v∈occ, c∈unocc}
                |V_cv|² / (ε_c − ε_v)² · L_η(ε_c − ε_v − ω),

with the velocity operator V = ∂H/∂k. Its local (kinetic) part in gradwave's eV/Å
units is 2·(ħ²/2m)·(k+G) — the same convention as the static ε∞ build in
``postscf.dielectric``. The nonlocal ``[V_nl, r]`` commutator is a documented
refinement (dielectric.py obtains it by a finite difference of the KB tables in k);
this first cut uses the local part, which is exact for a purely local potential.
Directions are averaged isotropically. ε₁(ω) follows by Kramers–Kronig; the
refractive index n, extinction κ and absorption coefficient α(ω) from ε(ω).

Norm-conserving, nspin=1, insulators. The BZ sum reuses the SCF k-mesh + weights;
the SCF orbitals are re-diagonalized with extra conduction bands (band_structure
keeps only eigenvalues, so the eigenvectors the matrix elements need are rebuilt).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gradwave.constants import E2, HBAR2_2M
from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.dtypes import CDTYPE
from gradwave.solvers.davidson import davidson_batched_ms

_HBARC = 1973.269804  # eV·Å  (ħc), for α(ω) = 2ω κ / ħc


def _kramers_kronig(omega: np.ndarray, eps2: np.ndarray) -> np.ndarray:
    """ε₁(ω) = 1 + (2/π) P∫₀^∞ ω' ε₂(ω') / (ω'² − ω²) dω'  (discrete, PV)."""
    dw = omega[1] - omega[0]
    w2 = omega**2
    eps1 = np.ones_like(omega)
    for i in range(len(omega)):
        d = w2 - w2[i]
        d[i] = np.inf  # principal value: drop the singular point
        eps1[i] += (2.0 / np.pi) * np.sum(omega * eps2 / d) * dw
    return eps1


@torch.no_grad()
def optical_epsilon(
    res: Any, *, omega_max: float = 20.0, n_omega: int = 600, eta: float = 0.1,
    n_extra_bands: int = 8, diago_tol: float = 1e-9, verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """(ω, ε₁, ε₂, α_cm, info) for a converged insulating SCFResult ``res``.

    ω [eV], ε₁/ε₂ dimensionless (isotropic average), α in cm⁻¹.
    """
    system = res.system
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("task: optics currently supports nspin=1 only")
    bk, grid = system.batch, system.grid
    if bk is None:
        raise ValueError("optics needs system.batch (the k-batched geometry)")
    device = res.v_eff.device
    omega_vol = float(grid.volume)

    nelec = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                          for i in range(len(system.species_of_atom)))))
    nocc = nelec // 2
    nbands = nocc + int(n_extra_bands)

    # re-diagonalize the SCF mesh, KEEPING eigenvectors (band_structure discards them)
    p_b = projectors_b(bk, system.positions)
    h = BatchedHamiltonian(bk, grid.shape, res.v_eff, p_b)
    c0 = torch.zeros(bk.nk, nbands, bk.npw_max, dtype=CDTYPE, device=device)
    diag = torch.arange(nbands, device=device)
    c0[:, diag, diag] = 1.0
    out = davidson_batched_ms(h.apply, c0, bk.t, bk.mask, tol=diago_tol,
                              max_iter=80, mixed_precision=False)
    eig = out.eigenvalues                 # (nk, nbands) [eV]
    evec = out.eigenvectors               # (nk, nbands, npw_max) complex (padded 0)

    kpg = bk.kpg                           # (nk, npw_max, 3) [Å⁻¹]
    mask = bk.mask.to(eig.dtype)           # (nk, npw_max)
    kw = torch.as_tensor(system.kweights, dtype=eig.dtype, device=device)
    kw = kw / kw.sum()

    omega = torch.linspace(1e-3, omega_max, n_omega, device=device, dtype=eig.dtype)
    eps2 = torch.zeros(n_omega, device=device, dtype=eig.dtype)
    prefac = 16.0 * np.pi**2 * E2 / omega_vol
    vfac = 2.0 * HBAR2_2M

    for ik in range(bk.nk):
        w = evec[ik]                                    # (nbands, npw_max)
        e = eig[ik]                                     # (nbands,)
        vf = (vfac * kpg[ik]) * mask[ik][:, None]       # (npw_max, 3) [eV·Å]
        absv2 = torch.zeros((nbands, nbands), device=device, dtype=eig.dtype)
        for a in range(3):
            # V^a_{cv} = Σ_G conj(c_c[G]) · [2·HBAR2_2M·(k+G)_a] · c_v[G]
            va = (w.conj() * vf[:, a][None, :]) @ w.t()
            absv2 = absv2 + va.abs() ** 2
        absv2 = absv2 / 3.0                              # isotropic average
        de = e[nocc:nbands, None] - e[None, :nocc]       # (ncond, nocc)
        good = de > 1e-4
        def_ = de[good]
        oscf = absv2[nocc:nbands, :nocc][good] / def_**2  # |V|²/Δ²
        lor = (eta / np.pi) / ((omega[:, None] - def_[None, :]) ** 2 + eta**2)
        eps2 = eps2 + prefac * float(kw[ik]) * (lor @ oscf)

    om = omega.cpu().numpy()
    e2 = eps2.cpu().numpy()
    e1 = _kramers_kronig(om, e2)
    modulus = np.sqrt(e1**2 + e2**2)
    kappa = np.sqrt(np.maximum((modulus - e1) / 2.0, 0.0))
    alpha_cm = (2.0 * om / _HBARC) * kappa * 1e8         # Å⁻¹ → cm⁻¹
    info = {"n_bands": int(nbands), "n_occ": int(nocc), "eps_static": float(e1[0])}
    if verbose:
        print(f"[optics] eps1(0)={e1[0]:.2f}, eps2 peak {om[int(np.argmax(e2))]:.2f} eV, "
              f"{bk.nk} k, {nocc}+{n_extra_bands} bands")
    return om, e1, e2, alpha_cm, info
