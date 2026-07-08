"""Hubbard U as a determinable quantity, not just an input.

Two capabilities:

1. `energy_derivative_u` — the exact analytic dE_total/dU. At SCF convergence
   the KS energy is stationary in the density, so by Hellmann–Feynman the total
   derivative w.r.t. the parameter U equals the *partial*:
       dE/dU = Σ_{I,σ} ½ Tr[ n^{Iσ}(1 − n^{Iσ}) ]   (per manifold, U_eff=U−J).
   This makes U a first-class differentiable parameter — the gradient a learning
   loop would backprop — with no finite differences or SCF re-runs.

2. `linear_response_u` — the code computes its OWN U from occupation response
   (Cococcioni–de Gironcoli). A rigid probe α_J·Σ_m|φ^J_m⟩⟨φ^J_m| is added to
   manifold J and the on-site occupation N_I = Tr[n^I] is measured:
       χ_{IJ}  = dN_I/dα_J  (interacting: SCF re-converged)
       χ0_{IJ} = dN_I/dα_J  (bare: one non-self-consistent diagonalization)
       U = (χ0^{-1} − χ^{-1})_II
   The χ^{-1} background subtraction removes the rigid-shift (delocalized)
   response, leaving the local Hubbard interaction.
"""

from __future__ import annotations

import torch

from gradwave.core.hubbard import (
    HubbardManifold,
    build_hubbard_projectors,
    hubbard_projectors,
    occupation_matrices,
)
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.loop import scf


def energy_derivative_u(res, manifolds: list[HubbardManifold]) -> float:
    """Exact dE_total/dU [dimensionless] at the converged +U point (HF)."""
    if res.hub_occ is None:
        raise ValueError("res has no Hubbard occupation matrices (run scf with hubbard=...)")
    de = 0.0
    for sp_mats in res.hub_occ:  # per spin
        for n in sp_mats:  # per site
            de += 0.5 * float((torch.trace(n) - torch.trace(n @ n)).real)
    return de


def _site_occupations(res, hub, hub_q) -> torch.Tensor:
    """Per-site total occupation N_I = Σ_σ Tr[n^{Iσ}] from a converged result."""
    n = torch.zeros(hub.n_sites, dtype=RDTYPE)
    for sp in range(res.nspin):
        occ_sp = res.occupations if res.nspin == 1 else res.occupations[sp]
        w = 0.5 * occ_sp if res.nspin == 1 else occ_sp
        cpad = _pad(res.coeffs if res.nspin == 1 else res.coeffs[sp],
                    hub.q_free.shape[-1])
        mats = occupation_matrices(hub_q, cpad, w, res.system.kweights, hub.sites)
        for i, m in enumerate(mats):
            n[i] += torch.trace(m).real
    return n * (2.0 if res.nspin == 1 else 1.0)


def _pad(coeffs_per_k, npw_max):
    nk = len(coeffs_per_k)
    nb = coeffs_per_k[0].shape[0]
    out = torch.zeros(nk, nb, npw_max, dtype=CDTYPE)
    for ik, c in enumerate(coeffs_per_k):
        out[ik, :, : c.shape[1]] = c.detach()
    return out


@torch.no_grad()
def _bare_response_occ(system, base_res, hub, hub_q, alpha_vec, xc, smearing, width):
    """One non-self-consistent diagonalization at frozen (converged) v_eff plus
    the rigid probe α, returning per-site total occupations N_I."""
    from gradwave.core.batch import BatchedHamiltonian, projectors_b
    from gradwave.core.occupations import (
        SCHEMES,
        find_fermi,
        occupations_and_entropy,
    )
    from gradwave.solvers.davidson import davidson_batched

    bk, grid = system.batch, system.grid
    nspin = base_res.nspin
    g_spin = 2.0 / nspin
    projs_b = projectors_b(bk, system.positions)
    veff = base_res.v_eff if nspin == 2 else base_res.v_eff[None]

    eigs_s, coeffs_s = [], []
    for sp in range(nspin):
        dij = torch.zeros(hub.nproj, hub.nproj, dtype=CDTYPE)
        for si, s in enumerate(hub.sites):
            st, dim = s["start"], s["dim"]
            dij[st:st + dim, st:st + dim] += alpha_vec[si] * torch.eye(dim, dtype=CDTYPE)
        h = BatchedHamiltonian(bk, grid.shape, veff[sp], projs_b,
                               hub_q=hub_q, hub_dij=dij.conj())
        c0 = base_res.coeffs[sp] if nspin == 2 else base_res.coeffs
        c0 = _pad(c0, bk.npw_max)
        dav = davidson_batched(h.apply, c0, bk.t, bk.mask, tol=1e-9)
        eigs_s.append(dav.eigenvalues.to(RDTYPE))
        coeffs_s.append(dav.eigenvectors.to(CDTYPE))

    scheme = SCHEMES[smearing]
    eigs_cat = torch.cat(eigs_s, dim=0)
    kw_cat = torch.cat([system.kweights] * nspin)
    mu = torch.as_tensor(find_fermi(eigs_cat, kw_cat, scheme, width,
                                    system.n_electrons, degeneracy=g_spin)).to(RDTYPE)
    n = torch.zeros(hub.n_sites, dtype=RDTYPE)
    for sp in range(nspin):
        occ, _ = occupations_and_entropy(eigs_s[sp], mu, scheme, width, degeneracy=g_spin)
        w = g_spin * occ if nspin == 1 else occ  # to electron units per spin
        w = 0.5 * w if nspin == 1 else w
        mats = occupation_matrices(hub_q, coeffs_s[sp], w, system.kweights, hub.sites)
        for i, m in enumerate(mats):
            n[i] += torch.trace(m).real
    return n * (2.0 if nspin == 1 else 1.0)


def linear_response_u(system, xc, l: int, species: int, *, site: int = 0,
                      alpha: float = 0.1, smearing="gaussian", width=0.05,
                      scf_kwargs=None) -> dict:
    """Compute the linear-response Hubbard U [eV] for a manifold.

    Perturbs one correlated `site`, measures the on-site occupation response,
    and returns χ0, χ, and U = (χ0^{-1} − χ^{-1})_{site,site}. Runs one base +
    two perturbed SCFs (interacting χ) and cheap one-shot solves (bare χ0)."""
    scf_kwargs = dict(scf_kwargs or {})
    man = [HubbardManifold(species=species, l=l, u=0.0, j=0.0)]  # U computed at U=0
    hub = build_hubbard_projectors(system, man)
    hub_q = hubbard_projectors(hub, system.positions)
    ns = hub.n_sites

    base = scf(system, xc, smearing=smearing, width=width, hubbard=man, **scf_kwargs)

    chi_cols, chi0_cols = [], []  # response of all sites to perturbing `site`
    for sgn in (+1.0, -1.0):
        av = [0.0] * ns
        av[site] = sgn * alpha
        # interacting: full SCF re-converged with the probe
        r = scf(system, xc, smearing=smearing, width=width, hubbard=man,
                hub_alpha=av, **scf_kwargs)
        chi_cols.append(_site_occupations(r, hub, hub_q))
        # bare: one diagonalization at the base self-consistent potential
        chi0_cols.append(_bare_response_occ(system, base, hub, hub_q,
                                            torch.tensor(av), xc, smearing, width))

    # central difference dN_I/dα_site
    chi_col = (chi_cols[0] - chi_cols[1]) / (2 * alpha)   # (ns,)
    chi0_col = (chi0_cols[0] - chi0_cols[1]) / (2 * alpha)
    # symmetric 2-site (equivalent atoms): [[a,b],[b,a]] known from one column
    if ns == 2:
        chi = torch.tensor([[chi_col[site], chi_col[1 - site]],
                            [chi_col[1 - site], chi_col[site]]])
        chi0 = torch.tensor([[chi0_col[site], chi0_col[1 - site]],
                             [chi0_col[1 - site], chi0_col[site]]])
        u = float((torch.linalg.inv(chi0) - torch.linalg.inv(chi))[site, site])
    else:  # single-site scalar estimate
        chi, chi0 = chi_col[site], chi0_col[site]
        u = float(1.0 / chi0 - 1.0 / chi)
    return {"U_eV": u, "chi": chi_col[site].item(), "chi0": chi0_col[site].item(),
            "chi_col": chi_col.tolist(), "chi0_col": chi0_col.tolist()}
