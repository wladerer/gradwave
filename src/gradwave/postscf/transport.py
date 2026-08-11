"""Differentiable NEGF quantum transport — Landauer/Caroli transmission through a
block-tridiagonal device between two semi-infinite leads.

A device region (a chain of principal layers, onsite blocks H_ii coupled to their
neighbours by H_{i,i+1}) is attached to two semi-infinite leads. The retarded
Green's function

    G(E) = [ (E + i·η) I − H_device − Σ_L(E) − Σ_R(E) ]⁻¹

carries the coherent transport, with the lead self-energies Σ(E) obtained from the
lead surface Green's function by Sancho-Rubio decimation. The Caroli formula gives
the transmission

    T(E) = Tr[ Γ_L G_{1N} Γ_R G_{1N}† ],   Γ = i(Σ − Σ†),

and the zero-bias conductance is G = G₀·T(E_F) with G₀ = 2e²/h the conductance
quantum. In the ballistic (scattering-free) limit T(E) equals the integer number
of open channels; a scattering region suppresses it.

Everything is torch and batched over the energy grid, so it is DIFFERENTIABLE:
`autograd` flows through the matrix inverse, giving exact dT/d(parameter) —
dT/d(gate), dT/d(atomic position), dT/d(field) — for inverse device design, which
no non-AD transport code offers.

Basis-agnostic: it consumes block-tridiagonal Hamiltonian blocks (from a localized
/ Wannier / tight-binding representation, or the DG-ALB block Hamiltonian). The
blocks must be Hermitian-consistent: `H_ji = H_ij†`.

References: Datta, *Electronic Transport in Mesoscopic Systems* (1995);
Sancho, Sancho & Rubio, J. Phys. F 15, 851 (1985) (decimation).
"""

from __future__ import annotations

import torch

from gradwave.dtypes import CDTYPE

# Conductance quantum G₀ = 2e²/h [siemens]; conductance = G0_SIEMENS · T(E_F).
G0_SIEMENS = 7.748091729e-5


def _as_batched_z(energies: torch.Tensor, eta: float, n: int) -> torch.Tensor:
    """(E + i·η)·I as a batched (ne, n, n) complex operator."""
    e = torch.as_tensor(energies, dtype=CDTYPE).reshape(-1)
    z = (e + 1j * eta)[:, None, None]
    return z * torch.eye(n, dtype=CDTYPE, device=z.device)


def surface_green_function(
    h00: torch.Tensor,
    h01: torch.Tensor,
    energies: torch.Tensor,
    eta: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> torch.Tensor:
    """Retarded surface Green's function g(E) of a semi-infinite lead of identical
    principal layers — onsite ``h00`` (n×n), inter-layer coupling ``h01`` (layer
    n → n+1) — by Sancho-Rubio decimation. Batched over ``energies``; the returned
    g is the surface of a lead extending AWAY from that surface via ``h01``.

    Args:
        h00: (n, n) Hermitian onsite block.
        h01: (n, n) coupling from a layer to the next (H_{i,i+1}).
        energies: (ne,) real energies.
        eta: retarded broadening (E → E + iη).
    Returns:
        g: (ne, n, n) complex surface Green's function.
    """
    n = h00.shape[-1]
    z = _as_batched_z(energies, eta, n)
    h00c = h00.to(CDTYPE)
    eps_s = z - h00c                       # surface (accumulated) inverse-Green operator
    eps_b = z - h00c                       # bulk
    alpha = h01.to(CDTYPE).expand_as(eps_s).clone()          # couples n → n+1
    beta = h01.conj().transpose(-1, -2).to(CDTYPE).expand_as(eps_s).clone()  # n → n-1
    for _ in range(max_iter):
        g_b = torch.linalg.inv(eps_b)
        a_gb = alpha @ g_b
        b_gb = beta @ g_b
        eps_s = eps_s - a_gb @ beta
        eps_b = eps_b - a_gb @ beta - b_gb @ alpha
        alpha = a_gb @ alpha
        beta = b_gb @ beta
        if float(alpha.abs().amax()) < tol:
            break
    return torch.linalg.inv(eps_s)


def lead_self_energy(
    h00: torch.Tensor,
    h01: torch.Tensor,
    energies: torch.Tensor,
    side: str = "left",
    eta: float = 1e-6,
    **kw,
) -> torch.Tensor:
    """Self-energy Σ(E) a semi-infinite lead adds to the device layer it couples to.

    ``side='left'`` — the left lead's surface couples to the device's FIRST layer
    through ``h01`` (lead → device); Σ_L = h01† g h01.
    ``side='right'`` — the right lead couples to the LAST layer through ``h01†``;
    Σ_R = h01 g h01†. Batched over energies. Differentiable.
    """
    if side == "left":
        g = surface_green_function(h00, h01, energies, eta, **kw)
        v = h01.to(CDTYPE)
        return v.conj().transpose(-1, -2) @ g @ v
    if side == "right":
        g = surface_green_function(h00, h01.conj().transpose(-1, -2), energies, eta, **kw)
        v = h01.to(CDTYPE)
        return v @ g @ v.conj().transpose(-1, -2)
    raise ValueError("side must be 'left' or 'right'")


def _assemble_device(onsites: list[torch.Tensor], hop: torch.Tensor) -> torch.Tensor:
    """Block-tridiagonal device Hamiltonian from onsite blocks + a single coupling
    ``hop`` = H_{i,i+1} (uniform along the chain)."""
    n = onsites[0].shape[-1]
    ell = len(onsites)
    dtype = CDTYPE
    hd = torch.zeros(ell * n, ell * n, dtype=dtype, device=onsites[0].device)
    hopc = hop.to(dtype)
    for i, blk in enumerate(onsites):
        hd[i * n:(i + 1) * n, i * n:(i + 1) * n] = blk.to(dtype)
        if i + 1 < ell:
            hd[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = hopc
            hd[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = hopc.conj().transpose(-1, -2)
    return hd


def transmission(
    onsites: list[torch.Tensor],
    hop: torch.Tensor,
    energies: torch.Tensor,
    lead_h00: torch.Tensor | None = None,
    lead_h01: torch.Tensor | None = None,
    eta: float = 1e-6,
    **kw,
) -> torch.Tensor:
    """Caroli transmission T(E) through a block-tridiagonal device.

    Args:
        onsites: device diagonal blocks [H_00, H_11, …, H_{L-1,L-1}] (each n×n).
        hop: uniform inter-layer coupling H_{i,i+1} (n×n), also the device↔lead
            coupling.
        energies: (ne,) real energies.
        lead_h00, lead_h01: lead onsite / coupling. Default: the device end blocks
            and ``hop`` (device embedded in leads of its own end material).
        eta: retarded broadening.
    Returns:
        T: (ne,) real transmission (∈ [0, n_channels]).
    """
    n = onsites[0].shape[-1]
    lh00_l = onsites[0] if lead_h00 is None else lead_h00
    lh00_r = onsites[-1] if lead_h00 is None else lead_h00
    lh01 = hop if lead_h01 is None else lead_h01
    sig_l = lead_self_energy(lh00_l, lh01, energies, "left", eta, **kw)
    sig_r = lead_self_energy(lh00_r, lh01, energies, "right", eta, **kw)

    hd = _assemble_device(onsites, hop)
    ne = sig_l.shape[0]
    d = hd.shape[-1]
    z = _as_batched_z(energies, eta, d)
    a = z - hd
    a[:, :n, :n] = a[:, :n, :n] - sig_l
    a[:, -n:, -n:] = a[:, -n:, -n:] - sig_r
    g = torch.linalg.inv(a)                                  # (ne, d, d)

    gam_l = 1j * (sig_l - sig_l.conj().transpose(-1, -2))
    gam_r = 1j * (sig_r - sig_r.conj().transpose(-1, -2))
    g1n = g[:, :n, -n:]                                       # left→right corner block
    t = torch.einsum(
        "eij,ejk,ekl,eil->e", gam_l, g1n, gam_r, g1n.conj()
    )
    return t.real.reshape(ne)


def conductance(
    onsites: list[torch.Tensor],
    hop: torch.Tensor,
    fermi_energy: float,
    eta: float = 1e-6,
    **kw,
) -> float:
    """Zero-bias Landauer conductance G = G₀·T(E_F) [siemens]."""
    t = transmission(onsites, hop, torch.tensor([fermi_energy]), eta=eta, **kw)
    return G0_SIEMENS * float(t[0])
