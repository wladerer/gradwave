"""Non-collinear spinor SCF (no spin-orbit yet) — see docs/noncollinear.md.

Spinors ride the existing k-batched machinery as DOUBLED plane-wave vectors
c (nk, nb, 2·npw_max): [.., :npw] = up component, [.., npw:] = down. The
batched Davidson operates on ℂ^{2npw} unchanged (kinetic/mask/preconditioner
tensors are simply concatenated); only the Hamiltonian apply knows about
spin, mixing the components in real space through

    V̂(r) = [v_H + v_loc + v_xc]·𝟙 + B⃗_xc(r)·σ⃗

with (v_xc, B⃗_xc) from ONE autograd call on the locally-collinear XC.
The nonlocal (scalar-relativistic) projectors act on each component
independently; SOC will add the 2×2 j-resolved structure here.

Density matrix by Pauli decomposition: ρ = Σf(|ψ↑|²+|ψ↓|²),
m_z = Σf(|ψ↑|²−|ψ↓|²), m_x = 2Σf Re(ψ↑*ψ↓), m_y = 2Σf Im(ψ↑*ψ↓).
Mixing runs on the 4-vector (ρ, m⃗) with Kerker on the ρ block ONLY
(the collinear lesson: Kerker's G=0 zero must never pin magnetization).

Each spinor band holds ONE electron (Fermi degeneracy g = 1). Build the
System with time_reversal=False: TR flips m⃗, so the reduced mesh is only
valid for collinear-limit checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from gradwave.core.batch import BatchedK, becp_b, box_to_sphere_b, g_to_r_b, projectors_b
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.fftbox import r_to_g
from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.core.xc.noncollinear import NoncollinearXC, vxc_and_bxc
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.guess import sad_density
from gradwave.scf.loop import System, _stack_dij
from gradwave.scf.mixing import PulayMixer
from gradwave.solvers.davidson import davidson_batched


class SpinorHamiltonian:
    """H apply on doubled vectors (nk, nb, 2·npw_max).

    Nonlocal: scalar-relativistic pseudos act per spin component with p;
    fully-relativistic pseudos use spinor projectors q on the DOUBLED axis
    (j-resolved SOC — see core/spinor_proj.py)."""

    def __init__(self, bk: BatchedK, shape, v_r, b_vec_r, p, q=None, dij_so=None):
        self.bk = bk
        self.shape = shape
        self.v_r = v_r  # (n1,n2,n3) scalar potential [eV]
        self.b_r = b_vec_r  # (3, n1,n2,n3) exchange field [eV]
        self.p = p  # (nk, nproj, npw_max) scalar projectors
        self.q = q  # (nk, nproj_so, 2·npw_max) spinor projectors (FR)
        self.dij_so = dij_so
        self.m = bk.npw_max

    def apply(self, c: torch.Tensor) -> torch.Tensor:
        bk, m = self.bk, self.m
        cu, cd = c[..., :m], c[..., m:]
        out_u = bk.t[:, None, :] * cu
        out_d = bk.t[:, None, :] * cd

        psi_u = g_to_r_b(cu, bk, self.shape)
        psi_d = g_to_r_b(cd, bk, self.shape)
        bx, by, bz = self.b_r[0], self.b_r[1], self.b_r[2]
        v_uu = self.v_r + bz
        v_dd = self.v_r - bz
        v_ud = torch.complex(bx, -by)  # ⟨↑|V̂|↓⟩ = Bx − iBy
        h_u = psi_u * v_uu + psi_d * v_ud
        h_d = psi_u * v_ud.conj() + psi_d * v_dd

        out_u = out_u + box_to_sphere_b(h_u, bk)
        out_d = out_d + box_to_sphere_b(h_d, bk)

        mask = bk.mask[:, None, :]
        out = torch.cat([out_u * mask, out_d * mask], dim=-1)

        if self.q is not None:  # spin-orbit (j-resolved) nonlocal
            b = torch.einsum("kpg,kbg->kbp", self.q.conj(), c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b,
                                     self.dij_so.to(c.dtype), self.q) \
                * torch.cat([mask, mask], dim=-1)
        elif self.p.shape[1]:
            dij = bk.dij_full.to(c.dtype)
            bu = torch.einsum("kpg,kbg->kbp", self.p.conj(), cu)
            bd = torch.einsum("kpg,kbg->kbp", self.p.conj(), cd)
            nl = torch.cat([torch.einsum("kbp,pq,kqg->kbg", bu, dij, self.p) * mask,
                            torch.einsum("kbp,pq,kqg->kbg", bd, dij, self.p) * mask],
                           dim=-1)
            out = out + nl
        return out


@dataclass
class NCResult:
    converged: bool
    n_iter: int
    energies: EnergyBreakdown
    fermi: float
    mag_vec: tuple  # ∫ m⃗ dr [μB]
    mag_abs: float  # ∫ |m⃗| dr [μB]
    rho: torch.Tensor
    m: torch.Tensor  # (3, grid)
    eigenvalues: torch.Tensor  # (nk, nb)
    system: System
    history: list = field(default_factory=list)
    coeffs: torch.Tensor | None = None  # (nk, nb, 2·npw_max) spinor coefficients


@torch.no_grad()
def scf_noncollinear(
    system: System,
    xc: NoncollinearXC,
    mag_vec_init,  # (na, 3) initial moment fraction·direction per atom
    smearing: str = "gaussian",
    width: float = 0.1,
    max_iter: int = 120,
    etol: float = 1e-8,
    rhotol: float = 1e-7,
    mixing_alpha: float = 0.5,
    mixing_history: int = 8,
    diago_tol: float = 1e-9,
    verbose: bool = True,
) -> NCResult:
    if system.rho_symmetrizer is not None:
        raise ValueError("noncollinear SCF requires use_symmetry=False")
    grid, bk = system.grid, system.batch
    vol, nk = grid.volume, len(system.spheres)
    device = system.positions.device
    nbands = 2 * system.nbands  # spinor bands hold one electron each
    m_pw = bk.npw_max
    mag_vec_init = torch.as_tensor(mag_vec_init, dtype=RDTYPE)

    # initial (ρ, m⃗): SAD total + atom-directed magnetization channels
    rho = sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                      system.n_electrons)
    m_chan = [
        sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                    None, atom_scale=[float(mag_vec_init[a, i])
                                      for a in range(len(system.species_of_atom))])
        for i in range(3)
    ]
    m = torch.stack(m_chan).to(device)
    rho = rho.to(device)

    mask_flat = grid.dens_mask.reshape(-1)
    g2_vec = grid.g2.reshape(-1)[mask_flat]
    ng = int(mask_flat.sum())
    kerker_mask = torch.cat([torch.ones(ng, dtype=torch.bool, device=device),
                             torch.zeros(3 * ng, dtype=torch.bool, device=device)])
    mixer = PulayMixer(torch.cat([g2_vec] * 4), alpha=mixing_alpha,
                       history=mixing_history, kerker=True, check_g0=False,
                       kerker_mask=kerker_mask)

    projs_b = projectors_b(bk, system.positions)
    q_so = dij_so = None
    if system.is_fr:
        from gradwave.core.spinor_proj import build_so_projectors

        q_so, dij_so = build_so_projectors(bk, system)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real

    # initial spinors: alternate up/down lowest plane waves
    c0 = torch.zeros(nk, nbands, 2 * m_pw, dtype=CDTYPE, device=device)
    for b in range(nbands):
        c0[:, b, (b // 2) + (b % 2) * m_pw] = 1.0
    coeffs = c0
    t2 = torch.cat([bk.t, bk.t], dim=-1)
    mask2 = torch.cat([bk.mask, bk.mask], dim=-1)

    scheme = SCHEMES[smearing]
    e_free_prev, converged, history = None, False, []
    mu = 0.0

    def vec_of(fields):
        return torch.cat([r_to_g(f.to(CDTYPE)).reshape(-1)[mask_flat] for f in fields])

    for it in range(1, max_iter + 1):
        rho_g_box = r_to_g(rho.to(CDTYPE))
        v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                               dim=(-3, -2, -1)) * grid.n_points).real
        v_xc, b_xc, _ = vxc_and_bxc(xc, rho, m, grid, rho_core=system.rho_core)
        v_r = v_h + v_xc + vloc_r

        tol_eff = max(diago_tol, 1e-3) if it == 1 else \
            max(diago_tol, min(1e-3, 0.03 * history[-1]["res"]))
        h = SpinorHamiltonian(bk, grid.shape, v_r, b_xc, projs_b, q=q_so, dij_so=dij_so)
        dav = davidson_batched(h.apply, coeffs, t2, mask2, tol=tol_eff)
        eigs, coeffs = dav.eigenvalues, dav.eigenvectors

        mu = float(find_fermi(eigs, system.kweights, scheme, width,
                              system.n_electrons, degeneracy=1.0))
        mu_t = torch.tensor(mu, dtype=RDTYPE, device=device)
        occ, s_ent = occupations_and_entropy(eigs, mu_t, scheme, width, degeneracy=1.0)
        entropy_term = -width * (system.kweights[:, None] * s_ent).sum()

        # Pauli-decomposed density matrix (accumulated per k to bound memory)
        rho_out = torch.zeros(grid.shape, dtype=RDTYPE, device=device)
        m_out = torch.zeros(3, *grid.shape, dtype=RDTYPE, device=device)
        for ik in range(nk):
            w = system.kweights[ik]
            cu = coeffs[ik:ik + 1, :, :m_pw]
            cd = coeffs[ik:ik + 1, :, m_pw:]
            bk1 = _slice_bk(bk, ik)
            pu = g_to_r_b(cu, bk1, grid.shape)[0]
            pd = g_to_r_b(cd, bk1, grid.shape)[0]
            f = (w * occ[ik]).to(pu.real.dtype)
            uu = torch.einsum("b,bxyz->xyz", f, pu.real**2 + pu.imag**2)
            dd = torch.einsum("b,bxyz->xyz", f, pd.real**2 + pd.imag**2)
            ud = torch.einsum("b,bxyz->xyz", f.to(CDTYPE), pu.conj() * pd)
            rho_out += uu + dd
            m_out[0] += 2.0 * ud.real
            m_out[1] += 2.0 * ud.imag
            m_out[2] += uu - dd
        rho_out, m_out = rho_out / vol, m_out / vol

        # energies
        rho_g_out = r_to_g(rho_out.to(CDTYPE))
        t_occ = (system.kweights[:, None] * occ).to(coeffs.real.dtype)
        e_kin = torch.einsum("kb,kbg,kg->", t_occ,
                             coeffs.real**2 + coeffs.imag**2, t2)
        e_h = hartree_energy(rho_g_out, grid.g2, vol)
        from gradwave.core.xc.noncollinear import energy_with_grid

        e_xc = energy_with_grid(xc, rho_out, m_out, grid, rho_core=system.rho_core)
        e_loc = local_energy(rho_g_out, vloc_g, vol)
        if q_so is not None:
            b_so = torch.einsum("kpg,kbg->kbp", q_so.conj(), coeffs)
            e_nl = nonlocal_energy([b_so[ik] for ik in range(nk)], dij_so, occ,
                                   system.kweights)
        else:
            bu = becp_b(projs_b, coeffs[..., :m_pw])
            bd = becp_b(projs_b, coeffs[..., m_pw:])
            dij = _stack_dij(system)
            e_nl = nonlocal_energy([bu[ik] for ik in range(nk)], dij, occ,
                                   system.kweights) \
                + nonlocal_energy([bd[ik] for ik in range(nk)], dij, occ,
                                  system.kweights)
        e_ew = ewald_energy(system.positions, system.charges, grid.cell)
        energies = EnergyBreakdown(kinetic=e_kin, hartree=e_h, xc=e_xc, local=e_loc,
                                   nonlocal_=e_nl, ewald=e_ew, smearing=entropy_term)
        e_free = float(energies.free_energy)

        vin, vout = vec_of([rho, *m]), vec_of([rho_out, *m_out])
        res_norm = float(torch.linalg.norm(vout - vin)) * vol
        de = abs(e_free - e_free_prev) if e_free_prev is not None else float("inf")
        history.append({"iter": it, "free_energy": e_free, "dE": de, "res": res_norm})
        if verbose:
            mv = [float(m_out[i].mean()) * vol for i in range(3)]
            print(f"  NC-SCF {it:3d}  F = {e_free:+.8f}  dE = {de:.2e}  "
                  f"|dρ,m| = {res_norm:.2e}  m⃗ = ({mv[0]:+.3f},{mv[1]:+.3f},{mv[2]:+.3f})",
                  flush=True)

        if de < etol and res_norm < rhotol and tol_eff <= diago_tol * 1.01:
            converged = True
            rho, m = rho_out, m_out
            break

        e_free_prev = e_free
        mixed = mixer.step(vin, vout)
        fields = []
        for c4 in range(4):
            gnew = torch.zeros(grid.n_points, dtype=CDTYPE, device=device)
            gnew[mask_flat] = mixed[c4 * ng:(c4 + 1) * ng]
            fields.append((torch.fft.ifftn(gnew.reshape(grid.shape) * grid.n_points,
                                           dim=(-3, -2, -1))).real)
        rho, m = fields[0], torch.stack(fields[1:])

    m_int = [float(m[i].mean()) * vol for i in range(3)]
    m_norm = torch.sqrt((m**2).sum(dim=0))
    return NCResult(
        converged=converged, n_iter=it, energies=energies, fermi=mu,
        mag_vec=tuple(m_int), mag_abs=float(m_norm.mean()) * vol,
        rho=rho, m=m, eigenvalues=eigs, system=system, history=history,
        coeffs=coeffs,
    )


def _slice_bk(bk: BatchedK, ik: int) -> BatchedK:
    return BatchedK(
        npw=bk.npw[ik:ik + 1], mask=bk.mask[ik:ik + 1], flat_idx=bk.flat_idx[ik:ik + 1],
        kpg=bk.kpg[ik:ik + 1], t=bk.t[ik:ik + 1],
        proj_phase_free=bk.proj_phase_free[ik:ik + 1],
        proj_atom_index=bk.proj_atom_index, dij_full=bk.dij_full,
    )
