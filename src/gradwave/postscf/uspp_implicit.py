"""Differentiability through the USPP/PAW SCF (task #58).

Milestone 1 — dE/dθ by stationarity: at the converged generalized SCF
point, E_total is stationary w.r.t. the S-orthonormal orbitals and the
occupations, and becsum is a function of the orbitals — so the total
derivative w.r.t. an XC-functional parameter θ is the PARTIAL derivative at
fixed state:

    dE/dθ = ∂E_xc^grid[ρ_tot + ρ_core; θ]/∂θ + Σ_a ∂E_1c^a[becsum; θ]/∂θ

Milestone 2 — the self-consistent adjoint for DENSITY-dependent losses,
on the composite response vector x = (ρ_tot on the grid, becsum) — the same
vector the Pulay mixer extrapolates. Writing the independent-particle
response to a perturbation pair p = (δv(r), δD_bare) as χ̃p (a generalized
Sternheimer solve per occupied band: the grid field enters BOTH as a local
potential and through its augmentation cross term ∫δv Q into D, exactly the
way a v_eff increment enters H), and the self-consistency kernel in the
same split basis as the BLOCK-DIAGONAL K = diag(K_Hxc, H_1c) with
K_Hxc = Hartree + f_xc (autograd HVP of the grid E_xc at ρ+ρ_core) and
H_1c = ∂²E_1c/∂becsum² (OneCenter.hvp_becsum), the χ̃/K factorization makes
both operators SYMMETRIC, so the one adjoint fixed point

    u = (v̄, 0) + K χ̃ u,        v̄(r) = ∂L/∂ρ(r)

gives every parameter gradient at once:

    dL/dθ = ⟨δρ_tot(χ̃u), ∂v_xc/∂θ⟩ + Σ_a Tr[δbec_a(χ̃u) · ∂ddd_a/∂θ]

(the ∫∂v_xc/∂θ Q cross term is absorbed into the first pairing because
δρ_tot already contains the augmentation response — that is what makes the
split-basis kernel block-diagonal). Anderson mixing on the composite u —
the NiO lesson: plain damping diverges near spin instabilities, and the
becsum↔ddd on-site mode is the stiffest direction here too.

Symmetry-reduced (IBZ) SCF points are supported. The symmetrized SCF map is
x_out = 𝒮 χ̃(Kx + p) with 𝒮 the (ρ, becsum) symmetrizers; both are group
averages of orthogonal operations, hence self-adjoint projections, so the
transposed fixed point just applies them to u before each χ̃:

    u = (v̄, 0) + K χ̃ 𝒮 u,        dL/dθ = ⟨χ̃𝒮u, p_θ⟩

The one wrinkle is ordering: the SCF symmetrizes becsum BEFORE building the
augmentation density and rho-symmetrizes after, but the augmentation map is
group-EQUIVARIANT (S_ρ∘Aug = Aug∘S_b — the correctness content of the
becsum symmetrizer), so the transpose of the composite collapses to
symmetrizing apply_chi0's two inputs and touching nothing inside. On the
symmetric subspace the weighted-IBZ response followed by symmetrization IS
the full-BZ response, so gradients match the full-mesh adjoint exactly, at
1/|G|-ish the Sternheimer cost. Insulators, nspin=1, no +U.
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import E2
from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.core.hamiltonian import becp, projectors
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.solvers.precond import teter


def uspp_energy_param_grads(res: dict, xc) -> dict[str, torch.Tensor]:
    """dE_total/dθ for every parameter of `xc` at a converged scf_uspp point.

    res: scf_uspp result (nspin=1). Includes the one-center term.
    """
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("dE/dθ for nspin=2 USPP not implemented")
    system = res["system"]
    grid = system.grid

    rho = res["rho"].detach()
    rho_xc = rho if system.rho_core is None else rho + system.rho_core
    sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
    e_theta = xc.energy(rho_xc, grid.volume, sigma)

    if any(p.is_paw for p in system.paws):
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        for a, sp in enumerate(system.species_of_atom):
            e_theta = e_theta + onec[sp].energy_theta(res["rho_ij_atoms"][a])

    grads = torch.autograd.grad(e_theta, list(xc.parameters()),
                                allow_unused=True)
    return {name: g for (name, _), g in
            zip(xc.named_parameters(), grads, strict=True)}


# ---------------------------------------------------------------------------
# Milestone 2: the composite (δρ, δbecsum) self-consistent adjoint
# ---------------------------------------------------------------------------


def _check_supported(res: dict):
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("USPP adjoint: nspin=2 is future work")
    if "hub_occ" in res:
        raise NotImplementedError("USPP adjoint: +U response not implemented")


def _occupied_uspp(res: dict, ik: int):
    occ = res["occupations"][ik]
    n_occ = int((occ > 1e-8).sum())
    if not torch.all((occ[:n_occ] - 2.0).abs() < 1e-8):
        raise NotImplementedError("USPP adjoint supports insulators only (occ = 2)")
    return res["coeffs"][ik][:n_occ], res["eigenvalues"][ik][:n_occ].to(RDTYPE)


class _ConvergedUSPP:
    """Frozen converged-state operators: per-k (H, S), occupied blocks,
    augmentation pairing and the one-center machinery."""

    def __init__(self, res: dict, xc):
        from gradwave.core.energies.hartree import hartree_potential_g
        from gradwave.core.energies.local_pp import local_potential_g
        from gradwave.scf.loop import vxc_potential
        from gradwave.scf.uspp import _HkS

        system = res["system"]
        grid = system.grid
        self.res, self.xc, self.system, self.grid = res, xc, system, grid
        self.vol, self.shape = grid.volume, grid.shape
        self.mask_flat = grid.dens_mask.reshape(-1)
        dev = system.positions.device

        rho = res["rho"].detach()
        core = system.rho_core
        self.rho_xc = rho if core is None else rho + core
        rho_g_box = r_to_g(rho.to(CDTYPE))
        v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                               dim=(-3, -2, -1)) * grid.n_points).real
        v_xc, _ = vxc_potential(xc, self.rho_xc, grid)
        vloc_g = local_potential_g(
            system.positions, torch.tensor(system.species_of_atom, device=dev),
            system.vloc_tables, grid.g_cart, self.vol)
        vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real
        self.v_eff = v_h + v_xc + vloc_r

        phase_arg = system.g_sphere @ system.positions.T  # (nGm, na)
        self.phase_pos = torch.exp(
            torch.complex(torch.zeros_like(phase_arg), phase_arg))

        # screened D at the converged state (∫v_eff Q + one-center ddd)
        dscr = system.proj_data[0].dij_full + self._aug_dmat(self.v_eff)
        self.is_paw = any(p.is_paw for p in system.paws)
        self.onec = None
        self.rho_ij = [m.detach() for m in res["rho_ij_atoms"]]
        if self.is_paw:
            from gradwave.scf.paw_onsite import OneCenter

            self.onec = {sp: OneCenter(system.paws[sp], xc)
                         for sp in set(system.species_of_atom)}
            dscr = dscr.clone()
            for a, sp in enumerate(system.species_of_atom):
                s0, s1 = system.atom_slices[a]
                _, ddd = self.onec[sp].energy_and_ddd(self.rho_ij[a])
                dscr[s0:s1, s0:s1] += ddd.to(dev)

        self.hks, self.c_occ, self.s_occ, self.eps_occ = [], [], [], []
        self.b_occ, self.shifts, self.t_band = [], [], []
        for ik, sph in enumerate(system.spheres):
            p = projectors(system.proj_data[ik], system.positions)
            hk = _HkS(sph, self.shape, self.v_eff, system.proj_data[ik], p,
                      dscr, system.q_full)
            c, eps = _occupied_uspp(res, ik)
            self.hks.append(hk)
            self.c_occ.append(c)
            self.s_occ.append(hk.s(c))
            self.eps_occ.append(eps)
            self.b_occ.append(becp(p, c))
            self.shifts.append(2.0 * float(eps.max() - eps.min()) + 10.0)
            self.t_band.append(torch.clamp(torch.einsum(
                "bg,g,bg->b", c.conj(), hk.t.to(c.dtype), c).real, min=1e-6))

    def _aug_dmat(self, w_r: torch.Tensor) -> torch.Tensor:
        """Block-diagonal ∫w(r) Q_ij(r−τ_a) d³r — same pairing the SCF uses
        to screen D with v_eff (a grid perturbation enters D through it)."""
        system = self.system
        w_g = r_to_g(w_r.to(CDTYPE)).reshape(-1)[self.mask_flat]
        out = torch.zeros_like(system.q_full)
        for a, sp in enumerate(system.species_of_atom):
            s0, s1 = system.atom_slices[a]
            contr = torch.einsum("ijg,g->ij", system.aug[sp].q_g.conj(),
                                 w_g * self.phase_pos[:, a])
            out[s0:s1, s0:s1] = (0.5 * (contr + contr.conj().T)).real
        return out

    def _sternheimer_k(self, ik: int, rhs, x0, tol: float, max_iter: int):
        """(H − ε_n S + α S|ψ⟩⟨ψ|S) δψ_n = rhs, in the S-metric conduction
        space (P_c = 1 − Σ|ψ⟩⟨ψ|S projects the WHOLE occupied subspace, so
        degenerate valence tops are handled together)."""
        hk = self.hks[ik]
        c, s, eps = self.c_occ[ik], self.s_occ[ik], self.eps_occ[ik]
        alpha = self.shifts[ik]

        def pc(x):  # 1 − Σ|ψ⟩⟨Sψ|
            return x - (x @ s.conj().T) @ c

        def pcd(y):  # 1 − Σ S|ψ⟩⟨ψ|
            return y - (y @ c.conj().T) @ s

        def a_apply(x):
            y = hk.h(x) - eps[:, None].to(CDTYPE) * hk.s(x)
            return pcd(y) + alpha * ((x @ s.conj().T) @ s)

        x = pc(x0)
        r = rhs - a_apply(x)
        z = pc(teter(r, hk.t, self.t_band[ik]))
        p = z
        rz = torch.einsum("bg,bg->b", r.conj(), z).real
        for _ in range(max_iter):
            ap = a_apply(p)
            pap = torch.einsum("bg,bg->b", p.conj(), ap).real
            a_cg = rz / torch.clamp(pap, min=1e-300)
            x = x + a_cg[:, None] * p
            r = r - a_cg[:, None] * ap
            if float(torch.linalg.norm(r, dim=1).max()) < tol:
                break
            z = pc(teter(r, hk.t, self.t_band[ik]))
            rz_new = torch.einsum("bg,bg->b", r.conj(), z).real
            p = z + (rz_new / torch.clamp(rz, min=1e-300))[:, None] * p
            rz = rz_new
        return pc(x)

    def apply_chi0(self, w_r: torch.Tensor, d_bare: list, dpsi_warm: list,
                   cg_tol: float, cg_max_iter: int):
        """Composite response χ̃(w, D_bare) → (δρ_tot(r), δbec per atom).

        w_r acts as a v_eff increment: local on the grid AND ∫w Q into D.
        d_bare: per-atom real (nm, nm) bare-D perturbations (one-center)."""
        system, grid = self.system, self.grid
        kw = system.kweights
        dpert = self._aug_dmat(w_r)
        for a, (s0, s1) in enumerate(system.atom_slices):
            dpert[s0:s1, s0:s1] += d_bare[a].to(dpert.dtype)

        drho_sm = torch.zeros(self.shape, dtype=RDTYPE)
        dbec = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE)
                for (s0, s1) in system.atom_slices]
        for ik, sph in enumerate(system.spheres):
            hk, c = self.hks[ik], self.c_occ[ik]
            psi_r = g_to_r(c, sph.flat_idx, self.shape)
            w_psi = box_to_sphere(r_to_g(psi_r * w_r), sph.flat_idx)
            beta_term = (self.b_occ[ik] @ dpert.to(CDTYPE)) @ hk.p
            rhs = w_psi + beta_term
            rhs = -(rhs - (rhs @ c.conj().T) @ self.s_occ[ik])  # −P_c† dV ψ
            dpsi = self._sternheimer_k(ik, rhs, dpsi_warm[ik], cg_tol,
                                       cg_max_iter)
            dpsi_warm[ik] = dpsi
            dpsi_r = g_to_r(dpsi, sph.flat_idx, self.shape)
            # f = 2, plus the c.c. pair (ψ*δψ + δψ*ψ)
            drho_sm += 4.0 * float(kw[ik]) * (psi_r.conj() * dpsi_r).real.sum(dim=0)
            b_d = becp(hk.p, dpsi)
            w2 = (2.0 * kw[ik]).to(CDTYPE)
            for a, (s0, s1) in enumerate(system.atom_slices):
                bo, bd = self.b_occ[ik][:, s0:s1], b_d[:, s0:s1]
                dbec[a] += w2 * (torch.einsum("bi,bj->ij", bd.conj(), bo)
                                 + torch.einsum("bi,bj->ij", bo.conj(), bd))
        dbec = [0.5 * (m + m.conj().T) for m in dbec]

        aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE)
        for a, sp in enumerate(system.species_of_atom):
            aug_sph = aug_sph + self.phase_pos[:, a].conj() * torch.einsum(
                "ij,ijg->g", dbec[a], system.aug[sp].q_g)
        aug_box = torch.zeros(grid.n_points, dtype=CDTYPE)
        aug_box[system.sphere_idx] = aug_sph / self.vol
        drho_aug = torch.fft.ifftn(aug_box.reshape(self.shape) * grid.n_points,
                                   dim=(-3, -2, -1)).real
        return drho_sm / self.vol + drho_aug, dbec

    def k_hxc_grid(self, drho: torch.Tensor) -> torch.Tensor:
        """(K_Hxc δρ)(r): Hartree kernel + f_xc·δρ (autograd HVP of the grid
        E_xc at the converged ρ_tot + ρ_core)."""
        grid = self.grid
        w_g = r_to_g(drho.to(CDTYPE))
        inv_g2 = torch.where(grid.g2 > 1e-12,
                             1.0 / torch.clamp(grid.g2, min=1e-12),
                             torch.zeros_like(grid.g2))
        kh = (torch.fft.ifftn(4.0 * math.pi * E2 * w_g * inv_g2,
                              dim=(-3, -2, -1)) * grid.n_points).real
        rho = self.rho_xc.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            sigma = (sigma_from_rho(rho, grid.g_cart)
                     if self.xc.needs_gradient else None)
            e_xc = self.xc.energy(rho, self.vol, sigma)
            (v_xc,) = torch.autograd.grad(e_xc, rho, create_graph=True)
            inner = (v_xc * drho.detach()).sum()
            (fxc_w,) = torch.autograd.grad(inner, rho)
        return kh + fxc_w * (grid.n_points / self.vol)

    def hvp_onecenter(self, dbec: list) -> list:
        """H_1c δbec: per-atom one-center Hessian-vector products (zero for
        bare USPP, which has no one-center energy)."""
        out = []
        for a, sp in enumerate(self.system.species_of_atom):
            m = 0.5 * (dbec[a] + dbec[a].conj().T)
            m = m.real if m.is_complex() else m
            if self.onec is None:
                out.append(torch.zeros_like(m))
            else:
                out.append(self.onec[sp].hvp_becsum(self.rho_ij[a], m))
        return out


def uspp_density_loss_param_grads(
    res: dict, xc, loss_fn, *, beta: float = 0.2, history: int = 8,
    outer_tol: float = 1e-9, max_outer: int = 100, cg_tol: float = 1e-8,
    cg_max_iter: int = 200, verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """dL/dθ of a density-dependent loss through the USPP/PAW SCF fixed
    point. loss_fn: rho(grid tensor) -> scalar torch tensor (pure,
    differentiable). Returns (L, {param_name: grad})."""
    _check_supported(res)
    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        grid, system = cs.grid, cs.system
        n_pts = grid.n_points

        rho_leaf = res["rho"].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            loss = loss_fn(rho_leaf)
            (vbar,) = torch.autograd.grad(loss, rho_leaf)

        nbec = [s1 - s0 for (s0, s1) in system.atom_slices]

        def split(u):
            w_r = u[:n_pts].reshape(grid.shape)
            mats, off = [], n_pts
            for n in nbec:
                mats.append(u[off:off + n * n].reshape(n, n))
                off += n * n
            return w_r, mats

        def join(w_r, mats):
            return torch.cat([w_r.reshape(-1)] + [m.reshape(-1) for m in mats])

        l_vec = join(vbar, [torch.zeros(n, n, dtype=torch.float64)
                            for n in nbec])
        dpsi_warm = [torch.zeros_like(c) for c in cs.c_occ]

        def symmetrize(w_r, d_bare):
            """𝒮ᵀu = 𝒮u (self-adjoint projections), mirroring the SCF's
            per-iteration symmetrization on the transposed side."""
            if system.rho_symmetrizer is not None:
                w_g = system.rho_symmetrizer.apply(r_to_g(w_r.to(CDTYPE)))
                w_r = (torch.fft.ifftn(w_g * n_pts, dim=(-3, -2, -1))).real
            if system.becsum_sym is not None:
                d_bare = [m.real for m in system.becsum_sym.apply(
                    [m.to(CDTYPE) for m in d_bare])]
            return w_r, d_bare

        # Anderson-accelerated fixed point u = l + K χ̃ u (plain damping
        # diverges for gain>1 modes — NiO lesson; the on-site becsum↔ddd
        # feedback is stiff in exactly the same way the SCF mixer sees).
        u = l_vec.clone()
        prev_u = prev_r = None
        hist_du, hist_dr = [], []
        drho = dbec = None
        for it in range(1, max_outer + 1):
            w_r, d_bare = symmetrize(*split(u))
            drho, dbec = cs.apply_chi0(w_r, d_bare, dpsi_warm, cg_tol,
                                       cg_max_iter)
            g_u = l_vec + join(cs.k_hxc_grid(drho),
                               cs.hvp_onecenter(dbec))
            r_vec = g_u - u
            rn = float(torch.linalg.norm(r_vec)) / max(
                1.0, float(torch.linalg.norm(u)))
            if verbose:
                print(f"  uspp-adjoint it {it:3d}: |r|/|u| = {rn:.3e}")
            if rn < outer_tol:
                break
            if prev_r is not None:
                hist_du.append(u - prev_u)
                hist_dr.append(r_vec - prev_r)
                if len(hist_dr) > history:
                    hist_du.pop(0)
                    hist_dr.pop(0)
            prev_u, prev_r = u, r_vec
            if hist_dr:
                dr_m = torch.stack(hist_dr, dim=1)  # (n_u, h)
                du_m = torch.stack(hist_du, dim=1)
                gamma = torch.linalg.lstsq(dr_m, r_vec[:, None]).solution[:, 0]
                u = u + beta * r_vec - (du_m + beta * dr_m) @ gamma
            else:
                u = u + beta * r_vec
        else:
            raise RuntimeError(
                f"USPP adjoint fixed point not converged ({rn:.2e} after "
                f"{max_outer} iterations)")

        # dL/dθ = ⟨δρ_tot, ∂v_xc/∂θ⟩ + Σ_a Tr[δbec_a ∂ddd_a/∂θ]
        params = list(xc.parameters())
        rho_fix = cs.rho_xc.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            sigma = (sigma_from_rho(rho_fix, grid.g_cart)
                     if xc.needs_gradient else None)
            e_xc = xc.energy(rho_fix, grid.volume, sigma)
            (v_xc,) = torch.autograd.grad(e_xc, rho_fix, create_graph=True)
            inner = (v_xc * drho.detach()).sum()
            if cs.onec is not None:
                for a, sp in enumerate(system.species_of_atom):
                    leaf = cs.onec[sp]._to_real_t(cs.rho_ij[a])
                    leaf = leaf.clone().requires_grad_(True)
                    e1 = cs.onec[sp].e1c_t([leaf])
                    (g1,) = torch.autograd.grad(e1, leaf, create_graph=True)
                    db = 0.5 * (dbec[a] + dbec[a].conj().T)
                    inner = inner + (g1 * db.real.detach()).sum()
            # ONE shared n_pts/Ω: u was seeded with the grid-gradient v̄
            # (= (Ω/n_pts)·physical δL/δρ), so the whole response Cu carries
            # that scale and BOTH pairings (grid and one-center trace) need
            # the same conversion — scaling only the grid term breaks the
            # becsum block.
            inner = inner * (n_pts / grid.volume)
            grads = torch.autograd.grad(inner, params, allow_unused=True)
    named = {
        name: (g if g is not None else torch.zeros_like(p))
        for (name, p), g in zip(xc.named_parameters(), grads, strict=True)
    }
    return loss.detach(), named
