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
1/|G|-ish the Sternheimer cost.

Smeared metals: occupations respond too. χ̃ decomposes exactly over the
COMPUTED band window (Davidson always carries empty buffer bands, so no
de Gironcoli θ̃ partition is needed):

  (a) response into the uncomputed complement — Sternheimer solves for the
      occupation-carrying bands with the WHOLE window projected out, so the
      shifted operator stays positive definite even with states straddling
      μ (the complement spectrum starts above the window top);
  (b) window↔window pairs — an explicit sum over states with divided-
      difference weights (f_n − f_m)/(ε_n − ε_m), which tend to f′ at
      degeneracies (safe at the Fermi surface where sharp-occupation
      denominators would blow up);
  (c) the Fermi-surface diagonal δf_n = f′(ε_n)(δε_n − δμ), with
      δμ = Σ w f′ δε / Σ w f′ from particle conservation — a rank-one
      coupling across the whole BZ, δε_n = ⟨ψ_n|δv + Σ δD|β⟩⟨β||ψ_n⟩.

Each piece is symmetric ((a)+(b) by pair symmetry of the weights, (c) is
diag + rank-one built from one vector), so χ̃ᵀ = χ̃ still holds and the
transposed fixed point is unchanged. f′ vanishes → (c) drops out and (b)
reduces to the occupied↔empty-window pairs: the insulator limit is exact,
not a special case. Free-energy gradients (M1) never needed any of this —
F is stationary in the occupations.

nspin=2: the composite vector doubles to (δρ↑, δρ↓, δbec↑, δbec↓) and every
structure above becomes a per-spin list. χ̃ is block-diagonal over spin
(each channel has its own bands, v_eff^σ, D^σ and Sternheimer solves,
g = 1) EXCEPT the Fermi-surface δμ, whose particle-conservation sums run
over BOTH channels — the shared Fermi level is the one genuine cross-spin
coupling in the response. K keeps its cross-spin blocks for free: the
Hartree kernel acts on δρ_tot and enters both channels, f_xc^{σσ'} is the
spin HVP of the grid E_xc (the NiO-validated kernel from
hubbard_u._k_hxc_spin), and the one-center Hessian's ↑↓ blocks come out of
the double backward through the joint E_1c(bec↑, bec↓). All blocks stay
symmetric (f_xc^{↑↓} = f_xc^{↓↑} by equality of mixed partials; the δμ term
is a rank-one of one composite vector), so the transposed fixed point and
the IBZ machinery carry over unchanged. The loss stays a functional of the
TOTAL density, so v̄ seeds both channels equally.

DFT+U: the composite vector gains the occupation-matrix response δn per
hub channel, the frozen window operators carry the converged V_U, and the
kernel block is the Dudarev second derivative δD_U = −(U−J)·herm(δn).
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
    """dE_total/dθ for every parameter of `xc` at a converged scf_uspp point
    (nspin=1 or 2). Includes the one-center term."""
    nspin = res.get("nspin", 1)
    system = res["system"]
    grid = system.grid
    core = system.rho_core

    if nspin == 1:
        rho = res["rho"].detach()
        rho_xc = rho if core is None else rho + core
        sigma = (sigma_from_rho(rho_xc, grid.g_cart)
                 if xc.needs_gradient else None)
        e_theta = xc.energy(rho_xc, grid.volume, sigma)
    else:
        c2 = 0.0 if core is None else 0.5 * core
        ru = res["rho_spin"][0].detach() + c2
        rd = res["rho_spin"][1].detach() + c2
        if xc.needs_gradient:
            s_uu = sigma_from_rho(ru, grid.g_cart)
            s_dd = sigma_from_rho(rd, grid.g_cart)
            s_tot = sigma_from_rho(ru + rd, grid.g_cart)
        else:
            s_uu = s_dd = s_tot = None
        e_theta = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tot)

    if any(p.is_paw for p in system.paws):
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        for a, sp in enumerate(system.species_of_atom):
            bec = (res["rho_ij_atoms"][a] if nspin == 1
                   else [res["rho_ij_atoms"][0][a], res["rho_ij_atoms"][1][a]])
            e_theta = e_theta + onec[sp].energy_theta(bec)

    grads = torch.autograd.grad(e_theta, list(xc.parameters()),
                                allow_unused=True)
    return {name: g for (name, _), g in
            zip(xc.named_parameters(), grads, strict=True)}


# ---------------------------------------------------------------------------
# Milestone 2: the composite (δρ, δbecsum) self-consistent adjoint
# ---------------------------------------------------------------------------


def _check_supported(res: dict):
    if res.get("nspin", 1) not in (1, 2):
        raise NotImplementedError("USPP adjoint: nspin must be 1 or 2")
    occ = res["occupations"]
    f_full = 2.0 if res.get("nspin", 1) == 1 else 1.0
    frac = bool(((occ > _F_CUT) & ((occ - f_full).abs() > _F_CUT)).any())
    if frac and res.get("smearing", "none") == "none":
        raise ValueError("USPP adjoint: fractional occupations but no "
                         "smearing metadata in the result dict")


_F_CUT = 1e-8  # bands above this occupation get Sternheimer solves


def _window_uspp(res: dict, isp: int, ik: int):
    """The full computed-band window at (spin, k) — coeffs, ε, f — plus the
    number of occupation-carrying bands to solve for. Bands are ε-sorted;
    the solved set must be a prefix (mp1/cold non-monotonicity lives at
    f ~ 1e-2, far above the cut)."""
    if res.get("nspin", 1) == 1:
        occ = res["occupations"][ik].to(RDTYPE)
        c, eps = res["coeffs"][ik], res["eigenvalues"][ik].to(RDTYPE)
    else:
        occ = res["occupations"][isp][ik].to(RDTYPE)
        c = res["coeffs"][isp][ik]
        eps = res["eigenvalues"][isp][ik].to(RDTYPE)
    n_solve = int((occ > _F_CUT).sum())
    if not torch.all(occ[:n_solve] > _F_CUT):
        raise NotImplementedError("USPP adjoint: non-prefix band occupations")
    return c, eps, occ, n_solve


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
        self.nspin = ns = res.get("nspin", 1)
        self.g_spin = 2.0 if ns == 1 else 1.0

        rho = res["rho"].detach()  # total density, both cases
        core = system.rho_core
        self.rho_xc = rho if core is None else rho + core
        self.rho_sp = ([rho] if ns == 1
                       else [r.detach() for r in res["rho_spin"]])
        rho_g_box = r_to_g(rho.to(CDTYPE))
        v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                               dim=(-3, -2, -1)) * grid.n_points).real
        vloc_g = local_potential_g(
            system.positions, torch.tensor(system.species_of_atom, device=dev),
            system.vloc_tables, grid.g_cart, self.vol)
        vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real
        if ns == 1:
            v_xc, _ = vxc_potential(xc, self.rho_xc, grid)
            self.veff_sp = [v_h + v_xc + vloc_r]
        else:
            from gradwave.scf.loop import vxc_spin_potential

            c2 = None if core is None else 0.5 * core
            v_up, v_dn, _ = vxc_spin_potential(
                xc,
                self.rho_sp[0] if core is None else self.rho_sp[0] + c2,
                self.rho_sp[1] if core is None else self.rho_sp[1] + c2,
                grid)
            self.veff_sp = [v_h + v_up + vloc_r, v_h + v_dn + vloc_r]

        phase_arg = system.g_sphere @ system.positions.T  # (nGm, na)
        self.phase_pos = torch.exp(
            torch.complex(torch.zeros_like(phase_arg), phase_arg))

        # screened D per spin at the converged state (∫v_eff^σ Q + ddd^σ)
        dscr_sp = [system.proj_data[0].dij_full + self._aug_dmat(v)
                   for v in self.veff_sp]
        self.is_paw = any(p.is_paw for p in system.paws)
        self.onec = None
        self.rho_ij_sp = (
            [[m.detach() for m in res["rho_ij_atoms"]]] if ns == 1
            else [[m.detach() for m in ch] for ch in res["rho_ij_atoms"]])
        if self.is_paw:
            from gradwave.scf.paw_onsite import OneCenter

            self.onec = {sp: OneCenter(system.paws[sp], xc)
                         for sp in set(system.species_of_atom)}
            dscr_sp = [d.clone() for d in dscr_sp]
            for a, sp in enumerate(system.species_of_atom):
                s0, s1 = system.atom_slices[a]
                _, ddd = self.onec[sp].energy_and_ddd(self._bec_at(a))
                if ns == 1:
                    dscr_sp[0][s0:s1, s0:s1] += ddd.to(dev)
                else:
                    for isp in range(ns):
                        dscr_sp[isp][s0:s1, s0:s1] += ddd[isp].to(dev)

        # DFT+U: rebuild the S-dressed orbital projectors and freeze the
        # converged V_U (the +U channel enters the response exactly like
        # the KB nonlocal one — a projector perturbation with a Hermitian
        # D — and its kernel is the Dudarev second derivative −(U−J)).
        # nspin=1 keeps the SCF's half-occupancy convention: ONE occupation
        # channel n_half built with 0.5·f, whose D applies to both spins.
        self.hub = None
        hub_d_sp = [None] * ns
        if "hub_occ" in res:
            from gradwave.core.batch import build_batched, projectors_b
            from gradwave.core.hubbard import HubbardManifold, hubbard_dmatrix
            from gradwave.scf.uspp_hubbard import build_uspp_hubbard

            sites = res["hub_sites"]
            man = {}
            for st in sites:
                sp = system.species_of_atom[st["atom"]]
                man[sp] = HubbardManifold(species=sp, l=st["l"], u=st["u"],
                                          j=st["j"])
            bk = build_batched(system.spheres, system.proj_data, device=dev)
            p_b = projectors_b(bk, system.positions)
            self.hub = build_uspp_hubbard(system, list(man.values()), bk, p_b)
            self.n_hub = [[m.detach() for m in ch] for ch in res["hub_occ"]]
            self.nsh = len(self.n_hub)  # 1 (n_half) or 2 hub channels
            self.hub_w = 0.5 if ns == 1 else 1.0  # δn weight per SCF f
            hub_d_sp = [hubbard_dmatrix(self.n_hub[min(isp, self.nsh - 1)],
                                        self.hub.sites, self.hub.nproj,
                                        dev).conj() for isp in range(ns)]

        # smearing scheme for f′ (Fermi-surface response weights); None for
        # fixed occupations, where every f′-term vanishes identically
        from gradwave.core.occupations import SCHEMES

        smear = res.get("smearing", "none")
        self.scheme = SCHEMES[smear] if smear != "none" else None
        self.width = float(res.get("width", 0.0))
        self.mu = res.get("fermi")

        self.hks = [[] for _ in range(ns)]
        self.c_win = [[] for _ in range(ns)]
        self.s_win = [[] for _ in range(ns)]
        self.eps_win = [[] for _ in range(ns)]
        self.f_win = [[] for _ in range(ns)]
        self.fp_win = [[] for _ in range(ns)]
        self.n_solve = [[] for _ in range(ns)]
        self.b_win = [[] for _ in range(ns)]
        self.bh_win = [[] for _ in range(ns)]  # ⟨Sφ|ψ⟩ window projections
        self.shifts = [[] for _ in range(ns)]
        self.t_band = [[] for _ in range(ns)]
        for ik, sph in enumerate(system.spheres):
            p = projectors(system.proj_data[ik], system.positions)
            sphi_k = (self.hub.sphi[ik, :, :sph.npw]
                      if self.hub is not None else None)
            for isp in range(ns):
                hk = _HkS(sph, self.shape, self.veff_sp[isp],
                          system.proj_data[ik], p, dscr_sp[isp],
                          system.q_full, hub_sphi=sphi_k,
                          hub_d=hub_d_sp[isp])
                c, eps, f, n_sv = _window_uspp(res, isp, ik)
                fp = self._f_prime(eps)
                if self.scheme is not None and (
                        n_sv == len(f) or float(fp[-1].abs()) > 1e-10):
                    raise ValueError(
                        "USPP adjoint: band window too thin — the top "
                        "computed band still carries occupation/Fermi-"
                        "surface weight; re-run the SCF with more nbands")
                self.hks[isp].append(hk)
                self.c_win[isp].append(c)
                self.s_win[isp].append(hk.s(c))
                self.eps_win[isp].append(eps)
                self.f_win[isp].append(f)
                self.fp_win[isp].append(fp)
                self.n_solve[isp].append(n_sv)
                self.b_win[isp].append(becp(p, c))
                self.bh_win[isp].append(
                    becp(sphi_k, c) if sphi_k is not None else None)
                self.shifts[isp].append(
                    2.0 * float(eps.max() - eps.min()) + 10.0)
                self.t_band[isp].append(torch.clamp(torch.einsum(
                    "bg,g,bg->b", c[:n_sv].conj(), hk.t.to(c.dtype),
                    c[:n_sv]).real, min=1e-6))

    def _bec_at(self, a: int):
        """becsum of atom a in the shape OneCenter expects: a matrix for
        nspin=1, a 2-list for spin."""
        if self.nspin == 1:
            return self.rho_ij_sp[0][a]
        return [self.rho_ij_sp[0][a], self.rho_ij_sp[1][a]]

    def _f_prime(self, eps: torch.Tensor) -> torch.Tensor:
        """df/dε per band (≤ 0, includes the spin degeneracy g = 2 for
        nspin=1, g = 1 per channel for spin) — computed by autograd through
        the scheme's occupation function, so every smearing stays
        scheme-consistent for free."""
        if self.scheme is None or self.width <= 0.0:
            return torch.zeros_like(eps)
        x = ((eps - self.mu) / self.width).detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.scheme.occupation(x)
            (dfdx,) = torch.autograd.grad(f.sum(), x)
        return self.g_spin * dfdx / self.width

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

    def _sternheimer_k(self, isp: int, ik: int, rhs, x0, tol: float,
                       max_iter: int):
        """(H − ε_n S + α S|ψ⟩⟨ψ|S) δψ_n = rhs, in the S-metric complement
        of the computed-band WINDOW (P_c = 1 − Σ|ψ⟩⟨ψ|S over every computed
        band): window↔window response goes through the explicit pair sum,
        and projecting the empties too keeps H − ε_n S positive definite
        for metallic ε_n at the Fermi level."""
        hk = self.hks[isp][ik]
        c, s = self.c_win[isp][ik], self.s_win[isp][ik]
        eps = self.eps_win[isp][ik][:self.n_solve[isp][ik]]
        alpha = self.shifts[isp][ik]

        def pc(x):  # 1 − Σ|ψ⟩⟨Sψ|
            return x - (x @ s.conj().T) @ c

        def pcd(y):  # 1 − Σ S|ψ⟩⟨ψ|
            return y - (y @ c.conj().T) @ s

        def a_apply(x):
            y = hk.h(x) - eps[:, None].to(CDTYPE) * hk.s(x)
            return pcd(y) + alpha * ((x @ s.conj().T) @ s)

        x = pc(x0)
        r = rhs - a_apply(x)
        # tol is absolute; floor it relative to |rhs| so CG stops at its
        # achievable precision instead of grinding at round-off — grinding
        # is where transient negative curvature appears (pap ≤ 0 from
        # round-off after stagnation → 1e300 step → Inf → NaN; observed on
        # FM Ni dn-channel solves with warm starts)
        tol_eff = max(tol, 1e-12 * float(torch.linalg.norm(rhs, dim=1).max()))
        z = pc(teter(r, hk.t, self.t_band[isp][ik]))
        p = z
        rz = torch.einsum("bg,bg->b", r.conj(), z).real
        for _ in range(max_iter):
            ap = a_apply(p)
            pap = torch.einsum("bg,bg->b", p.conj(), ap).real
            p2 = torch.einsum("bg,bg->b", p.conj(), p).real
            # per-band breakdown guard: freeze bands whose curvature is
            # non-positive or non-finite (their p is zeroed, so they stay
            # frozen); the operator is PD, so this only fires at the
            # round-off floor where the band is already converged
            ok = torch.isfinite(pap) & (pap > 1e-30 * p2.clamp_min(1e-300))
            if not bool(ok.any()):
                break
            a_cg = torch.where(ok, rz / pap.clamp_min(1e-300),
                               torch.zeros_like(rz))
            x = x + a_cg[:, None] * p
            r = r - a_cg[:, None] * ap
            if float(torch.linalg.norm(r, dim=1).max()) < tol_eff:
                break
            z = pc(teter(r, hk.t, self.t_band[isp][ik]))
            rz_new = torch.einsum("bg,bg->b", r.conj(), z).real
            beta = torch.where(ok, rz_new / rz.clamp_min(1e-300),
                               torch.zeros_like(rz))
            p = torch.where(ok[:, None], z + beta[:, None] * p,
                            torch.zeros_like(p))
            rz = rz_new
        return pc(x)

    def apply_chi0(self, w_sp: list, d_bare_sp: list, dpsi_warm: list,
                   cg_tol: float, cg_max_iter: int, d_hub_sp=None):
        """Composite response χ̃(w, D_bare, D_hub) → per-spin (δρ_σ(r),
        δbec_σ) and, with +U, the per-channel occupation-matrix response
        δn (returned third; None otherwise).

        w_sp: per-spin grid fields, each acting as a v_eff^σ increment
        (local on the grid AND ∫w^σ Q into D^σ). d_bare_sp: per-spin lists
        of per-atom real (nm, nm) bare-D perturbations (one-center).
        d_hub_sp: per-hub-channel lists of per-site Hermitian (dim, dim)
        V_U D-matrix perturbations (one channel for nspin=1, applied to
        both spins per the SCF's half-occupancy convention). The spin
        channels respond independently except through δμ: the shared
        Fermi level's particle-conservation sums run over BOTH channels."""
        system, grid = self.system, self.grid
        kw = system.kweights
        nsp = self.nspin
        dev = system.positions.device
        hubp_sp = [None] * nsp
        if self.hub is not None and d_hub_sp is not None:
            for isp in range(nsp):
                ch = d_hub_sp[min(isp, self.nsh - 1)]
                dmat = torch.zeros(self.hub.nproj, self.hub.nproj,
                                   dtype=CDTYPE, device=dev)
                for m, st in zip(ch, self.hub.sites, strict=True):
                    s0, d = st["start"], st["dim"]
                    dmat[s0:s0 + d, s0:s0 + d] = 0.5 * (m + m.conj().T)
                hubp_sp[isp] = dmat.conj()  # apply convention (D^T)
        dpert_sp = []
        for isp in range(nsp):
            dpert = self._aug_dmat(w_sp[isp])
            for a, (s0, s1) in enumerate(system.atom_slices):
                dpert[s0:s1, s0:s1] += d_bare_sp[isp][a].to(dpert.dtype)
            dpert_sp.append(dpert)

        drho_sm = [torch.zeros(self.shape, dtype=RDTYPE, device=dev)
                   for _ in range(nsp)]
        dbec = [[torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev)
                 for (s0, s1) in system.atom_slices] for _ in range(nsp)]
        # Fermi-surface accumulators: δρ_FS = A − δμ·B with δμ = num/den
        # assembled only after the (spin, k) loops — particle conservation
        # couples every k-point AND both spin channels through the single
        # scalar δμ (the one cross-spin term in χ̃)
        num_mu = den_mu = 0.0
        a_r = [torch.zeros(grid.n_points, dtype=RDTYPE, device=dev)
               for _ in range(nsp)]
        b_r = [torch.zeros(grid.n_points, dtype=RDTYPE, device=dev)
               for _ in range(nsp)]
        a_bec = [[torch.zeros_like(m) for m in ch] for ch in dbec]
        b_bec = [[torch.zeros_like(m) for m in ch] for ch in dbec]
        dnh = a_hub = b_hub = None
        if self.hub is not None:
            def _hub_zeros():
                return [[torch.zeros(st["dim"], st["dim"], dtype=CDTYPE,
                                     device=dev)
                         for st in self.hub.sites] for _ in range(self.nsh)]
            dnh, a_hub, b_hub = _hub_zeros(), _hub_zeros(), _hub_zeros()
        for isp in range(nsp):
            for ik, sph in enumerate(system.spheres):
                hk, c = self.hks[isp][ik], self.c_win[isp][ik]
                ns = self.n_solve[isp][ik]
                f, eps = self.f_win[isp][ik], self.eps_win[isp][ik]
                fp = self.fp_win[isp][ik]
                wk = float(kw[ik])
                psi_r = g_to_r(c, sph.flat_idx, self.shape)
                dv_psi = (box_to_sphere(r_to_g(psi_r * w_sp[isp]),
                                        sph.flat_idx)
                          + (self.b_win[isp][ik]
                             @ dpert_sp[isp].to(CDTYPE)) @ hk.p)
                if hubp_sp[isp] is not None:
                    dv_psi = dv_psi + (self.bh_win[isp][ik]
                                       @ hubp_sp[isp]) @ hk.hub_sphi
                # ⟨ψ_m|δV|ψ_n⟩ over the whole window (Hermitian: δV is real
                # local + real-symmetric D, and the sphere projection
                # commutes with the c_m pairing)
                dvmat = torch.einsum("mg,ng->mn", c.conj(), dv_psi)

                # (a) complement response: solves for occupied bands
                rhs = dv_psi[:ns]
                rhs = -(rhs - (rhs @ c.conj().T) @ self.s_win[isp][ik])
                dpsi = self._sternheimer_k(isp, ik, rhs, dpsi_warm[isp][ik],
                                           cg_tol, cg_max_iter)
                dpsi_warm[isp][ik] = dpsi
                dpsi_r = g_to_r(dpsi, sph.flat_idx, self.shape)
                fw = f[:ns]
                # per-band f_n, plus the c.c. pair (ψ*δψ + δψ*ψ)
                drho_sm[isp] += 2.0 * wk * torch.einsum(
                    "b,bxyz->xyz", fw, (psi_r[:ns].conj() * dpsi_r).real)
                b_d = becp(hk.p, dpsi)
                for a, (s0, s1) in enumerate(system.atom_slices):
                    bo = self.b_win[isp][ik][:ns, s0:s1]
                    bd = b_d[:, s0:s1]
                    m1 = torch.einsum("b,bi,bj->ij", fw.to(CDTYPE),
                                      bd.conj(), bo)
                    dbec[isp][a] += wk * (m1 + m1.conj().T)
                if self.hub is not None:
                    # hub convention n_pq = Σ wf ⟨φ_p|ψ⟩⟨ψ|φ_q⟩ (conj on the
                    # SECOND index, opposite to becsum); nspin=1 carries the
                    # SCF's half-occupancy weight
                    bh = self.bh_win[isp][ik]
                    bh_d = becp(hk.hub_sphi, dpsi)
                    ich = min(isp, self.nsh - 1)
                    for si, st in enumerate(self.hub.sites):
                        s0, d = st["start"], st["dim"]
                        m1 = torch.einsum("b,bp,bq->pq", fw.to(CDTYPE),
                                          bh_d[:, s0:s0 + d],
                                          bh[:ns, s0:s0 + d].conj())
                        dnh[ich][si] += self.hub_w * wk * (m1 + m1.conj().T)

                # (b) window↔window pairs: divided-difference weights
                # (f_n−f_m)/(ε_n−ε_m) → f′ at degeneracies; diagonal
                # excluded (that is the FS term). Insulators: only
                # occ↔empty survive.
                de = eps[:, None] - eps[None, :]
                near = de.abs() < 1e-6
                wmat = torch.where(
                    near, 0.5 * (fp[:, None] + fp[None, :]),
                    (f[:, None] - f[None, :])
                    / torch.where(near, torch.ones_like(de), de))
                wmat.fill_diagonal_(0.0)
                m_pair = wmat.to(CDTYPE) * dvmat.mT  # W_nm ⟨ψ_m|δV|ψ_n⟩
                psi_flat = psi_r.reshape(len(f), -1)
                phi = m_pair @ psi_flat
                drho_sm[isp] += wk * (psi_flat.conj() * phi).real.sum(
                    dim=0).reshape(self.shape)
                for a, (s0, s1) in enumerate(system.atom_slices):
                    bw = self.b_win[isp][ik][:, s0:s1]
                    dbec[isp][a] += wk * torch.einsum(
                        "nm,ni,mj->ij", m_pair, bw.conj(), bw)
                if self.hub is not None:
                    bh = self.bh_win[isp][ik]
                    ich = min(isp, self.nsh - 1)
                    for si, st in enumerate(self.hub.sites):
                        s0, d = st["start"], st["dim"]
                        bhs = bh[:, s0:s0 + d]
                        dnh[ich][si] += self.hub_w * wk * torch.einsum(
                            "nm,mp,nq->pq", m_pair, bhs, bhs.conj())

                # (c) Fermi-surface diagonal: δf_n = f′_n (δε_n − δμ)
                fs = fp.abs() > 1e-14
                if bool(fs.any()):
                    deps = dvmat.diagonal().real[fs]
                    cfs = wk * fp[fs]
                    num_mu += float((cfs * deps).sum())
                    den_mu += float(cfs.sum())
                    dens = (psi_flat[fs].conj() * psi_flat[fs]).real
                    a_r[isp] += ((cfs * deps)[:, None] * dens).sum(dim=0)
                    b_r[isp] += (cfs[:, None] * dens).sum(dim=0)
                    for a, (s0, s1) in enumerate(system.atom_slices):
                        bwf = self.b_win[isp][ik][fs, s0:s1]
                        a_bec[isp][a] += torch.einsum(
                            "n,ni,nj->ij", (cfs * deps).to(CDTYPE),
                            bwf.conj(), bwf)
                        b_bec[isp][a] += torch.einsum(
                            "n,ni,nj->ij", cfs.to(CDTYPE), bwf.conj(), bwf)
                    if self.hub is not None:
                        bh = self.bh_win[isp][ik]
                        ich = min(isp, self.nsh - 1)
                        for si, st in enumerate(self.hub.sites):
                            s0, d = st["start"], st["dim"]
                            bhf = bh[fs, s0:s0 + d]
                            a_hub[ich][si] += self.hub_w * torch.einsum(
                                "n,np,nq->pq", (cfs * deps).to(CDTYPE),
                                bhf, bhf.conj())
                            b_hub[ich][si] += self.hub_w * torch.einsum(
                                "n,np,nq->pq", cfs.to(CDTYPE),
                                bhf, bhf.conj())
        dmu = num_mu / den_mu if abs(den_mu) > 1e-12 else 0.0
        drho_out = []
        for isp in range(nsp):
            drho_sm[isp] += (a_r[isp] - dmu * b_r[isp]).reshape(self.shape)
            for a in range(len(dbec[isp])):
                dbec[isp][a] += a_bec[isp][a] - dmu * b_bec[isp][a]
            dbec[isp] = [0.5 * (m + m.conj().T) for m in dbec[isp]]

            aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE,
                                  device=dev)
            for a, sp in enumerate(system.species_of_atom):
                aug_sph = aug_sph + self.phase_pos[:, a].conj() \
                    * torch.einsum("ij,ijg->g", dbec[isp][a],
                                   system.aug[sp].q_g)
            aug_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
            aug_box[system.sphere_idx] = aug_sph / self.vol
            drho_aug = torch.fft.ifftn(
                aug_box.reshape(self.shape) * grid.n_points,
                dim=(-3, -2, -1)).real
            drho_out.append(drho_sm[isp] / self.vol + drho_aug)
        if self.hub is not None:
            for ich in range(self.nsh):
                for si in range(len(self.hub.sites)):
                    m = dnh[ich][si] + a_hub[ich][si] - dmu * b_hub[ich][si]
                    dnh[ich][si] = 0.5 * (m + m.conj().T)
        return drho_out, dbec, dnh

    def k_hxc_grid(self, drho_sp: list) -> list:
        """(K_Hxc δρ)^σ(r) per spin: the Hartree kernel acts on δρ_tot and
        enters every channel; f_xc is the autograd HVP of the grid E_xc at
        the converged density (spin HVP for nspin=2 — the NiO-validated
        kernel from hubbard_u._k_hxc_spin, NLCC core split half/half)."""
        grid = self.grid
        w_g = r_to_g(sum(drho_sp).to(CDTYPE))
        inv_g2 = torch.where(grid.g2 > 1e-12,
                             1.0 / torch.clamp(grid.g2, min=1e-12),
                             torch.zeros_like(grid.g2))
        kh = (torch.fft.ifftn(4.0 * math.pi * E2 * w_g * inv_g2,
                              dim=(-3, -2, -1)) * grid.n_points).real
        scale = grid.n_points / self.vol
        if self.nspin == 1:
            rho = self.rho_xc.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                sigma = (sigma_from_rho(rho, grid.g_cart)
                         if self.xc.needs_gradient else None)
                e_xc = self.xc.energy(rho, self.vol, sigma)
                (v_xc,) = torch.autograd.grad(e_xc, rho, create_graph=True)
                inner = (v_xc * drho_sp[0].detach()).sum()
                (fxc_w,) = torch.autograd.grad(inner, rho)
            return [kh + fxc_w * scale]
        core = self.system.rho_core
        c2 = 0.0 if core is None else 0.5 * core
        ru = (self.rho_sp[0] + c2).detach().clone().requires_grad_(True)
        rd = (self.rho_sp[1] + c2).detach().clone().requires_grad_(True)
        with torch.enable_grad():
            if self.xc.needs_gradient:
                s_uu = sigma_from_rho(ru, grid.g_cart)
                s_dd = sigma_from_rho(rd, grid.g_cart)
                s_tot = sigma_from_rho(ru + rd, grid.g_cart)
            else:
                s_uu = s_dd = s_tot = None
            e_xc = self.xc.energy(ru, rd, self.vol, s_uu, s_dd, s_tot)
            vu, vd = torch.autograd.grad(e_xc, (ru, rd), create_graph=True)
            inner = ((vu * drho_sp[0].detach()).sum()
                     + (vd * drho_sp[1].detach()).sum())
            fu, fd = torch.autograd.grad(inner, (ru, rd))
        return [kh + fu * scale, kh + fd * scale]

    def hvp_onecenter(self, dbec_sp: list) -> list:
        """H_1c δbec per spin: per-atom one-center Hessian-vector products
        (zero for bare USPP, which has no one-center energy). For spin, the
        double backward through the joint E_1c(bec↑, bec↓) carries the
        cross-spin blocks automatically."""
        out = [[] for _ in range(self.nspin)]
        dev = self.system.positions.device
        for a, sp in enumerate(self.system.species_of_atom):
            ms = []
            for isp in range(self.nspin):
                m = 0.5 * (dbec_sp[isp][a] + dbec_sp[isp][a].conj().T)
                ms.append(m.real if m.is_complex() else m)
            if self.onec is None:
                for isp in range(self.nspin):
                    out[isp].append(torch.zeros_like(ms[isp]))
            elif self.nspin == 1:
                # one-center quadrature is CPU-anchored (_to_real_t);
                # bridge back to the composite vector's device
                out[0].append(self.onec[sp].hvp_becsum(
                    self.rho_ij_sp[0][a], ms[0]).to(dev))
            else:
                hu, hd = self.onec[sp].hvp_becsum(self._bec_at(a), ms)
                out[0].append(hu.to(dev))
                out[1].append(hd.to(dev))
        return out

    def k_hub(self, dnh: list) -> list:
        """Dudarev kernel per hub channel: δD_U = −(U−J)·herm(δn). The
        second derivative of E_U = Σ (U−J)/2 Tr[n − n²] is −(U−J) on the
        Hermitian part, and the D-matrix the SCF applies is built from the
        same channel's matrix (n_half for nspin=1), so no extra spin
        factors appear."""
        out = []
        for ch in dnh:
            out.append([-(st["u"] - st["j"]) * 0.5 * (m + m.conj().T)
                        for m, st in zip(ch, self.hub.sites, strict=True)])
        return out


def uspp_density_loss_param_grads(
    res: dict, xc, loss_fn, *, beta: float = 0.2, history: int = 8,
    outer_tol: float = 1e-9, max_outer: int = 100, cg_tol: float = 1e-8,
    cg_max_iter: int = 200, floor_tol: float | None = None,
    kerker_q0: float | None = None, verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """dL/dθ of a density-dependent loss through the USPP/PAW SCF fixed
    point. loss_fn: rho(grid tensor of the TOTAL density) -> scalar torch
    tensor (pure, differentiable) — for nspin=2 the loss stays a functional
    of ρ_tot, so its gradient seeds both spin channels equally. Returns
    (L, {param_name: grad}).

    floor_tol: opt-in stagnation acceptance. Some systems' achievable
    outer floor wanders (the O₂ vacuum spin-f_xc lesson, and it drifts
    with the functional parameters during training); when the residual
    stops improving for 15 iterations and the best seen is below
    floor_tol, the best iterate is used instead of raising. None keeps
    the strict behavior for validation work.

    kerker_q0: opt-in Kerker preconditioning of the outer residual's
    grid blocks, factor G²/(G²+q0²) (q0 in Å⁻¹), floored at 0.05 so the
    G≈0 modes still converge (u lives in the potential channel — its
    G=0 component is not fixed by charge conservation). The fixed point
    is unchanged. Measured: no help on small metals (Anderson with
    history is already near-exact there — Al 3→7 its), but it breaks
    the vacuum-cell stagnation floor (O₂ spin: floored at 1.4e-4
    without, converges to 1.4e-5 with q0=1.5) — the floor is low-G
    residual noise amplified through 4π/G² in the vacuum region."""
    _check_supported(res)
    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        grid, system = cs.grid, cs.system
        n_pts = grid.n_points
        nsp = cs.nspin

        rho_leaf = res["rho"].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            loss = loss_fn(rho_leaf)
            (vbar,) = torch.autograd.grad(loss, rho_leaf)

        nbec = [s1 - s0 for (s0, s1) in system.atom_slices]
        nsh = cs.nsh if cs.hub is not None else 0
        hub_dims = ([st["dim"] for st in cs.hub.sites]
                    if cs.hub is not None else [])

        def split(u):
            """Flat u → (per-spin grid fields, per-spin per-atom becsum
            mats, per-channel per-site hub mats or None). Layout mirrors
            the mixer for the first two blocks; the hub block appends the
            V_U D-matrix perturbations as (Re, Im) pairs (n is complex
            Hermitian, unlike the real becsum block)."""
            w_sp, off = [], 0
            for _ in range(nsp):
                w_sp.append(u[off:off + n_pts].reshape(grid.shape))
                off += n_pts
            mats_sp = []
            for _ in range(nsp):
                mats = []
                for n in nbec:
                    mats.append(u[off:off + n * n].reshape(n, n))
                    off += n * n
                mats_sp.append(mats)
            hub_sp = None
            if nsh:
                hub_sp = []
                for _ in range(nsh):
                    ch = []
                    for d in hub_dims:
                        re = u[off:off + d * d].reshape(d, d)
                        im = u[off + d * d:off + 2 * d * d].reshape(d, d)
                        ch.append(torch.complex(re, im))
                        off += 2 * d * d
                    hub_sp.append(ch)
            return w_sp, mats_sp, hub_sp

        def join(w_sp, mats_sp, hub_sp=None):
            parts = ([w.reshape(-1) for w in w_sp]
                     + [m.reshape(-1) for mats in mats_sp for m in mats])
            if nsh:
                for ch in hub_sp:
                    for m in ch:
                        parts.append(m.real.reshape(-1))
                        parts.append(m.imag.reshape(-1))
            return torch.cat(parts)

        dev = vbar.device
        zero_bec = [[torch.zeros(n, n, dtype=torch.float64, device=dev)
                     for n in nbec] for _ in range(nsp)]
        zero_hub = ([[torch.zeros(d, d, dtype=CDTYPE, device=dev)
                      for d in hub_dims] for _ in range(nsh)] if nsh else None)
        l_vec = join([vbar] * nsp, zero_bec, zero_hub)
        dpsi_warm = [[torch.zeros_like(c[:n_sv]) for c, n_sv in
                      zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
                     for isp in range(nsp)]

        def symmetrize(w_sp, d_bare_sp):
            """𝒮ᵀu = 𝒮u (self-adjoint projections), mirroring the SCF's
            per-iteration symmetrization on the transposed side; applied
            per spin channel."""
            if system.rho_symmetrizer is not None:
                w_sp = [
                    (torch.fft.ifftn(
                        system.rho_symmetrizer.apply(r_to_g(w.to(CDTYPE)))
                        * n_pts, dim=(-3, -2, -1))).real
                    for w in w_sp]
            if system.becsum_sym is not None:
                d_bare_sp = [
                    [m.real for m in system.becsum_sym.apply(
                        [m.to(CDTYPE) for m in ch])]
                    for ch in d_bare_sp]
            return w_sp, d_bare_sp

        kfac = None
        if kerker_q0 is not None:
            g2 = grid.g2
            kfac = (g2 / (g2 + kerker_q0 * kerker_q0)).clamp_min(0.05)

        def precondition(r):
            """Kerker-filter the grid blocks of a residual; becsum and
            hub blocks pass through (they carry no G²-divergent kernel)."""
            if kfac is None:
                return r
            w_sp, mats_sp, hub_sp = split(r)
            w_sp = [
                torch.fft.ifftn(r_to_g(w.to(CDTYPE)) * kfac * n_pts,
                                dim=(-3, -2, -1)).real
                for w in w_sp]
            return join(w_sp, mats_sp, hub_sp)

        # Anderson-accelerated fixed point u = l + K χ̃ u (plain damping
        # diverges for gain>1 modes — NiO lesson; the on-site becsum↔ddd
        # feedback is stiff in exactly the same way the SCF mixer sees).
        # With kerker_q0 the Anderson recursion runs on the preconditioned
        # residual; the convergence check stays on the raw one.
        u = l_vec.clone()
        prev_u = prev_r = None
        hist_du, hist_dr = [], []
        drho = dbec = None
        rn_best, u_best, it_best = float("inf"), None, 0
        for it in range(1, max_outer + 1):
            w_sp, d_bare_sp, d_hub_sp = split(u)
            w_sp, d_bare_sp = symmetrize(w_sp, d_bare_sp)
            drho, dbec, dnh = cs.apply_chi0(w_sp, d_bare_sp, dpsi_warm,
                                            cg_tol, cg_max_iter,
                                            d_hub_sp=d_hub_sp)
            g_u = l_vec + join(cs.k_hxc_grid(drho),
                               cs.hvp_onecenter(dbec),
                               cs.k_hub(dnh) if nsh else None)
            r_vec = g_u - u
            rn = float(torch.linalg.norm(r_vec)) / max(
                1.0, float(torch.linalg.norm(u)))
            if verbose:
                print(f"  uspp-adjoint it {it:3d}: |r|/|u| = {rn:.3e}")
            if rn < outer_tol:
                break
            if rn < rn_best:
                rn_best, u_best, it_best = rn, u.clone(), it
            if (floor_tol is not None and rn_best < floor_tol
                    and it - it_best >= 15):
                if verbose:
                    print(f"  uspp-adjoint: floored at {rn_best:.2e} "
                          f"(no improvement for {it - it_best} its)")
                w_sp, d_bare_sp, d_hub_sp = split(u_best)
                w_sp, d_bare_sp = symmetrize(w_sp, d_bare_sp)
                drho, dbec, dnh = cs.apply_chi0(
                    w_sp, d_bare_sp, dpsi_warm, cg_tol, cg_max_iter,
                    d_hub_sp=d_hub_sp)
                break
            p_vec = precondition(r_vec)
            if prev_r is not None:
                hist_du.append(u - prev_u)
                hist_dr.append(p_vec - prev_r)
                if len(hist_dr) > history:
                    hist_du.pop(0)
                    hist_dr.pop(0)
            prev_u, prev_r = u, p_vec
            if hist_dr:
                dr_m = torch.stack(hist_dr, dim=1)  # (n_u, h)
                du_m = torch.stack(hist_du, dim=1)
                gamma = torch.linalg.lstsq(dr_m, p_vec[:, None]).solution[:, 0]
                u = u + beta * p_vec - (du_m + beta * dr_m) @ gamma
            else:
                u = u + beta * p_vec
        else:
            raise RuntimeError(
                f"USPP adjoint fixed point not converged ({rn:.2e} after "
                f"{max_outer} iterations)")

        # dL/dθ = Σ_σ ⟨δρ_σ, ∂v_xc^σ/∂θ⟩ + Σ_a Σ_σ Tr[δbec_aσ ∂ddd_aσ/∂θ]
        params = list(xc.parameters())
        with torch.enable_grad():
            if nsp == 1:
                rho_fix = cs.rho_xc.detach().clone().requires_grad_(True)
                sigma = (sigma_from_rho(rho_fix, grid.g_cart)
                         if xc.needs_gradient else None)
                e_xc = xc.energy(rho_fix, grid.volume, sigma)
                (v_xc,) = torch.autograd.grad(e_xc, rho_fix,
                                              create_graph=True)
                inner = (v_xc * drho[0].detach()).sum()
            else:
                core = system.rho_core
                c2 = 0.0 if core is None else 0.5 * core
                ru = (cs.rho_sp[0] + c2).detach().clone().requires_grad_(True)
                rd = (cs.rho_sp[1] + c2).detach().clone().requires_grad_(True)
                if xc.needs_gradient:
                    s_uu = sigma_from_rho(ru, grid.g_cart)
                    s_dd = sigma_from_rho(rd, grid.g_cart)
                    s_tot = sigma_from_rho(ru + rd, grid.g_cart)
                else:
                    s_uu = s_dd = s_tot = None
                e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tot)
                vu, vd = torch.autograd.grad(e_xc, (ru, rd),
                                             create_graph=True)
                inner = ((vu * drho[0].detach()).sum()
                         + (vd * drho[1].detach()).sum())
            if cs.onec is not None:
                for a, sp in enumerate(system.species_of_atom):
                    leaves = []
                    for isp in range(nsp):
                        leaf = cs.onec[sp]._to_real_t(cs.rho_ij_sp[isp][a])
                        leaves.append(leaf.clone().requires_grad_(True))
                    e1 = cs.onec[sp].e1c_t(leaves)
                    g1s = torch.autograd.grad(e1, leaves, create_graph=True)
                    for isp in range(nsp):
                        db = 0.5 * (dbec[isp][a] + dbec[isp][a].conj().T)
                        # g1s lives on the CPU one-center graph; pair there
                        # and let .to() carry the graph across devices
                        pair = (g1s[isp] * db.real.detach().cpu()).sum()
                        inner = inner + pair.to(inner.device)
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
