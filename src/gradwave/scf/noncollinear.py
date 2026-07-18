"""Non-collinear spinor SCF (no spin-orbit yet).

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
from gradwave.dtypes import CDTYPE, CDTYPE_LOW, RDTYPE, RDTYPE_LOW
from gradwave.scf.common import symmetrize_rho
from gradwave.scf.guess import sad_density
from gradwave.scf.loop import System, _stack_dij
from gradwave.scf.mixing import PulayMixer
from gradwave.scf.moment_penalty import field_coeff
from gradwave.solvers.davidson import davidson_batched


class SpinorHamiltonian:
    """H apply on doubled vectors (nk, nb, 2·npw_max).

    Nonlocal: scalar-relativistic pseudos act per spin component with p;
    fully-relativistic pseudos use spinor projectors q on the DOUBLED axis
    (j-resolved SOC — see core/spinor_proj.py)."""

    def __init__(self, bk: BatchedK, shape, v_r, b_vec_r, p, q=None, dij_so=None):
        self.bk = bk
        self.shape = shape
        self.p = p  # (nk, nproj, npw_max) scalar projectors
        self.q = q  # (nk, nproj_so, 2·npw_max) spinor projectors (FR)
        self.dij_so = dij_so
        self.m = bk.npw_max
        # Precompute the 2×2 potential blocks once (fixed per H): the diagonal
        # spin channels v ± Bz and the off-diagonal Bx − iBy. Nonmagnetic runs
        # (B⃗ ≡ 0) take a fast path that skips the spin-flip term entirely.
        bx, by, bz = b_vec_r[0], b_vec_r[1], b_vec_r[2]
        self.b_zero = float(b_vec_r.abs().max()) == 0.0
        self._v_uu = v_r + bz  # ⟨↑|V̂|↑⟩ (real)
        self._v_dd = v_r - bz  # ⟨↓|V̂|↓⟩ (real)
        self._v_ud = torch.complex(bx, -by)  # ⟨↑|V̂|↓⟩ (complex)
        self._cache: dict = {}

    def _tables(self, cdtype):
        """Working-precision copies of the fixed tensors (cached per dtype)."""
        cached = self._cache.get(cdtype)
        if cached is None:
            from gradwave.dtypes import real_of

            rdtype = real_of(cdtype)
            cached = {
                "t": self.bk.t.to(rdtype),
                "v_uu": self._v_uu.to(rdtype),
                "v_dd": self._v_dd.to(rdtype),
                "v_ud": self._v_ud.to(cdtype),
                "p": self.p.to(cdtype),
                "q": None if self.q is None else self.q.to(cdtype),
                "dij_so": None if self.dij_so is None else self.dij_so.to(cdtype),
                "dij": self.bk.dij_full.to(cdtype),
            }
            self._cache[cdtype] = cached
        return cached

    def _band_chunk(self, nk: int, device, elem_bytes: int = 16) -> int:
        """Bands per chunk: the potential mix holds ~6 dense-grid temporaries
        (two ψ components + products); keep each under ~250 MB on GPU and
        ~400 MB on CPU. The CPU bound matters at many k: an unchunked apply on
        a 144-k SOC metal materializes (nk, 2·nb, grid) FFT temporaries — 8+ GB
        — and OOM-kills small-RAM hosts (asus, 14 GB). elem_bytes lets the fp32
        draft (8 B) take twice the bands of fp64."""
        n = self.shape[0] * self.shape[1] * self.shape[2]
        budget = 2.5e8 if device.type == "cuda" else 4.0e8
        return max(1, int(budget / (elem_bytes * n * max(nk, 1))))

    def apply(self, c: torch.Tensor) -> torch.Tensor:
        bk, m = self.bk, self.m
        tab = self._tables(c.dtype)
        t_r = tab["t"]
        cu, cd = c[..., :m], c[..., m:]
        out_u = t_r[:, None, :] * cu
        out_d = t_r[:, None, :] * cd

        nk, nb = c.shape[0], c.shape[1]
        chunk = self._band_chunk(nk, c.device, c.element_size())
        for lo in range(0, nb, chunk):
            hi = min(lo + chunk, nb)
            nbc = hi - lo
            # fuse both spinor components into a single batched FFT pair
            cud = torch.cat([cu[:, lo:hi], cd[:, lo:hi]], dim=1)
            psi = g_to_r_b(cud, bk, self.shape)
            psi_u, psi_d = psi[:, :nbc], psi[:, nbc:]
            if self.b_zero:  # B⃗ = 0: diagonal spin blocks, no spin flip
                h_u = psi_u * tab["v_uu"]
                h_d = psi_d * tab["v_dd"]
            else:
                v_ud = tab["v_ud"]
                h_u = psi_u * tab["v_uu"] + psi_d * v_ud
                h_d = psi_u * v_ud.conj() + psi_d * tab["v_dd"]
            hud = box_to_sphere_b(torch.cat([h_u, h_d], dim=1), bk)
            out_u[:, lo:hi] += hud[:, :nbc]
            out_d[:, lo:hi] += hud[:, nbc:]

        mask = bk.mask[:, None, :]
        out = torch.cat([out_u * mask, out_d * mask], dim=-1)

        # nonlocal, band-chunked like the FFT mix above: the unchunked einsums
        # materialize (nk, nb, 2·npw) temporaries — at 384 k and a 240-vector
        # Davidson block that is a >5 GB spike per temporary, which OOM-killed
        # the A100 FePt run through allocator fragmentation. In-place adds on
        # band slices bound the spike at the chunk size.
        if self.q is not None:  # spin-orbit (j-resolved) nonlocal
            q, dso = tab["q"], tab["dij_so"]
            mask2 = torch.cat([mask, mask], dim=-1)
            for lo in range(0, nb, chunk):
                hi = min(lo + chunk, nb)
                b = torch.einsum("kpg,kbg->kbp", q.conj(), c[:, lo:hi])
                out[:, lo:hi] += torch.einsum("kbp,pq,kqg->kbg", b, dso, q) * mask2
        elif self.p.shape[1]:
            dij, p = tab["dij"], tab["p"]
            for lo in range(0, nb, chunk):
                hi = min(lo + chunk, nb)
                bu = torch.einsum("kpg,kbg->kbp", p.conj(), cu[:, lo:hi])
                bd = torch.einsum("kpg,kbg->kbp", p.conj(), cd[:, lo:hi])
                out[:, lo:hi, :m] += torch.einsum("kbp,pq,kqg->kbg", bu, dij, p) * mask
                out[:, lo:hi, m:] += torch.einsum("kbp,pq,kqg->kbg", bd, dij, p) * mask
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
    mag_mixing_alpha: float | None = None,  # separate step for m⃗ (None → max(mixing_alpha,0.6))
    adaptive: bool = True,  # back off mixing on a stalled/oscillating residual
    diago_tol: float = 1e-9,
    verbose: bool = True,
    nonmagnetic: bool = False,  # pin m⃗ ≡ 0 (QE's domag=false): nonmagnetic + SOC
    mixed_precision: bool = False,  # opt-in fp32 draft (situational — see scf())
    constrain_dirs=None,  # (na,3) unit target directions ê_I for constrained moments
    constrain_lambda: float = 0.0,  # penalty strength λ [eV/μB²] (Ma-Dudarev)
    atom_weights=None,  # (na,*grid) Hirshfeld weights; required when constraining
    constrain_mode: str = "perp",  # "perp" (direction only) or "vector" (+magnitude)
    constrain_target_mag=None,  # per-atom |M| target [μB] for mode="vector"
) -> NCResult:
    if system.rho_symmetrizer is not None and not nonmagnetic:
        raise ValueError(
            "noncollinear SCF with a nonzero m⃗ requires use_symmetry=False "
            "(time reversal and the space group act on m⃗); the nonmagnetic "
            "(m⃗ ≡ 0) case keeps the full crystal symmetry — pass nonmagnetic=True"
        )
    grid, bk = system.grid, system.batch
    vol, nk = grid.volume, len(system.spheres)
    device = system.positions.device
    mp_crossover = 1e-5
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
    n_chan = 1 if nonmagnetic else 4
    # magnetization-aware mixing: ρ and m⃗ ride one packed vector but must NOT
    # share a single step. QE/VASP mix the spin channel separately from charge;
    # here m⃗ gets its own step through the mixer's per-component step_scale.
    # The failure mode is moment COLLAPSE: at a small mixing_alpha the charge is
    # under-relaxed and the magnetization, dragged toward the transient small
    # m_out before the exchange field self-consistifies, decays into the wrong
    # nonmagnetic basin (bcc O2: |M| → 0 at alpha=0.4 while alpha=0.7 keeps the
    # triplet). Decoupling the m⃗ step with a floor keeps the magnetization mixed
    # vigorously enough to hold the magnetic branch regardless of the charge
    # step; the adaptive backoff below is the counterweight against overshoot.
    if mag_mixing_alpha is None:
        mag_mixing_alpha = max(mixing_alpha, 0.6)
    base_step_scale = None
    if nonmagnetic:
        m = torch.zeros_like(m)
        mixer = PulayMixer(g2_vec, alpha=mixing_alpha, history=mixing_history,
                           kerker=True, check_g0=True)
    else:
        kerker_mask = torch.cat([torch.ones(ng, dtype=torch.bool, device=device),
                                 torch.zeros(3 * ng, dtype=torch.bool, device=device)])
        ratio = mag_mixing_alpha / mixing_alpha if mixing_alpha > 0 else 1.0
        base_step_scale = torch.cat([
            torch.ones(ng, dtype=RDTYPE, device=device),
            torch.full((3 * ng,), float(ratio), dtype=RDTYPE, device=device)])
        mixer = PulayMixer(torch.cat([g2_vec] * 4), alpha=mixing_alpha,
                           history=mixing_history, kerker=True, check_g0=False,
                           kerker_mask=kerker_mask, step_scale=base_step_scale)

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
    # adaptive mixing-backoff state: a global step multiplier layered on top of
    # base_step_scale, cut when the residual stops falling (see the loop below).
    adapt_mult, last_backoff, stall_window = 1.0, 0, 6

    # nonmagnetic + SOC keeps the full crystal symmetry (m⃗ ≡ 0): reduce k to
    # the IBZ in setup_system and symmetrize ρ each step, exactly as the scalar
    # path does. m⃗ is never symmetrized (it is pinned to zero here).
    def symmetrize(r_out):
        return symmetrize_rho(system.rho_symmetrizer, r_out, grid)

    def vec_of(fields):
        return torch.cat([r_to_g(f.to(CDTYPE)).reshape(-1)[mask_flat] for f in fields])

    for it in range(1, max_iter + 1):
        rho_g_box = r_to_g(rho.to(CDTYPE))
        v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                               dim=(-3, -2, -1)) * grid.n_points).real
        v_xc, b_xc, _ = vxc_and_bxc(xc, rho, m, grid, rho_core=system.rho_core)
        if nonmagnetic:
            b_xc = torch.zeros_like(b_xc)
        elif constrain_dirs is not None:
            # Constraining field B_c(r) = Σ_I (∂E_p/∂M_I) w_I(r) pins each atomic
            # moment M_I = ∫ w_I m⃗ dr toward its target ê_I. ∂E_p/∂M_I comes from
            # autograd on penalty_energy (gradwave.scf.moment_penalty), so the
            # direction-only "perp" and magnitude-robust "vector" penalties share
            # one definition. It adds to the exchange field b_xc (= δE/δm⃗).
            cf = vol / grid.n_points
            m_at = torch.einsum("axyz,ixyz->ai", atom_weights, m) * cf   # M_I (na,3)
            g = field_coeff(m_at, constrain_dirs, constrain_lambda,
                            constrain_mode, constrain_target_mag)
            b_xc = b_xc + torch.einsum("ai,axyz->ixyz", g, atom_weights)
        v_r = v_h + v_xc + vloc_r

        tol_eff = max(diago_tol, 1e-3) if it == 1 else \
            max(diago_tol, min(1e-3, 0.03 * history[-1]["res"]))
        use_low = mixed_precision and tol_eff > mp_crossover
        cdtype = CDTYPE_LOW if use_low else CDTYPE
        t2_solve = t2.to(RDTYPE_LOW) if use_low else t2
        h = SpinorHamiltonian(bk, grid.shape, v_r, b_xc, projs_b, q=q_so, dij_so=dij_so)
        dav = davidson_batched(h.apply, coeffs.to(cdtype), t2_solve, mask2, tol=tol_eff)
        eigs = dav.eigenvalues.to(RDTYPE)
        coeffs = dav.eigenvectors.to(CDTYPE)
        if use_low:
            # fp32 draft: renormalize spinors in fp64 so the electron count
            # (ρ at G=0) stays conserved through mixing (see collinear scf)
            coeffs = coeffs / torch.linalg.norm(
                coeffs, dim=-1, keepdim=True).clamp_min(1e-30)

        mu = float(find_fermi(eigs, system.kweights, scheme, width,
                              system.n_electrons, degeneracy=1.0))
        mu_t = torch.tensor(mu, dtype=RDTYPE, device=device)
        occ, s_ent = occupations_and_entropy(eigs, mu_t, scheme, width, degeneracy=1.0)
        entropy_term = -width * (system.kweights[:, None] * s_ent).sum()

        # Pauli-decomposed density matrix (accumulated per k to bound memory)
        rho_out = torch.zeros(grid.shape, dtype=RDTYPE, device=device)
        m_out = torch.zeros(3, *grid.shape, dtype=RDTYPE, device=device)
        nbc = h._band_chunk(1, coeffs.device, coeffs.element_size())
        for ik in range(nk):
            w = system.kweights[ik]
            bk1 = _slice_bk(bk, ik)
            for lo in range(0, nbands, nbc):
                hi = min(lo + nbc, nbands)
                cu = coeffs[ik:ik + 1, lo:hi, :m_pw]
                cd = coeffs[ik:ik + 1, lo:hi, m_pw:]
                pu = g_to_r_b(cu, bk1, grid.shape)[0]
                pd = g_to_r_b(cd, bk1, grid.shape)[0]
                f = (w * occ[ik, lo:hi]).to(pu.real.dtype)
                uu = torch.einsum("b,bxyz->xyz", f, pu.real**2 + pu.imag**2)
                dd = torch.einsum("b,bxyz->xyz", f, pd.real**2 + pd.imag**2)
                ud = torch.einsum("b,bxyz->xyz", f.to(CDTYPE), pu.conj() * pd)
                rho_out += uu + dd
                m_out[0] += 2.0 * ud.real
                m_out[1] += 2.0 * ud.imag
                m_out[2] += uu - dd
        rho_out, m_out = rho_out / vol, m_out / vol
        rho_out = symmetrize(rho_out)  # no-op unless IBZ symmetry is active

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

        if nonmagnetic:
            m_out = torch.zeros_like(m_out)
            vin, vout = vec_of([rho]), vec_of([rho_out])
        else:
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
        # adaptive fallback: a residual that stops decreasing over a window
        # (stall) or bounces (limit cycle at a frustrated moment / SOC) means
        # the step is too aggressive for the local Jacobian. Halve the global
        # step multiplier and drop the DIIS history so the pre-stall vectors
        # stop fighting the recovery, instead of silently running to max_iter.
        if (adaptive and it - last_backoff >= stall_window
                and it > 2 * stall_window and adapt_mult > 0.1):
            recent = min(h["res"] for h in history[-stall_window:])
            before = min(h["res"] for h in history[-2 * stall_window:-stall_window])
            if recent > 0.9 * before:
                adapt_mult = max(0.5 * adapt_mult, 0.1)
                mixer.step_scale = (adapt_mult if base_step_scale is None
                                    else base_step_scale * adapt_mult)
                mixer.reset()
                last_backoff = it
                if verbose:
                    print(f"  NC-SCF: residual stalled — mixing step x{adapt_mult:.2f}",
                          flush=True)
        mixed = mixer.step(vin, vout)
        fields = []
        for c4 in range(n_chan):
            gnew = torch.zeros(grid.n_points, dtype=CDTYPE, device=device)
            gnew[mask_flat] = mixed[c4 * ng:(c4 + 1) * ng]
            fields.append((torch.fft.ifftn(gnew.reshape(grid.shape) * grid.n_points,
                                           dim=(-3, -2, -1))).real)
        rho = fields[0]
        if not nonmagnetic:
            m = torch.stack(fields[1:])

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
