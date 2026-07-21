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
System with time_reversal=False: TR flips m⃗, so the plain TR-reduced mesh
is only valid for collinear-limit checks. For real k-savings pass
setup_system(..., use_symmetry=True, magmoms=...): k then folds into the
MAGNETIC IBZ of the Shubnikov group (anti-unitary g·T ops act as −W⁻ᵀ) and
(ρ, m⃗) are re-symmetrized over the full magnetic group each iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from gradwave.core.batch import BatchedK, becp_b, projectors_b
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.fftbox import r_to_g
from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.core.xc.noncollinear import NoncollinearXC, vxc_and_bxc
from gradwave.dtypes import CDTYPE, CDTYPE_LOW, RDTYPE, RDTYPE_LOW
from gradwave.scf.common import (
    MP_CROSSOVER,
    adaptive_diago_tol,
    convergence_gate,
    record_iteration,
    symmetrize_rho,
)
from gradwave.scf.guess import sad_density
from gradwave.scf.loop import System, _stack_dij
from gradwave.scf.mixing import PulayMixer
from gradwave.scf.moment_penalty import field_coeff
from gradwave.scf.spinor_common import (
    apply_local_spinor,
    pauli_density_accumulate,
    spinor_band_chunk,
    spinor_potential_blocks,
    spinor_pw_seed,
)
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
        # Precompute the 2×2 potential blocks once (fixed per H): v_uu/v_dd
        # are ⟨↑|V̂|↑⟩/⟨↓|V̂|↓⟩ (real), v_ud is ⟨↑|V̂|↓⟩ (complex); nonmagnetic
        # runs (B⃗ ≡ 0) take a fast path that skips the spin-flip term.
        self.b_zero, self._v_uu, self._v_dd, self._v_ud = \
            spinor_potential_blocks(v_r, b_vec_r)
        self._cache: dict = {}

    def _tables(self, cdtype):
        """Working-precision copies of the fixed tensors (cached per dtype)."""
        cached = self._cache.get(cdtype)
        if cached is None:
            from gradwave.dtypes import real_of

            rdtype = real_of(cdtype)
            p = self.p.to(cdtype)
            q = None if self.q is None else self.q.to(cdtype)
            cached = {
                "t": self.bk.t.to(rdtype),
                "v_uu": self._v_uu.to(rdtype),
                "v_dd": self._v_dd.to(rdtype),
                "v_ud": self._v_ud.to(cdtype),
                "p": p,
                # conjugates cached too: they are constant per H but consumed
                # in every band chunk of every Davidson round
                "p_conj": p.conj().resolve_conj(),
                "q": q,
                "q_conj": None if q is None else q.conj().resolve_conj(),
                "dij_so": None if self.dij_so is None else self.dij_so.to(cdtype),
                "dij": self.bk.dij_full.to(cdtype),
            }
            self._cache[cdtype] = cached
        return cached

    def _band_chunk(self, nk: int, device, elem_bytes: int = 16) -> int:
        """Bands per chunk bounding the dense-grid temporaries — the shared
        spinor heuristic (scf/spinor_common.py)."""
        return spinor_band_chunk(self.shape, nk, device, elem_bytes)

    def apply(self, c: torch.Tensor) -> torch.Tensor:
        bk, m = self.bk, self.m
        tab = self._tables(c.dtype)
        t_r = tab["t"]
        cu, cd = c[..., :m], c[..., m:]
        out_u = t_r[:, None, :] * cu
        out_d = t_r[:, None, :] * cd

        nk, nb = c.shape[0], c.shape[1]
        chunk = self._band_chunk(nk, c.device, c.element_size())
        apply_local_spinor(out_u, out_d, cu, cd, bk, self.shape, chunk,
                           tab["v_uu"], tab["v_dd"], tab["v_ud"], self.b_zero)

        mask = bk.mask[:, None, :]
        out = torch.cat([out_u * mask, out_d * mask], dim=-1)

        # nonlocal, band-chunked like the FFT mix above: the unchunked einsums
        # materialize (nk, nb, 2·npw) temporaries — at 384 k and a 240-vector
        # Davidson block that is a >5 GB spike per temporary, which OOM-killed
        # the A100 FePt run through allocator fragmentation. In-place adds on
        # band slices bound the spike at the chunk size.
        if self.q is not None:  # spin-orbit (j-resolved) nonlocal
            q, qc, dso = tab["q"], tab["q_conj"], tab["dij_so"]
            mask2 = torch.cat([mask, mask], dim=-1)
            for lo in range(0, nb, chunk):
                hi = min(lo + chunk, nb)
                b = torch.einsum("kpg,kbg->kbp", qc, c[:, lo:hi])
                out[:, lo:hi] += torch.einsum("kbp,pq,kqg->kbg", b, dso, q) * mask2
        elif self.p.shape[1]:
            dij, p, pc = tab["dij"], tab["p"], tab["p_conj"]
            for lo in range(0, nb, chunk):
                hi = min(lo + chunk, nb)
                bu = torch.einsum("kpg,kbg->kbp", pc, cu[:, lo:hi])
                bd = torch.einsum("kpg,kbg->kbp", pc, cd[:, lo:hi])
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
    formalism: str = "noncollinear"  # result-type tag shared by all four SCF drivers


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
    precond_op=None,  # callable r→P·r on the density-total block (charge channel),
    # overriding constant Kerker there — e.g. a fitted learned_precond filter
    mixer_hook=None,  # research probe: called (it, vin, vout) each step pre-mix
) -> NCResult:
    # A plain RhoSymmetrizer (paramagnetic group) is only valid with m⃗ ≡ 0.
    # A MagneticSymmetrizer (setup_system(..., magmoms=...)) carries the
    # Shubnikov group of the moment configuration and symmetrizes m⃗ too, so
    # magnetic runs on the magnetic IBZ are allowed. The seeded mag_vec_init
    # must match the magmoms the group was built from — symmetrization
    # projects every iteration onto that magnetic symmetry.
    mag_sym_active = hasattr(system.rho_symmetrizer, "apply_m")
    if system.rho_symmetrizer is not None and not (nonmagnetic or mag_sym_active):
        raise ValueError(
            "noncollinear SCF with a nonzero m⃗ requires use_symmetry=False "
            "(time reversal and the space group act on m⃗) or a MAGNETIC "
            "symmetry system (setup_system(..., magmoms=...)); the nonmagnetic "
            "(m⃗ ≡ 0) case keeps the full crystal symmetry — pass nonmagnetic=True"
        )
    grid, bk = system.grid, system.batch
    vol, nk = grid.volume, len(system.spheres)
    device = system.positions.device
    mp_crossover = MP_CROSSOVER
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

    if precond_op is not None:
        # override constant Kerker on the density-total (charge) block; m⃗ blocks
        # keep their own step (base_step_scale) and are untouched by this operator.
        mixer.precond_op = precond_op
        mixer.precond_slice = slice(0, ng)

    projs_b = projectors_b(bk, system.positions)
    q_so = dij_so = None
    if system.is_fr:
        from gradwave.core.spinor_proj import build_so_projectors

        q_so, dij_so = build_so_projectors(bk, system)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real

    # initial spinors: alternate up/down lowest plane waves
    coeffs = spinor_pw_seed(nk, nbands, m_pw, device)
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
    # path does. With a MagneticSymmetrizer, m⃗ is symmetrized as well —
    # spatially like ρ but mixed by the per-op axial 3×3 (s_T·det(S)·S).
    def symmetrize(r_out):
        return symmetrize_rho(system.rho_symmetrizer, r_out, grid)

    def symmetrize_m(m_r):
        if not mag_sym_active or nonmagnetic:
            return m_r
        m_g = torch.stack([r_to_g(m_r[i].to(CDTYPE)) for i in range(3)])
        m_g = system.rho_symmetrizer.apply_m(m_g)
        return torch.fft.ifftn(m_g * grid.n_points, dim=(-3, -2, -1)).real

    def vec_of(fields):
        return torch.cat([r_to_g(f.to(CDTYPE)).reshape(-1)[mask_flat] for f in fields])

    for it in range(1, max_iter + 1):
        t_it = time.perf_counter()
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

        tol_eff = adaptive_diago_tol(it, history, diago_tol,
                                     system.n_electrons, schedule="linear")
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

        # Pauli-decomposed density matrix — the shared band-chunked, fused-FFT
        # accumulation (scf/spinor_common.py)
        nbc = h._band_chunk(nk, coeffs.device, coeffs.element_size())
        w_kb = system.kweights[:, None] * occ
        rho_out, m_out = pauli_density_accumulate(
            coeffs, w_kb, bk, grid.shape, m_pw, nbands, nbc, device)
        rho_out, m_out = rho_out / vol, m_out / vol
        rho_out = symmetrize(rho_out)  # no-op unless IBZ symmetry is active
        m_out = symmetrize_m(m_out)  # no-op unless MAGNETIC symmetry is active
        if nonmagnetic:
            # pin m⃗ ≡ 0 BEFORE E_xc so the pinned state's energy sees no
            # eigensolver noise in m_out (mirror the b_xc zeroing above)
            m_out = torch.zeros_like(m_out)

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

        if nonmagnetic:  # m_out already pinned to 0 above (before E_xc)
            vin, vout = vec_of([rho]), vec_of([rho_out])
        else:
            vin, vout = vec_of([rho, *m]), vec_of([rho_out, *m_out])
        res_norm = float(torch.linalg.norm(vout - vin)) * vol
        de = record_iteration(history, it, e_free, e_free_prev, res_norm, t_it)
        if verbose:
            mv = [float(m_out[i].mean()) * vol for i in range(3)]
            print(f"  NC-SCF {it:3d}  F = {e_free:+.8f}  dE = {de:.2e}  "
                  f"|dρ,m| = {res_norm:.2e}  m⃗ = ({mv[0]:+.3f},{mv[1]:+.3f},{mv[2]:+.3f})",
                  flush=True)

        if convergence_gate(de, res_norm, tol_eff, etol, rhotol, diago_tol):
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
        if mixer_hook is not None:
            mixer_hook(it, vin, vout)
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
