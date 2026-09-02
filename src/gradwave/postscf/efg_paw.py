"""Plane-wave/PAW electric-field-gradient (EFG) via the Petrilli-Blöchl reconstruction.

The electric field gradient at nucleus κ — the observable behind the solid-state-NMR
quadrupolar coupling ``C_Q = eQ V_zz/h`` — is the traceless second derivative of the
electrostatic potential, ``V_ab = -∂²φ/∂r_a∂r_b`` evaluated at the nucleus. This module
reconstructs it from a converged plane-wave USPP/PAW ground state (``scf.uspp_loop.scf_uspp``
→ ``USPPResult``), so a single PW calculation yields both the NMR shielding (``postscf.gipaw``)
and the EFG, and both scale to supercells. It mirrors the all-electron FLAPW EFG
(``gradwave.flapw.efg``) in sign/normalization so the two paths cross-validate directly.

Physics [Petrilli, Blöchl, Margl & Schwarz, PRB 57, 14690 (1998)]
----------------------------------------------------------------
Within the PAW frozen-core reconstruction the all-electron valence density is

    n(r) = ñ(r) + Σ_R [ n¹_R(r) − ñ¹_R(r) ],

with ñ the smooth pseudo valence density, n¹ the on-site all-electron one-centre density and
ñ¹ its pseudo counterpart. The EFG is the field of this (electron) density plus the ionic
point charges +Z_val, and because the operator (3r̂_a r̂_b − δ_ab)/r³ is linear in the density
it splits into three additive site tensors:

1. **Lattice / plane-wave part.** From the SMOOTH pseudo valence density ñ and the ionic point
   charges. The electron piece is an absolutely-convergent reciprocal sum over the smooth
   density (which decays in G),

       V^smooth_ab(R) = −4π e² Σ_{G≠0} ñ(G) (G_a G_b/G² − δ_ab/3) e^{iG·R},

   and the ionic piece is the traceless field gradient of the +Z_val point-charge lattice,
   done by an Ewald split (:func:`ionic_efg`) because the bare point-charge G-sum is only
   conditionally convergent. CRITICAL: the stored ``USPPResult.rho`` is the TOTAL density
   ñ + n̂ (it already carries the PAW compensation/augmentation charge n̂ — the classic EFG
   sign/double-count trap). The compensation n̂ is an auxiliary charge for the Hartree energy,
   NOT part of the physical density, so it is removed here (``ñ = rho − n̂`` via
   ``uspp_frozen.aug_density_from_becsum``); the on-site term (2) is then the clean AE−PS
   difference with no compensation piece to subtract.

2. **PAW on-site correction.** The l=2 field of the AE−PS one-centre density difference,

       V^onsite_ab(R) = e² Σ_ij ρ_ij M^ab_{I_i I_j} [ ⟨rφ_i|1/r³|rφ_j⟩ − ⟨rφ̃_i|1/r³|rφ̃_j⟩ ],

   with the on-site density matrix ρ_ij (the PAW becsum ``USPPResult.rho_ij_atoms``), the l=2
   angular tensor M^ab_IJ = ∫ Y_I(Ω)(3r̂_a r̂_b − δ_ab)Y_J(Ω) dΩ over the real spherical
   harmonics in which the becsum is indexed (:func:`_efg_angular_tensor`), and the radial AE−PS
   ⟨1/r³⟩ difference (:class:`EFGOnSite`, the ``gipaw.PAWOnSite`` partial-wave/r³ bridge reused,
   with the EFG l=2 selection ``|l_i−l_j|≤2≤l_i+l_j``, ``l_i+l_j`` even). A spherical (closed-
   shell) ρ_ij gives zero: no l=2 asphericity, no on-site EFG.

3. **Frozen core → zero.** A spherical frozen core carries no l=2 moment and contributes no EFG
   at its own nucleus; its monopole is folded into the +Z_val ionic point charge of the other
   sites. (Aspherical frozen-core polarization — Sternheimer antishielding — is a known
   ≤10–20% effect for 3d cations; documented, not implemented, exactly as in the FLAPW path.)

Convention (locked to ``flapw.efg``): V is in eV/Å² with the electron number density entering
as +e² (the Hartree-of-electron-density sign), verified against the FLAPW ``efg_tensor`` on a
p_z state (both give V_zz = e²·(4/5)∫R²/r dr). ``V_zz`` is the largest-magnitude eigenvalue,
``η = |V_xx−V_yy|/|V_zz|`` ∈ [0,1]. ``C_Q`` uses the same isotope table and prefactor as
``flapw.nmr.quadrupolar_coupling`` (``C_Q[MHz] = 2.4180·Q[barn]·V_zz[eV/Å²]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from gradwave.constants import E2
from gradwave.core.energies.ewald import _ACC, _g_vectors, _image_vectors, _max_pair_offset
from gradwave.core.fftbox import r_to_g
from gradwave.core.gaunt import ylm_np
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.flapw.nmr import NUCLEAR_Q, quadrupolar_coupling
from gradwave.postscf.gipaw import PAWOnSite, _angular_grid
from gradwave.postscf.uspp_frozen import aug_density_from_becsum, screen_phase
from gradwave.pseudo.upf_paw import PAWData

if TYPE_CHECKING:
    from gradwave.scf.results import USPPResult
    from gradwave.scf.uspp_setup import USPPSystem


# ---------------------------------------------------------------------------
# tensor observables
# ---------------------------------------------------------------------------
def _tensor_observables(v: np.ndarray) -> tuple[float, float]:
    """``(V_zz, η)`` from a symmetric traceless 3×3 EFG tensor, ``|V_xx|≤|V_yy|≤|V_zz|`` and
    ``η = |V_xx−V_yy|/|V_zz|`` — same principal-axis convention as ``flapw.efg._tensor_from_v``."""
    w = np.linalg.eigvalsh(np.asarray(v, dtype=float))
    order = np.argsort(np.abs(w))
    v_xx, v_yy, v_zz = w[order[0]], w[order[1]], w[order[2]]
    eta = abs((v_xx - v_yy) / v_zz) if abs(v_zz) > 1e-30 else 0.0
    return float(v_zz), float(eta)


def _traceless(v: np.ndarray) -> np.ndarray:
    """Symmetric traceless part of a 3×3 tensor (the physical EFG; the trace is the local
    charge density by Poisson and is not observed)."""
    v = 0.5 * (v + v.T)
    return v - np.eye(3) * (np.trace(v) / 3.0)


# ---------------------------------------------------------------------------
# on-site PAW correction
# ---------------------------------------------------------------------------
def _efg_angular_tensor(lmax: int, nx: int = 20, nphi: int = 40) -> np.ndarray:
    """``M[a, b, I, J] = ∫ Y_I(Ω)(3 r̂_a r̂_b − δ_ab) Y_J(Ω) dΩ`` over the REAL spherical
    harmonics (``core.gaunt.ylm_np`` convention, index l²+m), I, J ≤ (lmax+1)².

    The angular factor of the on-site EFG operator (l=2 spherical tensor). Direct Gauss-
    Legendre(cosθ)×uniform(φ) quadrature, exact for the band-limited integrand (|Y_I Y_J| has
    degree 2·lmax, the r̂r̂ factor adds 2). Traceless in (a, b): Σ_a M[a,a] = ∫Y_I(3−3)Y_J = 0."""
    rhat, wgt = _angular_grid(nx, nphi)
    nlm = (lmax + 1) ** 2
    y = ylm_np(lmax, rhat)[:, :nlm]  # (npt, nlm)
    ang = 3.0 * rhat[:, :, None] * rhat[:, None, :] - np.eye(3)[None]  # (npt, 3, 3)
    return np.einsum("p,pab,pI,pJ->abIJ", wgt, ang, y, y)


@dataclass(frozen=True)
class EFGOnSite:
    """PAW on-site EFG operator for one species — the ``gipaw.PAWOnSite`` partial-wave/radial
    bridge specialised to the l=2 field gradient.

    Reuses ``PAWOnSite`` to read the AE/PS partial waves (r·φ_i), the (channel, l, m) projector
    index map matching the becsum layout, and the radial mesh; adds the EFG-specific radial
    ⟨1/r³⟩ AE−PS difference (masked to the l=2-coupling channels ``|l_i−l_j|≤2≤l_i+l_j``,
    ``l_i+l_j`` even — which also discards the 1/r³-singular s–s pair, whose angular factor is
    zero anyway) and the l=2 angular tensor :func:`_efg_angular_tensor`."""

    element: str
    idx: list[tuple[int, int, int]]  # (channel, l, m-slot) per becsum column
    r3_diff: np.ndarray  # (nproj, nproj) EFG-masked ∫[(rφ_i)(rφ_j) − (rφ̃_i)(rφ̃_j)]/r³ dr
    ang: np.ndarray  # (3, 3, nlm, nlm) angular tensor M^ab_IJ

    @property
    def n_mexp(self) -> int:
        """Number of m-expanded projector columns (= becsum dimension)."""
        return len(self.idx)

    @classmethod
    def from_paw(cls, paw: PAWData) -> EFGOnSite:
        # A bare ultrasoft / GBRV dataset (is_paw=False) carries no AE/PS partial waves, so the
        # Petrilli-Blöchl on-site l=2 term cannot be reconstructed — and the raw failure is an
        # opaque IndexError deep in the index-map read. Fail early with an actionable message:
        # EFG requires a PAW dataset (measured — see experiments/autoapw/efg_eta_anion.md Front A).
        if not paw.is_paw or not paw.aewfc:
            raise ValueError(
                f"efg_paw needs a PAW dataset for '{paw.element}': this dataset has "
                f"is_paw={paw.is_paw} and {len(paw.aewfc)} AE partial waves, so it carries no "
                "on-site l=2 density for the electric-field-gradient. Use a PAW (kjpaw) "
                "pseudopotential, not a bare ultrasoft/GBRV one.")
        base = PAWOnSite.from_paw(paw)  # reuse the partial-wave / index-map reading
        r = base.r
        rab = base.rab
        ch_l = base.ch_l
        w3 = (rab / r**3)[None, :]
        r3_full = (base.rphi_ae * w3) @ base.rphi_ae.T - (base.rphi_ps * w3) @ base.rphi_ps.T
        li = ch_l[:, None]
        lj = ch_l[None, :]
        couple = (np.abs(li - lj) <= 2) & (li + lj >= 2) & ((li + lj) % 2 == 0)
        r3_diff = r3_full * couple
        lmax = int(ch_l.max()) if len(ch_l) else 0
        return cls(element=base.element, idx=base.idx, r3_diff=r3_diff,
                   ang=_efg_angular_tensor(lmax))

    def tensor(self, becsum: Tensor) -> Tensor:
        """On-site EFG tensor (3, 3) [eV/Å²] for one atom from its on-site density matrix
        ``becsum`` (m-expanded ``rho_ij_atoms`` layout):

            V^onsite_ab = e² Σ_pq Re(ρ_pq) M^ab_{lm(p),lm(q)} R³_{ch(p),ch(q)}.

        ρ is Hermitian and M·R³ real-symmetric, so only Re ρ contributes. Traceless in (a, b)
        (M is), symmetric, and zero for a spherical (m-diagonal, l=0) ρ."""
        d = np.asarray(becsum.detach().real.cpu().numpy(), dtype=float)
        n = self.n_mexp
        if d.shape != (n, n):
            raise ValueError(
                f"becsum shape {d.shape} != m-expanded projector count {n} for {self.element}")
        lm = np.array([l * l + m for (_, l, m) in self.idx], dtype=int)
        ch = np.array([i for (i, _, _) in self.idx], dtype=int)
        radial = self.r3_diff[ch[:, None], ch[None, :]]  # (n, n)
        ang = self.ang[:, :, lm[:, None], lm[None, :]]  # (3, 3, n, n)
        v = E2 * np.einsum("pq,abpq->ab", d, ang * radial[None, None])
        return torch.as_tensor(_traceless(v), dtype=RDTYPE)


# ---------------------------------------------------------------------------
# lattice / plane-wave parts
# ---------------------------------------------------------------------------
def smooth_density_efg(
    rho_g: Tensor, g_cart: Tensor, g2: Tensor, positions: Tensor
) -> Tensor:
    """Reciprocal-space EFG of the smooth (pseudo) valence density at every atomic site.

        V^smooth_ab(R) = −4π e² Σ_{G≠0} ñ(G) (G_a G_b/G² − δ_ab/3) e^{iG·R}.

    ``rho_g`` = ñ(G) on the dense FFT box (``core.fftbox.r_to_g`` of the SMOOTH density ñ, i.e.
    ``rho − n̂``), coefficients in e/Å³ (physics Fourier-series convention ñ(r)=Σ_G ñ(G)e^{iG·r}).
    ``g_cart`` (…, 3) and ``g2`` (…) are ``grid.g_cart``/``grid.g2`` [Å⁻¹, Å⁻²]; ``positions``
    (nsite, 3) Cartesian Å. Returns (nsite, 3, 3) [eV/Å²]. The −4π e² and the minus sign are the
    electron (charge −ñ) contribution in the module's +e²-electron convention; the smooth density
    decays in G so this sum converges absolutely (no Ewald needed for the electron part)."""
    gc = g_cart.reshape(-1, 3).to(RDTYPE)
    g2f = g2.reshape(-1).to(RDTYPE)
    ng = rho_g.reshape(-1).to(CDTYPE)  # ñ(G), box-order matching g_cart/g2
    keep = g2f > 1e-12
    gc = gc[keep]
    g2f = g2f[keep]
    ng = ng[keep]
    # traceless projector P_ab(G) = G_a G_b/G² − δ_ab/3
    proj = gc[:, :, None] * gc[:, None, :] / g2f[:, None, None]
    proj = proj - torch.eye(3, dtype=RDTYPE, device=proj.device)[None] / 3.0
    phase = positions.to(RDTYPE) @ gc.T  # (nsite, ng)
    weight = torch.complex(torch.cos(phase), torch.sin(phase)) * ng[None, :]  # ñ(G) e^{iG·R}
    v = -4.0 * math.pi * E2 * torch.einsum("sg,gab->sab", weight, proj.to(CDTYPE)).real
    return v.to(RDTYPE)


def ionic_efg(
    positions: np.ndarray, charges: np.ndarray, cell: np.ndarray, eta: float | None = None
) -> np.ndarray:
    """Traceless EFG of the ionic +Z_val point-charge lattice at every site, by Ewald splitting.

        V^ion_ab(R_κ) = −e² Σ'_{j,R} Z_j T^real_ab(R_κ − R_j − R)
                        + e² (4π/Ω) Σ_{G≠0} (G_a G_b/G²) e^{−G²/4β²} Σ_j Z_j cos(G·(R_κ − R_j)),

    the real-space (erfc) Hessian ``T^real`` and the reciprocal (erf) Hessian, with the exact self
    pair (R_κ − R_j − R = 0) excluded from the real sum; the reciprocal self term is isotropic and
    removed by the final traceless projection. Reuses the ``core.energies.ewald`` image/G lists and
    splitting parameter (β = √η, η = (π/Ω^{1/3})² by default) for a machine-precision result.

    ``positions`` (na, 3) Cartesian Å, ``charges`` (na,) valence Z (positive), ``cell`` (3, 3) Å
    rows. Returns (na, 3, 3) [eV/Å²]. Validated against a direct real-space point-charge lattice
    sum (:mod:`tests`)."""
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    cell = np.asarray(cell, dtype=float)
    na = positions.shape[0]
    omega = abs(np.linalg.det(cell))
    if eta is None:
        eta = (math.pi / omega ** (1.0 / 3.0)) ** 2
    beta = math.sqrt(eta)
    rcut = _ACC / beta
    gcut = 2.0 * beta * _ACC
    pad = _max_pair_offset(positions)
    images = _image_vectors(cell, rcut + pad)  # (nR, 3), includes R=0
    gvecs = _g_vectors(cell, gcut)  # (nG, 3), excludes G=0
    g2 = np.einsum("gi,gi->g", gvecs, gvecs)
    g_exp = np.exp(-g2 / (4.0 * eta)) / g2  # (nG,)
    sqrt_pi = math.sqrt(math.pi)

    out = np.zeros((na, 3, 3), dtype=float)
    for k in range(na):
        # ---- real space: d[j, R] = R_k − R_j − R ----
        d = positions[k][None, None, :] - positions[:, None, :] - images[None, :, :]  # (na,nR,3)
        r2 = np.einsum("jra,jra->jr", d, d)
        alive = r2 > 1e-12
        r = np.sqrt(np.where(alive, r2, 1.0))
        erfc = np.where(alive, _erfc(beta * r), 0.0)
        eg = np.where(alive, np.exp(-(beta**2) * r2), 0.0)
        # T^real coefficients: d_a d_b · c_dd − δ_ab · c_id
        c_dd = 3.0 * erfc / r**5 + (2.0 * beta / sqrt_pi) * eg * (3.0 / r**4 + 2.0 * beta**2 / r**2)
        c_id = erfc / r**3 + (2.0 * beta / sqrt_pi) * eg / r**2
        c_dd = np.where(alive, c_dd, 0.0)
        c_id = np.where(alive, c_id, 0.0)
        zc_dd = charges[:, None] * c_dd  # (na, nR)
        zc_id = charges[:, None] * c_id
        v_real = np.einsum("jr,jra,jrb->ab", zc_dd, d, d)
        v_real = v_real - np.eye(3) * float(np.einsum("jr->", zc_id))
        v_k = -E2 * v_real
        # ---- reciprocal space ----
        gd = gvecs @ (positions[k][:, None] - positions.T)  # (nG, na)
        s = (charges[None, :] * np.cos(gd)).sum(axis=1)  # (nG,)
        v_recip = np.einsum("g,ga,gb->ab", s * g_exp, gvecs, gvecs)
        v_k = v_k + E2 * (4.0 * math.pi / omega) * v_recip
        out[k] = _traceless(v_k)
    return out


def _erfc(x: np.ndarray) -> np.ndarray:
    """Vectorised complementary error function (SciPy if available, else a torch fallback) so the
    ionic Ewald stays numpy-native like ``core.energies.ewald``."""
    try:
        from scipy.special import erfc as _sp_erfc

        return np.asarray(_sp_erfc(x), dtype=float)
    except ImportError:  # pragma: no cover - scipy is a hard dep, guard only for safety
        return torch.erfc(torch.as_tensor(x, dtype=RDTYPE)).cpu().numpy()


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def _sum_spin_becsum(
    rho_ij_atoms: list[Tensor] | list[list[Tensor]], nspin: int
) -> list[Tensor]:
    """Per-atom total becsum (summed over spin channels). ``rho_ij_atoms`` is a flat per-atom list
    for nspin=1, or ``[spin][atom]`` for nspin=2; the EFG uses the total (spin-summed) density
    matrix (the physical charge, not the magnetisation)."""
    if nspin == 2:
        spin_lists = cast("list[list[Tensor]]", rho_ij_atoms)  # [spin][atom]
        return [spin_lists[0][a] + spin_lists[1][a] for a in range(len(spin_lists[0]))]
    return list(cast("list[Tensor]", rho_ij_atoms))


def efg_paw(
    result: USPPResult | dict[str, Any],
    *,
    isotopes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per-site PAW electric field gradient of a converged USPP/PAW ground state.

    ``result`` is a ``USPPResult`` (or its dict view) from ``scf.uspp_loop.scf_uspp``; it must be a
    PAW run (partial waves present) for the on-site term. ``isotopes`` optionally maps a chemical
    element to the NMR isotope to report ``C_Q`` for (e.g. ``{"O": "17O", "Al": "27Al"}``); sites
    whose element is absent from the map, or whose isotope is unknown to
    ``flapw.nmr.NUCLEAR_Q``, simply omit the ``C_Q`` entry.

    Returns one dict per atom with:

    - ``element``, ``site`` (index),
    - ``V`` (3, 3) traceless EFG tensor [eV/Å²] and its pieces ``V_smooth`` / ``V_ion`` /
      ``V_onsite`` (for forensics / cross-validation),
    - ``V_zz`` [eV/Å²] (largest-magnitude eigenvalue) and ``eta`` ∈ [0, 1],
    - ``C_Q`` (a ``quadrupolar_coupling`` dict) when an isotope is available.
    """
    system: USPPSystem = result["system"]
    rho: Tensor = result["rho"]
    nspin = int(result.get("nspin", 1) or 1)
    becsum = _sum_spin_becsum(result["rho_ij_atoms"], nspin)

    grid = system.grid
    positions = system.positions
    species_of_atom = list(system.species_of_atom)
    paws: list[PAWData] = list(system.paws)

    # smooth density ñ = rho − n̂ (drop the PAW compensation charge; it is not physical density)
    n_aug = aug_density_from_becsum(system, [b.to(CDTYPE) for b in becsum], screen_phase(system))
    n_smooth = rho.to(RDTYPE) - n_aug.to(RDTYPE)
    rho_g = r_to_g(n_smooth.to(CDTYPE))

    v_smooth = smooth_density_efg(rho_g, grid.g_cart, grid.g2, positions)  # (nsite, 3, 3)
    v_ion = ionic_efg(
        positions.detach().cpu().numpy(),
        system.charges.detach().cpu().numpy(),
        np.asarray(grid.cell, dtype=float),
    )  # (nsite, 3, 3)

    onsites = {sp: EFGOnSite.from_paw(p) for sp, p in enumerate(paws)}

    out: list[dict[str, Any]] = []
    for a, sp in enumerate(species_of_atom):
        v_os = onsites[sp].tensor(becsum[a]).detach().cpu().numpy()
        v_sm = v_smooth[a].detach().cpu().numpy()
        v_io = v_ion[a]
        v_tot = _traceless(v_sm + v_io + v_os)
        v_zz, eta = _tensor_observables(v_tot)
        element = paws[sp].element
        entry: dict[str, Any] = {
            "element": element,
            "site": a,
            "V": torch.as_tensor(v_tot, dtype=RDTYPE),
            "V_smooth": torch.as_tensor(_traceless(v_sm), dtype=RDTYPE),
            "V_ion": torch.as_tensor(v_io, dtype=RDTYPE),
            "V_onsite": torch.as_tensor(v_os, dtype=RDTYPE),
            "V_zz": v_zz,
            "eta": eta,
        }
        isotope = None if isotopes is None else isotopes.get(element)
        if isotope is not None and isotope in NUCLEAR_Q:
            entry["C_Q"] = quadrupolar_coupling(v_zz, eta, isotope)
        out.append(entry)
    return out
