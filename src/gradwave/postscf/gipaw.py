"""GIPAW ground-state shielding terms (Phase 1): frozen-core Lamb + diamagnetic
augmentation, on the PAW partial-wave path.

The merged ``postscf.kgeometry_nmr.sigma_shielding`` computes the *bare* (pseudo,
smooth-valence) NMR chemical shielding. The full GIPAW shielding is

    σ = σ_bare + σ_core + σ_aug,   σ_aug = σ_dia_aug + σ_para_aug,

with σ_core the frozen-core diamagnetic (Lamb) constant, σ_dia_aug the on-site
AE−PS *ground-state* diamagnetic correction, and σ_para_aug the on-site
*paramagnetic response* (the hard, magnetic-linear-response term — Phase 2). This
module builds the two ground-state pieces and the PAW partial-wave/projector
plumbing (the "PAW-path bridge") that Phase 2's paramagnetic term reuses.

Physics
-------
**Frozen-core Lamb term** [Lamb, Phys. Rev. 60, 817 (1941)]. A closed-shell
frozen core carries no first-order paramagnetic current, so its shielding is
exactly the diamagnetic integral over the all-electron core density:

    σ_core = (1/3)(e²/mc²) ∫ ρ_core(r)/r d³r
           = (1/3) r_e · 4π ∫₀^∞ ρ_core(r) r dr,

with r_e = e²/mc² = 2·α²·(ℏ²/2m)/e² the classical electron radius (all from
``gradwave.constants``). A per-element constant; it cancels in chemical-shift
*differences* but is the numerically dominant piece of the *absolute* σ (Si core
≈ 838 ppm vs a bare valence σ of tens of ppm).

**Diamagnetic augmentation** [Pickard & Mauri, PRB 63, 245101 (2001); Yates,
Pickard & Mauri, PRB 76, 024401 (2007)]. The diamagnetic shielding operator at a
nucleus (measuring r from that nucleus) is

    O^dia_αβ(r) = (e²/2mc²) [ r²δ_αβ − r_α r_β ] / r³
                = (1/2) r_e · (1/r) · [ δ_αβ − r̂_α r̂_β ],

whose isotropic trace (1/3)Tr = (1/3)r_e/r reproduces Lamb. The on-site GIPAW
correction replaces the smooth pseudo valence density by the true AE valence
density *inside* the augmentation sphere:

    σ^dia_aug_αβ(R) = Σ_ij D_ij [ ⟨φ_i|O^dia_αβ|φ_j⟩ − ⟨φ̃_i|O^dia_αβ|φ̃_j⟩ ],

with D_ij = Σ_nk f_nk ⟨ψ̃_nk|p_i⟩⟨p_j|ψ̃_nk⟩ the on-site density matrix (the PAW
"becsum", ``USPPResult.rho_ij_atoms``). Because O^dia has a common 1/r radial
weight, the matrix element factorizes into a radial AE−PS difference

    R^diff_ij = ∫ [ (rφ_i)(rφ_j) − (rφ̃_i)(rφ̃_j) ] / r dr

(compactly supported: φ = φ̃ outside the sphere) and an angular tensor
M^αβ_IJ = ∫ Y_I(Ω)[δ_αβ − r̂_α r̂_β]Y_J(Ω) dΩ over the real spherical harmonics
in which the becsum is indexed. Environment-dependent through D_ij; no magnetic
response, so FLAPW/literature-validatable as a ground-state quantity.

PAW-path bridge
---------------
:class:`PAWOnSite` reads the AE/PS partial waves (r·φ_i, r·φ̃_i), the projector
channel/l/m index map (matching ``scf.uspp_setup._mexp_index_map`` and hence the
becsum layout), the radial mesh, and precomputes the (φ−φ̃) radial differences
and the diamagnetic angular tensor. :func:`augmentation_charge_residual` is a
self-consistency check independent of any SCF: it reconstructs the parsed
augmentation charges ∫Q_ij d³r from the partial waves (a partial-wave
completeness/norm relation), agreeing to ~1e-9. Phase 2's paramagnetic term
reuses this bridge for the on-site velocity/L_R matrix elements ⟨φ_i|·|φ_j⟩.

nspin = 1, ground-state; the becsum enters as a plain per-atom matrix so this
module imports no SCF driver (it takes ``paws``/``species_of_atom``/becsum as
data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from gradwave.constants import ALPHA_FS, E2, HBAR2_2M
from gradwave.core.gaunt import ylm_np
from gradwave.dtypes import RDTYPE
from gradwave.pseudo.upf_paw import PAWData

# classical electron radius r_e = e²/mc² = 2 α² (ℏ²/2m) / e²  [Å]
R_E_ANG = 2.0 * ALPHA_FS**2 * HBAR2_2M / E2


def core_lamb_shielding(paw: PAWData) -> float:
    """Frozen-core diamagnetic (Lamb) isotropic shielding of a PAW dataset, in
    ppm — a per-element constant.

        σ_core = (1/3) r_e · 4π ∫ ρ_core(r) r dr

    over the all-electron core density ``paw.ae_core_rho`` (spherical, e/Å³) on
    the UPF radial mesh. Ground-state density integral, no magnetic response;
    cross-validates against the FLAPW all-electron-core Lamb term (Si ≈ 838 vs
    811 ppm, O ≈ 271 vs 270 ppm — a few %).
    """
    if paw.ae_core_rho is None:
        raise ValueError(
            f"{paw.element}: PAW dataset carries no AE core density "
            "(PP_AE_NLCC) — cannot form σ_core"
        )
    rho = np.asarray(paw.ae_core_rho, dtype=float)
    r = np.asarray(paw.r, dtype=float)
    rab = np.asarray(paw.rab, dtype=float)
    integral = 4.0 * math.pi * float(np.sum(rho * r * rab))  # ∫ρ_core/r d³r
    return (1.0 / 3.0) * R_E_ANG * integral * 1e6


def _index_map(paw: PAWData) -> list[tuple[int, int, int]]:
    """[(channel i, l, real-harmonic m-slot)] in projector-column order — the
    exact layout of ``scf.uspp_setup._mexp_index_map`` and hence of the becsum
    (``rho_ij_atoms``) columns."""
    out: list[tuple[int, int, int]] = []
    for i, b in enumerate(paw.betas):
        for m in range(2 * b.l + 1):
            out.append((i, b.l, m))
    return out


def _dia_angular_tensor(lmax: int, nx: int = 20, nphi: int = 40) -> np.ndarray:
    """M[α, β, I, J] = ∫ Y_I(Ω)[δ_αβ − r̂_α r̂_β]Y_J(Ω) dΩ over the REAL spherical
    harmonics (``core.gaunt.ylm_np`` convention, index l²+m), I, J ≤ (lmax+1)².

    Direct Gauss-Legendre(cosθ)×uniform(φ) quadrature (exact for the band-limited
    integrand: |Y_I Y_J| has degree 2·lmax, the r̂r̂ factor adds 2). The isotropic
    trace (1/3)Σ_α M[α,α] = (2/3)δ_IJ (used as a self-consistency check)."""
    xg, wx = np.polynomial.legendre.leggauss(nx)
    theta = np.arccos(xg)
    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    wgt = (wx[:, None] * np.full(nphi, 2.0 * np.pi / nphi)[None, :]).reshape(-1)
    st = np.sin(th).reshape(-1)
    rhat = np.stack(
        [st * np.cos(ph).reshape(-1), st * np.sin(ph).reshape(-1),
         np.cos(th).reshape(-1)],
        axis=-1,
    )  # (npt, 3)
    y = ylm_np(lmax, rhat)  # (npt, (lmax+1)²)
    nlm = (lmax + 1) ** 2
    y = y[:, :nlm]
    # ang[p, α, β] = δ_αβ − r̂_α r̂_β
    ang = np.eye(3)[None] - rhat[:, :, None] * rhat[:, None, :]  # (npt, 3, 3)
    return np.einsum("p,pab,pI,pJ->abIJ", wgt, ang, y, y)


@dataclass(frozen=True)
class PAWOnSite:
    """PAW-path bridge for the on-site GIPAW terms of one species.

    Holds the partial-wave/projector plumbing shared by σ_dia_aug (Phase 1) and
    the paramagnetic augmentation (Phase 2): AE/PS partial waves as r·φ on the
    radial mesh, the (channel, l, m) projector index map matching the becsum, and
    the precomputed diamagnetic radial AE−PS differences and angular tensor.
    """

    element: str
    r: np.ndarray  # radial mesh [Å]
    rab: np.ndarray  # radial dr weights [Å]
    aug_cutoff_idx: int
    ch_l: np.ndarray  # (nproj,) l of each beta channel
    rphi_ae: np.ndarray  # (nproj, nr) r·φ_i (AE) [Å^{-1/2}]
    rphi_ps: np.ndarray  # (nproj, nr) r·φ̃_i (PS)
    idx: list[tuple[int, int, int]]  # (channel, l, m-slot) per becsum column
    r_diff: np.ndarray  # (nproj, nproj) ∫[(rφ_i)(rφ_j) − (rφ̃_i)(rφ̃_j)]/r dr
    dia_ang: np.ndarray  # (3, 3, nlm, nlm) angular tensor M^αβ_IJ

    @property
    def n_mexp(self) -> int:
        """Number of m-expanded projector columns (= becsum dimension)."""
        return len(self.idx)

    @classmethod
    def from_paw(cls, paw: PAWData) -> PAWOnSite:
        if not paw.is_paw or not paw.aewfc:
            raise ValueError(
                f"{paw.element}: not a PAW dataset with partial waves "
                "(PP_FULL_WFC) — cannot build the on-site bridge"
            )
        r = np.asarray(paw.r, dtype=float)
        rab = np.asarray(paw.rab, dtype=float)
        nproj = paw.n_proj
        ch_l = np.array([b.l for b in paw.betas], dtype=int)
        rphi_ae = np.stack([np.asarray(w.rphi, dtype=float) for w in paw.aewfc])
        rphi_ps = np.stack([np.asarray(w.rphi, dtype=float) for w in paw.pswfc])
        # radial AE−PS ⟨1/r⟩ difference (compactly supported: φ = φ̃ for r>r_c)
        w = (rab / r)[None, :]
        r_diff = (rphi_ae * w) @ rphi_ae.T - (rphi_ps * w) @ rphi_ps.T
        lmax = int(ch_l.max()) if nproj else 0
        dia_ang = _dia_angular_tensor(lmax)
        return cls(
            element=paw.element,
            r=r,
            rab=rab,
            aug_cutoff_idx=paw.aug_cutoff_idx,
            ch_l=ch_l,
            rphi_ae=rphi_ae,
            rphi_ps=rphi_ps,
            idx=_index_map(paw),
            r_diff=r_diff,
            dia_ang=dia_ang,
        )

    def dia_aug_tensor(self, becsum: Tensor) -> Tensor:
        """Diamagnetic-augmentation shielding tensor (3, 3) [ppm] for one atom
        from its on-site density matrix ``becsum`` (m-expanded, the ``rho_ij_atoms``
        layout):

            σ^dia_aug_αβ = (1/2) r_e Σ_ab D_ab M^αβ_{lm(a),lm(b)} R^diff_{ch(a),ch(b)}.

        Hermitian D contracted with the real-symmetric M·R factor, so only Re D
        contributes; the isotropic trace equals :func:`dia_aug_iso`."""
        d = np.asarray(becsum.detach().real.cpu().numpy(), dtype=float)
        n = self.n_mexp
        if d.shape != (n, n):
            raise ValueError(
                f"becsum shape {d.shape} != m-expanded projector count {n} "
                f"for {self.element}"
            )
        lm = np.array([l * l + m for (_, l, m) in self.idx], dtype=int)
        ch = np.array([i for (i, _, _) in self.idx], dtype=int)
        radial = self.r_diff[ch[:, None], ch[None, :]]  # (n, n)
        ang = self.dia_ang[:, :, lm[:, None], lm[None, :]]  # (3, 3, n, n)
        sig = 0.5 * R_E_ANG * np.einsum("ab,ijab->ij", d, ang * radial[None, None])
        return torch.as_tensor(sig * 1e6, dtype=RDTYPE)

    def dia_aug_iso(self, becsum: Tensor) -> float:
        """Isotropic diamagnetic-augmentation shielding (1/3)Tr σ^dia_aug [ppm]
        for one atom — the literature-comparable magnitude. Equivalent to the
        direct l=0 (Lamb-like) form (1/3)r_e Σ_{a,b:(l,m)=(l',m')} D_ab R^diff."""
        return float(torch.einsum("ii->", self.dia_aug_tensor(becsum)) / 3.0)


def augmentation_charge_residual(paw: PAWData) -> float:
    """Max abs error of the partial-wave reconstruction of the parsed augmentation
    charges — a PAW-path self-consistency check independent of any SCF.

    For same-l channel pairs the monopole augmentation charge is the radial AE−PS
    overlap over the augmentation sphere,

        ∫ Q_ij d³r  =  ∫₀^{r_c} [ (rφ_i)(rφ_j) − (rφ̃_i)(rφ̃_j) ] dr  (l_i = l_j),

    which must equal the parsed ``paw.q`` (PP_Q). Agreement to ~1e-9 validates the
    partial-wave reading + radial quadrature (a completeness/norm relation) before
    any external comparison."""
    onsite = PAWOnSite.from_paw(paw)
    nc = onsite.aug_cutoff_idx
    rab = onsite.rab[:nc]
    ae = onsite.rphi_ae[:, :nc]
    ps = onsite.rphi_ps[:, :nc]
    q = np.asarray(paw.q, dtype=float)
    nproj = paw.n_proj
    max_err = 0.0
    for i in range(nproj):
        for j in range(nproj):
            if onsite.ch_l[i] != onsite.ch_l[j]:
                continue
            rec = float(np.sum((ae[i] * ae[j] - ps[i] * ps[j]) * rab))
            max_err = max(max_err, abs(rec - q[i, j]))
    return max_err


def ground_state_shielding(
    paws: list[PAWData],
    species_of_atom: list[int],
    rho_ij_atoms: list[Tensor],
) -> dict[str, Tensor]:
    """Per-site GIPAW ground-state shielding terms from a PAW SCF's becsum.

    Takes the plain PAW data (``system.paws``, ``system.species_of_atom``) and the
    per-atom on-site density matrices (``USPPResult.rho_ij_atoms``) — no SCF driver
    import. Returns, per atom:

    - ``core`` (nsite,): the frozen-core Lamb constant σ_core [ppm].
    - ``dia_aug`` (nsite, 3, 3): the diamagnetic-augmentation tensor [ppm].
    - ``dia_aug_iso`` (nsite,): its isotropic (1/3)Tr [ppm].

    The full GIPAW σ is σ_bare (``kgeometry_nmr.sigma_shielding``) + core + dia_aug
    + the Phase-2 paramagnetic augmentation. Add ``core`` and ``dia_aug`` to the
    bare tensor (see ``sigma_shielding(core_ppm=...)``)."""
    onsites = {sp: PAWOnSite.from_paw(p) for sp, p in enumerate(paws)}
    core_by_sp = {sp: core_lamb_shielding(p) for sp, p in enumerate(paws)}
    nsite = len(species_of_atom)
    core = torch.empty(nsite, dtype=RDTYPE)
    dia = torch.empty(nsite, 3, 3, dtype=RDTYPE)
    dia_iso = torch.empty(nsite, dtype=RDTYPE)
    for a, sp in enumerate(species_of_atom):
        core[a] = core_by_sp[sp]
        t = onsites[sp].dia_aug_tensor(rho_ij_atoms[a])
        dia[a] = t
        dia_iso[a] = torch.einsum("ii->", t) / 3.0
    return {"core": core, "dia_aug": dia, "dia_aug_iso": dia_iso}
