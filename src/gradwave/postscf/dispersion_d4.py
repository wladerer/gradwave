"""Grimme DFT-D4 dispersion with Becke–Johnson damping — D4(BJ).

D4 extends D3(BJ) with *charge-dependent* C6 coefficients: the reference
dynamic polarizabilities are reweighted by a classical electronegativity-
equilibration (EEQ) partial charge through a ζ scaling. Two ingredients change
relative to ``dispersion.py`` (D3):

  1. **EEQ charges.** Solve the bordered linear system

         [ A  1 ] [ q ]   [ X ]
         [ 1ᵀ 0 ] [ λ ] = [ Q ]

     with X_A = −χ_A + κ_A √(CN^EEQ_A), diagonal A_AA = η_A + √(2/π)/α_A, and
     off-diagonal A_AB = erf(γ_AB r_AB)/r_AB, γ_AB = 1/√(α_A²+α_B²). CN^EEQ is
     the plain error-function coordination number (no EN weighting), smoothly
     capped at 8.

  2. **Charge-dependent C6.** C6_AB = Σ_ab W_a^A W_b^B C6ref_ab, with
     W_a^A = ζ_a(q_A) · gw_a, gw_a the Gaussian CN weight (in the D4 covalent
     CN, which *does* carry the |EN_A−EN_B| pair weighting) and

         ζ(q) = exp{ ga·[1 − exp( gc·η·(1 − q_ref/q_mod) )] },
         q_mod = q + Z_eff,   q_ref = q_ref,a + Z_eff.

     The pairwise reference C6 (``C6REF``) are charge-independent and are
     vendored precomputed (see ``scripts/gen_d4_params.py``).

The two-body BJ energy is then identical in form to D3:

    E_disp = −½ Σ'_{A,B} [ s6 C6_AB/(r⁶+R0⁶) + s8 C8_AB/(r⁸+R0⁸) ],
    C8_AB = 3 C6_AB √(Q_A Q_B),  R0_AB = a1 √(C8_AB/C6_AB) + a2.

Everything is a differentiable function of the Cartesian positions **and the
cell**, so forces (−∂E/∂τ) and stress ((1/Ω)∂E/∂ε) come straight from autograd.

**Periodic EEQ (Ewald of the erf-Coulomb hardness matrix).** The off-diagonal
kernel erf(γ_AB r)/r has the bare-Coulomb 1/r tail, so its lattice sum is only
conditionally convergent and needs an Ewald split. With a splitting parameter κ
(fixed off the cell volume, exactly like ``energies/ewald.py``),

    erf(γr)/r = [erf(γr) − erf(κr)]/r  (short-ranged, real space)
              +  erf(κr)/r             (smooth tail, reciprocal space).

The reciprocal part is *identical* to the bare-Coulomb Ewald reciprocal sum
because the screening only touches short range — the 1/r tail is the same:

    A_AB^rec = (4π/Ω) Σ_{G≠0} e^{−G²/4κ²}/G² cos(G·r_AB).

The self (A=B, L=0) real-space term is the finite limit 2(γ_AA−κ)/√π; combined
with the reciprocal on-site value 2κ/√π it reconstructs √(2/π)/α_A + the atom's
own periodic-image self-interaction. A uniform-background constant −π/(κ²Ω) is
added to every A_AB; because it is a rank-1 constant on the charge block it is
absorbed by the total-charge Lagrange multiplier and leaves the charges q
invariant (so it never affects the D4 energy, which only reads q). As Ω → ∞ the
reciprocal sum and background vanish and A → erf(γ_AB r)/r: the molecular limit.

Units: positions/cell in Å, energy in eV. Reference tables (``_d4_params``) are
atomic (Bohr, Hartree); conversion happens at this boundary only. Reference:
Caldeweyher, Bannwarth, Grimme, J. Chem. Phys. 150, 154122 (2019). The periodic
EEQ Ewald mirrors the multicharge convention (Caldeweyher et al.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.dtypes import RDTYPE
from gradwave.grids import reciprocal_cell
from gradwave.postscf import _d4_params as P
from gradwave.postscf._dispersion_geom import (
    image_labels as _image_labels,
)
from gradwave.postscf._dispersion_geom import (
    pair_distances as _pair_distances,
)


class _PeriodicSetup(TypedDict):
    """Image/G labels and the Ewald kappa fixed off the detached ref_cell —
    per-key heterogeneous types (label arrays vs. the scalar kappa), so a
    plain dict[str, ...] would union them all together at every key."""

    cn_lab: np.ndarray
    e_lab: np.ndarray
    eeq_real_lab: np.ndarray
    g_lab: np.ndarray
    kappa: float

# Ewald splitting accuracy for the periodic EEQ Coulomb matrix: erfc(_EEQ_ACC)
# and e^{−_EEQ_ACC²} bound the last shell (6.0 → ~1e-16, machine converged and
# cheap for unit-cell systems). Mirrors ``energies/ewald._ACC``.
_EEQ_ACC = 6.0


@dataclass(frozen=True)
class D4Config:
    """Resolved D4(BJ) damping constants (atomic units) and total charge.

    Build from a functional name with :meth:`from_functional`, or pass the four
    damping constants directly. ``charge`` is the total molecular/cell charge fed
    to the EEQ model. ``cutoff``/``cn_cutoff`` are real-space image radii (Bohr)
    for the periodic dispersion and coordination-number sums (ignored for
    molecules).
    """

    s6: float
    s8: float
    a1: float
    a2: float  # Bohr
    charge: float = 0.0
    cutoff: float = 40.0  # Bohr (~21 Å): dispersion real-space image radius
    cn_cutoff: float = 20.0  # Bohr (~10.6 Å): coordination-number image radius

    @classmethod
    def from_functional(cls, functional: str, *, charge: float = 0.0,
                        cutoff_ang: float | None = None,
                        cn_cutoff_ang: float | None = None,
                        **overrides: float) -> D4Config:
        key = functional.lower().replace("_", "-")
        if key not in P.BJ_PARAMS:
            raise ValueError(
                f"no D4(BJ) parameters vendored for functional {functional!r}; "
                f"available: {sorted(P.BJ_PARAMS)}"
            )
        s6, s8, a1, a2 = P.BJ_PARAMS[key]
        kw = dict(s6=s6, s8=s8, a1=a1, a2=a2, charge=charge)
        if cutoff_ang is not None:
            kw["cutoff"] = cutoff_ang / BOHR_ANG
        if cn_cutoff_ang is not None:
            kw["cn_cutoff"] = cn_cutoff_ang / BOHR_ANG
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def resolve(
        cls,
        functional: str,
        *,
        charge: float = 0.0,
        cutoff_ang: float,
        cn_cutoff_ang: float,
        s6: float | None = None,
        s8: float | None = None,
        a1: float | None = None,
        a2: float | None = None,
    ) -> D4Config:
        """Build from a functional preset with optional per-constant overrides.

        Overrides use the published D4(BJ) units (a2 in Bohr). If no preset
        exists for ``functional`` all four constants must be supplied. Mirrors
        :meth:`dispersion.D3Config.resolve` so the calculator/api wiring is
        symmetric.
        """
        overrides = {"s6": s6, "s8": s8, "a1": a1, "a2": a2}
        given = {k: v for k, v in overrides.items() if v is not None}
        key = functional.lower().replace("_", "-")
        if key in P.BJ_PARAMS:
            return cls.from_functional(
                key, charge=charge, cutoff_ang=cutoff_ang,
                cn_cutoff_ang=cn_cutoff_ang, **given
            )
        if len(given) != 4:
            raise ValueError(
                f"no D4(BJ) preset for functional {functional!r}; supply all of "
                f"s6, s8, a1, a2 explicitly (given: {sorted(given)})"
            )
        return cls(
            charge=charge, cutoff=cutoff_ang / BOHR_ANG,
            cn_cutoff=cn_cutoff_ang / BOHR_ANG, **given
        )


# ---------------------------------------------------------------------------
# reference-table assembly (per-atom padded tensors)
# ---------------------------------------------------------------------------

def _covered_elements() -> set[int]:
    return set(P.NREF)


def _check_coverage(atomic_numbers: list[int]) -> None:
    missing = sorted({int(z) for z in atomic_numbers} - _covered_elements())
    if missing:
        raise NotImplementedError(
            f"D4(BJ) reference data not vendored for element(s) Z={missing}; "
            f"covered subset is Z={sorted(_covered_elements())}. Extend "
            f"scripts/gen_d4_params.py to add them."
        )


def _element_tensors(
    atomic_numbers: list[int], dtype: torch.dtype, device: torch.device
) -> dict[str, torch.Tensor]:
    """Per-atom EEQ params, element scalars, reference blocks, and pair C6.

    Returns a dict of tensors. Reference blocks are padded to ``nref_max`` with
    a boolean ``valid`` mask; ``c6ref`` is (na, na, nref_max, nref_max).
    """
    z = [int(v) for v in atomic_numbers]
    _check_coverage(z)
    na = len(z)
    nref_max = max(P.NREF[e] for e in z)

    def scalar(table):
        return torch.tensor([table[e] for e in z], dtype=dtype, device=device)

    covcn_ref = np.zeros((na, nref_max))
    refc = np.zeros((na, nref_max))
    refq = np.zeros((na, nref_max))
    valid = np.zeros((na, nref_max), dtype=bool)
    for a, e in enumerate(z):
        n = P.NREF[e]
        covcn_ref[a, :n] = P.REF_COVCN[e]
        refc[a, :n] = P.REF_C[e]
        refq[a, :n] = P.REF_Q[e]
        valid[a, :n] = True

    c6ref = np.zeros((na, na, nref_max, nref_max))
    for a, ea in enumerate(z):
        for b, eb in enumerate(z):
            block = P.C6REF[(ea, eb)]
            na_, nb_ = P.NREF[ea], P.NREF[eb]
            c6ref[a, b, :na_, :nb_] = np.asarray(block)

    t = lambda arr, dt=dtype: torch.as_tensor(arr, dtype=dt, device=device)  # noqa: E731
    return {
        "chi": scalar(P.EEQ_CHI), "eta": scalar(P.EEQ_ETA),
        "kcn": scalar(P.EEQ_KCN), "rad": scalar(P.EEQ_RAD),
        "gam": scalar(P.GAM), "zeff": scalar(P.ZEFF),
        "r4r2": scalar(P.R4R2), "rcov": scalar(P.COV_D3), "en": scalar(P.EN),
        "covcn_ref": t(covcn_ref), "refc": t(refc), "refq": t(refq),
        "valid": t(valid, torch.bool), "c6ref": t(c6ref),
    }


# ---------------------------------------------------------------------------
# real-space image enumeration (integer labels off a detached cell)
# ---------------------------------------------------------------------------

def _g_labels(cell_ang: np.ndarray, gcut_bohr: float) -> np.ndarray:
    """Integer reciprocal labels m with 0 < |m·B| ≤ gcut. (nG, 3), off a detached cell.

    B = reciprocal_cell(cell) has rows b_i (Bohr⁻¹); G = m·B. The Cartesian G are
    rebuilt on the autograd cell in :func:`_eeq_amat_periodic` so the reciprocal
    sum rides the strain graph.
    """
    cell = np.asarray(cell_ang, dtype=np.float64) / BOHR_ANG  # Bohr
    b = reciprocal_cell(cell)  # rows b_i, Bohr⁻¹
    bounds = [
        int(np.ceil(gcut_bohr * np.linalg.norm(cell[i]) / (2.0 * math.pi))) + 1
        for i in range(3)
    ]
    axes = [np.arange(-n, n + 1) for n in bounds]
    m = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    g = m @ b
    g2 = np.einsum("ij,ij->i", g, g)
    keep = (g2 > 1e-12) & (g2 <= gcut_bohr**2 + 1e-9)
    return m[keep].astype(np.int64)


# ---------------------------------------------------------------------------
# coordination numbers (image sums; differentiable in positions and cell)
# ---------------------------------------------------------------------------

def _erf_count(r: torch.Tensor, r0: torch.Tensor, kcn: float) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(-kcn * (r / r0 - 1.0)))


def _cn_d4(
    pos_bohr: torch.Tensor,
    rcov: torch.Tensor,
    en: torch.Tensor,
    images_bohr: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """D4 covalent CN: erf counting with the |EN_A−EN_B| pair weighting."""
    r, self_mask = _pair_distances(pos_bohr, images_bohr)
    r0 = (rcov[:, None] + rcov[None, :])[:, :, None]
    endiff = torch.abs(en[:, None] - en[None, :])[:, :, None]
    weight = P.D4_K4 * torch.exp(-((endiff + P.D4_K5) ** 2) / P.D4_K6)
    contrib = weight * _erf_count(r, r0, P.KCN_D4)
    drop = self_mask | (r > cutoff)
    contrib = torch.where(drop, torch.zeros_like(contrib), contrib)
    return contrib.sum(dim=(1, 2))


def _cn_eeq(
    pos_bohr: torch.Tensor, rcov: torch.Tensor, images_bohr: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """EEQ CN: plain erf counting, smoothly capped at ``CN_EEQ_MAX``."""
    r, self_mask = _pair_distances(pos_bohr, images_bohr)
    r0 = (rcov[:, None] + rcov[None, :])[:, :, None]
    contrib = _erf_count(r, r0, P.KCN_EEQ)
    drop = self_mask | (r > cutoff)
    contrib = torch.where(drop, torch.zeros_like(contrib), contrib)
    cn = contrib.sum(dim=(1, 2))
    cnmax = torch.tensor(P.CN_EEQ_MAX, dtype=cn.dtype, device=cn.device)
    return torch.log1p(torch.exp(cnmax)) - torch.log1p(torch.exp(cnmax - cn))


# ---------------------------------------------------------------------------
# EEQ hardness matrix (molecular and periodic) + bordered solve
# ---------------------------------------------------------------------------

def _eeq_amat_molecular(pos_bohr: torch.Tensor, T: dict[str, torch.Tensor]) -> torch.Tensor:
    """Molecular EEQ Coulomb matrix A (na,na): erf(γ r)/r, diag η + √(2/π)/α."""
    na = pos_bohr.shape[0]
    dt, dev = pos_bohr.dtype, pos_bohr.device
    d = pos_bohr[:, None, :] - pos_bohr[None, :, :]
    eye = torch.eye(na, dtype=torch.bool, device=dev)
    offset = torch.zeros(3, dtype=dt, device=dev)
    offset[0] = 1.0
    d = d + eye[..., None].to(dt) * offset
    r = torch.linalg.norm(d, dim=-1)

    rad = T["rad"]
    gamma = 1.0 / torch.sqrt(rad[:, None] ** 2 + rad[None, :] ** 2)
    off = torch.erf(gamma * r) / r
    diag = T["eta"] + P.SQRT_2_OVER_PI / rad
    return torch.where(eye, diag.diag_embed(), off)


def _eeq_amat_periodic(
    pos_bohr: torch.Tensor,
    T: dict[str, torch.Tensor],
    cell_bohr: torch.Tensor,
    real_images_bohr: torch.Tensor,
    g_labels_int: np.ndarray,
    kappa: float,
) -> torch.Tensor:
    """Periodic EEQ Coulomb matrix A (na,na) via Ewald of erf(γ r)/r.

    ``cell_bohr`` (3,3, on the autograd graph, rows a_i, Bohr) and ``pos_bohr``
    carry positions/cell derivatives; ``real_images_bohr`` are Cartesian lattice
    vectors rebuilt on that cell; ``g_labels_int`` are integer reciprocal labels
    (detached); ``kappa`` is the fixed Ewald splitting parameter (Bohr⁻¹).
    """
    dt, dev = pos_bohr.dtype, pos_bohr.device
    rad = T["rad"]
    gamma = 1.0 / torch.sqrt(rad[:, None] ** 2 + rad[None, :] ** 2)  # (na,na)
    omega = torch.abs(torch.linalg.det(cell_bohr))
    sqrt_pi = math.sqrt(math.pi)

    # --- real space: [erf(γr) − erf(κr)]/r, short-ranged and smooth at r→0 ---
    r, self_mask = _pair_distances(pos_bohr, real_images_bohr)  # (na,na,nR)
    ga = gamma[:, :, None]
    kernel = (torch.erf(ga * r) - torch.erf(kappa * r)) / r
    # A=B, L=0 finite limit 2(γ_AA − κ)/√π (r was shifted off 0; override it)
    self_val = (2.0 / sqrt_pi) * (gamma - kappa)  # (na,na); diagonal used only
    kernel = torch.where(self_mask, self_val[:, :, None].expand_as(kernel), kernel)
    a_real = kernel.sum(dim=2)  # (na,na)

    # --- reciprocal space: (4π/Ω) Σ_{G≠0} e^{−G²/4κ²}/G² cos(G·r_AB) ---
    b = 2.0 * math.pi * torch.linalg.inv(cell_bohr).transpose(-2, -1)  # rows b_i
    g_lab = torch.as_tensor(g_labels_int, dtype=dt, device=dev)
    gvec = g_lab @ b  # (nG,3), on graph
    g2 = (gvec * gvec).sum(-1)  # (nG,)
    rij = pos_bohr[:, None, :] - pos_bohr[None, :, :]  # (na,na,3)
    phase = rij @ gvec.T  # (na,na,nG)
    pref = torch.exp(-g2 / (4.0 * kappa * kappa)) / g2  # (nG,)
    a_recip = (4.0 * math.pi / omega) * (pref[None, None, :] * torch.cos(phase)).sum(-1)

    # --- neutralizing background (rank-1 constant; leaves charges invariant) ---
    a_bg = -math.pi / (kappa * kappa * omega)

    a = a_real + a_recip + a_bg
    a = a + torch.diag_embed(T["eta"])  # atomic hardness on the diagonal
    return a


def _eeq_solve(
    A: torch.Tensor, T: dict[str, torch.Tensor], cn_eeq: torch.Tensor, total_charge: float
) -> torch.Tensor:
    """Solve the charge-constrained bordered EEQ system for partial charges q."""
    na = A.shape[0]
    dt, dev = A.dtype, A.device
    A_full = torch.zeros((na + 1, na + 1), dtype=dt, device=dev)
    A_full[:na, :na] = A
    A_full[:na, na] = 1.0
    A_full[na, :na] = 1.0
    # floor CN inside the sqrt: √(CN) has a divergent derivative at CN→0 (an
    # isolated/low-coordination atom), so a bare sqrt(0) poisons the backward
    # pass with a 0·∞ NaN. The clamp is forward-inert for any bonded atom
    # (CN≫1e-14, so the molecular references stay bit-for-bit) and sends the
    # dead-branch gradient to 0, which is the correct CN→0 limit.
    rhs = -T["chi"] + T["kcn"] * torch.sqrt(cn_eeq.clamp_min(1e-14))
    x = torch.cat([rhs, torch.tensor([total_charge], dtype=dt, device=dev)])
    sol = torch.linalg.solve(A_full, x)
    return sol[:na]


# ---------------------------------------------------------------------------
# charge-dependent reference weights and atomic C6
# ---------------------------------------------------------------------------

def _zeta(gam: torch.Tensor, qref: torch.Tensor, qmod: torch.Tensor) -> torch.Tensor:
    """ζ charge scaling (``gam`` already premultiplied by gc)."""
    eps = torch.finfo(gam.dtype).eps
    scale = torch.exp(gam * (1.0 - qref / (qmod - eps)))
    return torch.where(qmod > 0.0, torch.exp(P.GA * (1.0 - scale)),
                       torch.exp(torch.tensor(P.GA, dtype=gam.dtype, device=gam.device)))


def _reference_weights(
    covcn: torch.Tensor, q: torch.Tensor, T: dict[str, torch.Tensor]
) -> torch.Tensor:
    """W_a^A = ζ_a(q_A) · gw_a, Gaussian CN weights normalised over references."""
    valid = T["valid"]
    dcn = covcn[:, None] - T["covcn_ref"]  # (na, nref)
    tmp = torch.exp(-dcn * dcn)
    wf = P.WF
    # refc is 1 (single Gaussian) or 3 (three-Gaussian sum), Grimme's trick
    gw1 = tmp ** wf
    gw3 = tmp ** wf + tmp ** (2 * wf) + tmp ** (3 * wf)
    gw = torch.where(T["refc"] == 3, gw3, gw1)
    gw = torch.where(valid, gw, torch.zeros_like(gw))
    norm = gw.sum(dim=1, keepdim=True)
    # fallback (all references underflow): unit weight on the max-refcovcn ref
    degenerate = norm.squeeze(1) < 1e-300
    if bool(degenerate.any()):
        masked = torch.where(valid, T["covcn_ref"], torch.full_like(T["covcn_ref"], -1.0))
        top = masked.argmax(dim=1)
        onehot = torch.zeros_like(gw)
        onehot[torch.arange(gw.shape[0]), top] = 1.0
        gw = torch.where(degenerate[:, None], onehot, gw)
        norm = torch.where(degenerate[:, None], torch.ones_like(norm), norm)
    gw = gw / norm

    zeff = T["zeff"][:, None]
    gam = T["gam"][:, None] * P.GC
    zeta = _zeta(gam, T["refq"] + zeff, q[:, None] + zeff)
    zeta = torch.where(valid, zeta, torch.zeros_like(zeta))
    return zeta * gw


def _atomic_c6(weights: torch.Tensor, c6ref: torch.Tensor) -> torch.Tensor:
    """C6_AB = Σ_ab W_a^A W_b^B C6ref_ab, (na,na)."""
    return torch.einsum("ia,jb,ijab->ij", weights, weights, c6ref)


# ---------------------------------------------------------------------------
# periodic setup helper (image labels + Ewald parameter, off a detached cell)
# ---------------------------------------------------------------------------

def _periodic_setup(
    ref_cell_ang: np.ndarray, cfg: D4Config, dtype: torch.dtype, device: torch.device
) -> _PeriodicSetup:
    """Image/G labels and the Ewald κ (all fixed off the detached ``ref_cell``)."""
    rc = np.asarray(ref_cell_ang, dtype=np.float64)
    omega_bohr = abs(np.linalg.det(rc / BOHR_ANG))
    kappa = math.pi / omega_bohr ** (1.0 / 3.0)  # Ewald splitting param, Bohr⁻¹
    rcut = _EEQ_ACC / kappa
    gcut = 2.0 * kappa * _EEQ_ACC
    return {
        "cn_lab": _image_labels(rc, cfg.cn_cutoff),
        "e_lab": _image_labels(rc, cfg.cutoff),
        "eeq_real_lab": _image_labels(rc, rcut),
        "g_lab": _g_labels(rc, gcut),
        "kappa": kappa,
    }


# ---------------------------------------------------------------------------
# energy (the differentiable core)
# ---------------------------------------------------------------------------

def dispersion_energy(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    atomic_numbers: list[int],
    cfg: D4Config,
    *,
    ref_cell: np.ndarray | None = None,
) -> torch.Tensor:
    """D4(BJ) dispersion energy [eV], differentiable in ``positions`` and ``cell``.

    positions (na,3) Å. ``cell`` (3,3) Å tensor or ``None`` (molecule). For the
    periodic case the EEQ charges come from an Ewald sum of the erf-Coulomb
    hardness matrix (see the module docstring). ``ref_cell`` (Å, detached numpy)
    fixes the integer image/G labels and the Ewald κ; defaults to ``cell`` and is
    passed by the stress path so the cell can ride the ε-graph.
    """
    dev, dt = positions.device, positions.dtype
    pos_b = positions / BOHR_ANG

    T = _element_tensors(atomic_numbers, dt, dev)

    if cell is None:
        zero_img = torch.zeros((1, 3), dtype=dt, device=dev)
        inf = math.inf
        cn_eeq = _cn_eeq(pos_b, T["rcov"], zero_img, inf)
        A = _eeq_amat_molecular(pos_b, T)
        q = _eeq_solve(A, T, cn_eeq, cfg.charge)
        covcn = _cn_d4(pos_b, T["rcov"], T["en"], zero_img, inf)
        e_images = zero_img
        e_cutoff = inf
    else:
        rc = np.asarray(ref_cell if ref_cell is not None else cell.detach().cpu().numpy())
        S = _periodic_setup(rc, cfg, dt, dev)
        cell_b = cell / BOHR_ANG  # (3,3) Bohr, rows a_i, on the autograd graph

        def cart(labels):
            lab = torch.as_tensor(labels, dtype=dt, device=dev)
            return lab @ cell_b

        cn_eeq = _cn_eeq(pos_b, T["rcov"], cart(S["cn_lab"]), cfg.cn_cutoff)
        A = _eeq_amat_periodic(pos_b, T, cell_b, cart(S["eeq_real_lab"]),
                               S["g_lab"], S["kappa"])
        q = _eeq_solve(A, T, cn_eeq, cfg.charge)
        covcn = _cn_d4(pos_b, T["rcov"], T["en"], cart(S["cn_lab"]), cfg.cn_cutoff)
        e_images = cart(S["e_lab"])
        e_cutoff = cfg.cutoff

    weights = _reference_weights(covcn, q, T)
    c6 = _atomic_c6(weights, T["c6ref"])  # (na,na)
    c8 = 3.0 * c6 * T["r4r2"][:, None] * T["r4r2"][None, :]
    r0 = cfg.a1 * torch.sqrt(c8 / c6) + cfg.a2

    r, self_mask = _pair_distances(pos_b, e_images)  # (na,na,nR)
    r2 = r * r
    r6 = r2 ** 3
    r8 = r6 * r2
    e6 = c6[:, :, None] / (r6 + (r0 ** 6)[:, :, None])
    e8 = c8[:, :, None] / (r8 + (r0 ** 8)[:, :, None])
    term = cfg.s6 * e6 + cfg.s8 * e8
    drop = self_mask | (r > e_cutoff)
    term = torch.where(drop, torch.zeros_like(term), term)
    e_hartree = -0.5 * term.sum()
    return e_hartree * HARTREE_EV


def eeq_charges(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    atomic_numbers: list[int],
    charge: float = 0.0,
    *,
    cfg: D4Config | None = None,
) -> torch.Tensor:
    """EEQ partial charges (na,) [e], differentiable in ``positions`` and ``cell``.

    Molecular (``cell=None``) or periodic. For the periodic case a ``cfg`` may be
    given to control the CN image radius; otherwise defaults are used.
    """
    dt, dev = positions.dtype, positions.device
    pos_b = positions / BOHR_ANG
    T = _element_tensors(atomic_numbers, dt, dev)
    if cell is None:
        zero_img = torch.zeros((1, 3), dtype=dt, device=dev)
        cn_eeq = _cn_eeq(pos_b, T["rcov"], zero_img, math.inf)
        A = _eeq_amat_molecular(pos_b, T)
        return _eeq_solve(A, T, cn_eeq, charge)
    cfg = cfg if cfg is not None else D4Config(s6=1.0, s8=0.0, a1=0.0, a2=0.0,
                                               charge=charge)
    rc = np.asarray(cell.detach().cpu().numpy())
    S = _periodic_setup(rc, cfg, dt, dev)
    cell_b = cell / BOHR_ANG
    lab = lambda x: torch.as_tensor(x, dtype=dt, device=dev) @ cell_b  # noqa: E731
    cn_eeq = _cn_eeq(pos_b, T["rcov"], lab(S["cn_lab"]), cfg.cn_cutoff)
    A = _eeq_amat_periodic(pos_b, T, cell_b, lab(S["eeq_real_lab"]),
                           S["g_lab"], S["kappa"])
    return _eeq_solve(A, T, cn_eeq, charge)


# ---------------------------------------------------------------------------
# forces & stress (autograd wrappers — mirror postscf/forces.py, stress.py)
# ---------------------------------------------------------------------------

def _cell_tensor(
    cell: torch.Tensor | np.ndarray | None, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    if cell is None:
        return None
    return torch.as_tensor(np.asarray(cell, dtype=np.float64), dtype=dtype, device=device)


def dispersion_forces(
    positions: torch.Tensor,
    cell: torch.Tensor | np.ndarray | None,
    atomic_numbers: list[int],
    cfg: D4Config,
) -> torch.Tensor:
    """F_A = −∂E_disp/∂τ_A, (na,3) [eV/Å], via autograd (molecular or periodic)."""
    dev = positions.device
    pos = positions.detach().clone().to(RDTYPE).requires_grad_(True)
    cell_t = _cell_tensor(cell, RDTYPE, dev)
    e = dispersion_energy(pos, cell_t, atomic_numbers, cfg)
    (grad,) = torch.autograd.grad(e, pos)
    return -grad


def dispersion_stress(
    positions: torch.Tensor, cell: torch.Tensor | np.ndarray | None,
    atomic_numbers: list[int], cfg: D4Config,
) -> torch.Tensor:
    """σ_αβ = (1/Ω) ∂E_disp/∂ε_αβ, (3,3) [eV/Å³] (tension-positive), via autograd.

    Symmetric strain ε with r→(1+ε)r, a_i→(1+ε)a_i, τ→(1+ε)τ, exactly the
    ``postscf/stress.py`` convention; integer image/G labels and the Ewald κ stay
    fixed at ε=0 (passed via ``ref_cell``).
    """
    if cell is None:
        raise ValueError("dispersion stress requires a periodic cell")
    dev = positions.device
    cell0 = np.asarray(cell, dtype=np.float64)
    omega0 = abs(np.linalg.det(cell0))
    pos0 = positions.detach().to(RDTYPE)
    a0 = torch.as_tensor(cell0, dtype=RDTYPE, device=dev)

    eps = torch.zeros(3, 3, dtype=RDTYPE, device=dev, requires_grad=True)
    f_map = torch.eye(3, dtype=RDTYPE, device=dev) + eps
    pos_e = pos0 @ f_map.T
    cell_e = a0 @ f_map.T
    e = dispersion_energy(pos_e, cell_e, atomic_numbers, cfg, ref_cell=cell0)
    (grad,) = torch.autograd.grad(e, eps)
    return 0.5 * (grad + grad.T) / omega0
