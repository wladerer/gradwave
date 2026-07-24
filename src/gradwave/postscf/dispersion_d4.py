"""Grimme DFT-D4 dispersion with Becke–Johnson damping — D4(BJ) (molecular).

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

Everything is a differentiable function of the Cartesian positions, so forces
(−∂E/∂τ) come straight from autograd. **Scope:** this module implements the
*molecular* (non-periodic) case only. A periodic EEQ needs an Ewald treatment
of the long-range erf-Coulomb charge matrix (the erf(γr)/r tail is ~1/r); that,
and the resulting stress, are deferred (see the D4 PR / dispersion notebook).

Units: positions in Å, energy in eV. Reference tables (``_d4_params``) are
atomic (Bohr, Hartree); conversion happens at this boundary only. Reference:
Caldeweyher, Bannwarth, Grimme, J. Chem. Phys. 150, 154122 (2019).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.dtypes import RDTYPE
from gradwave.postscf import _d4_params as P


@dataclass(frozen=True)
class D4Config:
    """Resolved D4(BJ) damping constants (atomic units) and total charge.

    Build from a functional name with :meth:`from_functional`, or pass the four
    damping constants directly. ``charge`` is the total molecular charge fed to
    the EEQ model.
    """

    s6: float
    s8: float
    a1: float
    a2: float  # Bohr
    charge: float = 0.0

    @classmethod
    def from_functional(cls, functional: str, *, charge: float = 0.0,
                        **overrides: float) -> D4Config:
        key = functional.lower().replace("_", "-")
        if key not in P.BJ_PARAMS:
            raise ValueError(
                f"no D4(BJ) parameters vendored for functional {functional!r}; "
                f"available: {sorted(P.BJ_PARAMS)}"
            )
        s6, s8, a1, a2 = P.BJ_PARAMS[key]
        kw = dict(s6=s6, s8=s8, a1=a1, a2=a2, charge=charge)
        kw.update(overrides)
        return cls(**kw)


# ---------------------------------------------------------------------------
# reference-table assembly (per-atom padded tensors)
# ---------------------------------------------------------------------------

def _covered_elements() -> set[int]:
    return set(P.NREF)


def _check_coverage(atomic_numbers) -> None:
    missing = sorted({int(z) for z in atomic_numbers} - _covered_elements())
    if missing:
        raise NotImplementedError(
            f"D4(BJ) reference data not vendored for element(s) Z={missing}; "
            f"covered subset is Z={sorted(_covered_elements())}. Extend "
            f"scripts/gen_d4_params.py to add them."
        )


def _element_tensors(atomic_numbers, dtype, device):
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
# coordination numbers (molecular, all pairs; differentiable in positions)
# ---------------------------------------------------------------------------

def _distances(pos_bohr: torch.Tensor):
    """(na,na) distances with the diagonal shifted off zero (kept finite)."""
    na = pos_bohr.shape[0]
    d = pos_bohr[:, None, :] - pos_bohr[None, :, :]
    eye = torch.eye(na, dtype=torch.bool, device=pos_bohr.device)
    offset = torch.zeros(3, dtype=pos_bohr.dtype, device=pos_bohr.device)
    offset[0] = 1.0
    d = d + eye[..., None].to(pos_bohr.dtype) * offset
    return torch.linalg.norm(d, dim=-1), eye


def _erf_count(r, r0, kcn):
    return 0.5 * (1.0 + torch.erf(-kcn * (r / r0 - 1.0)))


def _cn_d4(pos_bohr, rcov, en, eye):
    """D4 covalent CN: erf counting with the |EN_A−EN_B| pair weighting."""
    r, _ = _distances(pos_bohr)
    r0 = rcov[:, None] + rcov[None, :]
    endiff = torch.abs(en[:, None] - en[None, :])
    weight = P.D4_K4 * torch.exp(-((endiff + P.D4_K5) ** 2) / P.D4_K6)
    contrib = weight * _erf_count(r, r0, P.KCN_D4)
    contrib = contrib.masked_fill(eye, 0.0)
    return contrib.sum(dim=1)


def _cn_eeq(pos_bohr, rcov, eye):
    """EEQ CN: plain erf counting, smoothly capped at ``CN_EEQ_MAX``."""
    r, _ = _distances(pos_bohr)
    r0 = rcov[:, None] + rcov[None, :]
    contrib = _erf_count(r, r0, P.KCN_EEQ).masked_fill(eye, 0.0)
    cn = contrib.sum(dim=1)
    cnmax = torch.tensor(P.CN_EEQ_MAX, dtype=cn.dtype, device=cn.device)
    return torch.log1p(torch.exp(cnmax)) - torch.log1p(torch.exp(cnmax - cn))


# ---------------------------------------------------------------------------
# EEQ partial charges (bordered linear solve; differentiable in positions)
# ---------------------------------------------------------------------------

def _eeq_charges(pos_bohr, T, cn_eeq, total_charge):
    """Solve the constrained EEQ system for partial charges q (na,)."""
    na = pos_bohr.shape[0]
    dt, dev = pos_bohr.dtype, pos_bohr.device
    r, eye = _distances(pos_bohr)

    rad = T["rad"]
    gamma = 1.0 / torch.sqrt(rad[:, None] ** 2 + rad[None, :] ** 2)
    off = torch.erf(gamma * r) / r
    diag = T["eta"] + P.SQRT_2_OVER_PI / rad
    A = torch.where(eye, diag.diag_embed(), off)

    # bordered system with the total-charge Lagrange constraint
    A_full = torch.zeros((na + 1, na + 1), dtype=dt, device=dev)
    A_full[:na, :na] = A
    A_full[:na, na] = 1.0
    A_full[na, :na] = 1.0

    x = torch.empty(na + 1, dtype=dt, device=dev)
    rhs = -T["chi"] + T["kcn"] * torch.sqrt(cn_eeq)
    x = torch.cat([rhs, torch.tensor([total_charge], dtype=dt, device=dev)])
    sol = torch.linalg.solve(A_full, x)
    return sol[:na]


# ---------------------------------------------------------------------------
# charge-dependent reference weights and atomic C6
# ---------------------------------------------------------------------------

def _zeta(gam, qref, qmod):
    """ζ charge scaling (``gam`` already premultiplied by gc)."""
    eps = torch.finfo(gam.dtype).eps
    scale = torch.exp(gam * (1.0 - qref / (qmod - eps)))
    return torch.where(qmod > 0.0, torch.exp(P.GA * (1.0 - scale)),
                       torch.exp(torch.tensor(P.GA, dtype=gam.dtype, device=gam.device)))


def _reference_weights(covcn, q, T):
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


def _atomic_c6(weights, c6ref):
    """C6_AB = Σ_ab W_a^A W_b^B C6ref_ab, (na,na)."""
    return torch.einsum("ia,jb,ijab->ij", weights, weights, c6ref)


# ---------------------------------------------------------------------------
# energy (the differentiable core)
# ---------------------------------------------------------------------------

def dispersion_energy(
    positions: torch.Tensor,
    cell,
    atomic_numbers,
    cfg: D4Config,
) -> torch.Tensor:
    """D4(BJ) dispersion energy [eV], differentiable in ``positions``.

    positions (na,3) Å. ``cell`` must be ``None`` (molecular): the periodic EEQ
    (Ewald) is out of scope.
    """
    if cell is not None:
        raise NotImplementedError(
            "D4(BJ) currently supports molecules only (cell=None); the periodic "
            "EEQ charge model requires an Ewald sum of the erf-Coulomb matrix."
        )
    dev, dt = positions.device, positions.dtype
    pos_b = positions / BOHR_ANG

    T = _element_tensors(atomic_numbers, dt, dev)
    _, eye = _distances(pos_b)

    cn_eeq = _cn_eeq(pos_b, T["rcov"], eye)
    q = _eeq_charges(pos_b, T, cn_eeq, cfg.charge)
    covcn = _cn_d4(pos_b, T["rcov"], T["en"], eye)

    weights = _reference_weights(covcn, q, T)
    c6 = _atomic_c6(weights, T["c6ref"])  # (na,na)
    c8 = 3.0 * c6 * T["r4r2"][:, None] * T["r4r2"][None, :]
    r0 = cfg.a1 * torch.sqrt(c8 / c6) + cfg.a2

    r, _ = _distances(pos_b)
    r6 = r ** 6
    r8 = r6 * r * r
    e6 = c6 / (r6 + r0 ** 6)
    e8 = c8 / (r8 + r0 ** 8)
    term = (cfg.s6 * e6 + cfg.s8 * e8).masked_fill(eye, 0.0)
    e_hartree = -0.5 * term.sum()
    return e_hartree * HARTREE_EV


def eeq_charges(positions, cell, atomic_numbers, charge: float = 0.0) -> torch.Tensor:
    """EEQ partial charges (na,) [e], differentiable in ``positions`` (molecular)."""
    if cell is not None:
        raise NotImplementedError("EEQ charges: molecular (cell=None) only for now.")
    dt, dev = positions.dtype, positions.device
    pos_b = positions / BOHR_ANG
    T = _element_tensors(atomic_numbers, dt, dev)
    _, eye = _distances(pos_b)
    cn_eeq = _cn_eeq(pos_b, T["rcov"], eye)
    return _eeq_charges(pos_b, T, cn_eeq, charge)


# ---------------------------------------------------------------------------
# forces (autograd wrapper — mirrors postscf/dispersion.py)
# ---------------------------------------------------------------------------

def dispersion_forces(positions, cell, atomic_numbers, cfg: D4Config) -> torch.Tensor:
    """F_A = −∂E_disp/∂τ_A, (na,3) [eV/Å], via autograd (molecular)."""
    if cell is not None:
        raise NotImplementedError("D4(BJ) forces: molecular (cell=None) only for now.")
    pos = positions.detach().clone().to(RDTYPE).requires_grad_(True)
    e = dispersion_energy(pos, None, atomic_numbers, cfg)
    (grad,) = torch.autograd.grad(e, pos)
    return -grad
