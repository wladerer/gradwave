"""Grimme DFT-D3 dispersion with Becke–Johnson damping — D3(BJ) (Layer A).

A geometric, SCF-independent pairwise correction to the total energy:

    E_disp = −½ Σ'_{A,B,L}  [ s6 C6_AB / (r^6 + f_AB^6)
                            + s8 C8_AB / (r^8 + f_AB^8) ],
    C8_AB = 3 C6_AB √(Q_A Q_B),   f_AB = a1 √(C8_AB/C6_AB) + a2   (BJ radius),

with the primed sum over ordered atom pairs and lattice images L, excluding the
A=B self term at L=0. C6_AB is interpolated from CN-resolved reference values by
a Gaussian weighting in the fractional coordination numbers CN_A (Grimme et al.,
J. Chem. Phys. 132, 154104 (2010); BJ damping: Grimme, Ehrlich, Goerigk,
J. Comput. Chem. 32, 1456 (2011)).

Everything here is a differentiable function of the Cartesian positions and the
cell, so forces (−∂E/∂τ) and stress ((1/Ω)∂E/∂ε) come straight from autograd —
exactly the pattern of ``postscf/forces.py`` and ``postscf/stress.py``. The two
guards borrowed from ``energies/ewald.py`` matter here too: integer image labels
are fixed off a detached cell (so the cell can ride the strain graph), and the
masked self-pair distance is shifted off r=0 *before* the norm so double-backward
stays finite.

Units: positions/cell in Å, energy in eV. The reference tables (``_d3_params``)
are atomic (Bohr, Hartree); conversion happens at this boundary only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.dtypes import RDTYPE
from gradwave.grids import reciprocal_cell
from gradwave.postscf._d3_params import (
    BJ_PARAMS,
    C6AB,
    K1,
    K3,
    MAXC,
    MXC,
    R2R4,
    RCOV,
)


@dataclass(frozen=True)
class D3Config:
    """Resolved D3(BJ) damping + cutoffs, all in atomic units (Bohr).

    Build from a functional name with :meth:`from_functional`, or pass the four
    damping constants directly. Cutoffs are radii for the real-space image sums.
    """

    s6: float
    s8: float
    a1: float
    a2: float  # Bohr
    cutoff: float = 40.0  # Bohr (~21 Å): dispersion real-space image radius
    cn_cutoff: float = 20.0  # Bohr (~10.6 Å): coordination-number image radius

    @classmethod
    def from_functional(
        cls,
        functional: str,
        *,
        cutoff_ang: float | None = None,
        cn_cutoff_ang: float | None = None,
        **overrides: float,
    ) -> D3Config:
        key = functional.lower().replace("_", "-")
        if key not in BJ_PARAMS:
            raise ValueError(
                f"no D3(BJ) parameters vendored for functional {functional!r}; "
                f"available: {sorted(BJ_PARAMS)}"
            )
        s6, s8, a1, a2 = BJ_PARAMS[key]
        kw = dict(s6=s6, s8=s8, a1=a1, a2=a2)
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
        cutoff_ang: float,
        cn_cutoff_ang: float,
        s6: float | None = None,
        s8: float | None = None,
        a1: float | None = None,
        a2: float | None = None,
    ) -> D3Config:
        """Build from a functional preset with optional per-constant overrides.

        Overrides use the published D3(BJ) units (a2 in Bohr). If no preset
        exists for ``functional`` all four constants must be supplied.
        """
        overrides = {"s6": s6, "s8": s8, "a1": a1, "a2": a2}
        given = {k: v for k, v in overrides.items() if v is not None}
        key = functional.lower().replace("_", "-")
        if key in BJ_PARAMS:
            return cls.from_functional(
                key, cutoff_ang=cutoff_ang, cn_cutoff_ang=cn_cutoff_ang, **given
            )
        if len(given) != 4:
            raise ValueError(
                f"no D3(BJ) preset for functional {functional!r}; supply all of "
                f"s6, s8, a1, a2 explicitly (given: {sorted(given)})"
            )
        return cls(
            cutoff=cutoff_ang / BOHR_ANG, cn_cutoff=cn_cutoff_ang / BOHR_ANG, **given
        )


# ---------------------------------------------------------------------------
# reference-table assembly (atomic-number-indexed dense tensors)
# ---------------------------------------------------------------------------

def _covered_elements() -> set[int]:
    return set(MXC)


def _check_coverage(atomic_numbers) -> None:
    missing = sorted({int(z) for z in atomic_numbers} - _covered_elements())
    if missing:
        raise NotImplementedError(
            f"D3(BJ) reference C6 not vendored for element(s) Z={missing}; "
            f"covered subset is Z={sorted(_covered_elements())}. Extend "
            f"scripts/gen_d3_params.py to add them."
        )


def _reference_tensors(atomic_numbers, dtype, device):
    """Per-atom-pair reference grids for the CN interpolation.

    Returns C6R, CN1, CN2, VALID each (na, na, MAXC, MAXC) plus per-atom rcov,
    r2r4 (na,). C6R[a,b,i,j] is the reference C6 of the (elem_a ref i, elem_b
    ref j) pair; CN1/CN2 the associated reference coordination numbers.
    """
    z = [int(v) for v in atomic_numbers]
    _check_coverage(z)
    zmax = max(z)
    # dense Z-indexed lookup (small: zmax≤~30 in the vendored subset)
    c6z = np.zeros((zmax + 1, zmax + 1, MAXC, MAXC))
    cn1z = np.zeros_like(c6z)
    cn2z = np.zeros_like(c6z)
    valz = np.zeros((zmax + 1, zmax + 1, MAXC, MAXC), dtype=bool)
    for (za, zb), entries in C6AB.items():
        if za > zmax or zb > zmax:
            continue
        for i, j, c6, cni, cnj in entries:
            c6z[za, zb, i - 1, j - 1] = c6
            cn1z[za, zb, i - 1, j - 1] = cni
            cn2z[za, zb, i - 1, j - 1] = cnj
            valz[za, zb, i - 1, j - 1] = True
    za = np.array(z)[:, None]
    zb = np.array(z)[None, :]
    C6R = torch.as_tensor(c6z[za, zb], dtype=dtype, device=device)
    CN1 = torch.as_tensor(cn1z[za, zb], dtype=dtype, device=device)
    CN2 = torch.as_tensor(cn2z[za, zb], dtype=dtype, device=device)
    VALID = torch.as_tensor(valz[za, zb], dtype=torch.bool, device=device)
    rcov = torch.as_tensor([RCOV[v - 1] for v in z], dtype=dtype, device=device)
    r2r4 = torch.as_tensor([R2R4[v - 1] for v in z], dtype=dtype, device=device)
    return C6R, CN1, CN2, VALID, rcov, r2r4


# ---------------------------------------------------------------------------
# real-space image enumeration (integer labels off a detached cell)
# ---------------------------------------------------------------------------

def _image_labels(cell_ang: np.ndarray | None, rcut_bohr: float) -> np.ndarray:
    """Integer lattice labels n with |n·cell| ≤ rcut (incl. n=0). (nR, 3).

    Mirrors ``ewald._image_vectors`` but returns integer labels so the Cartesian
    images can be rebuilt on the autograd cell (the stress path); a molecule
    (cell is None) has only the n=0 image.
    """
    if cell_ang is None:
        return np.zeros((1, 3), dtype=np.int64)
    cell = np.asarray(cell_ang, dtype=np.float64) / BOHR_ANG  # Bohr
    binv = reciprocal_cell(cell) / (2.0 * math.pi)  # rows b_i/2π
    bounds = [int(np.ceil(rcut_bohr * np.linalg.norm(binv[i]))) + 1 for i in range(3)]
    axes = [np.arange(-n, n + 1) for n in bounds]
    n = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    r = n @ cell
    return n[np.linalg.norm(r, axis=1) <= rcut_bohr + 1e-9].astype(np.int64)


def _pair_distances(pos_bohr: torch.Tensor, images_bohr: torch.Tensor):
    """(na,na,nR) distances |τ_a − τ_b + L| and the self-pair mask.

    The masked A=B, L=0 separation is shifted off zero *before* the norm so the
    dead branch keeps finite second derivatives (the Ewald double-backward
    guard); callers must still exclude self pairs via the returned mask.
    """
    na = pos_bohr.shape[0]
    d = (
        pos_bohr[:, None, None, :]
        - pos_bohr[None, :, None, :]
        + images_bohr[None, None, :, :]
    )
    dev, dt = pos_bohr.device, pos_bohr.dtype
    img0 = torch.linalg.norm(images_bohr, dim=-1) < 1e-12
    self_mask = torch.eye(na, dtype=torch.bool, device=dev)[:, :, None] & img0[None, None, :]
    offset = torch.zeros(3, dtype=dt, device=dev)
    offset[0] = 1.0
    d = d + self_mask[..., None].to(dt) * offset
    r = torch.linalg.norm(d, dim=-1)
    return r, self_mask


def _coordination_numbers(pos_bohr, rcov, images_bohr, cn_cutoff):
    """CN_A = Σ'_{B,L} 1/(1+exp(−k1(rco_AB/r − 1))), differentiable in positions."""
    r, self_mask = _pair_distances(pos_bohr, images_bohr)
    rco = (rcov[:, None] + rcov[None, :])[:, :, None]  # (na,na,1)
    damp = 1.0 / (1.0 + torch.exp(-K1 * (rco / r - 1.0)))
    drop = self_mask | (r > cn_cutoff)
    damp = torch.where(drop, torch.zeros_like(damp), damp)
    return damp.sum(dim=(1, 2))


def _c6_from_cn(cn, C6R, CN1, CN2, VALID):
    """CN-interpolated C6_AB (na,na) = Σ_ij w_ij C6ref_ij / Σ_ij w_ij,
    w_ij = exp(k3[(CN1−CN_A)² + (CN2−CN_B)²]); log-shifted for stability."""
    dcn = (CN1 - cn[:, None, None, None]) ** 2 + (CN2 - cn[None, :, None, None]) ** 2
    expo = K3 * dcn
    neg_inf = torch.full_like(expo, float("-inf"))
    expo = torch.where(VALID, expo, neg_inf)
    m = expo.amax(dim=(2, 3), keepdim=True)  # finite: ≥1 valid ref per covered pair
    w = torch.where(VALID, torch.exp(expo - m), torch.zeros_like(expo))
    return (w * C6R).sum(dim=(2, 3)) / w.sum(dim=(2, 3))


# ---------------------------------------------------------------------------
# energy (the differentiable core)
# ---------------------------------------------------------------------------

def dispersion_energy(
    positions: torch.Tensor,
    cell,
    atomic_numbers,
    cfg: D3Config,
    *,
    ref_cell: np.ndarray | None = None,
) -> torch.Tensor:
    """D3(BJ) dispersion energy [eV]. Differentiable in ``positions`` and ``cell``.

    positions (na,3) Å. cell (3,3) Å tensor or None (molecule). ``ref_cell``
    (Å, detached numpy) sets the integer image labels; defaults to ``cell`` and
    is passed explicitly by the stress path so the cell can ride the ε-graph.
    """
    dev = positions.device
    dt = positions.dtype
    pos_b = positions / BOHR_ANG

    C6R, CN1, CN2, VALID, rcov, r2r4 = _reference_tensors(atomic_numbers, dt, dev)

    if cell is None:
        cn_lab = e_lab = np.zeros((1, 3), dtype=np.int64)
        cell_b = None
    else:
        rc = np.asarray(ref_cell if ref_cell is not None else cell.detach().cpu().numpy())
        cn_lab = _image_labels(rc, cfg.cn_cutoff)
        e_lab = _image_labels(rc, cfg.cutoff)
        cell_b = cell / BOHR_ANG  # (3,3) Bohr, rows a_i, on the autograd graph

    def cart(labels):
        if cell_b is None:
            return torch.zeros((1, 3), dtype=dt, device=dev)
        lab = torch.as_tensor(labels, dtype=dt, device=dev)
        return lab @ cell_b

    # coordination numbers (short cutoff) → CN-interpolated C6/C8/BJ radius
    cn = _coordination_numbers(pos_b, rcov, cart(cn_lab), cfg.cn_cutoff)
    c6 = _c6_from_cn(cn, C6R, CN1, CN2, VALID)  # (na,na)
    c8 = 3.0 * c6 * r2r4[:, None] * r2r4[None, :]
    f = cfg.a1 * torch.sqrt(c8 / c6) + cfg.a2  # BJ radius (na,na), Bohr

    # dispersion image sum (long cutoff)
    r, self_mask = _pair_distances(pos_b, cart(e_lab))
    r2 = r * r
    r6 = r2 ** 3
    r8 = r6 * r2
    f6 = (f ** 6)[:, :, None]
    f8 = (f ** 8)[:, :, None]
    e6 = c6[:, :, None] / (r6 + f6)
    e8 = c8[:, :, None] / (r8 + f8)
    term = cfg.s6 * e6 + cfg.s8 * e8
    drop = self_mask | (r > cfg.cutoff)
    term = torch.where(drop, torch.zeros_like(term), term)
    e_hartree = -0.5 * term.sum()
    return e_hartree * HARTREE_EV


# ---------------------------------------------------------------------------
# forces & stress (autograd wrappers — mirror postscf/forces.py, stress.py)
# ---------------------------------------------------------------------------

def _cell_tensor(cell, dtype, device):
    if cell is None:
        return None
    return torch.as_tensor(np.asarray(cell, dtype=np.float64), dtype=dtype, device=device)


def dispersion_forces(positions, cell, atomic_numbers, cfg: D3Config) -> torch.Tensor:
    """F_A = −∂E_disp/∂τ_A, (na,3) [eV/Å], via autograd."""
    dev = positions.device
    pos = positions.detach().clone().to(RDTYPE).requires_grad_(True)
    cell_t = _cell_tensor(cell, RDTYPE, dev)
    e = dispersion_energy(pos, cell_t, atomic_numbers, cfg)
    (grad,) = torch.autograd.grad(e, pos)
    return -grad


def dispersion_stress(positions, cell, atomic_numbers, cfg: D3Config) -> torch.Tensor:
    """σ_αβ = (1/Ω) ∂E_disp/∂ε_αβ, (3,3) [eV/Å³] (tension-positive), via autograd.

    Symmetric strain ε with r→(1+ε)r, a_i→(1+ε)a_i, τ→(1+ε)τ, exactly the
    ``postscf/stress.py`` convention; integer image labels stay fixed at ε=0.
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
