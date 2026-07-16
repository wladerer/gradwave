"""Projected density of states (Layer C post-processing).

Projects the converged Kohn-Sham states onto the pseudo-atomic orbitals
(PP_PSWFC) to resolve the DOS by atom, angular momentum, and magnetic quantum
number. The atomic-orbital projectors reuse the Kleinman-Bylander structure of
core/hubbard.py,

    phi_{a,nlm}(k+G) = (4 pi / sqrt(Omega)) (-i)^l Y_lm(k+G^) F_nl(|k+G|)
                       e^{-i (k+G).tau_a},   F_nl(q) = sbt(l, r^2 R_nl, r, rab, q)

and the projection is Loewdin-orthonormalized so the per-state weights obey the
sum rule up to the plane-wave truncation, which the spilling parameter reports.

Coverage here: norm-conserving, nspin=1 and 2. USPP/PAW (the S-metric) and the
noncollinear/SOC projections extend this in the same module.

Reference: D. Sanchez-Portal et al., the Loewdin population analysis behind QE's
projwfc.x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.radial import sbt
from gradwave.scf.loop import SCFResult

_MINUS_I_POW = [1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j]  # (-i)^l
_M_LABELS = {0: [""], 1: ["z", "x", "y"],  # real-harmonic order of ylm_all
             2: ["z2", "xz", "yz", "x2-y2", "xy"],
             3: ["z3", "xz2", "yz2", "zx2-zy2", "xyz", "xx2-3yy2", "3xx2-yy2"]}


@dataclass
class AOColumn:
    """One atomic-orbital projector column."""

    atom: int
    species: int
    label: str        # e.g. "3D"
    l: int
    m: int            # 0..2l, real-harmonic index


@dataclass
class ProjectedDOS:
    energy_eV: np.ndarray            # (npoints,)
    total: np.ndarray                # (npoints,) or (2, npoints) for nspin=2
    groups: dict                     # group label -> same shape as total
    spilling: float                  # fraction of KS weight outside the AO span
    fermi_eV: float | None
    nspin: int
    group_by: str

    def to_dict(self) -> dict:
        """JSON-ready block (lists, not arrays); the parsing target for the
        analysis layer."""
        def _col(a):
            return np.asarray(a).tolist()
        return {
            "energy_eV": _col(self.energy_eV),
            "total": _col(self.total),
            "groups": {k: _col(v) for k, v in self.groups.items()},
            "spilling": self.spilling,
            "fermi_eV": self.fermi_eV,
            "nspin": self.nspin,
            "group_by": self.group_by,
        }


def _atomic_columns(system) -> list[AOColumn]:
    """Every PP_PSWFC orbital of every atom, expanded over m."""
    cols = []
    for a, sp in enumerate(system.species_of_atom):
        pswfc = getattr(system.upfs[sp], "pswfc", ())
        if not pswfc:
            raise ValueError(
                f"{system.upfs[sp].element}: the pseudopotential carries no "
                "PP_PSWFC atomic orbitals, so a projected DOS is not available "
                "(SG15 ONCV omits them; use a PseudoDojo or psl pseudo)")
        for o in pswfc:
            for m in range(2 * o.l + 1):
                cols.append(AOColumn(a, sp, o.label, o.l, m))
    return cols


def _ao_projectors_k(system, sph, cols, device):
    """AO projectors q (nproj, npw) on one G-sphere, phased at the positions."""
    vol = system.grid.volume
    kpg = sph.kpg.to(device)
    npw = sph.npw
    qmag = np.sqrt(sph.kpg2.cpu().numpy())
    l_max = max(c.l for c in cols)
    y = ylm_all(l_max, kpg)  # (npw, (l_max+1)^2)
    # radial form factors F_nl(q), cached per (species, orbital label)
    fcache: dict[tuple, torch.Tensor] = {}
    for sp in set(system.species_of_atom):
        u = system.upfs[sp]
        for o in u.pswfc:
            key = (sp, o.label)
            if key not in fcache:
                fcache[key] = torch.as_tensor(
                    sbt(o.l, o.rchi * u.r, u.r, u.rab, qmag),
                    dtype=RDTYPE, device=device)
    # phases e^{-i(k+G).tau_a}
    phase_arg = kpg @ system.positions.to(device).T  # (npw, na)
    phase = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))

    q = torch.zeros(len(cols), npw, dtype=CDTYPE, device=device)
    orb_of = {(c.species, c.label) for c in cols}
    fkey = {(sp, lab): fcache[(sp, lab)] for (sp, lab) in orb_of}
    for p, c in enumerate(cols):
        pref = (4.0 * math.pi / math.sqrt(vol)) * _MINUS_I_POW[c.l]
        yl = y[:, c.l * c.l + c.m]
        q[p] = pref * (fkey[(c.species, c.label)] * yl).to(CDTYPE) * phase[:, c.atom]
    return q


def _lowdin_weights(becp, overlap, floor=1e-8):
    """Loewdin-orthonormalized |<phi~_p|psi_b>|^2 from raw <phi_p|psi_b> and the
    AO overlap <phi_i|phi_j>. becp (nb, nproj), overlap (nproj, nproj)."""
    w, v = torch.linalg.eigh(overlap)
    w = w.clamp_min(floor)
    o_inv_sqrt = (v * w.rsqrt()) @ v.conj().T          # O^{-1/2}, Hermitian
    proj = becp @ o_inv_sqrt                            # (nb, nproj)
    return (proj.real ** 2 + proj.imag ** 2)            # (nb, nproj)


def _group_key(col: AOColumn, group_by: str):
    if group_by == "total":
        return "total"
    if group_by == "atom":
        return f"atom{col.atom + 1}"
    if group_by == "l":
        return f"atom{col.atom + 1}:{col.label}"
    ml = _M_LABELS.get(col.l, [str(m) for m in range(2 * col.l + 1)])
    suffix = ml[col.m] if col.m < len(ml) else str(col.m)
    return f"atom{col.atom + 1}:{col.label}{('_' + suffix) if suffix else ''}"


@torch.no_grad()
def projected_dos(res, *, width: float = 0.1, npoints: int = 800, window=None,
                  group_by: str = "l") -> ProjectedDOS:
    """Löwdin-projected DOS of a converged norm-conserving SCF.

    group_by is one of 'atom', 'l' (atom + orbital), 'lm' (adds m), or 'total'.
    Spin channels come back stacked on axis 0 for nspin=2.
    """
    if not isinstance(res, SCFResult):
        raise NotImplementedError(
            "projected DOS currently supports the norm-conserving SCFResult; "
            "USPP/PAW and noncollinear are separate paths")
    system = res.system
    device = res.rho.device
    nspin = int(getattr(res, "nspin", 1))
    cols = _atomic_columns(system)

    eig = res.eigenvalues if nspin == 2 else res.eigenvalues[None]
    coeffs = res.coeffs if nspin == 2 else [res.coeffs]
    kw = system.kweights.to(device)
    g_spin = 2.0 if nspin == 1 else 1.0

    # per (spin, k) Löwdin weights and eigenvalues
    all_e = eig.reshape(nspin, -1).cpu().numpy()           # (nspin, nk*nb)
    weights = np.zeros((nspin, all_e.shape[1], len(cols)))  # (nspin, states, nproj)
    kweight_state = np.zeros((nspin, all_e.shape[1]))
    nb = eig.shape[-1]
    for isp in range(nspin):
        for ik, sph in enumerate(system.spheres):
            c = coeffs[isp][ik].to(device)                 # (nb, npw)
            q = _ao_projectors_k(system, sph, cols, device)  # (nproj, npw)
            becp = torch.einsum("bg,pg->bp", c, q.conj())    # <phi_p|psi_b>
            overlap = torch.einsum("ig,jg->ij", q.conj(), q)  # <phi_i|phi_j>
            wgt = _lowdin_weights(becp, overlap).cpu().numpy()  # (nb, nproj)
            sl = slice(ik * nb, (ik + 1) * nb)
            weights[isp, sl] = wgt
            kweight_state[isp, sl] = float(kw[ik])

    # spilling: 1 - <sum_p weight>_states, kweighted over every (spin, k, band).
    # A complete AO basis captures every state, so this is 0; the plane-wave
    # truncation and the finite orbital set leave a positive remainder.
    captured = (weights.sum(axis=2) * kweight_state).sum()
    spilling = float(1.0 - captured / kweight_state.sum())

    # energy grid + gaussian broadening
    if window is None:
        window = (all_e.min() - 10 * width, all_e.max() + 10 * width)
    grid = np.linspace(window[0], window[1], npoints)
    inv = 1.0 / (width * math.sqrt(2 * math.pi))

    def broaden(state_weight, isp):
        e = all_e[isp]
        g = (np.exp(-0.5 * ((grid[:, None] - e[None, :]) / width) ** 2) * inv
             * (kweight_state[isp] * g_spin * state_weight)[None, :]).sum(axis=1)
        return g

    labels = sorted({_group_key(c, group_by) for c in cols})
    groups = {}
    for lab in labels:
        mask = np.array([_group_key(c, group_by) == lab for c in cols])
        per_spin = [broaden(weights[isp][:, mask].sum(axis=1), isp)
                    for isp in range(nspin)]
        groups[lab] = per_spin[0] if nspin == 1 else np.stack(per_spin)
    total = [broaden(weights[isp].sum(axis=1), isp) for isp in range(nspin)]
    total = total[0] if nspin == 1 else np.stack(total)

    return ProjectedDOS(
        energy_eV=grid, total=total, groups=groups, spilling=spilling,
        fermi_eV=None if res.fermi is None else float(res.fermi),
        nspin=nspin, group_by=group_by,
    )
