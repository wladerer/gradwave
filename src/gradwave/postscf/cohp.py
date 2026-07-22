"""Crystal Orbital Hamilton Population (COHP), Layer-C post-processing.

COHP resolves the band-structure energy into atom-pair bonding contributions.
Following the projected-COHP idea (Deringer/Dronskowski, the method LOBSTER
implements for plane-wave codes) we express the Kohn-Sham states in the
Loewdin-orthonormalized pseudo-atomic-orbital basis that core/pdos.py already
builds, then weight the energy-resolved density matrix by the Hamiltonian:

    COHP_{IJ}(E) = 2 Re sum_{p in I, q in J} H~_{pq}
                   sum_n <phi~_p|psi_n> <psi_n|phi~_q> delta(E - eps_n),

summed over the atomic orbitals p on atom I and q on atom J. Bonding states give
COHP < 0 (energy-lowering), antibonding COHP > 0 (the Hamilton sign convention).
The integral to the Fermi level, ICOHP, is the standard scalar bond descriptor.

The AO-basis Hamiltonian is reconstructed band-limited from the converged
spectrum,

    H~(k) = P(k)^dagger diag(eps(k)) P(k),    P_{np} = <phi~_p(k)|psi_n(k)>,

i.e. the Kohn-Sham Hamiltonian projected onto the span of the computed bands and
rewritten in the Loewdin AO basis. This needs only the projections and the
eigenvalues, so the SAME routine serves the collinear (nspin 1, 2), the
noncollinear, and the fully-relativistic (spin-orbit) spectra: the spinor
structure and SOC enter entirely through psi_n and the spinor AO projectors,
which core/pdos.py already assembles for the j-resolved PDOS. The band-limited
form is exact per k in the limit of a complete AO span and all bands; the finite
plane-wave truncation leaves the same spilling core/pdos.py reports, and the
finite band window is why an unoccupied-state tail is missing above the highest
computed eigenvalue. Both are reported so a caller can judge convergence.

Sum rule (used as the internal validation, since QE carries no COHP reference):
summing COHP over EVERY atom pair including the on-site blocks and integrating to
E_F reproduces the band-structure energy sum_n f_n eps_n, up to the spilling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gradwave.postscf.pdos import (
    _ao_projectors_k,
    _ao_spinor_projectors_k,
    _atomic_columns,
    _atomic_columns_so,
    _broaden,
    _lowdin_project,
    _unpack_result,
)


@dataclass
class COHP:
    """Atom-pair-resolved COHP. `pair_cohp[label](E)` is bonding-negative; the
    integral to E_F is `pair_icohp[label]`. `total`/`total_icohp` sum the selected
    off-site pairs. `spilling` is the plane-wave-truncation weight lost by the AO
    projection; `band_window_eV` is the highest computed eigenvalue (states above
    it are absent from the band-limited Hamiltonian)."""

    energy_eV: np.ndarray               # (npoints,)
    pair_cohp: dict                     # "i-j" -> (npoints,)
    pair_icohp: dict                    # "i-j" -> float [eV]
    total: np.ndarray                   # (npoints,)
    total_icohp: float
    spilling: float
    fermi_eV: float | None
    kind: str                           # "collinear" | "noncollinear" | "soc"
    band_window_eV: float
    pairs: list                         # [(i, j, distance_A)]

    def to_dict(self) -> dict:
        def _col(a):
            return np.asarray(a).tolist()
        return {
            "energy_eV": _col(self.energy_eV),
            "pair_cohp": {k: _col(v) for k, v in self.pair_cohp.items()},
            "pair_icohp": dict(self.pair_icohp),
            "total": _col(self.total),
            "total_icohp": self.total_icohp,
            "spilling": self.spilling,
            "fermi_eV": self.fermi_eV,
            "kind": self.kind,
            "band_window_eV": self.band_window_eV,
            "pairs": [[int(i), int(j), float(d)] for i, j, d in self.pairs],
        }


def _min_image_dist(system, i: int, j: int) -> float:
    """Nearest-image |tau_i - tau_j| [A] under the periodic cell."""
    cell = np.asarray(system.grid.cell, dtype=float)
    pos = system.positions.detach().cpu().numpy()
    d = pos[i] - pos[j]
    frac = d @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return float(np.linalg.norm(frac @ cell))


def _select_pairs(system, pairs, rcut: float):
    """Resolve the requested atom pairs to [(i, j, dist)] with i < j. An explicit
    `pairs` list wins; otherwise every distinct atom pair within `rcut`."""
    na = len(system.species_of_atom)
    if pairs is not None:
        chosen = {(min(i, j), max(i, j)) for i, j in pairs if i != j}
    else:
        chosen = {(i, j) for i in range(na) for j in range(i + 1, na)
                  if _min_image_dist(system, i, j) <= rcut}
    out = [(i, j, _min_image_dist(system, i, j)) for i, j in sorted(chosen)]
    return out


def _pair_block_weights(proj, eig, atom_of, i, j, factor):
    """Per-band COHP weight w_b for the (i, j) atom block, given one k-point's
    Loewdin amplitudes proj (nb, nbasis) and eigenvalues eig (nb,).

    H~ = proj^dagger diag(eig) proj is the band-limited AO Hamiltonian; the block
    contribution to <psi_b|H~|psi_b> is
        w_b = factor * Re sum_{p in i, q in j} conj(proj_bp) H~_pq proj_bq,
    with factor 2 for an off-site pair (i != j, the block plus its Hermitian
    conjugate) and 1 for an on-site block."""
    mi = torch.as_tensor(atom_of == i, device=proj.device)
    mj = torch.as_tensor(atom_of == j, device=proj.device)
    ai = proj[:, mi]                                  # (nb, ni)
    aj = proj[:, mj]                                  # (nb, nj)
    # H~ restricted to the (i, j) block: (proj_i)^dagger diag(eig) proj_j
    hij = (ai.conj() * eig[:, None]).transpose(0, 1) @ aj   # (ni, nj)
    tmp = ai.conj() @ hij                              # (nb, nj)
    w = factor * (tmp * aj).real.sum(dim=1)            # (nb,)
    return w.cpu().numpy()


def _accumulate(proj_per_k, eig_per_k, kw, atom_of, pair_list, g_spin, fermi):
    """Core spectral accumulation shared by every formalism. Returns per-pair
    (energies, weights) samples, the direct (step-occupation) ICOHP, and the
    all-pairs total ICOHP used for the sum-rule check."""
    energies = {p[:2]: [] for p in pair_list}
    weights = {p[:2]: [] for p in pair_list}
    icohp = {p[:2]: 0.0 for p in pair_list}
    na = int(atom_of.max()) + 1 if len(atom_of) else 0
    total_all_icohp = 0.0
    fermi = -np.inf if fermi is None else float(fermi)
    for ik, (proj, eig) in enumerate(zip(proj_per_k, eig_per_k, strict=True)):
        w_k = float(kw[ik])
        e_np = eig.cpu().numpy()
        occ_mask = e_np < fermi                        # step occupation to E_F
        for (i, j) in ((p[0], p[1]) for p in pair_list):
            w = _pair_block_weights(proj, eig, atom_of, i, j, 2.0) * g_spin
            energies[(i, j)].append(e_np)
            weights[(i, j)].append(w * w_k)
            icohp[(i, j)] += float((w[occ_mask] * w_k).sum())
        # sum rule: every atom pair including the on-site diagonal
        for a in range(na):
            wa = _pair_block_weights(proj, eig, atom_of, a, a, 1.0) * g_spin
            total_all_icohp += float((wa[occ_mask] * w_k).sum())
        for a in range(na):
            for b in range(a + 1, na):
                wab = _pair_block_weights(proj, eig, atom_of, a, b, 2.0) * g_spin
                total_all_icohp += float((wab[occ_mask] * w_k).sum())
    return energies, weights, icohp, total_all_icohp


def _finalize(energies, weights, icohp, pair_list, width, npoints, window,
              spilling, fermi, kind, band_window, total_all_icohp):
    all_e = np.concatenate([np.concatenate(v) for v in energies.values()]) \
        if energies else np.array([0.0])
    if window is None:
        window = (all_e.min() - 10 * width, all_e.max() + 10 * width)
    grid = np.linspace(window[0], window[1], npoints)

    pair_cohp, pair_icohp = {}, {}
    total = np.zeros(npoints)
    for (i, j, _dist) in pair_list:
        lab = f"{i + 1}-{j + 1}"
        e = np.concatenate(energies[(i, j)])
        w = np.concatenate(weights[(i, j)])
        curve = _broaden(grid, e, w, width)
        pair_cohp[lab] = curve
        pair_icohp[lab] = icohp[(i, j)]
        total += curve
    return COHP(
        energy_eV=grid, pair_cohp=pair_cohp, pair_icohp=pair_icohp,
        total=total, total_icohp=float(sum(pair_icohp.values())),
        spilling=spilling, fermi_eV=None if fermi is None else float(fermi),
        kind=kind, band_window_eV=float(band_window), pairs=pair_list,
    ), total_all_icohp


def _spilling_from_weights(captured_kw, total_kw):
    return float(1.0 - captured_kw / total_kw) if total_kw else 0.0


@torch.no_grad()
def cohp(res, *, pairs=None, rcut: float = 3.0, width: float = 0.1,
         npoints: int = 800, window=None):
    """Atom-pair COHP of a converged collinear SCF (norm-conserving, nspin 1/2).

    `pairs` selects atom index tuples (0-based); the default is every atom pair
    within `rcut` angstrom. Spin channels are summed. Returns a `COHP`.
    """
    system, nspin, eig, coeffs, fermi, device, _ = _unpack_result(res)
    cols = _atomic_columns(system)
    atom_of = np.array([c.atom for c in cols])
    pair_list = _select_pairs(system, pairs, rcut)
    kw = system.kweights.to(device)
    g_spin = 2.0 if nspin == 1 else 1.0

    # Loewdin amplitudes P_{np} = <phi~_p|psi_n> per (spin, k), flattened over spin
    proj_per_k, eig_per_k, kw_flat = [], [], []
    captured_kw = total_kw = 0.0
    band_window = -np.inf
    for isp in range(nspin):
        for ik, sph in enumerate(system.spheres):
            c = coeffs[isp][ik].to(device)                       # (nb, npw)
            q = _ao_projectors_k(system, sph, cols, device)      # (nproj, npw)
            becp = torch.einsum("bg,pg->bp", c, q.conj())
            overlap = torch.einsum("ig,jg->ij", q.conj(), q)
            proj = _lowdin_project(becp, overlap)                # (nb, nproj)
            e = eig[isp, ik].to(device)
            proj_per_k.append(proj)
            eig_per_k.append(e)
            kw_flat.append(float(kw[ik]))
            w = float(kw[ik])
            captured_kw += (proj.real ** 2 + proj.imag ** 2).sum(1).cpu().numpy().sum() * w
            total_kw += proj.shape[0] * w
            band_window = max(band_window, float(e.max()))
    spilling = _spilling_from_weights(captured_kw, total_kw)

    energies, weights, icohp, tot = _accumulate(
        proj_per_k, eig_per_k, kw_flat, atom_of, pair_list, g_spin, fermi)
    out, _tot = _finalize(energies, weights, icohp, pair_list, width, npoints,
                          window, spilling, fermi, "collinear", band_window, tot)
    out._sumrule_icohp = _tot  # all-pairs incl on-site, for validation
    return out


def _spinor_proj_per_k(res, cols, spinor, device):
    """Per-k Loewdin amplitudes for a spinor SCF. `spinor=True` uses the (j, mj)
    spin-angular AO projectors (SOC); `spinor=False` uses scalar AOs replicated on
    the two spin components (scalar-relativistic noncollinear). Returns
    (proj_per_k, eig_per_k, spilling, band_window)."""
    system = res.system
    m_pw = system.batch.npw_max
    kw = system.kweights.to(device)
    eig = res.eigenvalues
    proj_per_k, eig_per_k = [], []
    captured_kw = total_kw = 0.0
    band_window = -np.inf
    for ik, sph in enumerate(system.spheres):
        npw = sph.npw
        c = res.coeffs[ik].to(device)                            # (nb, 2*m_pw)
        cu, cd = c[:, :npw], c[:, m_pw:m_pw + npw]
        if spinor:
            qu, qd = _ao_spinor_projectors_k(system, sph, cols, device)
            becp = (torch.einsum("bg,pg->bp", cu, qu.conj())
                    + torch.einsum("bg,pg->bp", cd, qd.conj()))
            overlap = (torch.einsum("pg,qg->pq", qu.conj(), qu)
                       + torch.einsum("pg,qg->pq", qd.conj(), qd))
            proj = _lowdin_project(becp, overlap)                # (nb, nproj)
        else:
            q = _ao_projectors_k(system, sph, cols, device)
            overlap = torch.einsum("ig,jg->ij", q.conj(), q)
            pu = _lowdin_project(torch.einsum("bg,pg->bp", cu, q.conj()), overlap)
            pd = _lowdin_project(torch.einsum("bg,pg->bp", cd, q.conj()), overlap)
            # spin-summed charge COHP: stack the two components side by side so the
            # band-limited H~ carries the full spinor character of psi_n
            proj = torch.cat([pu, pd], dim=1)                    # (nb, 2*nproj)
        e = eig[ik].to(device)
        proj_per_k.append(proj)
        eig_per_k.append(e)
        w = float(kw[ik])
        cap = (proj.real ** 2 + proj.imag ** 2).sum(1).cpu().numpy().sum()
        captured_kw += cap * w
        total_kw += proj.shape[0] * w
        band_window = max(band_window, float(e.max()))
    spilling = _spilling_from_weights(captured_kw, total_kw)
    return proj_per_k, eig_per_k, spilling, band_window


@torch.no_grad()
def cohp_noncollinear(res, *, pairs=None, rcut: float = 3.0, width: float = 0.1,
                      npoints: int = 800, window=None):
    """Charge (spin-summed) atom-pair COHP of a noncollinear spinor SCF.

    Scalar pseudo-atomic orbitals are projected per spin component and the two
    are stacked, so the band-limited Hamiltonian carries the full spinor
    character of each state. Works with or without spin-orbit coupling; for a
    fully-relativistic pseudo `cohp_soc` gives the j-resolved projectors instead.
    """
    from gradwave.scf.noncollinear import NCResult
    if not isinstance(res, NCResult):
        raise NotImplementedError("cohp_noncollinear expects a noncollinear NCResult")
    system = res.system
    device = res.coeffs.device
    cols = _atomic_columns(system)
    atom_of = np.tile([c.atom for c in cols], 2)      # up-block then down-block
    pair_list = _select_pairs(system, pairs, rcut)

    proj_per_k, eig_per_k, spilling, band_window = _spinor_proj_per_k(
        res, cols, spinor=False, device=device)
    kw = [float(w) for w in system.kweights]
    energies, weights, icohp, tot = _accumulate(
        proj_per_k, eig_per_k, kw, atom_of, pair_list, 1.0, res.fermi)
    out, _tot = _finalize(energies, weights, icohp, pair_list, width, npoints,
                          window, spilling, res.fermi, "noncollinear",
                          band_window, tot)
    out._sumrule_icohp = _tot
    return out


@torch.no_grad()
def cohp_soc(res, *, pairs=None, rcut: float = 3.0, width: float = 0.1,
             npoints: int = 800, window=None):
    """Atom-pair COHP of a fully-relativistic (spin-orbit) spinor SCF.

    Projects the spinor states onto spin-angular |n l j mj> atomic orbitals built
    from the FR pseudo's PP_PSWFC radials, so spin-orbit coupling enters the
    band-limited Hamiltonian through both the states and the projector basis.
    """
    from gradwave.scf.noncollinear import NCResult
    if not isinstance(res, NCResult):
        raise NotImplementedError("cohp_soc expects a fully-relativistic NCResult")
    system = res.system
    if not getattr(system, "is_fr", False):
        raise NotImplementedError(
            "cohp_soc needs a fully-relativistic (SOC) pseudo; use "
            "cohp_noncollinear for scalar-relativistic noncollinear SCF")
    device = res.coeffs.device
    cols = _atomic_columns_so(system)
    atom_of = np.array([c.atom for c in cols])
    pair_list = _select_pairs(system, pairs, rcut)

    proj_per_k, eig_per_k, spilling, band_window = _spinor_proj_per_k(
        res, cols, spinor=True, device=device)
    kw = [float(w) for w in system.kweights]
    energies, weights, icohp, tot = _accumulate(
        proj_per_k, eig_per_k, kw, atom_of, pair_list, 1.0, res.fermi)
    out, _tot = _finalize(energies, weights, icohp, pair_list, width, npoints,
                          window, spilling, res.fermi, "soc", band_window, tot)
    out._sumrule_icohp = _tot
    return out
