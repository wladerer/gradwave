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

Two routes build the AO Hamiltonian H~ (the `method` argument):
  operator   (default, collinear)  H~ = <phi~|H^|phi~>, the converged Kohn-Sham
             operator applied to the AO projectors. A constant potential shift
             adds C·1 in the orthonormal basis, so off-site COHP is energy-zero
             invariant. This is the correct route for solids; on diamond it gives
             the right bonding/antibonding shape and a bonding (negative) ICOHP.
  eigenvalue (band-limited)  H~ = P^dagger diag(eps) P. Cheap, needs only the
             projections + eigenvalues, and is what the spinor (noncollinear/SOC)
             paths use for now (the spinor operator route is a follow-up). But it
             carries the plane-wave energy zero, which leaks into off-site COHP
             through the incomplete band set (diamond: ~65 eV of ICOHP per eV of
             shift), so it is only reliable for well-separated atoms (O2, Bi2).

QUANTITATIVE STATUS (NOT yet calibrated to LOBSTER — do not ship as such). On
diamond (PBE) LOBSTER reports IpCOHP ~= -9.64 eV per C-C bond. Two gaps remain:
  1. Bond resolution. This atom-pair COHP uses Bloch AO projectors, so pair
     (i, j) is the interaction of atom i with the WHOLE atom-j sublattice (all
     periodic images), not a single bond — for diamond ~4 nearest bonds. A
     per-bond (per-image R) resolution is needed to compare to LOBSTER directly.
  2. Basis. A Loewdin-orthonormalized *pseudo-atomic* basis is more diffuse than
     LOBSTER's contracted local orbitals and develops larger off-site elements,
     so the operator-route per-bond magnitude overshoots (~2x) and the band-
     limited eigenvalue route undershoots (~2x); the true value is bracketed
     between them. A contracted / projected local basis is needed for a
     quantitative match.
Sign and bonding/antibonding shape are correct (diamond: COHP<0 valence, >0
conduction; spilling physical after the projection-conjugation fix in
core/pdos.py). Treat absolute solid-state ICOHP as not-yet-validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gradwave.dtypes import CDTYPE
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
    it are absent from the band-limited Hamiltonian). Two spilling metrics are
    reported: `spilling` over every band (total spilling) and `charge_spilling`
    over the occupied manifold — the latter bounds how well the AO span captures
    the real electron density, and hence how much to trust the occupied ICOHP.

    The k-point- and band-resolved COHP is exposed unbroadened: `band_cohp[label]`
    is (nblocks, nb), the Hamilton population of each eigenstate (k, n) on that
    bond (bonding < 0), with `band_energies` (nblocks, nb) the matching eps_n(k)
    and `block_kpts`/`block_kweights` the k-points and weights. A "block" is one
    (spin, k) for a collinear nspin=2 run and one k otherwise; blocks are ordered
    spin-major, so a collinear nspin=2 array reshapes to (nspin, nk, nb). The
    energy curve is exactly sum over blocks of a k-weighted Gaussian sum of these;
    `cohp_at_k` re-broadens a single block."""

    energy_eV: np.ndarray               # (npoints,)
    pair_cohp: dict                     # "i-j" -> (npoints,)
    pair_icohp: dict                    # "i-j" -> float [eV]
    total: np.ndarray                   # (npoints,)
    total_icohp: float
    spilling: float                     # total spilling, over every band
    charge_spilling: float              # over the occupied manifold
    fermi_eV: float | None
    kind: str                           # "collinear" | "noncollinear" | "soc"
    band_window_eV: float
    pairs: list                         # [(i, j, distance_A)]
    method: str = "operator"            # "operator" (<phi~|H^|phi~>) or "eigenvalue"
    nspin: int = 1
    nk: int = 0
    block_kpts: np.ndarray = None       # (nblocks, 3) fractional
    block_kweights: np.ndarray = None   # (nblocks,)
    band_energies: np.ndarray = None    # (nblocks, nb) [eV]
    band_cohp: dict = None              # "i-j" -> (nblocks, nb) [eV], bonding < 0
    bond_images: dict = None            # "i-j" -> (n1,n2,n3), set by resolve_images
    basis: str = "pswfc"                # projector basis: "pswfc" or "iao"
    rmsp: float = None                  # LOBSTER G-space projection residual [0,1]

    def cohp_at_k(self, label: str, block: int, *, width: float = 0.1, grid=None):
        """Re-broaden the COHP of one (spin, k) block for a pair into a curve.
        Returns (energy_grid, cohp). `grid` defaults to `self.energy_eV`."""
        grid = self.energy_eV if grid is None else np.asarray(grid)
        e = self.band_energies[block]
        w = self.band_cohp[label][block] * float(self.block_kweights[block])
        return grid, _broaden(grid, e, w, width)

    def bands_reshaped(self, label: str) -> np.ndarray:
        """`band_cohp[label]` reshaped to (nspin, nk, nb)."""
        return self.band_cohp[label].reshape(self.nspin, self.nk, -1)

    def to_dict(self) -> dict:
        def _col(a):
            return None if a is None else np.asarray(a).tolist()
        return {
            "energy_eV": _col(self.energy_eV),
            "pair_cohp": {k: _col(v) for k, v in self.pair_cohp.items()},
            "pair_icohp": dict(self.pair_icohp),
            "total": _col(self.total),
            "total_icohp": self.total_icohp,
            "spilling": self.spilling,
            "charge_spilling": self.charge_spilling,
            "fermi_eV": self.fermi_eV,
            "kind": self.kind,
            "band_window_eV": self.band_window_eV,
            "pairs": [[int(i), int(j), float(d)] for i, j, d in self.pairs],
            "method": self.method,
            "nspin": self.nspin,
            "nk": self.nk,
            "block_kpts": _col(self.block_kpts),
            "block_kweights": _col(self.block_kweights),
            "band_energies": _col(self.band_energies),
            "band_cohp": {k: _col(v) for k, v in (self.band_cohp or {}).items()},
            "bond_images": (None if self.bond_images is None else
                            {k: list(map(int, v)) for k, v in self.bond_images.items()}),
            "basis": self.basis,
            "rmsp": self.rmsp,
        }


def _min_image_dist(system, i: int, j: int) -> float:
    """Nearest-image |tau_i - tau_j| [A] under the periodic cell."""
    cell = np.asarray(system.grid.cell, dtype=float)
    pos = system.positions.detach().cpu().numpy()
    d = pos[i] - pos[j]
    frac = d @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return float(np.linalg.norm(frac @ cell))


def _nearest_image_R(system, i: int, j: int) -> np.ndarray:
    """Integer lattice vector R of the image of atom j nearest atom i, i.e. the
    single bond the min-image distance picks out. The COHP off-site block sums the
    whole j sublattice; this R selects one image for a per-bond resolution."""
    cell = np.asarray(system.grid.cell, dtype=float)
    pos = system.positions.detach().cpu().numpy()
    frac = (pos[i] - pos[j]) @ np.linalg.inv(cell)
    return np.round(frac).astype(int)


def _accumulate_images(proj_per_k, htilde_per_k, eig_per_k, kw, kpts, atom_of,
                       pair_list, images, nspin, nk, g_spin, fermi):
    """Per-bond COHP: restrict each pair to a single image R of atom j.

    The Bloch AO Hamiltonian H~_pq(k) is the interaction of orbital p (home cell)
    with the whole atom-j sublattice. Its real-space transform along the bond,

        h_pq(R) = sum_k w_k e^{-2 pi i k.R} H~_pq(k),

    is the hopping to the single image R, and the matching density term carries the
    conjugate phase e^{+2 pi i k.R}, so the per-state weight for bond (i, j, R) is

        w_b^R = 2 g_spin Re[ e^{2 pi i k.R} sum_{p in i, q in j}
                             conj(proj_bp) h_pq(R) proj_bq ].

    Summing w_b^R over the Born-von-Karman image shell reconstructs the sublattice
    weight `_accumulate` returns (validated in the tests); at Gamma R=0 and the two
    are identical. h_pq(R) is only the true hopping on a full (unreduced) k-mesh, so
    per-image resolution needs time_reversal=False / use_symmetry=False for R != 0."""
    raw = {p[:2]: [None] * len(proj_per_k) for p in pair_list}
    icohp = {p[:2]: 0.0 for p in pair_list}
    fermi = -np.inf if fermi is None else float(fermi)
    for (i, j) in ((p[0], p[1]) for p in pair_list):
        R = np.asarray(images[(i, j)], dtype=float)
        mi = torch.as_tensor(atom_of == i, device=proj_per_k[0].device)
        mj = torch.as_tensor(atom_of == j, device=proj_per_k[0].device)
        # real-space hopping h_pq(R) per spin channel (blocks are spin-major)
        hR = []
        for s in range(nspin):
            acc = None
            for ik in range(nk):
                b = s * nk + ik
                ph = np.exp(-2j * np.pi * float(np.dot(kpts[b], R)))
                cs = torch.as_tensor(float(kw[b]) * ph, dtype=CDTYPE,
                                     device=htilde_per_k[b].device)
                blk = htilde_per_k[b][mi][:, mj] * cs
                acc = blk if acc is None else acc + blk
            hR.append(acc)
        for s in range(nspin):
            for ik in range(nk):
                b = s * nk + ik
                proj = proj_per_k[b]
                ph = np.exp(2j * np.pi * float(np.dot(kpts[b], R)))
                tmp = proj[:, mi].conj() @ hR[s]          # (nb, nj)
                wc = (tmp * proj[:, mj]).sum(dim=1).cpu().numpy()  # (nb,) complex
                w = 2.0 * g_spin * np.real(wc * ph)
                raw[(i, j)][b] = w
                occ = eig_per_k[b].cpu().numpy() < fermi
                icohp[(i, j)] += float((w[occ] * float(kw[b])).sum())
    return raw, icohp


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


def _iao_projectors_k(phi, psi_occ, floor=1e-8):
    """Intrinsic Atomic Orbitals in the plane-wave basis (Knizia, JCTC 2013),
    norm-conserving metric <a|b> = sum_G conj(a_G) b_G.

    `phi` (nproj, npw) is the free-atom minimal AO basis (the PP_PSWFC projectors)
    and `psi_occ` (nocc, npw) the occupied KS states (orthonormal). Returns the
    IAOs A (nproj, npw):

        A = [ O O~ + (1-O)(1-O~) ] phi = phi - O phi - O~ phi + 2 O O~ phi,

    with O = |psi><psi| the occupied projector and O~ = |psi~><psi~| the projector
    onto the occupied MOs *depolarised* into the minimal basis and reorthonormalised
    (psi~ = orthonormalise(phi S^{-1} <phi|psi>), S = <phi|phi>). By construction the
    IAOs span the occupied manifold exactly, so the occupied-state projection has
    zero spilling — the contracted/localised basis the COHP docstring calls for,
    built with no external tables (Knizia 2013; periodic form arXiv:2407.00852)."""
    Phi, Psi = phi, psi_occ
    eye = torch.eye(Phi.shape[0], dtype=Phi.dtype, device=Phi.device)
    B = torch.einsum("pg,ng->pn", Phi.conj(), Psi)          # <phi_p|psi_n>
    S = torch.einsum("pg,qg->pq", Phi.conj(), Phi)          # <phi_p|phi_q>
    coeff = torch.linalg.solve(S + floor * eye, B)          # S^{-1} <phi|psi>
    psi_t = torch.einsum("pn,pg->ng", coeff, Phi)           # P^{B2} psi (nocc, npw)
    T = torch.einsum("ng,mg->nm", psi_t.conj(), psi_t)      # <psi~|psi~>
    w, v = torch.linalg.eigh(T)
    tis = (v * w.clamp_min(floor).rsqrt()) @ v.conj().T     # T^{-1/2}
    psi_o = torch.einsum("mn,ng->mg", tis, psi_t)           # orthonormal psi~
    o_phi = torch.einsum("pn,ng->pg", B.conj(), Psi)        # O phi
    bt = torch.einsum("pg,ng->pn", Phi.conj(), psi_o)       # <phi_p|psi~_n>
    ot_phi = torch.einsum("pn,ng->pg", bt.conj(), psi_o)    # O~ phi
    u = torch.einsum("ng,pg->np", Psi.conj(), ot_phi)       # <psi_n|O~ phi_p>
    oot_phi = torch.einsum("np,ng->pg", u, Psi)             # O O~ phi
    return Phi - o_phi - ot_phi + 2.0 * oot_phi


def _o_inv_sqrt(overlap, floor=1e-8):
    """O^{-1/2} (Hermitian) of the AO overlap, near-singular modes clamped."""
    w, v = torch.linalg.eigh(overlap)
    w = w.clamp_min(floor)
    return (v * w.rsqrt()) @ v.conj().T


def _htilde_eig(proj, eig):
    """Band-limited AO Hamiltonian H~ = P^dagger diag(eps) P (nbasis, nbasis).
    Cheap (needs only projections + eigenvalues) but carries the plane-wave
    energy zero, which leaks into off-site elements when the band set is
    incomplete — see the module docstring."""
    return (proj.conj() * eig[:, None]).transpose(0, 1) @ proj


def _htilde_operator(q, o_inv_sqrt, h_apply):
    """Operator-route AO Hamiltonian H~ = O^{-1/2} <phi|H^|phi> O^{-1/2}.

    `q` (nbasis, npw) are the RAW (non-orthonormal) AO projectors, `o_inv_sqrt`
    the Loewdin factor of their overlap, and `h_apply(q) -> (nbasis, npw)` applies
    the converged Kohn-Sham Hamiltonian. A constant potential shift adds C·1 in
    the orthonormal basis, so the off-site elements are energy-zero invariant —
    the fix for the eigenvalue route's offset contamination."""
    hraw = torch.einsum("pg,qg->pq", q.conj(), h_apply(q))     # <phi_p|H^|phi_q>
    return o_inv_sqrt @ hraw @ o_inv_sqrt


def _pair_block_weights(proj, htilde, atom_of, i, j, factor):
    """Per-band COHP weight w_b for the (i, j) atom block from one k-point's
    Loewdin amplitudes proj (nb, nbasis) and AO Hamiltonian htilde (nbasis,
    nbasis). The block contribution to <psi_b|H~|psi_b> is
        w_b = factor * Re sum_{p in i, q in j} conj(proj_bp) H~_pq proj_bq,
    with factor 2 for an off-site pair (the block plus its Hermitian conjugate)
    and 1 for an on-site block."""
    mi = torch.as_tensor(atom_of == i, device=proj.device)
    mj = torch.as_tensor(atom_of == j, device=proj.device)
    ai = proj[:, mi]                                  # (nb, ni)
    aj = proj[:, mj]                                  # (nb, nj)
    tmp = ai.conj() @ htilde[mi][:, mj]               # (nb, nj)
    w = factor * (tmp * aj).real.sum(dim=1)           # (nb,)
    return w.cpu().numpy()


def _accumulate(proj_per_k, htilde_per_k, eig_per_k, kw, atom_of, pair_list,
                g_spin, fermi):
    """Core spectral accumulation shared by every formalism. Keeps the per-block
    (i.e. per-(spin,k)) resolution: `block_e[b]` and `raw[pair][b]` are the
    eigenvalues and the per-eigenstate COHP weight w_{kn} (g_spin folded in, no
    k-weight) for block b, the unbroadened k/band-resolved COHP. `htilde_per_k[b]`
    is the AO Hamiltonian for block b (eigenvalue- or operator-route). Also
    returns the direct (step-occupation) ICOHP and the all-pairs sum-rule ICOHP."""
    block_e = []
    raw = {p[:2]: [] for p in pair_list}
    icohp = {p[:2]: 0.0 for p in pair_list}
    na = int(atom_of.max()) + 1 if len(atom_of) else 0
    total_all_icohp = 0.0
    fermi = -np.inf if fermi is None else float(fermi)
    for ik, (proj, htilde, eig) in enumerate(
            zip(proj_per_k, htilde_per_k, eig_per_k, strict=True)):
        w_k = float(kw[ik])
        e_np = eig.cpu().numpy()
        block_e.append(e_np)
        occ_mask = e_np < fermi                        # step occupation to E_F
        for (i, j) in ((p[0], p[1]) for p in pair_list):
            w = _pair_block_weights(proj, htilde, atom_of, i, j, 2.0) * g_spin
            raw[(i, j)].append(w)
            icohp[(i, j)] += float((w[occ_mask] * w_k).sum())
        # sum rule: every atom pair including the on-site diagonal
        for a in range(na):
            wa = _pair_block_weights(proj, htilde, atom_of, a, a, 1.0) * g_spin
            total_all_icohp += float((wa[occ_mask] * w_k).sum())
        for a in range(na):
            for b in range(a + 1, na):
                wab = _pair_block_weights(proj, htilde, atom_of, a, b, 2.0) * g_spin
                total_all_icohp += float((wab[occ_mask] * w_k).sum())
    return block_e, raw, icohp, total_all_icohp


def _finalize(block_e, raw, icohp, pair_list, kw, kpts, nspin, nk, width,
              npoints, window, spilling, charge_spilling, fermi, kind,
              band_window, total_all_icohp, method="operator"):
    band_energies = np.stack(block_e) if block_e else np.zeros((0, 0))
    kw = np.asarray(kw, dtype=float)
    all_e = band_energies.ravel() if band_energies.size else np.array([0.0])
    if window is None:
        window = (all_e.min() - 10 * width, all_e.max() + 10 * width)
    grid = np.linspace(window[0], window[1], npoints)

    pair_cohp, pair_icohp, band_cohp = {}, {}, {}
    total = np.zeros(npoints)
    for (i, j, _dist) in pair_list:
        lab = f"{i + 1}-{j + 1}"
        rawstack = np.stack(raw[(i, j)])               # (nblocks, nb), per state
        band_cohp[lab] = rawstack
        w = (rawstack * kw[:, None]).ravel()           # k-weight for the curve
        curve = _broaden(grid, band_energies.ravel(), w, width)
        pair_cohp[lab] = curve
        pair_icohp[lab] = icohp[(i, j)]
        total += curve
    return COHP(
        energy_eV=grid, pair_cohp=pair_cohp, pair_icohp=pair_icohp,
        total=total, total_icohp=float(sum(pair_icohp.values())),
        spilling=spilling, charge_spilling=charge_spilling,
        fermi_eV=None if fermi is None else float(fermi),
        kind=kind, band_window_eV=float(band_window), pairs=pair_list,
        method=method,
        nspin=nspin, nk=nk, block_kpts=np.asarray(kpts, dtype=float),
        block_kweights=kw, band_energies=band_energies, band_cohp=band_cohp,
    ), total_all_icohp


def _spilling_metrics(cap_blocks, occ_blocks, kw):
    """(total, charge) spilling from per-block captured weight and occupation.

    `cap_blocks[b]` (nb,) is the AO-captured weight of each state in [0, 1];
    `occ_blocks[b]` (nb,) its occupation in [0, 1]; `kw[b]` the k-weight.

    total  = 1 - <captured> over every band (the LOBSTER "total spilling")
    charge = 1 - <captured> over the OCCUPIED manifold (the "charge spilling"):
             the fraction of the real electron density the AO span misses, which
             is what bounds the reliability of the occupied-state ICOHP.
    """
    tot_cap = tot_w = ch_cap = ch_w = 0.0
    for cap, occ, w in zip(cap_blocks, occ_blocks, kw, strict=True):
        tot_cap += w * float(cap.sum())
        tot_w += w * cap.shape[0]
        ch_cap += w * float((occ * cap).sum())
        ch_w += w * float(occ.sum())
    total = 1.0 - tot_cap / tot_w if tot_w else 0.0
    charge = 1.0 - ch_cap / ch_w if ch_w else 0.0
    return float(total), float(charge)


def _step_occupations(eig_per_k, fermi):
    """Step occupations in [0, 1] at E_F per block. Used where the SCF result
    carries no stored occupations (the spinor NCResult); each spinor band holds
    one electron, so the step is 1 below E_F and 0 above."""
    f = -np.inf if fermi is None else float(fermi)
    return [(e.cpu().numpy() < f).astype(float) for e in eig_per_k]


@torch.no_grad()
def cohp(res, *, pairs=None, rcut: float = 3.0, width: float = 0.1,
         npoints: int = 800, window=None, method: str = "operator",
         resolve_images: bool = False, basis: str = "pswfc"):
    """Atom-pair COHP of a converged collinear SCF (norm-conserving, nspin 1/2).

    `pairs` selects atom index tuples (0-based); the default is every atom pair
    within `rcut` angstrom. Spin channels are summed.

    `method` selects the AO Hamiltonian: "operator" (default) evaluates
    H~ = <phi~|H^|phi~> with the converged Kohn-Sham operator (energy-zero
    invariant, the correct route for solids); "eigenvalue" uses the band-limited
    P^dagger diag(eps) P (cheaper, but carries the plane-wave energy zero). The
    operator route needs the norm-conserving KS Hamiltonian, so a USPP/PAW result
    falls back to the eigenvalue route.

    `resolve_images` restricts each pair to the single nearest image of atom j (the
    min-image bond) instead of the whole j sublattice, for a per-bond number
    comparable to LOBSTER. The image lattice vectors land in `bond_images`. Needs a
    full (unreduced) k-mesh for bonds that cross a cell boundary; exact at Gamma.

    `basis` selects the projector local basis: "pswfc" (default) the pseudo-atomic
    PP_PSWFC orbitals, or "iao" the Intrinsic Atomic Orbitals that span the occupied
    manifold exactly (zero occupied-state spilling, more localised off-site
    elements). "iao" needs the norm-conserving operator route. Returns a `COHP`.
    """
    from gradwave.scf.loop import SCFResult
    system, nspin, eig, coeffs, fermi, device, _ = _unpack_result(res)
    use_op = method == "operator" and isinstance(res, SCFResult)
    method = "operator" if use_op else "eigenvalue"
    if basis not in ("pswfc", "iao"):
        raise ValueError(f"basis must be 'pswfc' or 'iao', got {basis!r}")
    if basis == "iao" and not use_op:
        raise NotImplementedError(
            "basis='iao' needs the norm-conserving operator route "
            "(method='operator' on an SCFResult)")
    if use_op:
        from gradwave.core.hamiltonian import HamiltonianK, projectors
        shape = system.grid.shape

    cols = _atomic_columns(system)
    atom_of = np.array([c.atom for c in cols])
    pair_list = _select_pairs(system, pairs, rcut)
    kw = system.kweights.to(device)
    g_spin = 2.0 if nspin == 1 else 1.0

    # actual occupations normalized to [0, 1] per state (in [0,2] for nspin=1)
    occ_all = res.occupations
    # Loewdin amplitudes P_{np} = <phi~_p|psi_n> per (spin, k), flattened over spin
    proj_per_k, htilde_per_k, eig_per_k, kw_flat, kpts = [], [], [], [], []
    cap_blocks, occ_blocks = [], []
    band_window = -np.inf
    for isp in range(nspin):
        veff = res.v_eff if nspin == 1 else res.v_eff[isp]
        for ik, sph in enumerate(system.spheres):
            c = coeffs[isp][ik].to(device)                       # (nb, npw)
            e = eig[isp, ik].to(device)
            q = _ao_projectors_k(system, sph, cols, device)      # (nproj, npw)
            if basis == "iao":
                occ_mask = (e < fermi) if fermi is not None \
                    else torch.ones_like(e, dtype=torch.bool)
                q = _iao_projectors_k(q, c[occ_mask])            # occupied-span IAOs
            overlap = torch.einsum("ig,jg->ij", q.conj(), q)
            ois = _o_inv_sqrt(overlap)                           # O^{-1/2}
            # <phi~_p|psi> = becp @ O^{-1/2 T}; conj() matters for complex O off Gamma
            becp = torch.einsum("bg,pg->bp", c, q.conj())
            proj = becp @ ois.conj()                             # (nb, nproj)
            if use_op:
                pd = system.proj_data[ik]
                p = projectors(pd, system.positions).to(device)
                h = HamiltonianK(sph, shape, veff, pd, p)
                htilde = _htilde_operator(q, ois, h.apply)
            else:
                htilde = _htilde_eig(proj, e)
            proj_per_k.append(proj)
            htilde_per_k.append(htilde)
            eig_per_k.append(e)
            kw_flat.append(float(kw[ik]))
            kpts.append(np.asarray(sph.k_frac, dtype=float))
            cap_blocks.append((proj.real ** 2 + proj.imag ** 2).sum(1).cpu().numpy())
            occ_k = occ_all[ik] if nspin == 1 else occ_all[isp, ik]
            occ_blocks.append(occ_k.cpu().numpy() / g_spin)      # -> [0, 1]
            band_window = max(band_window, float(e.max()))
    spilling, charge_spilling = _spilling_metrics(cap_blocks, occ_blocks, kw_flat)

    block_e, raw, icohp, tot = _accumulate(
        proj_per_k, htilde_per_k, eig_per_k, kw_flat, atom_of, pair_list,
        g_spin, fermi)
    images = None
    if resolve_images:
        images = {(p[0], p[1]): _nearest_image_R(system, p[0], p[1])
                  for p in pair_list}
        raw, icohp = _accumulate_images(
            proj_per_k, htilde_per_k, eig_per_k, kw_flat, kpts, atom_of,
            pair_list, images, nspin, len(system.spheres), g_spin, fermi)
    out, _tot = _finalize(block_e, raw, icohp, pair_list, kw_flat, kpts, nspin,
                          len(system.spheres), width, npoints, window, spilling,
                          charge_spilling, fermi, "collinear", band_window, tot,
                          method=method)
    out._sumrule_icohp = tot  # all-pairs incl on-site, for validation
    out.basis = basis
    # RMSp: G-space projection residual. For the Loewdin reconstruction
    # X_n = P^{B2} psi_n, ||psi_n - X_n||^2 = 1 - captured_n, so the k-weighted RMS
    # equals sqrt(total spilling) (LOBSTER's model-independent metric).
    out.rmsp = float(np.sqrt(max(spilling, 0.0)))
    if images is not None:
        out.bond_images = {f"{i + 1}-{j + 1}": tuple(int(x) for x in R)
                           for (i, j), R in images.items()}
    return out


def projection_rmsp(res, *, basis: str = "pswfc", occupied_only: bool = False):
    """LOBSTER-style reciprocal-space projection residual RMSp, as a torch scalar.

        RMSp^2 = sum_{k,n} w_k || psi_n - P^{B2} psi_n ||^2 / sum_{k,n} w_k
               = sum_{k,n} w_k (1 - captured_n) / sum_{k,n} w_k,

    the k-weighted mean state spilling, computed directly from the projections. It
    is left DIFFERENTIABLE (no torch.no_grad): once the projector radial is built on
    the torch spherical-Bessel path (pseudo/radial_torch.py) rather than the numpy
    `sbt`, this is the objective a parameterised / contracted basis is optimised
    against. Caveat (see docs/plans/cohp-contracted-basis.md): minimising RMSp is the
    Sanchez-Portal spilling objective and favours a MORE diffuse/complete basis —
    right for band-energy fidelity, wrong for localised COHP, where IAO or a fitted
    minimal basis is the target. `occupied_only` restricts the sum to states below
    E_F (the charge RMSp). Norm-conserving collinear only."""
    from gradwave.scf.loop import SCFResult
    if not isinstance(res, SCFResult):
        raise NotImplementedError("projection_rmsp: norm-conserving SCFResult only")
    system, nspin, eig, coeffs, fermi, device, _ = _unpack_result(res)
    cols = _atomic_columns(system)
    kw = system.kweights.to(device)
    num = den = None
    for isp in range(nspin):
        for ik, sph in enumerate(system.spheres):
            c = coeffs[isp][ik].to(device)
            e = eig[isp, ik].to(device)
            q = _ao_projectors_k(system, sph, cols, device)
            if basis == "iao":
                occ_mask = (e < fermi) if fermi is not None \
                    else torch.ones_like(e, dtype=torch.bool)
                q = _iao_projectors_k(q, c[occ_mask])
            overlap = torch.einsum("ig,jg->ij", q.conj(), q)
            proj = torch.einsum("bg,pg->bp", c, q.conj()) @ _o_inv_sqrt(overlap).conj()
            cap = (proj.real ** 2 + proj.imag ** 2).sum(1)       # captured per band
            keep = (e < fermi) if (occupied_only and fermi is not None) \
                else torch.ones_like(e, dtype=torch.bool)
            wk = float(kw[ik])
            n = wk * (1.0 - cap[keep]).sum()
            d = wk * float(keep.sum())
            num = n if num is None else num + n
            den = d if den is None else den + d
    return (num / den).clamp_min(0.0).sqrt()


def _spinor_proj_per_k(res, cols, spinor, device):
    """Per-k Loewdin amplitudes for a spinor SCF. `spinor=True` uses the (j, mj)
    spin-angular AO projectors (SOC); `spinor=False` uses scalar AOs replicated on
    the two spin components (scalar-relativistic noncollinear). Returns
    (proj_per_k, eig_per_k, kpts, cap_blocks, spilling, band_window), where
    cap_blocks[k] is the per-state AO-captured weight used for charge spilling."""
    system = res.system
    m_pw = system.batch.npw_max
    kw = system.kweights.to(device)
    eig = res.eigenvalues
    proj_per_k, eig_per_k, kpts, cap_blocks = [], [], [], []
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
        kpts.append(np.asarray(sph.k_frac, dtype=float))
        w = float(kw[ik])
        cap = (proj.real ** 2 + proj.imag ** 2).sum(1).cpu().numpy()
        cap_blocks.append(cap)
        captured_kw += float(cap.sum()) * w
        total_kw += proj.shape[0] * w
        band_window = max(band_window, float(e.max()))
    spilling = 1.0 - captured_kw / total_kw if total_kw else 0.0
    return proj_per_k, eig_per_k, kpts, cap_blocks, float(spilling), band_window


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

    proj_per_k, eig_per_k, kpts, cap_blocks, spilling, band_window = \
        _spinor_proj_per_k(res, cols, spinor=False, device=device)
    kw = [float(w) for w in system.kweights]
    occ_blocks = _step_occupations(eig_per_k, res.fermi)
    _, charge_spilling = _spilling_metrics(cap_blocks, occ_blocks, kw)
    # spinor operator-route H~ needs the reconstructed SpinorHamiltonian (a
    # follow-up); the eigenvalue route is used here (fine for well-separated
    # atoms, energy-zero contaminated for short bonds — see the module docstring)
    htilde_per_k = [_htilde_eig(p, e) for p, e in zip(proj_per_k, eig_per_k,
                                                      strict=True)]
    block_e, raw, icohp, tot = _accumulate(
        proj_per_k, htilde_per_k, eig_per_k, kw, atom_of, pair_list, 1.0,
        res.fermi)
    out, _tot = _finalize(block_e, raw, icohp, pair_list, kw, kpts, 1,
                          len(system.spheres), width, npoints, window, spilling,
                          charge_spilling, res.fermi, "noncollinear",
                          band_window, tot, method="eigenvalue")
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

    proj_per_k, eig_per_k, kpts, cap_blocks, spilling, band_window = \
        _spinor_proj_per_k(res, cols, spinor=True, device=device)
    kw = [float(w) for w in system.kweights]
    occ_blocks = _step_occupations(eig_per_k, res.fermi)
    _, charge_spilling = _spilling_metrics(cap_blocks, occ_blocks, kw)
    htilde_per_k = [_htilde_eig(p, e) for p, e in zip(proj_per_k, eig_per_k,
                                                      strict=True)]
    block_e, raw, icohp, tot = _accumulate(
        proj_per_k, htilde_per_k, eig_per_k, kw, atom_of, pair_list, 1.0,
        res.fermi)
    out, _tot = _finalize(block_e, raw, icohp, pair_list, kw, kpts, 1,
                          len(system.spheres), width, npoints, window, spilling,
                          charge_spilling, res.fermi, "soc", band_window, tot,
                          method="eigenvalue")
    out._sumrule_icohp = _tot
    return out
