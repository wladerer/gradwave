"""Band symmetry analysis: irrep labels (Mulliken symbols) at a k-point.

For each little-group operation g = {S|t} (Sk ≡ k mod G), the action on a
Bloch state in plane waves is a Miller-index permutation plus phases:

    (O_g ψ)(r) = ψ(g⁻¹r)  ⇒  (O_g c)(G′) = c(G)·e^{−i(k+G′)·t},
    G′ = S G + G₀  ↔  m′ = W⁻ᵀ m + g₀,   g₀ = W⁻ᵀ k_frac − k_frac (integer)
    e^{−i(k+G′)·t} = e^{−2πi (k_frac + m′)·w}

The representation matrix D_mn(g) = ⟨ψ_m|O_g|ψ_n⟩ over a degenerate cluster
gives characters χ(g) = tr D — basis- and phase-convention independent.
Mulliken names are assigned by RULE, not table lookup:

    dim 1 → A/B by sign of χ(principal C_n); dim 2 → E_j from
    χ(C_n) = 2cos(2πj/n); dim 3 (cubic) → T_{1,2} by sign of χ(C₄)/χ(S₄);
    subscript 1/2 from χ(C₂′) (else χ(σ_v)); g/u from χ(i); ′/″ from χ(σ_h).

Caveats (reported, not hidden):
- Zone-boundary k in NON-symmorphic groups can carry projective reps —
  labels there may not match tabulated ordinary-rep names (warned).
- A cluster whose D matrices are not unitary (‖D‖_F² ≠ dim) indicates an
  accidental degeneracy fused by cluster_tol — labeled "?" with a warning.
- B₁/B₂-type subscripts depend on which C₂′/σ_v a textbook calls "primed";
  our deterministic axis choice may differ from a given table's orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.hamiltonian import HamiltonianK, build_projector_data, projectors
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere
from gradwave.pseudo.kb import beta_form_factors
from gradwave.solvers.davidson import davidson
from gradwave.symmetry import find_spacegroup


@dataclass
class IrrepCluster:
    energies: list
    dim: int
    label: str
    characters: dict  # class name -> character (real part shown)
    warning: str = ""


@dataclass
class KPointIrreps:
    k_frac: np.ndarray
    n_ops: int
    clusters: list = field(default_factory=list)

    def __str__(self):
        lines = [f"k = {np.round(self.k_frac, 6)}  (little group: {self.n_ops} ops)"]
        for c in self.clusters:
            e = ", ".join(f"{x:9.4f}" for x in c.energies)
            warn = f"   ! {c.warning}" if c.warning else ""
            lines.append(f"  {c.label:>5s} (dim {c.dim}):  {e} eV{warn}")
        return "\n".join(lines)


def _cartesian_rotation(w_mat: np.ndarray, cell: np.ndarray) -> np.ndarray:
    a_t = np.asarray(cell, dtype=float).T
    return a_t @ w_mat @ np.linalg.inv(a_t)


def _classify_op(s: np.ndarray):
    """(kind, order, axis): kind ∈ E, i, C (proper), sigma, S (other improper)."""
    det = float(np.linalg.det(s))
    r = s * np.sign(det)
    tr = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(tr)
    if theta < 1e-6:
        return ("E", 1, None) if det > 0 else ("i", 2, None)
    order = int(round(2.0 * math.pi / theta))
    w_, v = np.linalg.eigh((r + r.T) / 2.0)
    axis = v[:, np.argmax(w_)]
    if det > 0:
        return "C", order, axis
    if abs(theta - math.pi) < 1e-6:
        return "sigma", 2, axis  # mirror; axis = plane normal
    return "S", order, axis


def little_group(k_frac: np.ndarray, sg, cell: np.ndarray) -> list[dict]:
    """Operations with W⁻ᵀk ≡ k (mod 1), with Cartesian classification."""
    k_frac = np.asarray(k_frac, dtype=float)
    ops = []
    for w_mat, w_vec in zip(sg.rotations, sg.translations, strict=True):
        w_inv_t = np.round(np.linalg.inv(w_mat).T).astype(np.int64)
        g0 = w_inv_t @ k_frac - k_frac
        if np.max(np.abs(g0 - np.round(g0))) > 1e-6:
            continue
        s = _cartesian_rotation(w_mat, cell)
        kind, order, axis = _classify_op(s)
        ops.append(dict(W=w_mat, w=np.asarray(w_vec, float), Winv_t=w_inv_t,
                        g0=np.round(g0).astype(np.int64), S=s,
                        kind=kind, order=order, axis=axis))
    return ops


def _rep_matrix(c: np.ndarray, miller: np.ndarray, k_frac: np.ndarray, op) -> np.ndarray:
    """D_mn = ⟨ψ_m|O_g|ψ_n⟩ for a band block c (nb, npw)."""
    index = {tuple(m): i for i, m in enumerate(miller)}
    mprime = miller @ op["Winv_t"].T + op["g0"]
    perm = np.array([index[tuple(m)] for m in mprime])
    phase = np.exp(-2j * math.pi * ((k_frac + mprime) @ op["w"]))
    c_rot = np.zeros_like(c)
    c_rot[:, perm] = c * phase[None, :]
    return np.conj(c) @ c_rot.T


def _principal(ops):
    """(n, axis) of the principal rotation; n=1 if no proper rotations."""
    best = (1, None)
    axes = {}
    for op in ops:
        if op["kind"] == "C":
            key = tuple(np.round(np.abs(op["axis"]), 6))
            axes.setdefault(key, []).append(op)
            if op["order"] > best[0]:
                best = (op["order"], op["axis"])
    # cubic detection: more than one axis carrying order-3+ rotations
    high_axes = {tuple(np.round(np.abs(o["axis"]), 6))
                 for o in ops if o["kind"] == "C" and o["order"] >= 3}
    cubic = len(high_axes) > 1
    return best[0], best[1], cubic


def _chi(clusters_chi, ops, select):
    """Real class character used as a Mulliken discriminant.

    Not the plain class mean: at a zone-boundary k the members of one class can
    carry projective phases (e.g. the three C₂′ of graphene's K little group come
    out as {1, ω, ω²}, the cube roots of unity), so ``mean(Re χ)`` collapses to
    ~0 and its sign is numerical noise — the label then flips between platforms.
    The gauge-coherent representative is the class member whose character is real
    (largest |Re|, imaginary part ~0); its sign is the physical discriminant and
    is stable. Reduces to the mean for an ordinary class where all members agree.
    """
    vals = [chi for chi, op in zip(clusters_chi, ops, strict=True) if select(op)]
    if not vals:
        return None
    rep = max(vals, key=lambda x: abs(np.real(x)))
    return float(np.real(rep))


def _mulliken(chis: list, ops: list, dim: int) -> str:
    n, axis, cubic = _principal(ops)

    def par(op_axis):
        return axis is not None and op_axis is not None and \
            abs(abs(float(np.dot(op_axis, axis))) - 1.0) < 1e-6

    chi_i = _chi(chis, ops, lambda o: o["kind"] == "i")
    chi_cn = _chi(chis, ops, lambda o: o["kind"] == "C" and o["order"] == n and par(o["axis"]))
    chi_sh = _chi(chis, ops, lambda o: o["kind"] == "sigma" and par(o["axis"]))
    chi_c2p = _chi(chis, ops, lambda o: o["kind"] == "C" and o["order"] == 2
                   and o["axis"] is not None and not par(o["axis"]))
    chi_sv = _chi(chis, ops, lambda o: o["kind"] == "sigma" and o["axis"] is not None
                  and not par(o["axis"]))
    chi_c4 = _chi(chis, ops, lambda o: o["kind"] == "C" and o["order"] == 4)
    chi_s4 = _chi(chis, ops, lambda o: o["kind"] == "S" and o["order"] == 4)

    if cubic:
        if dim == 3:
            disc = chi_c4 if chi_c4 is not None else chi_s4
            base = "T" + ("1" if (disc is None or disc > 0) else "2")
        elif dim == 2:
            base = "E"
        elif dim == 1:
            disc = chi_c4 if chi_c4 is not None else chi_s4
            base = "A" + ("1" if (disc is None or disc > 0) else "2")
        else:
            return "?"
    else:
        # D2-type: three inequivalent C2 axes, no higher rotation
        c2_axes = {tuple(np.round(np.abs(o["axis"]), 4))
                   for o in ops if o["kind"] == "C" and o["order"] == 2}
        if n == 2 and len(c2_axes) >= 2 and dim == 1:
            axes_sorted = sorted(c2_axes, key=lambda a: (-abs(a[2]), -abs(a[1])))
            chi_axes = [
                _chi(chis, ops, lambda o, ax=ax: o["kind"] == "C" and o["order"] == 2
                     and tuple(np.round(np.abs(o["axis"]), 4)) == ax)
                for ax in axes_sorted
            ]
            if all(c is not None and c > 0 for c in chi_axes):
                base = "A"
            else:
                pos = [i for i, c in enumerate(chi_axes) if c is not None and c > 0]
                base = f"B{pos[0] + 1}" if pos else "B?"
        elif dim == 1:
            base = "A" if (chi_cn is None or chi_cn > 0) else "B"
            disc = chi_c2p if chi_c2p is not None else chi_sv
            if disc is not None and n > 2:
                base += "1" if disc > 0 else "2"
            elif disc is not None and n == 2:
                base += "1" if disc > 0 else "2"
        elif dim == 2:
            base = "E"
            if chi_cn is not None and n >= 5:
                j = int(round(n * math.acos(max(-1.0, min(1.0, chi_cn / 2.0)))
                              / (2.0 * math.pi)))
                base += str(max(j, 1))
        else:
            return "?"

    if chi_i is not None:
        base += "g" if chi_i / dim > 0 else "u"
    elif chi_sh is not None:
        base += "'" if chi_sh / dim > 0 else "''"
    return base


def _class_name(op) -> str:
    if op["kind"] in ("E", "i"):
        return op["kind"]
    if op["kind"] == "sigma":
        return "sigma"
    return f"{op['kind']}{op['order']}"


def band_irreps(res, k_frac, nbands: int | None = None, cluster_tol: float = 1e-3,
                diago_tol: float = 1e-10) -> KPointIrreps:
    """Solve at k on the converged potential and label bands by irrep."""
    system = res.system
    grid = system.grid
    cell = grid.cell
    device = res.v_eff.device
    nbands = nbands or system.nbands
    k_frac = np.asarray(k_frac, dtype=float)

    frac = system.positions.cpu().numpy() @ np.linalg.inv(cell)
    sg = system.sym or find_spacegroup(cell, frac, system.species_of_atom)
    ops = little_group(k_frac, sg, cell)

    sph = build_gsphere(grid, system.ecut, k_frac, device=device)
    q = np.sqrt(sph.kpg2.cpu().numpy())
    beta_tables = [torch.as_tensor(beta_form_factors(u, q), dtype=RDTYPE, device=device)
                   for u in system.upfs]
    beta_ls = [[b.l for b in u.betas] for u in system.upfs]
    dij_species = [torch.as_tensor(u.dij, dtype=RDTYPE, device=device) for u in system.upfs]
    pd = build_projector_data(sph, system.species_of_atom, beta_tables, beta_ls,
                              dij_species, grid.volume)
    p = projectors(pd, system.positions)
    h = HamiltonianK(sph, grid.shape, res.v_eff, pd, p)
    c0 = torch.zeros(nbands, sph.npw, dtype=CDTYPE, device=device)
    c0[torch.arange(nbands), torch.arange(nbands)] = 1.0
    # warm start from SCF orbitals when k coincides with a mesh point —
    # keeps symmetric degenerate subspaces intact and converges in a few steps
    for ik, s_scf in enumerate(system.spheres):
        if np.max(np.abs((s_scf.k_frac - k_frac + 0.5) % 1.0 - 0.5)) < 1e-9:
            nb0 = min(nbands, res.coeffs[ik].shape[0])
            c0[:nb0] = res.coeffs[ik][:nb0].to(device)
            break
    out = davidson(h.apply, c0, HBAR2_2M * sph.kpg2, tol=diago_tol, max_iter=300)

    eigs = out.eigenvalues.cpu().numpy()
    coeffs = out.eigenvectors.cpu().numpy()
    miller = sph.miller.cpu().numpy()

    solve_warn = ""
    if float(out.residual_norms.max()) > 1e-6:
        solve_warn = f"eigensolve residual {float(out.residual_norms.max()):.1e}"

    # Origin gauge: an origin shift s multiplies χ(g) by e^{−2πi g0(g)·s}
    # (invariant at Γ where g0 = 0). Tabulated labels correspond to the
    # standard origin; pick s ∈ {0, ±origin_shift} making characters
    # maximally real (ordinary crystallographic reps have real characters
    # for these groups).
    shift = getattr(sg, "origin_shift", None)
    gauge_candidates = [np.zeros(3)]
    if shift is not None and np.abs(shift).max() > 1e-8:
        gauge_candidates += [np.asarray(shift, float), -np.asarray(shift, float)]

    nonsym_warn = solve_warn
    if any(np.abs(op["w"] - np.round(op["w"])).max() > 1e-6 for op in ops) and \
       any(np.abs(op["g0"]).max() > 0 for op in ops) and (
           shift is None or np.abs(shift).max() < 1e-8):
        warn2 = "non-symmorphic zone-boundary k: labels may be projective"
        nonsym_warn = f"{solve_warn}; {warn2}" if solve_warn else warn2

    result = KPointIrreps(k_frac=k_frac, n_ops=len(ops))
    start = 0
    while start < nbands:
        stop = start + 1
        while stop < nbands and eigs[stop] - eigs[stop - 1] < cluster_tol:
            stop += 1
        block = coeffs[start:stop]
        dim = stop - start
        chis_raw, unitary = [], True
        for op in ops:
            d = _rep_matrix(block, miller, k_frac, op)
            chis_raw.append(np.trace(d))
            if abs(np.linalg.norm(d) ** 2 - dim) > 1e-3 * dim:
                unitary = False
        # gauge-correct characters to the standard origin
        best_chis, best_cost = chis_raw, float("inf")
        for s_gauge in gauge_candidates:
            cand = [chi * np.exp(-2j * math.pi * float(op["g0"] @ s_gauge))
                    for chi, op in zip(chis_raw, ops, strict=True)]
            cost = sum(abs(np.imag(c)) for c in cand)
            if cost < best_cost - 1e-9:
                best_chis, best_cost = cand, cost
        chis = best_chis
        label = _mulliken(chis, ops, dim) if unitary else "?"
        warn = nonsym_warn if unitary else "cluster not closed under group (accidental degeneracy?)"
        # aggregate characters by class for display
        char = {}
        for chi, op in zip(chis, ops, strict=True):
            char.setdefault(_class_name(op), []).append(np.real(chi))
        char = {k2: float(np.mean(v)) for k2, v in char.items()}
        result.clusters.append(IrrepCluster(
            energies=[float(e) for e in eigs[start:stop]], dim=dim,
            label=label, characters=char, warning=warn,
        ))
        start = stop
    return result
