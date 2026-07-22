"""Bader charge analysis — grid partitioning of ρ(r) into atomic basins.

The Bader (QTAIM) partition splits real space at the zero-flux surfaces of the
charge density: every grid point is walked uphill along ∇ρ until it reaches a
local maximum (an "attractor"), and the volume draining into each attractor is
that basin. Integrating ρ over a basin gives the electron count assigned to the
attractor; summing basins by nearest nucleus gives per-atom electrons, and the
net Bader charge is q_a = Z_a^val − N_a.

Method — the on-grid steepest-ascent scheme of Henkelman, Arnaldsson & Jónsson
(Comput. Mater. Sci. 36, 354, 2006): at each grid point the ascent step goes to
the neighbour (of the 26 around it) that maximises (ρ_nb − ρ_0)/|Δr|, with |Δr|
the true Cartesian hop length so non-orthogonal cells are handled correctly. The
ascent map is a forest whose roots are the maxima; basins are found by pointer
jumping (vectorised path compression) rather than tracing each path in Python,
so the whole partition is a handful of grid-sized tensor ops. The on-grid method
has grid-aligned basin surfaces (an O(1/N) bias shared by all pure-grid Bader
codes); the Yu–Trinkle weight method removes it and is the natural refinement.

Pseudopotential caveat (important): gradwave's ρ is the VALENCE density. For
norm-conserving / USPP results it is the smooth pseudo-density with no nuclear
cusp, so basins partition valence charge only (the analog of running `bader` on
a bare CHGCAR without AECCAR) and charges are valence-referenced. PAW results
carry the augmented density, which is much closer to all-electron inside the
augmentation sphere and gives the most meaningful basins. `add_core=True` folds
the NLCC partial-core density back onto the grid where present, sharpening the
maxima at the nuclei.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# 26 nearest-neighbour offsets on the grid (all of {-1,0,1}^3 minus the origin).
_OFFSETS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
     if (i, j, k) != (0, 0, 0)],
    dtype=np.int64,
)


@dataclass
class BaderResult:
    """Per-atom Bader populations and the underlying basin decomposition.

    charges     (na,)   net Bader charge q_a = Z_a^val − N_a [e]; + is cationic
    electrons   (na,)   integrated valence electrons N_a in the atom's basins [e]
    volumes     (na,)   integrated Bader volume per atom [Å³]
    moments     (na,)   integrated spin moment ∫(ρ↑−ρ↓) over the basins [μ_B],
                        or None for a non-spin-polarised result
    valence     (na,)   Z_a^val the charge is referenced to [e]
    total_electrons     ∫ρ dr over the whole cell [e] (a partition sanity check)
    n_attractors        number of density maxima found
    attractor_frac      (n_attr,3) fractional coords of each attractor
    attractor_atom      (n_attr,)  index of the nucleus each attractor is bound to
    attractor_dist      (n_attr,)  distance attractor→its nucleus [Å]
    attractor_charge    (n_attr,)  electrons in each basin [e]
    nonnuclear          indices of attractors farther than `nna_tol` from every
                        nucleus (candidate non-nuclear attractors)
    """

    charges: np.ndarray
    electrons: np.ndarray
    volumes: np.ndarray
    moments: np.ndarray | None
    valence: np.ndarray
    total_electrons: float
    n_attractors: int
    attractor_frac: np.ndarray
    attractor_atom: np.ndarray
    attractor_dist: np.ndarray
    attractor_charge: np.ndarray
    nonnuclear: np.ndarray

    def __repr__(self) -> str:  # concise, atom-by-atom
        lines = ["BaderResult(", "  atom   Z_val   electrons    charge     volume"]
        for a in range(len(self.charges)):
            lines.append(
                f"  {a:4d}  {self.valence[a]:6.2f}  {self.electrons[a]:9.4f}  "
                f"{self.charges[a]:+9.4f}  {self.volumes[a]:9.3f}"
            )
        lines.append(
            f"  Σe⁻ = {self.electrons.sum():.4f} (∫ρ = {self.total_electrons:.4f}), "
            f"{self.n_attractors} attractors"
        )
        if len(self.nonnuclear):
            lines.append(f"  ⚠ {len(self.nonnuclear)} non-nuclear attractor(s)")
        lines.append(")")
        return "\n".join(lines)


def _ascent_map(rho: torch.Tensor, cell: np.ndarray) -> torch.Tensor:
    """Flat index of each grid point's steepest-ascent neighbour (self at a max).

    Incremental over the 26 offsets so peak memory stays O(grid), not O(26·grid).
    """
    n1, n2, n3 = rho.shape
    dev = rho.device
    # Cartesian hop vectors and lengths for each offset: Δr = Σ (dₐ/nₐ) a_a.
    dr = (_OFFSETS / np.array([n1, n2, n3])) @ cell  # (26, 3) Å
    dist = np.linalg.norm(dr, axis=1)  # (26,) Å

    best_score = torch.zeros_like(rho)  # self is the default; its score is 0
    best = torch.zeros((3, n1, n2, n3), dtype=torch.int64, device=dev)
    for off, d in zip(_OFFSETS.tolist(), dist.tolist(), strict=True):
        di, dj, dk = off
        # ρ at the neighbour p+off, brought back onto p's cell by a −off roll.
        rho_nb = torch.roll(rho, shifts=(-di, -dj, -dk), dims=(0, 1, 2))
        score = (rho_nb - rho) / d
        upd = score > best_score
        best_score = torch.where(upd, score, best_score)
        for c, dc in enumerate(off):
            best[c] = torch.where(upd, torch.full_like(best[c], dc), best[c])

    ii, jj, kk = torch.meshgrid(
        torch.arange(n1, device=dev), torch.arange(n2, device=dev),
        torch.arange(n3, device=dev), indexing="ij",
    )
    ti = (ii + best[0]) % n1
    tj = (jj + best[1]) % n2
    tk = (kk + best[2]) % n3
    return (ti * (n2 * n3) + tj * n3 + tk).reshape(-1)


def _basins(parent: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """Root (attractor flat index) of every point, by pointer jumping.

    The ascent map strictly increases ρ along each step, so the graph is acyclic
    and the doubling walk parent ← parent[parent] converges to the roots.
    """
    for _ in range(max_iter):
        nxt = parent[parent]
        if torch.equal(nxt, parent):
            return parent
        parent = nxt
    raise RuntimeError("Bader basin walk did not converge — density has a plateau/cycle")


def bader(
    res,
    add_core: bool = False,
    nna_tol: float = 0.5,
    vacuum_threshold: float | None = None,
) -> BaderResult:
    """Partition the SCF density into atomic Bader basins.

    res                the SCFResult / USPPResult holding ρ(r) and geometry.
    add_core           add the NLCC partial-core density (system.rho_core) back
                       onto the grid before partitioning, where the pseudos
                       carry one. Sharpens the maxima at the nuclei; the extra
                       core charge is NOT counted in the reported electrons, so
                       charges stay valence-referenced.
    nna_tol            an attractor farther than this (Å) from every nucleus is
                       flagged as a candidate non-nuclear attractor.
    vacuum_threshold   if given, basins whose attractor density is below this
                       (e/Å³) are treated as vacuum: their charge/volume is not
                       assigned to any atom (use for slabs/molecules-in-a-box).

    Returns a BaderResult. Uses the total density; for a spin-polarised result
    the spin density is partitioned too, giving a per-atom moment.
    """
    system = res.system
    grid = system.grid
    cell = np.asarray(grid.cell, dtype=np.float64)
    shape = tuple(int(n) for n in grid.shape)
    volume = float(grid.volume)
    n_points = shape[0] * shape[1] * shape[2]
    voxel = volume / n_points

    rho = res.rho.detach().to(torch.float64)
    if rho.shape != shape:
        rho = rho.reshape(shape)
    rho_for_ascent = rho
    if add_core and getattr(system, "rho_core", None) is not None:
        rho_for_ascent = rho + system.rho_core.detach().to(torch.float64).reshape(shape)

    # 1) steepest-ascent forest → 2) basin roots by pointer jumping.
    parent = _ascent_map(rho_for_ascent, cell)
    roots = _basins(parent)
    attractors, basin_of_point = torch.unique(roots, return_inverse=True)
    n_attr = int(attractors.numel())

    # Per-basin integrated electrons / volume / spin, on the reported ρ.
    rho_flat = rho.reshape(-1)
    basin_charge = torch.zeros(n_attr, dtype=torch.float64, device=rho.device)
    basin_charge.scatter_add_(0, basin_of_point, rho_flat * voxel)
    basin_volume = torch.zeros(n_attr, dtype=torch.float64, device=rho.device)
    basin_volume.scatter_add_(0, basin_of_point, torch.full_like(rho_flat, voxel))

    basin_moment = None
    rho_spin = getattr(res, "rho_spin", None)
    if rho_spin is not None:
        spin = (rho_spin[0] - rho_spin[1]).detach().to(torch.float64).reshape(-1)
        basin_moment = torch.zeros(n_attr, dtype=torch.float64, device=rho.device)
        basin_moment.scatter_add_(0, basin_of_point, spin * voxel)

    # Attractor fractional coordinates from their flat indices.
    n1, n2, n3 = shape
    fi = torch.div(attractors, n2 * n3, rounding_mode="floor")
    rem = attractors - fi * (n2 * n3)
    fj = torch.div(rem, n3, rounding_mode="floor")
    fk = rem - fj * n3
    attr_frac = torch.stack(
        [fi / n1, fj / n2, fk / n3], dim=1
    ).cpu().numpy().astype(np.float64)

    # Bind each attractor to its nearest nucleus (minimum image).
    pos = system.positions.detach().cpu().numpy().astype(np.float64)
    inv_cell = np.linalg.inv(cell)
    nuc_frac = pos @ inv_cell  # (na, 3)
    na = pos.shape[0]
    dfrac = attr_frac[:, None, :] - nuc_frac[None, :, :]  # (n_attr, na, 3)
    dfrac -= np.round(dfrac)
    dcart = dfrac @ cell
    d2 = np.einsum("...i,...i->...", dcart, dcart)  # (n_attr, na)
    attr_atom = d2.argmin(axis=1)
    attr_dist = np.sqrt(d2[np.arange(n_attr), attr_atom])

    charge_np = basin_charge.cpu().numpy()
    volume_np = basin_volume.cpu().numpy()
    moment_np = None if basin_moment is None else basin_moment.cpu().numpy()

    # Vacuum basins (optional) are dropped from atom assignment.
    attr_rho = rho_for_ascent.reshape(-1)[attractors].cpu().numpy()
    keep = np.ones(n_attr, dtype=bool)
    if vacuum_threshold is not None:
        keep = attr_rho >= vacuum_threshold

    electrons = np.zeros(na, dtype=np.float64)
    volumes = np.zeros(na, dtype=np.float64)
    moments = None if moment_np is None else np.zeros(na, dtype=np.float64)
    for b in range(n_attr):
        if not keep[b]:
            continue
        a = attr_atom[b]
        electrons[a] += charge_np[b]
        volumes[a] += volume_np[b]
        if moments is not None:
            moments[a] += moment_np[b]

    valence = system.charges.detach().cpu().numpy().astype(np.float64)
    charges = valence - electrons
    nonnuclear = np.where(attr_dist > nna_tol)[0]

    return BaderResult(
        charges=charges,
        electrons=electrons,
        volumes=volumes,
        moments=moments,
        valence=valence,
        total_electrons=float(rho_flat.sum().item() * voxel),
        n_attractors=n_attr,
        attractor_frac=attr_frac,
        attractor_atom=attr_atom,
        attractor_dist=attr_dist,
        attractor_charge=charge_np,
        nonnuclear=nonnuclear,
    )
