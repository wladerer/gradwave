"""Track 6 — matrix-free symmetry block-diagonalization at OFF-Γ k-points.

Runs a real gradwave SCF, then at several high-symmetry k builds H(k) and
block-diagonalizes it MATRIX-FREE via the star construction (blockdiag_matfree):
the symmetry-adapted basis is built star-by-star (O(Σs³), never npw×npw) and each
irrep block is assembled by applying gradwave's HamiltonianK.apply to the block's
sparse basis vectors — no dense H in the solve path.

diamond fcc: Γ, L, X   (X is the nonsymmorphic PROJECTIVE case — watch it break)
bcc Fe:      Γ, H, P, N (symmorphic — ordinary reps everywhere)

Run:  uv run python experiments/symbolic/track6_offgamma.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.api import build_system, run_scf
from gradwave.core.hamiltonian import HamiltonianK, projectors
from gradwave.grids import build_gsphere
from gradwave.inputs import load_input
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.postscf.irreps import little_group
from gradwave.symmetry import find_spacegroup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blockdiag_matfree import adapted_basis, solve_blocks_matfree, sparse_reps  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def scf(inp_path: str):
    inp = load_input(inp_path)
    system = build_system(inp)
    t0 = time.perf_counter()
    res = run_scf(inp, system=system, verbose=False)
    dt = time.perf_counter() - t0
    frac = system.positions.cpu().numpy() @ np.linalg.inv(system.grid.cell)
    sg = find_spacegroup(system.grid.cell, frac, system.species_of_atom)
    return system, res, sg, dt


def h_at_k(system, v_eff, k_frac):
    grid = system.grid
    sph = build_gsphere(grid, system.ecut, np.asarray(k_frac, float))
    beta_ls, dij_sp = species_projector_tables(system.upfs, None)
    pd = projector_data_at_k(sph, system.species_of_atom, system.upfs,
                             beta_ls, dij_sp, grid.volume, None)
    p = projectors(pd, system.positions)
    return HamiltonianK(sph, grid.shape, v_eff, pd, p), sph


def dense_and_sym(h, sph, ops, k_frac):
    """Dense H(k) (validation oracle) + its Reynolds-symmetrized version."""
    npw = sph.npw
    H = h.apply(torch.eye(npw, dtype=torch.complex128)).transpose(0, 1).numpy()
    H = 0.5 * (H + H.conj().T)
    miller = sph.miller.cpu().numpy()
    perms, phases = sparse_reps(miller, ops, np.asarray(k_frac, float))
    Hsym = np.zeros_like(H)
    for P, ph in zip(perms, phases):
        M = (ph[:, None] * H) * np.conj(ph)[None, :]
        Hsym[np.ix_(P, P)] += M
    Hsym /= len(ops)
    return H, Hsym, float(np.abs(Hsym - H).max())


def analyze_k(system, v_eff, sg, kname, kfrac):
    cell = system.grid.cell
    ops = little_group(np.asarray(kfrac, float), sg, cell)
    h, sph = h_at_k(system, v_eff, kfrac)
    npw = sph.npw
    miller = sph.miller.cpu().numpy()
    Ws = [np.asarray(op["W"], int) for op in ops]

    H, Hsym, asym = dense_and_sym(h, sph, ops, kfrac)
    ev_sym = np.sort(np.linalg.eigvalsh(Hsym).real)

    U, sizes, diag = adapted_basis(miller, ops, Ws, np.asarray(kfrac, float))
    ev_blk, napply, peak = solve_blocks_matfree(h.apply, U, sizes)
    method_err = float(np.abs(ev_blk - ev_sym).max())

    N3, s3 = npw ** 3, diag["sum_s3"]
    red = N3 / sum(s ** 3 for s in sizes)
    projective = method_err > 1e-6
    tag = "PROJECTIVE — plain class sums insufficient" if projective else "ordinary — clean"
    print(f"  {kname:2s}  |G_k|={len(ops):2d}  npw={npw:4d}  "
          f"blocks={len(sizes):2d} max={max(sizes):3d}  "
          f"reduce={red:5.1f}×  basisΣs³/npw³={s3/N3:.4f}  "
          f"blk-vs-full={method_err:.1e}  [{tag}]")
    if asym > 1e-6:
        print(f"        (density glide-breaking at this k: {asym:.1e} eV)")
    return not projective


def main() -> None:
    print("Track 6: matrix-free symmetry block-diagonalization off Γ")
    print("(basisΣs³/npw³ ≪ 1 ⇒ near-linear basis build; reduce = Σn³ diag speedup)\n")

    print("DIAMOND (C, Fd-3m, nonsymmorphic):")
    system, res, sg, dt = scf(str(HERE / "diamond_scf.yaml"))
    print(f"  [SCF {dt:.0f}s, {sg.international}]")
    for kn, kf in [("Γ", (0, 0, 0)), ("L", (0.5, 0.5, 0.5)), ("X", (0.5, 0.0, 0.5))]:
        analyze_k(system, res.v_eff, sg, kn, kf)

    print("\nbcc Fe (Im-3m, symmorphic, metallic):")
    system, res, sg, dt = scf(str(HERE / "fe_bcc_scf.yaml"))
    print(f"  [SCF {dt:.0f}s, {sg.international}]")
    for kn, kf in [("Γ", (0, 0, 0)), ("H", (0.5, -0.5, 0.5)),
                   ("P", (0.25, 0.25, 0.25)), ("N", (0.0, 0.0, 0.5))]:
        analyze_k(system, res.v_eff, sg, kn, kf)


if __name__ == "__main__":
    main()
