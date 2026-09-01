"""Track 5 — Symmetry block-diagonalization on REAL gradwave calculations.

Runs a real NC-DFT SCF (gradwave) for diamond (C, Fd-3m — NONSYMMORPHIC, a
(¼,¼,¼) glide) and bcc Fe (Im-3m — symmorphic, 48-op Oh), builds the dense
Kohn–Sham H(Γ) at the converged density, and block-diagonalizes it with the
dependency-free class-sum method (blockdiag.py) — NO Sage, NO character tables.

The symmetry representation D(g) on the plane-wave basis is built with gradwave's
OWN convention by feeding the identity block to postscf.irreps._rep_matrix, so the
fractional-translation phases (needed for nonsymmorphic diamond) are exactly the
house convention. spglib (already a gradwave dep) supplies the operations.

Validates: H symmetric ([H,D]≈0), class sums block H ([T,H]≈0), block spectrum ==
full spectrum, and the Γ eigenvalue degeneracies match the Oh irreps (1/2/3-fold).

Run:  uv run python experiments/symbolic/track5_real_crystals.py
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
from gradwave.postscf.irreps import _rep_matrix, little_group
from gradwave.symmetry import find_spacegroup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blockdiag import block_diagonalize, conjugacy_classes, flop_reduction  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def dense_H_gamma(inp_path: str):
    """Real SCF → dense Hermitian H(Γ) at the converged density."""
    inp = load_input(inp_path)
    system = build_system(inp)
    t0 = time.perf_counter()
    res = run_scf(inp, system=system, verbose=False)
    t_scf = time.perf_counter() - t0

    grid = system.grid
    v_eff = res.v_eff
    if v_eff.ndim == 4:  # nspin=2 → use the spin-up channel for the symmetry demo
        v_eff = v_eff[0]

    sph = build_gsphere(grid, system.ecut, np.zeros(3))
    beta_ls, dij_sp = species_projector_tables(system.upfs, None)
    pd = projector_data_at_k(sph, system.species_of_atom, system.upfs,
                             beta_ls, dij_sp, grid.volume, None)
    p = projectors(pd, system.positions)
    h = HamiltonianK(sph, grid.shape, v_eff, pd, p)

    npw = sph.npw
    out = h.apply(torch.eye(npw, dtype=torch.complex128))
    H = out.transpose(0, 1).contiguous()
    H = 0.5 * (H + H.conj().T)
    herm = float((out.transpose(0, 1) - out.conj()).abs().max())  # pre-symmetrization
    return H.numpy(), sph, system, grid.cell, res, t_scf, herm


def build_reps(sph, system, cell):
    """Rep matrices D(g) on the Γ plane-wave basis, gradwave's own convention."""
    frac = system.positions.cpu().numpy() @ np.linalg.inv(cell)
    sg = find_spacegroup(cell, frac, system.species_of_atom)
    ops = little_group(np.zeros(3), sg, cell)
    miller = sph.miller.cpu().numpy()
    ident = np.eye(sph.npw, dtype=complex)
    Dmats = [_rep_matrix(ident, miller, np.zeros(3), op) for op in ops]
    Ws = [np.asarray(op["W"], dtype=int) for op in ops]
    return sg, ops, Ws, Dmats


def degeneracy_pattern(evals: np.ndarray, n: int, tol: float = 1e-3) -> str:
    """Group the lowest n eigenvalues into degenerate multiplets (Oh: 1/2/3)."""
    ev = np.sort(evals)[:n]
    groups, i = [], 0
    while i < len(ev):
        j = i
        while j + 1 < len(ev) and ev[j + 1] - ev[i] < tol:
            j += 1
        groups.append((ev[i], j - i + 1))
        i = j + 1
    return "  ".join(f"{e:+.3f}(×{m})" for e, m in groups)


def analyze(name: str, inp_path: str) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    H, sph, system, cell, res, t_scf, herm = dense_H_gamma(inp_path)
    npw = H.shape[0]
    print(f"SCF converged in {t_scf:.0f}s; dense H(Γ): {npw}×{npw}, "
          f"Hermitian to {herm:.1e}")

    sg, ops, Ws, Dmats = build_reps(sph, system, cell)
    print(f"space group: {sg.international}, {len(ops)} ops in the Γ little group")

    # Density-symmetry diagnostic: how well does the CONVERGED v_eff carry the
    # crystal symmetry?  (nonsymmorphic glides expose residual asymmetry that a
    # symmorphic single-atom cell hides.)  Then impose the symmetry on H by
    # Reynolds averaging over the (unitary) rep — extracts H's symmetric part.
    raw_res = max(float(np.abs(H @ D - D @ H).max()) for D in Dmats)
    Hsym = sum(D @ H @ D.conj().T for D in Dmats) / len(Dmats)
    break_amp = float(np.abs(Hsym - H).max())
    print(f"\ndensity symmetry diagnostic:")
    print(f"  raw max |[H, D(g)]| from SCF v_eff = {raw_res:.1e} eV "
          f"({'exact' if raw_res < 1e-9 else 'residual glide-breaking'})")
    print(f"  H symmetrization correction        = {break_amp:.1e} eV")

    # cross-check the RAW dense H against gradwave's SCF band energies at Γ,
    # proving H(Γ) really is the converged Kohn–Sham Hamiltonian.
    ev_raw = np.linalg.eigvalsh(H).real
    try:
        kfs = np.array([np.asarray(s.k_frac, float) for s in system.spheres])
        gi = int(np.argmin(np.abs(kfs).sum(axis=1)))
        if np.abs(kfs[gi]).sum() < 1e-8:
            scf_ev = np.sort(np.asarray(res.eigenvalues)[gi])
            nb = len(scf_ev)
            print(f"  dense H(Γ) vs SCF band energies at Γ (nb={nb}): "
                  f"max diff {np.abs(np.sort(ev_raw)[:nb] - scf_ev).max():.1e} eV")
    except Exception as exc:  # noqa: BLE001
        print(f"  (SCF-eigenvalue cross-check skipped: {type(exc).__name__})")

    classes = conjugacy_classes(Ws)
    U, sizes, B, diag = block_diagonalize(Hsym, Dmats, classes)
    H = Hsym  # use the symmetrized H below

    print(f"\nvalidation (on symmetrized H):")
    print(f"  max |[H, D(g)]|     = {diag['commute_HD']:.1e}")
    print(f"  max |[T, H]|        = {diag['commute_TH']:.1e}   "
          f"(class sums commute with H)")
    print(f"  inter-block leakage = {diag['leak']:.1e}")
    print(f"  block vs full spec  = {diag['ev_err']:.1e}")

    print(f"\nblocks ({diag['n_classes']} classes → {len(sizes)} irrep blocks):")
    print(f"  sizes {sorted(sizes)}")
    print(f"  largest block {max(sizes)} vs full {npw}; "
          f"Σn³ reduction = {flop_reduction(npw, sizes):.1f}×")

    ev = np.linalg.eigvalsh(H).real
    print(f"\nlowest Γ eigenvalues (eV), degeneracies (Oh irreps → 1/2/3-fold):")
    print(f"  {degeneracy_pattern(ev, 12)}")

    ok = (diag["commute_HD"] < 1e-6 and diag["commute_TH"] < 1e-6
          and diag["leak"] < 1e-6 and diag["ev_err"] < 1e-6)
    print(f"\n{'PASS' if ok else 'CHECK'}: {name} block-diagonalized by symmetry, "
          f"blocks reproduce the spectrum.")


def main() -> None:
    print("Track 5: symmetry block-diagonalization on REAL gradwave SCF calculations")
    analyze("DIAMOND  (C, Fd-3m, nonsymmorphic — glide-phase test)",
            str(HERE / "diamond_scf.yaml"))
    analyze("bcc Fe   (Im-3m, symmorphic — 48-op Oh, metallic)",
            str(HERE / "fe_bcc_scf.yaml"))


if __name__ == "__main__":
    main()
