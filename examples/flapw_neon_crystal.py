"""AutoAPW: a differentiable all-electron FLAPW crystal from scratch, in PyTorch.

gradwave is a pseudopotential plane-wave code; this example drives the parallel
*all-electron* full-potential-LAPW pipeline in ``gradwave.flapw`` — no
pseudopotential, the bare nuclear ``-Z/r`` handled by muffin-tin augmentation.
Three stages, each checked against an independent reference:

  1. Atomic KS SCF (log-mesh Numerov + tridiagonal radial eigensolver, LDA XC)
     vs the NIST LDA atomic reference eigenvalues.
  2. The empty-lattice gate: with V=0 in the sphere the LAPW secular equation
     must reproduce the free-electron bands ½|k+G|² — the correctness anchor.
  3. Self-consistent crystal FLAPW for simple-cubic Ne (LAPW Bloch solve ->
     interstitial FFT density + sphere augmentation density + frozen core ->
     Weinert Coulomb -> LDA -> Anderson mixing), validated two ways: the dilute
     limit (a=10 Bohr) recovers the isolated atom, and the a=6 Bohr crystal
     2s-2p splitting matches Elk 11.0.2 (an established all-electron FLAPW code).

The "Auto" in AutoAPW: the augmented-basis primitives are differentiable by
construction. The final panel shows autograd through the muffin-tin radial
match — ∂u_l(R_MT)/∂E_l from a single backward pass, exact against a
finite-difference check — the hook for fitting basis/linearization parameters
by gradient descent.

Run:
    uv run python examples/flapw_neon_crystal.py --outdir examples
Runtime: ~3-5 min on CPU (two self-consistent crystal loops).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, HBAR2_2M
from gradwave.flapw import (
    NIST_LDA_EV,
    atomic_scf,
    build_matrices,
    build_matrices_multi,
    crystal_scf,
    log_mesh,
    numerov_log,
    solve_geneig,
)

# Elk 11.0.2 all-electron FLAPW, simple-cubic Ne a=6 Bohr, LSDA (PW92), rgkmax 7:
# 2s -1.263267 Ha, 2p -0.426409 Ha at Γ  ->  2s-2p splitting 22.77 eV.
ELK_NE_SPLIT_EV = 22.77


def _free_electron(kfrac, L, ecut, nbands):
    b = 2 * np.pi / L
    nmax = int(np.ceil(np.sqrt(ecut / HBAR2_2M) / b)) + 1
    e = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = b * (np.array([i, j, m]) + np.asarray(kfrac))
                if HBAR2_2M * (kg @ kg) <= ecut:
                    e.append(HBAR2_2M * (kg @ kg))
    return np.sort(e)[:nbands]


def stage_atom():
    """Atomic Ne LDA SCF vs NIST."""
    r, dx = log_mesh(1e-5, 28.0, 2500)
    eigs, _ = atomic_scf("Ne", r, dx)
    print("  Ne atomic KS eigenvalues (eV)      mine        NIST-LDA     |Δ|")
    rows = []
    for lab in ("2s", "2p"):
        mine, ref = eigs[lab], NIST_LDA_EV["Ne"][lab]
        rows.append((lab, mine, ref))
        print(f"    {lab:>3}  {mine:14.3f}  {ref:12.3f}  {abs(mine - ref):8.3f}")
    return rows


def stage_empty_lattice():
    """Empty-lattice correctness gate: V=0 -> free-electron bands."""
    L, R, lmax, ecut = 6.0, 1.0, 6, 60.0
    r, dx = log_mesh(1e-4, R + 1.0, 1200)
    v = torch.zeros_like(r)
    El = {lg: 2.5 for lg in range(lmax + 1)}
    H, S, _ = build_matrices([0.1, 0.0, 0.0], L, R, lmax, El, ecut, r, dx, v)
    ev = solve_geneig(0.5 * (H + H.T), S, 6)
    ref = _free_electron([0.1, 0.0, 0.0], L, ecut, 6)
    err = float(np.abs(ev - ref).max())
    print(f"  LAPW bands vs ½|k+G|² (V=0), max|Δ| = {err:.2e} eV")
    return ev, ref


def _lapw_neon_split(a_bohr, R=1.4, ecut=300.0):
    """Single-shot LAPW 2s-2p splitting at Γ for simple-cubic Ne with the atomic potential."""
    L = a_bohr * BOHR_ANG
    r, dx = log_mesh(1e-5, 28.0, 2500)
    at, v_at = atomic_scf("Ne", r, dx)
    v0 = float(v_at.numpy()[np.argmin(np.abs(r.numpy() - R))])
    v_mt = torch.where(r <= R, v_at - v0, torch.zeros_like(r))
    el = {0: at["2s"] - v0, 1: at["2p"] - v0, 2: -5.0 - v0}
    species = {"Ne": {"R": R, "v": v_mt, "El": el}}
    H, S, _ = build_matrices_multi([0.0, 0.0, 0.0], L, [([0.0, 0.0, 0.0], "Ne")],
                                   2, ecut, r, dx, species)
    ev = solve_geneig(H, S, 6)
    return float(ev[1:4].mean() - ev[0])       # 2p (3-fold) minus 2s


def stage_crystal():
    """Crystal FLAPW, compared via the zero-independent 2s-2p splitting.

    The muffin-tin interstitial zero wanders, so absolute eigenvalues are not comparable; the
    2s-2p splitting is the physical, reference-grade quantity — that is what Elk comparisons use.
    """
    r, dx = log_mesh(1e-5, 28.0, 2500)
    at, _ = atomic_scf("Ne", r, dx)
    atom_split = at["2p"] - at["2s"]
    print(f"    atomic Ne 2s-2p splitting            {atom_split:6.2f} eV")

    # (a) frozen atomic potential, single LAPW solve — deterministic; the validated Elk cross-check.
    frozen_split = _lapw_neon_split(6.0, R=1.4)
    print(f"    a=6 Bohr crystal (frozen potential)  {frozen_split:6.2f} eV   "
          f"vs Elk {ELK_NE_SPLIT_EV:.2f}  (|Δ| {abs(frozen_split - ELK_NE_SPLIT_EV):.2f})")

    # (b) self-consistent crystal — self-consistency shifts the splitting toward Elk.
    cryst, _ = crystal_scf(6.0, "Ne", R=1.4, ecut=200.0, iters=40)
    scf_split = cryst["2p"] - cryst["2s"]
    print(f"    a=6 Bohr crystal (self-consistent)   {scf_split:6.2f} eV   "
          f"vs Elk {ELK_NE_SPLIT_EV:.2f}  (|Δ| {abs(scf_split - ELK_NE_SPLIT_EV):.2f})")

    # (c) dilute limit — the self-consistent crystal splitting must recover the isolated atom.
    dilute, atomic = crystal_scf(10.0, "Ne", R=2.0, ecut=120.0, iters=25)
    dilute_split = dilute["2p"] - dilute["2s"]
    print(f"    a=10 Bohr (dilute) recovers atom     {dilute_split:6.2f} eV   "
          f"vs atom {atom_split:.2f}  (|Δ| {abs(dilute_split - atom_split):.2f})")

    return atom_split, frozen_split, scf_split, dilute_split


def stage_autograd():
    """The 'Auto' in AutoAPW: autograd through the radial match, vs finite difference."""
    r, dx = log_mesh(1e-4, 3.0, 1500)
    v = -8.0 / r  # a bare hydrogenic well (eV·Å units via E2 folded into r later); shape only
    R_MT = 1.4
    iR = int(np.argmin(np.abs(r.numpy() - R_MT)))

    def u_at_R(E):
        u = numerov_log(0, E, r, dx, v)
        u = u / torch.sqrt((u[r <= R_MT] ** 2 * r[r <= R_MT] * dx).sum())
        return u[iR]

    E = torch.tensor(-6.0, dtype=torch.float64, requires_grad=True)
    u = u_at_R(E)
    (grad_ad,) = torch.autograd.grad(u, E)
    h = 1e-4
    with torch.no_grad():
        grad_fd = (u_at_R(E + h) - u_at_R(E - h)) / (2 * h)
    print(f"  ∂u_l(R_MT)/∂E_l :  autograd {float(grad_ad):+.5f}   "
          f"finite-diff {float(grad_fd):+.5f}   |Δ| {abs(float(grad_ad) - float(grad_fd)):.1e}")
    return float(grad_ad), float(grad_fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1] Atomic all-electron LDA SCF")
    stage_atom()
    print("\n[2] Empty-lattice LAPW correctness gate")
    ev, ref = stage_empty_lattice()
    print("\n[3] Crystal FLAPW — 2s-2p splitting (simple-cubic Ne)")
    atom_split, frozen_split, scf_split, dilute_split = stage_crystal()
    print("\n[4] Differentiable augmented basis (autograd)")
    stage_autograd()

    if args.no_plot:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axb, axc) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: empty-lattice LAPW bands vs free electron.
    axb.plot(range(len(ref)), ref, "o", ms=9, mfc="none", mec="k", label="½|k+G|² (exact)")
    axb.plot(range(len(ev)), ev, "x", ms=8, color="tab:red", label="LAPW (this code)")
    axb.set_xlabel("band index")
    axb.set_ylabel("energy (eV)")
    axb.set_title(f"Empty-lattice gate\nmax|Δ| = {float(np.abs(ev - ref).max()):.1e} eV")
    axb.legend(fontsize=8)

    # Panel B: Ne 2s-2p splitting (zero-independent) — atom, dilute, a=6 frozen, a=6 SCF, Elk.
    labels = ["atom", "a=10\ndilute", "a=6\nfrozen", "a=6\nself-cons.", "Elk 11\n(a=6)"]
    splits = [atom_split, dilute_split, frozen_split, scf_split, ELK_NE_SPLIT_EV]
    colors = ["0.6", "tab:blue", "tab:orange", "tab:green", "k"]
    axc.bar(labels, splits, color=colors)
    axc.set_ylabel("2s–2p splitting (eV)")
    axc.set_ylim(min(splits) - 0.3, max(splits) + 0.3)
    axc.axhline(ELK_NE_SPLIT_EV, color="k", ls=":", lw=0.8)
    axc.set_title("Ne all-electron FLAPW\n2s–2p splitting vs Elk 11.0.2")
    for i, s in enumerate(splits):
        axc.text(i, s + 0.03, f"{s:.2f}", ha="center", fontsize=8)

    fig.suptitle("AutoAPW — differentiable all-electron FLAPW in PyTorch", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = outdir / "flapw_neon_crystal.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
