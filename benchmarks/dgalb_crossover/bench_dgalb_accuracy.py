"""DG-ALB accuracy — the two-scale (element <-> crystal) representability test.

The crossover study proved DG-ALB is *fast*; this asks whether it is *correct*:
can ~M adaptive local basis functions built from an ISOLATED extended element
represent the true CRYSTAL occupied orbitals in the element's core region to
sub-meV, and how big a buffer does that take?

Method (sidesteps the grid-commensurability problem — no shared FFT grid):
  1. CRYSTAL reference: real SCF on an N x N x N conventional-cell Si supercell
     -> converged occupied orbitals {psi_n} (plane-wave coeffs at Gamma).
  2. Pick the crystal's central conventional cell as the CORE region; sample
     K = core_pts^3 real-space points r_k in it (Cartesian).
  3. Evaluate psi_n(r_k) by explicit Fourier sum  sum_G c_{n,G} e^{iG.r_k}
     -> Psi (K x N_occ).  <- the "truth" in the core.
  4. For each buffer size k (element = central k x k x k conv cells, an ISOLATED
     periodic cell): real SCF on that element, diagonalise its converged H for
     M_max states -> {phi_j}. Evaluate phi_j on the SAME core points r_k (mapped
     into the element frame) -> Phi (K x M_max).  <- the ALB span.
  5. Residual of the occupied subspace outside span{phi_1..phi_M}, per M:
        res_n = 1 - ||Q^H psi_n||^2 / ||psi_n||^2,   Q = orthonormal(Phi[:, :M]).
     Report mean/max over occupied n vs M (ALBs) and k (buffer).

Correctness self-check: k = N (element == crystal) must give res -> 0 once
M >= N_occ (the crystal's own eigenstates then span its occupied orbitals). A
nonzero floor there flags a frame/evaluation bug.

REAL throughout: two genuine gradwave SCF solves + exact Fourier evaluation +
linear algebra. No placeholder potential. The one modelling choice is the core
sampling density (core_pts); residuals are reported as fractions (and an eV
proxy = fraction x occupied bandwidth).

Self-contained (installed gradwave symbols). Heavy SCFs -> run on asus.

Run:  uv run python benchmarks/dgalb_crossover/bench_dgalb_accuracy.py \
          --crystal 3 --buffers 1,2,3 --ecut-ry 12 --m-list 8,16,32,64,96 --threads 8
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.batch import BatchedHamiltonian, build_batched, projectors_b
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.davidson import davidson_batched

RY = 13.605693122994
CDTYPE = torch.complex128
RDTYPE = torch.float64
A_SI = 5.43
_OCC_TOL = 1e-5

_BASE = np.array(
    [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ]
)


def si_cells(n, a=A_SI):
    """Isolated n x n x n conventional-cell Si box (atoms at standard sites)."""
    frac = np.vstack(
        [(_BASE + [i, j, k]) / [n, n, n]
         for i in range(n) for j in range(n) for k in range(n)]
    )
    cell = a * np.diag([n, n, n]).astype(float)
    return cell, frac @ cell, [0] * (8 * n**3)


def _crystal_ref(system, xc, n_iter=25):
    """Real SCF; use its converged OCCUPIED orbitals directly as the reference
    (no re-diagonalisation — the SCF already spans them). Returns
    (kpg_cart (npw,3), occ_coeffs (N_occ, npw), N_occ, res)."""
    res = scf(system, xc, smearing="none", max_iter=n_iter, etol=1e-6, rhotol=1e-5,
              verbose=False)
    sph = system.spheres[0]
    occ = int((res.occupations[0] > _OCC_TOL).sum().item())
    return sph.kpg.to(RDTYPE), res.coeffs[0][:occ], occ, res


def _element_alb(system, xc, m_states, n_iter=25):
    """Real SCF, then diagonalise the converged H for m_states eigenvectors
    (occupied + empty) — the ALB candidate pool. Returns
    (kpg_cart (npw,3), coeffs (m_states, npw))."""
    res = scf(system, xc, smearing="none", max_iter=n_iter, etol=1e-6, rhotol=1e-5,
              verbose=False)
    sph = system.spheres[0]
    pd = system.proj_data[0]
    bk = build_batched([sph], [pd])
    p = projectors_b(bk, system.positions)
    ham = BatchedHamiltonian(bk, system.grid.shape, res.v_eff, p)
    npw = sph.npw
    m = min(m_states, npw)
    torch.manual_seed(0)
    x0 = torch.randn(1, m, npw, dtype=CDTYPE)
    t = (HBAR2_2M * sph.kpg2).unsqueeze(0)
    mask = torch.ones(1, npw, dtype=torch.bool)
    dav = davidson_batched(ham.apply, x0, t, mask, tol=1e-6, max_iter=40)
    return sph.kpg.to(RDTYPE), dav.eigenvectors[0]


def _eval_on_points(kpg_cart, coeffs, r_pts):
    """psi_j(r_k) = sum_G c_{j,G} e^{i G.r_k}. coeffs (nb, npw), r_pts (K,3)
    Cartesian in the SAME frame as kpg_cart. Returns (K, nb) complex."""
    phase = torch.exp(1j * (r_pts @ kpg_cart.T))     # (K, npw)
    return phase @ coeffs.T                           # (K, nb)


def _core_points(crystal_n, core_pts):
    """Uniform sample points inside the crystal's central conventional cell,
    Cartesian in the crystal frame. Returns (K,3) and the cell corner offset."""
    c0 = (crystal_n - 1) // 2                          # central conv-cell index (odd N)
    lo = c0 * A_SI
    xs = (np.arange(core_pts) + 0.5) / core_pts * A_SI + lo
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return torch.tensor(pts, dtype=RDTYPE), c0


def _residuals(Psi, Phi_M):
    """Fraction of each reference column outside span(Phi_M). Psi (K,Nocc),
    Phi_M (K,M). Returns per-column residual array (Nocc,)."""
    Q, _ = torch.linalg.qr(Phi_M, mode="reduced")     # (K, M) orthonormal cols
    proj = Q.conj().T @ Psi                            # (M, Nocc)
    num = (proj.abs() ** 2).sum(dim=0)                 # ||Q^H psi_n||^2
    den = (Psi.abs() ** 2).sum(dim=0) + 1e-30
    return (1.0 - (num / den).real).clamp(min=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crystal", type=int, default=3, help="crystal conv cells/side (odd)")
    ap.add_argument("--buffers", default="1,2,3", help="element sizes k (conv cells/side), k<=crystal")
    ap.add_argument("--ecut-ry", type=float, default=12.0)
    ap.add_argument("--m-list", default="8,16,32,64,96", help="ALB counts M to test")
    ap.add_argument("--core-pts", type=int, default=12, help="core sample points per axis")
    ap.add_argument("--m-max", type=int, default=128, help="element eigenstates to build")
    ap.add_argument("--upf", default="tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    upf = parse_upf(args.upf)
    xc = LDA_PW92()
    ecut = args.ecut_ry * RY
    m_list = [int(x) for x in args.m_list.split(",")]
    buffers = [int(x) for x in args.buffers.split(",")]

    print(f"# crystal={args.crystal}^3 conv cells ({8*args.crystal**3} atoms)  "
          f"ecut={args.ecut_ry} Ry  core_pts={args.core_pts}^3  threads="
          f"{torch.get_num_threads()}  (fp64)", flush=True)

    # ---- crystal reference ----
    cell, pos, spc = si_cells(args.crystal)
    csys = setup_system(cell=cell, positions=pos, species_of_atom=spc, upfs=[upf],
                        ecut=ecut, kmesh=(1, 1, 1), use_symmetry=False)
    kpg_c, coeffs_occ, n_occ, cres = _crystal_ref(csys, xc)
    r_pts, c0 = _core_points(args.crystal, args.core_pts)
    Psi = _eval_on_points(kpg_c, coeffs_occ, r_pts)    # (K, N_occ)
    occ_ev = cres.eigenvalues[0][:n_occ]
    bandwidth = float((occ_ev.max() - occ_ev.min()).item())
    print(f"# crystal: N_occ={n_occ}  occ bandwidth={bandwidth:.2f} eV  "
          f"K={r_pts.shape[0]} core points", flush=True)
    print(f"{'buffer_k':>8} {'elem_atoms':>10} {'M':>4} {'M/atom':>7} "
          f"{'mean_res':>10} {'max_res':>10} {'~E_err_meV':>11}", flush=True)

    for k in buffers:
        # element = central k x k x k conv-cell block, isolated periodic cell.
        # Its corner sits at (crystal-k)//2 conv cells; core cell offset within.
        ecell, epos, espc = si_cells(k)
        esys = setup_system(cell=ecell, positions=epos, species_of_atom=espc, upfs=[upf],
                            ecut=ecut, kmesh=(1, 1, 1), use_symmetry=False)
        kpg_e, coeffs_e = _element_alb(esys, xc, args.m_max)
        # map crystal core points into the element frame: element corner is the
        # central-block corner in the crystal, at ((crystal-k)//2)*a per axis.
        corner = ((args.crystal - k) // 2) * A_SI
        r_e = r_pts - corner
        Phi = _eval_on_points(kpg_e, coeffs_e, r_e)    # (K, M_max)
        elem_atoms = 8 * k**3
        for M in m_list:
            if M > Phi.shape[1]:
                continue
            res = _residuals(Psi, Phi[:, :M])
            mean_r = float(res.mean().item())
            max_r = float(res.max().item())
            e_err = mean_r * bandwidth * 1000.0        # crude eV->meV proxy
            print(f"{k:>8} {elem_atoms:>10} {M:>4} {M/elem_atoms:>7.2f} "
                  f"{mean_r:>10.2e} {max_r:>10.2e} {e_err:>11.2f}", flush=True)


if __name__ == "__main__":
    main()
