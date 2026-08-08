"""DG-ALB crossover study — does the adaptive-local-basis reformulation buy
wall-time (not just memory), and above what atom count?

Compares, per SCF diagonalization step, the work DGDFT *replaces* against the
work it *introduces*, on Si supercells swept over atom count:

  t_PW   REAL. One `davidson_batched` solve for nb bands on the full global
         plane-wave sphere (fixed iteration count → potential-independent,
         deterministic timing). This is the object DG-ALB shrinks.

  t_DG = t_alb + t_glob
    t_alb  REAL. Build all adaptive local basis functions: for each of
           N_elem extended elements, a `davidson_batched` for M_elem states
           on a small extended-element sub-box. Elements are identical for a
           homogeneous Si supercell, so they batch (nk = N_elem) — the exact
           parallelism DGDFT exploits. Run in element-batch chunks to bound
           memory. This is the recurring per-SCF ALB regeneration cost.
    t_glob MODEL (dense upper bound). One dense `eigh` of the reduced global
           matrix, dimension D = M_elem x N_elem. The true DG global operator
           is BLOCK-SPARSE (nearest-neighbour element coupling only), so a
           real DG global solve is *faster* than this dense number — t_glob is
           a conservative ceiling, not the achievable cost.

What is REAL vs MODELLED (read before trusting a number):
  * t_PW and t_alb are genuine gradwave `davidson_batched` runs at the right
    npw / band counts and precision (complex128, fp64 — matches the SCF).
  * Extended elements are modelled as identical isolated Si sub-cells (buffer
    as a real Si halo via `--ext` conventional cells around a `--core`-cell
    core). Homogeneous Si → identical is exact and is DGDFT's best batching
    case; a heterogeneous cell would batch less perfectly.
  * t_glob's dimension D and sparsity are physical; its matrix entries are
    synthetic (a random Hermitian) — fine for a *timing* ceiling.
  * DG surface/flux (interior-penalty) assembly cost is NOT included; it is an
    O(N_elem) sparse-assembly term, small next to the solves, and there is no
    gradwave machinery to measure it yet.
  * v_eff is a fixed smooth placeholder and davidson runs a FIXED iteration
    count (tol=0): this is a timing study, not a converged SCF. Cost centres
    (FFT / GEMM / Rayleigh-Ritz) are set by npw, nb, M, D — not by potential
    values — so the crossover it locates is meaningful.

The plane-wave baseline is EXPECTED to OOM at large cells (the global sphere +
projector table blow past RAM — the very wall DG-ALB removes); those rows are
caught and reported as OOM while the DG columns keep going. That divergence is
itself the headline.

Self-contained: installed gradwave symbols only, so it runs against the
canonical checkout on asus.

Run:  uv run python benchmarks/dgalb_crossover/bench_dgalb.py --sizes 8,64,216
      uv run python benchmarks/dgalb_crossover/bench_dgalb.py --sizes 8,64,216,512 \
            --ecut-ry 20 --m-per-atom 8 --core 1 --ext 2 --threads 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from gradwave.core.batch import BatchedHamiltonian, build_batched, projectors_b
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.solvers.davidson import davidson_batched

RY = 13.605693122994
CDTYPE = torch.complex128
RDTYPE = torch.float64

# 8-atom conventional Si cell (fractional), tiled to build cores/supercells.
_BASE = np.array(
    [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ]
)


def si_cells(nx: int, ny: int, nz: int, a: float = 5.43):
    """Si supercell of nx*ny*nz conventional cells → 8*nx*ny*nz atoms."""
    frac = np.vstack(
        [(_BASE + [i, j, k]) / [nx, ny, nz]
         for i in range(nx) for j in range(ny) for k in range(nz)]
    )
    cell = a * np.diag([nx, ny, nz]).astype(float)
    return cell, frac @ cell, [0] * (8 * nx * ny * nz)


def _cubeish(cells: int):
    """Factor `cells` into (nx,ny,nz) as close to cubic as possible."""
    best = (cells, 1, 1)
    for nx in range(1, cells + 1):
        if cells % nx:
            continue
        rem = cells // nx
        for ny in range(1, rem + 1):
            if rem % ny:
                continue
            nz = rem // ny
            if max(nx, ny, nz) - min(nx, ny, nz) < max(best) - min(best):
                best = (nx, ny, nz)
    return best


def _smooth_veff(shape, device):
    """Deterministic smooth real field on the grid (timing placeholder)."""
    axes = [torch.linspace(0, 2 * np.pi, n, dtype=RDTYPE, device=device) for n in shape]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    return (-4.0 * (torch.cos(gx) + torch.cos(gy) + torch.cos(gz))).to(RDTYPE)


def _build_system(nx, ny, nz, ecut_ry, upf):
    cell, pos, spc = si_cells(nx, ny, nz)
    system = setup_system(
        cell=cell, positions=pos, species_of_atom=spc, upfs=[upf],
        ecut=ecut_ry * RY, kmesh=(1, 1, 1), use_symmetry=False,
    )
    return system


def _time_davidson(bk, shape, v_eff, p, nb, n_iter, seed=0):
    """Fixed-iteration davidson_batched over nk = bk.mask.shape[0]. Returns
    (seconds, npw_max, peak_note). Raises on OOM (caller catches)."""
    ham = BatchedHamiltonian(bk, shape, v_eff, p)
    nk, m = bk.mask.shape
    torch.manual_seed(seed)
    x0 = torch.randn(nk, nb, m, dtype=CDTYPE) * bk.mask[:, None, :]
    # warmup (allocs, caches) then timed run
    davidson_batched(ham.apply, x0.clone(), bk.t, bk.mask, tol=0.0, max_iter=2)
    t0 = time.perf_counter()
    davidson_batched(ham.apply, x0.clone(), bk.t, bk.mask, tol=0.0, max_iter=n_iter)
    return time.perf_counter() - t0


def _tile_batch(sphere, pd, positions, reps, ecut_ry):
    """A BatchedK of `reps` identical elements + phased projectors (nk=reps)."""
    bk = build_batched([sphere] * reps, [pd] * reps)
    p = projectors_b(bk, positions)                      # (reps, nproj, npw)
    return bk, p


def _peak_gb(nk, nb, npw, n_grid, nproj):
    """Rough peak-RAM estimate (GB) of a batched davidson run, complex128.
    Covers the projector table (+ its cached conjugate), the CPU box temporary
    (unchunked), and the growing subspace (~max_dim=4*nb)."""
    return nk * 16.0 * (2 * nproj * npw + 3 * nb * n_grid + 13 * nb * npw) / 1e9


def run_size(total_atoms_req, ecut_ry, m_per_atom, core, ext, n_iter, upf, elem_batch,
             mem_budget_gb):
    # ---- element geometry -------------------------------------------------
    # core = conv cells per side of the DG element; ext = conv cells per side
    # of the extended element (core + buffer halo). Atoms per core element:
    core_atoms = 8 * core**3
    # number of elements to tile the requested total (round to a cube of cores)
    n_elem_req = max(1, round(total_atoms_req / core_atoms))
    ex, ey, ez = _cubeish(n_elem_req)
    n_elem = ex * ey * ez
    total_atoms = n_elem * core_atoms
    m_elem = int(m_per_atom * core_atoms)

    upf_data = parse_upf(upf)

    # ---- extended-element solve (ALB build unit) --------------------------
    esys = _build_system(ext, ext, ext, ecut_ry, upf_data)
    e_sphere = esys.spheres[0]
    e_pd = esys.proj_data[0]
    e_shape = esys.grid.shape
    e_pos = esys.positions
    e_veff = _smooth_veff(e_shape, e_pos.device)
    npw_e = e_sphere.npw
    n_grid_e = int(np.prod(e_shape))
    nproj_e = e_pd.f_ylm_phase_free.shape[0]

    # time ALB build in element-batch chunks; shrink the batch to fit the
    # memory budget so a big extended element never trips the OS OOM-killer.
    fit_batch = elem_batch
    while fit_batch > 1 and _peak_gb(fit_batch, m_elem, npw_e, n_grid_e, nproj_e) > mem_budget_gb:
        fit_batch -= 1
    t_alb = 0.0
    done = 0
    while done < n_elem:
        b = min(fit_batch, n_elem - done)
        bk_b, p_b = _tile_batch(e_sphere, e_pd, e_pos, b, ecut_ry)
        t_alb += _time_davidson(bk_b, e_shape, e_veff, p_b, m_elem, n_iter)
        done += b

    # ---- reduced global solve (dense upper bound) -------------------------
    D = m_elem * n_elem
    t_glob = float("nan")
    glob_note = ""
    if 3 * D * D * 16.0 / 1e9 > mem_budget_gb:      # A + eigvalsh workspace
        glob_note = "OOM(est)"
    else:
        try:
            torch.manual_seed(1)
            A = torch.randn(D, D, dtype=CDTYPE)
            A = A + A.conj().T
            t0 = time.perf_counter()
            torch.linalg.eigvalsh(A)      # eigenvalues suffice for a timing ceiling
            t_glob = time.perf_counter() - t0
            del A
        except RuntimeError as exc:
            glob_note = "OOM" if "alloc" in str(exc).lower() or "memory" in str(exc).lower() else "ERR"

    t_dg = t_alb + (0.0 if np.isnan(t_glob) else t_glob)

    # ---- plane-wave baseline (the object DG replaces) ---------------------
    # setup_system is cheap (sphere/grid metadata only); estimate the davidson
    # peak from it and skip the run gracefully if it would exceed the budget —
    # the OS OOM-killer is uncatchable, so we must not trigger it.
    npw_pw = nb_pw = float("nan")
    t_pw = float("nan")
    pw_note = ""
    try:
        nx, ny, nz = _cubeish(n_elem * core**3)      # same atom count, cubic-ish
        psys = _build_system(nx, ny, nz, ecut_ry, upf_data)
        p_sphere = psys.spheres[0]
        npw_pw = p_sphere.npw
        nb_pw = int(psys.nbands)
        nproj_pw = psys.proj_data[0].f_ylm_phase_free.shape[0]
        n_grid_pw = int(np.prod(psys.grid.shape))
        if _peak_gb(1, nb_pw, npw_pw, n_grid_pw, nproj_pw) > mem_budget_gb:
            pw_note = "OOM(est)"
            del psys
        else:
            bk_pw = build_batched([p_sphere], [psys.proj_data[0]])
            p_pw = projectors_b(bk_pw, psys.positions)
            pw_veff = _smooth_veff(psys.grid.shape, psys.positions.device)
            t_pw = _time_davidson(bk_pw, psys.grid.shape, pw_veff, p_pw, nb_pw, n_iter)
            del psys, bk_pw, p_pw
    except RuntimeError as exc:
        pw_note = "OOM" if "alloc" in str(exc).lower() or "memory" in str(exc).lower() else "ERR"

    speedup = (t_pw / t_dg) if (not np.isnan(t_pw) and t_dg > 0) else float("nan")

    return {
        "atoms": total_atoms, "n_elem": n_elem, "m_elem": m_elem,
        "npw_pw": npw_pw, "nb_pw": nb_pw, "npw_e": npw_e, "D": D,
        "dim_red": (npw_pw / D) if not np.isnan(npw_pw) else float("nan"),
        "t_pw": t_pw, "pw_note": pw_note,
        "t_alb": t_alb, "t_glob": t_glob, "glob_note": glob_note, "t_dg": t_dg,
        "speedup": speedup,
    }


def _fmt(x, spec):
    return ("{:" + spec + "}").format(x) if isinstance(x, (int,)) or not np.isnan(x) else "     —  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="8,64,216", help="target total atom counts")
    ap.add_argument("--ecut-ry", type=float, default=20.0)
    ap.add_argument("--m-per-atom", type=float, default=8.0, help="ALBs kept per core atom")
    ap.add_argument("--core", type=int, default=1, help="conv cells/side of a DG element")
    ap.add_argument("--ext", type=int, default=2, help="conv cells/side of the extended element")
    ap.add_argument("--n-iter", type=int, default=12, help="fixed davidson iterations (timing)")
    ap.add_argument("--elem-batch", type=int, default=8, help="elements solved per batch (memory)")
    ap.add_argument("--mem-budget-gb", type=float, default=9.0,
                    help="skip runs whose estimated peak exceeds this (avoids OS OOM-kill)")
    ap.add_argument("--upf", default="tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-header", action="store_true",
                    help="skip the banner + column header (per-size subprocess driver)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if not args.no_header:
        print(f"# torch threads={torch.get_num_threads()}  ecut={args.ecut_ry} Ry  "
              f"M/atom={args.m_per_atom}  core={args.core}c ext={args.ext}c  "
              f"n_iter={args.n_iter}  (fp64)", flush=True)
        print("# t_pw = full plane-wave davidson (REAL); t_dg = ALB build (REAL) + dense "
              "global eigh (MODEL, dense upper bound)", flush=True)
        hdr = ("atoms", "n_elem", "npw_pw", "nb_pw", "npw_e", "M_el", "D", "npw/D",
               "t_pw_s", "t_alb_s", "t_glob_s", "t_dg_s", "PW/DG")
        print(("{:>6} {:>6} {:>8} {:>6} {:>7} {:>5} {:>7} {:>7} "
               "{:>9} {:>9} {:>9} {:>9} {:>7}").format(*hdr), flush=True)
    for tok in args.sizes.split(","):
        r = run_size(int(tok), args.ecut_ry, args.m_per_atom, args.core, args.ext,
                     args.n_iter, args.upf, args.elem_batch, args.mem_budget_gb)
        pw = r["pw_note"] or f"{r['t_pw']:.2f}"
        gl = r["glob_note"] or f"{r['t_glob']:.2f}"
        sp = "  —  " if np.isnan(r["speedup"]) else f"{r['speedup']:.2f}x"
        dr = "  —  " if np.isnan(r["dim_red"]) else f"{r['dim_red']:.0f}x"
        npw_pw = "OOM" if np.isnan(r["npw_pw"]) else f"{int(r['npw_pw'])}"
        nb_pw = "—" if np.isnan(r["nb_pw"]) else f"{int(r['nb_pw'])}"
        print(("{:>6} {:>6} {:>8} {:>6} {:>7} {:>5} {:>7} {:>7} "
               "{:>9} {:>9.2f} {:>9} {:>9.2f} {:>7}").format(
            r["atoms"], r["n_elem"], npw_pw, nb_pw, r["npw_e"], r["m_elem"],
            r["D"], dr, pw, r["t_alb"], gl, r["t_dg"], sp), flush=True)


if __name__ == "__main__":
    main()
