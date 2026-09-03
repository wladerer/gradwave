"""Where does the symmetry-path time actually go? Decompose Path B at fixed N.

Baseline eigh is O(N³) LAPACK. The symmetry path's *algorithmic* cost is Σ eigh(nλ)
= O(Σnλ³) ≈ N³/R. Everything else (building the adapted basis, assembling U†HU) is
overhead — and in this prototype it's pure Python. This isolates the three parts so
we can see whether the flop win (small block-eighs) is real and merely buried under
Python overhead, or absent.
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
from bench_walltime import median_time, spacegroup, symmetrize  # noqa: E402
from blockdiag_matfree import adapted_basis  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def run(name, yaml, se):
    inp = load_input(str(HERE / yaml))
    system = build_system(inp)
    res = run_scf(inp, system=system, verbose=False)
    v_eff = res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]
    sg = spacegroup(system)
    grid = system.grid
    sph = build_gsphere(grid, se, np.zeros(3))
    npw = sph.npw
    miller = sph.miller.cpu().numpy()
    ops = little_group(np.zeros(3), sg, grid.cell)
    Ws = [np.asarray(op["W"], int) for op in ops]
    beta_ls, dij_sp = species_projector_tables(system.upfs, None)
    pd = projector_data_at_k(sph, system.species_of_atom, system.upfs,
                             beta_ls, dij_sp, grid.volume, None)
    p = projectors(pd, system.positions)
    h = HamiltonianK(sph, grid.shape, v_eff, pd, p)
    H = h.apply(torch.eye(npw, dtype=torch.complex128)).transpose(0, 1).numpy()
    H = 0.5 * (H + H.conj().T)
    H = symmetrize(H, miller, ops, np.zeros(3))

    U, sizes, _ = adapted_basis(miller, ops, Ws, np.zeros(3))
    R = npw ** 3 / sum(s ** 3 for s in sizes)

    # precompute blocks (assembly) so we can time eigh alone
    blocks, s = [], 0
    for r in sizes:
        Ul = U[:, s:s + r]
        Hl = Ul.conj().T @ (H @ Ul)
        blocks.append(0.5 * (Hl + Hl.conj().T))
        s += r

    tA, _ = median_time(lambda: np.linalg.eigvalsh(H))
    t_basis, _ = median_time(lambda: adapted_basis(miller, ops, Ws, np.zeros(3)))
    t_asm, _ = median_time(lambda: [U[:, a:a + r].conj().T @ (H @ U[:, a:a + r])
                                    for a, r in _slices(sizes)])
    t_eigh, _ = median_time(lambda: [np.linalg.eigvalsh(b) for b in blocks])

    print(f"\n{name}: N={npw}, {len(sizes)} blocks, maxblk={max(sizes)}, "
          f"1/R={1/R:.3f} (flop win {R:.0f}×)")
    print(f"  baseline  eigvalsh(H)          : {tA*1e3:8.1f} ms")
    print(f"  ── symmetry path parts ──")
    print(f"  (algorithmic) Σ eigh(blocks)   : {t_eigh*1e3:8.1f} ms   "
          f"→ {tA/t_eigh:5.1f}× vs baseline  [the real flop win]")
    print(f"  (overhead) block assembly U†HU : {t_asm*1e3:8.1f} ms")
    print(f"  (overhead) adapted_basis build : {t_basis*1e3:8.1f} ms   [pure Python]")
    print(f"  eigh-only would win {tA/t_eigh:.0f}×; overhead currently swamps it "
          f"({(t_basis+t_asm)/t_eigh:.0f}× the eigh cost).")


def _slices(sizes):
    a = 0
    for r in sizes:
        yield a, r
        a += r


if __name__ == "__main__":
    run("diamond", "diamond_bench.yaml", 700)
    run("bcc Fe", "fe_bench.yaml", 900)
