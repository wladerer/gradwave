"""Track 7 — collinear-spin (nspin=2) matrix-free symmetry block-diagonalization.

Ferromagnetic bcc Fe (its real ground state). For a collinear FM the uniform
magnetization leaves the full spatial point group intact, so BOTH spin channels
H↑(k), H↓(k) are block-diagonalized by the SAME (spatial) symmetry-adapted basis.
The projectors are spin-independent, so the two Hamiltonians share pd/p and differ
only through res.v_eff[isp].

Run:  uv run python experiments/symbolic/track7_spin.py
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
from blockdiag_matfree import adapted_basis, solve_blocks_matfree  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def main() -> None:
    print("Track 7: collinear-spin (FM bcc Fe) matrix-free block-diagonalization\n")
    inp = load_input(str(HERE / "fe_fm_scf.yaml"))
    system = build_system(inp)
    t0 = time.perf_counter()
    res = run_scf(inp, system=system, verbose=False)
    print(f"[SCF nspin=2 in {time.perf_counter() - t0:.0f}s]  v_eff shape {tuple(res.v_eff.shape)}")
    try:
        occ = np.asarray(res.occupations)          # (2, nk, nb)
        wk = np.asarray(res.kweights) if hasattr(res, "kweights") else None
        if wk is not None:
            m = (occ[0] * wk[:, None]).sum() - (occ[1] * wk[:, None]).sum()
            print(f"[magnetic moment ≈ {m:.2f} μB/cell]")
    except Exception:  # noqa: BLE001
        pass

    grid = system.grid
    frac = system.positions.cpu().numpy() @ np.linalg.inv(grid.cell)
    sg = find_spacegroup(grid.cell, frac, system.species_of_atom)

    for kname, kfrac in [("Γ", (0, 0, 0)), ("H", (0.5, -0.5, 0.5))]:
        kf = np.asarray(kfrac, float)
        sph = build_gsphere(grid, system.ecut, kf)
        npw = sph.npw
        miller = sph.miller.cpu().numpy()
        ops = little_group(kf, sg, grid.cell)
        Ws = [np.asarray(op["W"], int) for op in ops]

        beta_ls, dij_sp = species_projector_tables(system.upfs, None)
        pd = projector_data_at_k(sph, system.species_of_atom, system.upfs,
                                 beta_ls, dij_sp, grid.volume, None)
        p = projectors(pd, system.positions)

        U, sizes, diag = adapted_basis(miller, ops, Ws, kf)
        red = npw ** 3 / sum(s ** 3 for s in sizes)

        evs = {}
        for isp, name in ((0, "↑"), (1, "↓")):
            h = HamiltonianK(sph, grid.shape, res.v_eff[isp], pd, p)  # shared pd, p
            ev, _, _ = solve_blocks_matfree(h.apply, U, sizes)
            evs[name] = ev

        print(f"\n{kname}: |G_k|={len(ops)}, npw={npw}, "
              f"{len(sizes)} blocks (max {max(sizes)}), reduce={red:.1f}×")
        lo = min(len(evs["↑"]), len(evs["↓"]), 8)
        split = evs["↑"][:lo] - evs["↓"][:lo]
        print(f"  both spins block-diagonalized by the SAME spatial basis")
        print(f"  lowest ↑ (eV): {np.round(evs['↑'][:lo], 3)}")
        print(f"  lowest ↓ (eV): {np.round(evs['↓'][:lo], 3)}")
        print(f"  exchange splitting ↑−↓ (eV): {np.round(split, 3)}")


if __name__ == "__main__":
    main()
