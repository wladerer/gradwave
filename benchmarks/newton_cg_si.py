"""Go/no-go: exact-Hvp Newton-CG vs first-order joint vs nested BFGS+SCF (Si).

Perturbed-supercell relaxation head-to-head on the metric that matters for the
exact-Hvp thesis (ideas.md step 5): total Hamiltonian applications and wall time
to the same force gate. Positions-only (fixed cell) so all three optimize the
identical objective and the perturbation is a clean atomic displacement.

Usage:
  uv run python benchmarks/newton_cg_si.py [--size 16|64] [--ecut RY]
     [--kmesh N] [--device cpu|cuda] [--methods newton,joint,nested]
     [--fmax 0.02] [--disp 0.12] [--seed 0]

Prints one table with, per method: total H-applies (in the joint band·k unit),
wall seconds, converged?, final fmax, and final energy (fresh SCF at the relaxed
geometry) so correctness and cost are both auditable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994


def _build_si(size: int, disp: float, seed: int):
    """Diamond-Si supercell (16 or 64 atoms), one snapshot randomly displaced."""
    from ase.build import bulk

    n = {16: (2, 1, 1), 64: (2, 2, 2)}[size]
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * n
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    pos = atoms.get_positions().astype(np.float64)
    rng = np.random.default_rng(seed)
    pos_pert = pos + disp * rng.standard_normal(pos.shape)
    return cell, pos, pos_pert, len(atoms)


def _pseudo():
    from gradwave.pseudo.upf import parse_upf
    root = Path(__file__).parents[1]
    return parse_upf(root / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")


def _scf_energy_fmax(cell, pos, species, upf, ecut, kmesh, device):
    """Fresh converged SCF energy [eV] and Hellmann-Feynman fmax [eV/Å]."""
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.postscf.forces import forces as hf_forces
    from gradwave.scf.loop import scf, setup_system

    system = setup_system(cell=cell, positions=pos, species_of_atom=species,
                          upfs=[upf], ecut=ecut, kmesh=kmesh,
                          use_symmetry=False).to(device)
    res = scf(system, LDA_PW92(), verbose=False)
    f = float(np.abs(hf_forces(res, xc=LDA_PW92()).cpu().numpy()).max())
    return float(res.energies.free_energy), f


def run(size, ecut, kmesh, device, methods, fmax, disp, seed):
    from gradwave.calculator import GradWave
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.opt.joint import count_h_applies, joint_relax
    from gradwave.opt.newton import newton_cg_relax

    upf = _pseudo()
    pseudo_path = str(Path(__file__).parents[1]
                      / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    cell, pos0, posp, natom = _build_si(size, disp, seed)
    species = [0] * natom
    km = (kmesh, kmesh, kmesh)
    rows = []

    print(f"# Si-{natom}  ecut={ecut/RY:.0f} Ry  kmesh={km}  device={device}  "
          f"fmax={fmax}  disp={disp} Å  seed={seed}", flush=True)

    if "newton" in methods:
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        t0 = time.time()
        res = newton_cg_relax(cell, posp, species, [upf], LDA_PW92(),
                              ecut=ecut, kmesh=km, fmax=fmax, fix_cell=True,
                              max_newton=60, device=device, verbose=True)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        e, f = _scf_energy_fmax(cell, res.positions, species, upf, ecut, km, device)
        rows.append(("newton-cg", res.h_equiv, wall, res.converged, f, e,
                     f"grad {res.n_grad} hvp {res.n_hvp} trial {res.n_trial} "
                     f"seed {res.h_seed}"))

    if "joint" in methods:
        t0 = time.time()
        res = joint_relax(cell, posp, species, [upf], LDA_PW92(),
                          ecut=ecut, kmesh=km, fmax=fmax, fix_cell=True,
                          max_closures=4000, device=device, verbose=True)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        e, f = _scf_energy_fmax(cell, res.positions, species, upf, ecut, km, device)
        rows.append(("joint-lbfgs", res.h_equiv, wall, res.converged, f, e,
                     f"closures {res.n_closures} seed {res.h_seed}"))

    if "nested" in methods:
        from ase import Atoms
        from ase.optimize import BFGS

        atoms = Atoms(f"Si{natom}", positions=posp.copy(), cell=cell.copy(),
                      pbc=True)
        atoms.calc = GradWave(ecut=ecut, pseudopotentials={"Si": pseudo_path},
                              xc="lda", kpts=km, use_symmetry=False,
                              device=device)
        t0 = time.time()
        with count_h_applies() as ctr:
            opt = BFGS(atoms, logfile=None)
            conv = opt.run(fmax=fmax, steps=120)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        e, f = _scf_energy_fmax(cell, atoms.get_positions(), species, upf,
                                ecut, km, device)
        rows.append(("nested-bfgs", ctr.count, wall, bool(conv), f, e,
                     f"steps {opt.nsteps}"))

    w = "%-13s %14s %10s %6s %10s %16s  %s"
    print(w % ("method", "H-applies", "wall_s", "conv", "fmax", "E_eV", "detail"))
    for name, h, wall, conv, f, e, detail in rows:
        print(w % (name, f"{h:,}", f"{wall:.1f}", conv, f"{f:.4f}",
                   f"{e:.5f}", detail))
    if device == "cuda":
        print(f"# peak CUDA mem: "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16, choices=[16, 64])
    ap.add_argument("--ecut", type=float, default=12.0, help="Ry")
    ap.add_argument("--kmesh", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", default="newton,joint,nested")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--disp", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run(a.size, a.ecut * RY, a.kmesh, a.device, a.methods.split(","),
        a.fmax, a.disp, a.seed)
