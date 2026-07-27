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
    """Diamond-Si cell (2 primitive, or 16/64-atom supercell), one snapshot
    randomly displaced."""
    from ase.build import bulk

    if size == 2:
        atoms = bulk("Si", "diamond", a=5.43)          # 2-atom primitive
    else:
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


def mem_probe(size, ecut, kmesh, device, seed):
    """Peak memory for one gradient + one Hessian-vector product on the joint
    energy, checkpointing on vs off — the concrete deliverable-3 result
    (checkpointing is what lets the 64-atom Hvp fit a 6 GB card)."""
    import torch as _t

    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.dtypes import CDTYPE, RDTYPE
    from gradwave.opt.joint import (
        _coeffs_from_z,
        joint_energy,
        lowdin,
        teter_precond,
    )
    from gradwave.pseudo.radial_torch import RadialTables
    from gradwave.scf.loop import scf, setup_system

    upf = _pseudo()
    cell, pos0, _, natom = _build_si(size, 0.0, seed)
    species = [0] * natom
    km = (kmesh, kmesh, kmesh)
    system = setup_system(cell=cell, positions=pos0, species_of_atom=species,
                          upfs=[upf], ecut=ecut, kmesh=km,
                          use_symmetry=False).to(device)
    xc = LDA_PW92()
    n_occ = int(round(system.n_electrons / 2))
    res = scf(system, xc, max_iter=3, verbose=False, etol=0.0, rhotol=0.0,
              diago_tol=1e-4)
    coeffs0 = [lowdin(c[:n_occ].to(CDTYPE)) for c in res.coeffs]
    precond = [teter_precond(sph.kpg2, 1.0).to(RDTYPE) for sph in system.spheres]
    npws = [sph.npw for sph in system.spheres]
    tabs = [RadialTables(u, device=device) for u in system.upfs]
    occ = _t.full((len(system.spheres), n_occ), 2.0, dtype=RDTYPE, device=device)
    a0 = np.asarray(system.grid.cell)
    print(f"# Si-{natom} mem probe  ecut={ecut/RY:.0f} Ry  kmesh={km}  "
          f"n_occ={n_occ}  npw~{npws[0]}", flush=True)

    for ckpt in (True, False):
        eps = _t.zeros(3, 3, dtype=RDTYPE, device=device)
        frac = _t.tensor(pos0 @ np.linalg.inv(a0), dtype=RDTYPE, device=device,
                         requires_grad=True)
        zs = [_t.view_as_real((coeffs0[ik] / precond[ik][None, :].to(CDTYPE))
              .contiguous()).clone().requires_grad_(True)
              for ik in range(len(system.spheres))]
        leaves = [frac, *zs]
        if device == "cuda":
            _t.cuda.empty_cache()
            _t.cuda.reset_peak_memory_stats()
        try:
            e = joint_energy(system, xc, tabs, eps, frac,
                             _coeffs_from_z(zs, precond, npws), occ,
                             checkpoint=ckpt)
            g = _t.autograd.grad(e, leaves, create_graph=True)
            v = [_t.randn_like(t) for t in leaves]
            _ = _t.autograd.grad(sum((gi * vi).sum() for gi, vi
                                     in zip(g, v, strict=True)), leaves)
            peak = (_t.cuda.max_memory_allocated() / 1e9
                    if device == "cuda" else float("nan"))
            print(f"  checkpoint={ckpt!s:5s}  Hvp OK   peak={peak:.2f} GB",
                  flush=True)
        except RuntimeError as exc:
            print(f"  checkpoint={ckpt!s:5s}  FAILED   {str(exc)[:80]}",
                  flush=True)


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
    ap.add_argument("--size", type=int, default=16, choices=[2, 16, 64])
    ap.add_argument("--ecut", type=float, default=12.0, help="Ry")
    ap.add_argument("--kmesh", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", default="newton,joint,nested")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--disp", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem-probe", action="store_true",
                    help="measure peak grad+Hvp memory (checkpoint on/off) and exit")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.mem_probe:
        mem_probe(a.size, a.ecut * RY, a.kmesh, a.device, a.seed)
    else:
        run(a.size, a.ecut * RY, a.kmesh, a.device, a.methods.split(","),
            a.fmax, a.disp, a.seed)
