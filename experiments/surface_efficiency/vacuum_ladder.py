#!/usr/bin/env python
"""CO/Pt(111) vacuum-ladder A/B: plain-periodic vs open-boundary (ESM) SCF.

Backlog item ``surface_vacuum_ladder`` (docs/h100_backlog.md #1). This is the
committed, parameterized driver the H100 runner executes; the earlier throwaway
version lived on asus at /tmp/surface_eff.

Single-point (rigid geometry, no relaxation) SCFs on a Pt(111) 2x2xN slab with a
CO (or H) adsorbate, driven through ``gradwave.api.run_scf`` with the ESM slab
boundary ``inp.scf.boundary="open_z"`` A/B'd against plain 3D-periodic. For each
(system, vacuum, boundary) it records converged E_total, adsorbate forces, the
work function on both faces, npw, FFT-grid size and wall time, written
incrementally to a results JSON so partial progress survives a crash. One SCF at
a time, hard iteration cap.

The decisive gate is the **atop - fcc site-preference difference** and its
plateau vs vacuum: the banked finding (memory/surface-slab-efficiency-stack.md)
is that ESM ``open_z`` gives asymmetric faces and converges the ladder by
~6 A/face, while plain-periodic washes the adsorbate dipole and fails to
converge the thin-vacuum slab. This driver reproduces that A/B at scale.

Compute note: real Pt-slab SCFs OOM the RTX 3050 and take ~10-90 min each on
asus CPU -- this is an H100 datacenter run. Locally, only ``--precheck`` (a
no-SCF npw / FFT-grid pre-check) is meant to be run.

Usage:
  # free, no-SCF geometry pre-check (npw + grid ratio thin vs thick vacuum):
  uv run python experiments/surface_efficiency/vacuum_ladder.py --precheck \
      --out precheck.json

  # full ladder (H100): plain vs ESM at each vacuum, atop + fcc, + clean slab:
  uv run python experiments/surface_efficiency/vacuum_ladder.py \
      --system Pt --adsorbate CO --vacuums 5 6 8 15 --kmesh 4 4 1 \
      --device cuda --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

RY_EV = 13.605693122994

# --------------------------------------------------------------------------- #
# Geometry (ASE fcc111 + add_adsorbate; exact per-face vacuum via center()).
# c stays perpendicular to a,b (fcc111 orthogonal=False), the ESM slab
# constraint. These helpers are import-safe (no torch / no SCF) so the unit
# test can exercise them without heavy deps.
# --------------------------------------------------------------------------- #
A0 = {"Pt": 3.97, "Au": 4.17}  # PBE fcc lattice constants [A]
CO_D = 1.15      # C-O distance [A]
CO_HEIGHT = 1.95  # C height above the site [A]
H_HEIGHT = 1.0   # H height above the site [A]


def _co_molecule():
    from ase import Atoms
    return Atoms("CO", positions=[[0, 0, 0], [0, 0, CO_D]])


def clean_slab(vac_face, element="Pt", a0=None, layers=3):
    from ase.build import fcc111
    a0 = a0 if a0 is not None else A0[element]
    slab = fcc111(element, size=(2, 2, layers), a=a0, vacuum=None, orthogonal=False)
    slab.center(vacuum=vac_face, axis=2)
    return slab


def slab_with_adsorbate(vac_face, site, element="Pt", adsorbate="CO", a0=None, layers=3):
    from ase.build import add_adsorbate, fcc111
    a0 = a0 if a0 is not None else A0[element]
    slab = fcc111(element, size=(2, 2, layers), a=a0, vacuum=None, orthogonal=False)
    if adsorbate == "CO":
        add_adsorbate(slab, _co_molecule(), height=CO_HEIGHT, position=site, mol_index=0)
        n_ads = 2
    elif adsorbate == "H":
        add_adsorbate(slab, "H", height=H_HEIGHT, position=site)
        n_ads = 1
    else:
        raise ValueError(f"unknown adsorbate {adsorbate!r}")
    slab.center(vacuum=vac_face, axis=2)
    ads_idx = list(range(len(slab) - n_ads, len(slab)))
    return slab, ads_idx


def gas_molecule(adsorbate="CO", box=12.0):
    from ase import Atoms
    if adsorbate == "CO":
        mol = _co_molecule()
    elif adsorbate == "H":
        mol = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    else:
        raise ValueError(f"unknown adsorbate {adsorbate!r}")
    mol.set_cell([box, box, box])
    mol.set_pbc(True)
    mol.center()
    return mol


def actual_vacuum_per_face(atoms):
    z = atoms.get_positions()[:, 2]
    c = float(atoms.get_cell()[2, 2])
    return 0.5 * (c - (z.max() - z.min()))


def cell_c(atoms):
    return float(atoms.get_cell()[2, 2])


def check_orthogonal(atoms):
    """max |cos angle| between c and {a,b}; ~0 means c perpendicular to plane."""
    cell = np.asarray(atoms.get_cell())
    c = cell[2]
    return float(max(abs(np.dot(c, cell[a])) / (np.linalg.norm(c) * np.linalg.norm(cell[a]))
                     for a in (0, 1)))


# --------------------------------------------------------------------------- #
# Pseudopotential resolution: search a list of dirs (env + repo defaults) so the
# split delta_gauge (Pt/Au) + qe-fixture (C/O/H) pseudos both resolve.
# --------------------------------------------------------------------------- #
_DEFAULT_PMAP = {
    "Pt": "Pt.upf", "Au": "Au.upf",
    "C": "C_ONCV_PBE-1.2.upf", "O": "O_ONCV_PBE-1.2.upf", "H": "H_ONCV_PBE-1.2.upf",
}


def _repo_root():
    # experiments/surface_efficiency/vacuum_ladder.py -> repo root is two up
    return Path(__file__).resolve().parents[2]


def _pseudo_search_dirs(extra):
    dirs = []
    if extra:
        dirs.append(Path(extra))
    env = os.environ.get("GW_PSEUDO_PATH", "")
    dirs += [Path(p) for p in env.split(":") if p]
    root = _repo_root()
    dirs += [root / "benchmarks/delta_gauge/pseudos", root / "tests/fixtures/qe/pseudos"]
    return dirs


def resolve_pseudos(symbols, pmap, search_dirs):
    """Map each element symbol to an absolute UPF path found in search_dirs."""
    resolved = {}
    for sym in sorted(set(symbols)):
        fname = pmap.get(sym)
        if fname is None:
            raise KeyError(f"no pseudo mapping for element {sym!r}")
        for d in search_dirs:
            cand = Path(d) / fname
            if cand.is_file():
                resolved[sym] = str(cand)
                break
        else:
            raise FileNotFoundError(f"pseudo {fname!r} for {sym} not in {search_dirs}")
    return resolved


# --------------------------------------------------------------------------- #
# no-SCF pre-check (free): npw + FFT-grid-point ratio thin vs thick vacuum.
# --------------------------------------------------------------------------- #
def precheck(cfg):
    from gradwave.grids import build_fft_grid, build_gsphere
    ecut = cfg.ecut_ry * RY_EV
    ecutrho = cfg.ecutrho_ry * RY_EV

    def grid_stats(atoms, tag):
        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        grid = build_fft_grid(cell, ecutrho)
        sph = build_gsphere(grid, ecut, (0.0, 0.0, 0.0))
        return dict(tag=tag, vac_face=round(actual_vacuum_per_face(atoms), 3),
                    c=round(cell_c(atoms), 3), fft_shape=list(grid.shape),
                    n_grid_points=int(grid.n_points), npw_gamma=int(sph.npw),
                    ortho_cos=round(check_orthogonal(atoms), 12))

    out = {"kind": "precheck", "ecut_eV": ecut, "ecutrho_eV": ecutrho,
           "system": cfg.system, "adsorbate": cfg.adsorbate, "systems": []}
    thick_vac, thin_vac = max(cfg.vacuums), min(cfg.vacuums)
    builders = [("clean_slab", lambda v: clean_slab(v, cfg.system))]
    builders.append((f"{cfg.adsorbate}_ontop",
                     lambda v: slab_with_adsorbate(v, "ontop", cfg.system, cfg.adsorbate)[0]))
    for name, build in builders:
        st = grid_stats(build(thick_vac), f"{name}_thick{thick_vac:g}")
        sn = grid_stats(build(thin_vac), f"{name}_thin{thin_vac:g}")
        entry = {"thick": st, "thin": sn,
                 "npw_drop_pct": round(100 * (1 - sn["npw_gamma"] / st["npw_gamma"]), 1),
                 "ngrid_drop_pct": round(100 * (1 - sn["n_grid_points"] / st["n_grid_points"]), 1)}
        out["systems"].append(entry)
        print(f"{name}: thick c={st['c']} npw={st['npw_gamma']} ngrid={st['n_grid_points']} "
              f"-> thin npw={sn['npw_gamma']} ngrid={sn['n_grid_points']}  "
              f"npw drop {entry['npw_drop_pct']}%  "
              f"ngrid drop {entry['ngrid_drop_pct']}%", flush=True)
    Path(cfg.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {cfg.out}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# One SCF (the H100 path).
# --------------------------------------------------------------------------- #
def _make_input(atoms, kmesh, boundary, cfg, pseudos):
    from gradwave.inputs import Input
    from gradwave.inputs.models import KPointsParams, SCFParams, SmearingParams
    return Input(
        atoms=atoms,
        pseudo_dir=Path("/"),  # ignored: pseudo_map holds absolute paths
        pseudo_map=pseudos,
        ecut=cfg.ecut_ry * RY_EV, ecutrho=cfg.ecutrho_ry * RY_EV, xc="pbe",
        kpoints=KPointsParams(mesh=tuple(kmesh)),
        smearing=SmearingParams(type="gaussian", width=cfg.width),
        scf=SCFParams(boundary=boundary, max_iter=cfg.max_iter,
                      etol=cfg.etol, rhotol=cfg.rhotol),
        nspin=1, symmetry=True, device=cfg.device, verbose=False,
        output_checkpoint=False,
    )


def run_one(atoms, kmesh, boundary, cfg, pseudos, adsorbate_idx=None, save_ckpt=None):
    import torch

    from gradwave.api._common import XC_REGISTRY
    from gradwave.api.scf import run_scf
    from gradwave.api.system import build_system
    from gradwave.postscf.forces import forces
    from gradwave.postscf.work_function import work_function

    inp = _make_input(atoms, kmesh, boundary, cfg, pseudos)
    system = build_system(inp)
    npw = int(max(s.npw for s in system.spheres))
    n_grid = int(system.grid.n_points)
    if cfg.device != "cpu":
        system = system.to(cfg.device)
    xc = XC_REGISTRY["pbe"]()
    t0 = time.time()
    res = run_scf(inp, system=system, verbose=False)
    wall = time.time() - t0

    fmax, f_ads = None, None
    try:
        f = forces(res, xc=xc).detach().cpu().numpy()
        fmax = float(np.linalg.norm(f, axis=1).max())
        if adsorbate_idx:
            f_ads = float(np.linalg.norm(f[adsorbate_idx], axis=1).max())
    except Exception as exc:
        print(f"    [forces failed] {exc}", flush=True)

    wf = None
    try:
        wf = [float(x) for x in work_function(res, open_axis=2, both_faces=True)]
    except Exception as exc:
        print(f"    [workfn failed] {exc}", flush=True)

    if save_ckpt is not None:
        try:
            from gradwave.io.checkpoint import save_checkpoint
            save_checkpoint(res, save_ckpt)
            print(f"    [ckpt] {save_ckpt}", flush=True)
        except Exception as exc:
            print(f"    [ckpt failed] {exc}", flush=True)

    del torch  # (imported for side effects / thread config upstream)
    return dict(
        boundary=boundary, converged=bool(res.converged), n_iter=int(res.n_iter),
        e_total=float(res.energies.total), e_free=float(res.energies.free_energy),
        fermi=float(res.fermi), npw=npw, n_grid=n_grid, fmax=fmax,
        f_adsorbate_max=f_ads, work_function_faces=wf, wall_s=round(wall, 1),
        n_atoms=len(atoms),
    )


def run_ladder(cfg):
    import torch

    from gradwave.api.system import build_system  # noqa: F401  (import sanity)

    if cfg.device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    out_path = Path(cfg.out)
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {"runs": {}}
    runs = results.setdefault("runs", {})
    results["meta"] = dict(
        kind="vacuum_ladder", system=cfg.system, adsorbate=cfg.adsorbate,
        ecut_eV=cfg.ecut_ry * RY_EV, ecutrho_eV=cfg.ecutrho_ry * RY_EV,
        width=cfg.width, kmesh=list(cfg.kmesh), max_iter=cfg.max_iter,
        etol=cfg.etol, rhotol=cfg.rhotol, device=cfg.device,
        vacuums=list(cfg.vacuums), boundaries=list(cfg.boundaries), sites=list(cfg.sites),
    )

    # collect the element set up front to resolve pseudos once
    symbols = {cfg.system} | ({"C", "O"} if cfg.adsorbate == "CO" else {"H"})
    pseudos = resolve_pseudos(symbols, dict(_DEFAULT_PMAP, **cfg.pseudo_map),
                              _pseudo_search_dirs(cfg.pseudo_dir))
    results["meta"]["pseudos"] = pseudos

    def do(key, atoms, kmesh, boundary, adsorbate_idx=None, save_ckpt=None):
        if key in runs and runs[key].get("e_total") is not None:
            print(f"[skip] {key} (E={runs[key]['e_total']:.4f})", flush=True)
            return
        print(f"[run ] {key}  vac={actual_vacuum_per_face(atoms):.2f} "
              f"c={cell_c(atoms):.2f} natoms={len(atoms)}", flush=True)
        try:
            r = run_one(atoms, kmesh, boundary, cfg, pseudos, adsorbate_idx, save_ckpt)
            r["vac_face"] = round(actual_vacuum_per_face(atoms), 3)
            runs[key] = r
            print(f"[done] {key}  E={r['e_total']:.4f} conv={r['converged']} "
                  f"it={r['n_iter']} npw={r['npw']} wall={r['wall_s']}s "
                  f"wf={r['work_function_faces']} f_ads={r['f_adsorbate_max']}", flush=True)
        except Exception as exc:
            runs[key] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"[FAIL] {key}: {exc}", flush=True)
        out_path.write_text(json.dumps(results, indent=2))

    slab_k = tuple(cfg.kmesh)
    gas_k = (1, 1, 1)

    def _ck(tag, vac, b):
        # checkpoint the clean slab + adsorbate at the plateau vacuum, ESM only
        # (shared with the tt_rank_slab probe #4).
        if abs(vac - cfg.ckpt_vac) < 1e-6 and b == "open_z":
            return str(ckpt_dir / f"{cfg.system}_{cfg.adsorbate}_{tag}_v{vac:g}.pt")
        return None

    # 0) gas reference (both boundaries) for E_ads
    for b in cfg.boundaries:
        do(f"{cfg.adsorbate}_gas__{b}", gas_molecule(cfg.adsorbate), gas_k, b)

    # 1) DECISIVE first: clean slab + atop at every vacuum, both boundaries.
    for vac in cfg.vacuums:
        for b in cfg.boundaries:
            do(f"slab__v{vac:g}__{b}", clean_slab(vac, cfg.system), slab_k, b,
               save_ckpt=_ck("slab", vac, b))
            for site in cfg.sites:
                atoms, ads_idx = slab_with_adsorbate(vac, site, cfg.system, cfg.adsorbate)
                do(f"slab_{cfg.adsorbate}_{site}__v{vac:g}__{b}", atoms, slab_k, b,
                   ads_idx, _ck(f"{cfg.adsorbate}_{site}", vac, b))

    print("CAMPAIGN_DONE", flush=True)
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", default="Pt", choices=["Pt", "Au"])
    p.add_argument("--adsorbate", default="CO", choices=["CO", "H"])
    p.add_argument("--vacuums", type=float, nargs="+", default=[5.0, 6.0, 8.0, 15.0],
                   help="vacuum per face [A]")
    p.add_argument("--kmesh", type=int, nargs=3, default=[4, 4, 1])
    p.add_argument("--sites", nargs="+", default=["ontop", "fcc"])
    p.add_argument("--boundaries", nargs="+", default=["periodic", "open_z"],
                   choices=["periodic", "open_z", "open_z_metal"])
    p.add_argument("--ecut-ry", type=float, default=40.0)
    p.add_argument("--ecutrho-ry", type=float, default=160.0)
    p.add_argument("--width", type=float, default=0.20, help="smearing width [eV]")
    p.add_argument("--max-iter", type=int, default=80, help="hard SCF iteration cap")
    p.add_argument("--etol", type=float, default=1e-6)
    p.add_argument("--rhotol", type=float, default=1e-5)
    p.add_argument("--ckpt-vac", type=float, default=6.0,
                   help="vacuum at which to save a density checkpoint (ESM) for tt_rank")
    p.add_argument("--device", default=os.environ.get("GW_DEVICE", "cpu"))
    p.add_argument("--pseudo-dir", default=os.environ.get("GW_PSEUDO_DIR"),
                   help="extra pseudo search dir (searched before repo defaults)")
    p.add_argument("--pseudo-map", default="{}",
                   help="JSON overriding element->filename pseudo names")
    p.add_argument("--ckpt-dir", default="ckpt")
    p.add_argument("--out", default="vacuum_ladder_results.json")
    p.add_argument("--precheck", action="store_true",
                   help="no-SCF npw / FFT-grid pre-check only (free, laptop-safe)")
    args = p.parse_args(argv)
    args.pseudo_map = json.loads(args.pseudo_map)
    return args


def main(argv=None):
    cfg = _parse_args(argv)
    if cfg.precheck:
        precheck(cfg)
    else:
        run_ladder(cfg)


if __name__ == "__main__":
    main()
