"""M2-hardening: generality (GAP 1) + amortization crossover (GAP 2).

One harness, three jobs, all driven through ``api.run_relax`` (the production
nested BFGS+SCF engine), feature OFF (pulay + local_tf) vs ON (same +
chi0_precond, build-once/reuse behind the ρ(M) auto-abstain gate). Both arms
follow the byte-identical BFGS trajectory (a preconditioner cannot move the
fixed point), so this is an apples-to-apples cost measurement.

GAP 1 — generality. Does the 4-layer 1.46× wall win hold on a DIFFERENT, harder
surface? ``--system`` selects:
  * ``al_slab``  (default): Al(100), ``--layers`` thick, ``--nx/--ny`` lateral
    supercell — a larger cell tests whether the win grows with system size.
  * ``h_al``: an H adatom (ontop, relaxes to hollow) on an Al(100) slab — the
    asymmetric-adsorbate inhomogeneity that is the real catalysis target.

GAP 2 — amortization crossover. Per ionic step we snapshot the process FFT tally
(``opcount``; cumulative and never reset, so a snapshot/since delta around each
``_calculate_nc`` captures that step's SCF *plus* — on ON's first step — the
one-time gate+build). We emit the cumulative-FFT and wall curves so the reader
can see WHERE the total-FFT ratio crosses from ~1.0× (build not amortized) up
toward the running-FFT ratio as the trajectory lengthens. Give BFGS ≥20 steps
(harder rattle / more steps) and read the curve.

Usage (asus only):
    uv run python experiments/chi0_precond_crux/m2_hardening.py \
        --system al_slab --layers 6 --steps 25 --rattle 0.10 \
        --out .../results/gap2_crossover.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import add_adsorbate, fcc100

from gradwave.core import opcount
from gradwave.inputs import load_input

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
PSEUDO = _ROOT / "tests" / "fixtures" / "qe" / "pseudos"

_PSEUDO_MAP = {
    "Al": "Al_ONCV_PBE-1.2.upf",
    "H": "H_ONCV_PBE-1.2.upf",
    "C": "C_ONCV_PBE-1.2.upf",
    "O": "O_ONCV_PBE-1.2.upf",
}


def build_atoms(args):
    """The test structure for ``--system``."""
    slab = fcc100("Al", size=(args.nx, args.ny, args.layers), a=4.05,
                  vacuum=args.vacuum)
    if args.system == "h_al":
        # single H adatom, ontop of a surface Al (asymmetric: one face only);
        # ontop is a saddle so BFGS drives a real multi-step relaxation toward
        # the hollow site — exactly the inhomogeneous adsorbate regime.
        add_adsorbate(slab, "H", height=1.5, position="ontop")
    slab.rattle(stdev=args.rattle, seed=args.seed)
    return slab


def _pseudo_map(atoms) -> dict[str, str]:
    return {s: _PSEUDO_MAP[s] for s in sorted(set(atoms.get_chemical_symbols()))}


def _write_input(tmp: Path, atoms, *, kmesh, ecut_ry, steps, fmax, chi0, nbands):
    cell = atoms.cell.array.tolist()
    pos = atoms.get_positions().tolist()
    species = atoms.get_chemical_symbols()
    pmap = _pseudo_map(atoms)
    body = f"""
structure:
  cell: {cell}
  positions: {{cart: {pos}}}
  species: {species}
symmetry: false
nbands: {nbands}
pseudopotentials:
  dir: {PSEUDO}
  map: {json.dumps(pmap)}
ecut: {ecut_ry * RY}
kpoints: {{mesh: [{kmesh}, {kmesh}, 1]}}
smearing: {{type: gaussian, width: 0.1}}
scf:
  max_iter: 150
  etol: 1.0e-9
  rhotol: 1.0e-8
  mixing: {{scheme: pulay, alpha: 0.7, precond: local_tf, chi0_precond: {str(chi0).lower()}}}
task: relax
relax: {{optimizer: bfgs, fmax: {fmax}, max_steps: {steps}, method: nested,
        extrapolation: reuse}}
"""
    p = tmp / f"in_{'on' if chi0 else 'off'}.yaml"
    p.write_text(body)
    return load_input(p)


def _install_perstep_probe():
    """Monkeypatch the NC calculate to record per-ionic-step FFT + SCF iters.

    ``opcount`` counts are process-cumulative and never reset; ``scf()`` toggles
    counting on for its run and off on return, and ``Chi0PrecondCache.update``
    (the one-time gate+build) does likewise inside the SAME call. So a
    snapshot/since delta around ``_calculate_nc`` captures exactly that ionic
    step's total FFT — including, on ON's first converged step, the gate+build."""
    from gradwave import calculator

    if getattr(calculator.GradWave, "_perstep_probe_installed", False):
        return
    orig = calculator.GradWave._calculate_nc

    def wrapped(self):
        prev = opcount.snapshot()
        orig(self)
        d = int(opcount.since(prev)["fft"])
        self._fft_steps = [*getattr(self, "_fft_steps", []), d]
        self._iter_steps = [*getattr(self, "_iter_steps", []),
                            int(self.last_result.n_iter)]

    wrapped._perstep_probe_installed = True  # type: ignore[attr-defined]
    calculator.GradWave._calculate_nc = wrapped
    calculator.GradWave._perstep_probe_installed = True


def _run(tag, inp):
    from gradwave.api import run_relax

    opcount.enable()
    prev = opcount.snapshot()
    t0 = time.time()
    relax, atoms, _frames = run_relax(inp, verbose=True)
    dt = time.time() - t0
    ffts = int(opcount.since(prev)["fft"])
    cache = getattr(atoms.calc, "_chi0_cache", None)
    fft_steps = list(getattr(atoms.calc, "_fft_steps", []))
    iter_steps = list(getattr(atoms.calc, "_iter_steps", []))
    row = {
        "tag": tag,
        "converged": bool(relax["converged"]),
        "n_steps": int(relax["n_steps"]),
        "energy_eV": float(relax["energy_eV"]),
        "fmax_eV_ang": float(relax["fmax_eV_ang"]),
        "scf_total_iter": int(relax.get("scf_total_iter", sum(iter_steps))),
        "scf_iter_per_step": iter_steps,
        "fft_per_step": fft_steps,
        "total_fft": ffts,
        "wall_s": round(dt, 1),
        "positions_ang": relax["positions_ang"],
        "chi0": None if cache is None else {
            "engaged": bool(getattr(cache, "engaged", False)),
            "rho": cache.rho,
            "n_col": (None if getattr(cache, "precond", None) is None
                      else int(cache.precond.n_col)),
            "gate_ffts": int(getattr(cache, "gate_ffts", 0)),
            "build_ffts": int(getattr(cache, "build_ffts", 0)),
            "precond_calls": (None if getattr(cache, "precond", None) is None
                              else int(cache.precond.n_calls)),
        },
    }
    print(f"\n>>> {tag}: {relax['n_steps']} steps  conv={relax['converged']}  "
          f"E={relax['energy_eV']:.8f} eV  scf_iters={row['scf_total_iter']}  "
          f"FFT={ffts}  {dt:.0f}s\n", flush=True)
    return row


def _crossover(off, on):
    """Cumulative-FFT and wall ratios as a function of ionic-step count."""
    fo, fn = off["fft_per_step"], on["fft_per_step"]
    io, ino = off["scf_iter_per_step"], on["scf_iter_per_step"]
    k = min(len(fo), len(fn))
    curve = []
    co = cn = cio = cino = 0
    for i in range(k):
        co += fo[i]
        cn += fn[i]
        cio += io[i]
        cino += ino[i]
        curve.append({
            "step": i + 1,
            "cum_fft_off": co, "cum_fft_on": cn,
            "fft_ratio_off_over_on": round(co / max(1, cn), 3),
            "cum_iter_off": cio, "cum_iter_on": cino,
            "iter_ratio_off_over_on": round(cio / max(1, cino), 3),
        })
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["al_slab", "h_al"], default="al_slab")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nx", type=int, default=1)
    ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--vacuum", type=float, default=8.0)
    ap.add_argument("--kmesh", type=int, default=4)
    ap.add_argument("--ecut", type=float, default=25.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--rattle", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    _install_perstep_probe()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent

    atoms0 = build_atoms(args)
    natoms = len(atoms0)
    nbands = 6 * natoms
    meta = {"args": vars(args), "natoms": natoms, "nbands": nbands,
            "formula": atoms0.get_chemical_formula()}
    print(f"=== {args.system} {atoms0.get_chemical_formula()} "
          f"({natoms} atoms, {args.nx}x{args.ny}x{args.layers}), "
          f"kmesh={args.kmesh}, rattle={args.rattle} Å, steps≤{args.steps} ===",
          flush=True)

    rows = []
    for chi0 in (False, True):
        inp = _write_input(tmp, atoms0.copy(), kmesh=args.kmesh,
                           ecut_ry=args.ecut, steps=args.steps, fmax=args.fmax,
                           chi0=chi0, nbands=nbands)
        rows.append(_run("ON" if chi0 else "OFF", inp))
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    off, on = rows[0], rows[1]
    dE = abs(on["energy_eV"] - off["energy_eV"])
    dpos = float(np.abs(np.array(on["positions_ang"])
                        - np.array(off["positions_ang"])).max())
    curve = _crossover(off, on)
    verdict = {
        "dE_eV": dE,
        "d_pos_ang": dpos,
        "same_answer": bool(dE < 1e-6 and dpos < 1e-4),
        "fft_off": off["total_fft"], "fft_on": on["total_fft"],
        "fft_ratio_off_over_on": round(off["total_fft"]
                                       / max(1, on["total_fft"]), 3),
        "wall_ratio_off_over_on": round(off["wall_s"]
                                        / max(1e-9, on["wall_s"]), 3),
        "scf_iters_off": off["scf_total_iter"],
        "scf_iters_on": on["scf_total_iter"],
        "iter_ratio_off_over_on": round(off["scf_total_iter"]
                                        / max(1, on["scf_total_iter"]), 3),
        "chi0_engaged": on["chi0"]["engaged"] if on["chi0"] else False,
        "chi0_rho": on["chi0"]["rho"] if on["chi0"] else None,
        "crossover_curve": curve,
    }
    meta["verdict"] = verdict
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    print("\n===== M2-HARDENING VERDICT =====")
    print(f"  system: {meta['formula']} ({natoms} atoms)  steps={off['n_steps']}")
    print(f"  faithfulness: dE={dE:.2e} eV  d|pos|={dpos:.2e} Å  "
          f"SAME={verdict['same_answer']}")
    print(f"  scf iters: OFF={off['scf_total_iter']}  ON={on['scf_total_iter']}  "
          f"→ {verdict['iter_ratio_off_over_on']}×")
    print(f"  wall: OFF={off['wall_s']}s  ON={on['wall_s']}s  "
          f"→ {verdict['wall_ratio_off_over_on']}×")
    print(f"  total FFT: OFF={off['total_fft']}  ON={on['total_fft']}  "
          f"→ {verdict['fft_ratio_off_over_on']}×")
    if on["chi0"]:
        c = on["chi0"]
        print(f"  chi0: engaged={c['engaged']} ρ={c['rho']} n_col={c['n_col']} "
              f"gate_ffts={c['gate_ffts']} build_ffts={c['build_ffts']} "
              f"precond_calls={c['precond_calls']}")
    if curve:
        print("  crossover (cum FFT ratio OFF/ON by step): "
              + " ".join(f"{c['step']}:{c['fft_ratio_off_over_on']}"
                         for c in curve))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
