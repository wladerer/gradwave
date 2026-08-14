"""NC inexact-Newton SCF finisher — self-contained benchmark (prototype).

Compares the shipped linear density mixer against the exact-Jacobian Newton
finisher (scf.newton_nc, wired into scf/loop.py via `newton_finish=`) on
norm-conserving metals, and isolates the two inner-solve levers (Kerker
preconditioning; Eisenstat–Walker adaptive forcing) by reporting the
inner-apply count per Newton step for each variant.

The Newton step solves (I − χ₀ K_Hxc) δ = r on the SCF density residual using
the existing response matvecs (scf.implicit.{apply_chi0, apply_k_hxc}) plus a
preconditioned Anderson inner solve. It reaches the SAME fixed point as the
mixer (exact at convergence) — the ΔE column is the correctness gate (< 1e-8 eV).

Systems (default set; --systems to subset):
  Al-fcc      easy nearly-free-electron metal (nspin=1)         — mixer wins fast
  Fe-bcc-nm   nonmagnetic bcc Fe, stiffer d-band metal (nspin=1)
  Fe-bcc-fm   FERROMAGNETIC bcc Fe (nspin=2, start_mag) — the STIFF case where
              the default mixer needs many iterations and the Newton iteration
              cut is large enough to matter

Variants per system: mixer, newton-plain (no lever), newton-kerker,
newton-stoner (Kerker + Stoner spin precond on the mag channel, nspin=2 only),
newton-ew (EW + Kerker), newton-ew-ston (EW + Kerker + Stoner). Each newton
variant is measured; the table shows iters / wall / ΔE / total inner
χ₀-applies / applies-per-step. The Stoner lever isolates on Fe-bcc-fm: compare
newton-kerker vs newton-stoner (mag channel un-preconditioned vs preconditioned).

Run start-to-finish (writes its own EXIT marker at the end):
  uv run python benchmarks/newton_scf/nc_newton_bench.py
  uv run python benchmarks/newton_scf/nc_newton_bench.py --systems Fe-bcc-fm
  uv run python benchmarks/newton_scf/nc_newton_bench.py --chi0-tol 1e-6   # cheaper per-apply
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks" / "delta_gauge"))
from lattices import geometry  # noqa: E402

# WIEN2k reference per-atom volumes (Å³/atom) for the cubic elemental metals.
_V0 = {"Al": 16.4796, "Fe": 11.3436, "Cu": 11.9511, "Ni": 10.8876}


def _pseudo(elem):
    from gradwave.pseudo.upf import parse_upf
    return parse_upf(str(_ROOT / "benchmarks" / "delta_gauge" / "pseudos" / f"{elem}.upf"))


# --- system catalogue -------------------------------------------------------
# Each case: (elem, struct, ecut Ry, kmesh, nspin, smearing, width, start_mag,
#             mixing_scheme, nbands, elong).
_CASES = {
    "Al-fcc":    dict(elem="Al", struct="fcc", ecut=30, kmesh=8, nspin=1,
                      smearing="cold", width=0.2, start_mag=None,
                      mixing_scheme=None, nbands=None),
    "Fe-bcc-nm": dict(elem="Fe", struct="bcc", ecut=45, kmesh=8, nspin=1,
                      smearing="cold", width=0.2, start_mag=None,
                      mixing_scheme=None, nbands=12),
    "Fe-bcc-fm": dict(elem="Fe", struct="bcc", ecut=45, kmesh=8, nspin=2,
                      smearing="gaussian", width=0.1, start_mag=[0.7, 0.7],
                      mixing_scheme="pulay", nbands=24),
    # fcc Ni FM: the near-critical-Stoner magnet where history mixing plateaus /
    # collapses the moment (scf.spin_precond was validated here, not on Fe) — the
    # system where preconditioning the magnetization inner mode should bite.
    "Ni-fcc-fm": dict(elem="Ni", struct="fcc", ecut=45, kmesh=8, nspin=2,
                      smearing="gaussian", width=0.1, start_mag=[0.5],
                      mixing_scheme="pulay", nbands=16),
}


def _build(case, kmesh_override=None):
    from gradwave.scf.loop import setup_system

    cell, pos, _ = geometry(case["struct"], case["elem"], _V0[case["elem"]])
    # bcc/fcc primitive cells here are 1 atom; a magnetic case needs one
    # start_mag entry per atom, so size start_mag to the basis.
    natom = pos.shape[0]
    km = case["kmesh"] if kmesh_override is None else kmesh_override
    system = setup_system(cell, pos, [0] * natom, [_pseudo(case["elem"])],
                          ecut=case["ecut"] * RY, kmesh=(km, km, km),
                          nbands=case["nbands"], use_symmetry=False)
    return system, natom


def _xc(case):
    if case["nspin"] == 2:
        from gradwave.core.xc.spin import SpinPBE
        return SpinPBE()
    from gradwave.core.xc.pbe import PBE
    return PBE()


def _run(system, case, natom, *, newton, newton_kwargs, switch, max_iter):
    from gradwave.scf.loop import scf

    sm = case["start_mag"]
    start_mag = None if sm is None else sm[:natom]
    t0 = time.time()
    res = scf(system, _xc(case), smearing=case["smearing"], width=case["width"],
              nspin=case["nspin"], start_mag=start_mag,
              mixing_scheme=case["mixing_scheme"], max_iter=max_iter,
              etol=1e-10, rhotol=1e-8, verbose=False,
              newton_finish=newton, newton_switch=switch,
              newton_kwargs=newton_kwargs)
    wall = time.time() - t0
    return res, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="Al-fcc,Fe-bcc-nm,Fe-bcc-fm",
                    help="comma list from: " + ",".join(_CASES))
    ap.add_argument("--kmesh", type=int, default=None, help="override every case's kmesh")
    ap.add_argument("--switch", type=float, default=1e-3, help="mixer→Newton threshold")
    ap.add_argument("--inner-tol", type=float, default=1e-7,
                    help="fixed inner tol (non-EW). Must be tight enough to reach the "
                         "target rhotol or the density residual floors and the finisher "
                         "stalls (esp. nspin=2 magnetic); tighter is often CHEAPER (fewer "
                         "stalled outer steps).")
    ap.add_argument("--chi0-tol", type=float, default=1e-8,
                    help="Sternheimer tol per χ₀ apply (looser = cheaper apply)")
    ap.add_argument("--max-inner", type=int, default=60)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    base_kw = dict(inner_tol=a.inner_tol, max_inner=a.max_inner, beta=a.beta,
                   chi0_tol=a.chi0_tol)
    # (label, newton?, extra newton_kwargs). Levers isolated:
    variants = [
        ("mixer",          False, None),
        ("newton-plain",   True,  dict(ew=False, precond=False)),
        ("newton-kerker",  True,  dict(ew=False, precond=True)),
        ("newton-stoner",  True,  dict(ew=False, precond=True, spin_precond=True)),
        ("newton-ew",      True,  dict(ew=True, precond=True)),
        ("newton-ew-ston", True,  dict(ew=True, precond=True, spin_precond=True)),
    ]

    hdr = ("%-11s %-14s %5s %8s %11s %8s  %s" %
           ("system", "variant", "iters", "wall_s", "dE_eV", "applies", "applies/step"))
    print(f"# NC inexact-Newton finisher benchmark  switch={a.switch:g}  "
          f"inner_tol={a.inner_tol:g}  chi0_tol={a.chi0_tol:g}  threads={a.threads}",
          flush=True)

    rows = []
    for name in a.systems.split(","):
        name = name.strip()
        if name not in _CASES:
            print(f"# skip unknown system {name!r}", flush=True)
            continue
        case = _CASES[name]
        try:
            system, natom = _build(case, a.kmesh)
        except Exception:  # noqa: BLE001
            print(f"# {name}: BUILD FAILED\n{traceback.format_exc()}", flush=True)
            continue
        km = case["kmesh"] if a.kmesh is None else a.kmesh
        print(f"\n# {name}: {case['elem']} {case['struct']} nspin={case['nspin']} "
              f"ecut={case['ecut']} Ry kmesh={km}^3 (nk_TR={len(system.spheres)}) "
              f"nbands={system.nbands} grid={tuple(system.grid.shape)} "
              f"smear={case['smearing']} {case['width']}", flush=True)
        print("# " + hdr, flush=True)

        e_ref = None
        for label, newton, extra in variants:
            nkw = None if not newton else {**base_kw, **(extra or {})}
            try:
                res, wall = _run(system, case, natom, newton=newton,
                                 newton_kwargs=nkw, switch=a.switch,
                                 max_iter=a.max_iter)
            except Exception:  # noqa: BLE001
                print("  %-11s %-14s  RUN FAILED\n%s" %
                      (name, label, traceback.format_exc()), flush=True)
                continue
            F = float(res.energies.free_energy)
            if label == "mixer":
                e_ref = F
            dE = float("nan") if e_ref is None else F - e_ref
            ni = res.newton_info
            appl = "" if ni is None else str(ni["total_applies"])
            per = "" if ni is None else str(ni["applies_per_step"])
            line = ("  %-11s %-14s %5d %8.1f %11.2e %8s  %s" %
                    (name, label, res.n_iter, wall, dE, appl, per))
            print(line + ("" if res.converged else "  [NOT CONVERGED]"), flush=True)
            rows.append((name, label, res.n_iter, wall, dE, appl,
                         res.converged))

    print("\n# ==== summary ====", flush=True)
    print("# " + hdr, flush=True)
    for name, label, it, wall, dE, appl, conv in rows:
        print("  %-11s %-14s %5d %8.1f %11.2e %8s  %s" %
              (name, label, it, wall, dE, appl,
               "OK" if conv else "NOT-CONVERGED"), flush=True)
    gate = all(abs(dE) < 1e-8 for *_x, dE, _a, _c in rows if not np.isnan(dE))
    print(f"# correctness gate (all |ΔE| < 1e-8 eV vs mixer): "
          f"{'PASS' if gate else 'FAIL'}", flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
