"""SCF eigensolver benchmark battery.

Runs the full self-consistent field with every registered block eigensolver
(gradwave.solvers.registry) across a set of small systems spanning the solver-
relevant classes — insulator, simple metal, magnetic metal, antiferromagnet,
heavy/relativistic — and records, per (system x solver):

    * SCF iterations and wall time
    * eigh (Rayleigh-Ritz) self-time and its share of wall time
    * converged? and the final self-consistency residual |drho|
    * total energy, and its match to the Davidson baseline (correctness gate)

Emits one machine-readable JSON per system under results/, an aggregate
summary.json, and regenerates the auto-table block in RESEARCH.md.

This is a HARNESS, not a torture test: cutoffs / k-meshes are the small end of
sensible so each system runs in a few minutes. Physical accuracy is not the
point — the correctness gate compares each solver against Davidson on the SAME
Hamiltonian, so any valid, well-conditioned H exercises it.

Run (single process; torch auto-caps to <=8 CPU threads on import):

    PYTHONPATH=src uv run python benchmarks/solver_battery/run.py
    PYTHONPATH=src uv run python benchmarks/solver_battery/run.py --only si_insulator al_metal
    PYTHONPATH=src uv run python benchmarks/solver_battery/run.py --solvers davidson chebyshev
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.common import MP_CROSSOVER
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.registry import available, is_registered

ROOT = Path(__file__).resolve().parents[2]
PSEUDOS = ROOT / "tests" / "fixtures" / "qe" / "pseudos"
RESULTS = Path(__file__).resolve().parent / "results"
# Mixed-precision sweep writes here so it never clobbers the round-1 fp64 JSONs
# in results/ (owned by PR #86's knowledge base).
RESULTS_MP = RESULTS / "mixed_precision"
RESEARCH = Path(__file__).resolve().parent / "RESEARCH.md"
RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Correctness gate: a solver's converged total energy must match Davidson's to
# this absolute tolerance (eV). Recorded exactly regardless; anything above is
# flagged as a finding.
E_MATCH_TOL = 1e-6

# Mixed-precision accuracy gate: |E_mixed − E_fp64| per atom must stay at or
# below this (eV/atom) or the fp32 draft is flagged as corrupting the answer.
MP_E_TOL_PER_ATOM = 1e-6

# ---------------------------------------------------------------------------
# eigh (Rayleigh-Ritz) timer: wrap torch.linalg.eigh with a resettable
# accumulator. Every registered solver builds its subspace matrix and calls
# torch.linalg.eigh, so this captures the shared Rayleigh-Ritz cost across the
# whole SCF without touching solver code.
# ---------------------------------------------------------------------------
_ORIG_EIGH = torch.linalg.eigh
_EIGH = {"t": 0.0, "n": 0}


def _timed_eigh(*args, **kwargs):
    t0 = time.perf_counter()
    out = _ORIG_EIGH(*args, **kwargs)
    _EIGH["t"] += time.perf_counter() - t0
    _EIGH["n"] += 1
    return out


torch.linalg.eigh = _timed_eigh


def _sync(device):
    """Barrier so wall-clock timing is honest on cuda (ops are async there)."""
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _upf(name):
    return parse_upf(PSEUDOS / name)


# ---------------------------------------------------------------------------
# System battery. Each entry: build() -> fresh System, scf kwargs, and metadata.
# A fresh System is built per solver run so nothing leaks between solves.
# ---------------------------------------------------------------------------
def _fcc(a, frac, elems, ecut, kmesh, disp=None, **kw):
    cell = a / 2.0 * FCC
    pos = np.asarray(frac, dtype=np.float64) @ cell
    if disp is not None:  # Cartesian Å per-atom displacement (warm-start axis)
        pos = pos + np.asarray(disp, dtype=np.float64)
    return setup_system(cell, pos, list(range(len(frac))), elems,
                        ecut=ecut, kmesh=kmesh, **kw)


def _cubic(a, frac, species, elems, ecut, kmesh, disp=None, **kw):
    cell = a * np.eye(3)
    pos = np.asarray(frac, dtype=np.float64) @ cell
    if disp is not None:
        pos = pos + np.asarray(disp, dtype=np.float64)
    return setup_system(cell, pos, species, elems, ecut=ecut, kmesh=kmesh, **kw)


def _fcc_supercell(a, frac, elems_basis, ecut, kmesh, reps=(2, 2, 2),
                   disp=None, **kw):
    """Replicate the fcc-primitive basis (rows of a/2·FCC are the lattice
    vectors) into a reps supercell — MEDIUM systems for the small/medium axis.
    `elems_basis` is the per-basis-atom UPF list; every image reuses it."""
    prim = a / 2.0 * FCC
    base = np.asarray(frac, dtype=np.float64) @ prim
    nx, ny, nz = reps
    pos, elems = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = i * prim[0] + j * prim[1] + k * prim[2]
                for b, p in enumerate(base):
                    pos.append(p + shift)
                    elems.append(elems_basis[b])
    pos = np.asarray(pos, dtype=np.float64)
    if disp is not None:
        pos = pos + np.asarray(disp, dtype=np.float64)
    cell = np.asarray([nx * prim[0], ny * prim[1], nz * prim[2]])
    return setup_system(cell, pos, list(range(len(pos))), elems,
                        ecut=ecut, kmesh=kmesh, **kw)


SYSTEMS = {
    # --- insulators (fixed occupations, no smearing) ---
    "si_insulator": dict(
        cls="insulator", formula="Si2",
        build=lambda: _fcc(5.43, [[0, 0, 0], [0.25, 0.25, 0.25]],
                           [_upf("Si_ONCV_PBE-1.2.upf")] * 2, 25 * RY, (4, 4, 4)),
        # warm_build(disp): same cell/grid, atoms nudged by `disp` (Å) — the
        # perturbed geometry the warm-start axis reuses density+orbitals into.
        warm_build=lambda disp: _fcc(
            5.43, [[0, 0, 0], [0.25, 0.25, 0.25]],
            [_upf("Si_ONCV_PBE-1.2.upf")] * 2, 25 * RY, (4, 4, 4), disp=disp),
        scf=dict(xc=LDA_PW92(), smearing="none"),
    ),
    "mgo_insulator": dict(
        cls="insulator", formula="MgO",
        build=lambda: _fcc(4.21, [[0, 0, 0], [0.5, 0.5, 0.5]],
                           [_upf("Mg_ONCV_PBE-1.2.upf"), _upf("O_ONCV_PBE-1.2.upf")],
                           45 * RY, (4, 4, 4)),
        warm_build=lambda disp: _fcc(
            4.21, [[0, 0, 0], [0.5, 0.5, 0.5]],
            [_upf("Mg_ONCV_PBE-1.2.upf"), _upf("O_ONCV_PBE-1.2.upf")],
            45 * RY, (4, 4, 4), disp=disp),
        scf=dict(xc=LDA_PW92(), smearing="none"),
    ),
    # --- simple / noble metals (smeared, shared Fermi level) ---
    "al_metal": dict(
        cls="simple metal", formula="Al",
        build=lambda: _fcc(4.05, [[0, 0, 0]], [_upf("Al_ONCV_PBE-1.2.upf")],
                           22 * RY, (6, 6, 6)),
        scf=dict(xc=LDA_PW92(), smearing="gaussian", width=0.1),
    ),
    "cu_metal": dict(
        cls="noble metal", formula="Cu",
        build=lambda: _fcc(3.61, [[0, 0, 0]], [_upf("Cu_ONCV_PBE-1.2.upf")],
                           45 * RY, (6, 6, 6), nbands=16),
        scf=dict(xc=LDA_PW92(), smearing="gaussian", width=0.1),
    ),
    # --- magnetic metal (FM bcc Fe, collinear nspin=2) ---
    "fe_fm_metal": dict(
        cls="ferromagnetic metal", formula="Fe",
        build=lambda: _cubic(2.87, [[0, 0, 0]], [0], [_upf("Fe_ONCV_PBE-1.2.upf")],
                            45 * RY, (5, 5, 5), nbands=16),
        scf=dict(xc=LSDA_PW92(), smearing="gaussian", width=0.1, nspin=2,
                 start_mag=[0.4], mixing_scheme="johnson", max_iter=120),
    ),
    # --- antiferromagnet (2-atom bcc Cr, opposite start moments) ---
    "cr_afm_metal": dict(
        cls="antiferromagnet", formula="Cr2",
        build=lambda: _cubic(2.88, [[0, 0, 0], [0.5, 0.5, 0.5]], [0, 0],
                            [_upf("Cr_ONCV_PBE-1.2.upf")] * 2, 45 * RY, (4, 4, 4),
                            nbands=24),
        scf=dict(xc=LSDA_PW92(), smearing="gaussian", width=0.1, nspin=2,
                 start_mag=[0.6, -0.6], mixing_scheme="johnson", max_iter=120),
    ),
    # --- heavy / relativistic element (scalar-relativistic Bi; TRUE SOC needs
    #     the spinor SCF, a documented registry gap — see RESEARCH.md) ---
    "bi_heavy": dict(
        cls="heavy (scalar-rel)", formula="Bi",
        build=lambda: _fcc(5.05, [[0, 0, 0]], [_upf("Bi_ONCV_PBE_sr.upf")],
                           30 * RY, (6, 6, 6)),
        scf=dict(xc=LDA_PW92(), smearing="gaussian", width=0.1),
    ),
    # --- MEDIUM systems (small/medium axis) — 2×2×2 fcc supercells, sized so
    #     the fp64 GPU sweep fits a 6 GB RTX 3050 (probed: wfc blocks ~20-35 MB,
    #     npw_max ≲ 3.3k, nk=8). Symmetry ON keeps the irreducible k-set small;
    #     cutoffs stay at the small end (solver-relative timing, not physics). ---
    "si_insulator_m": dict(
        cls="insulator (medium)", formula="Si16",
        build=lambda: _fcc_supercell(
            5.43, [[0, 0, 0], [0.25, 0.25, 0.25]],
            [_upf("Si_ONCV_PBE-1.2.upf")] * 2, 25 * RY, (2, 2, 2),
            reps=(2, 2, 2), use_symmetry=True),
        scf=dict(xc=LDA_PW92(), smearing="none"),
    ),
    "cu_metal_m": dict(
        # Cu ONCV carries 19 valence e⁻/atom → 8 atoms = 152 e⁻ (76 occupied
        # bands); nbands=96 leaves a ~20-band smearing buffer. ecut 40 Ry / k222
        # keeps npw_max ≈ 2.7k so the fp64 subspace stays well under 6 GB.
        cls="noble metal (medium)", formula="Cu8",
        build=lambda: _fcc_supercell(
            3.61, [[0, 0, 0]], [_upf("Cu_ONCV_PBE-1.2.upf")], 40 * RY, (2, 2, 2),
            reps=(2, 2, 2), nbands=96, use_symmetry=True),
        scf=dict(xc=LDA_PW92(), smearing="gaussian", width=0.1),
    ),
}

# Systems used for the warm-start axis (positions-only perturbation → same FFT
# grid → density+orbital reuse via scf(start_from=...)). Restricted to the small
# 2-atom nspin=1 insulators: single-atom metals give a trivial (translational)
# perturbation, and nspin=2 cells fall outside the calculator's warm-start path.
WARM_SYSTEMS = ["si_insulator", "mgo_insulator"]


def _system_stats(system):
    npw = [int(s.npw) for s in system.spheres]
    return dict(nk=len(system.spheres), nbands=int(system.nbands),
                npw_max=max(npw), n_electrons=float(sum(system.charges).item()))


def run_system(name, spec, solvers, device="cpu", mixed=False):
    """Run every solver on one system; return the per-system record.

    `mixed` toggles scf(mixed_precision=...) for the whole slice — the matrix
    mode runs one (host, device, mp) slice per invocation, so every solver here
    shares the same precision and the Davidson correctness baseline is the
    same-precision Davidson run."""
    common = dict(etol=1e-9, rhotol=1e-8, verbose=False)
    rec = dict(system=name, cls=spec["cls"], formula=spec["formula"],
               baseline_solver="davidson", device=device, mixed_precision=mixed,
               solvers={})
    baseline_E = None
    stats = None
    for solver in solvers:
        print(f"  [{name}] {solver} ...", flush=True)
        kw = dict(common)
        kw.update(spec["scf"])
        xc = kw.pop("xc")
        try:
            system = spec["build"]()
            if stats is None:
                stats = _system_stats(system)
            if device != "cpu":
                system = system.to(device)
            _EIGH["t"] = 0.0
            _EIGH["n"] = 0
            _sync(device)
            t0 = time.perf_counter()
            res = scf(system, xc, eigensolver=solver, mixed_precision=mixed, **kw)
            _sync(device)
            wall = time.perf_counter() - t0
            eigh_t = _EIGH["t"]
            E = float(res.energies.total)
            final_res = float(res.history[-1]["res"]) if res.history else None
            if solver == "davidson":
                baseline_E = E
            dE = None if baseline_E is None else abs(E - baseline_E)
            entry = dict(
                converged=bool(res.converged), scf_iters=int(res.n_iter),
                wall_s=round(wall, 3), eigh_s=round(eigh_t, 3),
                eigh_share=round(eigh_t / wall, 4) if wall > 0 else None,
                eigh_calls=_EIGH["n"], final_res=final_res,
                energy_eV=E,
                dE_vs_baseline_eV=(None if dE is None else float(dE)),
                energy_match=(None if dE is None else bool(dE <= E_MATCH_TOL)),
                mag_total=(float(res.mag_total)
                           if getattr(res, "mag_total", None) else None),
                error=None,
            )
            flag = ("" if entry["energy_match"] in (None, True)
                    else f"  !! dE={dE:.2e} eV > {E_MATCH_TOL:.0e}")
            print(f"      iters={res.n_iter} conv={res.converged} "
                  f"wall={wall:.1f}s eigh%={entry['eigh_share']} "
                  f"E={E:.9f}{flag}", flush=True)
        except Exception as exc:
            entry = dict(error=f"{type(exc).__name__}: {exc}")
            print(f"      ERROR: {entry['error']}", flush=True)
        rec["solvers"][solver] = entry
    if stats:
        rec["stats"] = stats
    return rec


# ---------------------------------------------------------------------------
# Mixed-precision axis
# ---------------------------------------------------------------------------
# The battery's round 1 ran mixed_precision=False, so the fp32-draft payoff is
# unmeasured. Here each system runs a solver TWICE — fp64 baseline and
# scf(..., mixed_precision=True) — and we record wall/iters/energy/|Δρ| for each
# plus the derived speedup, iters delta, and energy error. Written to
# results/mixed_precision/ (never touches results/ or RESEARCH.md).
#
# fp32 draft H-applies attack the ~90% of SCF wall time that is per-iteration
# FFTs + nonlocal projectors (the solver battery proved eigh is ≤13%); the loop
# crosses back to fp64 once the diago tolerance drops below MP_CROSSOVER=1e-5.
# CheFSI is skipped (proven 10–35× too slow to be worth the runtime); lobpcg is
# included automatically if a later worker has registered it.
# ---------------------------------------------------------------------------
def _mp_solvers(requested):
    """Which solvers to sweep in the mixed-precision axis. Davidson always;
    lobpcg if registered; chebyshev is intentionally excluded (too slow)."""
    if requested:
        sv = [s for s in requested if s != "chebyshev"]
    else:
        sv = ["davidson"] + (["lobpcg"] if is_registered("lobpcg") else [])
    if "davidson" in sv:  # baseline first, stable order
        sv = ["davidson"] + [s for s in sv if s != "davidson"]
    return sv


def _scf_once(spec, solver, mixed, device="cpu"):
    """One full SCF solve; returns a per-run record (wall/iters/energy/|Δρ|).

    device: "cpu" (default) or "cuda" — the freshly built System is moved to the
    chosen device before the solve so the whole SCF runs there. On cuda we
    synchronize around the timer so wall time reflects real kernel execution
    (torch cuda ops are async; without the sync the wall would only clock launch
    overhead)."""
    common = dict(etol=1e-9, rhotol=1e-8, verbose=False)
    kw = dict(common)
    kw.update(spec["scf"])
    xc = kw.pop("xc")
    system = spec["build"]()
    natoms = len(system.positions)
    if device != "cpu":
        system = system.to(device)
    _EIGH["t"] = 0.0
    _EIGH["n"] = 0
    _sync(device)
    t0 = time.perf_counter()
    res = scf(system, xc, eigensolver=solver, mixed_precision=mixed, **kw)
    _sync(device)
    wall = time.perf_counter() - t0
    E = float(res.energies.total)
    final_res = float(res.history[-1]["res"]) if res.history else None
    return dict(
        mixed_precision=mixed, device=device, natoms=natoms,
        converged=bool(res.converged), scf_iters=int(res.n_iter),
        wall_s=round(wall, 3), eigh_s=round(_EIGH["t"], 3),
        eigh_share=round(_EIGH["t"] / wall, 4) if wall > 0 else None,
        eigh_calls=_EIGH["n"], final_res=final_res, energy_eV=E, error=None,
    )


def _mp_compare(fp64, mixed):
    """Derived metrics: speedup, iters delta, energy error, accuracy gate."""
    if fp64.get("error") or mixed.get("error"):
        return dict(error="one or both runs failed")
    speedup = (fp64["wall_s"] / mixed["wall_s"]) if mixed["wall_s"] > 0 else None
    e_err = abs(mixed["energy_eV"] - fp64["energy_eV"])
    natoms = fp64["natoms"] or 1
    e_err_per_atom = e_err / natoms
    return dict(
        speedup=(None if speedup is None else round(speedup, 3)),
        iters_delta=mixed["scf_iters"] - fp64["scf_iters"],
        energy_error_eV=e_err,
        energy_error_eV_per_atom=e_err_per_atom,
        energy_ok=bool(e_err_per_atom <= MP_E_TOL_PER_ATOM),
        # a regression = mixed is slower than fp64 (extra fp32-noise SCF steps
        # outweighing the per-iteration speedup)
        regresses=bool(speedup is not None and speedup < 1.0),
    )


def run_system_mp(name, spec, solvers, device="cpu"):
    """Run each solver at fp64 and mixed precision on one system; return the
    per-system record with derived comparison metrics."""
    rec = dict(system=name, cls=spec["cls"], formula=spec["formula"],
               device=device, solvers={})
    for solver in solvers:
        print(f"  [{name}] {solver}: fp64 vs mixed", flush=True)
        runs = {}
        for mixed in (False, True):
            tag = "mixed" if mixed else "fp64"
            try:
                r = _scf_once(spec, solver, mixed, device=device)
                print(f"      {tag:5s} iters={r['scf_iters']} "
                      f"conv={r['converged']} wall={r['wall_s']:.1f}s "
                      f"|Δρ|={_fmt(r['final_res'], '{:.1e}')} "
                      f"E={r['energy_eV']:.9f}", flush=True)
            except Exception as exc:
                # OOM is expected on a 6 GB GPU for the larger k-meshes (fp64
                # doubles memory). Record "OOM" and free the cache so the next
                # system starts from a clean allocator instead of cascading.
                is_oom = "out of memory" in str(exc).lower()
                r = dict(mixed_precision=mixed, device=device,
                         oom=is_oom, error=f"{type(exc).__name__}: {exc}")
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tag_msg = "OOM" if is_oom else "ERROR"
                print(f"      {tag:5s} {tag_msg}: {r['error']}", flush=True)
            runs[tag] = r
        comp = _mp_compare(runs["fp64"], runs["mixed"])
        if not comp.get("error"):
            flag = "" if comp["energy_ok"] else (
                f"  !! E err {comp['energy_error_eV_per_atom']:.2e} eV/atom "
                f"> {MP_E_TOL_PER_ATOM:.0e}")
            reg = "  REGRESSES" if comp["regresses"] else ""
            print(f"      -> speedup={comp['speedup']}x "
                  f"iters_delta={comp['iters_delta']:+d}{reg}{flag}", flush=True)
        rec["solvers"][solver] = dict(fp64=runs["fp64"], mixed=runs["mixed"],
                                      comparison=comp)
    return rec


def render_mp_table(records):
    """Plain-text summary table for the log (NOT written to RESEARCH.md)."""
    hdr = (f"{'System':<26} {'Solver':<9} {'fp64 s':>8} {'mixed s':>8} "
           f"{'speedup':>8} {'fp64 it':>8} {'mix it':>7} {'E err/at eV':>13} "
           f"{'gate':>5}")
    lines = [hdr, "-" * len(hdr)]
    for rec in records:
        for solver, e in rec["solvers"].items():
            f, m, c = e["fp64"], e["mixed"], e["comparison"]
            sysc = f"{rec['formula']} {rec['system']}"
            if c.get("error"):
                lines.append(f"{sysc:<26} {solver:<9} ERROR")
                continue
            gate = "ok" if c["energy_ok"] else "FLAG"
            lines.append(
                f"{sysc:<26} {solver:<9} {f['wall_s']:>8.1f} {m['wall_s']:>8.1f} "
                f"{c['speedup']:>7.2f}x {f['scf_iters']:>8d} {m['scf_iters']:>7d} "
                f"{c['energy_error_eV_per_atom']:>13.2e} {gate:>5}")
    return "\n".join(lines)


def run_mixed_precision(names, solvers, meta, device="cpu"):
    # cpu keeps writing flat into results/mixed_precision/ (identical to before);
    # cuda writes to results/mixed_precision/cuda/ so the GPU sweep never
    # clobbers the committed CPU baseline it is meant to be contrasted against.
    out_dir = RESULTS_MP if device == "cpu" else RESULTS_MP / device
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        print(f"[{name}]", flush=True)
        rec = run_system_mp(name, SYSTEMS[name], solvers, device=device)
        rec["meta"] = meta
        records.append(rec)
        (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2))
    (out_dir / "summary.json").write_text(
        json.dumps(dict(meta=meta, records=records), indent=2))
    print("\n" + render_mp_table(records))
    print(f"\nwrote {len(records)} mixed-precision results to {out_dir}/ "
          f"(RESEARCH.md untouched)")


# ---------------------------------------------------------------------------
# Warm-start axis
# ---------------------------------------------------------------------------
# Question: how much of a second SCF does density + wavefunction reuse buy after
# a small ionic move? For each small nspin=1 system we converge a COLD reference
# SCF, nudge every atom by ~0.05 Å (positions-only → the FFT grid is unchanged),
# then run the perturbed geometry twice: (a) cold and (b) warm-started via
# scf(start_from=res0). start_from carries the reference density AND orbitals, so
# loop.py's _seed_density/_seed_orbitals reuse both — the same mechanism
# calculator.py rides between ionic/relaxation steps (minus the atomic-ρ
# extrapolation, which is a further optimization on top of plain reuse). We
# record the perturbed-SCF wall + iteration count for cold vs warm.
# ---------------------------------------------------------------------------
def _perturb(natoms, amp=0.05, seed=0):
    """Deterministic per-atom Cartesian displacement whose largest single
    component is `amp` Å (reproducible across runs and hosts)."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((natoms, 3))
    return d * (amp / np.abs(d).max())


def _warm_seed(res0, new_system):
    """Calculator-style extrapolated warm-start seed for a same-grid ionic move.

    Mirrors calculator._warm_start's norm-conserving, same-FFT-grid branch: the
    superposition-of-atoms part of ρ travels with the atoms to their NEW
    positions (the sad_density delta) while the bonding remainder of the
    converged reference density is reused, and the reference orbitals ride along
    (QE wfc-extrapolation analogue). Plain reuse (start_from=res0) instead leaves
    the atomic charge at the OLD positions — for a ~50 mÅ move that stale seed
    can lose to a fresh SAD guess, which is why the extrapolated form is what
    calculator.py actually uses between ionic steps."""
    from gradwave.scf.guess import sad_density

    prev_sys = res0.system
    pos_new = new_system.positions
    pos_old = prev_sys.positions.to(pos_new.device)
    soa, ne, tabs = (prev_sys.species_of_atom, prev_sys.n_electrons,
                     prev_sys.upfs)
    delta = (sad_density(new_system.grid, pos_new, soa, tabs, ne)
             - sad_density(prev_sys.grid, pos_old, soa, tabs, ne))
    return {"system": prev_sys, "nspin": 1, "rho": res0.rho.detach() + delta,
            "rho_spin": None, "coeffs": res0.coeffs}


def run_warm_start(names, meta, device="cpu", mixed=False, solver="davidson"):
    """Cold reference → perturb → {cold, warm} second SCF, per system."""
    common = dict(etol=1e-9, rhotol=1e-8, verbose=False)
    records = []
    for name in names:
        spec = SYSTEMS[name]
        if "warm_build" not in spec:
            print(f"  [{name}] no warm_build — skipped", flush=True)
            continue
        kw = dict(common)
        kw.update(spec["scf"])
        xc = kw.pop("xc")
        print(f"[warm:{name}]", flush=True)

        def _solve(system, start_from, _xc=xc, _kw=kw):
            if device != "cpu":
                system = system.to(device)
            _sync(device)
            t0 = time.perf_counter()
            res = scf(system, _xc, eigensolver=solver, mixed_precision=mixed,
                      start_from=start_from, **_kw)
            _sync(device)
            return res, time.perf_counter() - t0

        try:
            res0, w0 = _solve(spec["warm_build"](None), None)  # cold reference
            natoms = int(res0.system.positions.shape[0])
            disp = _perturb(natoms, amp=0.05, seed=0)
            res_c, wc = _solve(spec["warm_build"](disp), None)       # cold restart
            warm_sys = spec["warm_build"](disp)
            if device != "cpu":
                warm_sys = warm_sys.to(device)
            # calculator-style density-extrapolated + orbital reuse warm start
            res_w, ww = _solve(warm_sys, _warm_seed(res0, warm_sys))
            de = abs(float(res_w.energies.total) - float(res_c.energies.total))
            rec = dict(
                system=name, cls=spec["cls"], formula=spec["formula"],
                device=device, mixed_precision=mixed, solver=solver,
                natoms=natoms, disp_amp_A=0.05,
                reference=dict(scf_iters=int(res0.n_iter), wall_s=round(w0, 3),
                               converged=bool(res0.converged),
                               energy_eV=float(res0.energies.total)),
                cold=dict(scf_iters=int(res_c.n_iter), wall_s=round(wc, 3),
                          converged=bool(res_c.converged),
                          energy_eV=float(res_c.energies.total)),
                warm=dict(scf_iters=int(res_w.n_iter), wall_s=round(ww, 3),
                          converged=bool(res_w.converged),
                          energy_eV=float(res_w.energies.total)),
                iters_saved=int(res_c.n_iter) - int(res_w.n_iter),
                speedup=round(wc / ww, 3) if ww > 0 else None,
                warm_cold_energy_diff_eV=de,
            )
            print(f"      cold iters={rec['cold']['scf_iters']} wall={wc:.1f}s | "
                  f"warm iters={rec['warm']['scf_iters']} wall={ww:.1f}s | "
                  f"saved={rec['iters_saved']} speedup={rec['speedup']}x | "
                  f"ΔE(warm−cold)={de:.2e} eV", flush=True)
        except Exception as exc:
            rec = dict(system=name, error=f"{type(exc).__name__}: {exc}")
            print(f"      ERROR: {rec['error']}", flush=True)
        rec["meta"] = meta
        records.append(rec)
    return records


def render_warm_table(records):
    """Plain-text warm-start summary for the log."""
    hdr = (f"{'System':<24} {'cold it':>7} {'warm it':>7} {'saved':>6} "
           f"{'cold s':>8} {'warm s':>8} {'speedup':>8} {'ΔE eV':>10}")
    lines = [hdr, "-" * len(hdr)]
    for r in records:
        if r.get("error"):
            lines.append(f"{r['system']:<24} ERROR: {r['error']}")
            continue
        sysc = f"{r['formula']} {r['system']}"
        spd = _fmt(r["speedup"], "{:.2f}") + "x"
        lines.append(
            f"{sysc:<24} {r['cold']['scf_iters']:>7d} {r['warm']['scf_iters']:>7d} "
            f"{r['iters_saved']:>6d} {r['cold']['wall_s']:>8.1f} "
            f"{r['warm']['wall_s']:>8.1f} {spd:>8} "
            f"{r['warm_cold_energy_diff_eV']:>10.1e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matrix mode — one coarse (host, device, mp) slice per invocation
# ---------------------------------------------------------------------------
# Runs the full solver battery (correctness gate + timing) at a FIXED precision
# plus the warm-start axis, writing JSON + a printed summary table into
# results/matrix/<host>_<device>_<mp>/. Never edits RESEARCH.md — this is sweep
# output, contrasted across slices, not the committed round-1 knowledge base.
# The sweep fans several of these out inside one gwq job (small mp-off, small
# mp-on, medium both, all on one --device).
# ---------------------------------------------------------------------------
def run_matrix(names, solvers, warm_names, meta, device, mixed, out_dir,
               do_warm=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        print(f"[{name}]", flush=True)
        rec = run_system(name, SYSTEMS[name], solvers, device=device, mixed=mixed)
        rec["meta"] = meta
        records.append(rec)
        (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2))
    warm_records = []
    if do_warm and warm_names:
        print("\n[warm-start axis]", flush=True)
        warm_records = run_warm_start(warm_names, meta, device=device, mixed=mixed)
        (out_dir / "warm_start.json").write_text(
            json.dumps(dict(meta=meta, records=warm_records), indent=2))
    (out_dir / "summary.json").write_text(json.dumps(
        dict(meta=meta, records=records, warm_start=warm_records), indent=2))
    print("\n" + render_table(records, meta))
    if warm_records:
        print("\nwarm-start axis:\n" + render_warm_table(warm_records))
    print(f"\nwrote matrix slice ({len(records)} systems"
          f"{', + warm-start' if warm_records else ''}) to {out_dir}/ "
          f"(RESEARCH.md untouched)")


# ---------------------------------------------------------------------------
# RESEARCH.md auto-table regeneration (between markers; prose is preserved)
# ---------------------------------------------------------------------------
TABLE_START = "<!-- RESULTS TABLE START -->"
TABLE_END = "<!-- RESULTS TABLE END -->"


def _fmt(v, spec="{}"):
    return "—" if v is None else spec.format(v)


def render_table(records, meta):
    lines = [TABLE_START,
             f"_Auto-generated by `benchmarks/solver_battery/run.py` on "
             f"{meta['timestamp']} — {meta['host']}, torch {meta['torch']}, "
             f"{meta['threads']} threads. Do not edit by hand; edit prose "
             f"outside the markers._", "",
             "| System | Class | Solver | SCF iters | Wall (s) | eigh (s) | "
             "eigh % | Converged | Final |Δρ| | E (eV) | ΔE vs Davidson (eV) |",
             "|---|---|---|--:|--:|--:|--:|:--:|--:|--:|--:|"]
    for rec in records:
        for i, (solver, e) in enumerate(rec["solvers"].items()):
            sysc = f"{rec['formula']} `{rec['system']}`" if i == 0 else ""
            clsc = rec["cls"] if i == 0 else ""
            if e.get("error"):
                lines.append(f"| {sysc} | {clsc} | {solver} | ERROR | | | | | | | "
                             f"{e['error']} |")
                continue
            conv = "yes" if e["converged"] else "**NO**"
            dE = e["dE_vs_baseline_eV"]
            dEc = ("baseline" if solver == rec["baseline_solver"]
                   else (_fmt(dE, "{:.2e}") + ("" if e["energy_match"] else " ⚠")))
            lines.append(
                f"| {sysc} | {clsc} | {solver} | {e['scf_iters']} | "
                f"{_fmt(e['wall_s'], '{:.1f}')} | {_fmt(e['eigh_s'], '{:.2f}')} | "
                f"{_fmt(e['eigh_share'], '{:.0%}') if e['eigh_share'] is not None else '—'} | "
                f"{conv} | {_fmt(e['final_res'], '{:.1e}')} | "
                f"{_fmt(e['energy_eV'], '{:.6f}')} | {dEc} |")
    lines += ["", TABLE_END]
    return "\n".join(lines)


def update_research(records, meta):
    table = render_table(records, meta)
    text = RESEARCH.read_text()
    if TABLE_START in text and TABLE_END in text:
        pre = text[: text.index(TABLE_START)]
        post = text[text.index(TABLE_END) + len(TABLE_END):]
        RESEARCH.write_text(pre + table + post)
    else:  # append if markers are missing
        RESEARCH.write_text(text.rstrip() + "\n\n" + table + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of system names to run")
    ap.add_argument("--solvers", nargs="*", default=None,
                    help="subset of registered solvers (default: all)")
    ap.add_argument("--mixed-precision", action="store_true",
                    help="run the mixed-precision axis instead: each solver at "
                         "fp64 and mixed_precision=True; writes to "
                         "results/mixed_precision/ and leaves RESEARCH.md alone")
    ap.add_argument("--matrix", action="store_true",
                    help="run one coarse (host, device, mp) matrix slice: the "
                         "full solver battery at a FIXED precision (--mixed) plus "
                         "the warm-start axis, into "
                         "results/matrix/<host>_<device>_<mp>/. RESEARCH.md "
                         "untouched. Fan several of these out per gwq job.")
    ap.add_argument("--mixed", choices=("on", "off"), default="off",
                    help="matrix mode precision: 'on' runs every SCF with "
                         "mixed_precision=True, 'off' (default) fp64.")
    ap.add_argument("--warm-systems", nargs="*", default=None,
                    help="matrix mode: warm-start systems (default: the small "
                         "nspin=1 insulators). Empty list / --no-warm skips it.")
    ap.add_argument("--no-warm", action="store_true",
                    help="matrix mode: skip the warm-start axis.")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="device the SCF runs on. cpu (default) is unchanged; "
                         "cuda moves each System to the GPU before every solve "
                         "(consumer NVIDIA fp64 is ~1/32-1/64 of fp32, so this "
                         "is where fp32-draft mixed precision should pay off). "
                         "GPU mixed-precision results write to "
                         "results/mixed_precision/cuda/.")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() "
                         "is False (no CUDA build / no visible GPU).")

    names = args.only or list(SYSTEMS)

    if args.matrix:
        mixed = args.mixed == "on"
        mp_tag = "mp-on" if mixed else "mp-off"
        solvers = args.solvers or available()
        if "davidson" in solvers:  # baseline first for the correctness gate
            solvers = ["davidson"] + [s for s in solvers if s != "davidson"]
        warm_names = ([] if args.no_warm else
                      (args.warm_systems if args.warm_systems is not None
                       else [n for n in WARM_SYSTEMS if n in names]))
        out_dir = RESULTS / "matrix" / f"{platform.node()}_{device}_{mp_tag}"
        meta = dict(
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            host=platform.node(), torch=torch.__version__, device=device,
            gpu=(torch.cuda.get_device_name(0) if device == "cuda" else None),
            threads=torch.get_num_threads(), solvers=solvers, systems=names,
            warm_systems=warm_names, mode="matrix", mixed_precision=mixed,
            E_match_tol_eV=E_MATCH_TOL, mp_crossover=MP_CROSSOVER)
        print(f"matrix slice: {len(names)} systems x {len(solvers)} solvers "
              f"({', '.join(solvers)}) [{mp_tag}] on {meta['host']} "
              f"[device={device}, {meta['threads']} threads, torch {meta['torch']}]"
              f" -> {out_dir}/")
        run_matrix(names, solvers, warm_names, meta, device, mixed, out_dir,
                   do_warm=not args.no_warm)
        return

    if args.mixed_precision:
        solvers = _mp_solvers(args.solvers)
        meta = dict(
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            host=platform.node(), torch=torch.__version__, device=device,
            gpu=(torch.cuda.get_device_name(0) if device == "cuda" else None),
            threads=torch.get_num_threads(), solvers=solvers, systems=names,
            mode="mixed_precision", mp_crossover=MP_CROSSOVER,
            mp_e_tol_eV_per_atom=MP_E_TOL_PER_ATOM)
        print(f"mixed-precision axis: {len(names)} systems x {len(solvers)} "
              f"solvers ({', '.join(solvers)}) x [fp64, mixed] on {meta['host']} "
              f"[device={device}, {meta['threads']} threads, "
              f"torch {meta['torch']}, crossover {MP_CROSSOVER:.0e}]")
        run_mixed_precision(names, solvers, meta, device=device)
        return

    solvers = args.solvers or available()
    if "davidson" in solvers:  # baseline first, for the correctness gate
        solvers = ["davidson"] + [s for s in solvers if s != "davidson"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = dict(
        timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        host=platform.node(), torch=torch.__version__, device=device,
        gpu=(torch.cuda.get_device_name(0) if device == "cuda" else None),
        threads=torch.get_num_threads(), solvers=solvers, systems=names,
        E_match_tol_eV=E_MATCH_TOL)
    print(f"battery: {len(names)} systems x {len(solvers)} solvers "
          f"({', '.join(solvers)}) on {meta['host']} "
          f"[device={device}, {meta['threads']} threads, torch {meta['torch']}]")

    records = []
    for name in names:
        print(f"[{name}]", flush=True)
        rec = run_system(name, SYSTEMS[name], solvers, device=device)
        rec["meta"] = meta
        records.append(rec)
        (RESULTS / f"{name}.json").write_text(json.dumps(rec, indent=2))

    (RESULTS / "summary.json").write_text(
        json.dumps(dict(meta=meta, records=records), indent=2))
    update_research(records, meta)
    print(f"\nwrote {len(records)} results to {RESULTS}/ and updated {RESEARCH.name}")


if __name__ == "__main__":
    main()
