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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.registry import available

ROOT = Path(__file__).resolve().parents[2]
PSEUDOS = ROOT / "tests" / "fixtures" / "qe" / "pseudos"
RESULTS = Path(__file__).resolve().parent / "results"
RESEARCH = Path(__file__).resolve().parent / "RESEARCH.md"
RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Correctness gate: a solver's converged total energy must match Davidson's to
# this absolute tolerance (eV). Recorded exactly regardless; anything above is
# flagged as a finding.
E_MATCH_TOL = 1e-6

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


def _upf(name):
    return parse_upf(PSEUDOS / name)


# ---------------------------------------------------------------------------
# System battery. Each entry: build() -> fresh System, scf kwargs, and metadata.
# A fresh System is built per solver run so nothing leaks between solves.
# ---------------------------------------------------------------------------
def _fcc(a, frac, elems, ecut, kmesh, **kw):
    cell = a / 2.0 * FCC
    pos = np.asarray(frac, dtype=np.float64) @ cell
    return setup_system(cell, pos, list(range(len(frac))), elems,
                        ecut=ecut, kmesh=kmesh, **kw)


def _cubic(a, frac, species, elems, ecut, kmesh, **kw):
    cell = a * np.eye(3)
    pos = np.asarray(frac, dtype=np.float64) @ cell
    return setup_system(cell, pos, species, elems, ecut=ecut, kmesh=kmesh, **kw)


SYSTEMS = {
    # --- insulators (fixed occupations, no smearing) ---
    "si_insulator": dict(
        cls="insulator", formula="Si2",
        build=lambda: _fcc(5.43, [[0, 0, 0], [0.25, 0.25, 0.25]],
                           [_upf("Si_ONCV_PBE-1.2.upf")] * 2, 25 * RY, (4, 4, 4)),
        scf=dict(xc=LDA_PW92(), smearing="none"),
    ),
    "mgo_insulator": dict(
        cls="insulator", formula="MgO",
        build=lambda: _fcc(4.21, [[0, 0, 0], [0.5, 0.5, 0.5]],
                           [_upf("Mg_ONCV_PBE-1.2.upf"), _upf("O_ONCV_PBE-1.2.upf")],
                           45 * RY, (4, 4, 4)),
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
}


def _system_stats(system):
    npw = [int(s.npw) for s in system.spheres]
    return dict(nk=len(system.spheres), nbands=int(system.nbands),
                npw_max=max(npw), n_electrons=float(sum(system.charges).item()))


def run_system(name, spec, solvers):
    """Run every solver on one system; return the per-system result record."""
    common = dict(etol=1e-9, rhotol=1e-8, verbose=False)
    rec = dict(system=name, cls=spec["cls"], formula=spec["formula"],
               baseline_solver="davidson", solvers={})
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
            _EIGH["t"] = 0.0
            _EIGH["n"] = 0
            t0 = time.perf_counter()
            res = scf(system, xc, eigensolver=solver, **kw)
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
        except Exception as exc:  # noqa: BLE001 - record and continue the battery
            entry = dict(error=f"{type(exc).__name__}: {exc}")
            print(f"      ERROR: {entry['error']}", flush=True)
        rec["solvers"][solver] = entry
    if stats:
        rec["stats"] = stats
    return rec


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
    args = ap.parse_args()

    solvers = args.solvers or available()
    if "davidson" in solvers:  # baseline first, for the correctness gate
        solvers = ["davidson"] + [s for s in solvers if s != "davidson"]
    names = args.only or list(SYSTEMS)

    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = dict(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        host=platform.node(), torch=torch.__version__,
        threads=torch.get_num_threads(), solvers=solvers, systems=names,
        E_match_tol_eV=E_MATCH_TOL)
    print(f"battery: {len(names)} systems x {len(solvers)} solvers "
          f"({', '.join(solvers)}) on {meta['host']} "
          f"[{meta['threads']} threads, torch {meta['torch']}]")

    records = []
    for name in names:
        print(f"[{name}]", flush=True)
        rec = run_system(name, SYSTEMS[name], solvers)
        rec["meta"] = meta
        records.append(rec)
        (RESULTS / f"{name}.json").write_text(json.dumps(rec, indent=2))

    (RESULTS / "summary.json").write_text(
        json.dumps(dict(meta=meta, records=records), indent=2))
    update_research(records, meta)
    print(f"\nwrote {len(records)} results to {RESULTS}/ and updated {RESEARCH.name}")


if __name__ == "__main__":
    main()
