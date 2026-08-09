"""Full pipeline for ONE adsorbate–surface pair (the run-through target).

  1. relax the clean slab                         → E_clean
  2. relax the adsorbate at each (111) site        → E(site)   [warm-started]
  3. relax the gas reference (H2 for *H, CO for *CO)→ E_gas
  4. adsorption energy + CHE free energy + report

Robust-by-default: every energy is written to results/<pair>.json as it lands, so
a mid-run drop loses nothing and a re-run skips finished pieces.

    uv run python run_pair.py Pt H            # one pair, production settings (GPU)
    uv run python run_pair.py Pt H --debug    # tiny/fast, CPU — for local debugging
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import che  # noqa: E402
import config  # noqa: E402

STRUCT = HERE / "structures"
RESULTS = HERE / "results"
GAS_FOR = {"H": "H2", "CO": "CO"}
GAS_SCALE = {"H": 0.5, "CO": 1.0}  # ½H2 for *H, CO for *CO


def _provenance() -> dict:
    """One-time environment/provenance snapshot so a result is reproducible and a
    slow run can be blamed on a throttled/loaded host. All best-effort."""
    info: dict = {}
    try:
        from gradwave import runinfo as ri
        info["git"] = ri._git_commit()
        info["cpu"], info["memory"] = ri.cpu_info(), ri.memory_info()
        info["gpu"], info["thermal"] = ri.gpu_info(), ri.thermal_info()
    except Exception as e:
        info["runinfo_err"] = str(e)[:120]
    try:
        import torch

        import gradwave
        info["torch"] = torch.__version__
        info["gradwave"] = getattr(gradwave, "__version__", "?")
        info["cuda"] = bool(torch.cuda.is_available())
        info["device"] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu")
        info["torch_threads"] = torch.get_num_threads()
    except Exception as e:
        info["ver_err"] = str(e)[:120]
    info["config"] = dict(ecut=config.ECUT, ecutrho=config.ECUTRHO,
                          kpts_slab=config.KPTS_SLAB, kpts_gas=config.KPTS_GAS,
                          smearing=config.SMEARING, width=config.WIDTH, xc=config.XC,
                          fmax=config.FMAX, max_steps=config.MAX_STEPS)
    return info


def _final_fmax(log_path) -> float | None:
    try:
        return float(Path(log_path).read_text().strip().splitlines()[-1].split()[-1])
    except Exception:
        return None


def _diagnostics(calc, opt, fmax, max_steps, wall, log_path, do_wf) -> dict:
    """Per-stage diagnostics (all best-effort — never raises). Surfaces the SILENT
    failure modes: a metal converging as an insulator, nbands too small to hold the
    smearing tail, and a geometry that hit the BFGS step cap without converging."""
    d: dict = {"wall_s": round(wall, 1), "final_fmax": _final_fmax(log_path)}
    try:
        d["bfgs_steps"] = int(opt.nsteps)
        ff = d["final_fmax"]
        d["bfgs_capped"] = bool(opt.nsteps >= max_steps and (ff is None or ff > fmax))
    except Exception as e:
        d["bfgs_err"] = str(e)[:80]
    try:
        r = calc.last_result
        d["scf_converged"] = bool(r.converged)
        d["scf_n_iter"] = int(r.n_iter)
        d["fermi_eV"] = float(r.fermi)
        d["n_electrons"] = float(getattr(r, "n_electrons", 0.0) or 0.0)
        d["kerker_used"] = getattr(r, "kerker_used", None)
        occ = r.occupations.detach().to("cpu")
        full = 2.0 if getattr(r, "nspin", 1) == 1 else 1.0
        d["fractional_occ_count"] = int(((occ > 1e-3) & (occ < full - 1e-3)).sum())
        d["is_metallic"] = bool(d["fractional_occ_count"] > 0)
        d["max_top_band_occ"] = float(occ[..., -1].max())  # ≳1e-3 ⇒ raise nbands
        dr = getattr(r, "drho_scf", None)
        if dr is not None:
            d["scf_residual"] = float(dr.abs().max())
    except Exception as e:
        d["scf_diag_err"] = str(e)[:80]
    if do_wf:
        # work_function reads res.v_eff, which only the NC SCFResult exposes. The
        # USPP/PAW path (this campaign) returns a USPPResult without it — a correct Φ
        # for a PAW slab needs v_eff reconstructed from the converged density
        # (gauge-consistent with E_F), a dedicated post-processing step. Record
        # cleanly rather than erroring or computing a wrong-by-a-constant value.
        r = calc.last_result
        if getattr(r, "formalism", "nc") == "nc":
            try:
                from gradwave.postscf.work_function import work_function
                d["work_function_eV"] = float(work_function(r))
            except Exception as e:
                d["wf_err"] = str(e)[:80]
        else:
            d["work_function"] = "n/a: uspp/paw result carries no v_eff (Φ via post-proc)"
    return d


def _relax(atoms, calc, fmax, max_steps, tag, do_wf=True):
    atoms.calc = calc
    t = time.time()
    # logfile: per-step energy/fmax text; trajectory: every ionic step's geometry
    # (with forces) so the relaxation path can be replayed / restarted / visualized
    opt = BFGS(atoms, logfile=str(RESULTS / f"{tag}.log"),
               trajectory=str(RESULTS / f"{tag}.traj"))
    opt.run(fmax=fmax, steps=max_steps)
    e = float(atoms.get_potential_energy())
    write(RESULTS / f"{tag}_relaxed.xyz", atoms)  # final geom for r2SCAN/diff stretch
    wall = time.time() - t
    # diagnostics sidecar — best-effort, must never break the campaign
    tail = ""
    try:
        diag = _diagnostics(calc, opt, fmax, max_steps, wall, RESULTS / f"{tag}.log",
                            do_wf)
        (RESULTS / f"{tag}.diag.json").write_text(json.dumps(diag, indent=2))
        tail = f", scf_iter={diag.get('scf_n_iter', '?')}"
        if diag.get("bfgs_capped"):
            tail += "  ⚠STEP-CAP"
        if diag.get("is_metallic") is False:
            tail += "  ⚠NON-METALLIC"
        if diag.get("max_top_band_occ", 0.0) > 1e-3:
            tail += "  ⚠NBANDS-LOW"
    except Exception as ex:  # diagnostics are never allowed to fail the run
        tail = f"  [diag err: {str(ex)[:50]}]"
    print(f"    [{tag}] E={e:.4f} eV  ({opt.nsteps} steps, {wall:.0f}s{tail})",
          flush=True)
    return e


def run_pair(metal: str, ads: str, dbg: dict | None = None) -> dict:
    RESULTS.mkdir(exist_ok=True)
    try:  # provenance snapshot (best-effort; never blocks the run)
        (RESULTS / "runinfo.json").write_text(json.dumps(_provenance(), indent=2))
    except Exception:
        pass
    out_f = RESULTS / f"{metal}_{ads}.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {"metal": metal, "ads": ads}

    def save():
        out_f.write_text(json.dumps(res, indent=2))

    p = dbg or {}
    kw_slab = dict(kpts=p.get("kpts_slab"), device=p.get("device"),
                   ecut=p.get("ecut"), ecutrho=p.get("ecutrho"))
    fmax, steps = p.get("fmax", config.FMAX), p.get("max_steps", config.MAX_STEPS)

    # A fresh calculator per stage (the safe, known-good behavior the Pt-H run used).
    # An earlier optimization reused one calculator across the slab + sites to build
    # form factors once and warm-start each stage, but the adsorbate changes the atom
    # count vs the clean slab (16 → 17/18) and the USPP/PAW warm-start extrapolates a
    # per-atom becsum that cannot cross a natoms change — it crashes ("tensor 18 vs
    # 16"). A natoms-aware calc reuse (bail to a cold start on atom-count change, like
    # the orbital seed already does) is future work; correctness first here.

    # 1. clean slab
    if "e_clean" not in res:
        slab = read(STRUCT / f"slab_{metal}.xyz")
        res["e_clean"] = _relax(slab, config.make_calc(**kw_slab), fmax, steps,
                                f"{metal}_slab")
        save()

    # 2. adsorbate at each site
    res.setdefault("sites", {})
    for site in ("ontop", "bridge", "fcc", "hcp"):
        if site in res["sites"]:
            continue
        a = read(STRUCT / f"{metal}_{ads}_{site}.xyz")
        res["sites"][site] = _relax(a, config.make_calc(**kw_slab), fmax, steps,
                                    f"{metal}_{ads}_{site}")
        save()

    # 3. gas reference
    gas = GAS_FOR[ads]
    if "e_gas" not in res:
        g = read(STRUCT / f"gas_{gas}.xyz")
        kw_gas = dict(is_gas=True, device=p.get("device"),
                      ecut=p.get("ecut"), ecutrho=p.get("ecutrho"))
        res["e_gas"] = _relax(g, config.make_calc(**kw_gas), fmax, steps,
                              f"gas_{gas}", do_wf=False)  # Φ undefined for a molecule
        save()

    # 4. thermodynamics
    e_gas_ref = GAS_SCALE[ads] * res["e_gas"]
    summary = che.summarize(metal, ads, res["sites"], res["e_clean"], e_gas_ref)
    res["adsorption"] = {
        "best_site": summary.best_site, "e_ads": summary.e_ads,
        "dg_ads": summary.dg_ads, "per_site_dE": summary.per_site,
    }
    save()
    print(che.report(summary), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metal", choices=["Pt", "Au"])
    ap.add_argument("ads", choices=["H", "CO"])
    ap.add_argument("--debug", action="store_true", help="tiny CPU profile")
    args = ap.parse_args()
    run_pair(args.metal, args.ads, config.DEBUG if args.debug else None)
