"""run(): task dispatch composing the drivers and summary blocks."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gradwave.api._common import SPIN_XC_REGISTRY
from gradwave.api.dispersion import _apply_dispersion
from gradwave.api.elastic import run_elastic
from gradwave.api.eos import run_eos
from gradwave.api.phonons import run_phonons
from gradwave.api.relax import run_relax
from gradwave.api.scf import run_scf
from gradwave.api.summary import (
    _bands_extra,
    _base_summary,
    _cohp_summary_block,
    _error_estimate_block,
    _pdos_summary_block,
    _write_volumetric,
    build_summary,
)
from gradwave.api.system import build_system
from gradwave.inputs import Input

if TYPE_CHECKING:

    from gradwave.postscf.magnetism import MagneticReport

logger = logging.getLogger(__name__)


def run_magnetism(inp: Input, verbose: bool = True) -> MagneticReport:
    """Characterize the magnetism of the input system (task: magnetism). Builds a
    non-collinear XC from inp.xc, runs `characterize_magnetism`, and returns the
    MagneticReport."""
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.postscf.magnetism import characterize_magnetism

    system = build_system(inp)
    if inp.device != "cpu":
        system = system.to(inp.device)
    xc = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())
    m = inp.magnetism
    smtype = inp.smearing.type if inp.smearing.type != "none" else "gaussian"
    return characterize_magnetism(
        system, xc, exchange=m.exchange, ref_atom=m.ref_atom, lam=m.lam,
        delta=m.delta, seed_scale=m.seed_scale, smearing=smtype,
        width=inp.smearing.width, max_iter=inp.scf.max_iter, etol=inp.scf.etol,
        rhotol=inp.scf.rhotol, mixing_alpha=inp.scf.mixing.alpha, verbose=verbose)


# post-SCF tasks whose run() branch is a bare "run it, wrap the result" —
# collapsed into one data-driven branch below
_POSTSCF_RUNNERS = {"eos": run_eos, "elastic": run_elastic,
                    "phonons": run_phonons}


def run(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Execute inp.task and write <task>.json, <task>.out and (for SCF
    state) checkpoint.pt into inp.output_dir.

    Under ``inp.distributed`` (a torchrun launch), every rank computes the
    identical, correctly-reduced result (see ``scf.loop.scf``'s ``dist_ctx``
    handling), so file writing below is gated to rank 0 to avoid every rank
    racing to write the same files."""
    from gradwave.output import write_output
    from gradwave.runinfo import ProcessMeter, machine_snapshot, provenance_block

    if inp.distributed and inp.task not in ("scf", "bands", "relax", "eos"):
        raise NotImplementedError(
            f"distributed: true is wired for task: scf | bands | relax | eos "
            f"(got task: {inp.task!r}) — elastic/phonons/magnetism don't route "
            f"through the k-point-sharded SCF path yet (see "
            f"docs/manual/distributed.md)"
        )
    snap = machine_snapshot()
    meter = ProcessMeter()
    t0 = time.time()
    res = None
    _frames = None
    if inp.task == "scf":
        res = run_scf(inp, verbose=verbose)
        disp_block = _apply_dispersion(res, inp) if inp.dispersion.enabled else None
        summary = build_summary(res, inp, "scf", runtime_s=time.time() - t0)
        if disp_block is not None:
            summary["dispersion"] = disp_block
        if inp.error_estimate:
            summary["error_estimate"] = _error_estimate_block(res, inp)
        if inp.projections.enabled:
            summary["pdos"] = _pdos_summary_block(res, inp)
        if inp.projections.cohp.enabled:
            summary["cohp"] = _cohp_summary_block(res, inp)
    elif inp.task == "relax":
        relax, _atoms, _frames = run_relax(inp, verbose=verbose)
        summary = _base_summary(inp, "relax")
        summary["relax"] = relax
        summary["runtime_s"] = round(time.time() - t0, 2)
        # error estimate at the FINAL geometry: the calculator caches its
        # last converged SCF, so the estimate describes the relaxed state
        res_final = getattr(_atoms.calc, "last_result", None)
        if inp.error_estimate and res_final is not None:
            summary["error_estimate"] = _error_estimate_block(res_final, inp)
    elif inp.task == "bands":
        res = run_scf(inp, verbose=verbose)
        summary = build_summary(res, inp, "bands",
                                extra=_bands_extra(inp, res, verbose),
                                runtime_s=time.time() - t0)
        if inp.error_estimate:
            summary["error_estimate"] = _error_estimate_block(res, inp)
    elif inp.task == "magnetism":
        # no error_estimate block here: magnetism runs are spinor SCFs,
        # outside every estimator's coverage (it would always be
        # available: false)
        report = run_magnetism(inp, verbose=verbose)
        if verbose:
            print(report.summary())
        summary = _base_summary(inp, "magnetism")
        summary["magnetism"] = {
            "ordering": report.ordering,
            "total_moment_muB": round(report.total_moment, 4),
            "atomic_moments_muB": report.moment_magnitudes,
            "moment_vectors_muB": report.moment_vectors,
            "exchange_J_meV": None if report.exchange_J is None else
            {str(i): round(J * 1000, 3) for i, J in report.exchange_J.items()},
            "dmi_meV": None if report.dmi is None else
            {str(i): round(d * 1000, 4) for i, d in report.dmi.items()},
            "curie_temperature_mfa_K": None if report.curie_temperature_mfa is None
            else round(report.curie_temperature_mfa),
        }
        summary["runtime_s"] = round(time.time() - t0, 2)
    elif inp.task in _POSTSCF_RUNNERS:
        block = _POSTSCF_RUNNERS[inp.task](inp, verbose=verbose)
        summary = _base_summary(inp, inp.task)
        summary[inp.task] = block
        summary["runtime_s"] = round(time.time() - t0, 2)
    else:
        raise ValueError(
            f"unknown task {inp.task!r} "
            f"(scf | relax | bands | magnetism | eos | elastic | phonons)")

    if inp.distributed:
        from gradwave.distributed import current_rank, maybe_destroy_process_group

        # All collective communication finished inside run_scf's result gather;
        # from here on every rank works purely locally (file writing below is
        # rank-0-only). Tear the process group down now so a torchrun launch
        # exits cleanly instead of leaking it (#216). Guarded (no-op if never
        # initialized), so the single-process path never touches it.
        maybe_destroy_process_group()
        if current_rank() != 0:
            return summary

    outdir = Path(inp.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if res is not None and inp.output_checkpoint:
        from gradwave.checkpoint import save_checkpoint

        ck = save_checkpoint(res, outdir / "checkpoint.pt",
                             wavefunctions=inp.output_wavefunctions)
        outputs["checkpoint"] = ck.name
    if inp.task == "relax" and _frames:
        from ase.io import write as ase_write

        ase_write(str(outdir / "relax.xyz"), _frames, format="extxyz")
        outputs["trajectory"] = "relax.xyz"
    if res is not None and inp.output_volumetric.any():
        outputs.update(_write_volumetric(res, inp.output_volumetric, outdir, verbose))
    # SCF flight-recorder sidecar: the full per-iteration trace, opted in via
    # scf.trace. The compact diagnostics ride in the JSON regardless; this is the
    # heavy per-iteration corpus (schema-versioned scf_trace.json). Rank-0 only,
    # inheriting the gate above.
    _rec = getattr(res, "recorder", None) if res is not None else None
    if inp.scf.trace and _rec is not None and getattr(_rec, "iters", None):
        (outdir / "scf_trace.json").write_text(json.dumps(_rec.to_trace_dict(), indent=1))
        outputs["scf_trace"] = "scf_trace.json"
    summary["provenance"] = provenance_block(snap, meter)
    summary["outputs"] = {**outputs, "json": f"{inp.task}.json",
                          "report": f"{inp.task}.out"}
    (outdir / f"{inp.task}.json").write_text(json.dumps(summary, indent=1))
    write_output(summary, outdir / f"{inp.task}.out")
    if verbose:
        print(f"wrote {outdir / inp.task}.json / .out"
              + (" / checkpoint.pt" if "checkpoint" in outputs else ""))
    return summary
