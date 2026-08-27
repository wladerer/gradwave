"""Command-line interface.

    gradwave init relax -o input.yaml  # write a starter input for a task
    gradwave input.yaml                # run, outputs to the YAML's output.dir
    gradwave input.yaml -o results/    # override the output directory
    gradwave validate input.yaml       # parse and check, run nothing
    gradwave plot out/scf.json         # convergence / bands / DOS figure

The explicit `gradwave run input.yaml` form still works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gradwave._version import __version__

if TYPE_CHECKING:
    from gradwave.inputs import Input

_COMMANDS = {"init", "run", "validate", "plot"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradwave", description="Differentiable plane-wave DFT")
    parser.add_argument("--version", action="version",
                        version=f"gradwave {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a calculation from a YAML input")
    p_run.add_argument("input", help="path to input.yaml")
    p_run.add_argument("-o", "--output", metavar="DIR",
                       help="output directory (overrides output.dir)")
    p_run.add_argument("-r", "--restart", metavar="PATH",
                       help="checkpoint.pt to warm-start from (overrides the "
                            "restart: key in the YAML)")
    p_run.add_argument("-q", "--quiet", action="store_true")
    p_run.add_argument("--log-level", metavar="LEVEL",
                       choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                       help="stream gradwave.* diagnostics to stderr at this "
                            "level (DEBUG shows SCF/solver/mixer branch tracing)")

    p_init = sub.add_parser(
        "init", help="write a starter input for a task (relax, bands, ...)")
    p_init.add_argument("template", nargs="?",
                        help="template name; omit to list the available ones")
    p_init.add_argument("-o", "--output", metavar="FILE",
                        help="write here instead of stdout")
    p_init.add_argument("--force", action="store_true",
                        help="overwrite an existing --output file")

    p_val = sub.add_parser(
        "validate", help="parse and check an input without running it")
    p_val.add_argument("input", help="path to input.yaml")

    p_plot = sub.add_parser(
        "plot", help="plot a result JSON (scf convergence, bands, or dos)")
    p_plot.add_argument("result", help="path to <task>.json")
    p_plot.add_argument("-o", "--output", metavar="FILE",
                        help="figure file (default: alongside the JSON)")
    p_plot.add_argument("--kind",
                        choices=("auto", "scf", "bands", "dos", "pdos", "cohp",
                                 "phonons", "eos", "elastic"),
                        default="auto")
    p_plot.add_argument("--width", type=float, default=0.1,
                        help="DOS broadening [eV]")
    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    from gradwave.io import templates

    if not args.template:
        print("available templates (gradwave init <name>):")
        for name, desc in templates.summaries().items():
            print(f"  {name:14s} {desc}")
        return 0
    try:
        text = templates.render(args.template)
    except KeyError:
        print(f"error: unknown template {args.template!r}; choices: "
              f"{', '.join(templates.names())}", file=sys.stderr)
        return 1
    if args.output:
        out = Path(args.output)
        if out.exists() and not args.force:
            print(f"error: {out} exists (use --force to overwrite)",
                  file=sys.stderr)
            return 1
        out.write_text(text)
        print(f"wrote {out}  —  edit the structure and pseudopotentials, then "
              f"`gradwave validate {out}`")
        return 0
    sys.stdout.write(text)
    return 0


def _load_checked(path: str) -> tuple[Input | None, int | None]:
    """Load an input, turning the schema errors into a one-line message and a
    non-zero exit rather than a traceback. Returns (Input, None) or (None, rc)."""
    from gradwave.inputs import InputError, load_input

    try:
        return load_input(path), None
    except (InputError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return None, 1


def _banner() -> str:
    """The startup wordmark. A ``╱╲`` wave motif — a nod to the plane-wave basis
    — plus the version, so a bare `gradwave run` gives immediate feedback while
    the (silent, slow) pseudopotential parse and basis build get under way."""
    return (f"\n  ╱╲╱╲╱╲   gradwave  v{__version__}\n"
            f"  ╲╱╲╱╲╱   differentiable plane-wave DFT\n")


def _flapw_summary_lines(inp: Input) -> list[str]:
    """The at-a-glance block for the all-electron FLAPW / EFG tasks, whose
    knobs live in the ``flapw`` block rather than the plane-wave ``ecut`` and
    pseudopotentials."""
    import numpy as np

    a = inp.atoms
    fp = inp.flapw
    task = inp.task + (f"  ({inp.nmr.task})" if inp.task == "nmr" else "")
    lines = [
        f"  task        {task}",
        f"  structure   {a.get_chemical_formula()}  ({len(a)} atoms)",
        f"  cell [Å]    {np.array2string(a.cell.array, precision=4)}",
        f"  flapw ecut  {fp.ecut:g}   lmax {fp.lmax}   "
        + ("full-potential" if fp.fullpot else "muffin-tin"),
        f"  mt radii Å  {dict(fp.radii)}",
        f"  kpoints     mesh {list(inp.kpoints.mesh)}",
        f"  smearing    {fp.smearing:g} eV",
        f"  device      {inp.device}",
        f"  output_dir  {inp.output_dir}",
    ]
    if inp.task == "nmr" and inp.nmr.task == "efg" and inp.nmr.isotopes:
        lines.append(f"  isotopes    {inp.nmr.isotopes}")
    return lines


def _summary_lines(inp: Input) -> list[str]:
    """The input at a glance — shared by `validate` and the `run` startup block."""
    if inp.task == "flapw" or (inp.task == "nmr" and inp.nmr.task == "efg"):
        return _flapw_summary_lines(inp)
    import numpy as np

    a = inp.atoms
    task = inp.task
    if task == "relax":
        task += "  (variable cell)" if inp.relax.cell else "  (positions only)"
    return [
        f"  task        {task}",
        f"  structure   {a.get_chemical_formula()}  ({len(a)} atoms)",
        f"  cell [Å]    {np.array2string(a.cell.array, precision=4)}",
        f"  ecut [eV]   {inp.ecut:g}"
        + (f"   ecutrho {inp.ecutrho:g}" if inp.ecutrho else ""),
        f"  xc          {inp.xc}",
        f"  kpoints     mesh {list(inp.kpoints.mesh)} shift "
        f"{list(inp.kpoints.shift)}",
        f"  smearing    {inp.smearing.type}"
        + (f" ({inp.smearing.width} eV)" if inp.smearing.type != "none" else ""),
        f"  nspin       {inp.nspin}"
        + ("  noncollinear" if inp.noncollinear else ""),
        f"  pseudos     {inp.pseudo_map}",
        f"  device      {inp.device}",
        f"  output_dir  {inp.output_dir}",
    ]


def _cmd_validate(args: argparse.Namespace) -> int:
    inp, rc = _load_checked(args.input)
    if rc is not None:
        return rc
    assert inp is not None
    print(f"ok: {args.input}")
    for line in _summary_lines(inp):
        print(line)
    return 0


def _print_nmr(nmr: dict[str, Any]) -> int:
    """Render the per-site NMR table to stdout: an EFG (V_zz / η / C_Q) or a
    magnetic-shielding (σ_iso / Δσ / η) block, mirroring the ``nmr.out`` report."""
    if nmr.get("observable") == "shielding":
        kind = ("absolute GIPAW" if nmr.get("method") == "gipaw_absolute"
                else "bare valence")
        print(f"NMR magnetic shielding ({kind}, plane-wave GIPAW): "
              f"{nmr['n_sites']} sites")
        for s in nmr["sites"]:
            line = (f"  site {s['site']:>3d} {(s.get('species') or '?'):>3s}  "
                    f"σ_iso = {s['sigma_iso_ppm']:10.3f} ppm  "
                    f"Δσ = {s['sigma_aniso_ppm']:10.3f} ppm  "
                    f"η = {s['sigma_eta']:.3f}")
            if "delta_iso_ppm" in s:
                line += f"  δ_iso = {s['delta_iso_ppm']:10.3f} ppm"
            print(line)
        return 0
    conv = bool(nmr.get("converged"))
    print(f"{'converged' if conv else 'NOT CONVERGED'}: electric field "
          f"gradient, {nmr['n_sites']} sites")
    for s in nmr["sites"]:
        line = (f"  site {s['site']:>3d} {s['species']:>3s}  "
                f"V_zz = {s['V_zz_eV_ang2']:11.4f} eV/Å²  η = {s['eta']:.4f}")
        if "C_Q_MHz" in s:
            line += f"  C_Q = {s['C_Q_MHz']:9.4f} MHz ({s['isotope']})"
        print(line)
    return 0 if conv else 1


def _cmd_run(args: argparse.Namespace) -> int:
    import dataclasses

    from gradwave import configure_logging

    if args.log_level:
        configure_logging(args.log_level)
    inp, rc = _load_checked(args.input)
    if rc is not None:
        return rc
    assert inp is not None
    if args.output:
        inp = dataclasses.replace(inp, output_dir=Path(args.output))
    if args.restart:
        inp = dataclasses.replace(inp, restart=Path(args.restart))
    if not args.quiet:
        # Print before importing api (torch) and before the silent build, so the
        # user sees what they launched instead of a blank terminal. flush=True:
        # the following build can run for seconds with no further output.
        print(_banner())
        for line in _summary_lines(inp):
            print(line)
        print("\n  preparing system — parsing pseudopotentials, building the "
              "plane-wave basis and form factors …\n", flush=True)
    from gradwave.api import run

    summary = run(inp, verbose=inp.verbose and not args.quiet)
    if not args.quiet:
        # timing + peak memory footer (VASP-style), from the provenance block the
        # run always records — surfaced here rather than left in the JSON only.
        proc = (summary.get("provenance") or {}).get("process") or {}
        if proc:
            foot = f"\n  {proc.get('wall_s', 0):.1f}s wall"
            if proc.get("cpu_s") is not None:
                foot += f" · {proc['cpu_s']:.0f}s cpu"
            if proc.get("effective_threads"):
                foot += f" ({proc['effective_threads']:.1f}× threads)"
            if proc.get("peak_rss_gb") is not None:
                foot += f" · peak RSS {proc['peak_rss_gb']:.2f} GB"
            if "cuda_peak_alloc_gb" in proc:
                foot += f" · CUDA peak {proc['cuda_peak_alloc_gb']:.2f} GB"
            print(foot)
    scf = summary.get("scf")
    if scf is not None:
        e = scf["energies_eV"]
        print(f"{'converged' if scf['converged'] else 'NOT CONVERGED'}: "
              f"F = {e['free_energy']:.8f} eV ({scf['n_iter']} iterations)")
        return 0 if scf["converged"] else 1
    relax = summary.get("relax")
    if relax is not None:
        # A relax that reaches the ionic-step limit still yields a valid
        # trajectory and a usable last geometry, so exit 0 signals the run
        # executed. Convergence is a quality flag carried by relax.converged.
        print(f"{'converged' if relax['converged'] else 'NOT CONVERGED'}: "
              f"E = {relax['energy_eV']:.8f} eV, fmax = "
              f"{relax['fmax_eV_ang']:.4f} eV/Å ({relax['n_steps']} steps)")
        return 0
    eos = summary.get("eos")
    if eos is not None:
        print(f"V0 = {eos['v0_ang3_per_atom']:.4f} Å³/atom, "
              f"B0 = {eos['b0_GPa']:.2f} GPa, B0' = {eos['b0_prime']:.3f}")
        return 0 if eos.get("all_converged", True) else 1
    elastic = summary.get("elastic")
    if elastic is not None:
        print(f"K = {elastic['bulk_modulus_GPa']['hill']:.1f} GPa, "
              f"G = {elastic['shear_modulus_GPa']['hill']:.1f} GPa, "
              f"E = {elastic['young_modulus_GPa']:.1f} GPa, "
              f"ν = {elastic['poisson_ratio']:.3f} "
              f"({'stable' if elastic['mechanically_stable'] else 'UNSTABLE'})")
        return 0 if elastic.get("all_converged", True) else 1
    phonons = summary.get("phonons")
    if phonons is not None:
        fmin = phonons["min_frequency_cm1"]
        print(f"phonons: {tuple(phonons['supercell'])} supercell, "
              f"min ω = {fmin:.1f} cm⁻¹ "
              f"({'all real' if fmin > -1.0 else 'IMAGINARY modes'})")
        return 0 if fmin > -1.0 else 1
    flapw = summary.get("flapw")
    if flapw is not None:
        span = flapw.get("band_span_eV")
        print(f"{'converged' if flapw['converged'] else 'NOT CONVERGED'}: "
              f"FLAPW SCF, {flapw.get('n_bands')} bands"
              + (f", band span {span:.4f} eV" if span is not None else ""))
        return 0 if flapw["converged"] else 1
    nmr = summary.get("nmr")
    if nmr is not None:
        return _print_nmr(nmr)
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from gradwave.io import analysis

    summary = analysis.load(args.result)
    kind = args.kind
    if kind == "auto":
        if "phonons" in summary:
            kind = "phonons"
        elif "eos" in summary:
            kind = "eos"
        elif "elastic" in summary:
            kind = "elastic"
        elif "bands" in summary:
            kind = "bands"
        elif "pdos" in summary:
            kind = "pdos"
        else:
            kind = "scf"
    out = args.output or str(
        Path(args.result).with_suffix("")) + f".{kind}.png"
    if kind == "scf":
        analysis.plot_scf(summary, path=out)
    elif kind == "bands":
        analysis.plot_bands(summary, path=out)
    elif kind == "pdos":
        if analysis._is_noncollinear_block(summary.get("pdos")):
            analysis.plot_spin_texture(summary, path=out)
        else:
            analysis.plot_pdos(summary, path=out)
    elif kind == "cohp":
        analysis.plot_cohp(summary, path=out)
    elif kind == "phonons":
        analysis.plot_phonons(summary, path=out)
    elif kind == "eos":
        analysis.plot_eos(summary, path=out)
    elif kind == "elastic":
        analysis.plot_elastic(summary, path=out)
    else:
        analysis.plot_dos(summary, path=out, width=args.width)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # bare `gradwave input.yaml [...]` is a run
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "run")
    args = _build_parser().parse_args(argv)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "plot":
        return _cmd_plot(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
