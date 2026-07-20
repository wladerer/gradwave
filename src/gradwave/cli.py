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

from gradwave import __version__

_COMMANDS = {"init", "run", "validate", "plot"}


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="gradwave", description="Differentiable plane-wave DFT")
    parser.add_argument("--version", action="version",
                        version=f"gradwave {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a calculation from a YAML input")
    p_run.add_argument("input", help="path to input.yaml")
    p_run.add_argument("-o", "--output", metavar="DIR",
                       help="output directory (overrides output.dir)")
    p_run.add_argument("-q", "--quiet", action="store_true")

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
                        choices=("auto", "scf", "bands", "dos", "pdos"),
                        default="auto")
    p_plot.add_argument("--width", type=float, default=0.1,
                        help="DOS broadening [eV]")
    return parser


def _cmd_init(args) -> int:
    from gradwave import templates

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


def _load_checked(path):
    """Load an input, turning the schema errors into a one-line message and a
    non-zero exit rather than a traceback. Returns (Input, None) or (None, rc)."""
    from gradwave.inputs import InputError, load_input

    try:
        return load_input(path), None
    except (InputError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return None, 1


def _cmd_validate(args) -> int:
    import numpy as np

    inp, rc = _load_checked(args.input)
    if inp is None:
        return rc
    a = inp.atoms
    formula = a.get_chemical_formula()
    print(f"ok: {args.input}")
    print(f"  task        {inp.task}")
    print(f"  structure   {formula}  ({len(a)} atoms)")
    print(f"  cell [Å]    {np.array2string(a.cell.array, precision=4)}")
    print(f"  ecut [eV]   {inp.ecut:g}"
          + (f"   ecutrho {inp.ecutrho:g}" if inp.ecutrho else ""))
    print(f"  xc          {inp.xc}")
    print(f"  kpoints     mesh {list(inp.kpoints.mesh)} shift "
          f"{list(inp.kpoints.shift)}")
    print(f"  smearing    {inp.smearing.type}"
          + (f" ({inp.smearing.width} eV)" if inp.smearing.type != "none" else ""))
    print(f"  nspin       {inp.nspin}"
          + ("  noncollinear" if inp.noncollinear else ""))
    print(f"  pseudos     {inp.pseudo_map}")
    print(f"  device      {inp.device}")
    print(f"  output_dir  {inp.output_dir}")
    return 0


def _cmd_run(args) -> int:
    import dataclasses

    from gradwave.api import run

    inp, rc = _load_checked(args.input)
    if inp is None:
        return rc
    if args.output:
        inp = dataclasses.replace(inp, output_dir=Path(args.output))
    summary = run(inp, verbose=inp.verbose and not args.quiet)
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
    return 0


def _cmd_plot(args) -> int:
    from gradwave import analysis

    summary = analysis.load(args.result)
    kind = args.kind
    if kind == "auto":
        if "bands" in summary:
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
    else:
        analysis.plot_dos(summary, path=out, width=args.width)
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
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
