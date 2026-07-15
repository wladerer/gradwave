"""Command-line interface.

    gradwave input.yaml                # run, outputs to the YAML's output.dir
    gradwave input.yaml -o results/    # override the output directory
    gradwave plot out/scf.json         # convergence / bands / DOS figure

The explicit `gradwave run input.yaml` form still works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gradwave import __version__

_COMMANDS = {"run", "plot"}


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

    p_plot = sub.add_parser(
        "plot", help="plot a result JSON (scf convergence, bands, or dos)")
    p_plot.add_argument("result", help="path to <task>.json")
    p_plot.add_argument("-o", "--output", metavar="FILE",
                        help="figure file (default: alongside the JSON)")
    p_plot.add_argument("--kind", choices=("auto", "scf", "bands", "dos"),
                        default="auto")
    p_plot.add_argument("--width", type=float, default=0.1,
                        help="DOS broadening [eV]")
    return parser


def _cmd_run(args) -> int:
    import dataclasses

    from gradwave.api import run
    from gradwave.inputs import load_input

    inp = load_input(args.input)
    if args.output:
        inp = dataclasses.replace(inp, output_dir=Path(args.output))
    summary = run(inp, verbose=not args.quiet)
    scf = summary.get("scf")
    if scf is not None:
        e = scf["energies_eV"]
        print(f"{'converged' if scf['converged'] else 'NOT CONVERGED'}: "
              f"F = {e['free_energy']:.8f} eV ({scf['n_iter']} iterations)")
        return 0 if scf["converged"] else 1
    relax = summary.get("relax")
    if relax is not None:
        print(f"{'converged' if relax['converged'] else 'NOT CONVERGED'}: "
              f"E = {relax['energy_eV']:.8f} eV, fmax = "
              f"{relax['fmax_eV_ang']:.4f} eV/Å ({relax['n_steps']} steps)")
        return 0 if relax["converged"] else 1
    return 0


def _cmd_plot(args) -> int:
    from gradwave import analysis

    summary = analysis.load(args.result)
    kind = args.kind
    if kind == "auto":
        kind = "bands" if "bands" in summary else "scf"
    out = args.output or str(
        Path(args.result).with_suffix("")) + f".{kind}.png"
    if kind == "scf":
        analysis.plot_scf(summary, path=out)
    elif kind == "bands":
        analysis.plot_bands(summary, path=out)
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
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "plot":
        return _cmd_plot(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
