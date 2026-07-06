"""Command-line interface:  gradwave run input.yaml"""

from __future__ import annotations

import argparse
import sys

from gradwave import __version__
from gradwave.api import run
from gradwave.inputs import load_input


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gradwave", description="Differentiable plane-wave DFT")
    parser.add_argument("--version", action="version", version=f"gradwave {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run a calculation from a YAML input file")
    p_run.add_argument("input", help="path to input.yaml")
    p_run.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "run":
        inp = load_input(args.input)
        summary = run(inp, verbose=not args.quiet)
        e = summary["free_energy_eV"]
        print(f"{'converged' if summary['converged'] else 'NOT CONVERGED'}: "
              f"F = {e:.8f} eV ({summary['n_iter']} iterations)")
        return 0 if summary["converged"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
