# High-level API

The Layer C surface. These functions and classes take an input specification,
run a task, and return results ready for analysis or ASE. For the file schema
and the analysis helpers in prose, see [Inputs and outputs](../io.md).

## Task runner

`gradwave.api` mirrors the YAML input. `run` executes a task and writes the
result files; the lower functions return the native result objects if you want
to keep them in memory.

::: gradwave.api.run

::: gradwave.api.run_scf

::: gradwave.api.run_relax

::: gradwave.api.build_system

::: gradwave.api.build_summary

## Input schema

`load_input` parses a YAML file into a frozen `Input`. The nested dataclasses
hold the validated defaults for each block of the schema.

::: gradwave.inputs.load_input

::: gradwave.inputs.Input

::: gradwave.inputs.SCFParams

::: gradwave.inputs.MixingParams

::: gradwave.inputs.SmearingParams

::: gradwave.inputs.KPointsParams

::: gradwave.inputs.RelaxParams

::: gradwave.inputs.BandsParams

## ASE calculator

`GradWave` is a standard ASE `Calculator`. Attach it to an `Atoms` object and
any ASE optimizer, filter, or molecular-dynamics driver works. It reuses the
converged density between ionic steps automatically. The formalism is detected
from the pseudopotentials, and energy, forces, and stress are all exact
autograd quantities.

::: gradwave.calculator.GradWave

## Checkpoints

A checkpoint stores the converged density (and becsum for USPP/PAW) so an SCF
can restart. Wavefunctions are excluded by default.

::: gradwave.checkpoint.save_checkpoint

::: gradwave.checkpoint.load_checkpoint

::: gradwave.checkpoint.as_start_from

## Analysis

These helpers turn a result file (or an in-memory summary dict) into tidy
pandas frames and matplotlib figures. `gradwave plot` wraps the plotting
functions.

::: gradwave.analysis.load

::: gradwave.analysis.scf_frame

::: gradwave.analysis.eigenvalues_frame

::: gradwave.analysis.bands_frame

::: gradwave.analysis.dos_frame

::: gradwave.analysis.plot_scf

::: gradwave.analysis.plot_bands

::: gradwave.analysis.plot_dos

## Command line

::: gradwave.cli.main
