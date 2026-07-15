# Reference

The terse command-line and entry-point tables. The [Inputs and outputs](io.md) page
covers the file schema, checkpoints, and analysis in full, and the
[API reference](api/index.md) documents every public function and class from its
docstrings.

## CLI

    gradwave input.yaml                 # run, outputs to the YAML's output.dir
    gradwave input.yaml -o results/     # override the output directory
    gradwave plot out/scf.json          # scf convergence, bands, or dos figure
    gradwave run input.yaml             # explicit run form, still valid

## Output files

| file | contents |
|---|---|
| `<task>.json` | machine summary, the parsing target |
| `<task>.out` | human report |
| `checkpoint.pt` | restartable SCF state, density and becsum by default |

`<task>` is `scf`, `relax`, or `bands`. See [Inputs and outputs](io.md) for the JSON
key list, the checkpoint API, and the analysis helpers.

## Key entry points

| symbol | purpose |
|---|---|
| `scf`, `setup_system` (`scf.loop`) | norm-conserving SCF |
| `scf_uspp`, `setup_uspp` (`scf.uspp`) | USPP/PAW SCF |
| `GradWave` (`calculator`) | ASE calculator, energy, forces, stress |
| `LearnableX`, `LearnableSpinX` (`core.xc.learnable`) | learnable exchange |
| `energy_param_grads` (`core.xc.learnable`) | free dE/dθ at convergence |
| `uspp_density_loss_param_grads` (`postscf.uspp_implicit`) | density-loss adjoint |

The [API reference](api/index.md) expands each of these with full signatures,
grouped by subsystem.
