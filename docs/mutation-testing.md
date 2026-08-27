# Mutation testing

Mutation testing asks the one question coverage can't: *do the tests actually
kill bugs, or just run the code?* It injects one small bug at a time (a `+`→`-`
swap, a `<`→`<=` flip, a constant `± 1`, a dropped sign, an `and`↔`or`) and
checks whether some test **fails** (the mutant is *killed*) or all pass (the
mutant *survives* — a spot no test constrains).

## How to run it

`scripts/mutation_probe.py` does this in place (mutmut 3.x's copy-to-`mutants/`
sandbox is incompatible with our editable src-layout install). It is **on-demand
and belongs on asus, never in per-PR CI** — it re-runs the covering tests once
per mutant, which is inherently many-test-runs expensive.

```bash
# one module + its fast covering tests (run on asus)
make mutation TARGET=src/gradwave/core/energies/hartree.py \
              TESTS="tests/unit/test_hartree.py tests/unit/test_esm_hartree.py"
```

Survivors print as `file:line  <mutation>  — <source line>` — each an actionable
"add or extend a test that pins this."

## Reading the output — the important part

**Kill-rate is a poor proxy for test quality on this codebase.** A 2026-08 probe
of four numerical-core modules (with only their *fast unit* covering tests, so
the SCF/integration tier that also exercises them was excluded) measured:

| module | unit-test kill-rate |
|---|---|
| `core/occupations.py` | 82% |
| `core/energies/ewald.py` | 62% |
| `core/energies/hartree.py` | 29% |
| `scf/mixing.py` | 28% |

Those low numbers are **mostly not test holes** — they are dominated by
**equivalent mutants**: mutations that no *correct* test should ever kill, because
they don't change the answer. On numerical / iterative DFT code these are
everywhere, and recognizing the categories is the whole skill:

- **Splitting / convergence parameters the physics is invariant to.**
  `ewald.py:_ACC = 8.0` is the real/reciprocal Ewald split — the total energy is
  invariant to it *by construction*. `mixing.py:alpha = 0.7`, `history = 8`
  change the SCF's convergence *path*, not its fixed point. Pinning these values
  in a test would be a **bad** test (over-fitting to an incidental default).
- **Safety margins and iteration caps.** `occupations.py`'s bisection bracket
  `lo = eigs.min() - 10*width - 1.0` and `max_iter = 200` still work when
  perturbed — the root is found either way.
- **Symmetric-grid no-ops.** The `inv_g2` symmetrization in `hartree_potential_r`
  is a no-op on an orthogonal cell, so mutating it survives any cubic-cell test.

The **signal** — the small subset worth acting on — is a mutation to a
**load-bearing physical constant or operation** that a correctness test *should*
pin but doesn't. Distinguish it by asking: *would flipping this change a
physically-meaningful result?* If yes and it survived, that's a real gap.

## Worked example (the gap this doc's PR closed)

The probe flagged `hartree.py:70` (`v_g = 4.0 * math.pi * E2 * rho_g * ...`, the
Hartree prefactor) and `:65-67` (the Nyquist-plane symmetrization) as survivors.
Real signal: `hartree_energy` and `hartree_potential_g` were pinned to ~1e-10,
but `hartree_potential_r` (the real-space symmetrized path) was exercised only by
the integration tier — so a 25% prefactor error survived the fast suite.
`tests/unit/test_hartree.py::test_real_space_potential_matches_full_g_path_triclinic`
now pins it against the full complex-FFT reference **on a non-orthogonal cell**
(where the symmetrization is load-bearing), killing both the prefactor and the
symmetrization mutants.

## Guidance

- **Don't mass-produce tests from a survivor list.** Most survivors are
  equivalent mutants; a test that pins one is worse than no test.
- **For a true measure, include the integration tier** in `TESTS`. The fast-unit
  numbers above understate coverage because every SCF exercises Ewald, Hartree,
  occupations, and mixing.
- **Best-fit modules** are stateless numerical kernels with closed-form or
  metamorphic oracles (energies, forces, transforms). **Worst-fit** are
  self-correcting iterative algorithms (mixing, optimizers), where a high
  equivalent-mutant rate makes the score nearly meaningless.
