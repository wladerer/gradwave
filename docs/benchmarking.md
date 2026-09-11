# Benchmarking against Quantum ESPRESSO — apples-to-apples

Timing comparisons against QE (or any other code) are only meaningful when both
codes solve the *same* problem. Two settings dominate the wall time and are easy
to get wrong, both of which silently inflate gradwave's measured gap:

## 1. IBZ reduction (symmetry) — the k-point count

QE reduces a Monkhorst–Pack mesh to the irreducible Brillouin zone using the
full crystal space group *by default*. gradwave does the same on its production
path (`Input.symmetry = True`, wired through `api.run`/`run_scf` →
`api.build_system` → `setup_system(use_symmetry=True)` →
`symmetry.reduce_mesh`). Density is symmetrized every SCF step
(`RhoSymmetrizer`) and forces via `symmetry.symmetrize_forces`, so the reduced
run is correct, not an approximation.

The trap is `setup_system(use_symmetry=False)`, which is the default of that
low-level entry point. With it, k-reduction falls back to *time-reversal only*
(`kpoints.monkhorst_pack`), which folds k ≡ −k but applies no point-group
operations. On an **even** mesh that does almost nothing: every mesh point has
half-integer components and is its own −k mod G, so an even mesh is not reduced
at all.

Measured on the 64-atom Si diamond supercell (2×2×2 of the conventional cell,
Fd-3m, full 48-rotation point group):

| mesh | time-reversal only (`use_symmetry=False`) | full space group (`use_symmetry=True`, = QE) |
|---|---|---|
| 2×2×2 | 8 k-points | **4** k-points |
| 4×4×4 | 36 k-points | 10 k-points |

A benchmark that calls `setup_system` without `use_symmetry=True` therefore
solves 8 k-points where QE (and gradwave's own `api.run`) solve 4 — a factor of
~2 in wall time attributed to gradwave that is purely a harness misconfiguration.
The same effect gives 36 vs 10 on the 4×4×4 mesh; a run reporting "nk=36" from a
4³ mesh (or "nk=8" from a 2³ mesh) is on the time-reversal-only path, not the
IBZ.

**Recipe:** benchmark through `api.run` (symmetry on by default), or if calling
`setup_system` directly, pass `use_symmetry=True`. `benchmarks/bench_matrix.py`
does this for every case.

## 2. Band count — the `nbands` heuristic vs the insulator count

`scf.setup_common.default_nbands` adds a 20% buffer over the occupied count (min
+4). That buffer is needed for metals (bands cross the Fermi level and the
smearing must see empty states), but an insulator run with `smearing="none"`
does not need it. For Si-64 the default gives 154 bands (256 valence e⁻ → 128
occupied → 128×1.2 = 154); QE, run as an insulator, solves exactly 128.

The band count is a first-class knob: set `Input.nbands` (YAML `nbands:`) or the
`nbands=` argument of `setup_system`/`scf`. For an apples-to-apples insulator
comparison, pass `nbands = n_valence_electrons / 2`.

The default heuristic is intentionally left metal-safe; do not lower it globally.
Set the explicit knob for insulator benchmarks instead.

## Corrected gap accounting (Si-64, single process)

The earlier reported Si-64 gap against QE compared a gradwave run on the
time-reversal-only path (nk=8) with the default 154 bands against QE's nk=4 /
128 bands — i.e. gradwave was solving roughly `8/4 × 154/128 ≈ 2.4×` more work
than QE. Correcting both settings (symmetry on → nk=4, `nbands=128`) is the
apples-to-apples configuration; the residual gap is then the genuine
per-k-point, per-band cost difference, not a k-mesh/band-count artifact. Always
report the QE-comparable config (nk and nbands printed by the harness) alongside
the wall times so the comparison is auditable.
