# ChaosField spectral-radius kill test

Falsification test for the **ChaosField** rung of the response-kernel program
(`docs/plans/response-kernel-program.md`): an asynchronous, barrier-free
chaotic-relaxation SCF converges only if the fixed-point Jacobian M = K_Hxc·χ₀
satisfies ρ(|M|) < 1 *without* the nonlocal Kerker preconditioner (a local
real-space actor field cannot apply Kerker's long-range term). Measure ρ(M) on a
simple metal as a function of box size.

## Run

```bash
uv run python benchmarks/chaosfield_spectral_radius/spectral_radius.py
```

Al fcc primitive supercells + a Si contrast, PBE, Fermi-Dirac smearing, coarse
size-consistent k-meshes (conservative: coarse k under-counts the metal intraband
enhancement, biasing ρ downward). ρ(M) via `scf.soft_mode.dominant_screening_eigenvalue`
(largest-magnitude eigenvalue of M — the negative Hartree charge-sloshing mode),
soft-mode margin via `max_real_screening_eigenvalue`.

## Result (2026-08-09, asus) — blocker CONFIRMED

| cell | box L | ρ(M) | λ_max^real |
|---|---|---|---|
| Al, 1 atom | 2.86 Å | 0.42 | +0.001 |
| Al, 8 atoms | 5.73 Å | **2.27** | +0.001 |
| Si, 2 atoms | 3.84 Å | 1.06 | +0.002 |

- ρ(M) grows steeply with box length (0.42 → 2.27 for a 2× cell) and is already
  2.3× over the async threshold at 8 atoms; ρ(|M|) ≥ ρ(M) makes the true bound worse.
- The sub-1 value at 1 atom is the degenerate small-box limit (large G_min ⇒ weak
  Hartree charge mode), not a counterexample — it pins the crossing.
- λ_max^real ≈ +0.001 (margin 1 − λ ≈ 1): no CDW/magnetic instability; this is the
  plain screening wall, so the result is not a special-case artifact.

**Conclusion.** An unpreconditioned async SCF diverges on any physically-relevant
metal cell and worsens with size, by construction. ChaosField survives only in the
demoted *async-fine + globally-synced coarse-space* form, not as a by-construction
async cure. Part (b) of the plan's kill test (straggler/barrier-waste) is now moot:
reclaimable waste is irrelevant when the iteration does not converge.

## Caveats

- Two Al sizes (the 27-atom point was dropped — the 11-electron ONCV Al makes it
  ~300 electrons, too heavy for a quick test). Two points across a 2× box already
  establish the crossing + growth direction; more sizes would sharpen the exponent,
  not the verdict.
- Power-iteration residuals are loose (≈1–4e-2 at n_iter 30–40); the magnitudes
  (ρ ≈ 2.3 ≫ 1) are unambiguous well beyond that tolerance.
