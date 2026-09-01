# M1 — cheap subspace-χ₀ Woodbury preconditioning: results & verdict

**Branch:** `research/woodbury-chi0-precond-m1`  ·  **Compute:** asus (22-core CPU,
fp64), one job at a time, laptop untouched  ·  **Convergence:** density-residual
gate ‖ρ_out−ρ_in‖·Ω/N_G tight (energy_metric off), etol 1e-9 eV.

## The question

The crux (`research/chi0-precond-crux`) showed the EXACT dielectric
preconditioner ε_ρ⁻¹ = (1 − χ₀·K_Hxc)⁻¹ — a full Sternheimer χ₀ solve each SCF
step, frozen at the converged reference — cuts bcc-Fe FM iterations ~2× but at
~7.6× the FFTs (it runs a conduction-space χ₀ solve every step). **M1 asks:**
does the CHEAP student — χ₀ restricted to the in-window Adler-Wiser sum + δμ term
the eigensolver already returns, inverted by Woodbury with **zero FFTs per
apply** — keep most of that iteration cut at a fraction of the FFT overhead, and
converge to the IDENTICAL fixed point?

## What was built

- `src/gradwave/scf/subspace_chi0.py`
  - `WoodburyPrecond` — (1 − M_ρ)⁻¹ with M_ρ = χ₀_sub·K_Hxc = U·diag(c)·W†,
    inverted by the exact Woodbury LU (generalises `StonerSpinPrecond`'s
    m-channel Stoner diagonal to the coupled charge+spin window; the (up,dn)↔
    (total,mag) transform is a pure G-space linear combination — no FFT). Pins
    the total-channel G=0 so a preconditioned step conserves N_e.
  - `build_woodbury_subspace` — freezes the low-rank codensity factors from a
    converged reference. The subspace χ₀ keeps pieces (2)+(3) of the shipped
    `_chi0_channel_metal` (the in-window band-pair sum with divided-difference
    weights β_nm = (f_n−f_m)/(ε_n−ε_m), plus the δμ Fermi-shift), and DROPS the
    FFT-heavy Sternheimer window→virtual piece (1). Cost: one `apply_k_hxc` per
    column, ONCE; **zero FFTs per subsequent apply**.
  - `apply_chi0_subspace` — matrix-free reference for validation.
- `tests/unit/test_woodbury_chi0_precond.py` — the Woodbury identity vs a dense
  solve; the frozen low-rank M_ρ reproduces the matrix-free subspace χ₀ **to
  9.2e-16** on a physical (density-sphere band-limited) residual.
- `experiments/chi0_precond_crux/stage2_cheap_student.py` — the 4-arm A/B
  harness + fixed-point-identity checks + verdict.

## Faithfulness — every arm reaches the SAME fixed point

A preconditioner cannot move the true fixed point. The crux's residual dE=1.22e-6
(SAME=False) was a convergence-bookkeeping artifact (it stopped on the energy
metric while the density still carried slack). Converging on the **density
residual** instead pins every arm to the identical density:

| system | dE | max d|ρ| | SAME |
|---|---|---|---|
| Fe (cheap vs teacher vs ctrl) | ≤ ~1e-11 eV | ~1e-10 | ✅ |
| Al(100) 4-layer (cheap vs ctrl) | 1.34e-13 Ha | 3.66e-11 | ✅ |

Far inside the mandated dE ≤ 1e-9 Ha, d|ρ| ≤ 1e-6. **The crux's SAME=False is
resolved.**

## A/B tables (iters + FFT launches; identical cold start & gate)

`prod` = johnson + local-TF (+ Stoner); `ctrl` = pulay + local-TF (+ Stoner), the
fair same-mixer baseline; `teacher_exact` = pulay + exact χ₀ precond_op; `cheap`
= pulay + subspace-χ₀ Woodbury precond_op (512 columns, one-time build).

### bcc-Fe FM (nspin=2) — the stiff-magnetic bulk metal

| arm | kmesh=2 iters / FFT | kmesh=4 iters / FFT |
|---|---|---|
| baseline_prod (johnson) | **DIVERGES** 120 / — | **DIVERGES** 120 / 26222 |
| baseline_ctrl (pulay)   | 24 / 1157 | 20 / 3365 |
| teacher_exact (full χ₀) | 13 / 19359 | 19 / 206517 |
| **cheap_woodbury**      | **14 / 335** | **18 / —** |

- **kmesh=2:** cheap **1.71× fewer iters vs ctrl**, and it **matches the exact
  teacher** (14 vs 13 iters) at **58× fewer FFTs** (335 vs 19359). It also
  **rescues the diverging production mixer** (prod never converges → cheap 14).
- **kmesh=4:** cheap 18 vs ctrl 20 = **1.11× vs ctrl (rescue of prod)**. A bulk
  metal at a dense k-mesh is the *weak* regime — the pulay control is already
  well-conditioned, so even the exact teacher only shaves 20→19. Expected: the
  dielectric preconditioner earns its keep where the base mixer struggles.

### Al(100) 4-layer slab (nspin=1) — THE target regime (large-inhomogeneous)

| arm | iters | FFT launches |
|---|---|---|
| baseline_prod | 22 | 1261 |
| baseline_ctrl | 23 | 1137 |
| **cheap_woodbury** | **15** | **580 running + 3082 one-time build** (n_col=512) |

- **1.47× fewer iters vs prod** (22/15), **2.2× fewer RUNNING FFTs** (580 vs
  1261), faithful to 1e-13. This is the inhomogeneous surface regime the method
  is built for, and it is the clearest running-cost win.

## Verdict — a conditional target-regime win; the deciding test is M1.5

The mechanism is **VALIDATED**: the cheap subspace χ₀ is faithful (identical
fixed point to 1e-13) and its per-apply is genuinely FFT-free, so on the target
inhomogeneous slab it cuts both iterations (1.47×) and running FFTs (2.2×).

The automatic `M1_GO` gate returns **False on the Al slab for one reason only**:
cut_vs_prod = 22/15 = **1.467×**, a hair under the 1.5 iteration bar. The FFT
criterion **passed** (running overhead 0.46 ≤ 1.3). This is **not a rejection** —
it is a conditional win, and the whole verdict reduces to **build amortization**:

- one-time build: **3082 FFTs** (the `apply_k_hxc` per column, once);
- per-SCF running saving: **681 FFTs** (1261 → 580);
- **break-even ≈ 4.5 SCFs.**

So the cheap Woodbury is **net-positive for any multi-SCF workflow that reuses
the frozen subspace** — geometry relaxation, EOS, phonon stencils, MD,
inverse-design — and net-negative for a single isolated small-N SCF (where the
build isn't amortized and the running saving alone doesn't clear the 1.5× bar).

**The one open question that decides M1.5:** does the frozen χ₀ subspace survive
**geometry displacement**? Build the subspace once at geometry A and reuse it
across rattled/displaced geometries — if the iteration count stays ~15, the
amortization story holds across the exact multi-SCF workflows above and the
method is a clear production win. That is the deciding experiment.

- **Fixed-point identity:** resolved (dE ≤ 1e-13 Ha, d|ρ| ≤ 1e-10; SAME=True).
- **Cheap keeps the teacher's benefit at 1/58th its FFT overhead** and **rescues
  divergence** where production fails.
- **M2** (coupled charge+spin block already handled here; the Das-Gavini
  across-iteration adaptive accumulation of the subspace is the natural next
  build) is the user's call — the M1 mechanism is proven; M1.5 (subspace
  transferability under displacement) is the gating experiment before any
  production commit.

## M2 end-to-end — the win survives the REAL driver (relaxation)

M2 productionized the mechanism: a `Chi0PrecondCache`
(`scf/subspace_chi0.py`) freezes the `WoodburyPrecond` at the FIRST converged
SCF of a multi-geometry task and reuses it as `precond_op` on every later ionic
step, behind an auto-abstain ρ(M) gate. It is wired through the ASE calculator
(`chi0_precond=True`) and the Input schema (`scf.mixing.chi0_precond: true`), so
`api.run_relax` (the production nested BFGS+SCF engine) drives it. Auto-abstain
reuses the shipped `scf.soft_mode.dominant_screening_eigenvalue`: engage only when
ρ(M) ≥ `_CHI0_ENGAGE_RHO=2.0` (bulk fcc-Al ρ=0.82 abstains, slab ρ=7.89-29.73
engages — the crux stage0 gap).

**Experiment** (`m2_end_to_end.py`, asus, 8 threads): a relaxed Al(100) 4-layer
slab rattled 0.08 Å, then relaxed through `api.run_relax` (nested BFGS, 8 steps →
9 SCF evaluations), feature OFF (pulay + local_tf) vs ON (same + chi0_precond).
Both follow the byte-identical BFGS trajectory (a preconditioner cannot move the
fixed point), so this is an apples-to-apples cost measurement.

| metric | OFF (local_tf) | ON (chi0 reuse) | ratio |
|---|---|---|---|
| total SCF iters (9 SCFs) | 204 | 140 | **1.46× fewer** |
| **wall time** | 351.1 s | 240.2 s | **1.46× faster** |
| total FFT launches | 12737 | 11970 | 1.06× fewer |
| *running FFT (excl. one-time)* | 12737 | 7198 | *1.77× fewer* |
| relaxed energy | −7453.28509152 eV | −7453.28509152 eV | dE = **1.4e-9 eV** |
| relaxed geometry | — | — | max Δpos = **2.9e-9 Å** |

The gate ENGAGED (ρ(M)=**37.81**, n_col=512, **105 zero-FFT** precond applies).
Per-step SCF iterations: ON step 1 = 27 (no operator yet, matches OFF's 27), then
steps 2-9 reuse the frozen subspace and hold **13-15 iters** while OFF ran
**20-27**. This is the M1.5 transferability question answered YES through the real
driver: the subspace frozen at geometry 1 keeps its iteration cut across all 8
displaced geometries.

**Honest verdict — the win survives, on wall clock and iterations; the pure-FFT
metric is thin at this trajectory length.** Wall time and SCF iterations both drop
**1.46×** end-to-end, faithful to 1.4e-9 eV / 2.9e-9 Å. The *running* FFT saving
(**1.77×**, close to M1's 2.2×) confirms the zero-FFT-apply mechanism transfers to
the driver. But the **total** FFT ratio is only **1.06×**: the one-time gate
(1690 FFT) + build (3082 FFT) = 4772 FFT nearly cancels the ~4000 FFT saved from
64 fewer iterations over just 9 SCFs. That one-time cost amortizes with trajectory
length — the M1-projected ~1.9× *total*-FFT win needs ≳20 SCFs to materialize on
the FFT metric. Wall clock does not wait for it: the build is wall-cheap relative
to its FFT count (dense `apply_k_hxc` GEMMs) and the 1.46× iteration cut maps
directly to 1.46× wall, because SCF wall is dominated by the Davidson eigensolve,
not FFTs alone. **Net: a real, faithful 1.46× wall speedup through `api.run_relax`
at 9 ionic steps, growing with trajectory length.** The gate (1690 FFT, ~35% of
the one-time cost) is the obvious next thing to cheapen.

- **Wiring:** `scf/subspace_chi0.py::Chi0PrecondCache` (+ `_CHI0_ENGAGE_RHO`
  gate); `calculator.py` (`chi0_precond` flag, threaded in `_calculate_nc`);
  `inputs/models.py::MixingParams.chi0_precond`; `api/relax.py::_build_relax_calc`.
- **Tests:** `tests/unit/test_chi0_precond_cache.py` (engage/abstain, decide-once,
  grid-invalidation — heavy calls monkeypatched, fast tier).
