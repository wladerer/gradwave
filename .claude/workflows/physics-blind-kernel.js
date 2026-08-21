export const meta = {
  name: 'physics-blind-kernel',
  description: 'Sanitize a numerical subsystem into a domain-free spec, attack it with a physics-blind agent, calibrate the attacks primitive costs by measurement, then vet against code + measured history',
  whenToUse: 'When optimization ideas keep converging on "how codes in this domain do it" and you want algorithmic attacks from unanchored CS/math priors. Args: a target description, e.g. "src/gradwave/flapw/scf.py per-iteration kernel" plus current wall-clock fractions if known.',
  phases: [
    { title: 'Sanitize', detail: 'read the target; emit a spec with all domain vocabulary stripped' },
    { title: 'Attack', detail: 'physics-blind agent proposes ranked algorithmic attacks from the spec alone' },
    { title: 'Calibrate', detail: 'micro-benchmark the primitives each attack hinges on, in the real regime, on the target hardware' },
    { title: 'Vet', detail: 'map attacks to code sites; strike tried/measured; rank survivors using MEASURED constants + falsifiable crossover predictions' },
  ],
}

// Validated 2026-08-20 on the FLAPW secular kernel: the blind agent independently rediscovered a
// measured result it could not have known (dense small-N applies beating FFT applies — the gated
// PW ToeplitzApply finding), then produced two attacks the domain-anchored analysis had missed
// (augmented Rayleigh-Ritz in span[V_prev, U]; iterative p-mode B-deflation).
//
// Calibrate phase ADDED 2026-08-21 after a measured failure: on the sigma-shielding pipeline the
// vet ranked a dense eigh-prepass Sternheimer solver #1 at an estimated 10-30x — reasoning
// structurally ("one factorization amortized over many solves") — but it measured 2.3x SLOWER at
// 2^3 and 1.15x slower at 4^3, because at npw~few-hundred with a fast preconditioned CG the dense
// eigh is itself expensive and the amortization base (small nk x nocc) is too small. The method is
// reliable at FINDING candidates and KILLING already-measured negatives, but systematically weak at
// predicting CONSTANT FACTORS in an un-measured regime (it defaults to op-counting / asymptotics).
// The calibration phase closes that loop with a cheap measurement BEFORE any build is authorized.

const target = args || 'src/gradwave/flapw/scf.py per-iteration kernel'

phase('Sanitize')
const spec = await agent(
  `Read the code of ${target} and produce a SANITIZED computational specification for an ` +
  `algorithms expert who must never learn the application domain. Rules — the spec may contain ` +
  `ONLY: array shapes and dtypes; operator STRUCTURE (diagonal, Toeplitz/index-difference, ` +
  `low-rank/Gram, banded, ...); the dependency graph of the pipeline; measured or estimated cost ` +
  `fractions per step; loop structure, and — critically — WHAT CHANGES vs WHAT IS FIXED between ` +
  `loop iterations and between problem instances. Strip every domain noun (no physics, chemistry ` +
  `or application words; rename quantities to neutral symbols). Include concrete sizes. Do not ` +
  `include file paths or code identifiers. Return only the spec text.`,
  { label: 'sanitize', phase: 'Sanitize' })

phase('Attack')
const attacks = await agent(
  `You are given an abstract numerical computation. Analyze it purely as computer science and ` +
  `numerical mathematics. Do NOT try to guess the application domain; do NOT read any files or ` +
  `repositories — reason from the spec alone. Propose the fastest known algorithmic approaches ` +
  `(structured matrices, sketching/randomized NLA, update/downdate algebra, Krylov recycling, ` +
  `tolerance forcing, batching/scheduling, ...). Deliver a ranked list of concrete attacks with ` +
  `expected asymptotic and constant-factor gains and numerical-stability risks for each. For each ` +
  `attack, name the SINGLE primitive operation whose measured cost decides whether it wins (e.g. ` +
  `"a dense n x n Hermitian eigendecomposition vs one preconditioned CG solve to tol at these ` +
  `sizes", "a batched size-N FFT vs a dense N x N GEMM") — the crossover primitive.\n\n` +
  `## The computation\n${spec}`,
  { label: 'blind-attack', phase: 'Attack', effort: 'high' })

phase('Calibrate')
const calibration = await agent(
  `You are an empirical performance-measurement agent (NOT physics-blind — you may use the real ` +
  `sizes and hardware). Below are algorithmic attacks on a numerical kernel, each naming a ` +
  `"crossover primitive" whose real cost decides whether it wins. Your job: MEASURE those ` +
  `primitives in the actual regime, so the downstream ranking uses measured constants instead of ` +
  `asymptotic guesses — a past run of this workflow over-ranked a dense-factorization attack at ` +
  `"10-30x" that measured SLOWER, because nobody timed the factorization against the iterative ` +
  `solve it replaced.\n\n` +
  `For each distinct crossover primitive across the attacks: write a SMALL self-contained ` +
  `micro-benchmark on SYNTHETIC operands at the spec's concrete sizes (a random complex128 ` +
  `Hermitian matrix of the right n for an eigh; a synthetic SPD operator of the right size and ` +
  `representative conditioning for a preconditioned CG to the stated tol, reporting BOTH per-solve ` +
  `wall AND iteration count; a batched FFT / dense GEMM at the right shapes; etc.). These are pure ` +
  `hardware-characterizing timings — they need neither the real codebase nor heavy setup. Run them ` +
  `on the TARGET hardware (asus, via pueue + scripts/qrun with absolute paths and a pinned ` +
  `OMP_NUM_THREADS matching how the real kernel runs; poll the EXIT marker, never tail -f a pipe). ` +
  `Do NOT run anything heavy locally.\n\n` +
  `Return a table: per primitive, the measured per-call wall (and iteration count where iterative), ` +
  `at each relevant size; then for every attack, compute its DERIVED CROSSOVER — the amortization ` +
  `count / problem size at which the attack's primitive beats the baseline primitive — and state ` +
  `plainly whether the regime in the spec reaches that crossover. Flag any attack whose measured ` +
  `crossover is NOT reached in the target regime as "predicted negative — do not build".\n\n` +
  `## Attacks (with their crossover primitives)\n${attacks}\n\n## Concrete sizes\n${spec}`,
  { label: 'calibrate', phase: 'Calibrate' })

phase('Vet')
const vetted = await agent(
  `Below are algorithmic attacks proposed for the computation implemented in ${target}, by an ` +
  `agent that never saw the code or domain, PLUS a calibration report with MEASURED primitive ` +
  `costs and derived crossovers in the real regime. Map each attack to concrete code sites, then ` +
  `STRIKE any attack this project has already tried and measured (check the project memory notes ` +
  `and experiments/ docs for closed avenues — items recorded as "measured, no win" or gated off ` +
  `after regression) AND strike any attack the calibration flags as "predicted negative — crossover ` +
  `not reached". \n\n` +
  `For each survivor: implementation sketch, effort estimate, which existing test gates ` +
  `(metamorphic suite, nulls, reference comparisons) validate it, and — using the MEASURED ` +
  `constants from calibration, NOT asymptotics — an expected-gain INTERVAL with an explicit ` +
  `WORST CASE. Then, for each, a falsifiable crossover prediction and the single CHEAPEST ` +
  `experiment that would refute it. RANK by (gain x probability of success) / effort, using the ` +
  `worst-case gain. Any survivor whose worst-case gain is < 1x, or whose crossover the calibration ` +
  `did not already confirm, is BUILD-BLOCKED: mark it "probe before building" and name the probe. ` +
  `Also: if a sibling implementation of the same observable already exists in the module and is ` +
  `faster, say so — do not rank a new build above an existing faster path.\n\n` +
  `Separately list any exact, zero-numerical-risk levers (symmetry reductions, exact ` +
  `dedup) that the blind agent flagged as out-of-scope — these cannot measure negative and should ` +
  `be surfaced even when parked.\n\n` +
  `## Attacks\n${attacks}\n\n## Calibration (measured)\n${calibration}`,
  { label: 'vet', phase: 'Vet' })

return { spec, attacks, calibration, vetted }
