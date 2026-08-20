export const meta = {
  name: 'physics-blind-kernel',
  description: 'Sanitize a numerical subsystem into a domain-free spec, attack it with a physics-blind agent, vet the attacks against the code and measured history',
  whenToUse: 'When optimization ideas keep converging on "how codes in this domain do it" and you want algorithmic attacks from unanchored CS/math priors. Args: a target description, e.g. "src/gradwave/flapw/scf.py per-iteration kernel" plus current wall-clock fractions if known.',
  phases: [
    { title: 'Sanitize', detail: 'read the target; emit a spec with all domain vocabulary stripped' },
    { title: 'Attack', detail: 'physics-blind agent proposes ranked algorithmic attacks from the spec alone' },
    { title: 'Vet', detail: 'map attacks to code sites; strike anything already tried/measured; rank survivors' },
  ],
}

// Validated 2026-08-20 on the FLAPW secular kernel: the blind agent independently rediscovered a
// measured result it could not have known (dense small-N applies beating FFT applies — the gated
// PW ToeplitzApply finding), then produced two attacks the domain-anchored analysis had missed
// (augmented Rayleigh-Ritz in span[V_prev, U]; iterative p-mode B-deflation).

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
  `expected asymptotic and constant-factor gains and numerical-stability risks for each.\n\n` +
  `## The computation\n${spec}`,
  { label: 'blind-attack', phase: 'Attack', effort: 'high' })

phase('Vet')
const vetted = await agent(
  `Below are algorithmic attacks proposed for the computation implemented in ${target}, by an ` +
  `agent that never saw the code or domain. Map each attack to concrete code sites, then STRIKE ` +
  `any attack this project has already tried and measured (check the project memory notes and ` +
  `experiments/ docs for closed avenues — e.g. items recorded as "measured, no win" or gated off ` +
  `after regression). For survivors: implementation sketch, effort estimate, expected gain in ` +
  `THIS codebase's terms, and which existing test gates (metamorphic suite, nulls, reference ` +
  `comparisons) validate each. Rank by (gain x probability of success) / effort.\n\n` +
  `## Attacks\n${attacks}`,
  { label: 'vet', phase: 'Vet' })

return { spec, attacks, vetted }
