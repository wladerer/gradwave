export const meta = {
  name: 'gradwave-moonshot-ideas-r5',
  description: 'Round 5 MOONSHOT round: wall-breaking-by-construction reformulations (representation/discretization/dynamics/learned-operator/exact-structure/exotic-substrate) -> rank by boldness+credible-mechanism -> research -> report',
  phases: [
    { title: 'Ideate', detail: '6 moonshot lenses generate a bold pool' },
    { title: 'Rank', detail: 'rank by novelty x wall-break x credible mechanism' },
    { title: 'Research', detail: 'per idea: prior art + gradwave fit + the one hard blocker' },
    { title: 'Report', detail: 'ranked, publication-if-it-works framing' },
  ],
}

const BRIEF = `# gradwave acceleration — ROUND 5, MOONSHOT ROUND

## The mandate (read first)
The user reviewed four prior rounds and found the incremental systems/campaign/mixing ideas unexciting; the MOONSHOTS were the most interesting. This round is EXCLUSIVELY moonshots: high-risk, high-reward, STRUCTURALLY TRANSFORMATIVE ideas that break the four walls BY CONSTRUCTION (a fundamentally different representation, discretization, formulation, learned operator, exploited exact structure, or compute substrate) rather than working around them. The bar is "this would be a genuine research contribution / a paper if it worked." Incremental systems, campaign scheduling, caching, mixing tweaks, and setup amortization are OUT OF SCOPE this round. Bold-but-with-a-credible-mechanism beats both safe and vague.

## What gradwave is
Differentiable plane-wave DFT in PyTorch (pure eager, fp64/complex128, validates sub-meV vs Quantum ESPRESSO). END-TO-END DIFFERENTIABLE: learned XC, learned pseudos, inverse design, exact response/Hessians via autograd. Core objects: plane-wave orbitals c(G) on an ecut G-sphere, an FFT-based H-apply (two complex128 FFTs + nonlocal projector GEMMs per band), batched Davidson eigensolver, density mixing, the whole thing autograd-differentiable.

## The four walls a moonshot should break BY CONSTRUCTION
(1) consumer-GPU fp64 tax (fp64 at 1/64 fp32); (2) small cells / the O(npw) storage and O(npw^2) memory cliff capping ~110-128 atoms; (3) the sub-meV / 1e-13 eV accuracy gate; (4) mixing-limited iteration count (~2.3x QE on metals). Rounds 1-4 mostly went AROUND these. A moonshot changes the representation/math so a wall stops existing.

## Hard constraints a moonshot must respect (or explicitly, credibly address)
- Must stay (or be made) AUTOGRAD-DIFFERENTIABLE end to end — that is gradwave's whole identity. A representation whose gradients are ill-conditioned (e.g. truncated-SVD gradients at near-degenerate singular values) must say how it handles that.
- Must reach sub-meV at convergence, OR be an exact-at-convergence accelerator (seed/preconditioner/coarse-space with an exact corrector) where the approximation costs only speed. Say which.
- METALS are the hard case (partially-occupied oscillatory Fermi-surface states) and the most-used one — an idea that only works for smooth insulators is fine but must SAY so.
- Pure-eager PyTorch, no custom kernels today; needing a new kernel/layer is allowed for a moonshot but must be named as part of the cost.

## Do NOT re-propose (rounds 1-4 + docs). You MAY explore a different mechanism in the same broad CLASS, but not these specific items:
QTT-Orbitals (quantized-tensor-train orbitals) and the other round-4 moonshots (Wannier Courier reverse-downfolding, Conformal Scout Cascade, CompressedTape, GammaPrime) are SAVED/deferred — do not restate them. Also excluded: fp64 emulation, stochastic DFT, direct minimization/ensemble-CG, sketched Rayleigh-Ritz, s-step Davidson, FOE density matrix, Shirley basis, stick/pruned FFT, meta-learned SCF map, adaptive BZ, CheFSI/LOBPCG/RMM-DIIS, fp32 drafting, CUDA graphs, sync-free Davidson, torch.compile-on-apply, Gamma-real wavefunctions, learned RADIAL preconditioner, Delta-tier, GreedyManifold reduced-basis POD (a DIFFERENT reduced/compressed representation is allowed), HessNet, Continuation-Sweep, GradTrust, RESPA-SCF, warm fork-server/daemon/CRIU, Converge-to-Answer, Hull-Chasing, ML density initializer. Also don't re-pitch generic "use a neural network potential" (that is just MLIP; a moonshot must do something an MLIP cannot).

## Reusable substrate (for "what it builds on")
core/fftbox.py (FFT convention), grids.py (G-sphere/FFT grid), core/batch.py + core/hamiltonian.py (batched H-apply), solvers/davidson.py, scf/loop.py + implicit.py (chi0/K_Hxc response, IFT adjoint), opt/joint.py + opt/newton.py (direct-min/Newton-CG, Lowdin/Teter/checkpointed density), symmetry.py (IBZ + magnetic), postscf response/forces/stress, learned-XC/pseudo autograd stack, dtypes.py.

## Novelty bar
Publication-if-it-works. Prefer ideas that make a WALL STOP EXISTING. Name the one hard technical blocker honestly. A labelled longshot with a real mechanism is exactly what this round wants.`

const IDEA_POOL_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ideas'],
  properties: {
    ideas: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'pitch', 'wall_broken', 'mechanism', 'hard_blocker'],
      properties: {
        name: { type: 'string' },
        pitch: { type: 'string', description: '2-4 sentences: the transformative idea' },
        wall_broken: { type: 'string', description: 'which of the 4 walls it breaks BY CONSTRUCTION' },
        mechanism: { type: 'string', description: 'the concrete technical mechanism, not a vibe' },
        hard_blocker: { type: 'string', description: 'the single hardest technical obstacle, stated honestly' },
      },
    } },
  },
}

const RANK_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ranked', 'merged_note'],
  properties: {
    ranked: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'pitch', 'wall_broken', 'why_bold', 'credibility', 'hard_blocker'],
      properties: {
        name: { type: 'string' }, pitch: { type: 'string' }, wall_broken: { type: 'string' },
        why_bold: { type: 'string' }, credibility: { type: 'string', description: 'is there a plausible mechanism, or is it vague/magical?' },
        hard_blocker: { type: 'string' },
      },
    } },
    merged_note: { type: 'string' },
  },
}

const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name', 'prior_art', 'exists_in_gradwave', 'builds_on', 'if_it_works', 'the_blocker', 'blocker_tractable', 'feasibility', 'verdict', 'sources'],
  properties: {
    name: { type: 'string' },
    prior_art: { type: 'string', description: 'has anyone done this in DFT or the source field? result?' },
    exists_in_gradwave: { type: 'string', enum: ['absent', 'partial', 'present'] },
    builds_on: { type: 'string' },
    if_it_works: { type: 'string', description: 'the transformative payoff, with scale' },
    the_blocker: { type: 'string', description: 'the one hard technical obstacle' },
    blocker_tractable: { type: 'string', description: 'is the blocker plausibly solvable, or fatal? honest read' },
    feasibility: { type: 'string', enum: ['moderate', 'hard', 'research_project', 'moonshot'] },
    verdict: { type: 'string', enum: ['pursue', 'promising_moonshot', 'interesting_longshot', 'fatal_flaw'] },
    sources: { type: 'array', items: { type: 'string' } },
  },
}

const REPORT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ranked', 'best_moonshots', 'has_a_real_path', 'summary'],
  properties: {
    ranked: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'one_line', 'verdict', 'if_it_works', 'the_blocker', 'builds_on', 'feasibility'],
      properties: {
        name: { type: 'string' }, one_line: { type: 'string' },
        verdict: { type: 'string', enum: ['pursue', 'promising_moonshot', 'interesting_longshot', 'fatal_flaw'] },
        if_it_works: { type: 'string' }, the_blocker: { type: 'string' }, builds_on: { type: 'string' }, feasibility: { type: 'string' },
      },
    } },
    best_moonshots: { type: 'array', items: { type: 'string' }, description: 'boldest with a credible mechanism' },
    has_a_real_path: { type: 'array', items: { type: 'string' }, description: 'moonshots whose blocker looks actually tractable' },
    summary: { type: 'string' },
  },
}

const LENSES = [
  { key: 'representation', title: 'Representation revolution',
    brief: 'Represent orbitals/density in a fundamentally different, compressive form that breaks the O(npw) storage / fp64-tax / size walls by construction. Implicit neural representations / neural fields for orbitals; hierarchical/butterfly or HSS-structured orbital blocks; wavelet or multiresolution coefficients; a learned adaptive per-system basis; Gaussian-process / RKHS orbital fields; hybrid real-space-localized + plane-wave. NOT quantized tensor trains (saved). Each must say how it stays autograd-differentiable and reaches sub-meV.' },
  { key: 'discretization', title: 'Discretization revolution',
    brief: 'Replace the plane-wave basis / uniform FFT grid for the KS equations with a different discretization that changes root scaling or conditioning: adaptive real-space multiresolution (MADNESS-style wavelets), finite elements / discontinuous Galerkin, spectral elements, sinc-DVR, gausslets/discontinuous bases, or a mixed basis. The moonshot is a per-atom/per-region adaptive resolution that a global FFT cannot express — attacking the size wall and the metal conditioning at the root.' },
  { key: 'dynamics', title: 'Dynamics / sampling reformulation',
    brief: 'Reach the KS ground state via a different dynamical or probabilistic process with better scaling/parallelism: rigorous imaginary-time or diffusion-based propagation of the orbitals/density; a stochastic differential equation / Langevin sampler whose stationary distribution is the KS density; tensor-network imaginary-time evolution; a homotopy/numerical-continuation of the entire self-consistency from a trivially-solvable limit. Must reach the exact fixed point, not an approximation.' },
  { key: 'learnedop', title: 'Learned operators that stay EXACT at convergence',
    brief: 'A learned component that accelerates but cannot corrupt the answer because an exact corrector guarantees convergence. Neural-operator preconditioner for the Davidson/mixing linear systems; ML-learned multigrid COARSE SPACE / interpolation operator; ML-predicted invariant-subspace or deflation vectors seeding the eigensolver; an operator-learned fixed-point accelerator with an exact residual gate. Distinct from the rejected meta-learned-SCF-map and learned-RADIAL-preconditioner: name the exactness guarantee and why it beats the analytic Teter/Kerker/chi0 operators already shipped.' },
  { key: 'structure', title: 'Exploit exact structure no PW code uses',
    brief: 'Find and exploit exact mathematical structure the plane-wave formulation leaves on the table: hidden low-rank / hierarchical (H-matrix, HSS, butterfly) structure in the nonlocal or exchange operator for a fast-transform apply; group-theoretic block-diagonalization of H beyond the shipped IBZ reduction (little groups, projective irreps); an exact tensor factorization of the Hamiltonian; exact fast multipole for the Hartree/exchange. Exact by construction, so accuracy is never at risk; the win is a lower-complexity apply.' },
  { key: 'substrate', title: 'Radically different compute substrate or execution model',
    brief: 'Change the machine or the execution model, not the math. Analog/optical/neuromorphic or in-memory-compute FFT and linear algebra for the fp32-tolerant inner work; probabilistic-bit / thermodynamic-computing samplers for the density; a fully asynchronous / dataflow nonlinear-solver execution that removes the SCF/eigensolver serialization; a chunked out-of-core formulation that makes cell size disk-bound not RAM-bound. Name the substrate and the realistic 10-100x lever, and be honest about availability.' },
]

phase('Ideate')
const pools = await parallel(LENSES.map((L) => () => agent(
  BRIEF + '\n\n## YOUR MOONSHOT LENS: ' + L.title + '\n' + L.brief +
  '\n\nPropose 5 MOONSHOT ideas through this lens — publication-if-it-works, wall-breaking-by-construction. Nothing on the exclusion list (a different mechanism in the same broad class is fine). For each: a memorable name, a 2-4 sentence pitch, which wall it breaks by construction, the CONCRETE technical mechanism (not a vibe), and the single hardest technical blocker stated honestly. Reward boldness, but every idea needs a plausible mechanism a skeptic could evaluate — no magic.',
  { schema: IDEA_POOL_SCHEMA, phase: 'Ideate', label: 'ideate:' + L.key, effort: 'high' }
)))
const candidates = pools.filter(Boolean).flatMap((p) => p.ideas)
log('Ideation produced ' + candidates.length + ' moonshot candidates across ' + LENSES.length + ' lenses')

phase('Rank')
const rank = await agent(
  BRIEF + '\n\n## CANDIDATE POOL (' + candidates.length + ')\n' + JSON.stringify(candidates, null, 2) +
  '\n\nYou are the curator for a MOONSHOT round. Dedup/merge. DROP anything on the exclusion list or that is vague/magical with no evaluable mechanism (this round wants bold-WITH-a-credible-mechanism, not hand-waving). Do NOT drop an idea for being hard, risky, metals-limited, or needing new infrastructure — those are expected. RANK by (how cleanly it breaks a wall by construction) x (boldness/novelty) x (credibility of the mechanism), keeping diversity across the six lenses. Return the top 14, each with why it is bold, an honest read of its credibility (real mechanism vs vague), which wall it breaks, and its hardest blocker, plus a note on what you merged or dropped-as-vague.',
  { schema: RANK_SCHEMA, phase: 'Rank', effort: 'high' }
)
const top = rank.ranked.slice(0, 14)
log('Ranked ' + top.length + ' moonshots forward to research')

phase('Research')
const researched = await parallel(top.map((idea) => () => agent(
  BRIEF + '\n\n## MOONSHOT TO ASSESS\n' + JSON.stringify(idea, null, 2) +
  '\n\nAssess this moonshot HONESTLY but generously (this round protects bold ideas; reserve fatal_flaw for a genuine showstopper). (1) PRIOR ART: has anyone done this in DFT/electronic-structure OR in the field it borrows from (applied math, ML, signal processing, HPC)? With what result? Web-research, cite real sources. (2) EXISTS IN GRADWAVE: check the repo at the gradwave path (grep/read src/gradwave and docs). (3) What substrate it builds on. (4) If_it_works: the transformative payoff with scale. (5) The_blocker: the ONE hardest technical obstacle. (6) blocker_tractable: is that blocker plausibly solvable (cite anyone who solved something similar) or is it fatal? Be specific. (7) feasibility (moderate/hard/research_project/moonshot) and verdict (pursue / promising_moonshot / interesting_longshot / fatal_flaw). Cite sources.',
  { schema: RESEARCH_SCHEMA, phase: 'Research', agentType: 'general-purpose', effort: 'medium', label: 'research:' + idea.name }
)))
const results = researched.filter(Boolean)
log('Assessed ' + results.length + ' moonshots')

phase('Report')
const report = await agent(
  BRIEF + '\n\n## ASSESSED MOONSHOTS\n' + JSON.stringify(results, null, 2) +
  '\n\nSynthesize the final ranked round-5 moonshot report. Rank by (verdict, then how cleanly it breaks a wall, then tractability of the blocker). For each: name, one-line pitch, verdict, if-it-works payoff, the one blocker, what it builds on, feasibility. best_moonshots = the boldest with a genuinely credible mechanism. has_a_real_path = the ones whose hardest blocker actually looks tractable (the ones worth a prototype). The summary must say which walls, if any, a moonshot could truly make STOP EXISTING, and which single moonshot has the best ratio of transformative-payoff to blocker-tractability. Honest and concrete; a labelled longshot with a real mechanism is the goal, magic is not.',
  { schema: REPORT_SCHEMA, phase: 'Report', effort: 'high' }
)
return { report, results, rank_note: rank.merged_note }