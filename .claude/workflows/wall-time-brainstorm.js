export const meta = {
  name: 'wall-time-brainstorm',
  description: 'WALL-TIME-ONLY acceleration round on the common case (small/medium metal cells, fp64): 4 pain-area lenses (dispatch, cold-start, redundant work, convergence) -> rank by wall-time lever x tractability -> vet against code + GH PRs + ideas.md so nothing already-done is re-proposed -> report',
  phases: [
    { title: 'Ideate', detail: '4 pain-area lenses generate a wall-time idea pool' },
    { title: 'Rank', detail: 'rank by wall-time lever x tractability x not-already-done, drop out-of-scope' },
    { title: 'Vet', detail: 'per idea: grep src + gh pr list + ideas.md -> does it already exist? + prior art + honest lever' },
    { title: 'Report', detail: 'ranked, net-new-only, with the one hard blocker each' },
  ],
}

const BRIEF = `# gradwave WALL-TIME round — reduce wall-clock time, nothing else

## The mandate (read first)
This round targets ONE metric: wall-clock time to a converged result, on the COMMON case
gradwave actually runs — small-to-medium cells (well under the ~110-128-atom dense cliff),
metals included (the hard, most-used case), fp64/complex128, pure-eager PyTorch. Ideas must
help this regime. The four big pain points, and the ONLY themes in scope:

1. DISPATCH — Python/PyTorch per-op dispatch and launch overhead. In eager fp64 on CPU the
   SCF step / H-apply / Davidson inner loop pay per-op Python + ATen dispatch cost that is a
   real fraction of wall time at these small tensor sizes. Fusing ops, capturing/compiling
   larger regions, cutting redundant allocations and host-device syncs, reducing the number
   of launched kernels per band/k.
2. COLD-START / SETUP — everything paid once before (or at the start of) the SCF: import time,
   pseudopotential radial FT / form-factor build, Ewald + local-PP build, FFT planning, the
   first SCF step from a cold guess, process startup. The first-run and short-run tax.
3. REDUNDANT / RE-DONE WORK — recomputing things that did not change, within one SCF, across
   SCF steps, across k-points, or across a campaign (relax / EOS / elastic / phonon / scan).
   Reuse, memo, transport, incremental update instead of rebuild.
4. CONVERGENCE BEHAVIOR — fewer SCF iterations / eigensolver passes to the SAME converged
   answer: stopping rules, mixing SCHEDULE and history control, restart heuristics, coupling
   solver accuracy to SCF progress. (NOT a new mixing PRECONDITIONER, and NOT a learned seed —
   see the hard exclusions; the obvious tolerance/precision knobs already ship, see below.)

## What gradwave is
Differentiable plane-wave DFT in PyTorch (pure eager, fp64/complex128, validated sub-meV vs
Quantum ESPRESSO). END-TO-END DIFFERENTIABLE (learned XC/pseudos, inverse design, exact
response/Hessians via autograd) — that identity is non-negotiable. Core inner loop: plane-wave
orbitals c(G) on an ecut G-sphere; H-apply = two complex128 FFTs + nonlocal projector GEMMs
per band; batched Davidson over (nk, nb, npw); density mixing; all autograd-differentiable.

## Hard constraints (respect or explicitly, credibly address)
- MUST stay autograd-differentiable end to end. An idea that breaks the graph (in-place graph
  capture, a non-differentiable cache on the forward path) must say how gradients survive.
- MUST be EXACT AT CONVERGENCE — sub-meV / 1e-13 eV gates hold at the fixed point. A speed idea
  may only change the PATH to the fixed point, never the fixed point. Say why the answer is
  unchanged.
- METALS are the important, hard case (partially-occupied oscillatory Fermi-surface states).
  An idea that only helps smooth insulators is essentially out of scope here (see below).
- Pure-eager PyTorch today. A new kernel / torch.compile region / graph capture is allowed but
  must be named as part of the cost, and must degrade safely (eager fallback) since the target
  hardware is CPU fp64 AND datacenter fp64 GPUs (A100/H100) that gradwave PRODUCTION-TESTS on — full-rate
  fp64, LAUNCH-bound at small op sizes, where cutting/fusing kernel launches is a REAL win. Do NOT discredit
  an idea for needing a datacenter GPU (we own them); the RTX 3050 is a crippled-fp64 DEV card only, never the
  fp64-GPU perf verdict. A purely fp32-GPU-only win is still out (sub-meV needs fp64)).

## OUT OF SCOPE — do NOT propose (these are the whole point of the constraints)
- INSULATOR-ONLY ideas. If it needs a gap / smooth density to work, it is out (Gamma-only-real,
  cross-k Taylor transport, most homotopy tricks live here). Metals must be covered.
- LARGE-CELL-ONLY ideas. Anything whose win only appears past ~100 atoms / attacks the O(npw^2)
  dense-allocation or O(N^3) scaling cliff is out (linear-scaling, DG/ALB adaptive local basis,
  stochastic DFT, tensor-train/QTT, Green's-function embedding, reduced-basis POD). Wrong axis.
- BETTER PRECONDITIONERS. Kerker, local Thomas-Fermi (local_tf.py), Teter (solvers/precond.py),
  Broyden/Johnson/Pulay-Anderson (mixing.py), the Stoner spin-preconditioner (spin_precond.py),
  AND a LEARNED multi-pole preconditioner (learned_precond.py) ALL already ship, and the "fit a
  better preconditioner" theme was reviewed and WALKED BACK. Do not propose a new mixing operator
  / dielectric model / learned preconditioner in any framing.
- ML / NEURAL / LEARNED INITIAL GUESSES / SEEDS. Atomic-orbital wavefunction seeding and the
  AO density-channel seed were both BUILT, measured 0-2 saved iters at neutral-to-worse wall,
  and REVERTED; by Rayleigh-Ritz invariance a rotation within the AO block converges identically,
  and the outer SCF count is MIXING-limited, not seed-quality-limited. The seed channel is
  structurally CLOSED. No learned/ML density or wavefunction initializer.
- THE RAYLEIGH-RITZ / VARIATIONAL-SUBSPACE-INVARIANCE WALL (general form of the above; internalize
  it before proposing anything touching the eigensolver's starting point or basis). Any idea whose
  only lever is a BETTER / CHEAPER / ROTATED / COARSE-THEN-FINE SUBSPACE — a reduced-ecut ladder, a
  multi-resolution or atomic-orbital or learned starting block, a smarter seed — changes ONLY the
  path to the fixed point, never the fixed point: by Rayleigh-Ritz it yields the IDENTICAL converged
  answer, and because the outer SCF count is set by the density-mixing dielectric conditioning (NOT
  orbital quality) it saves at most the first step's diagonalization, which adaptive_diago_tol already
  runs loose (~1e-3). Net ~neutral. This closure has now killed THREE separate proposals (AO
  wavefunction seeding, Huckel Ignition, and Basis Ladder reduced-ecut seed). Do NOT pitch
  subspace/basis/seed-quality as an SCF-count or wall-time win. The two candidate ways past this wall
  were BOTH MEASURED and BOTH FAIL on wall time: changing the MIXING (excluded — preconditioners,
  reviewed and walked back) and changing the ROOT-FINDING CLASS (inexact-Newton, prototyped in PR #272,
  now CLOSED) — the NC inexact-Newton finisher reaches the identical fixed point but is wall-NEGATIVE on
  every metal tested and REGRESSES on the stiff magnetic case (see the tried-and-shelved list below). So
  there is NO known way past this wall on wall time; do NOT re-pitch a better subspace, a preconditioner,
  OR a second-order/Newton/root-finding-class scheme as a wall-time win.
- Also already tried & shelved (do not restate): fp64 emulation (double-single/Ozaki), fp32
  drafting / precision migration (0.35x on this HW; note scf/options.py mixed_precision already
  does an fp32 draft Davidson while diago tol is loose), whole-step CUDA graphs (measured null),
  WHOLE-STEP / H-APPLY torch.compile (MEASURED 0.69-0.71x NET LOSS, 2026-08-09: inductor cannot
  codegen COMPLEX operators, so gradwave's complex128 inner loop compiles to a graph that falls
  every complex op back to eager aten plus guard overhead — slower, not faster; plus dynamic-nb
  recompile churn. Only REAL-valued regions compile: XC already ships via _XCCompileMixin. Do NOT
  re-pitch torch.compile on the COMPLEX inner loop as-is; the substrate structurally can't fuse
  complex. A real-split rewrite (view_as_real -> real pointwise -> view_as_complex) DOES let inductor
  fuse (5.59x on an isolated glue chain) — but the end-to-end real-split APPLY was PROTOTYPED and
  MEASURED 0.92-0.93x (Al/Fe, bit-identical): net-negative, because the apply has only ~3-4 pointwise
  ops between extern FFT/GEMM and the ~48% glue is DIFFUSE across the whole step (Davidson/density/XC/
  mixing), not in the apply. So the substrate lever is effectively CLOSED at the apply level; capturing
  the diffuse glue needs a whole-step real-split refactor (high risk, prior whole-step = null) or custom
  kernels (big cost) — do NOT re-pitch either as a clean win. Only free substrate slice: real-valued XC
  via openssl+_XCCompileMixin (already shipped)),
  RMM-DIIS, CheFSI/LOBPCG as a speed win (both shipped, not faster here), s-step/comm-avoiding
  Davidson, sketched Rayleigh-Ritz, FOE/KPM density matrix, Shirley k-interpolation, stick/pruned
  FFT, adaptive/multi-fidelity BZ, FFT arena + Morton layout (arena half already ships), smearing/
  temperature continuation as a mixing cure, Harris-Foulkes trust-region line search;
  INEXACT-NEWTON SCF FINISHER (MEASURED, PR #272 CLOSED, 2026-08-09: the NC Newton step
  (I - chi0 K_Hxc)delta=r reaches the exact fixed point but is wall-NEGATIVE on all metals — Al 13.5->17.9s,
  Fe-nm 52->64s (iters down but wall up because each chi0 apply is a Sternheimer solve) — and WORSE on
  BOTH iters and wall on stiff FM Fe nspin=2 (17->20 iters, 2x wall) because Kerker preconditions only the
  CHARGE channel, not the Stoner/spin mode the shipped johnson+spin-precond mixer already handles;
  Eisenstat-Walker adaptive inner tol STALLS. Do NOT re-pitch Newton/second-order/JFNK as a wall win);
  SPIN-BATCHED DAVIDSON (MEASURED NEGATIVE, 2026-08-09, branch spin-batched-davidson not merged: folding
  nspin into the k-batch to run ONE davidson over (2*nk,nb,npw) instead of two serial per-spin solves
  loses on CPU fp64 ~0.90x AND GPU fp64 ~0.89-0.93x — no launch latency to save on CPU, crippled-fp64
  (1/64) FFT/GEMM dominates on GPU; BUT this was measured on CPU + the CRIPPLED RTX 3050 ONLY, not on a datacenter fp64 GPU
  (A100/H100, which gradwave production-tests on) where full-rate fp64 + launch-bound execution could make the
  halved dispatch a REAL win — so NEGATIVE for CPU/dev-GPU, UNKNOWN (possibly positive) for datacenter GPUs, worth
  re-benchmarking there; plus a magnetic
  convergence-path wrinkle. Do NOT re-pitch spin/k-batch dispatch fusion);
  CampaignTolLadder ALREADY PROTOTYPED & MEASURED A WIN (branch feat/tol-ladder-relax, ~1.29x on a metal
  RELAXATION, -33% SCF iters, exact via full-tol re-solve; loose-first-step beats tight; RELAX-ONLY) — do
  NOT re-propose an fmax-driven / force-keyed outer-SCF stopping tol for relaxations; it's done.

## Already SHIPPED wall-time work (seed exclusion — a NET-NEW mechanism beyond these is fine,
## a restatement is not; the Vet phase will confirm against the live repo + PRs):
- Cold-start: threaded spherical-Bessel form-factor transform (pseudo/radial.py sbt, #19);
  form factors built once as splines on a q-grid with cell headroom and interpolated per cell
  (pseudo/kb.py beta_form_factors); per-species radial tables + augmentation tables cached
  (radial_torch.py, uspp_setup._aug_tables_cached); cache Ewald + local-PP over the frozen-position
  SCF loop (#21); cache per-(species,channel) beta_of_g in the strained projector (#180); memoized
  PW-basis+symmetry rebuild keyed on grid/op-set, positions-only moves reuse it (calculator._get_system,
  #121); parsed UPF/PAW cached per symbol; good-FFT-size box selection (grids.good_fft_size); lazy
  imports of pandas/matplotlib; sane default CPU thread count min(cores,8) (_threads.py, #77); a warm
  fork-server for campaign setup cost is DEFERRED/PROMISING (open entry in ideas.md, not yet built).
- Dispatch/batching: k-batched Davidson over (nk,nb,npw) with NO band-locking BY DESIGN (locking would
  break the uniform batch); batched H-apply (core/batch.py BatchedHamiltonian arena, _tab_cache); batch
  the PAW one-center correction over atoms, device-aware tables (#124); rfft the density->Hartree round
  trip (#91); CPU-offload the batched-QR / Davidson subspace solve on CUDA gated on measured fp64
  (#134/#174/#214, davidson._qr_offload); sync-free Davidson convergence flags (pinned-memory + CUDA
  events); torch.compile ONLY on the XC energy-density layer (core/xc/base._XCCompileMixin, #251, needs
  openssl on PATH or it silently runs eager).
- Redundant work: density extrapolation across ionic steps ("none"/"reuse"/"linear"/"quadratic",
  QE-style least-squares coeffs; calculator._warm_start, #231/#261); seed Davidson with the previous
  ionic step's eigenvectors (reuse_wavefunctions, #127); remap warm-start density AND coeffs across an
  ecut-fixed FFT-grid resize for EOS/variable-cell (calculator._remap_density_to_grid, #125); checkpoint
  warm-start bridge (checkpoint.as_start_from); distributed k-shard warm-start (scf/distributed.shard_start_from);
  memoized symmetry setup + single-pass properties (#121). NOTE the measured limits: same-position restart
  9->2 iters; band-path chunk warm-start HURTS (near-degenerate seeds stall Davidson ~2.5x, band paths are
  cold-started); cross-k subspace transport & cross-SCF invariant-subspace deflation were TRIED and shelved
  (batching-hostile).
- Convergence: QE-style ADAPTIVE DAVIDSON TOLERANCE already ships (scf/common.adaptive_diago_tol: loose
  ~1e-3 early, tightening as ~0.1*r^2/nelec) — do NOT re-propose "loosen the eigensolver early"; auto
  mixing-scheme resolvers incl. magnetic-aware johnson default (#90/#150/#205/#207); energy-metric
  convergence gate 1/2<r|K_Hxc|r> < entol (#210/#215); NC adaptive backoff halving the step + dropping DIIS
  history on a stalled residual (noncollinear._nc_adaptive_backoff); PulayMixer per-block adaptive damping;
  soft-mode deflation for stalled clusters (scf/soft_mode.py, solvers/deflation.py); occupation-aware
  adaptive band budget is MARGINAL (only the empty-headroom straggler gate survived); quasi-Newton
  mixer-Jacobian transfer across scan steps is MARGINAL and collides with the Stoner-poisoning negative.

## Reusable substrate (for "what it builds on")
core/fftbox.py (FFT convention), grids.py (G-sphere/FFT grid), core/batch.py + core/hamiltonian.py
(batched H-apply, BatchedHamiltonian arena), solvers/davidson.py, scf/loop.py + scf/mixing.py +
scf/common.py (adaptive_diago_tol) + scf/implicit.py (chi0/K_Hxc response, IFT adjoint), pseudo/radial.py
(sbt form-factor) + pseudo/kb.py (form-factor splines), calculator.py (warm-start machinery), checkpoint.py,
dtypes.py. Prior-work catalogue: docs/ideas.md ("Performance and scaling" + "Acceleration idea rounds:
evaluated and shelved") and docs/manual/performance.md (authoritative what-helps/what-does-not, which names
k-chunking the batched apply as a still-missing path).

## The bar
A concrete wall-time mechanism a skeptic can evaluate, that helps SMALL/MEDIUM METAL cells,
is NET-NEW beyond the shipped list, stays differentiable and exact-at-convergence, and names its
one hard blocker honestly. A measured 1.2-2x on the common path beats a vague 10x on a wrong-axis
regime. No preconditioners, no learned seeds, no insulator-only, no large-cell-only.`

const IDEA_POOL_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ideas'],
  properties: {
    ideas: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'pain_area', 'pitch', 'mechanism', 'wall_lever', 'metal_small_cell_fit', 'hard_blocker'],
      properties: {
        name: { type: 'string' },
        pain_area: { type: 'string', enum: ['dispatch', 'cold_start', 'redundant_work', 'convergence'] },
        pitch: { type: 'string', description: '2-4 sentences: the wall-time idea' },
        mechanism: { type: 'string', description: 'the concrete technical mechanism, not a vibe' },
        wall_lever: { type: 'string', description: 'WHERE the time goes today and the rough speedup, on WHAT cell size' },
        metal_small_cell_fit: { type: 'string', description: 'why it helps SMALL/MEDIUM METAL cells (not only insulators, not only large cells)' },
        hard_blocker: { type: 'string', description: 'the single hardest obstacle, stated honestly' },
      },
    } },
  },
}

const RANK_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ranked', 'dropped_note'],
  properties: {
    ranked: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'pain_area', 'pitch', 'mechanism', 'wall_lever', 'why_credible', 'hard_blocker'],
      properties: {
        name: { type: 'string' }, pain_area: { type: 'string' }, pitch: { type: 'string' },
        mechanism: { type: 'string' }, wall_lever: { type: 'string' },
        why_credible: { type: 'string', description: 'is there a real mechanism + honest magnitude, or is it vague?' },
        hard_blocker: { type: 'string' },
      },
    } },
    dropped_note: { type: 'string', description: 'what was merged, and what was dropped as out-of-scope / already-shipped / vague' },
  },
}

const VET_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name', 'already_in_gradwave', 'where_it_exists', 'net_new_delta', 'prior_art', 'builds_on', 'wall_win', 'the_blocker', 'blocker_tractable', 'in_scope', 'feasibility', 'verdict', 'sources'],
  properties: {
    name: { type: 'string' },
    already_in_gradwave: { type: 'string', enum: ['absent', 'partial', 'shipped', 'tried_and_shelved'],
      description: 'checked against src/gradwave, GH PRs (gh pr list/view), and docs/ideas.md' },
    where_it_exists: { type: 'string', description: 'file/symbol, PR number(s), and/or ideas.md section that already covers it (or "nothing found")' },
    net_new_delta: { type: 'string', description: 'if partial/shipped/tried: what, if anything, is genuinely NEW here vs what exists' },
    prior_art: { type: 'string', description: 'has anyone done this in DFT or the source field (HPC/ML-sys/numerics)? result? cite' },
    builds_on: { type: 'string' },
    wall_win: { type: 'string', description: 'honest wall-time payoff with rough magnitude and the cell regime it applies to' },
    the_blocker: { type: 'string', description: 'the one hardest technical obstacle' },
    blocker_tractable: { type: 'string', description: 'plausibly solvable or fatal? honest read' },
    in_scope: { type: 'string', enum: ['yes', 'borderline', 'no'], description: 'no = insulator-only, large-cell-only, a preconditioner, or a learned seed' },
    feasibility: { type: 'string', enum: ['quick_win', 'moderate', 'hard', 'research_project'] },
    verdict: { type: 'string', enum: ['pursue', 'promising', 'interesting_longshot', 'already_done', 'out_of_scope', 'fatal_flaw'] },
    sources: { type: 'array', items: { type: 'string' } },
  },
}

const REPORT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ranked', 'quick_wins', 'best_bets', 'already_done', 'summary'],
  properties: {
    ranked: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'pain_area', 'one_line', 'verdict', 'wall_win', 'net_new_delta', 'the_blocker', 'builds_on', 'feasibility'],
      properties: {
        name: { type: 'string' }, pain_area: { type: 'string' }, one_line: { type: 'string' },
        verdict: { type: 'string', enum: ['pursue', 'promising', 'interesting_longshot', 'already_done', 'out_of_scope', 'fatal_flaw'] },
        wall_win: { type: 'string' }, net_new_delta: { type: 'string' }, the_blocker: { type: 'string' },
        builds_on: { type: 'string' }, feasibility: { type: 'string' },
      },
    } },
    quick_wins: { type: 'array', items: { type: 'string' }, description: 'net-new, tractable, small-diff wall-time wins worth doing first' },
    best_bets: { type: 'array', items: { type: 'string' }, description: 'highest wall-time-lever x tractability, net-new' },
    already_done: { type: 'array', items: { type: 'string' }, description: 'ideas that turned out already shipped or tried-and-shelved (with the pointer)' },
    summary: { type: 'string', description: 'which pain area has the most live net-new headroom on the common metal-cell path, and the single best next thing to build' },
  },
}

const LENSES = [
  { key: 'dispatch', title: 'Dispatch / eager overhead',
    brief: `Cut Python + ATen per-op dispatch, kernel-launch count, host-device syncs, and redundant
allocations in the hot inner work (H-apply, Davidson subspace + Rayleigh-Ritz, mixing) at the SMALL
tensor sizes gradwave actually runs, where per-op overhead is a real fraction of wall. Fair game:
compiling / capturing LARGER regions than just XC (torch.compile on the H-apply or the whole Davidson
step, with an eager fallback), fusing the projector GEMMs or the FFT+multiply+iFFT chain, reusing
preallocated buffers to kill per-iteration allocation, removing .item()/.cpu() syncs, cutting per-band
Python loops in favor of a single batched launch. Respect: whole-step CUDA graphs already measured null,
FFT arena already ships, torch.compile already wraps XC, Davidson is already k-batched and sync-free — go
BEYOND those. Must stay differentiable and give the identical fixed point. Name what breaks under autograd
/ graph capture.` },
  { key: 'cold_start', title: 'Cold-start / setup / first-run tax',
    brief: `Attack the one-time cost paid before or at the very start of an SCF, which dominates SHORT
runs and every first run: package import time, the pseudopotential radial FT / form-factor build, the
Ewald + local-PP build, FFT-size selection / planning, and the first SCF step from a cold guess. Fair
game: precomputing / serializing form factors and Ewald tables keyed by (species, ecut, cell) so a
warm run skips the build across PROCESSES (the in-process caches already exist; a persistent on-disk
cache does not); lazy-importing heavy deps off the core path; a cheaper-but-exact first-step
Hamiltonian; overlapping setup with the first diagonalization; trimming import-time work. Respect: the
form-factor transform is already threaded (#19) and spline-cached (pseudo/kb.py), Ewald/local-PP already
cached WITHIN a frozen-position loop (#21), PW-basis+symmetry rebuild memoized (#121), thread count
already capped (#77), and a warm fork-server is already DEFERRED/PROMISING in ideas.md (you may sharpen
its exact-and-differentiable design, not merely restate it). Must be exact.` },
  { key: 'redundant_work', title: 'Redundant / re-done work (reuse & incremental update)',
    brief: `Find work that is recomputed when it did not change — within one SCF, across SCF steps,
across k-points, or across a campaign (relax / EOS / elastic / phonon / parameter scan) — and reuse,
transport, or incrementally update it instead of rebuilding. Fair game: an incremental H / potential
update when only part of the input changed (e.g. rebuild only the changed projector columns, already
done for STRAIN in #180 — generalize to atomic DISPLACEMENT); reusing converged subspaces or
factorizations across nearby campaign points where it survives the uniform batch; a persistent
content-addressed store of converged states keyed by geometry for a scan (differentiable-safe);
avoiding a full rebuild of the local potential when only the density moved a little. Respect: density
extrapolation across ionic steps (#231/#261), Davidson eigenvector seeding across ionic steps (#127),
warm-start remap across grid changes (#125), memoized symmetry (#121), and beta_of_g STRAIN caching
(#180) ALL ship; band-path chunk warm-start HURTS and is deliberately cold-started; cross-k transport
and cross-SCF deflation were shelved as batching-hostile. Go beyond. Must not change the converged
answer or break the graph.` },
  { key: 'convergence', title: 'Convergence behavior (fewer iters to the SAME answer)',
    brief: `Reduce the NUMBER of SCF iterations and eigensolver passes to the identical converged
result WITHOUT proposing a new mixing preconditioner and WITHOUT a learned/ML seed (both hard-excluded),
and WITHOUT re-proposing the adaptive Davidson tolerance or fp32-draft that ALREADY ship. What is still
open lives in SCHEDULE, STOPPING, and EFFORT ALLOCATION: smarter stopping rules that halt exactly when a
target OBSERVABLE (energy, force, stress) is converged rather than over-converging the density; adaptive
mixing-PARAMETER / history-DEPTH / restart control (managing the EXISTING mixer's schedule, not its
operator); detecting and short-circuiting charge-sloshing or limit cycles earlier; concentrating
eigensolver effort where the residual actually lives without breaking the uniform (nk,nb,npw) batch
(note: per-band locking is deliberately avoided today because it breaks the batch — an idea here must
say how it keeps the batch uniform or why the trade nets out); coupling the SCF stopping tolerance to
the cheapest sufficient eigensolver accuracy. Respect: adaptive_diago_tol, the energy-metric gate (#210),
auto mixing resolvers (#207), NC adaptive backoff, and that the outer count is MIXING-limited (so honest
wins here are stopping precision and effort allocation, not a new dielectric). Every idea must reach the
same fixed point; say why.` },
]

phase('Ideate')
const pools = await parallel(LENSES.map((L) => () => agent(
  BRIEF + '\n\n## YOUR PAIN-AREA LENS: ' + L.title + '\n' + L.brief +
  '\n\nPropose 6 WALL-TIME ideas through this lens for the COMMON case (small/medium METAL cells, fp64, ' +
  'eager PyTorch). Every idea must be NET-NEW beyond the shipped list, stay autograd-differentiable, and ' +
  'give the IDENTICAL converged answer. Nothing on the out-of-scope list (no preconditioners, no learned ' +
  'seeds, no insulator-only, no large-cell-only, no re-proposing adaptive diago tol / fp32 draft). For each: ' +
  'a memorable name, the pain_area, a 2-4 sentence pitch, the CONCRETE mechanism (not a vibe), the wall_lever ' +
  '(where the time goes today + rough speedup on what cell size), why it fits small/medium METAL cells, and ' +
  'the single hardest blocker stated honestly. A credible 1.2-2x on the real path beats a vague 10x on the ' +
  'wrong axis.',
  { schema: IDEA_POOL_SCHEMA, phase: 'Ideate', label: 'ideate:' + L.key, effort: 'high' }
)))
const candidates = pools.filter(Boolean).flatMap((p) => p.ideas)
log('Ideation produced ' + candidates.length + ' wall-time candidates across ' + LENSES.length + ' pain areas')

phase('Rank')
const rank = await agent(
  BRIEF + '\n\n## CANDIDATE POOL (' + candidates.length + ')\n' + JSON.stringify(candidates, null, 2) +
  '\n\nYou are the curator for a WALL-TIME round. Dedup/merge. DROP anything out-of-scope (insulator-only, ' +
  'large-cell-only, a preconditioner in any framing, a learned/ML seed), anything that merely restates a ' +
  'shipped item with no net-new delta (adaptive diago tol, fp32 draft, density extrapolation, form-factor ' +
  'caching, etc.), and anything vague with no evaluable mechanism or honest magnitude. Keep only ideas that ' +
  'plausibly help SMALL/MEDIUM METAL cells and stay differentiable + exact-at-convergence. RANK by (wall-time ' +
  'lever on the common path) x (tractability / small-diff-first) x (credibility of the mechanism), keeping ' +
  'coverage across the four pain areas. Return the top 12, each with why it is credible (real mechanism + ' +
  'honest magnitude vs vague), its wall_lever, and its hardest blocker, plus a note on what you merged and ' +
  'what you dropped as out-of-scope / already-shipped / vague.',
  { schema: RANK_SCHEMA, phase: 'Rank', effort: 'high' }
)
const top = rank.ranked.slice(0, 12)
log('Ranked ' + top.length + ' wall-time ideas forward to vetting')

phase('Vet')
const vetted = await parallel(top.map((idea) => () => agent(
  BRIEF + '\n\n## IDEA TO VET\n' + JSON.stringify(idea, null, 2) +
  '\n\nVet this wall-time idea HONESTLY. The single most important check: DOES IT ALREADY EXIST? Run all three:\n' +
  '(1) CODE: grep/read src/gradwave at the gradwave path for the relevant symbols (scf/mixing.py, scf/common.py, ' +
  'solvers/davidson.py, scf/loop.py, core/hamiltonian.py, core/batch.py, pseudo/radial.py, pseudo/kb.py, ' +
  'calculator.py, checkpoint.py, etc.).\n' +
  '(2) GH PRs: run `gh pr list --state all --search "<keywords>" --json number,title,state` and ' +
  '`gh pr view <n>` for any hit, to see if a merged/open PR already did it (the perf(...) / feat(scf) / ' +
  'feat(relax) PRs are the ones to check).\n' +
  '(3) docs/ideas.md + docs/manual/performance.md: read the "Performance and scaling" and "Acceleration idea ' +
  'rounds: evaluated and shelved" sections for a prior verdict on this exact idea or its class.\n' +
  'Set already_in_gradwave to absent / partial / shipped / tried_and_shelved and record where_it_exists ' +
  '(file/symbol, PR#, or ideas.md section). If partial/shipped/tried, state the net_new_delta — what, if ' +
  'anything, is genuinely new here (be strict; "same idea, different words" = no delta).\n' +
  'Then: (4) PRIOR ART in DFT or the source field (HPC / ML-systems / numerics), cite real sources. ' +
  '(5) builds_on. (6) wall_win: honest magnitude + cell regime. (7) the_blocker + blocker_tractable. ' +
  '(8) in_scope (no = insulator-only / large-cell-only / a preconditioner / a learned seed). ' +
  '(9) feasibility (quick_win/moderate/hard/research_project) and verdict (pursue / promising / ' +
  'interesting_longshot / already_done / out_of_scope / fatal_flaw). Cite sources.',
  { schema: VET_SCHEMA, phase: 'Vet', agentType: 'general-purpose', effort: 'medium', label: 'vet:' + idea.name }
)))
const results = vetted.filter(Boolean)
log('Vetted ' + results.length + ' ideas against code + PRs + ideas.md')

phase('Report')
const report = await agent(
  BRIEF + '\n\n## VETTED IDEAS\n' + JSON.stringify(results, null, 2) +
  '\n\nSynthesize the final ranked wall-time report. FIRST move every idea whose verdict is already_done or ' +
  'out_of_scope into their buckets with the pointer (PR#/file/ideas.md section) — do not rank those as live ' +
  'work. RANK the survivors by (verdict, then wall-time lever on the common metal-cell path, then blocker ' +
  'tractability / smallness of the diff). For each: name, pain_area, one-line, verdict, wall_win, ' +
  'net_new_delta, the one blocker, builds_on, feasibility. quick_wins = net-new, tractable, small-diff wins ' +
  'to do first. best_bets = highest lever x tractability, net-new. already_done = the ones that turned out ' +
  'shipped or tried-and-shelved, each with its pointer. The summary must say which of the four pain areas ' +
  'has the most LIVE net-new headroom on the common small/medium metal-cell path, and the single best next ' +
  'thing to build. Honest and concrete; a measured 1.2-2x that is real beats a hypothetical 10x on the ' +
  'wrong axis, and "this already ships in PR #X" is a valid and valuable finding.',
  { schema: REPORT_SCHEMA, phase: 'Report', effort: 'high' }
)
return { report, results, rank_note: rank.dropped_note }
