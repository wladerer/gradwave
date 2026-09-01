export const meta = {
  name: 'campaign-batching-brainstorm',
  description: 'CAMPAIGN-LEVEL wall-time round: what can we PARALLELIZE across workers or BATCH/REDUCE into a single calculation across a multi-config campaign (EOS/elastic/phonon/relax/scan/delta-gauge/inverse-design) -> rank by campaign wall lever x tractability -> vet against code + GH PRs + ideas.md -> report',
  phases: [
    { title: 'Ideate', detail: '5 lenses (batch-into-one / parallelize / reduce / transport / pipeline) generate a campaign pool' },
    { title: 'Rank', detail: 'rank by campaign wall lever x tractability x not-already-done' },
    { title: 'Vet', detail: 'per idea: grep src + gh pr list + ideas.md -> already exist? + prior art + honest lever + differentiable-or-forward-only' },
    { title: 'Report', detail: 'ranked, net-new-only, with the one hard blocker each' },
  ],
}

const BRIEF = `# gradwave CAMPAIGN-LEVEL wall-time round — parallelize, or stuff into ONE calculation

## The mandate (read first)
The SINGLE-SCF small-cell axis is EXHAUSTED (measured over three prior rounds + four prototypes: dispatch
micro-opts ~1-2%, torch.compile 0.71-0.92x, inexact-Newton wall-negative, spin-batched Davidson negative;
the shipped stack — batched Davidson, adaptive diago tol, johnson+spin-precond mixing, IBZ symmetry — is at
the practical eager-fp64 floor). This round changes SCOPE: reduce the wall-clock time of MULTI-CONFIGURATION
CAMPAIGNS, by two verbs — (1) PARALLELIZE independent work across cores/processes/machines, and (2) BATCH or
REDUCE many configurations into a SINGLE calculation so per-op dispatch AND per-config setup amortize. Plus
TRANSPORT converged information across points, and PIPELINE heterogeneous work.

## The campaign workloads (the target)
- EOS: energy vs volume (a few-to-10 volumes, same chemistry, different cell scale).
- Elastic: stress vs strain (6-24 strained cells from one reference).
- Phonon-FD: forces vs atomic displacement (up to 6*N_atom supercell SCFs from one reference).
- Relax: a SEQUENTIAL chain of ionic steps (NOT embarrassingly parallel — each needs the prior geometry).
- Delta-gauge / Delta-benchmark: elements x volumes (hundreds of independent small SCFs).
- Parameter scans (ecut/kmesh/U/lattice), inverse design (a gradient loop over a differentiable loss).

## What gradwave is
Differentiable plane-wave DFT in PyTorch (pure eager, fp64/complex128, sub-meV vs QE). Already BATCHES over
k-points and bands inside one eigensolve (davidson over (nk,nb,npw)); end-to-end autograd (learned XC/pseudos,
inverse design, exact response/Hessians). asus is a 22-core + RTX-3050 synced peer (ssh asus, dask pattern).

## Hard constraints (respect or credibly address)
- EXACT AT CONVERGENCE: sub-meV per-config results must be unchanged; a campaign speedup may only change HOW
  the set is computed, not each answer.
- METALS covered (partial occupations, per-config Fermi level).
- AUTOGRAD IS PROCESS-LOCAL and does NOT serialize: a SINGLE differentiable computation (learned-XC training,
  inverse-design gradient, exact response) CANNOT be split across processes — only INDEPENDENT FORWARD SCFs
  distribute. Say for each idea whether it is DIFFERENTIABLE (one graph: batch-into-one-tensor-op, symmetry
  reconstruction, response-tangent) or FORWARD-ONLY (multi-process/distributed). Batching configs into one
  tensor op KEEPS autograd; forking SCFs across processes does NOT.
- HARDWARE TARGETS (BOTH are production): CPU-fp64 (dev/CI, 22-core) AND DATACENTER fp64 GPUs (A100/H100)
  that gradwave PRODUCTION-TESTS on — full-rate fp64, where small-op DFT work is LAUNCH/DISPATCH-BOUND. So
  ideas that batch/fuse many small ops into fewer larger kernels (batch-into-one-tensor multi-config SCF,
  config/spin batching, fused applies) are REAL wins there: do NOT discredit an idea as "hardware-gated" or
  "needs a card we don't own" — we own them, and ideas.md ~1700's "gates the win to A100/H100" is a green
  light here, not a disqualifier. The RTX 3050 (asus) is a crippled-fp64 (1/64) DEV card for correctness/
  plumbing ONLY, never the perf verdict for an fp64-GPU idea. (A purely fp32-GPU-only win is still out — sub-meV needs fp64.)
- Pure-eager PyTorch; a new kernel/dep is allowed but named as a cost.

## Already SHIPPED / KNOWN (seed exclusion — a NET-NEW mechanism beyond these is fine, a restatement is not):
- IBZ k-point symmetry reduction (symmetry.reduce_mesh + RhoSymmetrizer) — the biggest shipped "compute only
  the irreducible set" win; magnetic (Shubnikov) folding too.
- Warm-start chains: density extrapolation + wavefunction reuse across ionic steps (#231/#127/#261); EOS
  sequential chaining; the warm-start survives an ecut-fixed grid resize (#125).
- Hub-and-spoke seeding: elastic seeds every strain spoke from the unstrained ref (#41/#229); phonon-FD seeds
  every +-displacement spoke from the undisplaced ref (phonons_supercell); Born-von-Karman TRANSLATIONAL
  reduction of the FD supercell already ships.
- Distributed k-point SCF sharding (scf/distributed, shard_start_from) — with a documented multi-GPU GATHER
  blocker (ideas.md ~1347). asus offload for embarrassingly-parallel sweeps; dask pattern in CLAUDE.md.
- Batched multi-structure SCF + the EOS-on-GPU question is ANALYZED in ideas.md (~1700) — READ it, extend it,
  do not re-derive it.
- DEFERRED/OPEN (may sharpen the exact+differentiable design, not merely restate): warm fork-server for
  cross-process setup amortization (ideas.md ~1818); Continuation-Sweep (analytic drho/dlambda response tangent
  drives EOS/elastic/scan as EXACT derivatives from one solve — gated on the metallic Fermi-surface response
  seed in scf/implicit.py, which IS now metal-capable: apply_chi0 has the metal path); GreedyManifold
  reduced-basis POD-Galerkin over a sweep (5-20x campaign, research-project); Delta-tier multifidelity; SplitFinish
  (fp32 bands/DOS overlapped vs fp64 SCF — evaluated SMALL, phonon example was physically wrong).
- BEING BUILT THIS SESSION — DO NOT re-propose: **DisplacementStar** = point-group reduction of the PHONON-FD
  displacement set (compute irreducible displacements, reconstruct the rest by group action). The ELASTIC-strain
  analog, batched-multi-config eigensolve, and other reductions ARE in scope.
- OUT: single-SCF per-iteration/dispatch micro-opts (that axis is exhausted — see the wall-time-brainstorm
  workflow); splitting a single autograd graph across processes (impossible); a new mixing preconditioner or a
  learned/ML seed (walked back); a learned surrogate as the WHOLE idea unless it carries an exact corrector.

## Reusable substrate
core/batch.py (batched (nk,nb,npw) machinery — the axis to extend with a config dimension), solvers/davidson.py,
scf/loop.py + scf/implicit.py (metal-capable apply_chi0/apply_k_hxc response — the continuation tangent), symmetry.py
(find_spacegroup, group action, atom_map, RhoSymmetrizer), postscf/phonons_supercell.py + phonons.py
(HessianSymmetry._select/.reconstruct), postscf/eos.py + elastic + api.run_eos/run_elastic/relax, scf/distributed.py,
calculator warm-start. Prior-work catalogue: docs/ideas.md ("Performance and scaling" + "evaluated and shelved").

## The bar
A concrete CAMPAIGN wall-time mechanism a skeptic can evaluate: name the workload (EOS/elastic/phonon/scan/
delta-gauge/inverse-design), the parallel-or-batch lever with rough magnitude, whether it stays EXACT and whether
it is DIFFERENTIABLE (one graph) or FORWARD-ONLY (multi-process), and the one hard blocker. A credible 2-5x on a
real campaign beats a vague 10x. No single-SCF micro-opts, no autograd-graph-splitting, no preconditioners/seeds.`

const IDEA_POOL_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ideas'],
  properties: {
    ideas: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'lens', 'campaign_workload', 'pitch', 'mechanism', 'campaign_lever', 'diff_or_forward', 'hard_blocker'],
      properties: {
        name: { type: 'string' },
        lens: { type: 'string', enum: ['batch_into_one', 'parallelize', 'reduce', 'transport', 'pipeline'] },
        campaign_workload: { type: 'string', description: 'EOS / elastic / phonon-FD / relax / scan / delta-gauge / inverse-design (which one(s))' },
        pitch: { type: 'string', description: '2-4 sentences: the campaign idea' },
        mechanism: { type: 'string', description: 'the concrete technical mechanism, not a vibe' },
        campaign_lever: { type: 'string', description: 'WHERE the campaign time goes today + rough speedup, on WHICH workload' },
        diff_or_forward: { type: 'string', enum: ['differentiable', 'forward_only'], description: 'one autograd graph, or independent forward SCFs across processes' },
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
      required: ['name', 'lens', 'campaign_workload', 'pitch', 'mechanism', 'campaign_lever', 'why_credible', 'hard_blocker'],
      properties: {
        name: { type: 'string' }, lens: { type: 'string' }, campaign_workload: { type: 'string' },
        pitch: { type: 'string' }, mechanism: { type: 'string' }, campaign_lever: { type: 'string' },
        why_credible: { type: 'string' }, hard_blocker: { type: 'string' },
      },
    } },
    dropped_note: { type: 'string', description: 'what merged, and what dropped as out-of-scope / already-shipped / vague' },
  },
}

const VET_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name', 'already_in_gradwave', 'where_it_exists', 'net_new_delta', 'prior_art', 'builds_on', 'campaign_win', 'diff_or_forward', 'the_blocker', 'blocker_tractable', 'in_scope', 'feasibility', 'verdict', 'sources'],
  properties: {
    name: { type: 'string' },
    already_in_gradwave: { type: 'string', enum: ['absent', 'partial', 'shipped', 'tried_and_shelved'] },
    where_it_exists: { type: 'string', description: 'file/symbol, PR number(s), and/or ideas.md section (or "nothing found")' },
    net_new_delta: { type: 'string', description: 'if partial/shipped/tried: what is genuinely NEW here vs what exists' },
    prior_art: { type: 'string', description: 'has anyone done this in DFT/HPC/ML-sys? result? cite' },
    builds_on: { type: 'string' },
    campaign_win: { type: 'string', description: 'honest campaign wall payoff + which workload + rough magnitude' },
    diff_or_forward: { type: 'string', enum: ['differentiable', 'forward_only'] },
    the_blocker: { type: 'string' },
    blocker_tractable: { type: 'string', description: 'plausibly solvable or fatal? honest read' },
    in_scope: { type: 'string', enum: ['yes', 'borderline', 'no'], description: 'no = single-SCF micro-opt / autograd-graph-split / preconditioner / learned seed' },
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
      required: ['name', 'lens', 'campaign_workload', 'one_line', 'verdict', 'campaign_win', 'diff_or_forward', 'net_new_delta', 'the_blocker', 'builds_on', 'feasibility'],
      properties: {
        name: { type: 'string' }, lens: { type: 'string' }, campaign_workload: { type: 'string' }, one_line: { type: 'string' },
        verdict: { type: 'string', enum: ['pursue', 'promising', 'interesting_longshot', 'already_done', 'out_of_scope', 'fatal_flaw'] },
        campaign_win: { type: 'string' }, diff_or_forward: { type: 'string' }, net_new_delta: { type: 'string' },
        the_blocker: { type: 'string' }, builds_on: { type: 'string' }, feasibility: { type: 'string' },
      },
    } },
    quick_wins: { type: 'array', items: { type: 'string' } },
    best_bets: { type: 'array', items: { type: 'string' } },
    already_done: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string', description: 'which lens x workload has the most live net-new campaign headroom, and the single best next thing to build' },
  },
}

const LENSES = [
  { key: 'batch_into_one', title: 'Batch many configs into ONE calculation',
    brief: `Fold N campaign configurations into a SINGLE batched calculation — config as a new batch axis
alongside k and band — so per-op ATen/Python dispatch AND per-config setup amortize across the whole set, in
ONE autograd graph (stays differentiable). Fair game: a batched multi-structure SCF/eigensolve over EOS volumes /
elastic strains / phonon displacements / a set of related structures that share cell/ecut/grid; a single
davidson over (nconfig*nk, nb, npw); batching the density/potential/XC/mixing over the config axis. Respect: the
(nk,nb,npw) batch, IBZ reduction, and the ideas.md ~1700 batched-multi-structure-SCF analysis all exist — go
BEYOND. Name what breaks: per-config Fermi level / occupations, different npw or grid per volume (EOS changes the
cell), heterogeneous convergence across configs (uniform no-lock batch rides the slowest), and the memory of
holding N configs at once. Say why it's still exact and how gradients survive.` },
  { key: 'parallelize', title: 'Parallelize independent work across cores/processes/machines',
    brief: `Distribute the embarrassingly-parallel FORWARD SCFs of a campaign across CPU cores / processes /
the asus 22-core+GPU peer (dask), with SHARED setup amortization so each worker skips the cold start. Fair game:
a warm process pool / fork-server that pays torch-import + pseudo-setup ONCE and forks per config (COW); a
work-stealing scheduler over EOS/elastic/phonon/delta-gauge points; CPU+GPU co-scheduling; retry/checkpoint. This
is FORWARD-ONLY (autograd graphs are process-local — say so; inverse-design/learned-XC gradients cannot use this).
Respect: k-point sharding + shard_start_from ship (with the multi-GPU gather blocker); asus offload + the dask
pattern exist; the warm fork-server is DEFERRED/PROMISING (sharpen its exact design, don't restate). Name the
realistic lever (fixed per-process cost ~3.5s import; so the win scales with #configs x fixed-cost / total).` },
  { key: 'reduce', title: 'Compute only the symmetry-irreducible work set',
    brief: `Compute only the group-IRREDUCIBLE members of a campaign and RECONSTRUCT the rest by symmetry —
exact, differentiable, the same lever IBZ k-reduction already exploits, extended to the CONFIG set. Fair game: the
ELASTIC analog of DisplacementStar — irreducible STRAINS (the strain tensor transforms under the point group; a
cubic metal needs 3 independent C_ij not 6-21 strained SCFs); irreducible STRUCTURES in a symmetric scan; higher
site-symmetry exploitation; combining config-symmetry with the shipped IBZ. Respect: IBZ k-reduction + phonon
Born-von-Karman translational reduction ship, and the PHONON-FD point-group displacement reduction (DisplacementStar)
is ALREADY BEING BUILT — do NOT restate it; the elastic-strain and structure-set reductions are the net-new targets.
Name the reconstruction transform and the low-symmetry / non-symmorphic corner cases that silently corrupt if wrong.` },
  { key: 'transport', title: 'Transport / continue converged info across campaign points',
    brief: `Reuse converged information from one campaign point to slash (or ANALYTICALLY ELIMINATE) the next,
beyond the shipped density warm-start. Fair game: RESPONSE-TANGENT continuation — the analytic drho/dlambda from ONE
converged solve (via the now-metal-capable apply_chi0/apply_k_hxc + IFT) gives EOS/elastic/scan observables as EXACT
DERIVATIVES, reading C_ij / B0 / Gruneisen off one solve instead of a stencil of SCFs (the deferred Continuation-Sweep);
transported subspaces/factorizations across nearby points where they survive batching; a reduced-basis / POD surrogate
built from a few solves with an EXACT residual corrector (differentiable). Respect: density extrapolation + wf-reuse
ship; Continuation-Sweep + GreedyManifold are DEFERRED (the metallic response seed that gated them is now available —
that's the unlock). Stays differentiable. Name the blocker (response conditioning near the Fermi surface / soft modes,
non-affine parameter dependence needing EIM/DEIM).` },
  { key: 'pipeline', title: 'Pipeline / overlap heterogeneous campaign work',
    brief: `Overlap work that today runs serially, so a campaign's wall is the max not the sum. Fair game: overlap
the fp32-tolerant POST-processing (bands / DOS / PDOS) or the NEXT config's setup (form factors, Ewald, grid) with the
CURRENT config's fp64 SCF; CPU+GPU co-scheduling (fp64 SCF on the 22 CPU cores while fp32-tolerant kernels run on the
3050); overlap checkpoint I/O and analysis with compute; prefetch the next relax/line-search geometry's setup. Respect:
SplitFinish (fp32 bands overlapped vs fp64 SCF) was evaluated SMALL and its phonon example was physically wrong — go
beyond it and be honest about the real overlap available. Say which parts are forward-only and where the fp64/fp32
accuracy boundary sits. Name the blocker (dependency chains that don't actually overlap; GIL/threading; the win is
capped at the smaller of the two overlapped costs).` },
]

phase('Ideate')
const pools = await parallel(LENSES.map((L) => () => agent(
  BRIEF + '\n\n## YOUR LENS: ' + L.title + '\n' + L.brief +
  '\n\nPropose 6 CAMPAIGN-LEVEL ideas through this lens. Every idea must be NET-NEW beyond the shipped/known list, ' +
  'target a named campaign workload, and give the IDENTICAL per-config converged answer. Nothing on the out-of-scope ' +
  'list (no single-SCF micro-opts, no autograd-graph-splitting, no preconditioners/learned-seeds, do not restate ' +
  'DisplacementStar or the fork-server verbatim). For each: a memorable name, the lens, the campaign_workload, a 2-4 ' +
  'sentence pitch, the CONCRETE mechanism, the campaign_lever (where the campaign time goes today + rough speedup on ' +
  'which workload), whether it is differentiable (one graph) or forward_only (multi-process), and the single hardest ' +
  'blocker stated honestly. A credible 2-5x on a real campaign beats a vague 10x.',
  { schema: IDEA_POOL_SCHEMA, phase: 'Ideate', label: 'ideate:' + L.key, effort: 'high' }
)))
const candidates = pools.filter(Boolean).flatMap((p) => p.ideas)
log('Ideation produced ' + candidates.length + ' campaign candidates across ' + LENSES.length + ' lenses')

phase('Rank')
const rank = await agent(
  BRIEF + '\n\n## CANDIDATE POOL (' + candidates.length + ')\n' + JSON.stringify(candidates, null, 2) +
  '\n\nYou are the curator for a CAMPAIGN-LEVEL round. Dedup/merge. DROP anything out-of-scope (single-SCF micro-opt, ' +
  'autograd-graph-split, preconditioner, learned seed, a restatement of DisplacementStar / the fork-server / a shipped ' +
  'item with no net-new delta) and anything vague with no evaluable mechanism or honest magnitude. Keep ideas that ' +
  'plausibly help a real campaign, stay exact per-config, and are honestly labelled differentiable vs forward-only. RANK ' +
  'by (campaign wall lever) x (tractability) x (credibility), keeping coverage across the 5 lenses and the campaign ' +
  'workloads. Return the top 12, each with why it is credible, its campaign_lever, and its hardest blocker, plus a note ' +
  'on what you merged and dropped.',
  { schema: RANK_SCHEMA, phase: 'Rank', effort: 'high' }
)
const top = rank.ranked.slice(0, 12)
log('Ranked ' + top.length + ' campaign ideas forward to vetting')

phase('Vet')
const vetted = await parallel(top.map((idea) => () => agent(
  BRIEF + '\n\n## IDEA TO VET\n' + JSON.stringify(idea, null, 2) +
  '\n\nVet this campaign idea HONESTLY. Most important: DOES IT ALREADY EXIST? Run all three:\n' +
  '(1) CODE: grep/read src/gradwave at the gradwave path (core/batch.py, scf/loop.py, scf/implicit.py, scf/distributed.py, ' +
  'symmetry.py, postscf/{phonons_supercell,phonons,eos,elastic}.py, api.py run_eos/run_elastic/run_relax, calculator.py).\n' +
  '(2) GH PRs: `gh pr list --state all --search "<keywords>" --json number,title,state` + `gh pr view <n>` for hits ' +
  '(perf/feat(scf)/feat(postscf)/bench PRs).\n' +
  '(3) docs/ideas.md: "Performance and scaling" (esp. ~1700 batched-multi-structure-SCF, ~1347 distributed gather, ' +
  '~1818 fork-server, Continuation-Sweep / GreedyManifold) and "evaluated and shelved".\n' +
  'Set already_in_gradwave and where_it_exists. If partial/shipped/tried, state the strict net_new_delta. Then: prior_art ' +
  '(DFT/HPC/ML-sys, cite), builds_on, campaign_win (honest magnitude + workload), diff_or_forward, the_blocker + ' +
  'blocker_tractable, in_scope (no = single-SCF micro-opt / autograd-split / preconditioner / learned seed), feasibility, ' +
  'verdict. Cite sources.',
  { schema: VET_SCHEMA, phase: 'Vet', agentType: 'general-purpose', effort: 'medium', label: 'vet:' + idea.name }
)))
const results = vetted.filter(Boolean)
log('Vetted ' + results.length + ' campaign ideas against code + PRs + ideas.md')

phase('Report')
const report = await agent(
  BRIEF + '\n\n## VETTED IDEAS\n' + JSON.stringify(results, null, 2) +
  '\n\nSynthesize the final ranked campaign-level report. FIRST move already_done / out_of_scope ideas into their buckets ' +
  'with the pointer. RANK the survivors by (verdict, then campaign wall lever, then blocker tractability). For each: name, ' +
  'lens, campaign_workload, one-line, verdict, campaign_win, diff_or_forward, net_new_delta, the one blocker, builds_on, ' +
  'feasibility. quick_wins = net-new, tractable, small-diff campaign wins to do first. best_bets = highest lever x ' +
  'tractability, net-new. already_done = shipped/shelved with the pointer. The summary must say which lens x workload has ' +
  'the most LIVE net-new campaign headroom and the single best next thing to build (DisplacementStar is already being built ' +
  '— name the best thing BESIDES it). Honest and concrete; "this already ships / is analyzed in ideas.md ~1700" is a valid ' +
  'and valuable finding.',
  { schema: REPORT_SCHEMA, phase: 'Report', effort: 'high' }
)
return { report, results, rank_note: rank.dropped_note }
