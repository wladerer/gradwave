export const meta = {
  name: 'consensus-review',
  description: 'Single-objective consensus review: coverage fan-out, refute-to-survive verify, judge-panel remediation',
  whenToUse: 'Deep review of gradwave (or a subtree) for ONE objective at a time (physics-correctness | duplication | complexity). Run once per objective.',
  phases: [
    { title: 'Scope' },
    { title: 'Find' },
    { title: 'Verify' },
    { title: 'Global' },
    { title: 'Remediate' },
  ],
}

// ---------- parameters ----------
// The runtime may deliver args as a JSON *string* (scriptPath mode) or an object.
const ARGS = typeof args === 'string' ? JSON.parse(args) : (args ?? {})
const REPO = ARGS.repo ?? '/home/wladerer/github/gradwave'
const TARGET = ARGS.target ?? 'src/gradwave'            // dir or glob under REPO; '.' = whole tree
const OBJECTIVE = ARGS.objective ?? 'physics-correctness'
const MODEL = ARGS.model            // e.g. 'opus' — pins every subagent; default inherits the session model
const FILES_PER_FINDER = ARGS.filesPerFinder ?? 6
const MAX_FINDERS = ARGS.maxFinders ?? 20
const SKEPTICS = ARGS.skeptics ?? 3
const VERIFY_TOP_PER_SLICE = ARGS.verifyTopPerSlice ?? 10 // rest pass through flagged unverified
const TOP_THEMES = ARGS.topThemes ?? (budget.total ? Math.min(6, Math.max(2, Math.floor(budget.remaining() / 300000))) : 3)

// ---------- objective rubrics ----------
const RUBRICS = {
  'physics-correctness': `PHYSICAL & NUMERICAL CORRECTNESS for plane-wave DFT in PyTorch (Kohn-Sham; norm-conserving / ultrasoft / PAW pseudopotentials; autodiff for forces/stress/phonons; implicit differentiation through the SCF). Hunt for:
- Unit inconsistency. Base units are eV and Angstrom. Flag any hidden Rydberg/Hartree/Bohr mixups, missing or double conversions, hard-coded constants in the wrong unit system.
- Precision/dtype. Tensors must be float64 or complex128. Flag float32 leaks, real<->complex confusion, dtype-promotion that silently downcasts.
- Sign conventions and factor-of-2 errors: spin/occupation factors, k-point weight normalization (sum to 1 vs to Nk), Fermi occupations, reciprocal-lattice 2*pi conventions.
- FFT normalization and grid handling (forward/backward norm, Nyquist/half-grid packing, real-FFT symmetry, wrap-around).
- Operator Hermiticity, gauge/phase conventions, charge/norm conservation.
- AUTODIFF CORRECTNESS (critical for this codebase): does the graph actually differentiate the quantity claimed? Flag .detach()/.item()/.numpy() that silently cut gradients, in-place ops that corrupt autograd, non-differentiable branches on tensor values, and places claiming a derivative (force = -dE/dx, stress, response) where the graph is broken or the sign is wrong.
- Iterative-solver stability: density mixing, preconditioning, convergence criteria that can diverge or false-converge.
- PBC / minimum-image / supercell handling.
For each finding cite the specific physical law, convention, or torch semantics violated.`,
  'duplication': `DUPLICATION & DIVERGENCE. Hunt for copy-pasted numerics, parallel implementations of the same operator/transform/integral, repeated ad-hoc unit conversions or physical constants, and once-identical copies that have DRIFTED (a bug fixed in one but not the other is the highest-value find). Prefer structural clones over textual similarity. For each, name the single canonical home it should collapse into and what the shared contract would be.`,
  'complexity': `UNNECESSARY COMPLEXITY. Hunt for long multi-job functions, deep nesting, tangled control flow, leaky abstractions, parameters threaded through many layers, implicit/global mutable state, and unclear tensor-shape/dtype contracts (shapes only knowable by reading the body). For each, name the data structure, type, or contract that would dissolve it.`,
}
const RUBRIC = RUBRICS[OBJECTIVE] ?? `Review for: ${OBJECTIVE}. Report concrete, evidence-backed issues.`

// ---------- schemas ----------
const SCOPE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    files: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { path: { type: 'string' }, loc: { type: 'integer' } }, required: ['path', 'loc'] } },
    map: { type: 'string', description: 'Repo map: modules, key tensors/data structures, and the unit/dtype/convention inventory a reviewer needs.' },
  }, required: ['files', 'map'],
}
const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { findings: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: {
      file: { type: 'string' }, line: { type: 'integer' },
      kind: { type: 'string' }, title: { type: 'string' },
      severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
      scope: { type: 'string', enum: ['local', 'global-candidate'] },
      evidence: { type: 'string' }, confidence: { type: 'number' },
      suggested_direction: { type: 'string' },
    }, required: ['file', 'line', 'kind', 'title', 'severity', 'scope', 'evidence', 'confidence'],
  } } }, required: ['findings'],
}
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
    corrected_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'none'] },
    confidence: { type: 'number' },
  }, required: ['refuted', 'reason', 'corrected_severity'],
}
const THEMES_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { themes: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: { id: { type: 'string' }, title: { type: 'string' }, rationale: { type: 'string' },
      issue_refs: { type: 'array', items: { type: 'string' } },
      priority: { type: 'number' } },
    required: ['id', 'title', 'rationale', 'priority'],
  } } }, required: ['themes'],
}
const PROPOSAL_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    proposal: { type: 'string', description: 'The concrete object / data structure / contract / formalism proposed.' },
    rationale: { type: 'string' },
    sketch: { type: 'string', description: 'Signature / type / pseudocode sketch.' },
    affected_files: { type: 'array', items: { type: 'string' } },
    invasiveness: { type: 'string', enum: ['low', 'medium', 'high'] },
    risks: { type: 'string' },
  }, required: ['proposal', 'rationale', 'invasiveness'],
}
const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    correctness: { type: 'number' }, physics_soundness: { type: 'number' },
    low_invasiveness: { type: 'number' }, maintainability: { type: 'number' },
    total: { type: 'number' }, best_idea: { type: 'string' }, verdict: { type: 'string' },
  }, required: ['total', 'verdict'],
}

// ---------- helpers ----------
const SEV = { critical: 4, high: 3, medium: 2, low: 1, none: 0 }
const SEV_NAME = ['none', 'low', 'medium', 'high', 'critical']

function dedupe(findings) {
  const byKey = new Map()
  for (const f of findings) {
    const key = `${f.file}:${Math.round((f.line || 0) / 8)}:${f.kind}`
    const prev = byKey.get(key)
    if (!prev) { byKey.set(key, { ...f, support: 1 }); continue }
    prev.support += 1
    if (SEV[f.severity] > SEV[prev.severity]) prev.severity = f.severity
    prev.confidence = Math.max(prev.confidence, f.confidence)
    if (!prev.evidence.includes(f.evidence)) prev.evidence += ` | ${f.evidence}`
  }
  return [...byKey.values()]
}

function priorityScore(issue) {
  return SEV[issue.severity] * 10 + (issue.confidence || 0) * 3 + (issue.support || 1)
}

const LENSES = [
  'CORRECTNESS lens: is the stated defect actually present in the code as written? Read the exact lines. If the claim misreads the code, refute.',
  'PHYSICS lens: even if the code matches the description, is it actually wrong physically/numerically, or is this a valid convention? Check units, signs, dtypes, autograd semantics against real DFT/torch behavior. If it is defensible, refute.',
  'REPRODUCE lens: construct the concrete input/state that would trigger the claimed wrong behavior. If you cannot construct one (dead code, guarded, impossible state), refute.',
]

async function verifyIssue(issue, n) {
  const votes = (await parallel(
    Array.from({ length: n }, (_, k) => () =>
      agent(
        `You are an adversarial reviewer of gradwave (differentiable plane-wave DFT, PyTorch). ${LENSES[k % LENSES.length]}\n\n` +
        `Read the actual code at ${REPO}/${issue.file} (around line ${issue.line}) before judging. Default to refuted=true when genuinely uncertain — we want only high-trust findings to survive.\n\n` +
        `CLAIMED ISSUE [${issue.severity}/${issue.kind}]: ${issue.title}\nEvidence: ${issue.evidence}`,
        { ...(MODEL ? { model: MODEL } : {}), schema: VERDICT_SCHEMA, phase: 'Verify', label: `verify:${issue.file.split('/').pop()}:${k}` },
      ),
    ),
  )).filter(Boolean)
  if (!votes.length) return { ...issue, survived: false, refutes: 0, votes: [] }
  const refutes = votes.filter(v => v.refuted).length
  const survived = refutes < Math.ceil(votes.length / 2)
  // consensus severity from non-refuting votes' corrected_severity
  const keep = votes.filter(v => !v.refuted).map(v => SEV[v.corrected_severity] ?? SEV[issue.severity])
  const consensus = keep.length ? SEV_NAME[Math.round(keep.reduce((a, b) => a + b, 0) / keep.length)] : issue.severity
  return { ...issue, survived, refutes, votesN: votes.length, severity: consensus, verifierReasons: votes.map(v => v.reason) }
}

async function judgePanel(theme, issues) {
  const context = issues.filter(i => (theme.issue_refs || []).some(r => i.title.includes(r) || r.includes(i.file)))
    .slice(0, 12).map(i => `- [${i.severity}] ${i.file}:${i.line} ${i.title}`).join('\n') ||
    issues.slice(0, 8).map(i => `- [${i.severity}] ${i.file}:${i.line} ${i.title}`).join('\n')
  const angles = [
    'MINIMAL-DIFF angle: the smallest change that fixes the issues without new abstractions.',
    'FORMALISM-FIRST angle: introduce the right type / data structure / contract / invariant that makes the whole class of bug unrepresentable. Be willing to refactor.',
    'AUTODIFF-AND-PERF angle: a design that also keeps the computation differentiable end-to-end and efficient on GPU/float64-complex128.',
  ]
  const proposals = (await parallel(angles.map((a, k) => () =>
    agent(
      `Theme: ${theme.title}\n${theme.rationale}\n\nGrounding issues in gradwave (${REPO}):\n${context}\n\n` +
      `Propose a remediation from this angle:\n${a}\n\nRead relevant code before proposing. Give a concrete object/contract/formalism with a signature/type sketch.`,
      { ...(MODEL ? { model: MODEL } : {}), schema: PROPOSAL_SCHEMA, phase: 'Remediate', label: `propose:${theme.id}:${k}` },
    ),
  ))).filter(Boolean)
  if (!proposals.length) return { theme, proposals: [], recommendation: 'No proposals produced.' }
  const scored = (await parallel(proposals.map((p, k) => () =>
    agent(
      `Score this remediation proposal for gradwave theme "${theme.title}" on correctness, physics_soundness, low_invasiveness, maintainability (0-10 each; total 0-40). Give best_idea (the single most valuable element to keep) and a one-line verdict.\n\n` +
      `PROPOSAL: ${p.proposal}\nSketch: ${p.sketch || '(none)'}\nInvasiveness: ${p.invasiveness}\nRisks: ${p.risks || '(none)'}`,
      { ...(MODEL ? { model: MODEL } : {}), schema: JUDGE_SCHEMA, phase: 'Remediate', label: `judge:${theme.id}:${k}` },
    ).then(j => ({ p, j })),
  ))).filter(Boolean)
  scored.sort((a, b) => (b.j.total || 0) - (a.j.total || 0))
  const winner = scored[0]
  const recommendation = await agent(
    `Synthesize the final recommended remediation for gradwave theme "${theme.title}". Start from the winning proposal, graft the best_idea from the runners-up, and state it as a concrete, actionable design (the object/contract/formalism, its signature, where it lives, and the migration path).\n\n` +
    `WINNER (score ${winner.j.total}): ${winner.p.proposal}\nSketch: ${winner.p.sketch || ''}\n\n` +
    `RUNNER-UP IDEAS: ${scored.slice(1).map(s => s.j.best_idea).filter(Boolean).join(' | ')}`,
    { ...(MODEL ? { model: MODEL } : {}), phase: 'Remediate', label: `synth:${theme.id}` },
  )
  return { theme, proposals, scores: scored.map(s => ({ proposal: s.p.proposal, total: s.j.total, verdict: s.j.verdict })), recommendation }
}

function chunk(arr, size) {
  const out = []
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size))
  return out
}

// ================= run =================
phase('Scope')
log(`Objective: ${OBJECTIVE} | target: ${TARGET} | repo: ${REPO}`)
const scope = await agent(
  `Enumerate the Python source files to review in gradwave and build a reviewer's map.\n\n` +
  `Run: git -C ${REPO} ls-files -- '${TARGET}/*.py' '${TARGET}/**/*.py' (fall back to find if the target is a single dir). Exclude tests/, benchmarks/, experiments/, examples/, scripts/, docs/. For each source file return an ABSOLUTE path (prefix ${REPO}/) and its line count (wc -l).\n\n` +
  `Then read the key interfaces (constants.py, dtypes.py, core/ headers, the target's __init__ and main modules) and write a concise MAP: the modules and their responsibilities, the central tensors/data structures and their shape+dtype conventions, and the unit/sign/normalization conventions in force (eV/Angstrom base units, complex128, k-point weights, 2*pi lattice, FFT norm). This map is handed to every reviewer so they are not shortsighted.`,
  { ...(MODEL ? { model: MODEL } : {}), schema: SCOPE_SCHEMA, label: 'scope' },
)
// deterministic clamp: the scope agent can over-enumerate; keep only files under the target path
const scopePrefix = TARGET === '.' ? `${REPO}/` : `${REPO}/${TARGET.replace(/[*].*$/, '').replace(/\/+$/, '')}`
let files = (scope.files || []).filter(f => f.path.endsWith('.py') && f.path.startsWith(scopePrefix))
if (!files.length) { log(`No .py files under ${scopePrefix} (scope returned ${(scope.files || []).length}); aborting.`); return { error: 'no files in target', target: TARGET, scopePrefix } }

// size finders to cap
let perFinder = FILES_PER_FINDER
let nFinders = Math.ceil(files.length / perFinder)
if (nFinders > MAX_FINDERS) { nFinders = MAX_FINDERS; perFinder = Math.ceil(files.length / nFinders); log(`Coarsening: ${files.length} files across ${MAX_FINDERS} finders (~${perFinder} files each) to respect maxFinders.`) }
const slices = chunk(files, perFinder)
log(`${files.length} files -> ${slices.length} finder slices; verifying top ${VERIFY_TOP_PER_SLICE}/slice with ${SKEPTICS} skeptics each.`)

// ---- Find -> Verify pipeline ----
const finderReports = []
const perSlice = await pipeline(
  slices,
  async (slice, _o, i) => {
    const list = slice.map(f => `${f.path} (${f.loc} loc)`).join('\n')
    const r = await agent(
      `You are reviewing gradwave (differentiable plane-wave DFT, PyTorch) for ONE objective only.\n\nOBJECTIVE RUBRIC:\n${RUBRIC}\n\nREPO MAP (global context so you are not shortsighted):\n${scope.map}\n\n` +
      `Review ONLY these files (read them fully):\n${list}\n\n` +
      `Report concrete, evidence-backed findings with exact file+line. Mark scope='global-candidate' if the issue likely recurs across the codebase or is really about a cross-file contract; else 'local'. Do not invent issues to fill a quota; precision over recall.`,
      { ...(MODEL ? { model: MODEL } : {}), schema: FINDINGS_SCHEMA, phase: 'Find', label: `find:${i}` },
    )
    const findings = r?.findings ?? []   // agent() returns null on terminal API error
    finderReports.push({ i, files: slice.map(f => f.path), findings })
    return findings
  },
  async (found) => {
    const issues = dedupe(found).sort((a, b) => priorityScore(b) - priorityScore(a))
    const toVerify = issues.slice(0, VERIFY_TOP_PER_SLICE)
    const passthrough = issues.slice(VERIFY_TOP_PER_SLICE).map(i => ({ ...i, survived: true, unverified: true }))
    const verified = await parallel(toVerify.map(issue => () => verifyIssue(issue, SKEPTICS)))
    return [...verified.filter(Boolean), ...passthrough]
  },
)
const allLocal = perSlice.flat().filter(Boolean)
const localSurvivors = allLocal.filter(i => i.survived)
log(`Local: ${allLocal.length} candidate issues -> ${localSurvivors.length} survived (${allLocal.filter(i => i.unverified).length} passed through unverified).`)

// ---- Global synthesis (barrier: needs all reports) ----
phase('Global')
const reportDigest = finderReports.map(r =>
  `### slice ${r.i} (${r.files.length} files)\n` + r.findings.map(f => `- [${f.severity}/${f.scope}] ${f.file}:${f.line} ${f.title}`).join('\n'),
).join('\n\n')
const globalCandidates = localSurvivors.filter(i => i.scope === 'global-candidate')
  .map(i => `- ${i.file}:${i.line} ${i.title}`).join('\n')
const globalRaw = (await parallel([0, 1].map(k => () =>
  agent(
    `You see EVERY local reviewer's report across gradwave. Your job is to find issues NO single-slice reviewer could see: the same convention implemented inconsistently across modules, drifted duplicate implementations, contract mismatches between producer and consumer, an abstraction that should be shared but is reinvented, or a systemic pattern the ${OBJECTIVE} rubric cares about.\n\n` +
    `REPO MAP:\n${scope.map}\n\nALL LOCAL FINDINGS:\n${reportDigest}\n\nGLOBAL-CANDIDATE flags from reviewers:\n${globalCandidates || '(none)'}\n\n` +
    `${k === 0 ? 'Focus on cross-file DUPLICATION and DRIFT.' : 'Focus on inconsistent CONVENTIONS/CONTRACTS across modules.'} Read code across files to confirm before reporting. Set scope='global-candidate' and cite every file involved in evidence.`,
    { ...(MODEL ? { model: MODEL } : {}), schema: FINDINGS_SCHEMA, phase: 'Global', label: `global:${k}` },
  ),
))).filter(Boolean)
let globalIssues = dedupe(globalRaw.flatMap(r => r.findings || [])).map(i => ({ ...i, scope: 'global' }))
globalIssues = (await parallel(globalIssues.map(g => () => verifyIssue(g, Math.min(2, SKEPTICS))))).filter(i => i && i.survived)
log(`Global: ${globalIssues.length} cross-file issues survived.`)

// ---- Remediation (judge panel per theme) ----
phase('Remediate')
const allIssues = [...localSurvivors, ...globalIssues].sort((a, b) => priorityScore(b) - priorityScore(a))
let remediations = []
if (allIssues.length) {
  const themed = await agent(
    `Cluster these surviving gradwave issues into remediation THEMES (a theme = a set of issues one design change would address). Rank by priority (severity x breadth). Give each a stable id.\n\n` +
    allIssues.slice(0, 60).map(i => `- [${i.severity}/${i.scope}] ${i.file}:${i.line} ${i.title}`).join('\n'),
    { ...(MODEL ? { model: MODEL } : {}), schema: THEMES_SCHEMA, label: 'themes' },
  )
  const top = (themed?.themes ?? []).sort((a, b) => (b.priority || 0) - (a.priority || 0)).slice(0, TOP_THEMES)
  log(`${(themed?.themes ?? []).length} themes -> judge-panel design on top ${top.length}.`)
  remediations = (await parallel(top.map(t => () => judgePanel(t, allIssues)))).filter(Boolean)
}

return {
  objective: OBJECTIVE, target: TARGET, repo: REPO,
  stats: {
    files: files.length, finders: slices.length,
    localCandidates: allLocal.length, localSurvived: localSurvivors.length,
    unverifiedPassthrough: allLocal.filter(i => i.unverified).length,
    globalIssues: globalIssues.length, themes: remediations.length,
  },
  localIssues: localSurvivors.map(i => ({ file: i.file, line: i.line, kind: i.kind, severity: i.severity, title: i.title, evidence: i.evidence, suggested_direction: i.suggested_direction, unverified: !!i.unverified, refutes: i.refutes })),
  globalIssues: globalIssues.map(i => ({ file: i.file, line: i.line, kind: i.kind, severity: i.severity, title: i.title, evidence: i.evidence })),
  remediations: remediations.map(r => ({ theme: r.theme.title, rationale: r.theme.rationale, recommendation: r.recommendation, scores: r.scores })),
}
