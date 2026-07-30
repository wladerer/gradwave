# Improving electronic convergence

What to reach for when an SCF is slow, oscillating, or converges to the wrong
physical branch. [Wisdom](wisdom.md#scf-and-mixing) has the underlying
lessons in full ("SCF and mixing", "Metals and smearing", "Spin"); this page
turns them into concrete recipes for the systems that are canonically hard to
converge, plus an investigated answer to whether bigger cells are
systematically harder. Every option named below is a real keyword on `scf`
(norm-conserving) or `scf_uspp` (ultrasoft/PAW) — see
`src/gradwave/scf/options.py` for the full current list (`SCFOptions` /
`MixerOptions`).

## Quick diagnostic table

| Symptom | Likely cause | Try |
|---|---|---|
| Metal SCF oscillates or sloshes | No Kerker damping on the charge-total mode | Kerker is on by default once `smearing != "none"`; if you turned it off, turn it back on |
| Moment collapses to zero silently | Default damping on the FM/Stoner mode | `mixing_scheme="johnson"` (near a Stoner instability) or a stronger seed + warm start |
| Energy looks converged but `converged=False` for many iterations | Density residual floors at occupation noise on a smeared metal | `scf_uspp(..., criterion="energy")`, or loosen `rhotol` and gate on `etol` |
| Semicore/PAW metal charge oscillates with a fixed gain per iteration | The on-site becsum/ddd mode is stiff and undamped | Already handled by `scf_uspp`'s composite (ρ, becsum) mixing; re-check `mixing_scheme` before hand-tuning further |
| Slab/molecule/defect cell converges worse than the bulk material | A single Kerker screening length over- or under-screens the vacuum/impurity region | `precond="local_tf"` |
| Two-component or semicore metal needs more iterations than a simple metal | Kerker's single pole cannot represent a multi-scale response | `scf.learned_precond` (prototype, see below) |
| A scan/EOS/relaxation flips between magnetic branches point to point | The mixer has no way to prefer the physical fixed point | Warm-start with `start_from=checkpoint.as_start_from(prev)` |

## Metals: smearing, Kerker, and mixing scheme

For a metal, the mixing scheme sets the iteration count — the smearing kernel
does not. On 1-atom fcc Pt (PAW, 40/400 Ry, 6×6×6, 0.2 eV), `johnson`
converges in 13 iterations against `pulay`'s 17 and `broyden`'s 20, while
gaussian, cold, and Methfessel-Paxton smearing land within one iteration of
each other at a fixed scheme (wisdom.md, "Metals and smearing"). Pick the
mixer for convergence and the smearing scheme for the physics you need
(entropy convention, occupation smoothness) — a smearing-width sweep will not
fix a stalled SCF.

Kerker itself is on by default whenever `smearing != "none"`, so you never
need to pass `kerker=True` for a plain metal. One thing worth knowing: `scf`
(norm-conserving) also turns Kerker on for an *insulator*, once the cell is
large enough that the smallest nonzero |G| drops below about 0.8 Å⁻¹
(`gradwave.scf.loop._resolve_kerker`, roughly an 8 Å cell edge). The
reasoning is that long-wavelength charge sloshing scales like
`4πe²χ/G²_min`, and once G_min is small enough it dominates mixing even
without a Fermi surface. `scf_uspp` does **not** carry this size-aware
policy — its Kerker default depends on `smearing != "none"` only. So a large
*insulating* USPP/PAW supercell that mixes slowly is a case where the
norm-conserving path would have switched Kerker on for you automatically and
`scf_uspp` will not: try `mixing_kerker=True` (or `precond="local_tf"`) by
hand. See
["Does convergence get harder for bigger cells?"](#does-convergence-get-harder-for-bigger-cells-investigated)
below for the fuller picture.

The default mixing scheme is resolved per `nspin`, and it's worth knowing
that **the norm-conserving and PAW/USPP drivers resolve it in opposite
directions**:

- `scf` (norm-conserving): `mixing_scheme=None` → `johnson` for `nspin=2`,
  `pulay` for `nspin=1`.
- `scf_uspp` (ultrasoft/PAW): `mixing_scheme=None` → `johnson` for `nspin=1`,
  `pulay` for `nspin=2`.

The PAW default is inverted because the composite (density, becsum) mixing
vector carries an on-site augmentation-charge mode that stays stiff even in a
*gapped, non-magnetic* PAW insulator — Kerker-preconditioned Johnson wins
there regardless of spin (Si 18→12 iterations, Cu 19→13, Pt 21→12, 8-atom Si
20→13, converged free energy bit-identical; see wisdom.md). For `nspin=2`
PAW, Johnson gives up the becsum step-damping that `pulay` relies on, which
helps only near a Stoner instability (below) and hurts on a robust
ferromagnet — so `pulay` stays the safer default there. An explicit
`mixing_scheme=` always overrides the automatic choice on either driver.

## Ferromagnetic / near-Stoner-instability metals (Ni, Fe)

This is the hardest case in the code: the magnetization mode's response can
amplify each mixing step instead of damping it (measured gain near −6 on fcc
Ni's spin mode). The symptom is either a silent collapse of the moment to the
nonmagnetic branch under default damping, or a persistent oscillation that
never collapses but never settles either. Recipe, roughly in the order to
try things:

```python
from gradwave.scf.uspp import scf_uspp
from gradwave import checkpoint

res = scf_uspp(
    system, xc, nspin=2, start_mag=[0.6],
    smearing="gaussian", width=0.1,
    mixing_scheme="johnson",     # near the Stoner boundary; NOT the nspin=2 default
    spin_precond=True,           # Stoner chi0-diagonal preconditioner on the m-channel
    criterion="energy",          # gate on the settled free-energy tail, rhotol as a safety net
)
res.mag_total   # check this is near the expected moment, not ~0
```

- **`mixing_scheme="johnson"`** is the biggest lever near the Stoner boundary
  specifically (fcc Ni PAW: 27 → 18 iterations) — don't apply it to every FM
  metal by default. On a robust ferromagnet like bcc Fe, `johnson` gives up
  the becsum step-damping that `pulay` relies on and *blows up* (29 → 93
  iterations, same converged moment and energy). Reach for `johnson` when the
  moment is marginal or collapsing; keep `pulay` (the default) when the
  ferromagnet is solid and just slow to converge.
- **`spin_precond=True`** (`scf_uspp` only, smeared `nspin=2`) builds a
  Stoner preconditioner from the current Fermi-surface bands — a
  `(I − χ₀^diag K_mm)⁻¹` operator on the magnetization channel, inverted
  cheaply and exactly via the Woodbury identity (`scf/spin_precond.py`). This
  is the "prefer preconditioning to step-size control" idea from wisdom.md:
  the expansive mode needs an operator, not a damping schedule. Outside the
  Stoner regime the operator is close to the identity and does no harm, so
  it's safe to leave on for any smeared magnetic metal, not just the
  marginal ones.
- **`adapt_step=True`** (`scf_uspp`) is a cheaper, opt-in stabilizer: it
  watches for a block's residual growing and cuts that block's step. It's
  enough to *stop the silent moment collapse* on FM Ni at the default
  `mixing_alpha`, but it isn't a substitute for the options above at tight
  convergence — its damping only ever tightens, so it over-reacts to the
  wild first few iterations and can leave the run permanently over-damped
  (measured: static `alpha=0.3` reaches |dρ| 2e-3 where `adapt_step` stalls
  at 2e-2 with the ρ-block floored). Use it for exploratory runs at an
  unknown damping; for production, hand-setting `mixing_alpha` is still the
  right call (0.3 for Ni is the validated value).
- **Warm-start chains** are the practical defense against branch selection
  itself, which no mixer can fix: the nonmagnetic solution is a genuine
  stationary point tens of meV away, and any starting guess can land there.
  Across an EOS or displacement scan, seed each point from the previous
  converged result:

  ```python
  ckpt = checkpoint.load_checkpoint(...)
  res = scf_uspp(system, xc, nspin=2, start_from=checkpoint.as_start_from(ckpt), ...)
  ```

  Pair this with an explicit moment-gate check (`assert res.mag_total >
  threshold`, or compare against `reference_moment_magnitudes` for
  non-collinear cases): the mixer will not tell you it picked the wrong
  branch, only that it converged.
- Gate FM metals on the **energy tail**: `criterion="energy"` in `scf_uspp`
  converges on a settled 3-iteration free-energy tail rather than the density
  residual — the honest criterion for a smeared metal, since its density
  residual floors at occupation noise while the free energy has long since
  settled.

## Semicore / PAW stiff on-site (becsum/ddd) modes

`scf_uspp` already mixes density and becsum together as one composite
vector — this needs no flag, because mixing becsum outside that shared
history produces a charge oscillation that grows every iteration on
semicore-metal PAW. The one tunable here is `bec_step_scale` (`MixerOptions`,
or `opts=SCFOptions(...)`), which defaults to a mixer-aware value: `1.0` for
`johnson`, `0.4` otherwise. The `0.4` value damps the becsum step for
`pulay`/`broyden`; Johnson's normalized multisecant update doesn't need it,
and leaving it at `0.4` under Johnson costs about 11 iterations on FM Ni.
**If you change `mixing_scheme` by hand, re-check `bec_step_scale`** rather
than carrying over a value tuned for the old scheme — the defaults already
do this for you, so only override it if you've re-measured.

## Insulators vs metals — what's actually different

A small, gapped, non-magnetic cell doesn't need any of the metal machinery
above: Kerker is off by default (it actually *slows* convergence on an
insulator, with no long-wavelength charge instability to suppress), and the
plain default mixer (`pulay` on the NC path) is usually enough. Two things
still carry over from the metal story:

- **Cell size can flip Kerker on even for an insulator** on the NC path (see
  above) — the long-wavelength sloshing that Kerker fixes is not exclusively
  a Fermi-surface effect, it is a small-|G| effect, and a large enough
  insulating cell reaches the same regime.
- **PAW insulators inherit the on-site augmentation stiffness** regardless of
  the gap, which is why `scf_uspp`'s default is `johnson` even for `nspin=1`
  insulators (see the mixing-scheme table above) — a norm-conserving
  insulator has no such mode and stays on `pulay`.

## Mixer scheme cheat sheet

| Scheme | What it is | Reach for it when |
|---|---|---|
| `pulay` (DIIS) | Bordered least-squares extrapolation over a residual history, Kerker-weighted metric | The general default: NC insulators/simple metals, PAW `nspin=2` (safer near a robust ferromagnet) |
| `broyden` | Limited-memory Broyden's second method (QE `mixing_mode='plain'`), sequential secant updates | Rarely the first choice here — unweighted, it can be poisoned by wild early residuals near a Stoner instability, exactly the failure mode Johnson fixes |
| `johnson` | Normalized, Tikhonov-regularized multisecant Broyden (QE's actual default scheme) | NC `nspin=2` magnetic metals, PAW `nspin=1` (any spin), and explicitly near a Stoner boundary on PAW `nspin=2` (fcc Ni) |

All three converge to bit-identical energies when everything else is held
fixed — this is purely an iteration-count/robustness choice, never an
accuracy one.

## Inhomogeneous cells: local-TF and learned multipole preconditioners

Bare Kerker uses one screening length `q0` for the whole cell. That's correct
for a roughly-uniform bulk metal, and wrong wherever the density isn't
uniform — a slab, a molecule in vacuum, or an alloy or defect, where no
single bulk `q0` is right everywhere. Two opt-in preconditioners go past the
single pole; both are exact drop-in replacements for Kerker — they reproduce
it in the trivial limit and never change the converged energy, only the path
to it.

**`precond="local_tf"`** (`scf.local_tf`, works with both `scf` and
`scf_uspp`) makes the screening wavevector track the local density,
`q²(r) = min(q²_TF(r), q0_max²)`, solved with a short warm-started CG each
mixing step. It's neutral on a homogeneous bulk (fcc Al 8×8×8: 9 iterations
either way), and a real win as the cell gets more inhomogeneous:

| Cell | `kerker` | `local_tf` |
|---|---|---|
| bulk fcc Al (NC) | 9 | 9 |
| Al(100) slab, 4 layers | 21 | 17 (1.24×) |
| Al(100) slab, 6 layers | 27 | 21 (1.29×) |
| bulk fcc Al PAW | 10 | 10 |
| Al(100) PAW slab, 4 layers | 23 | 19 (1.21×) |

(`benchmarks/bench_precond.py`.) The margin widens as the slab gets
thicker — that's the cell becoming more inhomogeneous, not just bigger. See
the next section for why that distinction matters.

**`scf.learned_precond.MultipoleKerkerPrecond`** replaces the single Kerker
pole with a learned sum, `f_θ(G²) = Σ_i w_i·G²/(G²+q_i²)`, fit
(`fit_multipole`) against a short probe SCF's own residual response
(`response_from_residuals`) — a capability that only exists because the
solver is differentiable. This is a prototype, not yet a turnkey "converges
faster" flag: it ties Kerker on single-scale systems (fcc Al, and fcc
Pt+SOC at 9 vs 9 iterations) because there's nothing multi-scale to exploit,
and it gives the first real win on a genuinely multi-scale response — fcc
Cu's 3s3p semicore/d-band system, 10 → 8 iterations. It's a measured
*negative* on bcc Fe (`nspin=2`): the bottleneck there is branch selection
(moment collapse), which no preconditioner shape can fix, so the headroom
over `johnson` is thin. The in-flight research notes live in
`docs/ideas.md` ("Learned multi-pole density-mixing preconditioner") — read
that before extending this feature rather than re-deriving it; this page
only covers the user-facing recipe.

## Does convergence get harder for bigger cells? (investigated)

The textbook worry: as a periodic cell grows, the smallest nonzero
reciprocal-lattice vector `G_min = 2π/L` shrinks, and Kerker's filter
`G²/(G²+q0²)` damps those modes almost to zero — so does a bigger cell need
systematically more preconditioning care, or does it converge in more
iterations regardless? Three separate things were checked, rather than
assumed:

### What the Kerker formula predicts

`scf/mixing.py` implements exactly `R̃(G) = R(G)·G²/(G²+q0²)`, with `q0`
(default 1.1 Å⁻¹) a single *intensive* material property — a bulk screening
length, independent of cell size. The physical effect Kerker exists to
cure — the metallic Lindhard/Thomas-Fermi response diverging as `G→0` — does
genuinely reach more modes as a cell grows and `G_min` shrinks, since more
G-shells fall inside the poorly-conditioned small-G region. But because `q0`
is intensive, the same filter shape keeps taming each new shell correctly:
growing the cell doesn't change what the right `q0` is for a homogeneous bulk
material.

The codebase has already reasoned about this. `scf.loop._resolve_kerker`
turns Kerker on even for an *insulator* once the cell edge reaches roughly
8 Å (`G_min` below 0.8 Å⁻¹), for exactly the `4πe²χ/G²_min` reason above. That
policy exists only on the norm-conserving `scf` driver — `scf_uspp`'s Kerker
default is metal-only (`smearing != "none"`), with no cell-size term. So a
large **insulating USPP/PAW supercell is the one case the codebase doesn't
automatically protect** — worth trying `mixing_kerker=True` or
`precond="local_tf"` by hand if one converges slowly.

### What the test suite and benchmarks contain today

`tests/integration/test_supercell_identity.py` builds a rattled
(no-symmetry) Si supercell to cross-check against its folded primitive
cell — but its own docstring is explicit that this is "an identity at solver
tolerance, not a convergence statement." Both runs are simply asserted
`converged` at a tight `rhotol=1e-9`/`etol=1e-10`, and iteration counts are
never compared or reported.

`benchmarks/bench_matrix.py` is closer: its docstring literally says
"materials diversity + Si supercell size scaling," and it defines `si2`
(2-atom primitive, 4×4×4 k-mesh), `si8` (8-atom, 2×2×2 k-mesh, same physical
density), and `si64` (64-atom, Γ-only) cases, printing `res.n_iter` for each.
**This is exactly the harness needed to answer the question — but no run of
it is recorded anywhere in the repository**: no `benchmarks/results/` entry,
no numbers quoted in `docs/ideas.md`, `wisdom.md`, or `performance.md`. The
honest answer is that gradwave has never measured its own SCF iteration-count
scaling with supercell size and written the result down.

### Why no live run tonight

The task allowed a small, cheap empirical check if the machine was free.
`uptime` showed a load average of 101 (settling to ~59 a few minutes later)
on this 22-core box — clearly other heavy jobs were running, and CLAUDE.md
is explicit about checking load before adding CPU-heavy work. So no
`bench_matrix.py` run was launched; this is a disclosed, deliberate
omission, not an oversight.

**Bottom line:** the Kerker math predicts no inherent iteration-count
blowup for a *homogeneous* bulk cell as it grows, as long as the screening
length is applied at all — and gradwave's own size-aware Kerker policy for
NC insulators is evidence the codebase has already reasoned about this. The
one scaling effect actually measured in-tree is *inhomogeneity*, not raw
size (the local-TF slab table above), and the one real gap is that PAW/USPP
insulators don't get the automatic size-based protection the NC driver gives
them. No empirical supercell iteration-count sweep is committed anywhere in
gradwave today; `benchmarks/bench_matrix.py` exists to produce one and
should be run — on a quiet box, or via `./scripts/gwq bench` — before this
section can say more than what the math predicts.
