# Symbolic algebra + algorithm design for plane-wave DFT

Exploration notes. Goal: use symbolic computation (SymPy / SageMath) and a bit of
group theory to reduce **time or memory** in gradwave's SCF path. Nothing here is
committed to yet — this is a map of where the math actually pays off, with the
derivations spelled out so we can decide before writing a line of kernel code.

## 0. The governing idea: symbolic algebra is a *build-time* tool

You never call Sage inside `scf.loop.scf`. Symbolic computation earns its keep at
**design time**: derive a closed form / exact table / simplified expression once,
offline, then emit a fast numerical kernel (PyTorch / NumPy) that ships in the
package. The pipeline is always

    Sage/SymPy  →  simplify + common-subexpression-elimination (CSE)  →  generated .py kernel  →  gradwave

This is exactly how the reference libraries are built: **libxc** is Maple → C,
**libcint / PySCF** integral code is SymPy → C. gradwave's `xc/r2scan.py` even
says in its docstring that it was "transcribed from libxc's own Maple source" —
we would be closing that loop and generating the transcription instead of doing
it by hand.

### What it will NOT do
Be honest about the dominant cost. In plane-wave DFT the memory and wall-clock
are dominated by the **eigensolver working set**: the occupied-plus-buffer
wavefunction blocks (`N_band × N_pw × N_k` complex) and the Davidson subspace,
with `H` applied matrix-free (never stored). Symbolic algebra does **not** shrink
that block. The only lever that does is **symmetry reduction** (Track 3). Tracks 1,
2, and 4 attack the *setup*, *XC*, and *nonlocal/augmentation* layers — real, but
secondary to the eigensolver. Keep that ordering honest when we pick.

Ranked by (payoff / effort) for a first pass: **1 → 4 → 2 → 3**.

---

## Track 1 — Real Gaunt coefficients as exact sparse tables

**Target module:** `core/gaunt.py` (feeds USPP/PAW augmentation `Q_ij(r⃗)` and the
nonlocal projector angular coupling). **Lever:** memory + flops in the augmentation
contraction. **Tool:** SymPy is sufficient. **Risk:** low; self-contained.

### Current state
`gaunt.py` computes the *real* Gaunt coefficients

    G^{LM}_{l₁m₁,l₂m₂} = ∫ Ȳ_{LM} Ȳ_{l₁m₁} Ȳ_{l₂m₂} dΩ

(bars = real spherical harmonics) by **Gauss–Legendre × uniform-φ quadrature**.
That is numerically exact for the band-limited integrand, but it produces a *dense
float table* and throws away the exact algebraic structure — in particular it does
not know, a priori, which coefficients are **identically zero**.

### The symbolic content
Work through the complex Gaunt coefficient, which has a closed form, then rotate to
the real basis.

**Complex Gaunt (Wigner 3j form):**

    ∫ Y_{L}^{M} Y_{l₁}^{m₁} Y_{l₂}^{m₂} dΩ
      = √[ (2l₁+1)(2l₂+1)(2L+1) / 4π ]
        · ( l₁ l₂ L ; 0  0  0 )
        · ( l₁ l₂ L ; m₁ m₂ −M )

The two `( · )` are Wigner 3j symbols, which SymPy gives **exactly** as
`rational × √(rational)` (`sympy.physics.wigner.gaunt`, `wigner_3j`).

**Selection rules — this is the whole point.** The coefficient is exactly zero
unless *all* of:

  1. `M = m₁ + m₂`                              (azimuthal / m-coupling)
  2. `|l₁ − l₂| ≤ L ≤ l₁ + l₂`                  (triangle inequality)
  3. `l₁ + l₂ + L` is **even**                  (parity — from the `(·;0 0 0)` 3j)

For the real harmonics the m-rule 1 becomes a small fixed set of `cos/sin`
product-to-sum couplings (`M ∈ {|m₁±m₂|}` with sign/normalization bookkeeping),
but rules 2 and 3 carry over unchanged. The consequence: for d-channel projectors
(`l₁,l₂ ≤ 2`, so `L` up to 4) **a large fraction of the (L,M,l₁m₁,l₂m₂) table is
structurally zero**. Right now those zeros are stored as `~1e-16` floats and
multiplied through the augmentation anyway.

### The win
- **Memory:** store the table as `(index_list, value_list)` sparse triples instead
  of a dense `(L+1)² × (l₁+1)² × (l₂+1)²` array.
- **Flops:** the augmentation `Q_ij(r⃗) = Σ_{LM} c^{LM}_{ij} Ȳ_{LM}(r̂) Q^L_{ij}(r)`
  contraction iterates only over the nonzero `(L,M)` per `(i,j)` pair.
- **Exactness / verification:** the symbolic values are a ground-truth oracle for
  the existing quadrature (→ a gradcheck test), and the constants baked into the
  generated kernel are correctly-rounded float64 of exact `rational×√`, not
  quadrature output.

### Sketch of the offline generator
```
from sympy.physics.wigner import gaunt          # complex Gaunt, exact
# 1. build U: real-Ylm  ←  complex-Ylm  (unitary, matches core/ylm.py's convention:
#    √2 cos/sin pairs, NO Condon–Shortley — verify against ylm_np's normalization)
# 2. real_gaunt = contract three U's against the complex gaunt table (exact rationals)
# 3. drop exact zeros → sparse (L,M,l1,m1,l2,m2, value)
# 4. emit a .py module: index arrays + float64 values; assert vs gaunt.py quadrature
```
The one subtlety to nail is the **convention match**: `core/ylm.py` /
`gaunt.ylm_np` use `Y_{l,+m} = √2 N_lm P_l^m cos(mφ)` with Condon–Shortley removed.
The `U` matrix must reproduce exactly that, or the signs won't line up. The
existing `l ≤ 3` unit test in `gaunt.py` is the anchor to validate against.

**Exploratory math to actually do first:** by hand, tabulate the nonzero real
Gaunt coefficients for `l₁ = l₂ = 2` (d–d, `L ≤ 4`) and count zeros vs nonzeros —
that number *is* the memory/flop savings, and confirms the effort is worth it
before building the generator.

---

## Track 2 — Symbolic XC kernels: CSE'd, tape-free, with analytic f_xc

**Target module:** `core/xc/` (esp. `r2scan.py`, `pbe.py`, `_pbe_kernels.py`).
**Lever:** SCF-iteration XC evaluation speed + memory, *and* unlocking cheap second
derivatives for phonons/response. **Tool:** SymPy. **Risk:** medium (must match
libxc pointwise).

### Current state
The XC functionals are written as differentiable PyTorch expressions and **every
derivative is obtained by autograd**: `xc/base.py` — "v_xc = δE_xc/δρ is obtained
by autograd"; `r2scan.py` — "v_xc, the meta-GGA v_τ = ∂e/∂τ, forces … all fall out
of autograd." Correct and elegant. But:

- Autograd retains a **tape** (the full expression graph) per grid batch to compute
  `v_xc`. r2SCAN's expression is deep (regularized α, switching functions,
  gradient-corrected exchange enhancement) → a large graph held in memory.
- The **second derivative f_xc = δ²E_xc/δρ²** (and the mixed `∂²e/∂ρ∂σ`,
  `∂²e/∂σ²`, τ-terms) needed for **DFPT / phonons / linear response** requires
  **double-backward** through that graph — memory- and time-heavy. `postscf.phonons`
  is exactly the consumer, and parts of that machinery are still library-only.

### The math a GGA/mGGA kernel must produce
For a meta-GGA the energy density is `e_xc(ρ, σ, τ)` with `σ = |∇ρ|²`. The
potential is the functional derivative, which for the gradient term is a divergence:

    v_xc = ∂e/∂ρ  −  ∇·( 2 (∂e/∂σ) ∇ρ )          (the ∇· is applied numerically on the grid)
    v_τ  = ∂e/∂τ

So the *pointwise* kernel must emit `e`, `∂e/∂ρ`, `∂e/∂σ`, `∂e/∂τ`; the divergence
is done by gradwave on the FFT grid (unchanged). For response you additionally want
the pointwise second derivatives

    ∂²e/∂ρ², ∂²e/∂ρ∂σ, ∂²e/∂ρ∂τ, ∂²e/∂σ², ∂²e/∂σ∂τ, ∂²e/∂τ²

which are the `f_xc` kernel — analytic, no double-backward.

### The pipeline
```
import sympy as sp
rho, sig, tau = sp.symbols('rho sigma tau', positive=True)
e = r2scan_energy_density(rho, sig, tau)      # transcribe the closed form ONCE
firsts  = [sp.diff(e, v)            for v in (rho, sig, tau)]
seconds = [sp.diff(e, a, b)         for a,b in second_pairs]
repl, reduced = sp.cse([e, *firsts, *seconds])   # common-subexpression elimination
# emit a flat PyTorch function: no autograd tape, one fused pass over the grid
```
`cse` is what turns "one huge expression differentiated six ways" into a compact
straight-line kernel that reuses the shared intermediates (the `α`, the switching
functions, the enhancement factor appear in every derivative).

### The win
- **Memory:** no retained autograd graph for `v_xc` — the generated kernel is a
  pure forward function returning energy + all needed derivatives.
- **Speed:** CSE'd single pass beats forward-expression-then-backward-pass; the
  shared subexpressions are computed once.
- **New capability:** analytic `f_xc` for DFPT/phonons without double-backward —
  this is the part that could move `postscf.phonons` / linear-response forward,
  not just speed up what exists.

### Landmines (write these into the test plan up front)
- **Match libxc pointwise.** The whole value of r2SCAN is bit-level agreement with
  libxc/QE. Any regeneration must reproduce the existing pyscf/libxc validation
  tests to tolerance. Start with **LDA-PW92** (`lda_pw92.py`, 56 lines) as the
  trivial proof-of-pipeline, then **PBE**, then r2SCAN.
- **Density floor / branches.** `base.py` uses `RHO_FLOOR_AU`; the symbolic
  expression must carry the same regularization (SymPy `Piecewise`) or the
  generated kernel will NaN in vacuum.
- **Spin.** Exchange spin-scales `E_x[ρ↑,ρ↓] = ½ Σ E_x[2ρ_s]`; correlation is a
  function of `(ρ, σ_tot, τ_tot, ζ)`. Generate the collinear-spin kernel, not just
  the closed-shell one, to be useful.
- **Units.** Kernels work in a.u. internally, eV/Å³ out — keep the `to_au` /
  `HARTREE_EV`/`BOHR_ANG` boundary identical to the current code.

**Exploratory math to do first:** do the full LDA-PW92 `e → v_xc → f_xc` by hand in
SymPy and diff the generated kernel against the current autograd path on a random
density grid. If that round-trips to 1e-12, the pipeline is proven and r2SCAN is
"just" a bigger transcription.

---

## Track 3 — Symmetry-adapted plane waves: block-diagonalize H

**Target:** the eigensolver itself (`solvers/davidson.py`, the H application in
`core/hamiltonian.py`), building on `symmetry.py` (spglib space groups already
present). **Lever:** the *dominant* one — eigensolver memory + diagonalization
cost. **Tool:** needs **full SageMath** (character tables, real/projective irreps).
**Risk:** high; largest project. This is the "algorithm design" track.

### The idea
At a k-point with little group `G_k` (the space-group operations that leave k
invariant mod a reciprocal lattice vector), the Bloch Hamiltonian `H(k)` commutes
with the representation of `G_k` on the plane-wave basis. Group theory then
block-diagonalizes it. The projection operator onto irrep Γ is

    P^Γ = (d_Γ / |G_k|) · Σ_{g ∈ G_k} χ^Γ(g)* · D(g)

where `D(g)` permutes-and-phases plane waves (`g` maps `G → RG`, with a phase
`e^{−i(RG)·τ}` for a fractional translation τ in nonsymmorphic groups), `χ^Γ` is
the character, `d_Γ` the irrep dimension. Applying `P^Γ` to the plane-wave basis
yields **symmetry-adapted plane waves** that group the G-vector stars into irrep
blocks. In that basis

    H(k) = ⊕_Γ H_Γ ,     Σ_Γ dim(H_Γ) = N_pw

so instead of one `N_pw × N_pw` eigenproblem (matrix-free, but the *subspace* the
Davidson solver carries is `N_pw`-tall) you solve several smaller independent ones.

### The win and the cost
- **Memory:** the Davidson subspace and the diagonalization both scale with the
  block size, not `N_pw`. Cost drops from `~N³` toward `Σ_Γ n_Γ³` with
  `Σ n_Γ = N`. For a 48-operation cubic little group the largest block can be a
  small fraction of `N` → a real, multiplicative memory/time reduction at
  high-symmetry k (Γ, X, L …), which are exactly the k-points band structures and
  many meshes hit most.
- **Why Sage:** you need **character tables and irreducible representations of
  space groups**, including **projective (ray) representations** for nonsymmorphic
  groups where the fractional translations introduce k-dependent phase factors.
  Sage has the group-theory machinery (`gap` backend); spglib gives the operations
  but not the irreps. This is the one place spend on a full Sage build is justified.

### Honest caveats
- Only high-symmetry k-points block meaningfully; general k in the mesh have
  trivial little group and see no benefit. So the win is real but *k-point
  dependent* — quantify the mesh-averaged saving before committing.
- The nonlocal projector and augmentation terms must be expressed in the adapted
  basis too — nontrivial plumbing through `hamiltonian.py`.
- This is weeks, not days. It's the right *eventual* target if the setup-layer
  tracks prove the approach and we want the eigensolver lever.

---

## Track 4 — Symbolic ground-truth oracles for `tests/gradcheck`

**Target:** `tests/gradcheck`, `tests/unit`. **Lever:** correctness insurance +
byproduct of Tracks 1–2. **Tool:** SymPy. **Risk:** none.

Generate exact reference values from symbolic derivations to pin the numerical
code: XC energy/potential/kernel at fixed `(ρ,σ,τ)`; real Gaunt coefficients;
Ewald real+reciprocal split for a tiny cell; structure factors. These are cheap,
they fall out of Tracks 1 and 2 for free, and they convert "matches quadrature /
matches autograd" (numerical vs numerical) into "matches closed form" (numerical
vs exact). Good first PR on its own because it's pure additive test coverage.

---

## Tooling

- **SymPy** covers Tracks 1, 2, 4 with no heavy build: `uv add --dev sympy` in the
  gradwave env. Ships pure-Python; loads instantly.
- **SageMath** (v10.9 is in nixpkgs unstable) is only needed for Track 3's space-
  group irreps. It's a large multi-hour source build on NixOS — defer until we
  actually commit to the symmetry track. When we do, it goes in a dev shell /
  `nix shell nixpkgs#sage`, not the runtime env (Sage is never a gradwave runtime
  dependency — it only runs the offline generator).
- Generated kernels are checked into `src/gradwave/…` as ordinary `.py`; the
  generator scripts live under `experiments/` or `scripts/` and are re-runnable but
  not a build-time dependency (same posture as the QE fixture regenerator).

## Suggested first move (when we go from notes to code)

1. **Track 4 + Track 1 warm-up, by hand:** tabulate nonzero real Gaunt coeffs for
   d–d coupling in SymPy, count the zeros (that number justifies Track 1), and drop
   the exact values in as a gradcheck oracle against `gaunt.py`. One sitting, zero
   risk, proves the `symbolic → verify` half of the pipeline.
2. **Track 2 proof-of-pipeline on LDA-PW92:** symbolic `e → v_xc → f_xc`, CSE,
   generate a torch kernel, diff against the current autograd path. Proves the
   `symbolic → codegen` half.
3. Decide r2SCAN codegen (Track 2 full) vs symmetry blocking (Track 3) based on
   which lever — XC speed/f_xc vs eigensolver memory — matters more for the
   workloads you actually run.
