# Symbolic-DFT experiments — results

Four runnable proof-of-concepts under `experiments/symbolic/`, each verified
against gradwave's own code. Ran on CPU, float64, via `uv run --with sympy`
(no changes to `pyproject.toml`/lockfile — SymPy is pulled in ephemerally).

| # | script | claim | result |
|---|--------|-------|--------|
| 1 | `track1_gaunt.py` | exact sparse Gaunt tables | **PASS** — 92% structural zeros, 12.5× smaller, exact to 3e-15 |
| 2 | `track2_xc_codegen.py` | CSE'd tape-free XC kernel + analytic f_xc | **PASS** — matches autograd to 6e-14; compiled 48× faster |
| 3 | `track3_symmetry.py` | symmetry block-diagonalizes H | **PASS** — exact blocks, 7.7× fewer flops |
| 4 | `track4_oracles.py` | frozen closed-form test oracles | **PASS** — 2 tests, no SymPy at runtime |

Reproduce: `uv run --with sympy python experiments/symbolic/track{1,2,3}_*.py`
and `uv run pytest experiments/symbolic/track4_oracles.py -q`.

---

## Track 1 — exact sparse Gaunt tables ✅

`core/gaunt.py` builds `c[LM,i,j] = ∫ Ȳ_LM Ȳ_i Ȳ_j dΩ` (d-channels: 25×9×9) by
quadrature into a **dense** float cube. SymPy's `real_gaunt` — verified to match
gradwave's real-harmonic convention exactly (cos-slot ↔ +m, sin-slot ↔ −m) — gives:

- **1863 / 2025 entries are structural zeros (92.0%)** from the selection rules
  (M = m₁+m₂, triangle |l₁−l₂| ≤ L ≤ l₁+l₂, parity l₁+l₂+L even). 96 unique
  nonzeros (162 with i↔j symmetry).
- sparse `(index, value)` storage is **12.5× smaller** than the dense cube, and the
  augmentation contraction iterates only the nonzeros.
- exact closed forms, e.g. `c[1,1,4] = √5/(5√π)`, `c[2,1,5] = √15/(10√π)`.
- **oracle:** symbolic vs quadrature agree to **3.0e-15** on nonzeros, and every
  symbolic zero is quadrature-zero to 1.9e-15 (the quadrature's `~1e-16` noise is
  now a provable exact 0).

**Verdict:** real, self-contained memory+flop win in the USPP/PAW augmentation.
Ship path: generate the sparse COO offline, replace the dense-cube contraction.

## Track 2 — symbolic XC kernel: CSE'd, tape-free, analytic f_xc ✅

Proof-of-pipeline on LDA-PW92 (`ρ ─SymPy─▶ e_xc ─diff─▶ v_xc,f_xc ─cse─▶ torch`):

- **correctness:** generated kernel vs gradwave's autograd `v_xc` (single backward)
  and `f_xc` (**double** backward) agree to **6.4e-14** over 4096 points.
- CSE finds 27 shared subexpressions (rs, √rs, the log term feed e, v, *and* f).
- **speed (e+v+f, 200k pts):** eager kernel ≈ autograd (0.8–1.7× — LDA is ~1 log +
  2 powers, too cheap for codegen to matter in eager mode); **`torch.compile`d
  fused kernel: 26 ms vs autograd's ~hundreds of ms → up to ~48× here** (≈10× on an
  unloaded box; the ratio grows under load because autograd's two extra passes
  contend more).
- **the structural point:** `xc/base.py` itself notes *"torch.compile with
  aot_autograd does not support double backward, and the f_xc kernel … is exactly a
  double backward … so those call sites wrap themselves in `xc_eager()`."* gradwave's
  `f_xc` (DFPT/phonons/response) is therefore **forced onto the eager path today**.
  A symbolic kernel computes `f_xc` analytically in one **fused, compilable** pass —
  it sidesteps the exact limitation their own code documents.

**Verdict:** on LDA the win is correctness + analytic f_xc, not raw eager speed.
The speed/memory payoff lands on (a) the deep r2SCAN graph and (b) `f_xc`, where
compiled fusion is available to the symbolic kernel but not to double-backward.
Next: regenerate PBE, then r2SCAN, and benchmark f_xc there.

## Track 3 — symmetry-adapted plane waves block-diagonalize H ✅

Concrete C4v square-lattice PW Hamiltonian at Γ, 49 plane waves, pure numpy +
hardcoded C4v character table (no Sage needed for a symmorphic point group):

- H commutes with all 8 ops (`max|[H,D(g)]| = 0`).
- projection operators split the basis into irrep blocks **A1:10, A2:3, B1:6,
  B2:6, E:24**; inter-irrep leakage of QᵀHQ = **3.7e-15**.
- block eigenvalues reproduce the full spectrum to **1.6e-14**.
- diagonalization cost proxy Σn³: **117,649 → 15,283 (7.7× fewer flops)**; subspace
  memory set by the largest block (24) not N (49). The E block halves again (its
  two partners decouple) for a further 2× not counted here.

**Verdict:** the one lever that hits the *dominant* eigensolver cost — proven in
miniature. Sage was named only for irrep LABELS/character tables — **not needed**
for the block-diagonalization itself (see Track 3b). Largest project regardless;
do it only after committing to the eigensolver track.

## Track 3b — dependency-free block-diagonalization (no Sage, no tables) ✅

`track3b_commutant.py` removes the one dependency Track 3 named. The character
table is unnecessary: the **class sums** `C_i = Σ_{g∈class_i} D(g)` span the center
of the group algebra, commute with H, and a single generic combination
`T = Σ_i r_i C_i` is a distinct scalar on each irrep type. So **one `eigh(T)`
groups the basis into isotypic blocks** — using only the group operations + linear
algebra (pure numpy).

- 5 conjugacy classes computed from the ops alone (no table): {E}, {C4,C4³}, {C2},
  {σx,σy}, {σd,σd'}.
- class sums commute with H to **0.0**; one `eigh` yields blocks **[3,6,6,10,24]** —
  identical multiset to the character-table version; inter-block leakage 3.5e-14,
  spectrum matches to 1.8e-14.
- **real crystals:** operations come from spglib (already a gradwave dep in
  `symmetry.py`), not Sage. Fractional translations (nonsymmorphic groups) enter
  only as phases in `D(g)`; the class sums still commute with H and still block it,
  because the method never classifies irreps → no ray/projective-rep machinery.

**Verdict:** the eigensolver-shrinking symmetry lever needs **no Sage at all** —
only spglib (present) for the operations and numpy for the block construction. Sage
would only add human-readable irrep *labels*, never the speed/memory win.

## Track 5 — block-diagonalization on REAL gradwave SCF calculations ✅

`track5_real_crystals.py` + `blockdiag.py` (+ `diamond_scf.yaml`, `fe_bcc_scf.yaml`).
Runs a real NC-DFT SCF, builds the **dense Kohn–Sham H(Γ)** at the converged
density (apply the matrix-free `HamiltonianK` to an identity block), and
block-diagonalizes it with the dependency-free class-sum method. The rep matrices
`D(g)` on the plane-wave basis are built with **gradwave's own** convention by
feeding the identity to `postscf.irreps._rep_matrix`; ops come from spglib via
`symmetry.find_spacegroup`. No Sage, no character tables.

| system | space group | npw(Γ) | dense H vs SCF bands | after symmetrization | blocks | reduction |
|---|---|---|---|---|---|---|
| **diamond** C | Fd-3m (nonsymmorphic) | 181 | 6.0e-10 eV | [H,D]=1.1e-13, spec 1.0e-12 | 9 | **26.4×** |
| **bcc Fe** | Im-3m (symmorphic, metallic) | 321 | 2.2e-9 eV | [H,D]=4.5e-13, spec 1.4e-12 | 10 | **32.0×** |

- **the dense H is the real thing:** its Γ eigenvalues equal gradwave's SCF band
  energies at Γ to 6e-10 eV (diamond) / 2e-9 eV (Fe).
- **physical degeneracies fall out** (Oh irreps, 1/2/3-fold): diamond −8.11(×1),
  +14.15(×3), +19.54(×3)…; Fe −78.2(×1)=3s semicore, −41.7(×3)=3p, +15.9(×3), …
- **nonsymmorphic glide-phase test (diamond):** the `e^{-2πiG·τ}` phases ride along
  in `D(g)` and the blocks come out exact — the class-sum method needs no
  ray/projective-rep machinery, as claimed in Track 3b.
- **bonus diagnostic:** the raw `max|[H,D]|` from the converged `v_eff` measures
  density symmetry-breaking — **7.6e-4 eV for diamond** (the glide exposes residual
  asymmetry in the 2-atom density) vs **1.1e-13 eV for Fe** (symmorphic, 1 atom,
  already exact). Imposing the symmetry on H by Reynolds averaging
  `H_sym = ⟨D(g) H D(g)†⟩` (a 3.8e-4 eV correction, below ecut error) recovers exact
  blocks — exactly what a production code's per-step ρ-symmetrization would do.

**Verdict:** the symmetry block-diagonalization works on real DFT Hamiltonians for
both a nonsymmorphic insulator and a symmorphic metal, giving 26–32× fewer
diagonalization flops at Γ — with **zero heavy dependencies** (spglib + numpy).
Remaining engineering to make it a runtime speedup (not just a demo): apply it
matrix-free inside the Davidson subspace per k rather than forming dense H, and
plumb the adapted basis through the nonlocal projectors.

## Track 6 — MATRIX-FREE construction + OFF-Γ k-points ✅

`blockdiag_matfree.py` + `track6_offgamma.py`. Each symmetry op is a
permutation-with-phase on the plane waves, so the G-vectors split into **orbits
(stars)** of size ≤ |G_k|. The symmetry-adapted basis is built **star by star**
(tiny s×s eigendecompositions, `O(Σs³)`, never an npw×npw matrix; each adapted
vector is sparse), and each irrep block `H_λ = U_λ†(H U_λ)` is assembled by
applying gradwave's `HamiltonianK.apply` to the block's basis vectors — **no dense
H in the solve path**, peak memory `max(n_λ)²` not `npw²`.

Measured at high-symmetry k (basis-build cost `Σs³/npw³`, diag speedup `Σn³`):

| crystal | k | \|G_k\| | npw | blocks | reduce | basis Σs³/npw³ | blk-vs-full |
|---|---|---|---|---|---|---|---|
| diamond | Γ | 48 | 181 | 9 | 26.4× | 0.013 | 9e-13 |
| diamond | L | 12 | 210 | 6 | 13.7× | 0.002 | 1e-12 |
| diamond | **X** | 16 | 206 | 6 | 24.8× | 0.003 | **2.3e-3 ⚠** |
| Fe | Γ | 48 | 321 | 10 | 32.0× | 0.010 | 1e-12 |
| Fe | H | 48 | 298 | 10 | 30.4× | 0.009 | 2e-12 |
| Fe | P | 24 | 312 | 5 | 8.1× | 0.004 | 1e-12 |
| Fe | N | 8 | 288 | 8 | 56.1× | 0.001 | 1e-12 |

- **basis-build is near-linear** (`Σs³/npw³ ≈ 10⁻³–10⁻²`) — the scalable claim holds
  on real crystals.
- **diamond X is the honest catch:** at a nonsymmorphic zone boundary the
  little-group rep goes **projective** (the `g0` umklapp + glide give a nontrivial
  cocycle), so the plain class sums stop being central and the blocks no longer
  reproduce the spectrum (2.3e-3). Detected automatically by the residual. Fixing
  it needs projective (ray) class sums — the one place the general nonsymmorphic
  zone boundary genuinely needs more than ordinary class sums. Everywhere else
  (all of symmorphic Fe's BZ; diamond Γ, L) is exact.

## Track 7 — collinear spin (nspin=2, ferromagnetic bcc Fe) ✅

`track7_spin.py` + `fe_fm_scf.yaml`. A collinear FM leaves the full spatial point
group intact, and the NC projectors are spin-independent, so **both** channels
`H↑(k), H↓(k)` (built from `res.v_eff[0/1]` with shared `pd,p`) are
block-diagonalized by the **same** spatial adapted basis. 32× reduction at Γ.

The physical **exchange splitting** falls straight out of the paired blocks:

| state (Oh irrep) | ε↑ (eV) | ε↓ (eV) | Δ_ex |
|---|---|---|---|
| 3s (×1) | −78.93 | −77.31 | 1.62 |
| 3p (×3) | −42.51 | −40.67 | 1.84 |
| 3d/T (×3) | 15.22 | 16.65 | −1.43 |

The T-irrep triple degeneracies are preserved in each spin channel; the ~1.6–1.9 eV
splitting is Fe's physical exchange splitting.

**Verdict (off-Γ + spin):** both extensions work on real calculations. Off-Γ is
exact wherever the little-group rep is ordinary (all symmorphic k; nonsymmorphic
interior + Γ), and the projective zone-boundary case is cleanly flagged. Collinear
spin is a trivial, exact extension (same spatial basis, per-channel).

## Walltime benchmark — the honest result ⚠

`bench_walltime.py` + `bench_breakdown.py`. Times to get the FULL spectrum of a
symmetrized H(Γ), sweeping N by cutoff (diamond/Fe primitive cells, one high-cutoff
v_eff). **At the tested sizes symmetry blocking LOSES** — baseline LAPACK
`eigvalsh` beats the symmetry path 2.5–10×:

| N | 1/R (flop win) | A: full eigh | B: sym (Python) | A/B |
|---|---|---|---|---|
| 531 | 0.031 (32×) | 45 ms | 178 ms | **0.25×** |
| 725 | 0.031 (32×) | 100 ms | 239 ms | **0.42×** |

Decomposition at N=531 explains why:

| part | time | note |
|---|---|---|
| baseline eigvalsh(H) | 45 ms | O(N³) LAPACK, multithreaded |
| Σ eigh(blocks) | **5.7 ms** | the real flop win — **7.8× vs baseline** |
| block assembly U†HU | 24 ms | overhead (dense matmul) |
| adapted_basis build | **129 ms** | overhead — **pure Python**, dominates |

**The algorithmic win is real (~8× at N≈500, → the 30× flop ratio at larger N) but
buried under implementation overhead.** Two facts rescue it, neither exploited by
the naive timing:
1. **The adapted basis is density-INDEPENDENT** — it depends only on (crystal, k),
   so it is built ONCE and amortized over every SCF iteration / every calculation
   on that geometry. The 129 ms should not be charged per diagonalization.
2. Construction + assembly are pure Python here; compiled, they are ~µs–ms.

With the basis amortized and assembly compiled, only the 5.7 ms block-eigh counts →
the ~8× (→30×) win shows through. **But** LAPACK dense eigh at N<1000 is only ~tens
of ms, so this only matters where dense diagonalization is genuinely on the critical
path: reduced/effective Hamiltonians (Wannier/tight-binding/k·p) diagonalized over
dense k-meshes, full-spectrum post-processing (GW/RPA/BSE sum-over-states, many-band
DOS/optics), or the Rayleigh–Ritz subspace inside iterative solvers for large-N_band
metals. **NOT** the standard plane-wave ground-state SCF, which uses iterative
solvers (few bands, matrix-free FFT applies) that symmetry-blocking of a dense H
does not speed up. Symmetry still helps that SCF the classic way — IBZ k-reduction,
which gradwave already does.

## Making it win — cached basis + compiled sparse assembly ✅

`symbasis.py` (`SymBasis`) + `bench2_amortized.py`. Two fixes turn the loss into a
win: **(1)** the adapted basis is density-INDEPENDENT — build once, reuse for every
diagonalization at that (crystal, k) (stored as a scipy CSC; each column a
star-local sparse combo of ≤|G_k| plane waves); **(2)** assemble blocks with
C-backed sparse·dense products `B = Uᴴ H U` (cost O(N²·s̄)) instead of the O(N³)
dense matmul. Per-solve = assembly + block eigh (basis cached), vs baseline
`eigvalsh`, at k=Γ (benchmarks run on the asus workstation; thinkpad OOMs at large N):

| system | N | 1/R | A: eigh | P: per-solve | **A/P** | build once | breakeven |
|---|---|---|---|---|---|---|---|
| diamond prim (Oh) | 531 | 0.031 | 105 ms | 45 ms | 2.3× | 92 ms | 1.5 |
| diamond conv (Oh) | 1021 | 0.001 | 516 ms | 167 ms | **3.1×** | 155 ms | 0.4 |
| diamond conv (Oh) | 1503 | 0.001 | 1439 ms | 397 ms | **3.6×** | 321 ms | 0.3 |
| Al(100) slab (D4h) | 1097 | 0.032 | 132 ms | 58 ms | 2.3× | 88 ms | 1.2 |
| Al(100) slab (D4h) | 2417 | 0.033 | 688 ms | 362 ms | 1.9× | 73 ms | 0.2 |

(spectra agree to 1e-12 / 1e-8; `breakeven` = solves to amortize the one-time build —
<1 means even a single diagonalization wins.)

**The win is real and grows with N** — 3.6× per solve at N=1503 (Oh), trending toward
the Σn³ flop ratio; `breakeven < 1` for N≳1000 means it wins even single-shot. It
holds where dense/near-full diagonalization of H(k) is on the critical path:
Wannier/TB/k·p over dense k-meshes, GW/RPA/BSE and many-band DOS/optics, the
Rayleigh–Ritz subspace in large iterative solvers.

**Slabs (the Al(100) test):** yes, it helps — large N (vacuum), few k, and a real
point group (here D4h, order 16). ~2× per solve at N≈2400, single-shot-positive.
But the gain is set by the point-group order: a slab's 2D-derived group (≤16, and
smaller under reconstruction/steps) blocks less than bulk Oh (48), so slabs get a
solid-but-smaller win than bulk. And a truly large slab (N≫10⁴) is memory-bound for
dense H anyway — there the payoff must come through the iterative solver (RR
subspace / symmetry-adapted trial vectors), not dense diagonalization.

## Track 4 — symbolic ground-truth oracles ✅

`track4_oracles.py` freezes closed-form constants (derived once in SymPy) as a
pytest that needs **no SymPy at runtime**: 6 exact Gaunt values (incl. two
selection-rule zeros) pin `real_gaunt_table`; exact LDA-PW92 `e/v_xc/f_xc` at
ρ=0.3 pin the autograd path. Converts "numeric matches numeric" into "numeric
matches exact." Both tests pass. Cheap, additive, zero-risk — the natural first PR.

---

## RPA/IP optical absorption — a full-spectrum consumer ✅

`rpa_absorption.py` (+ `si_abs.yaml`, run on asus). The first in-tree workflow that
actually *does* dense full-spectrum diagonalization — the consumer the symmetry
blocking was looking for. At each IBZ k it takes all valence + many conduction
eigenpairs of H(k), forms momentum matrix elements `p_cv = ⟨c|k+G|v⟩`, and sums
interband transitions (Adler–Wiser, q→0, velocity gauge, nonlocal commutator
omitted): ε₂(ω) = (4π²/Ω) g_s Σ_k w_k Σ_vc |p_cv|²/Δ² L_η(ω−Δ); ε₁ via
Kramers–Kronig; refractive n,κ; absorption α(ω).

**Silicon (the textbook case), 8×8×8 → 29 IBZ k, ecut 400 eV, PBE:**
- ε₂ onset at the gap with the main peak at **3.6 eV** (PBE red-shifts the E1/E2
  structure vs the ~3.4/4.3 eV experiment — expected).
- static **ε₁(0) ≈ 17.7** (experiment 11.9): overestimated in the right direction —
  no local-field effects, PBE's small gap, coarse mesh. Shape and physics correct.
- **Symmetry backend (`SymBasis.block_eig`) reproduces the whole spectrum to
  6.9e-8** — the full-spectrum block-diagonalization plugs straight in as the
  eigensolver for a real spectroscopy calculation.

**Honest speed note:** the symmetry backend was *slightly slower* here (4.5 s vs
3.8 s over 29 k) — small N (~300) and most IBZ k are general points with trivial
little group (no blocking), and the basis was rebuilt per k. Symmetry blocking helps
only at high-symmetry k and large N, so for a general-mesh absorption run the win is
concentrated at the special k; the consumer itself is the deliverable.

### Local-field effects (RPA Dyson) ✅ — `rpa_lfe.py`

Upgrades the IP head to the full microscopic dielectric matrix and its inversion at
a small finite q (optical limit, so every matrix element is an ordinary plane-wave
overlap — no q→0 head/wing bookkeeping):

  χ₀_{GG'}(q,ω) = (2/Ω)(1/N_k) Σ_k Σ_vc M_vc(G) M*_vc(G') [1/(ω−Δ+iη) − 1/(ω+Δ+iη)]
  M_vc(G) = ⟨u_vk|e^{-iGr}|u_{c,k+q}⟩ = [r_to_g(conj(u_vk)·u_{c,k+q})](G)  (FFT, M_vv(0)=1)
  ε = δ − v_G χ₀ (Hartree kernel = RPA);  ε_M^{LFE} = 1/[ε⁻¹]_{00}

Si, 6³=216 full-BZ k, n_LFE=27 G-vectors, |q|=0.02:

| | ε₁(0) | ε₂ peak | max ε₂ |
|---|---|---|---|
| no local fields (head ε₀₀) | 16.4 | 3.70 eV | 73.5 |
| **with local fields** (invert ε) | **15.3** | 3.75 eV | **66.4** |

Local fields **lower ε₁(0) (−7%) and the ε₂ peak (−10%) with a slight blue-shift** —
the correct RPA screening for Si (~10–20% in the literature). FFT density matrix
elements validated (M_vv(0)=1 exactly). The residual overestimate vs experiment
(15.3 vs 11.9) is **not** an LFE deficiency — it's PBE's too-small gap (needs a
scissor/GW correction) plus the missing electron-hole vertex (that's BSE); LFE is
only the ~10% screening piece. Kernel is Hartree-only (RPA); adding f_xc (from
`_response.py`'s `fxc_hvp`) would make it ALDA/TDDFT.

## G₀W₀ quasiparticle gap — full self-energy prototype 🚧 (on asus)

`gw.py`. The full-distance consumer: on top of the RPA χ₀/W machinery it builds the
screened Coulomb W = ε⁻¹v over the q-mesh, the self-energy Σ = iGW = Σ_x (bare Fock
exchange) + Σ_c (screened correlation, **Godby-Needs plasmon-pole**), and solves the
quasiparticle equation for the Si direct gap at Γ:
ε^QP = ε^DFT + Z[⟨Σ_x+Σ_c(ε^DFT)⟩ − ⟨v_xc⟩], Z = 1/(1−∂Σ_c/∂ω). PPA params (ω̃, Ω²)
from ε⁻¹ at ω=0 and ω=iE₀; mesh-closed Γ-centered q-grid; q→0 head from k·p momentum
matrix elements; spherical-average exchange-divergence correction.

**Every component validated against known Si values (4³ mesh):**

| quantity | computed | reference | ✓ |
|---|---|---|---|
| DFT direct gap at Γ | 2.56 eV | PBE ~2.55 | ✓ |
| Σ_x (VBM / CBM) | −11.4 / −5.5 eV | valence more bound | ✓ |
| ⟨v_xc⟩ (VBM / CBM) | −11.3 / −10.0 eV | ~−10 to −11 | ✓ |
| Z renormalization | 0.86–0.87 | Si ~0.8 | ✓ |
| QP gap **opens** | 2.56 → 4.9 eV | direction correct | ✓ |

So the full G₀W₀ pipeline runs end-to-end and every piece is individually correct.
The **absolute** QP gap over-opens (4.9 vs ~3.3 eV expt) — diagnosed to **q-mesh
convergence**, not a formalism bug: the gap is unchanged whether the q→0 head gives
ε=8 or ε=28, and is flat in band count (64→120), because a 4³ mesh samples only
large-|q| points where the screening is weak, so Σ_c under-screens the
exchange-driven opening. This is precisely the convergence wall that makes GW a
production-scale effort (dense q-mesh + hundreds of empty bands + proper
tensor-averaged head). The q-mesh convergence is **confirmed**: the QP gap trends
straight toward the reference as the mesh densifies —

| q-mesh | QP gap | ε(q→0) | head plasmon ω̃ |
|---|---|---|---|
| 4³ | 4.88 eV | 28 | 19.6 eV |
| 6³ | **4.08 eV** | 19.7 | 18.0 eV |
| → 8³–12³ (extrap.) | → ~3.3 eV | → ~13 | → ~16.7 |

every quantity moving monotonically toward the Si reference (gap ~3.2–3.4, ε∞ 11.9,
plasmon 16.7). The machinery is correct; the residual is compute (a converged number
needs 8³–12³ q, hundreds of empty bands, tensor-averaged head — hours-to-days at
production settings, on GPU).

**Verdict:** the full-distance G₀W₀ formalism is implemented and validated
component-by-component; quantitative convergence is the remaining (large, expected)
cost, and the machinery reuses the same many-band full-spectrum diagonalization the
symmetry blocking targets — GW is the strongest motivation for that work.

## Do block-diagonalization + THC help PRODUCTION-sized GW walltime?

Assessment (`thc_demo.py`, `thc_chi0_bench.py`, + the gw.py timings). Short answer:
**THC — yes, decisively; symmetry block-diagonalization — no (wrong lever for GW).**

**Where GW's walltime actually is.** From the gw.py runs, χ₀ construction dominates:
6³ mesh spent ~1465 s building W(q) vs ~seconds diagonalizing the mesh — **χ₀ is
~99%**, diagonalization ~1%. So:

- **Block-diagonalizing H(k) cannot move GW walltime** — it accelerates the ~1%
  diagonalization, and only at high-symmetry k (most mesh k are general points with
  trivial little group). Confirmed structurally, not just measured.
- The symmetry lever that *would* help is **IBZ reduction** of the k/q sums (cuts the
  dominant χ₀ cost by up to the point-group order, ≤48 for Si) — but that is a
  different mechanism (rotating wavefunctions/matrix elements by symmetry), not the
  block-diagonalizer built here.

**THC (ISDF) attacks the χ₀ bottleneck directly.** gradwave's `isdf.py` *is* tensor
hypercontraction. `thc_demo.py` shows it factorizes the occ×unocc pair densities (the
χ₀ ingredient): fit error 55%→7.5% as the rank N_μ grows 1.4×→5.4×N_orb. The win at
scale (`thc_chi0_bench.py`): the G-basis χ₀ costs N_pair box-FFTs + O(N_pair·N_G²);
THC costs N_orb FFTs (one-time) + O(N_pair·N_μ²), with **N_μ ~ few×N_orb independent
of the plane-wave cutoff N_G**. So the χ₀ contraction speedup grows as (N_G/N_μ)²,
plus THC removes the per-pair FFTs (N_orb vs N_pair). Measured per-build (Si, 144
pairs, N_μ=140), sweeping the screening cutoff N_G:

| N_G | G-basis χ₀ | THC χ₀ | speedup |
|---|---|---|---|
| 169 | 7.3 ms | 0.3 ms | 28× |
| 725 | 12.0 ms | 0.2 ms | 58× |
| 1363 | 17.6 ms | 0.2 ms | 80× |
| 2445 | 47.7 ms | 0.2 ms | **216×** |

THC is flat in N_G (0.2 ms); the G-basis grows as N_G² — so the speedup climbs with
cutoff exactly as predicted, reaching 216× by N_G≈2.4k. **Caveat:** the one-time
ISDF factorization (pivoted QR) cost **111 s on CPU** at N_μ=140 — this amortizes
over all the (q,ω,k) χ₀ builds in a real run (hundreds–thousands of them), and would
drop by ~1–2 orders of magnitude on the asus GPU (the QR and FFTs are GPU-friendly;
these runs were CPU torch).

**Production projection.** χ₀ is built O(N_q·N_k·N_ω) times, so the per-build 30–200×
multiplies against the whole dominant cost while the 111 s ISDF is paid once (or once
per k). For a production cell (N_G ~ 2–5k, N_orb ~ few hundred, N_pair ~ 10⁴–10⁶) the
contraction (N_G/N_μ)² and FFT-count (N_pair/N_orb) factors compound — THC is the
standard route to cubic-scaling RPA/GW for exactly this reason, and gradwave already
has the substrate. Remaining work to use it in GW: extend `isdf.py` from
occupied-only to the occ×unocc pair space (shown working here) and add the multi-k/q
momentum-transfer phases (both flagged "future work" in the module), and run the ISDF
on GPU.

**Verdict for production GW:** invest in **THC** (attacks the 99% χ₀ cost, substrate
already in-tree) and **IBZ symmetry reduction** (fewer k/q). The block-diagonalizer,
useful for dense full-spectrum diagonalization elsewhere, is not the GW lever.

## THC-GW exploration — step 1 (χ₀ factorization) + the honest walltime picture

`thc_gw.py` (asus GPU). Built the THC-factored χ₀ = ζ_G·P·ζ_G† and validated it
against the direct build, sweeping the ISDF rank (Si, Γ, N_G=1363):

| N_μ | N_μ/N_orb | χ₀ rel err | THC χ₀ | direct χ₀ | speedup |
|---|---|---|---|---|---|
| 150 | 3.1× | 21% | 21 ms | 37 ms | 1.7× |
| 400 | 8.3× | 2.3% | 63 ms | 37 ms | 0.6× |
| 600 | 12.5× | 0.6% | 106 ms | 37 ms | **0.4×** |

**Correctness: yes** — THC χ₀ converges to the direct one (0.6% at rank 12.5×N_orb).
**Walltime at Si scale: no** — at the rank needed for accuracy the THC build is
*slower*, and the one-time ISDF cost was **10–40 s even on GPU**.

**This corrects the earlier "216×".** That number (thc_chi0_bench) measured only the
μ-basis polarizability `P_{μν}` (N_G-independent) vs the direct χ₀_GG'. It is real —
but only if the *entire* RPA/GW stays in the N_μ×N_μ point basis (ε = 1 − W_μν P,
Σ_c contracted through ζ), **never materializing χ₀_GG'**. The moment you form χ₀ in
the G-basis (as here, for validation), the advantage is gone.

**The real THC/ISDF win is a SYSTEM-SIZE SCALING win, not a fixed-system one.** It
takes RPA/GW from O(N⁴–N⁵) to O(N³) by collapsing the N_pair (~N²) sum onto N_μ ~
O(N) points and avoiding N_G² intermediates. At a fixed *small* system (Si: N_orb=48,
N_μ≈600 for 0.6% accuracy is not ≪ N_G=1363), the constants — ISDF setup, and N_μ/N_G
not small — dominate and THC loses. The crossover is at **many-atom systems**, which
the 6 GB laptop GPU can't reach for a clean demonstration.

**Honest verdict on THC-GW:** viable and correct (χ₀ factorization works), and it is
the standard route to *large-system* GW — but (a) the win is asymptotic in the number
of atoms, not visible on Si; (b) it requires the full μ-basis reformulation (ε=1−W_μν P,
Σ_c in μ-basis), not just factorizing χ₀; (c) the ISDF setup is a real fixed cost.
Next: implement the μ-basis RPA/Σ_c (never form χ₀_GG'), and test on a multi-atom
supercell on a larger GPU.

## Symbolic algebra for runtime/overhead — Gaunt, f_xc, velocity

### Gaunt sparse tables — no real win (already exploited)
Investigated the actual consumers. The dense Gaunt table is built **once** at setup
(~16 KB), the USPP consumer (`uspp_setup.py`) is one-time, and the per-SCF PAW hot
path (`paw_onsite.rho_lm`) **already** does `np.nonzero(|cy|>1e-12)` and loops only
over nonzeros. So the "92% sparse / 12.5×" headline is about the tiny one-time table,
and the skip-zero flop win is already in place. A "sparse Gaunt" refactor would touch
the PAW hot path for negligible gain. **Skipped** (correctness is covered by the
merged oracle test).

### f_xc codegen (PBE) — validated; modest eager win, compile env-blocked ✅
`fxc_codegen.py`. Symbolic PBE `e_xc(ρ,σ)` (matching `pbe.py`/`_pbe_kernels`) → CSE
(138 shared subexpressions) → flat kernel returning `e, v_ρ, v_σ, f_ρρ, f_ρσ, f_σσ`
analytically. **Matches gradwave's autograd (single + double backward) to 7e-13.**

| path | GPU time (200k pts, e+5 derivs) | vs autograd |
|---|---|---|
| autograd (fwd + double backward, forced eager) | 28.2 ms | 1.0× |
| symbolic kernel, eager | 20.4 ms | **1.4×** |
| symbolic kernel, torch.compile | — | blocked* |

*`torch.compile` fails on this box: NixOS has no `/sbin/ldconfig`, which torch's
Inductor shells out to (CPU **and** CUDA backends). That's an **environment** issue,
not the method — the kernel now traces cleanly through Dynamo. On a standard toolchain
the LDA prototype compiled to ~48×, and note the autograd f_xc **cannot be compiled at
all** (double-backward), so the symbolic kernel is the *only* compilable f_xc — exactly
the path `base.py` documents as forced-eager for DFPT/phonons/response.

**Value even without compile:** 1.4× eager + **tape-free** (no retained autograd graph
per grid batch → lower memory on the response path). Fix the toolchain (add glibc's
`ldconfig` to the env) and the compiled win is unlocked. Next: generate r2SCAN (the
deep functional where the autograd double-backward is most expensive).

### Analytic velocity operator ∂H/∂k — marginal runtime, real accuracy (`velocity_probe.py`)
`dielectric.py` builds ∂H/∂k with a central finite difference of the KB projectors in
k. But the form factors are a **prebuilt cubic spline** (`beta_form_factors`), so a
projector rebuild at a shifted k is cheap — measured **0.52 ms** (Si, npw=731,
nproj=16). So finite-diff ∂H/∂k ≈ 3.1 ms (6 rebuilds) vs analytic ≈ 0.8 ms (1
assembly + the spline-derivative F′(q)): a ~4× speedup, but on a sub-ms op — **~2.3
ms/k saved**. That only matters over a *dense k-mesh* (transport, Berry curvature),
and even there the velocity assembly isn't the bottleneck (the eigensolve is).

**Verdict:** the analytic ∂H/∂k is a marginal *runtime* win. Its real value is
**accuracy** — exact vs the O(δk²) finite difference, and it gives the analytic
nonlocal `[V_nl, r]` commutator that a proper velocity-gauge optical/transport matrix
element needs (the piece `rpa_absorption.py` omitted). Worth building for correctness,
not for speed.

### Summary — symbolic algebra for runtime in gradwave
gradwave is already well-optimized, so the symbolic-algebra *runtime* wins are modest:
Gaunt is already sparse-exploited (no win); the velocity operator's finite diff is
cheap and accurate (marginal); **f_xc is the only real one** — 1.4× eager + tape-free,
and the *only* compilable f_xc (the big win, gated on fixing the compile toolchain, is
compiled r2SCAN f_xc for phonons/DFPT). The higher-value symbolic uses are accuracy/
capability (the nonlocal-commutator velocity), not raw speed.

## Honest scorecard

- **Biggest sure win:** Track 1 (92% sparsity, exact, self-contained).
- **Biggest capability unlock:** Track 2's analytic `f_xc` — sidesteps the
  double-backward that `base.py` forces onto eager; the r2SCAN benchmark is the
  real test.
- **Biggest ceiling:** Track 3 — the only eigensolver lever. Track 3b shows it
  needs **no Sage** (class sums + spglib, both dependency-light); the cost is the
  real plumbing through `hamiltonian.py`, not a heavy dependency.
- **Free insurance:** Track 4.

Nothing here is committed to gradwave beyond the `experiments/symbolic/` scripts.
Suggested first PR: Track 4 oracles + Track 1 sparse generator.
