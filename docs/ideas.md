# Ideas and future work

A running backlog for gradwave, with enough reasoning attached that each item can
be picked up cold. Not commitments, just directions worth taking.

The open backlog comes first, grouped by theme, then a "Done and resolved" archive
that keeps the reasoning for items already built or settled as a measured negative.
Each idea carries a **status** line (open / prototyped / landed / tried-and-rejected,
with date and PR refs where they exist), then the idea, the evidence, and the next
step. The 2026-07-23 effort-vs-value survey directly below is the entry point and
the map over the sections. As items land, they move to the archive and the survey
is re-ranked.

Themes, in return order. Many-body electronic structure (exact exchange, hybrids,
RI/THC). Differentiable and learned functionals. Magnetism and spin-orbit coupling.
Post-SCF and spectroscopy. Coverage, correctness, and the error budget. Performance
and scaling.

# Open backlog

## Capability gaps: effort vs value (2026-07-23 survey)

A whole-code survey of the electronic-structure methods gradwave does *not* yet
have, ranked by return. Effort and value are calibrated to this codebase, to what
substrate already exists (the differentiable-by-construction design, the ISDF/ACE
Fock build, the nspin=1 autograd force/stress machinery) and the difficulty the
sections below already frame. Scale is Low / Med / High / V.High. Several rows have
a dedicated section further down that holds the real reasoning. This table is the
map, not a replacement.

For context, the ground-state semilocal-plus-hybrid-plus-magnetism surface is
already broad. LDA/PBE/r2SCAN, a self-consistent PBE0/HSE hybrid on a k-mesh, DFT+U
from linear response, the full spin-unpolarized to collinear to non-collinear to
SOC ladder, norm-conserving and USPP/PAW pseudos, forces/stress/bands/DOS/PDOS/COHP/
Bader/Γ-phonons/EOS, four smearing schemes, and Davidson plus CheFSI. The gaps are
mostly *beyond* ground-state GGA/hybrid DFT, plus a few coverage holes inside it.

| Feature | Effort | Value | Why / reusable substrate |
|---|---|---|---|
| nspin=2 forces / stress / bands | Low–Med | High | Plumbing — the nspin=1 autograd force/stress path and the noncollinear path both exist; thread the spin channel. Unblocks routine magnetic relaxation and bands. A coverage hole, not new physics. See "Full nspin=2 and PAW coverage". **LANDED (2026-07-24)** — nspin=2 forces (#45), stress (#58), NC and USPP/PAW bands (#45), KPM-DOS and ELF, and the dielectric/Born response (#65), plus fixed-spin-moment SCF (#45). |
| D3/D4 dispersion | Low | High | Geometry-only pairwise sum plus damping, no SCF change, trivially differentiable. Directly enables the layered / adsorption / 2D-magnet applications. **LANDED (2026-07-24)** — D3(BJ) energy, forces, and stress, wired through the CLI and the ASE calculator (#59, #69); D4 not attempted. |
| Elastic constants | Low–Med | High | `stress()` is already autodiff-through-strain; apply strain patterns and fit. The Phase-2 mechanical buildout. |
| Stress with DFT+U / fully-relativistic | Low–Med | Med | Removes two `NotImplementedError` guards; enables variable-cell for correlated and SOC systems. |
| NLCC force term | Low | Med | Small missing analytic term; correctness for NLCC-pseudo forces. **LANDED (2026-07-24)** — the core-correction force for LDA/GGA (#64) and meta-GGA (#68); the USPP / noncollinear meta-GGA NLCC edge stays gated. |
| Dipole correction + external E-field | Med | High | Sawtooth potential plus energy term. Unblocks slabs, RAIRS, and charged defects — an entire application class. See "RAIRS and a slab dipole moment". |
| Tetrahedron (Blöchl) BZ integration | Med | Med | Non-smooth weights need care against autograd, but smearing already covers metals; the win is insulator DOS/optics accuracy. |
| RPA correlation energy | Med–High | High | The ISDF/ACE substrate is already built for exact exchange; a differentiable RPA is novel and is physics GGA cannot do. See the RI/THC section. |
| Berry-phase polarization (modern theory) | Med | Med–High | Berry phase over k-strings; an independent route to Born charges, enables ferroelectrics, and is the precursor to topology. |
| Real-time / linear-response TDDFT | Med–High | Med–High | RT-TDDFT fits the differentiable design naturally and gives optical spectra far cheaper than BSE. |
| Meta-GGA under non-collinear / SOC | Med–High | Med | Wire the τ operator onto the spinor H-apply so r2SCAN meets magnetism+SOC. Narrower payoff. See "Learned meta-GGA". |
| Finite-q phonons / dispersion | High (DFPT) / Med (supercell) | High | No arbitrary-q DFPT today. The supercell frozen-phonon route is cheaper and unlocks thermodynamics and stability. See "Phonon band structures". **Supercell route LANDED (2026-07-24)** — dispersion + phonon DOS, norm-conserving nspin=1 (#61); arbitrary-q DFPT, the LO-TO nonanalytic term, and harmonic thermodynamics remain open. |
| Dielectric / Born for metals and nspin=2 | Med | Med | Generalize the current nspin=1-insulator-only path; needed for RAIRS and magnetic IR. **nspin=2 LANDED (2026-07-24)** — collinear spin-polarized ε∞ and Born charges through the E-field DFPT (#65); the metals (partial-occupation) path remains open. |
| Wannier functions / Wannier90 interface | Med | Med | Export overlaps (Med) or build MLWFs internally (higher); enables interpolation, transport hand-off, and topology. |
| Topological invariants (Z₂ / Chern) | Med (given Berry / Wannier) | Med | On-brand given the Bi₂Se₃ band-inversion demo, but niche; depends on the two rows above. |
| libxc binding / more functionals | Med (per functional if transcribed) | Med | Breadth (PBEsol, SCAN, B3LYP …), but a C binding breaks the pure-autodiff design and hand-transcription is Med each. A real tension with the ethos. |
| Higher-order MP smearing | Low | Low | Only MP-1 today; marginal accuracy gain. |
| Standalone Hartree–Fock | Low | Low | The Fock operator already exists inside the hybrids; rarely wanted for solids. |
| Extra PP formats (GTH / HGH / psp8) | Med | Low–Med | UPF already covers PseudoDojo / SG15 / PAW; mostly convenience. |
| GW quasiparticle | V.High | High | The standard ask for real gaps; ISDF helps but frequency integration plus the self-energy is a major build. |
| BSE (excitons / optical) | V.High | High | Needs GW first plus the electron–hole kernel; the highest-effort item here. |

The four quadrants.

- **Quick wins** (high value, low effort, do first). nspin=2 derivatives, D3
  dispersion, and elastic constants have since landed (elastic constants #41, the
  rest 2026-07-24); DFT+U and fully-relativistic stress remain.
- **Big bets** (high value, high effort). RPA → GW → BSE built in that order on the
  ISDF substrate, finite-q phonons, and the dipole/field boundary conditions.
- **On-brand niche** (differentiability showcase, moderate value). Berry-phase
  polarization, RT-TDDFT, topological invariants.
- **Low priority.** Extra PP formats, higher-order MP smearing, standalone HF, libxc
  breadth.

The largest genuinely-absent physics is the excited-state / many-body tier (GW, BSE,
TDDFT, RPA). Every gap, level alignment, and spectrum today comes from semilocal or
hybrid DFT with its self-interaction error. The two cheapest gaps this survey named,
dispersion and the nspin=2 derivatives, both landed on 2026-07-24; the many-body tier
is what is left at the top of the return ranking.

# Many-body electronic structure

Exact exchange is the largest missing physics in the code. Every energy, gap, force,
and adsorbate level gradwave produces comes from a GGA electronic structure with
self-interaction error, so band gaps come out too small and defect and adsorbate
levels land in the wrong place. A hybrid needs a Fock exchange operator applied each
SCF step, the O(N⁴) object that resolution of identity and tensor hypercontraction
exist to reduce. The scaling work and the accuracy work therefore overlap.

The CheFSI no-go, the EOS-batching analysis, and the 128-atom memory cliff (see the
performance sections) were all measured on a consumer RTX 3050, whose fp64 throughput
is a small fraction of its fp32 and whose 6 GB caps the grid. Those numbers bound that
card. A datacenter card with real fp64 changes the CheFSI arithmetic on its own. The
more durable approach is to cut the operation count itself, which helps on the CPU
path and on any GPU, and that is what RI and THC do.

## RI and tensor hypercontraction (ISDF)

**Status: first cut and self-consistent k-mesh hybrid SCF landed (2026-07-20);
learned hybrid landed; RPA and the fine-mesh/HSE tails open.**

### The framing

RI expands products of orbitals in an auxiliary basis so a four-center
electron-repulsion object factorizes through two- and three-center intermediates. In
a plane-wave code the Hartree term is already O(N log N) through the FFT, so RI is not
a Hartree win. Where it pays is exact exchange. The Fock term is the O(N⁴) bottleneck
that keeps hybrids out of the code, and RI on the orbital pair densities
`rho_ij(r) = psi_i*(r) psi_j(r)` is the standard route to make it affordable.

A plane-wave code already carries a complete auxiliary basis in the dense-grid plane
waves, so a pair density is exact on the grid. The cost problem is the number of
pairs, O(N²), each needing an FFT to Coulomb-couple. The fit is a linear solve against
a metric, differentiable end to end, so a learnable hybrid can carry the exchange-
mixing fraction and the range-separation length as trained parameters on top of an
RI-compressed Fock build.

Tensor hypercontraction factorizes the pair-product tensor into a small set of
interpolation points and vectors, collapsing an object O(N²) in orbital pairs and
O(N_grid) in real space to O(N) points times a compact factor. The plane-wave-native
form is interpolative separable density fitting (ISDF, Lu and Ying), which writes
`psi_i(r) psi_j(r) ≈ Σ_μ zeta_mu(r) psi_i(r_mu) psi_j(r_mu)` over a small chosen point
set `{r_mu}`. A QR-pivoted or centroidal-Voronoi point selection makes the rank grow
like N rather than N². It cuts the FLOP count of the exchange and correlation builds
directly, so unlike the CheFSI fp32 story it does not depend on a card's fp64
throughput. It is the standard enabling technique for affordable exact exchange and
RPA correlation in plane-wave codes (Qbox, PWDFT, the ISDF-K line). Its accuracy knob
is the interpolation-point count, and the point selection is the subtle part, so the
validation is budgeted against a direct plane-wave Fock build, not another
approximation, with the rank reported as a convergence parameter.

Build order was ISDF as a compression of the pair densities, validate the compressed
Fock exchange against direct to milli-eV, then the learnable-hybrid parameters and,
separately, the RPA correlation contraction. Each stage is a set of tensor
contractions, staying inside the differentiable-by-construction design.

### What landed (2026-07-20, all at single-k Γ unless noted)

- **ISDF energy.** `postscf/isdf.py`. The pivoted-QR interpolation-point selector
  (`select_interpolation_points`, exact pair matrix for small orbital sets,
  randomized Khatri–Rao sketch above a threshold), the `M S⁻¹` fit (`build_isdf`),
  the interpolation-vector Coulomb coupling (`_coulomb_coupling`, reusing the
  `hartree.py` G=0-excluded 4πe²/G² kernel), and the contracted exchange
  `E_x = −½ Σ_μν W_μν |D_μν|²` (`ISDFExchange.energy`). Validated against the direct
  O(N²) pair-FFT build `exchange_energy_direct`. Past the co-density rank, ISDF equals
  direct to machine precision (`tests/unit/test_isdf.py`, converged Γ Si in
  `tests/integration/test_isdf_vs_direct.py`); below saturation the error falls
  monotonically with rank (8-atom Si, 6.4 eV → 0.24 eV → 1e-14 as n_μ → 40 → 80 → 136,
  the 16·17/2 real-orbital co-density rank). The rank is the accuracy knob as promised.
- **Exchange operator + ACE.** `postscf/exchange.py`. The direct Fock operator
  `V_x φ = −Σ_j ψ_j v[ψ_j*φ]` (`exchange_operator_direct`, the O(N_occ²) reference);
  the ISDF operator `exchange_operator_isdf` (`V_x ψ_t = −Σ_μ v[ζ_μ] B(r,r_μ)
  ψ_t(r_μ)`, one Coulomb solve per interpolation vector instead of N_occ² pair
  solves); and adaptively-compressed exchange (`build_ace`, Lin Lin JCTC 2016), the
  Cholesky of the exchange matrix giving a low-rank `V_x ≈ −Σ_k |ξ_k⟩⟨ξ_k|` exact on
  the occupied subspace, the object a generalized-KS Hamiltonian carries. Validated
  (`tests/unit/test_exchange.py`, `tests/integration/test_exchange_operator.py`).
  Operator energy equals the contracted energy to machine precision, the ISDF operator
  saturates to direct (rel-err 2e-2 → 1e-13 as rank fills), and ACE reproduces
  `V_x ψ_n` on every occupied n to ~5e-14.
- **Multi-k exchange, range-separated kernel, learnable slot.**
  `postscf/coulomb_kernel.py`, `postscf/exchange_multik.py`. `coulomb_kernel` gives
  K(q+G) in `full` (PBE0), `short_range` (erfc, HSE), and `long_range` (erf) modes, ω
  differentiable, with K_sr+K_lr=K_full at q+G≠0. At q+G=0 the screened kernel is
  *finite* (π e²/ω²) while full/long-range drop the divergence, so **screened (HSE)
  exchange needs no singularity correction**, the reason to reach for it first
  (`tests/unit/test_coulomb_kernel.py`). `multik_exchange_energy` runs the full
  exchange over a k-mesh through the co-density momentum q = k′−k, the kernel at
  |q+G|². It reduces to the Γ direct build at one k-point (measured exact) and gives a
  finite real exchange on a 2×2×2 full-BZ Si mesh, requiring an unreduced mesh
  (`use_symmetry=False, time_reversal=False`) and the dense grid, both enforced by
  convention (`tests/integration/test_exchange_multik.py`). `HybridExchangeParams` /
  `hybrid_exchange_energy` is the learnable slot, α (sigmoid) and ω (softplus)
  mirroring `core/xc/learnable.py`, initialized HSE-like, with autograd dE/dω matching
  finite difference to 1e-9.
- **Self-consistent hybrid SCF + ISDF-K.** `postscf/hybrid.py` runs a self-consistent
  PBE0-form global hybrid at Γ. `ScaledExchangePBE` scales semilocal exchange by
  (1−α); `GammaFockExchange` is an SCF `fock` hook rebuilding the ACE operator from the
  current orbitals each iteration (lagging one step like DFT+U), adding α·V_x to
  `BatchedHamiltonian.apply` and α·E_x as the new `EnergyBreakdown.fock` term. The hook
  is guarded, default `fock=None` leaves every existing path bit-identical (golden
  energies and the E↔H gate still pass). Validated
  (`tests/integration/test_hybrid_scf.py`). α=0 reproduces PBE exactly; PBE0 (α=0.25)
  converges and opens the Γ Si gap 2.32 → 2.97 eV; the Fock energy matches the ACE
  energy on the converged orbitals. The spin factor (2/nspin in the energy, none in the
  operator) keeps energy and operator derivative-consistent. `postscf/isdf_k.py` is the
  multi-k ISDF acceleration, one shared interpolation set fit across all occupied
  orbitals over the BZ, contracting through a per-q Coulomb matrix V^q and per-k
  point-Grams A_k with N_μ FFTs instead of N_k²·N_occ² co-density FFTs. It reduces to
  the Γ ISDF build exactly and matches direct `multik_exchange_energy` at saturated rank
  (`tests/integration/test_isdf_k.py`).
- **k-mesh hybrid SCF.** `exchange_multik.multik_exchange_operator` is the direct
  multi-k Fock operator W_{tk} = V_x ψ̂_{tk}, summing over the whole BZ through
  q = k−k′ and the range-separated kernel (`coulomb_potential_q`). At one k-point it is
  exactly `exchange.exchange_operator_direct`, and its energy trace matches
  `multik_exchange_energy` for full and screened kernels (machine-precision gates,
  `tests/integration/test_hybrid_kmesh.py`). `hybrid.py::MultiKFockExchange` is the
  k-mesh `fock` hook, ACE-compressing per k and applying α·V_x block-by-block over the
  k batch plus α·E_x; `hybrid_scf` routes through it. α=0 reduces to the PBE SCF on a
  mesh exactly; PBE0 on a (2,1,1) full-BZ mesh converges, the Fock term equals
  α·(2/nspin)·`multik_exchange_energy` on the converged orbitals (~1e-13), and the gap
  opens relative to PBE. Screened caveat. The screened Fock *operator* is exact and
  energy-consistent, but `ScaledExchangePBE` scales the *whole* PBE exchange by (1−α),
  correct for full-range PBE0 but not complete HSE (which also range-separates the
  semilocal exchange, keeping long-range PBE and removing only the short-range
  fraction). Use `mode="full"` for a physically complete SCF until wPBE lands.
- **Learned hybrid.** The mixing α and screening ω train end to end.
  `hybrid.differentiable_hybrid_energy(res, params)` turns a converged hybrid SCF into
  a differentiable objective via the stationary-energy (Hellmann–Feynman) derivative.
  At self-consistency the density is variational, so dE_total/dθ = ∂E_total/∂θ, only
  the *explicit* θ-dependence of the exchange terms on the frozen converged orbitals
  survives. It returns a scalar equal to `res.energies.total` whose (α, ω) gradient is
  that exact derivative, and `hybrid_scf(..., params=)` solves at the current values
  (the SCF stays `no_grad`, the gradient rides the converged result). Validated
  (`tests/integration/test_learned_hybrid.py`). The differentiable energy equals the
  SCF total; dE/dα matches a finite difference of *re-converged* SCF energies to 6e-7
  (rel), dE/dω to 2e-3; a backward+optimizer step moves (α, ω).

### What remains, in build order

A Gygi–Baldereschi q+G=0 correction to complete unscreened `full`/PBE0 at fine meshes
(the divergent q+G=0 cell is dropped today, converging slowly in N_k); the range-
separated (wPBE) semilocal exchange to complete screened HSE; a truncated Coulomb for
physically isolated molecules; and the RPA correlation contraction. GW and BSE sit
above these, built on the same ISDF substrate, and are the top of the return ranking.

## Exact exchange and hybrid functionals

**Status: a learned PBE0-form hybrid SCF runs on a k-mesh, mixing and screening are
trainable.** gradwave self-consistently solves a PBE0 hybrid on a full-BZ k-mesh (gap
opening measured) and trains its mixing and screening against a target, through the
ISDF/ACE Fock build, the k-mesh lift, and `differentiable_hybrid_energy` (see the
ISDF section above for the landed pieces and their validation). This was the reason
the two scaling items above are worth building. The remaining reach is the same as
any hybrid, finer meshes (Gygi–Baldereschi) and a complete screened form (wPBE).

# Differentiable and learned functionals

## Learned meta-GGA and the kinetic energy density

**Status: infrastructure and r2SCAN landed (2026-07-21); forces, stress, and the
generalized-KS operator all built; USPP/PAW-τ, spinor-τ, and a learnable r2SCAN-form
functional remain open.**

Meta-GGA is the natural next rung for the differentiable-XC work and the cheaper
stepping stone before hybrids (roughly a week against a much larger EXX build). The
current learnable functional spans GGA form only, the two PBE parameters kappa and mu,
so it cannot fit, learn against, or compare with the functionals people use (SCAN,
r2SCAN all depend on the kinetic energy density τ). Meta-GGA is what lets `train_xc_paw`
learn a real functional rather than only recover PBE, and it is the item that most
directly distinguishes gradwave from a very well-validated second copy of QE.

### What landed

- **Infrastructure (nspin=1 and 2).** `core/metagga.py`. The kinetic-energy density
  `tau_b` (τ = ½ Σ_k w_k Σ_n f|∇ψ|² on the dense grid, one extra i(k+G) factor on the
  density-build FFT, differentiable in the coefficients) and the generalized-KS
  operator `metagga_tau_operator` (V_τψ = −½∇·(v_τ∇ψ) = −½ Σ_d i(k+G)_d·F[v_τ·F⁻¹[i(k+G)_d
  c]], Hermitian, band-chunked). The XC interface gained `needs_tau` plus an optional τ
  arg on both `XCFunctional` and `SpinXC`, backward compatible (every existing GGA/LDA
  path stays bit-identical). The SCF loop builds τ per channel each iteration and lags
  it one step (like the Fock/DFT+U rebuilds), extracting v_τ = ∂e_xc/∂τ by autograd on
  a τ leaf and injecting the operator additively into the H-apply through the same wrap
  the hybrid Fock uses, with v_xc evaluated at fixed τ so the multiplicative and τ-
  response pieces stay separated. τ is one FFT-per-band on the existing g→r path and
  the operator is the only genuinely new physics, making this a generalized-KS scheme
  that touches the H-apply. Validated intrinsically, no QE reference needed
  (`tests/unit/test_metagga.py`, `tests/integration/test_metagga_scf.py`). Single-plane-
  wave τ = ½|k+G|² exactly, τ ≥ τ_W, operator Hermiticity, constant v_τ ≡ c gives
  c·(−½∇²), the generalized-KS gate that the operator equals ∂E_xc/∂ψ* (dE/dλ = 2Re Σf
  ⟨φ|V_τ|ψ⟩ to 1e-5 vs finite difference), the stationary-energy identity
  dE_total/dλ = ∫τ, and τ-flat reduction to PBE/spin-PBE bit-for-bit.
- **r2SCAN.** `core/xc/r2scan.py`, spin-unpolarized `R2SCAN` and collinear
  `SpinR2SCAN`, transcribed from libxc's own Maple source so it matches libxc (hence
  QE's `input_dft='r2scan'`) pointwise, written as a differentiable PyTorch expression
  rather than a libxc binding so v_xc, v_τ, forces, and the learnable-parameter graph
  all follow from autograd. Validated against libxc via pyscf to machine precision
  across α spanning all three branches (`tests/unit/test_r2scan.py`). Unpolarized
  e_xc/vρ/vσ/vτ to 1e-11–1e-14, exchange and correlation each exact standalone, spin-
  polarized energy and per-channel vτ to 1e-15 vs libxc spin=1. The SCF converges (Si,
  11 iterations, same as PBE) and opens the gap 2.44 → 2.67 eV
  (`tests/integration/test_r2scan_scf.py`). Wired into the input path, `xc: r2scan`
  resolves through the registries (`api.py`, `calculator.py`, `inputs.py`), guarded off
  on the non-collinear spinor path (no τ there yet). The libxc pointwise match stands in
  for a `pw.x` fixture; a `si_r2scan_ci` QE fixture is a nice-to-have, not a correctness
  gap.
- **Forces need no τ term (verified).** The worry that the τ operator contributes a
  Pulay-like term absent from the HF forces was unfounded for the plane-wave case. The
  τ operator affects the orbitals and thus the self-consistent density, but at the SCF
  stationary point Hellmann–Feynman holds exactly as for GGA, so the meta-GGA force is
  the same three explicit-R terms (Ewald, local-PP structure factors, NL projectors)
  `forces()` already carries, functional-agnostic. Measured on Si (displaced atom,
  `forces()` vs finite difference), the gap falls 3.7e-3 (20 Ry) → 8.6e-5 (45 Ry),
  vanishing with grid density rather than plateauing, so it is real-space XC-grid
  egg-box (τ is exact in reciprocal space, only ∫e_xc(τ)dr on the grid is not), the
  known SCAN grid sensitivity, just larger than GGA. Meta-GGA forces want a denser
  cutoff than GGA for the same accuracy but are correct as-is. The reusable diagnostic
  (three knobs, SCF tolerance, FD step, grid, to tell a missing analytic term from
  egg-box/lag/FD noise) is in `docs/manual/wisdom.md` under "Meta-GGA forces and
  stress".
- **Stress.** Unlike forces, stress genuinely needs a τ term. Strain scales the plane-
  wave basis (the G-vectors), so ∇ψ, and hence τ = ½Σf|∇ψ|², has an explicit strain
  dependence a GGA (a functional of ρ, which only scales as 1/Ω) lacks.
  `stress._tau_strained` rebuilds τ from the strained (k+G) and the fixed coefficients
  on the autograd graph, so `stress()` carries the meta-GGA term (`_energy_strained`
  passes it to `xc.energy`). Validated on Si (asus). The ε=0 strained energy reproduces
  the SCF total (5e-8); autograd stress equals a finite difference of the strained
  energy (1e-10); the τ term is large (~0.57 eV/Å³, flipping the sign of the Si stress);
  and the fixed-basis stress converges to a re-converged FD (Pulay→0) at high cutoff to
  ~9e-6 eV/Å³. The prediction that stress (not forces) needs the τ term was correct.

### USPP/PAW τ (landed)

The collinear USPP/PAW path now carries the full meta-GGA. The smooth τ̃ = ½Σf|∇ψ̃|²
enters exactly as on the NC path (`core.metagga.tau_b` on the smooth orbitals, the
−½∇·(v_τ∇ψ̃) generalized-KS operator composed onto the batched USPP Hamiltonian in
`scf.uspp_batch.BatchedHS`; v_xc = ∂e/∂ρ|_τ rides the density argument, and there is no
τ compensation charge since the meta-GGA kernel is short-ranged). The one-center
augmentation is `τ¹ = ½ Σ_ij ρ_ij ∇φ_i·∇φ_j` on the PAW radial×angular grid, built for
the AE and PS partial waves and fed into the on-site XC quadrature
(`scf.paw_onsite.OneCenter._tau_rad`); the AE−PS difference is the τ correction, and the
on-site potential ddd_ij = ∂E_1c/∂ρ_ij picks up the τ term for free through the existing
autograd-through-`e1c_t`. Forces work with no dedicated smooth-τ term (the smooth τ̃ is a
functional of the position-free plane-wave coefficients, so it is a detached constant at
fixed coefficients; the one-center τ force rides ddd). We follow the NC convention of
building τ from valence orbitals only (no core-τ; the frozen core enters XC through
ρ_core), and bare (non-PAW) ultrasoft is rejected up front — it has no one-center sphere
to carry the AE/PS τ correction, matching QE. Stress and the noncollinear/SOC spinor USPP
path stay follow-ups. Validated: one-center ddd == finite difference, an r2SCAN PAW SCF
converges and is genuinely τ-dependent, and analytic forces == finite difference of the
SCF energy.

### What remains, in build order

(1) USPP/PAW meta-GGA stress (the strained smooth τ̃, mirroring `postscf.stress._tau_strained`)
and a QE r2SCAN-PAW reference fixture; (2) meta-GGA on the non-collinear/SOC spinor USPP
path (the one-center 2×2 τ-density-matrix augmentation); (3) a learnable r2SCAN-form
functional (α, and the enhancement-factor parameters as trained tensors) and the
`train_xc_paw` recovery test at meta-GGA level, now directly reachable since r2SCAN is
already a differentiable autograd expression. Validating a learnable meta-GGA against QE
`input_dft='scan'` or r2SCAN at pinned settings to milli-eV, then repeating
`train_xc_paw`, is the payoff.

## Differentiable pseudopotential correction

**Status: PROBED 2026-07-29 on Si. The machinery is validated end to end. An EOS-only
loss is rank-deficient, so the idea is shelved pending a multi-observable loss. Archive
branch `research/delta-learned-pseudo-probe`.**

The periodic-table Δ-gauge (`benchmarks/delta_gauge`) surfaced a concrete target. The
PseudoDojo standard UPF for Cu reproduces neither all-electron nor its own psp8 (B0 167
vs 141, Δ 7.9 meV/atom), and gradwave matches QE on that same UPF to 0.08 meV, so the
error is pseudization rather than implementation. Most transition metals carry a
smaller version of the same "stiff-metal" Δ floor. Worth using gradwave's
differentiability to address the error rather than only measure it.

Treat a small correction to the pseudopotential as a trained parameter and descend it
through the self-consistent solution against all-electron reference data, the way the
learned hybrid descends α and ω. Add a differentiable δv(r) (a few-parameter radial
form, or a correction to the local channel / a KB coefficient) and minimize a multi-
property loss, the EOS curve vs WIEN2k, or (better, once the all-electron anchor exists)
valence eigenvalues and logarithmic derivatives at the reference energies. The gradient
dLoss/dθ_pp flows through the same stationary-energy and Sternheimer machinery already
used for dE/dα and the density adjoint. The pseudopotential enters the energy through
the local potential and the nonlocal projectors, both already τ-differentiable for
forces, so the parameter graph is mostly in place. The result would be a corrected Cu
and, if it generalizes, a per-element learned correction that pulls the stiff-metal Δ
floor down. Care is needed. Keep the correction small and norm-conserving, and validate
against overfitting the EOS at the expense of transferability (band structure, a second
crystal structure).

**Probed on Si (2026-07-29).** A smooth four-parameter reciprocal-space correction to the
local form factor, `dv_loc(G) = Σ_k θ_k (G/μ_k)² exp(−(G/μ_k)²)` with μ_k = 1..4 Å⁻¹,
threaded into `System.vloc_atom` the way the alchemical local-table blend threads λ. The
credentials are exact. θ=0 reproduces the plain SCF bit-for-bit, and the Hellmann-Feynman
dE/dθ matches a full-reconvergence central finite difference to 3.2e-10 max relative error,
four decades below the repo's ~1e-5 FD floor, because the energy is exactly linear in θ at
frozen density.

The blocker is identifiability, not differentiability. A synthetic recovery oracle (perturb
Si by a known θ*, fit θ back from zero) drives the EOS shape residual from 0.935 to 0.077
meV/atom while |θ − θ*| stays at 0.364 and the fitted parameters collapse to a common
value. The SVD of the 5×4 Hellmann-Feynman EOS Jacobian explains it. The singular values
are 1.7e-2, 2.8e-4, 4.2e-6, 1.5e-7 (condition 1.1e5), so a 5-volume EOS window determines
one, at most two, of the four directions, and the recovery error sits in the two weakest.
This is rank deficiency of the loss, not an optimizer failure.

One real step against the WIEN2k Si EOS pulled Δ(Si) from 0.258 to 0.057 meV/atom at the
probe basis in 10 Adam steps, and the off-training forces changed by at most 2e-4 eV/Å.
Honesty caveat, the basis error at the probe settings is 0.149 meV/atom, comparable to the
Δ change, so part of what the correction learned is basis truncation rather than
pseudization. A real run must sit at the converged basis.

Next scope, in order. Add valence eigenvalues and displaced-cell forces to the EOS loss and
rerun the recovery oracle. Success there is the true go signal. Then Cu at the converged
basis along the leading Jacobian directions, or one Gauss-Newton step on the linear map
instead of Adam. The D_ij (KB coefficient) rung comes last, since the nonlocal channel has
the same linear-in-parameter HF structure but multiplies the null-space question.

## Alchemical composition channel and the band-gap design gradient

**Status: landed (2026-08-15). Differentiable substitution + the relaxed d(E_gap)/dλ
by composition DFPT, validated against finite difference; per-site vector λ, nspin=2,
and metals are the open tails.**

`scf/alchemical.py`. A per-atom weight λ blends two endpoint pseudopotentials across
every ionic piece — charge, local potential, KB projectors (both endpoints' columns
carried, the inactive one zero-coupled), and the NLCC core density — so composition is
a differentiable coordinate. `setup_alchemical_system` blends a whole cell A→B;
`setup_alchemical_substitution` transmutes a chosen subset of sites in a real
multi-species cell (perovskite/defect/doping). Not the VCA: the endpoints are real and
λ=0/1 reproduce the pure cells to SCF noise (verified).

Two gradients. `alchemical_energy_gradient` gives dE/dλ for free by Hellmann-Feynman —
at the SCF minimum the density is variational, so the response drops by the envelope
theorem and only the bare ionic terms survive (validated to ~3e-10 rel vs FD, the energy
being exactly linear in λ at frozen density). `alchemical_gap_gradient` gives the
*relaxed* d(E_gap)/dλ, the physically-meaningful design quantity: an eigenvalue is NOT
variational, so dε_i/dλ = ⟨ψ_i|dV_KS/dλ|ψ_i⟩ carries the self-consistent density
response. The bare perturbation dV_ion/dλ is a **local + nonlocal operator** (the local
form-factor change, the explicit NLCC core-XC term f_xc·∂ρ_core/∂λ, and the KB ∂D/∂λ
through the fixed projectors), so the response needs a Sternheimer solve with an operator
RHS — `_chi0_operator` adds the nonlocal piece to the local-field Sternheimer of
`scf/implicit.py` — plus a forward Dyson dressing (1−χ₀K_Hxc)⁻¹. Validated on two systems
(SiC→C, CsPbI3→CsPbCl3): analytic relaxed dε and d(E_gap)/dλ match a central FD of
re-converged eigenvalues/gaps to ~1e-4 eV (FD-limited). The **frozen (sudden) estimate is
off by >10×** on the perovskite (+8.9 vs +0.77 eV) and is ~0 while the relaxed is +2.4 eV
on SiC→C — the density response is the entire gradient, which is why the nonlocal build
was necessary rather than the cheap frozen matrix element.

### The aliovalent ladder (beyond isovalent)

The gap DFPT above is isovalent (ΔN=0, cell neutral, integer occupations, clean gap).
"Beyond isovalent" is a ladder of increasing difficulty, being built out as rungs:

**Rung 1 — per-site vector λ + charge-conserving co-substitution (landed 2026-08-15,
`alchemical_gap_gradient_per_site`).** The bare perturbation is parameterized by a per-atom
rate mask, so a one-hot mask gives a single site's ∂V_ion/∂λ_k and the per-site Jacobian
∂(E_gap)/∂λ_k falls out (one DFPT solve per site, shared edges). Because the perturbation is
linear and additive, Σ_k ∂/∂λ_k equals the coupled scalar gradient exactly (a built-in
consistency check). This enables the honest "beyond single-species isovalent" case: a
charge-conserving CO-substitution (a donor+acceptor pair whose TOTAL valence is invariant),
where each component is aliovalent but N stays fixed and insulating, so the coupled λ DFPT
applies directly. Validated on **MgO→NaCl** (Mg 10 + O 6 = 16 = Na 9 + Cl 7): coupled
d(E_gap)/dλ analytic +6.822 vs FD +6.821 (1 meV), Σ per-site = coupled exactly, and the
**frozen estimate has the WRONG SIGN** (−0.17 eV) — first-order APDFT band-gap alchemy would
mispredict the direction, the relaxed DFPT gets it right. Isovalent per-site (CsPbI3 3×X→Cl)
matches single-site FD to ~0.1 meV on non-degenerate-picking sites; the degenerate cubic-VBM
gauge dependence (rung-1 caveat) shifts one equivalent site by ~4 meV — the sum and coupled
are unaffected.

**Rung 2 — single-site aliovalent + Janak (landed 2026-08-15).** N(λ)=ΣZ(λ) follows the ionic
charge, fractional N mid-path → metallic (smearing), the gap dissolves, but the ENERGY
gradient stays clean via the Janak term μ·dN/dλ. `alchemical_energy_gradient` now dispatches
on the spec kind: `_substitution_energy_gradient` rebuilds the per-atom ionic terms from the
`SubstitutionSpec` (which now stashes `z_base`/`z_target`) and carries the Janak term (the
binary path already had it). Validated on **Si→As** (ΔZ=+1, one site in diamond Si, N=8.5,
metallic): analytic dF/dλ = −64.766 vs FD −64.766 (0.2 meV), the Janak share being +7.42 eV
(11.5 %); the **Si→C** isovalent control (ΔZ=0) has the Janak term vanish and still matches FD.
Per-state eigenvalue derivatives in the metallic case need the metallic χ₀ path
(`_chi0_channel_metal`, exists) — a follow-up. Grounding: von Lilienfeld APDFT (energy alchemy
is exact at first order by the envelope theorem; eigenvalue alchemy is not — the
frozen-vs-relaxed gaps above are the direct evidence).

**Rung 3 — charged-cell fixed-N (landed 2026-08-15).** Hold N fixed and transmute one site
aliovalently → the cell carries a net charge q(λ)=ΣZ(λ)−N. Built in three pieces:
(a) **net-charge SCF** — `setup_system(..., tot_charge=q)` sets `n_electrons=ΣZ−q` with the
implicit compensating −q jellium background (the G=0 electrostatics already carry it; the
occupation filling and density normalization target the reduced count). Empirically confirmed
consistent: a charged Na⁺ cell's energy follows the Makov-Payne law E(L)=E∞−q²α_M e²/(2L) with
the monopole coefficient matching the true simple-cubic α_M=2.837 to ~4 % (SCF noise) — gradwave
behaves like a standard PW code (QE/VASP: the periodic charged cell is the jellium cell, the
finite-size term is post-hoc), NOT a spurious η-monopole. (b) **Finite-size correction** —
`postscf/charged.py`: `makov_payne_correction(q, cell, ε)` computes the monopole term from the
cell's own Ewald-sum Madelung energy (exact α_M for any lattice, cubic reproduced to 6 digits),
and applying it collapses the box-size spread of the charged cell. Refinements (FNV / Lany-Zunger
for localized charge, the L⁻³ quadrupole term) are follow-ups — Kumagai & Oba, PRB 86, 045112.
(c) **Charged-cell alchemical gradient** — `setup_alchemical_substitution(..., n_electrons=N)`
holds N fixed, so an aliovalent transmutation charges the cell; the fixed-N energy gradient is
the bare Hellmann-Feynman ionic derivative (`grand_canonical=True`, no Janak since dN/dλ=0),
validated on Si→As at fixed N=8 (q=+0.5 at λ=0.5) vs FD to <5 meV.

**Rung 4 — grand-canonical fixed-μ (landed 2026-08-15).** `alchemical_energy_gradient(...,
grand_canonical=True)` returns the grand-potential derivative dΩ/dλ = dF/dλ − μ·dN/dλ. At
fixed chemical potential the Janak μ·dN/dλ bookkeeping cancels the −μN Legendre term, so the
grand-canonical alchemical gradient reduces to the **bare Hellmann-Feynman ionic derivative**
— the electron reservoir absorbs the composition-driven electron-count change. Validated on
Si→As (N=8.5, metallic): dΩ/dλ matches a central FD of Ω = F − μN (μ fixed at the reference
Fermi level) to <5 meV, and the Legendre relation dF = dΩ + μ·dN holds to 1e-6. This is the
standard *indirect* route to grand-canonical properties (Legendre transform of the canonical
fixed-N path — Sundararaman et al., JCP 146, 114104, 2017): a grand-canonical surface carries
no more information than the canonical one they are linked to. A *direct* constant-µ SCF
already exists via `target_mu`, but it requires `boundary='open_z_metal'` (an ESM slab with an
electron reservoir), so composing it with an alchemical substitution on a slab is the
electrochemistry follow-up.

Inverse design (capstone, validated 2026-08-15). `examples/perovskite_inverse_design.py`
closes the loop: two checks on the CsPbI3→CsPbCl3 path. (1) Path integration — ∫₀¹ dGap/dλ dλ
= +1.393 eV (trapezoid over 6 DFPT gradients) matches the endpoint ΔGap = +1.383 eV to 10 meV,
so the local gradient is a true global derivative. (2) A Newton loop on λ using dGap/dλ reaches
a 1.7 eV target in 3 steps (λ*=0.468), and a fresh direct SCF at λ* gives 1.7003 eV — a 0.3 meV
miss, gradient-driven composition design verified against ground truth. The scalar-λ path keeps
the cubic symmetry so the degenerate VBM shifts uniformly and the gradient stays exact; the
degenerate-edge caveat only matters for symmetry-breaking per-site design.

Arbitrary observable (landed 2026-08-15). `alchemical_observable_gradient(res, xc, observable)`
gives the relaxed dO/dλ for ANY differentiable density-functional observable O[ρ] — a multipole
moment, the charge in a region, a density overlap — as ⟨δO/δρ|dρ/dλ⟩, with δO/δρ by autograd
and dρ/dλ the composition DFPT response. This turns the gap-specific tool into general
composition→property design. Validated on SiC→C: the electron-count response dN/dλ = −4e-17 (the
isovalent response conserves electrons, an exact internal check) and the density overlap
d(∫ρ²)/dλ = −0.2111 matches FD to 7e-4. The band gap is the eigenvalue-observable special case
(a function of {ε_i} rather than of ρ); response-function observables (ε∞, Born charges) are
second-order (a mixed composition+field DFPT) and remain open.

Other open tails. nspin=2 and metals for the gap DFPT: the χ₀/K_Hxc kernels already have
spin-resolved and partial-occupation paths in `scf/implicit.py`, so these are threading, not
new physics. Degenerate band edges want the degenerate-subspace first-order form (the current
single-state expectation assumes a non-degenerate edge).

Response-function tier — composition derivative of ε∞ / Born charges (landed 2026-08-15,
FD; analytic 2n+1 open). A RESPONSE function is itself a second energy derivative (ε∞ ∝
d²E/dℰ²), so dε∞/dλ is a mixed THIRD-order derivative d³E/(dλ dℰ²). Two parts landed:
(a) an infrastructure fix — the E-field DFPT (`postscf/dielectric.py`) now runs on the
alchemical blended-projector System: `_shifted_projectors` rebuilds the STACKED [base;target]
KB columns at shifted k (the λ-weighting rides `bk.dij_full`), which it previously could not
(it rebuilt only the base per-species set, half the columns). The substitution spec stashes
the target UPFs for this. (b) `postscf/alchemical_response.alchemical_dielectric_gradient`
gives dε∞/dλ and dZ*/dλ by a central FD of `dielectric_born` along the composition path;
validated h-convergent on SiC→C (dε∞_iso/dλ = +14.77 at h=0.04 and 0.02, differing 0.014).
FD is the reliable route for a third-order quantity. The fully-ANALYTIC mixed derivative is a
2n+1 build (dε∞/dλ from first-order responses dρ/dℰ, dρ/dλ + the bare-λ RHS) whose one
genuinely new/expensive piece is the third functional derivative of E_xc (the ∂_λK_Hxc term,
a triple-backward HVP) plus the periodic-field position-operator assembly; the FD here is the
ground truth it would validate against. Binary (whole-cell) alchemical dielectric and the
metals/nspin=2 E-field paths are further follow-ups. No prior Raman/mixed-response machinery
existed — this is the first mixed-response derivative in the code.

Framing / citations: this whole line is the self-consistent upgrade to first-order band-gap
alchemy — APDFT (von Lilienfeld & Tuckerman, arXiv:1809.01647) and the AlxGa1-xAs direct-gap
design of Chang & von Lilienfeld (Phys. Rev. Materials 2, 073802, 2018) use the first-order
(frozen) alchemical derivative, which the frozen-vs-FD numbers here show is inadequate for
gaps (energy alchemy is fine at first order; eigenvalue alchemy needs the density response).

# Magnetism and spin-orbit coupling

## Differentiable spintronics: spin Hamiltonians, DMI, and inverse design

**Status: open; the pseudopotential blocker is cleared. Constrained-moment torque is
autograd-exact today.**

The constrained non-collinear framework (`postscf/moment_config.py`,
`scf/moment_penalty.py`) is differentiable where constrained-DFT in QE/VASP/FLEUR is
not. The per-atom torque `dW/de_I` is autograd-exact (validated to a finite difference
at ratio 1.000), not finite-differenced, and the magnitude-robust `vector` penalty
holds an arbitrary non-collinear texture at fixed moment instead of letting it
collapse. Every item below is cheap because of AD and awkward otherwise, roughly in
impact order.

- **Extract the spin Hamiltonian (J, D, K) by differentiating the torque.** The whole
  spintronics modeling stack (atomistic spin dynamics VAMPIRE/Spirit, micromagnetics
  mumax) runs on `H = -Σ J_ij S_i·S_j - Σ D_ij·(S_i×S_j) - Σ K_i (S_i·n)²`, and DFT's
  job is to parametrize it. Conventionally that is finite-difference energy mapping
  (fragile) or a separate Green's-function (LKAG) machinery. Here the Heisenberg J_ij,
  the Dzyaloshinskii–Moriya vectors D_ij, and the anisotropy K are derivatives of the
  torque, `d(dW/de_I)/de_J` being the exchange/DMI coupling tensor between sites I and
  J, so a second autograd pass over the torque gives the couplings directly, cross-terms
  included. DMI is the hardest of the three, needing SOC and non-collinearity (both
  present), setting skyrmion chirality and size, and notoriously noisy to finite-
  difference. A differentiable, magnitude-conserving DMI extractor is methodologically
  novel and connects small-cell plane-wave DFT to device-scale spin dynamics. The code
  cannot simulate a 50 nm skyrmion but can hand a clean spin Hamiltonian to a code that
  can. Second derivatives of the penalty scalar are within reach of the autograd path;
  the work is wiring the site-pair Hessian and validating J against a known magnet.
- **Chiral textures and the micromagnetic DMI from spin-spiral asymmetry.** The Fe
  spin-spiral demo (`examples/fe_spin_spiral.py`) traces E(theta) with inversion
  symmetry, so E(+q) = E(−q). Break inversion (an interface Co/Pt, Fe/Ir, or a B20 bulk
  MnSi, FeGe) with SOC on, and E(+q) ≠ E(−q), the chiral splitting whose q→0 slope is
  the micromagnetic DMI constant. A measured E(+q) − E(−q) is a direct extension of the
  committed spiral sweep (add SOC, rotate in the DMI-active plane, sweep signed q).
- **Inverse design.** Energy is differentiable w.r.t. atomic positions, moment
  directions, and in principle strain and composition, so a gradient search can
  optimize toward a target magnetic property, the strain that maximizes DMI, the
  composition that flips the easy axis. No finite-difference code can do this. Higher
  risk, since it needs gradients through the SCF and not just the envelope torque, but
  it is the fullest use of gradwave's differentiability applied to magnetism.
- **Demonstration vehicle, 2D magnets.** CrI3, Fe3GeTe2, CrSBr maximize impact per
  core-hour, small unit cells (tractable in plane waves), large anisotropy (clears the
  ~0.2 µeV rotation-invariance floor cubic Fe sits on), and open questions about their
  DMI, topological magnons, and stacking-dependent order. A J/D/K extraction on CrI3
  with DMI as the main target is the most tractable and novel option. The MAE map (next
  section) is the better visual deliverable and the natural second step once the SOC
  force-theorem path is in.

**The pseudopotential blocker is cleared.** DMI and single-ion anisotropy K need spin-
orbit coupling on a magnetic atom. The fully-relativistic magnetic pseudos (and iodine)
are in the fixtures, `Fe/Co/Ni/Cr/Pt_ONCV_PBE_FR-1.0.upf` and `I_ONCV_PBE_FR-1.1.upf`,
pulled from the SG15 ONCV set (quantum-simulation.org, the same source as the scalar
`_ONCV_PBE` pseudos). A magnetic FR pseudo correctly triggers the SOC path
(`system.is_fr` → j-resolved spinor projectors). The natural order is L1_0 FePt MAE
(`Fe_FR` + `Pt_FR`, ~2-3 meV/f.u., far above the 0.2 µeV floor) → hcp Co (~65 µeV) →
CrI3 (`Cr_FR` + `I_FR`) for the full J/K/DMI story. The Heisenberg-J machinery
(`postscf/spin_exchange.py`) and `characterize_magnetism` already work without SOC.

### Curie temperature: DFT J → Heisenberg Monte Carlo (landed 2026-08-16)

The spin-Hamiltonian edge, closed to an experimental number. `postscf/heisenberg_mc.py`
is a classical (continuous 3-vector) Heisenberg Metropolis MC on an arbitrary nn lattice
— greedy graph-colouring generalises the bipartite checkerboard, so it handles bcc *and*
fcc/hcp (the frustrated non-bipartite cases). It fills a real gap: the repo's only lattice
MC (`lattice_mc`) is Ising, which cannot give a magnetic T_c, and the only prior estimator
was mean field (`characterize_magnetism.curie_temperature_mfa`), which overshoots ~30 %.
Validated against the textbook classical coefficients k_B T_c/K = 2.054 (bcc) and 3.18
(fcc). `examples/fe_curie_temperature.py` runs the full pipeline for bcc Fe — relax
(spin-polarised EOS) → extract J (autograd-exact constrained-moment torque, no SOC) →
mean-field + MC T_c — and reproduces experiment: at the experimental lattice, M = 2.22 μB
(exp 2.22), J_01 = 179 meV, MFA T_c = 1386 K (33 % high), **MC T_c = 1125 K (8 % from the
1043 K experiment)** — the transverse-fluctuation correction is the whole difference. PBE
overbinds Fe (relaxed a = 2.75 Å vs 2.87), inflating T_c to 1468 K, so the experimental
lattice is the appropriate geometry (Pajda convention). Open follow-ups: multi-shell J
from a supercell (removes the nn-only approximation), fcc Co/Ni (the general MC path is in
place; Co running), the RPA/Tyablikov estimator, and quantum (S=1) corrections. This is the
one place small-cell DFT genuinely reaches an experimental observable others reach only
with much larger machinery.

## Magnetocrystalline anisotropy (MAE maps) and per-atom spin torques

**Status: force-theorem evaluator and per-direction magnetic-IBZ folding landed
(2026-07-19); band-resolved anisotropy is the one open tail.**

The constrained-moment work (`postscf/moment_config.py`) already produces half of this
for free. `constrained_moment_scf` returns a per-atom transverse torque `-dW/de_I`,
validated to a finite difference at ratio 1.000, which is the magnetic force-theorem
spin torque. Without SOC it is the inter-atomic exchange torque (what drives the config
search and sets a spin spiral's stiffness); with a fully-relativistic pseudo the same
torque picks up the on-site anisotropy term. Individual spin torques are what the module
returns today. The missing half was the global anisotropy, MAE maps E(theta, phi) over
the magnetization sphere.

The SOC path exists, `core/spinor_proj.py` builds the j = l ± ½ projectors and
`SpinorHamiltonian` accepts them (`q`, `dij_so`), and `NCResult.energies.free_energy`
gives a total energy per direction, so MAE = E(n1) − E(n2) is a direction sweep. The
efficient route is the torque method, one SOC evaluation per direction yielding
−dE/dn, integrated over the sphere.

**Proof-of-physics (2026-07-18).** The two-point MAE of L1_0 FePt by full SOC SCF is
correct, +2.55 meV/cell (+1.28 meV/atom), easy axis [001], in the literature band, both
orientations converged to ~1e-11 eV (`examples/fept_mae.py`, 144 k, asus CPU). The 48-k
mesh gives the wrong easy axis (−1.39 meV/cell toward [100]), the textbook sampling-
error sign flip measured directly. Three things were in the way, and the third was the
only real code.

- **Precision floor.** `test_noncollinear.py` pins rotation-invariance (MAE ≡ 0
  without SOC) to ~0.2 µeV, the noise floor. Cubic Fe's MAE is ~1 µeV/atom, right on
  it. Start on a high-anisotropy case, L1_0 FePt (~1 meV/atom), hcp Co (~65 µeV), or a
  uniaxial 2D magnet.
- **k-convergence.** Metal MAE converges painfully slowly in k (thousands of points,
  or fine-smearing / Fermi-surface-aware tricks). A cost problem, the reason the force
  theorem matters.
- **No force-theorem path for SOC (now built).** The cheap recipe is to converge the
  density scalar-relativistically once, add SOC non-self-consistently, and take
  occupied-band-energy differences per direction. The frozen-potential band-solve
  infrastructure already existed (`postscf/uspp_bands.py`, `core/gamma.py`, the
  one-shot solve in `postscf/hubbard_u.py`), it just was not connected to the SOC
  Hamiltonian plus a directional band sum.

**Force-theorem evaluator (2026-07-19).** `postscf/mae.py` `force_theorem_mae`. One
converged SOC SCF along a reference axis, then per direction a rigid rotation of
(m⃗, B_xc), exact for the locally-collinear XC since B_xc co-rotates with m⃗ and v_xc
depends only on (ρ, |m⃗|), one frozen-potential spinor diagonalization seeded with the
SU(2)-rotated reference spinors, and the occupied band free energy at that direction's
Fermi level. The anisotropy enters solely through the lattice-fixed SOC projectors, so
a scalar-relativistic system gives an exactly direction-independent band sum, the
correctness gate, measured invariant to <1e-6 eV over four directions
(`tests/integration/test_mae_force_theorem.py`). At a shared small FePt mesh the FT
difference tracks the two-SCF difference within the 30% gate. At scale
(`examples/fept_mae_force_theorem.py`, 144 k full mesh, 70 Ry), FT MAE [100]−[001] =
+2.673 meV/cell vs the self-consistent +2.552 (4.7%), [110] +2.713 (in-plane spread
0.04 meV), and the 45°-tilted [101] at +1.340 ≈ half of [100], the uniaxial K₁ sin²θ
form from four one-shot solves. Each direction costs ~11 min against ~84 min for a full
SCF (7.7×), which makes E(θ, φ) maps affordable. The global-spin-axis control and the
sweep/integrate layer are subsumed, the directions list is the sweep.

**Per-direction magnetic-IBZ folding (2026-07-19).** `force_theorem_mae(..., magmoms=)`
folds each one-shot solve into its own direction's Shubnikov IBZ. The per-atom
reference moments rotate with the direction, the magnetic group of the rotated texture
folds the mesh (`reduce_mesh_magnetic`), and because every folded representative is a
point of the full mesh the solve runs on a subset of the stored reference spheres with
folded weights, the SU(2)-rotated seeds gathering straight from the reference
coefficients. The reference SCF still needs the full mesh, only the evaluations fold.
The fold is exact for the collinear part of the frozen magnetization (ρ and |m⃗| carry
the crystal symmetry, the uniform rotated direction transforms as an axial vector); the
SOC-induced transverse textures formally break it, but the measured folded-vs-full
residual on small FePt is ~4e-12 eV, far below the force-theorem error (gate 1e-6). On
the 6×6×4 mesh the folds are [001]→30/144, [100]→48, generic (010)-plane tilt→56,
compounding with the 7.7× per-direction saving. `examples/fept_mae_map.py` uses this for
an E(θ) scan [001]→[100] with a K₁sin²θ + K₂sin⁴θ fit. Measured at scale (asus CPU,
2026-07-19, batched spinor density path), 7 directions in 889 s (~2.1 min each, against
~11 min unfolded pre-batching), K₁ = +2.6965 meV/cell, K₂ = −0.0358, max fit residual
0.0015 meV (FePt is uniaxial to 1%). The 45° point reproduces the unfolded full-mesh
+1.3398 meV to all printed digits, and [100]−[001] = +2.660 vs the self-consistent
+2.552 (4.2%). The whole 7-point map costs about an hour of CPU, one 2450 s reference
SCF plus 15 min of folded solves.

**Band-resolved anisotropy (open).** The MAE is a single number; the diagnostic that
explains it decomposes ΔF into per-k, per-band contributions by differencing the
spectra of two directions state by state (each at its own Fermi level) before summing.
Away from ε_F the SOC shifts cancel in the difference, so the net anisotropy lives in
near-degenerate band pairs within ~ξ_SOC of the Fermi level that split differently per
direction, in FePt the Pt-5d avoided crossings, which is also why coarse meshes flip
the sign. Everything needed is persisted, `MAEResult.save` keeps the full (nk, nb)
spectra and Fermi levels per direction. What to build. (1) A ΔE(k) map over the mesh
plus a band-index decomposition, occupation-weighted with the entropy term explicit.
(2) An unfolding step, since per-direction folds put two directions on different
k-subsets, either re-run the pair unfolded (minutes now) or expand folded spectra back
through the orbit maps the magnetic-symmetry machinery already computes. Caveats for
the docs, per-k contributions are gauge-sensitive (only the k-sum is physical), and
band indices must be matched through crossings if the decomposition is followed along
θ. Payoff is hotspot maps on the Fermi surface and a principled handle on how
alloying/strain moves the MAE.

# Post-SCF and spectroscopy

## Phonon band structures

**Status: supercell route landed (2026-07-24, #61); arbitrary-q DFPT, LO-TO, and
harmonic thermodynamics open.** `gamma_hessian` is Γ-only; the extension to finite q
gives dispersions.

`postscf/phonons_supercell.py` builds a supercell, displaces only the primitive
home-cell atoms (the Born–von-Kármán translational reduction, so the SCF count is
6·N_prim regardless of supercell size), symmetrizes the force constants and applies the
acoustic sum rule, then Fourier-folds `D(q)=Σ_R Φ/√(MμMν)·e^{iq·R}` onto the ASE
bandpath for the dispersion and onto an MP q-mesh for the phonon DOS. It reuses the
electronic bands-path builder and the PDOS broadener. Norm-conserving, nspin=1 (the
forces path). The 1×1×1 fold reproduces the analytic Γ route exactly (Si optical
521.46 cm⁻¹, acoustic within ~1 cm⁻¹ of zero). Wired through inputs (`PhononParams`),
the api dispatch, the output report, and `gradwave plot --kind phonons`.

Still open. The analytic force response at q with a q-dependent perturbation
(arbitrary-q DFPT), since the supercell route only reaches the commensurate q of its
supercell. For polar insulators the nonanalytic LO-TO term at q→Γ, which needs Born
charges and epsilon-infinity, both already in `dielectric.py`, so the polar correction
is reachable. And the harmonic thermodynamics from the q-mesh DOS (free energy,
entropy, heat capacity), which pairs with the EOS work for a full thermal equation of
state.

## RAIRS and a slab dipole moment

**Status: open. Vibrational frequencies and insulator/molecule IR intensities exist;
the metal-surface case is the gap.** We do vibrational frequencies
(`postscf/phonons.py`, validated against QE `ph.x` to 0.003% on Si) and IR intensities
for insulators and molecules (Born charges and epsilon-infinity from
`postscf/dielectric.py`). Metals are the gap.

`dielectric_born` refuses anything but an nspin=1 insulator, because it splits valence
from conduction with a projector `P_c` and a `(H − eps_v)` solve that goes singular at
a metal Fermi level. A bulk metal also has no IR-active optical phonon in the insulator
sense and no static Born charge, so "bulk metal IR" is not a real target.

The real target is RAIRS, the reflection-absorption IR of an adsorbate on a metal
surface (CO on Pt is the textbook case). The metal surface selection rule says only the
dynamic dipole perpendicular to the surface couples, and the slab surface-normal dipole
`mu_z` is well defined despite the metal because the vacuum gap gives a clean
reference. So sidestep the singular DFPT and finite-difference.

- New piece, a slab dipole-moment function `mu_z = integral z rho_tot(r) + ionic`,
  with the standard slab caveat that it needs a vacuum gap and a dipole correction so
  the two surfaces do not talk through the cell. The only genuinely new physics.
  gradwave has rho on the grid and the ion charges, so it is modest.
- Reuse, finite-difference the dynamic dipole. Displace each adsorbate atom by ±δ,
  compute `mu_z`, difference to `Ztilde_{s,zbeta} = d mu_z / d tau`. For CO that is 2
  atoms × 3 directions × 2 = 12 SCFs. Contract with the CO-projected Hessian modes from
  `gamma_hessian` and keep the z component.

Cost is finite-difference, roughly 12 extra slab SCFs on top of the Hessian, no DFPT.
It works on the metal because it uses `mu_z`, not `Z*`. Estimate 1 to 2 days, almost
all of it in the slab dipole routine and its validation.

Raman on a metal stays hard. The Raman tensor is d alpha / d Q and alpha itself is
ill-defined for a metal, and surface-enhanced Raman is dominated by electromagnetic
field enhancement rather than a clean DFT observable. Leave it.

If someone wants the metal's own far-IR response, that is a different deliverable, the
optical conductivity sigma(omega) (Drude plus interband). It extends the existing
E-field Sternheimer to finite frequency and adds the intraband Fermi-surface term
`dielectric.py` omits. Roughly a week, and it produces optical conductivity, not a
vibrational spectrum.

## Little and orbit groups for DFPT under symmetry-breaking perturbations

**Status: open.** The response calculations either use the full crystal symmetry or
drop to time-reversal only. A perturbation lowers the symmetry to its little group, and
we should reduce k-sampling and irreducible displacements by that residual group
instead of discarding symmetry outright.

- For a Γ-phonon column, the displaced-atom pattern has a little group, the site
  symmetry intersected with the displacement direction. Only symmetry-inequivalent
  columns need computing, the rest reconstructed by the group action (the
  `HessianSymmetry` reconstruction already does the reconstruction half for the full
  group; this generalizes it to the perturbation little group).
- For the E-field response, the little group is the subgroup that leaves the field
  vector invariant, and k reduces to that subgroup's IBZ rather than to time-reversal
  only.

Payoff is direct k-point and displacement-column savings on exactly the expensive
response runs, which is what QE `ph.x` does with its `modes_of_q` and small-group
machinery. The building blocks (`find_spacegroup`, `reduce_mesh`) exist; the work is
computing the little group of a given perturbation and threading it into the DFPT
drivers.

# Coverage, correctness, and error budget

## Full nspin=2 and PAW coverage for every feature

**Status: collinear nspin=2 post-SCF landed (2026-07-24); the residual gates are real
physics gaps, not spin threading.** Coverage is uneven across the postscf features.
Several are nspin=1 or NC only. The discretization-error force path is NC (nspin=1 or 2,
no USPP/PAW), and the noncollinear and SOC PDOS paths have their own constraints.

The nspin=2 gates came off the main post-SCF properties, since the per-spin machinery
already existed and only the spin loop was missing. Band structures on the norm-
conserving and USPP/PAW paths and atomic forces (#45), the Nielsen-Martin stress (#58),
KPM-DOS and ELF, and the dielectric/Born E-field DFPT (`_dielectric_born_spin`, #65),
each pinned by a nonmagnetic-limit or strain-FD self-oracle. Fixed-spin-moment SCF (a
fixed `tot_magnetization = N↑ − N↓`) landed alongside (#45). What remains gated is real
physics, the DFT+U and fully-relativistic stress, the metals (partial-occupation)
dielectric path, and the USPP / noncollinear meta-GGA NLCC force.

Make an explicit matrix of feature × {NC, USPP/PAW} × {nspin=1, 2} and close the gaps.
Most of the per-channel machinery exists, so the work is threading the spin index and
the S-metric or augmentation consistently, plus tests at each new cell. Unglamorous,
but it is what makes the code trustworthy on magnetic surfaces and spin-polarized
adsorbates. The SCF core is already even, the batched USPP/PAW eigensolve is validated
at nspin=2 (O2 triplet, batched vs per-k to 7e-12 eV, 21 iterations to QE's 20), so the
gaps are in the postscf property layer, not the solver.

## Error estimation: the rest of the budget, and what it can and cannot reach

**Status: smearing, k-point, and cutoff terms landed (2026-07-18); SCF-convergence
headline landed via trajectory extrapolation, the exact response form still open;
model-term tooling open.**

The discretization estimate (`postscf/discretization_error.py`) is one term in a larger
error budget, and the cleanest one because a plane-wave cutoff is a variational
truncation, so a converged truth exists at infinite basis, the energy is stationary
there, the error is second order, and a single cheap perturbative pass reaches it.
Density, energy, force (NC nspin=1/2), and now per-band eigenvalue and band-gap errors
all fall out of the same complement correction. This section records what the other
terms are, which share that structure, and whether adding them up ever gives the true
error. It never does, and the reason bounds the whole program.

Split the budget into a numerical part and a model part.

    E_computed - E_reality = (E_computed - E_KS_converged) + (E_KS_converged - E_reality)
                              \______ numerical ______/       \____ model (XC) ____/

`E_KS_converged` is the exact-basis, dense-k, fully self-consistent Kohn-Sham energy for
the chosen functional. The numerical term is the sum of the convergence errors (cutoff,
k-points, SCF, smearing, density grid, cell size). The model term is the XC functional
error. Only the first is reachable from inside a calculation.

The numerical terms are trackable, roughly additive, and individually cheap.

- SCF convergence error. Stopping at finite `rhotol` leaves a density residual
  `drho = rho_out - rho_in`. Because the energy is stationary at the fixed point, the
  error is `~ (1/2) <drho | K_Hxc + chi0 | drho>`, second order in the residual (why
  the energy converges as the square of the density). The kernel is exactly the
  operators `scf/implicit.py` already exposes for the Dyson dressing, so it is a few
  lines and one response application. The Harris-Foulkes vs Kohn-Sham energy pair at
  the last step brackets it as a zero-machinery cross-check.
- k-point sampling error. The largest untracked term for metals and small cells, and
  the one that does not share the variational structure, since BZ integration is a
  quadrature not a truncated variational space, so the complement trick does not
  transfer. Reachable by mesh extrapolation or integrand-smoothness estimates. Different
  math, higher value.
- Smearing / electronic temperature. The `E - (1/2)TS` (Methfessel-Paxton) T→0
  correction adds almost no cost, the entropy term is already in
  `shared_fermi_occupations`, so exposing the extrapolated energy is the whole task.
- Density-grid (`ecutrho`) and finite-size round out the list. The first is the same
  perturbative logic on the dense grid (USPP/PAW-relevant). The second is system-
  specific (Makov-Payne image corrections for charged/defect/molecular cells).

At leading order these add, but not exactly, since the axes couple (the basis error
depends on the density, which depends on the k-mesh), so the true numerical error
carries cross terms the per-axis estimates omit. Summed, they estimate the distance to
`E_KS_converged` well; they do not reproduce it to machine precision.

The model term, the XC error, is categorically not internally reachable. Converge every
numerical knob and you are left with the exact answer for an approximate functional,
off from reality by the XC error, and nothing in the run measures it. It needs external
reference (CCSD(T), QMC, experiment) or the exact functional. What is available is
weaker.

- Density-corrected decomposition (DC-DFT). The XC error splits into a functional-
  driven part (wrong functional on the exact density, not recoverable internally) and a
  density-driven part (`E_xc[rho_A] - E_xc[rho_B]` for a better density `rho_B`,
  computable and correctable). The textbook `rho_B` is the self-interaction-free
  Hartree-Fock density, which needs exact exchange the code does not have cheaply; the
  available proxy is the LDA↔PBE density sensitivity, and the differentiable machinery
  carries the resulting density change to any observable exactly as the force estimate
  does.
- Functional sensitivity. The `learnable.py` slot plus autograd give `d(observable)/d(XC
  parameters)` in one pass, a linearized single-run version of the BEEF ensemble spread.
  A variance, not the error, calibrated to whatever the parameters span.
- Self-interaction diagnostic. `E(N)` should be piecewise-linear in fractional electron
  number for the exact functional. The deviation is a direct, fully internal measure of
  the delocalization error that dominates gaps and charge transfer, needing only the
  fractional occupations the smearing path already supports.

Self-interaction and the density-driven error are decompositions of the XC term, not
independent channels; treating them as separate additive errors double-counts.

So, does the full budget give the true error? No, for two stacked reasons. The XC
(model) term is not knowable from inside the calculation, an irreducible unknown no
combination of internal estimates reaches; and even the numerical terms only sum to
leading order, missing their cross-coupling. What the numerical budget does provide is a
defensible estimate of the distance to `E_KS_converged`, you can drive the cutoff,
k-point, SCF, and smearing errors to near zero and know that you have. The XC error is
then both the largest remaining term for most production work and the only one you
cannot self-certify. The numerical errors are a solved problem in principle, the
accuracy that matters is in the functional, and the differentiable framework's advantage
on that term is sensitivity and the density-driven half, not an absolute bar.

**What landed.** `postscf/convergence_error.py` holds three estimators.
`estimate_smearing_error` (the scheme-matched `E0 = (E+F)/2` extrapolation with per-
scheme caveats) and `estimate_kpoint_error` (mesh extrapolation `E(N_k) → E_inf`, the
non-variational term that needs more than one run) are validated in
`tests/integration/test_convergence_error.py`. Together with the Ecut estimate in
`discretization_error.py`, those two plus the cutoff term are the trackable part of the
numerical budget that is built. `estimate_scf_error`'s headline is now the robust
piece, it extrapolates the recorded free-energy trajectory (`res.history`) as a
geometric tail, `E_inf - E_last ~ dE_last q/(1-q)`, giving a non-negative `denergy` and
an extrapolated `E_inf` from one run for any system (no χ0 solve), validated by
truncating a converged run's history and recovering the final energy, and by
`estimate_scf_error_bracket` against a loose/tight pair. This sidesteps, rather than
solves, the second-order response formula.

**Still open.** The exact second-order residual energy is
`1/2<x|(K_Hxc - chi0^-1)|x>` with `x` the dielectric-dressed density error, but the
code can only form `1/2<r|K_Hxc (1-chi0 K)^-1|r>`, which omits the `chi0^-1` kinetic-
response term and is not sign-definite. It is retained only as a labelled diagnostic
(`denergy_response`), never the headline. Pinning the exact Schur coupling is the same
missing term as the coarse-space Dyson refinement of δρ in [todo.md](todo.md), resolve
one and both resolve. The `chi0^-1` solve is numerically awkward by direct CG (chi0 is
only known through its forward action), so this is real work. Also open is the model-
term tooling, the fractional-charge self-interaction probe (self-contained, no second
functional) and the DC-DFT density-sensitivity piece, plus the smaller density-grid
(`ecutrho`) and finite-size terms.

**Probed 2026-07-29, shelved.** Two probes measured the discretization estimator's bias
independently and agreed. The machinery is validated in places but too fragile for
production runs today, so the owner's call is to record both and not productionize now.
Archive branches `research/error-calibration-probe` and `research/ecut-recommender-probe`.

The truthfulness grid (Si/C/MgO/NaCl and the metals Al/Cu, three loose cutoffs each) is a
consistent under-estimate. All 18 estimated/true ratios lie in [0.75, 0.98], never above 1,
rising monotonically toward 1 as the cutoff approaches the reference, and the per-system
geometric means agree to within a few percent across chemistry. A leave-one-system-out
calibration on the ecut fraction cuts the held-out log-ratio RMS from 0.1633 to 0.0367,
about 4.5x, a typical held-out miss of 4 percent. The band gap adds nothing at N=6, and a
per-element scale is untestable there since each element appears in one system.

The (ecut, k-mesh) cross term on fcc Al is second order for the total but first order for
the k line item. The largest interior |cross| is at most 1.0 percent of the summed
marginals but up to 119 percent of the k-point marginal below 0.65x of the converged
cutoff. Budget totals are safe to add, per-axis attributions are not.

The second probe read the whole error-vs-cutoff curve off one converged SCF by binning the
complement correction by its per-plane-wave kinetic energy T_G and cumulatively summing
from the top. The bin-and-cumsum reproduces the shipped `denergy` scalar to 2e-16, and the
curve tracks a real Si sweep within a factor of 2 over the whole annulus, a consistent
under-estimate near 0.6x that needs ~1.5x calibration. Inverting the calibrated curve gives
an energy ecut recommender. For Si the recommended vs true cutoff is 23.0 vs 26.5 Ry at a
10 meV/atom target, 31.0 vs 34.5 at 3, and 36.0 vs 38.5 at 1.

Two fragility findings drove the shelving. A pre-asymptotic probe is silently garbage, the
semicore Al_ONCV pseudo at a 15 Ry probe gives 140 eV/atom nonsense with no internal signal
that the probe is too loose, so a production tool must detect and refuse. And the
force-error curve does not extrapolate, the true force error is non-monotonic in cutoff and
hits the eggbox / k-noise floor within one or two steps, so only the probe cutoff itself is
trustworthy.

Revisit criteria. Run the ~20-system sweep (delta-gauge geometries and pseudos, one `gwq`
job) to test whether per-element structure emerges beyond the fraction slope, and build a
robust asymptotic-validity guard, before either the calibration layer or the recommender is
worth productionizing.

## Kato-Temple band-energy enclosures

**Status: open, scoped 2026-08-05.**

After diagonalization the solver already holds everything a rigorous eigenvalue
enclosure needs. Davidson returns Ritz pairs (theta_i, x_i) of the discretized H[rho]
together with their residual norms ||r_i||, the same norms it uses as its own
convergence criterion (`DavidsonResult.residual_norms` in
`src/gradwave/solvers/davidson.py`, formed as `torch.linalg.norm(r, dim=1)` and
compared against `tol`). Those norms are enough to certify where the true eigenvalues
of the diagonalized matrix lie, at no extra cost.

The mechanism is two classical bounds. Bauer-Fike is the linear certificate, for
Hermitian H some true eigenvalue lies within ||r|| of theta. Kato-Temple sharpens it to
quadratic, |lambda - theta| <= ||r||^2 / delta, with delta a lower bound on the gap from
theta to the rest of the spectrum. At Davidson's 1e-9 residual and a gap of order 1 eV
the enclosure is of order 1e-18 Ha, so diagonalization becomes a certified link in the
error chain for one already-computed norm per band.

The difficulty is delta. It must be a certified lower bound on the gap, but only the
computed Ritz values are in hand. The bootstrap runs Bauer-Fike first for coarse
enclosures on every band, uses those enclosures to certify the gaps, then applies
Kato-Temple to tighten. Near-degenerate clusters, metals at the Fermi level, break the
single-eigenvalue form and need the block variant, which certifies the cluster's
invariant subspace rather than the individual eigenvalues.

The scope is narrow and worth stating plainly. This certifies the eigenvalues of the
matrix actually diagonalized, at the ecut the calculation used, with fp64 arithmetic
taken as exact. It does not bound basis-set, SCF, or pseudopotential error. It composes
with the error-estimation module (`docs/manual/error-estimation.md`, with estimators in
`postscf/convergence_error.py` and `postscf/discretization_error.py`), which estimates
exactly those non-rigorous terms, so the report can label which bars are certified and
which are estimates. This strengthens the showcase claim already recorded below, that
per-calculation error bars plus spinor autograd have no published counterpart, since the
DFTK Cancès/Herbst guaranteed bounds are the standing competition.

Next step is a small postscf function taking eigenvalues and residual norms and
returning per-band enclosures, wired into the band-structure report as optional +/-
bars, with a unit test against a dense-diagonalized small matrix whose true eigenvalues
are known.

## Exact-constraint oracle suite for the learned XC

**Status: open, scoped 2026-08-05.**

The exact functional is unknown, but many exact constraints on it are known, and the
SCAN construction used about 17 of them. For a learned functional (the learned meta-GGA
line above) these constraints are guardrails against unphysical extrapolation, and in a
differentiable code the derivative-flavored ones become one-line autograd asserts on the
trained object rather than hand-derivations about a formula.

Five relations make up the autograd-natural set.

- Uniform coordinate scaling of exchange, E_x[rho_gamma] = gamma E_x[rho] with
  rho_gamma(r) = gamma^3 rho(gamma r). This holds automatically when the learned
  enhancement factor takes only dimensionless inputs, but a feature-normalization bug
  breaks it silently, so the test evaluates the functional on a density and its scaled
  copy.
- The Lieb-Oxford bound, E_xc >= -C ∫ rho^(4/3), enforced pointwise as a cap on the
  enhancement factor and hunted adversarially by gradient ascent over the input domain
  (the "Adversarial testing by gradient ascent" section below, the same machinery
  pointed at a new target).
- The Levy-Perdew virial relation, which connects E_x to its own functional derivative.
  Autograd supplies that derivative for free, so the identity ties the energy to the
  exact v_x the SCF uses and catches a broken backward.
- The spin-scaling relation for exchange.
- F(s → 0) = 1, recovering LDA in the slowly-varying limit.

The learnable functional (`LearnableX`, `LearnableSpinX` in
`src/gradwave/core/xc/learnable.py`) already builds in F(0) = 1 and the
Lieb-Oxford-motivated cap by construction, so the suite checks that the construction
survives training and feature handling rather than adding constraints the form lacks.

The same checks serve three roles. As fast-tier property tests they run pointwise on
shipped learned functionals with no SCF. As optional hinge-penalty regularizers they
enter training. And the adversarial ascent is the violation hunter.

The scope is necessary and not sufficient. Passing every constraint does not make a
functional accurate, it prevents unphysical extrapolation.

Next step is a `tests/unit` constraint suite parameterized over the learned functionals
(`LearnableX`, `LearnableSpinX`), plus a training-penalty option in the XC-learning
path. The energy-loss training gradients are already free at convergence
(`energy_param_grads` in `src/gradwave/core/xc/learnable.py`, with the driver in
`examples/train_xc_paw.py`), so the penalty is an added term on the same graph.

## Adversarial testing by gradient ascent

**Status: TRIED 2026-07-29, measured negative. Archive branch
`research/adversarial-testing-probe`.**

gradwave is differentiable end to end, so the tempting bug hunt is to wire two things that
should agree into a disagreement functional D(x), gradient-ascend D to surface worst-case
configurations that equal-budget random sampling misses, and pin each found maximum as a
regression fixture. The premise fails structurally whenever D is built from forces. The
cheap one-SCF-per-evaluation gradient is the frozen-density partial derivative, and for a
force-based D that is a second derivative of the energy, which Hellmann-Feynman stationarity
does not protect. Measured against the re-converged gradient on the FFT eggbox, the
frozen-density gradient overestimates by 2771, 18935, and 11985 at three probe points and
gets the sign wrong at one. It also points consistently toward grid-aligned symmetric
configurations, which are minima of D, so following it is descent, not ascent.

At equal budget the ascent found nothing random did not. Random uniform sampling reached
3.23e-3 eV/Å max |Σ F| over the voxel against ascent's 3.79e-3, and the ascent "win" was
its own random start rather than a followed gradient. A 15-point diagonal scan beat both.
The eggbox landscape itself is informative, D has exact zeros at grid-aligned AND half-voxel
shifts and maxima near 0.07 and 0.43 voxel, so the intuition of a maximum midway between
grid planes is wrong for the net force.

The maximizer's failure mode is a lesson for any optimizer in the loop. It preferentially
drives cells metallic or non-convergent, and one apparent 3.99 eV/Å "disagreement" was a
false positive from a non-converged SCF (converged=False under smearing="none" on a
band-overlap configuration). Per-evaluation SCF convergence, and physical validity, must be
a hard domain constraint on D, which applies to inverse design as much as to bug hunting.
The clean part is a keeper, the analytic-vs-finite-difference force cross-check ceiling is
5.06e-5 eV/Å (relative 1.4e-5), at the h² FD floor, confirming the two force paths agree
everywhere the comparison is defined.

Three revival paths are documented, in decreasing promise. Exact dD/dτ via the
implicit-function adjoint (`scf/implicit.py` already computes response-aware parameter
gradients), with the equal-budget bar raised because each evaluation then costs several
SCF-equivalents. Energy-based D functionals, whose first derivatives are Hellmann-Feynman
exact so the cheap gradient is correct. And dropping gradients for a coarse scan or Bayesian
optimization, which fits this smooth low-dimensional oscillatory landscape better per SCF
than ascent.

## SCF flight recorder: validating the convergence tags

**Status: DONE 2026-07-30 (validation), with a measured negative for auto-remediation.
Archive branch `research/convergence-case-studies`, stacked on `feat/scf-flight-recorder`.**

The flight recorder (`scf.recorder`, PR #203) reads the last five SCF iterations and
returns up to three convergence tags. Running it against the documented hard cases in
`docs/manual/wisdom.md` first surfaced three systematic false positives, all from
`diagnose()` inspecting the settled noise floor of a converged run rather than the
pathological phase of a stuck one. All three are fixed on PR #203 commit 659b845.

- Moment-collapse fired on every healthy ferromagnetic PAW metal. The USPP/PAW
  `seed_moment` is the smooth-grid moment of the SAD seed, many times the converged
  smooth moment (fcc Ni at `start_mag=0.6` seeds ~10.8 µB and holds ~0.79 µB), so the
  relative test `mag < 0.1·seed_moment` was trivially true. A 0.1 µB absolute floor
  fixes it.
- Charge-sloshing fired on a cleanly converged fcc Ni at a ~1e-6 residual, where the
  |G|-shell decomposition is roundoff. A 1e-4 residual floor fixes it.
- Both synthetic true-positive tests still fire, and replaying every stored trace gives
  zero tags on the converged battery.

The hard-case battery then re-validated wisdom.md's documented iteration counts almost
exactly (fcc Ni PAW 18 johnson and 27 pulay against 18 and 27, bcc Fe PAW pulay 30
against 29, the Al(100) kerker slabs 21 and 27, their local_tf counterparts 17 and 21).
Two structural gaps remain, both from the tag being post-hoc over the last five
iterations.

- Charge-sloshing misses the documented converging slab. The kerker Al slabs are
  long-wavelength dominated for 15 or more iterations, but a competent Pulay mixer drives
  the residual down almost monotonically, so the "not falling" requirement suppresses the
  tag. The 2-lowest-shell fraction alone is the discriminator, not the non-monotonicity.
- Level-crossing cannot report a crossing that resolves before the final window. fcc Pt
  reorders 86 times in iterations 2 to 4 (`[0, 64, 17, 5, 0, ...]`) then converges clean,
  so the window is all zeros.

The bcc Fe johnson blowup did not reproduce. wisdom.md records forced johnson going to 93
iterations on the USPP path against 29 for pulay, but on current code pulay took 30 and
forced johnson took 16, with no becsum oscillation left to key a tag against (annotated in
wisdom.md).

Part 2 tested trace-driven auto-remediation for charge-sloshing and it is a measured
negative. A mid-run detector that reads the same shell decomposition, run for `k` kerker
iterations and warm-restarted onto `local_tf` when it flags, costs more than it saves. At
`k=10` the detector flags but the total is 22 (Al 4-layer, 10+12) and 24 (Al 6-layer,
10+14) against `local_tf`-from-start 17 and 21. At `k=6` the sloshing signature has not
yet emerged (the first six iterations average a 2-lowest-shell fraction near 0.35, below
the 0.5 flag), so the detector never fires. Always-robust beats detect-and-remediate,
because the wasted kerker prefix sinks the switch. The follow-up worth taking is exposing
`local_tf` in the YAML schema as a first-class preconditioner choice, not building the
detector into the loop.

## Non-collinear and SOC SCF convergence: a transverse instability

**Status: DONE 2026-07-30 (campaign), archive branch `research/noncollinear-convergence`
(tip b472c1c), no PR. Follow-ons landed and in flight, see the dispositions below.**

The spinor SCF has a residual floor the energy does not share. No magnetic spinor run in
the matrix reaches rhotol 1e-5, under any mixer config (pulay, the #79 recipe, johnson to
200 iterations) on Ni, Fe, or a canted 2-atom cell, yet every arm that holds the FM
branch lands on the same free energy to about 4e-5 eV and the same moment to 1e-4 mu_B.
The fixed points are identical, so the floor is a residual-gate artifact, not a
fixed-point problem. The paired comparison isolates what the spinor path costs, the
scalar-pseudo Ni cell converges collinear in 13 iterations and floors at 80 through the
spinor path with SOC off in both.

The floor decomposes into two independent parts, measured by a `mixer_hook` probe with no
source changes. The transverse magnetization channels (m_x, m_y) start at machine zero
and are amplified by the SCF map at roughly 3x per iteration (3.9e-11 at iteration 10,
6.9e-6 at 20, 7.1e-5 at 40) until they saturate near 1e-4, with the residual power in the
two lowest |G| shells, the magnon-soft long-wavelength sector where the response gain is
near unity and mixing has nothing to contract. A transverse pin prototype holds m_x/m_y
at 1e-14 for all 80 iterations, which proves the amplification lives in the mixed state
and not the band solve. The longitudinal channel keeps its own near-Stoner floor of a few
1e-4 (dm_z 3.4e-4 under pulay), so the two channels fail independently and both need
treatment.

Full parity with the collinear nspin=2 default (johnson plus the quadratic schedule)
converges the spinor run in 16 iterations, but onto the nonmagnetic branch, 1.2 meV above
the FM answer the collinear path holds with the same mixer, and the pin does not save it.
The cause is the m-channel step boost `max(mixing_alpha, 0.6)`, a pulay-tuned guard
against moment collapse that inverts into a collapse accelerant under johnson's normalized
update. Cutting the m step to 0.3 (johnson plus quadratic plus `mag_mixing_alpha` 0.3)
holds the moment and gives the best honest arm, a floor 8x below the default that is still
30x above rhotol.

Recommendations, ranked by evidence, with their dispositions.

- Gate magnetic spinor runs on the energy, not the residual. Landed as the opt-in
  energy-metric gate (#210, next entry), with phase (c) spinor wiring and the YAML knobs
  in PR #215 (the best campaign arm converges in 10 iterations through the gate at the
  consensus F where every rhotol arm capped at 80, and `entol` 1e-4 is the calibrated
  magnetic-spinor tolerance).
- Expose the magnetic spinor knobs (`mag_mixer`, `mag_mixing_alpha`, `mag_diago_schedule`,
  `spin_precond`) through `inputs.py`/`api.py`, since the best measured arm cannot be
  selected from an input file today. Same branch.
- Make the m-step boost scheme-dependent before porting the #205 johnson default to
  `mag_mixer`, since the boost is a pulay-tuned guard that collapses the moment under
  johnson (measured twice, pin and no pin).
- Damp the transverse m channels at low q, a reverse of the Kerker exemption the m blocks
  enjoy, leaving the G=0 moment untouched. Built and rejected, see "Low-q transverse
  damping in the spinor SCF: no-go" below.
- `fix/ni-soc-convergence` is deletable, its substance (the `mag_mixer` selector and
  `build_stoner_precond_nc`) is on main verbatim.

One loose end from the branch audit. The plain NC collinear path (`scf/loop.py`) still
lacks the Stoner spin-preconditioner that the USPP and noncollinear paths carry.
`perf/magnetic-mixing` (df156ed) adds it there as an opt-in `spin_precond` flag and sits
unmerged, worth revisiting.

## Low-q transverse damping in the spinor SCF: no-go

**Status: DONE 2026-07-30 (37 arms), branch `research/transverse-damping` (tip 82b08bb),
no PR. Recommendation 4 of the campaign above, measured and rejected.**

The campaign's proposed cure for the transverse floor was a reverse-Kerker damping of the
m channels, suppress low-|G|, pass high-|G|, leave G=0 free. Built two ways and measured,
the design is structurally mismatched to the instability, and every form fails the
canted-cell kill criterion.

The amplified mode is a rigid rotation of the whole magnetization density, not a finite-q
magnon packet. Its residual grows in lockstep at G=0 and at finite G (5.8e-5 vs 8.2e-5 at
the floor), so no |G|-selective filter isolates it, and exempting G=0 while damping the
cloud tears the rotation apart rather than suppressing it.

The amplifier is the DIIS state recombination, not the mixing step. A residual-side hook
(the campaign's proposed `mixer_hook` insertion) leaves the growth bit-for-bit identical
to baseline through iteration 20. The same kernel applied to the mixer's total update in a
fixed lab frame kills the transverse floor 13x (dm_x 7.9e-6 against 1.2e-4 at the residual
hook, same kernel, same frame, different insertion point). The frame matters as much as
the insertion point, the instantaneous-moment frame tilts with the wandering G=0 noise and
self-defeats, the lab/seed-axis frame holds. Where the damping works, reverse-Kerker outperforms
the flat null 8x, so the |G| structure is real.

None of it lowers the total floor. The longitudinal near-Stoner channel binds at 2e-4 to
9e-4 regardless, the transverse treatments aggravate it, and the champion does not stack
with johnson (dm_x back to 1.9e-4). The best total floor stays the undamped johnson best
arm at 3.5e-4, and SOC is untouched at 1.5e-3 to 2.0e-3 under every treatment.

The kill criterion decides it. Two Fe moments seeded 90 degrees apart must still align,
and the baseline closes 88.7 to 0.6 degrees in eight iterations. Every damped arm fails,
the moment-frame hook drives the pair to 171.8 degrees near anti-alignment, the lab-frame
wrap lands its axis off the bisector, and the SOC lab-frame wrap never completes and
settles 8 meV into a different basin. The G=0-free design does not protect physical
rotation because interatomic alignment is a staggered rotation living at finite G, exactly
where the damping acts.

Verdict, no-go for production. The operative fix stays recommendation 1, gate magnetic
spinor runs on the energy (#210, #215). A future mixer-side cure must damp the update not
the residual, work in a fixed seed-axis frame, and treat G=0 together with its cloud. The
source hook it would need is a post-extrapolation `step_transform` callable inside
`PulayMixer.step`, since the existing `mixer_hook` is the wrong insertion point for any
recombination-driven instability.

## Self-consistent dressing of the Pulay pressure estimate: no-go

**Status: TRIED 2026-07-31 on branch `feat/pulay-estimator-accuracy`, measured and
rejected. Rung 3 of issue #227. The shipped estimator improvements are the iterative
annulus solver (rung 2) and the annulus-factor flatness finding (rung 1).**

The Pulay pressure estimator differentiates a frozen-density energy error, so the
open question was whether restoring self-consistency recovers the roughly 30 percent
that the estimate still misses after the rung-2 iterative annulus solver. The
independent-particle correction relaxes the occupied orbitals into the high-G annulus
in the frozen Kohn-Sham potential and reads off the second-order band-energy change.
Restoring self-consistency adds the Harris-like double-counting term of the total
energy, one half of the inner product of the density change with the Hartree plus xc
kernel applied to it, evaluated with the existing `apply_k_hxc` response primitive.

The measured effect on the silicon harness is negligible. On the diagonal solver the
ratio moves from 0.435 to 0.441 at 10 Ry and from 0.599 to 0.602 at 16 Ry. On the
iterative annulus solver it moves from 0.611 to 0.624 at 10 Ry and from 0.770 to 0.772
at 16 Ry. Every shift is one to two percent of the estimate, well inside the finite
difference noise and nowhere near the missing fraction. This is consistent with the
derivation. The double-counting term is a positive penalty that makes the total-energy
error slightly less negative than the band-energy correction, so it can only shrink the
estimate, not grow it toward the truth. The earlier `dyson=True` density-dressing path
in `estimate_density_error` was already documented as neutral to negative for the same
structural reason.

Verdict, no-go. The frozen-density approximation is not the dominant error in the
pressure estimate. The recoverable half lives in the diagonal kinetic resolvent, which
rung 2 addresses. The remaining gap after rung 2 sits in the annulus-only restriction of
the orbital correction and the second-order truncation, neither of which the
self-consistent dressing touches. The code path is not shipped. Numbers and the harness
are in `benchmarks/pulay_accuracy/`.

## Energy-metric SCF convergence gate

**Status: LANDED 2026-07-30 (#210, phases (a) and (b)), plan
`docs/plans/energy-metric-stopping.md`. Phase (c) in PR #215, phase (d) open.**

An opt-in stopping test that gates on the estimated energy error rather than the raw
density residual, triggered by the `convergence: energy` token against the default
`density`. It forms the kernel-only contraction `(1/2)<r|K_Hxc|r>` from the exact
response operators `scf/implicit.py` exposes, resolved per channel so the
mid-to-high-|G| precessing m-channel mode that carries little energy is down-weighted.
The chi0-dressed response metric was scoped out, since one chi0 application per iteration
needs a Sternheimer solve per band restricted to insulators, neither cheap nor applicable
to the metallic magnets the gate targets. On the stagnating Ni PAW pulay arm the
charge-channel estimate is 1.4e-8 eV against 1.7e-8 measured, and the arm terminates
converged at iteration 91 where the density gate sat at its 120-iteration cap. The
Harris-Foulkes/KS gap is recorded alongside on the NC path as the zero-machinery bracket.
Phase (c) threads the metric through `scf_noncollinear`, exposes the `scf.magnetic` YAML
block, and re-runs the Ni+SOC matrix (PR #215, `entol` 1e-4 calibrated for magnetic spinor
runs and the exact coupled (rho, m) f_xc HVP taken by double-backward). Phase (d) documents
it and flips the default only after a soak across the magnetic and metallic battery, still
open.

## Showcase figures: noncollinear magnetism and error estimates

**Status: open; several inputs half-built.** The validation record is tables of meV
agreements against QE, which persuades a methods reader and no one else. A small set of
figures would carry the two capabilities that distinguish the code, autograd through the
full spinor stack and per-calculation numerical error bars. DFTK has the Cancès/Herbst
error bounds and several codes do noncollinear SOC, but the combination, and anything
built on spinor autograd, has no published counterpart. Candidates roughly by impact,
each grounded in what exists in the tree.

Noncollinear magnetism.

- **MAE sphere for FePt.** `examples/fept_mae_map.py` already scans E(theta) along
  [001]→[100]. The full version is E(theta, phi) − E(easy) as a heatmap on the sphere
  (Mollweide), easy axis marked, with a few full-SCF anchor points overlaid to show the
  force-theorem accuracy. The per-direction magnetic-IBZ folding plus the 7.7× one-shot
  saving makes the map affordable, so the figure doubles as the cost story.
- **Torque against angle.** Plot the autograd dE/dtheta of the moment direction as a
  smooth curve and overlay finite-difference slopes of the E(theta) scan as points. One
  figure shows the SOC physics and that the spinor stack differentiates. The per-atom
  torque is validated in `moment_config`; the global-axis torque through the SOC energy
  is the piece the "MCA dE/dtheta" line still owes, so this is also its acceptance test.
- **Real-space magnetization texture.** A quiver plot of m⃗(r) on a plane through the
  cell, arrows colored by |m⃗|, density as a background contour. The strongest subject
  is a 120° Néel state on a triangular Mn or Cr lattice, where frustration forces
  genuine noncollinearity and a collinear code cannot represent the ground state at all.
  The Fe spiral (`examples/fe_spin_spiral.py`) is the already-computed fallback.
- **k-space spin texture.** `projected_dos_noncollinear` already Pauli-decomposes each
  state into (n, m_x, m_y, m_z). Coloring a band path by ⟨sigma_z⟩ with in-plane arrows
  at each k gives the spin-momentum-locking picture around Γ for Bi₂Se₃, whose SOC bands
  are validated (`examples/bi2se3_inversion.py`). Needs the per-state amplitudes routed
  onto a band path rather than binned into a DOS.
- **Magnetic-IBZ folding diagram.** The full mesh next to the Shubnikov IBZ for FePt
  m∥[001] (144→30) and m∥[100] (144→48), with the folded-vs-full residual quoted. A
  methods figure, narrower audience.

Error estimates.

- **Estimated against true error.** For a grid of systems and cutoffs, scatter the
  `discretization_error` estimate against the measured error to a converged-Ecut
  reference. Points on or bounded by the diagonal are the whole argument for trusting
  the estimator, the plot the Herbst/Levitt paper leads with. The EOS/Δ-factor
  infrastructure already produces the reference energies.
- **EOS with error bars.** E(V) at a deliberately modest cutoff with per-point error
  bars, overlaid on the converged curve, the bars visibly containing it, and the fitted
  a₀/B₀ carrying propagated uncertainties. The force version is displaced-Si force
  components with bars against converged forces.
- **Stacked error budget.** One bar per system, stacked into basis, SCF, smearing, and
  k-sampling terms from `discretization_error.py` plus `convergence_error.py`. Few other
  codes decompose the budget this way, so the figure shows the capability directly.

The strongest single figure is **MAE with numerical error bars**. Anisotropy energies
sit at tens of µeV to meV, exactly where convergence is the standing doubt, so E(n̂) −
E(easy) with a shaded numerical uncertainty band makes a scientific claim rather than a
benchmark claim, that the easy-axis assignment clears the error bar (or honestly, at
which Ecut/k-mesh it does not). The `discretization_error` estimator covers NC nspin=1/2
but not the spinor/SOC path, so it needs the spinor extension first, and this figure is
the reason to build it. Several inputs are half-built (the MAE scan, the spiral, the
Bi₂Se₃ band data, the EOS scans), so the marginal work is mostly plotting plus a few
targeted calculations, with the heavy SOC sweeps routed to the GPU box.

# Performance and scaling

## Acceleration frontier, 2024-2026 literature sweep

**Status: survey; two headline ideas were already implemented, the rest ranked below.**
A focused survey of recent literature (after the local-TF preconditioner landed) for
ideas that pass the filter "single GPU or CPU, small FFT-bound cell, fp64". Two headline
ideas were already in the code. The Gong and Dal Corso trick of batching the H-apply FFTs
across all bands and k-points into one call (arXiv:2412.01695, worth 6× on their small-
cell many-k H-apply) is exactly what `core/batch.py` does over `(nk, nb, grid)`, and the
CPU FFT is already MKL not pocketfft, so the "free 1.5-2× pocketfft to MKL" swap is not
available.

Measured on the RTX 3050 (2026-07-16, torch.profiler on 8 NC SCF iterations, aten-op
device time). This revises the "FFT-bound" framing for the GPU small-cell regime, which
came from CPU profiles and the molecule-in-large-box / USPP-Pt cases. For an ordinary
small crystal on the GPU the FFT is only about 12 percent.

    Si8 2x2x2 (nband 20, m~40, box 27^3): GPU-busy 2111 ms, launch/sync gap 996 ms
      = 32% of wall.  GEMM(bmm) 43%, eigh 21%, QR/ortho 14%, FFT 12%, other 10%.
    Si2 4x4x4 (nband  8, m~16, box 20^3): GPU-busy  652 ms, launch/sync gap 559 ms
      = 46% of wall.  QR/ortho 44%, GEMM 23%, FFT 12%, eigh 11%, other 10%.

Two things fall out. First, a small-cell GPU SCF is dense-linear-algebra-bound, not
FFT-bound, GEMM + eigh + QR being about 78 percent of GPU-busy time (small boxes make the
FFT cheap, and fp64 GEMM/eigh/QR pay the same 1/64 fp64 tax). Second, the launch/sync gap
is 32-46 percent of wall (profiler-inflated but consistent with the earlier finding that
eager dispatch of dozens of tiny kernels per Davidson round is the binding GPU
constraint), exactly what a whole-step CUDA graph reclaims. The eigh cliff is visible,
eigh 11 percent at m~16 vs 21 percent at m~40 (the n>32 cusolver-batched fallback,
measured 2.5-4.5× on its own). Reprioritized. (1) Whole-step CUDA graph to close the
32-46 percent launch gap. (2) Cut the dense subspace LA, RMM-DIIS now attractive because
it removes the Rayleigh-Ritz (eigh) and the subspace orthonormalization (QR), together 35
percent (Si8) to 54 percent (Si2) of GPU-busy, and a c64 subspace reduction on the NC
standard problem would dodge both the fp64 tax and the eigh cliff. (3) The FFT is no
longer the thing to chase on GPU small cells.

- **Whole-SCF-step CUDA-graph capture of the dispatch-bound glue.** The measured GPU
  negatives so far were an apply-only CUDA graph (1.0-1.1×, the back-to-back FFT kernels
  have no launch gap) and torch.compile on the XC functional in isolation. Neither
  touched the 55-65 percent of a step that is many-tiny-kernel real-valued glue between
  the FFTs (XC assembly, mixing, occupations, PAW one-center, density build). Capturing
  the whole step as one CUDA graph (the PyGraph line, arXiv:2503.19779, averages 1.18×
  and never regresses where naive reduce-overhead degrades up to 32 percent) removes the
  per-kernel launch overhead across that glue, exactly where an 8-core host plus a
  consumer GPU hurt most. CUDA-graph capture, unlike torch.compile fullgraph, tolerates
  the complex FFTs. It cannot speed the FFTs themselves. Estimate 1.2-1.5× on the non-FFT
  fraction, GPU only, needs measuring on the RTX 3050. The highest-value new software
  item. (Tried and found non-paying on the post-eigh math, see the Done section.)
- **The batched `eigh` size cliff (diagnostic, cheap).** `davidson_batched` calls
  `torch.linalg.eigh` on the `(nk, m, m)` subspace matrix with `m` about `2*nband`. On
  CUDA the fast `cusolverXsyevBatched` path is used only for `n <= 32`; above that
  PyTorch loops per-matrix (measured about 83× slower at the boundary, pytorch#175585).
  Every real system has `m > 32`, so the subspace diagonalization is probably on the slow
  per-k loop on the 3050. It is only about 5 percent of the CPU profile, but the cliff
  can inflate it on GPU. A ten-minute microbenchmark on asus settles whether it matters;
  if it does, cap or tile the subspace or split the batched solve.
- **ML density initializer, plane-wave-native.** "Global Plane Waves From Local
  Gaussians" (arXiv:2601.19966) and a transferability study (arXiv:2509.25724) report
  25-33 percent fewer SCF iterations, and show a density init transfers out of
  distribution where an ML-Hamiltonian init collapses. It only cuts iteration count, not
  per-iteration FFTs, so about a 1.3× ceiling on a single point, but it stacks with
  everything and its training set is the same shape as the learned-XC data. For MD and
  relaxation the cheaper analog is wavefunction/Grassmann extrapolation across geometries
  (about 3 iterations per step, JCTC 2022 1c00751), which QE and VASP already do and
  gradwave's warm-start approximates.

Skip, from the same sweep, because they do not transfer. Distributed GPU eigensolvers
(ELPA, ChASE, SIRIUS all lose on small subspace matrices), ML Hamiltonian predictors and
learned preconditioners (they need a localized basis, our kinetic preconditioner is
already analytic), tensor-core FP16 FFT (accuracy-fatal against QE-grade fp64), FP8-
emulated fp64 FFT (Blackwell-only, no FP8 on Ampere), NUFFT (our grid is uniform), and
VkFFT (wins only at large-prime grids; `good_fft_size` restricts to 2*3*5*7 radices cuFFT
already handles). RMM-DIIS is the one prototype-worthy eigensolver, it removes the
Rayleigh-Ritz that CheFSI could not, but the RR is cheap at small cell size so the win is
uncertain (built and rejected, see the Done section). The through-line matches the earlier
audit, on a single small SCF the consumer-GPU fp64 tax is the wall, and the durable gains
are throughput (batch many small structures), fewer iterations (learned or extrapolated
start), and a datacenter fp64 GPU.

## The 4x H100 session (2026-07-31)

**Status: DONE, five rounds on 4x H100 SXM (vast.ai, ~1 hr provisioned), data in
`benchmarks/results/h100-session/`, benchmark pack in `benchmarks/h100/` (#212). This
records the datacenter-fp64 numbers the RTX 3050 sections could only bound.**

The consumer-card ceilings the performance sections open with, the CheFSI no-go, the
128-atom memory cliff, the QR and RR-GEMM offload tuning, were all measured on a 6 GB RTX
3050 whose fp64 runs at 1/64 of fp32. This session ran the same battery on real fp64
hardware. Several verdicts hold and several invert, and the split is itself the lesson,
GPU tuning is device-class-specific and has to be benchmarked per card.

- **Minerals.** Cr2O3 eskolaite 1128 s to 46.8 s (24x vs the asus CPU), Fe2O3 hematite
  2275 s to 39.4 s (58x vs CPU, 15x vs the 3050), with identical iteration counts in both.
  These are the production-size magnetic USPP cases the 3050 ran in tens of minutes.
- **Delta gauge.** About 0.3 to 0.4 s per volume against the 3050's 14 to 97 s/vol, 35 to
  240x on the many-k one-atom chains the fp64 tax hit hardest. The complete 25-element
  table has median 0.845 meV/atom against QE, 20 of 25 under 2.0, and every outlier already
  understood, Cu 7.9 (the documented pseudopotential anomaly), Fe 5.4 (tracks pseudodojo),
  and Cr 11.9 (a nonmagnetic run of an AFM ground state). This is the rung-1 baseline for
  the delta-learned plan.
- **Memory.** Si-216 ran with no OOM at 29.4 GB peak and 3.9 s/iter, so the 3050's 128-atom
  cliff is a card limit, not a code limit. Linear extrapolation puts about 350 atoms inside
  an 80 GB card, which widens the glass-model scope for the kappa_min plan.
- **Hybrids.** A PBE0-form SCF runs in seconds on Si2 through Si16 at 0.33 GB peak, 4.6x
  the 32-thread CPU. ACE reproduces the direct Fock on the occupied subspace to 1e-13
  eV/atom (H100 oracle). Two limits recorded, ISDF is not wired into the hybrid SCF (the
  driver hard-codes direct Fock plus ACE per step, `exchange_operator_isdf` unused), and a
  metallic 16-atom 4x4-mesh slab PBE0 timed out at 600 s (full-BZ Fock).
- **Solvers.** Chebyshev loses to davidson at every size, see the CheFSI done-entry.
- **QR and RR-GEMM offload inversions.** The #214 QR-offload gate landed from this data,
  the 3050 measures an fp64/fp32 penalty ratio of 38.3 against the gate threshold of 8 and
  keeps offloading, the H100 measures near 1 and keeps the QR on-GPU (worth 13% there). The
  #195 RR-GEMM device-residency no-win inverts, at hematite's shape the fp64 GPU GEMM is
  1.26 ms against 139.6 ms CPU-resident, and fp32-compute (1.16 ms) has no margin over the
  fp64 tensor cores, so the fp32-subspace trick is permanently obsolete here. The
  consumer-card verdict stands (performance.md).
- **Multi-GPU.** The k-sharded SCF runs unmodified over 4 ranks via Gloo (all ranks
  identical to 1e-11) and scales at real size (2-rank Si-16 6x6x6 at 3.2 s/iter),
  production-blocked by #216, see the distributed record below.
- **Learned-U.** A linear-response U(Pt 5d) on bulk Pt solves in 12 s.

Repo gaps this session surfaced, no H pseudopotential in-repo (SG15 fetched and archived),
no selective-dynamics/atom-fixing in `inputs.py`, ISDF not wired into the hybrid SCF, a
`collect.py` delta-column bug (the H100 column was filled with reference values, trivial
fix), and a cosmetic "56 k(IBZ)" header printed under `symmetry: false` (a time-reversal
reduction shown with an IBZ label on the local shard, display only).

The +U metallic-adsorbate divergence and the vc-relax collapse are their own records below.

## Multi-GPU distributed SCF: status and the gather blocker

**Status: the k-point-sharded SCF (#196, #197) runs correctly across CUDA devices. The
#216 result-gather deadlock is fixed (#218, commits 9e81317 / 8afe0a5). IBZ symmetry
reduction composes with the sharding (2026-08-05): the shard unit is the reduced k-list,
the symmetrizers are k-set-independent and act on the post-all_reduce global
density/becsum, so the 5–14× symmetry factor and the rank count now multiply instead of
being mutually exclusive.**

The k-point-sharded distributed SCF runs unmodified across multiple CUDA devices. Gloo
stages CUDA tensors through host memory, so no NCCL build is needed, and a 4-rank run lands
energies identical to the single-rank result at 1e-11. At toy scale it is slower than one
GPU (startup- and collective-bound), but a real-scale Si-16 6x6x6 on 2 ranks reaches 3.2
s/iter against about 16.8 s/iter amortized single-rank with bit-identical energies, so the
sharding pays once per-k work dominates the collectives.

The #216 deadlock is resolved. The SCF previously completed correctly but hung in post-SCF
result reassembly, where `dist.all_gather_object` pickled each rank's list of large CUDA
coefficient tensors and both ranks blocked in `_object_to_tensor`. It was payload-dependent,
so Si-2 gathered fine while Si-16 6x6x6 deadlocked. #218 replaced the pickle-based object
gather with a sized raw-tensor `all_gather` staged through CPU, and added a
`maybe_destroy_process_group` teardown (called from `api.py`) so the process group closes
cleanly. NCCL plus heavy systems stays the eventual path once a NCCL build is wired in. Until
then Gloo host-staging carries the multi-GPU case.

## Large-U divergence on metallic adsorbate systems

**Status: DONE 2026-07-31 (H100 session round 4, `benchmarks/results/h100-session/round4/`,
flight traces recorded). A new convergence pathology class.**

DFT+U on Pt(111)+H with a bulk-derived U(Pt 5d) = 8.949 eV never converges, oscillating
about 30 eV across a 300-iteration cap under three mixing configs. The bare slab plus the
same U was fine, so the adsorbate is required. The threshold is (4, 6] eV, U=2 and U=4
converge in about 30 iterations while U=6 and U=8.95 never do.

The mechanism is occupation flip-flop and level-crossing, quantified from the recorder
traces. Occupation-reorder counts scale with U, 0 in the converging tails, 6650 total at
U=6, and 28597 at U=8.95, where the Fermi level swings 0.235 eV against the 0.14 eV
smearing width. When U/2 exceeds the smearing width the +U potential shift crosses levels
through the Fermi window faster than the one-step-lagged occupation update can track, and
the density limit-cycles.

Two candidate causes were ruled out. Charge sloshing is present in every run including the
converged ones, so the channel-resolved traces cleared it as a red herring. The adsorbate
dipole is exonerated by a symmetric double-H discriminator (net adsorbate dipole
cancelled), which still diverges at U=8.95 and still converges at U=4, so the divergence is
U-driven and geometry-independent.

The PBE binding is clean, E_b = -0.55 eV at z* = 0.95 A from an unrelaxed z-scan, plausible
against literature, and the learned-U solve itself ran in 12 s. Remedies to pursue, +U
occupation-matrix mixing or damping to kill the one-step lag, U-ramping into the SCF, and
caution transferring a bulk-derived U~9 eV to a metallic adsorbate system where the level
structure differs. Recorded for the manual under convergence and wisdom.

## Variable-cell relax collapses on soft framework cells

**Status: filed as #217 (H100 session round 5). Blocks kappa_min plan phase 2, worked
around by fixed-cell relax.**

A variable-cell relax of 9-atom alpha-quartz SiO2 (PBE, 30-40 Ry) reported converged at
fmax 0.009 eV/A in 52 steps, at a collapsed cell, a 4.916 to 3.623 A and volume 113.1 to
50.1 A^3, 44% of the start. The FFT grid and plane-wave basis are held fixed while the cell
shrinks, so the effective ecut rises with compression and the uncorrected
basis-incompleteness (Pulay) stress spuriously favors collapse. Hard framework oxides with
soft shear modes are the worst case.

The consequence for the kappa_min plan is that phase-2 equilibrium geometries must come
from ions-only relax at a fixed literature cell (or an EOS scan with the grid rebuilt per
volume) until #217 is fixed. The elastic machinery itself validated on the same session,
the trigonal C_ij pattern is correct and densities land within 0.5% of literature. The
Cahill-Pohl arithmetic checked out, kappa_min(SiO2) = 1.46 W/mK against literature 1.3 to
1.4 using literature moduli, while unrelaxed moduli overestimate it 1.9x, which is exactly
why the trustworthy geometry is the blocker. Candidate fixes, rebuild the grid when the
volume drifts past a threshold (with a mixing restart), or apply a Pulay-stress correction
along the path.

## Learned multi-pole density-mixing preconditioner

**Status: closed on real systems (2026-08-05, slab test); fit hardened (2026-08-04,
#232). The fit is multi-seed, quality-weighted, and gated behind a Kerker abstention,
so losing to Kerker is impossible at fit time. The battery is 1 synthetic win, 5
real-system ties, 0 losses, and the measured reading is that Pulay DIIS already
absorbs the single-pole reshaping the earlier wins were made of. The Al(100) slab
test then closed the line. On the 4- and 6-layer slabs the robust fit selected K=1
and tied bare Kerker (21 and 27 iterations) while local-TF kept its win (17 and 21).
The diagnosis is structural, not statistical. The multipole filter is radial in G,
but a slab's screening inhomogeneity is spatial, screen in the metal and not in the
vacuum, an operator that is r-dependent and off-diagonal in G. The radially averaged
probe sees the two regions smeared into one near-single-scale d(G), and the gate
correctly abstains. Bulk metals are single-scale (robust ties) and spatially
inhomogeneous cells need local-TF's operator family, so no tested real system
produces the radially staged response the mechanism wins on. Any successor lives in
the local-TF family (a learned local screening functional), not in radial poles.
Earlier milestones were the Cu win and the Cu₃Al harness and SCF oracle (2026-07-24,
#60), and the magnetization-channel extension stays a measured negative.**

The sweep above skips "learned preconditioners" because they need a localized basis and
our kinetic preconditioner is already analytic. That reason is about *eigensolver*
preconditioners. A learned *density-mixing* preconditioner is a different object. It
lives entirely in G-space, needs no localized basis, and generalizes the Kerker and
local-TF filters the code already ships. It is the approach `docs/manual/wisdom.md`
points at twice, prefer a preconditioner to step-size control, and the SCF iteration
count is set by density mixing, not by the initial wavefunction.

**The mechanism (`scf/learned_precond.py`).** Bare Kerker, R̃(G) = R(G)·G²/(G²+q0²), is
the single-pole long-wavelength approximation to the exact response preconditioner
ε⁻¹ = (1 − v_c χ₀)⁻¹. `MultipoleKerkerPrecond` replaces the one pole with a learned sum,
f_θ(G²) = Σ_i w_i·G²/(G²+q_i²), applied per density-sphere component where the mixer
applies Kerker (wired as `scf(..., precond_op=)` and `mixer.precond_op`, mirroring
local-TF). Two Kerker properties carry over by construction, f_θ(0) = 0 so the pinned G=0
charge is never touched, and the fixed point is unchanged so a bad filter costs
iterations, never accuracy. K=1, w=1 reproduces bare Kerker to round-off, so the single
pole is always inside the hypothesis class.

**The fit** is where a differentiable solver does something a non-differentiable one
cannot. In the diagonal model the error of a mixing step evolves as
e_{n+1}(G) = [1 − α·f_θ(G²)·d(G)]·ē(G), with d(G) = 1 − j(G) the response denominator and
ē the mixer's extrapolate. `fit_multipole` unrolls that recurrence and backpropagates the
residual to the pole weights and positions. `mixer="plain"` unrolls damped linear mixing
(ē = e_n) and minimizes the worst-shell rate, the wrong objective when the deployment
mixer is Pulay DIIS, as the first pass found. `mixer="diis"` (the default the benchmark
uses) unrolls the actual Pulay recurrence, ē = Σ c_i e_i with the DIIS coefficients from
the same bordered, Kerker-metric, Tikhonov-regularized solve `mixing.PulayMixer` runs, so
the filter is trained to complement the low-G work DIIS already does with its history
rather than duplicate it. The coefficients do not depend on the filter, but the e-history
they extrapolate does, so the gradient flows. gradwave can differentiate through the Pulay
recurrence, which other plane-wave mixers do not. `response_from_residuals` estimates d(G)
per |G|-shell from a short SCF captured through the new `scf` `mixer_hook`; probing a
d-band metal with plain damping sloshes, so it probes with Kerker ON and divides the
Kerker factor back out of the residual ratios to recover the bare d(G).

**Measured** (`benchmarks/bench_learned_precond.py`, `tests/unit/test_learned_precond.py`),
gaussian 0.1 eV, PBE, energy-gated rhotol 1e-6, all filters reaching the Kerker energy to
a few 1e-12 eV (fixed point unchanged, as designed).

- Synthetic two-scale response. Learned three-pole spectral radius 0.82 → 0.50, a 3.5×
  iteration ratio, and under a DIIS unroll the post-DIIS residual falls from 1e-5 to
  1e-16. Isolates the mechanism from DFT cost.
- fcc Al (30 Ry, 6×6×6). Kerker 7 iters, learned 7 (tie). The DIIS-aware fit clusters all
  three poles near q ≈ 1.0 Å⁻¹, correctly recognizing that a single-scale homogeneous
  metal has nothing for a radial filter to win, and does no harm. (The plain-fit first
  pass took 9 here, a loss, which named the DIIS fix.)
- fcc Cu (45 Ry, 6×6×6, 3s3p semicore d-band). Kerker 10 iters, learned **8**, a 20
  percent cut. The fit spreads its poles across q ≈ 0.07, 0.22, 0.70 Å⁻¹, the multi-scale
  shape a single Kerker pole cannot take. First real-system win.

The Cu result settles which objective is right, sharply. Its plain-mixing spectral radius
went up under the fit (0.39 → 0.82) while its DIIS iteration count went down (10 → 8). The
plain rate is not the deployment rate; only unrolling the real mixer predicts the real
win, and optimizing the plain rate actively mis-ranks filters. That is the load-bearing
lesson.

**Cu₃Al harness + SCF oracle (2026-07-24, #60).** `benchmarks/bench_learned_precond.py`
gains `run_cu3al`, an L1₂ Cu₃Al intermetallic (the same QE-validated cell and pseudos as
`test_metal_forces_vs_qe`) where two chemical species screen at two different lengths, the
multi-scale-charge frontier this section named after the Cu win. It compares bare Kerker
against the default DIIS-aware 3-pole fit and a wider 4-pole fit. The companion
`tests/integration/test_learned_precond_scf.py` is the end-to-end oracle that was missing,
deploying a genuinely multi-pole filter through the real SCF driver and asserting the
converged free energy and eigenvalues match bare Kerker to solver precision, with
f_θ(G=0)=0. The honest scope is unchanged, the filter earns iterations only on a genuinely
multi-scale charge response (the Cu d-band win, 10→8), and on a single-scale homogeneous
metal like Al it clusters its poles and ties Kerker within an iteration. The benchmark's
iteration counts are measured at run time, not committed as fixtures.

**Magnetism and SOC, and where the charge-channel filter stops (measured).** The
`precond_op` and `mixer_hook` hooks now reach `scf_noncollinear`, so the filter deploys on
the collinear nspin=2 path (total block) and the noncollinear/SOC path (charge block, m⃗
blocks keep their own step). fcc Pt nonmagnetic + SOC ties Kerker (9 vs 9) at a fixed point
identical to 2e-11 eV, the wiring is correct through the spinor SCF and Pt's charge
response is single-scale so the fit reproduces Kerker, as on Al. bcc Fe (nspin=2
ferromagnet) loses (12 vs 13), the fit is on the charge (total) block but the FM
convergence bottleneck is the magnetization channel (the Stoner mode with the measured
gain near −6) which a charge-block operator cannot touch, and the noisy nspin=2 probe
(plain damping wobbles the moment so d clamps) gives a slightly weak charge fit that costs
the extra iteration. The learned filter is a charge-multi-scale tool, and it neither helps
nor is meant to help a magnetization-channel problem.

That pointed at what looked like the real magnetism fix, the operator wisdom.md asks for
by name (the χ₀-diagonal preconditioner on the spin mode), a learned filter on the
magnetization block. The infrastructure is built and tested, a G=0-alive filter form
f_mag(G²) = w0 + Σ w_i·G²/(G²+q_i²) (the `const` term on `MultipoleKerkerPrecond`, since
Kerker's G=0 zero would freeze the moment), a `BlockPrecond` composite running bare Kerker
on the charge-total block and f_mag on the mag block, and the nspin=2 grid-spanning
`precond_op` wiring in `scf`. The first hypothesis it enabled was wrong, and the
measurement says so cleanly. The guess was that the near-critical uniform Stoner mode wants
damping (small w0, since its spin susceptibility is large so its inverse is small). On fcc
Ni near Stoner (PD_Ni NC, 45 Ry, 4×4×4, johnson, `benchmarks/bench_learned_precond.py ni`)
a w0 sweep under johnson gives baseline 12 iters at m = 0.537 µB, w0 = 0.4 collapses the
moment to 0 (23 iters to the wrong nonmagnetic branch), w0 = 0.6 gives 16 iters (holds the
moment but slower), w0 = 0.8 gives 12 iters (recovers the baseline). Damping the uniform
mode is backwards, the moment mode needs vigorous mixing to hold the ferromagnetic branch
(wisdom.md's moment-collapse warning seen from the preconditioner side), and reducing it
either collapses the moment or slows the run. There is no w0 that wins.

The lesson refines the direction. Charge sloshing is a linear, diagonal-in-G problem where
a filter shape is the right tool, and the Cu win is real. The FM convergence bottleneck is
branch selection, a nonlinear problem that vigorous moment mixing plus warm-start chains
(johnson) already handle, and a linear filter on the mag block does not have the right form
and can hurt. A mag-channel operator, if one helps at all, would have to come from the
exact spin susceptibility (`scf/implicit.py`'s χ₀ path), not a hand-shaped or linearly-
probed Kerker analog, and even then the headroom over johnson looks thin. Recorded as a
measured negative; the const-filter and BlockPrecond stay as reusable substrate.

**Fragility, measured (2026-08-04).** The Cu win did not survive an unrelated change.
Commit e5a8ac4 (#91, the rfft Hartree round trip), a last-bit-rounding-equivalent
change, flipped Cu from 10 to 8 back to a tie, and an automated bisect pinned the flip
to that commit. The fit was deterministic but fragile. The probe's d(G) estimate moves
at the last bit and the optimizer lands in a different local minimum. Two more probe
pathologies were on record, the Cu₃Al and Fe d(G) estimates saturate the [0, 2] clamp,
and a wide-seed 4-pole fit placed a pole at q = 0.034 Å⁻¹, overfit to lowest-shell
noise, and lost 2 iterations. The full battery had walked back to 0 wins, 4 ties, and
1 loss.

**Robust fit and the abstention gate (2026-08-04, #232).** `fit_multipole_robust`
hardens the fit at the one-iteration scale by four mechanisms, all deterministic.
Multi-seed fitting runs every pole count K = 1..K_max from a fixed set of seed
placements spread over the pole range and keeps the candidate minimizing the
unrolled-DIIS objective, so the choice between local minima is made by the objective
rather than by where floating-point noise dropped the seed. Probe-quality weighting
(`response_from_residuals(..., return_quality=True)`) turns the spread of the
per-iteration ratio samples into a per-shell confidence, zeroes shells whose mean
saturates the clamp, and sets a hard floor on pole positions at the lowest trustworthy
shell, which removes the q = 0.034 failure mode structurally. Model selection always
includes bare Kerker as a candidate and picks the smallest K within tolerance of the
best objective. The abstention gate deploys a K ≥ 2 fit only when it beats the best
single-pole candidate by a margin decisively above fit noise, and otherwise returns
bare Kerker exactly, so a noise-level win or loss becomes a guaranteed tie and losing
to Kerker is impossible at fit time. The regression tests pin the 1e-14 noise case,
the single-scale abstention, the fewer-poles preference, the pole floor, and the
deploy branch (`tests/unit/test_learned_precond.py`).

The calibration behind the gate is measured, and it explains the walk-back. Under the
DIIS-aware objective a free-weight single pole captures the headroom on every smooth
radial response tried (two-scale, three-scale, flat) to within 0.7 log units, while
1e-14 probe noise moves candidate objectives by up to 0.75, indistinguishable. The 10
to 30 log-unit predicted gains over bare Kerker are single-pole reshaping the real
Pulay mixer already absorbs with its history, which is why they never survived
deployment. The gate opens only on genuinely staged screening. A d(G) stepping down at
two separated scales gives K = 2 a 20 to 27 log-unit margin over the best single pole
and deploys.

**Battery (asus, 2026-08-04), all fixed points identical to ≤ 4e-11 eV.**

| case | kerker | robust fit | verdict | gate |
|---|---|---|---|---|
| synthetic two-scale (plain mixing) | ρ 0.82, ~93 iters | ρ 0.50, ~26 iters | win, 3.52× | mechanism case |
| fcc Al (30 Ry, 6×6×6) | 7 | 7 | tie | Kerker model-selected |
| fcc Cu (45 Ry, 6×6×6) | 10 | 10 | tie | Kerker model-selected |
| bcc Fe (nspin=2 FM, johnson) | 12 | 12 | tie | Kerker model-selected |
| fcc Pt (SOC, 45 Ry, 4×4×4) | 9 | 9 | tie | Kerker model-selected |
| L1₂ Cu₃Al (40 Ry, 2×2×2) | 11 | 11 | tie | abstained |

No case loses, and none can. The former Cu win is a structural tie rather than a
noise-dependent outcome, and the former Fe loss (13 vs 12) is gone. The synthetic
plain-mixing win is untouched, so the honest scope tightens further. The radial
multi-pole class earns iterations only where the deployment mixer cannot, under plain
or short-history mixing, or on a response with genuinely staged screening, and none of
the probed systems shows one.

**Next, in rough priority.** The 2026-08-04 battery narrows the search. A radial filter
under Pulay DIIS earns nothing on smooth responses, so the systems worth probing are the
ones whose screening is plausibly staged, PAW semicore metals and larger cells near the
charge-sloshing cliff, where the gate would open on real structure instead of abstaining.
A cleaner probe reads the response straight from the implicit-differentiation machinery
(`scf/implicit.py` already applies χ₀ and K_Hxc), making d exact and removing both the
probe SCF and the clamp saturation that costs Fe and Cu₃Al their low-shell data. And the
filter is radial (G-only) where the local-TF operator is spatial (r-only). A learned
operator that is both is the general form, and the two current preconditioners are its
limits. Amortizing the fit across a chemistry family only becomes interesting after some
system deploys.

## Second-order joint descent: exact Hvp Newton-CG (2026-07-27)

**Status: open; production joint substrate in flight.** The July performance campaign
established that the remaining software gains are in iteration counts, not kernel speed
(see wisdom.md, "GPU latency and precision"). This is the largest iteration-count idea,
second-order geometry optimization through exact Hessian-vector products, with the memory
math and the application argument attached.

**The mechanism.** BFGS spends its early ionic steps learning curvature from rank-two
updates, so every step is partly a probing move. A trust-region Newton-CG optimizer
(Steihaug) needs the Hessian only through products $Hv$, and autograd supplies exact
products by double-backward at 2 to 3 gradient-cost each. The trap in nested relaxation is
that second derivatives of the SCF-converged energy contain the orbital response
$d\psi/dR$, a Sternheimer solve per product. The joint functional from #123 dissolves the
trap, since $E(\text{strain}, R, Z)$ is an explicit function of all its variables and
double-backward on it is plain autograd. The CG loop then carries the electron-ion coupling
that BFGS-on-ions has to learn the hard way. No plane-wave code has this combination,
because none of them can afford exact $Hv$. Expected regime, 3 to 6 Newton steps at 5 to 15
CG products each against 15 to 50 quasi-Newton steps, which pays on soft-mode systems and
loses on 4-atom cells that relax in 2 steps.

**Memory envelope (the one hard requirement).** Retained activations for one gradient pass
scale as roughly $3 n_{bk} N_{grid} \times 16$ B, and double-backward holds 2 to 3 times
that. For 64-atom Si at 45 Ry ($n_{pw} \approx 44$k, grid $\approx 88^3$, $\approx 290$
band-k pairs) that is about 9 GB for the gradient graph and 20 to 27 GB for an $Hv$, and a
128-atom cell reaches 150 GB. Naive double-backward fits nothing we own. Band-chunk
gradient checkpointing (recompute the FFT sandwiches inside backward, retain per-chunk
summaries) drops the 64-atom footprint to 1 to 2 GB at 1.5 to 2× flops per product, and the
chunking to hang it on already exists in `core/batch.py`. Checkpointing is a prerequisite,
not an optimization.

**Why glassy insulators are the launch application.** Low-thermal-conductivity glasses and
anisotropic heat conduction are phonon-engineering problems on large, gapped, soft-mode
cells. Those cells are simultaneously the worst case for BFGS, the best case for exact
curvature, and inside the insulator coverage the current joint machinery already has. Two
byproducts land on the same graph. The converged joint Hessian projected onto the ionic
block is the dynamical matrix, so Hvp plus Lanczos gives matrix-free phonons with the
supercell-phonons module as the self-oracle. And a second application of the same trick
($Hv$ of $Hv$) gives third-order anharmonic force constants without displacement
combinatorics, the expensive ingredient of thermal-conductivity design.

**Metals.** Smooth at the functional level, a measured negative at first order (PR #126,
closed 2026-07-27 without merging). Three occupation-block strategies were tried, full
subspace rotation, detached frozen occupations, and a preconditioned Marzari-Vanderbilt
block with a degeneracy-robust eigh. The MV electronic solve itself is exact and eigh-free
(smeared Al converges to the SCF free energy within 1.2e-7 eV in 27 closures, occupations
to 1e-10), but the joint co-descent either never moved the atoms (the frozen-occupation
head-to-head was an artifact) or, once genuinely live, inverted the H-apply ratio to
~0.4x against nested BFGS, versus 7.8x for insulators. The bottleneck is ionic/ensemble
conditioning, not the occupation model, which is exactly the case for exact curvature.
Whether Newton-CG fixes it is the open question this plan tests, unresolved by #126,
which was first-order only.

**What remains, in order** (items 1-4 landed 2026-07-27, the production substrate in #139
and the gradgradcheck audit, chunk checkpointing, and Steihaug optimizer with Si benchmarks
in #148; the metals research branch `research/joint-descent-metals` closed unmerged in #126
and its worktree is prunable).

1. The production joint substrate (landed, #139).
2. A double-differentiability audit of the joint energy graph, `gradgradcheck` on a tiny
   cell. FFTs are linear and free, Cholesky is doubly differentiable in torch, and the
   known blocker is any custom `autograd.Function` without a double-backward, RobustEigh in
   particular. The insulator path avoids eigh via Cholesky orthonormalization, so the audit
   likely passes there as-is.
3. Band-chunk checkpointing through the density build (the memory table above).
4. The optimizer itself, Steihaug trust-region Newton-CG over the joint variables, with a
   preconditioned CG loop (Pfrommer-style ionic block, Teter-like electronic block) and the
   Cholesky parametrization handling gauge.
5. The validation ladder with one honest go/no-go number, total H-applies against nested
   BFGS and first-order joint descent on perturbed Si-16, then Si-64, then an amorphous cell
   near 100 atoms. The method earns default status only if the H-apply count drops on the
   amorphous cell. Measured on the H100 (#206, round 3), the nested-to-joint H-apply ratio is
   7.8x on Si-2, 7.69x on Si-16 (2x2x2), and 2.30x on Si-64 (Gamma-only, so the drop conflates
   atom count with k-point count and is not a clean size trend). The edge does not grow with
   cell size across this ladder, so the amorphous-cell default gate looks unlikely to be met,
   though joint stays faster at every size measured.
6. The phonon byproduct check, Hvp-Lanczos dynamical matrix against
   `postscf/phonons_supercell.py`.
7. Metals, gated solely on #129.
8. The third-derivative pipeline toward $\kappa$ as the follow-on, once 1 through 5 hold.

## Batched multi-structure SCF, and the EOS-on-GPU question

**Status: open; profiled, the batched path is the main structural gain.** Would an EOS go
faster by batching several volumes on the GPU at once?

Measured on the asus RTX 3050 for the 1-atom fcc Pt EOS (40/400 Ry, 12×12×12), a single
point sits at 100% nvidia-smi util but only 24.7 W of draw (the card's TGP is 35 to 80 W)
and 2.6 of 6 GB. The 100% util flag only means a kernel was in flight during the sample. The
low power and low memory say the GPU is not compute-saturated. For a system this small it is
launch and latency bound on many tiny kernels (small matmuls, a 35^3 FFT, per-k Davidson
steps), so there is real headroom. Yes, concurrency would help. Three ways, cheapest first.

- Run several volumes as concurrent processes sharing the GPU, plain backgrounding or CUDA
  MPS. Zero code. Two points fit in 6 GB (2 × 2.6). Likely 1.5 to 1.8× on a launch-bound
  system. The catch is that the current EOS chains the volumes with `start_from` warm
  starts, so they are serial by construction. Dropping the chain trades the warm-start
  iteration savings for the concurrency, close to a wash at N=2 but a win as the GPU empties.
- Batched multi-structure SCF, the main structural gain. Stack the volumes as independent
  k-blocks in one padded generalized Davidson, the way the batched Davidson already stacks
  k-points, so the small per-volume GEMMs become one big GEMM and the launch overhead
  amortizes. The SCF loop has to carry per-volume densities and potentials and mix them
  independently while sharing the linear algebra, substantial feature work. This is the
  version that fills the card. It generalizes past EOS to any embarrassingly-parallel set of
  small structures (displacement stencils for phonons, rattled configs for training data, a
  k-convergence sweep).
- CUDA streams to overlap independent kernels. Hard to orchestrate from PyTorch eager, low
  priority.

This only pays for small systems where a single SCF underfills the GPU. The slab already
uses more of the card, so batch structures for the cheap cases (bulk EOS, phonon stencils)
and run the heavy cases one at a time. The cleanest first target is a spin-spiral /
magnetic-dispersion sweep (`examples/fe_spin_spiral.py`). Every angle theta is the identical
cell, k-mesh, and band count, same FFT dims and tensor shapes, so the batch has zero
raggedness; only the per-point convergence count differs. That is cleaner than the EOS,
where the cells (and FFT boxes) vary with volume. The one wrinkle is the frustrated large-
angle points needing many more iterations than the collinear ones, so a lockstep batched
solve either over-iterates the easy members or needs per-member convergence masking. The
real blocker is the hardware, not the workload, on the RTX 3050 the sweep is fp64-bound and
7.8× slower than the CPU (it runs as concurrent CPU processes today). On a card with real
fp64 (A100/H100, fp64 = 1/2 fp32) and tens of GB, stacking these identical independent SCFs
to fill the device is where the batched path first pays off.

The best fit is GGA insulators. They are fixed-occupation, converge in few iterations, and
hold a small grid, so a single one badly underfills the card, exactly the regime where
stacking wins. A batch of GGA insulator structures is also the shape of a learned-XC
training set and an EOS or convergence sweep, so this feature and the meta-GGA training work
reinforce each other.

## Gamma-only real wavefunctions for slabs and molecules

**Status: representation built and validated; the FFT speedup did not appear on this CPU.**
At the Gamma point the orbitals can be taken real, because time reversal makes ψ(−G) = ψ*(G),
so only half the plane-wave sphere is independent. The foundation is built and validated in
`core/gamma.py`, gated to machine precision against the complex path (apply 1e-13, frozen-
potential eigenvalues 5e-14). It stores the half sphere, runs the local term on
`irfftn`/`rfftn`, and solves the eigenproblem as a real symmetric one in a feature embedding
where the half-sphere metric is the plain dot product, so the standard Davidson applies
unchanged.

The premise was a roughly 2× real-FFT win on the hottest kernel. That did not appear on the
available CPU. The forward-plus-inverse real transform measured 0.75× to 1.25× the complex
pair on non-power-of-two boxes at 63^3 and 72^3, so the H-apply came out 0.97× in isolation,
and the full solver ran slower still (directionally 0.6× to 0.8×) once the per-apply overhead
compounds over the Davidson iterations. The real-transform advantage is grid-size and library
dependent, and MKL did not deliver it here. The correctness is solid, so the remaining work is
measurement and integration.

- Re-measure on a GPU. cuFFT's real transform behaves differently from MKL's, and the memory
  story is better on the GPU, so the win may exist there. Check this first before investing
  more.
- Wire it into the SCF loop behind a flag, single Gamma k-point, insulators and molecules
  first, then metals at Gamma with smeared occupations. The density build, mixing, and energy
  assembly are unchanged, only the diagonalize call swaps.
- The memory angle stands on its own. The real-space fields are half the size, so this pairs
  with the size-ceiling item below independent of any speedup.

## Raising the system-size ceiling past the dense-allocation cliff

**Status: open, when-you-need-it. The fix is a tiling change, not an architecture change.**
The GPU probe found peak memory scaling roughly linearly to about 96 atoms on the 6 GB RTX
3050, then a hard cliff at 128 atoms from a single roughly 37 GB allocation, an O(npw²) dense
step (complex128 around 7.7 GB times the eigh workspace copies) that spikes at `npw` near 22k.
The practical ceiling is about 96 to 110 atoms at that cutoff, and the cliff is a specific
dense allocation, not gradual fill, so it is tileable rather than fundamental.

- Identify the O(npw²) step. The dense object that scales with the square of the plane-wave
  count, most likely a subspace-related workspace or the eigensolve's internal copies; the
  first task is to confirm which allocation trips at 128 atoms with a memory profile.
- Tile or avoid forming it. Block the offending contraction so the peak is bounded the way
  `BatchedHamiltonian.apply` and `density_b` already band-chunk their dense-grid temporaries,
  or restructure the step to never materialize the full O(npw²) array.

This only matters if larger cells become a goal, defects, bigger slabs, or supercells for
finite-q phonons. But it is the one thing standing between the sub-100-atom validation regime
and running the kind of system where the code would do new science. The ISDF work above is the
complementary approach, it lowers the operation count where this item lowers the peak memory.

## Davidson subspace Gram: conj-copy memory spike at large nk

**Status: open, deferred; two cheap fixes identified.** Measured on the A100 384-k FePt run
(job 14076535, 2026-07-18), with fragmentation already fixed (expandable_segments), the run
died at 32.2 GiB allocated when `davidson_batched`'s subspace overlap
`torch.einsum("kig,kjg->kij", v.conj(), hv)` requested another 6.68 GiB, since einsum
materializes `.conj()` as a full copy of the (nk, nsub, 2npw) subspace block. Two cheap fixes
when it next matters, chunk the Gram over k (the result is only (nk, nsub, nsub), tiny), or
restructure to avoid the conj copy (`(hv @ v.mH)`-style batched matmul conjugates lazily).
Deferred because the magnetic-IBZ fold (60–100 k instead of 384) removed the pressure, but any
future dense-k run without magnetic symmetry hits the same wall at nk·nsub·2npw·16 B ≈ 7 GiB
per copy.

## One-center ddd analytic derivative

**Status: open, low priority.** The one-center ddd is a named micro-cost from the performance
audit, 5% of the PAW profile through an autograd backward per iteration. It is already compiled
when `compile_xc=True` (the `energy_and_ddd` path is a single backward), so the remaining
question is only whether an analytic quadrature derivative beats the compiled autograd, a small
isolated experiment, not a feature.

## Campaign setup amortization: a warm fork-server for cold-start cost

**Status: open, prototype-worthy. Measured 2026-08-06. The setup cost it targets is real and
un-shardable, and the fix is pure systems engineering with zero effect on the physics.**

gradwave's per-process cold start is fixed serial work done before any physics: the torch
import, the one-time XC `torch.compile`, the UPF parse, and the form-factor / radial-table
build. None of it shards (k-point sharding and IBZ reduction leave it untouched, and it is
replicated on every distributed rank, see the distributed record above). For a big cell it is
noise, but gradwave's bread-and-butter is small cells (1-16 atoms) in large campaigns (an EOS
scan, a delta-gauge column, a phonon stencil, a rattled training set), where one SCF is seconds
and cold start is tens of seconds, so the campaign spends most of its wall clock re-doing
byte-identical warm-up.

Measured footprint of a warmed CPU process (asus, `import torch` + the gradwave stack + warmed
BLAS): RSS ~540 MB, of which the private/anonymous memory a snapshot must carry is ~320 MB,
dominated by torch's ~275 MB heap; gradwave's own imports plus physics tables are only ~45 MB.
The expensive state to preserve is almost entirely PyTorch, not anything DFT.

Warm the state once and reuse it, in one of three flavors, cheapest first.

- **Fork-server / zygote (the CPU quick win, zero disk).** Warm one parent, then `fork()` a
  child per job. Copy-on-write makes the fork O(page-table), a few milliseconds regardless of
  the ~500 MB RSS, and the child inherits the imports, parsed pseudos, and tables for free. Each
  child runs one SCF and exits, so a crash, OOM, or diverged solve dies with the child and never
  poisons the warm parent, which is the advantage over a single long-lived in-process daemon.
  This is Android's zygote and Gunicorn/uWSGI prefork. Low-effort route is the stdlib:
  `multiprocessing.set_forkserver_preload(["torch", "gradwave.api"])` +
  `set_start_method("forkserver")`, then point `pueue`/`gwq` at the resident server instead of
  launching a cold `uv run python -m gradwave` per slot.
- **Persistent in-process daemon.** One warm process serving structures over IPC, no fork. Same
  setup saving, no per-job fork cost, but no crash isolation, so a memory leak or segfault in one
  SCF corrupts the shared state. Needs real isolation discipline.
- **CRIU / durable image.** Freeze the warm process to a disk image (~300 MB raw, ~100-150 MB
  zstd; ~400-600 MB once the compiled XC is baked in) and thaw copies, including on the asus
  worker. The only flavor that ships across machines, at the cost of privileges (CAP_SYS_ADMIN,
  awkward on unprivileged NixOS), and it captures no GPU state.

Three hazards fix the shape of any implementation.

- **fork-after-threads deadlocks.** `fork()` duplicates only the calling thread; a lock held by
  an MKL/OpenMP/torch worker pool in the parent stays locked forever in the child, which then
  hangs on first use. The parent must be quiescent at fork, so inherit only thread-free state
  (imports, parsed pseudos, tables) and let each child spawn and tune its own threads after the
  fork (gradwave already caps threads at import). The stdlib forkserver enforces this by keeping
  the server process minimal and single-threaded.
- **CUDA is not fork-safe.** A forked child gets an invalid CUDA context, so a fork-server is
  CPU-only. That matches the regime, since small cells already belong on the CPU; a GPU warm pool
  would need an in-process persistent worker, not a fork.
- **CPython COW is leaky.** Refcount writes dirty shared pages, so the sharing erodes over a
  long-lived child. A short-lived one-SCF child exits before this matters.

Honest scope caveat. The single largest setup term, the XC `torch.compile`, is partly a
first-ever cost: Inductor caches compiled kernels to disk, so a second cold process with a warm
cache reloads them in seconds rather than the ~1-minute first trace (and on a box without
`openssl` on PATH the compile silently falls back to eager and never caches, see the
torch.compile note). So the durable per-job saving a fork-server banks is the torch import plus
the pseudo parse and table build (seconds each, thread-free, perfectly inheritable), plus
dodging any per-process recompile, on the order of seconds to tens of seconds per job across a
500-point campaign, at zero correctness risk because a fork changes only speed, never the
answer. Next step is a `multiprocessing` forkserver prototype behind `gwq` measuring
jobs-per-hour on a delta-gauge column against the cold-process baseline, before any daemon or
CRIU work.

## Soft-mode deflation for the near-critical response solve

**Status: open, planned (2026-08-06). Near-term; reuses the full shipped response stack. The core
(single-point deflated solve) is ~3-4 weeks and low-risk because exactness is unconditional. Tracked
in #257.**

The self-consistent response solve `scf/implicit.py::solve_adjoint` (and its USPP twin
`postscf/uspp_implicit.py`) solves `u = v̄ + K_Hxc[χ₀ u]`, i.e. `(1 − K_Hxc χ₀) u = v̄`, by
Anderson-accelerated fixed point. Its own docstring records the failure mode: near a spin/structural
instability an eigenvalue of `K_Hxc χ₀` approaches 1, the operator goes near-singular along the soft
mode, plain damping diverges and Anderson stalls (the "NiO lesson"). That is exactly the physically
interesting regime — incipient ferroelectrics, CDW/Peierls, Kohn anomalies, magnetic instabilities —
where phonons, Born charges, and the dielectric response are the deliverable, and gradwave's
differentiability makes those autograd-exact IF the solve stays tractable.

Deflate the soft near-null mode: extract the few offending eigenvectors of `(1 − K_Hxc χ₀)`, solve that
tiny subspace exactly, and run Anderson on the well-conditioned complement, cutting near-critical outer
iterations from hundreds to ~10. Each outer apply is a full Sternheimer sweep, so the win is large, and
the soft subspace moves slowly along an EOS / phonon-stencil / temperature trajectory, so recycling it
across the sweep compounds the saving. Exactness is unconditional — deflation only preconditions the
linear solve, so a wrong subspace costs iterations, never correctness, and the 1e-13 gate is preserved.

The one hard part is that `(1 − K_Hxc χ₀)` is non-normal (a product of two symmetric operators), so at a
true transition the soft mode goes defective (an exceptional point where left/right eigenvectors
coalesce) and clean deflation loses its footing. The fix is to work in the self-adjoint form
`|χ₀|^½ K_Hxc |χ₀|^½` (same eigenvalues, orthonormal eigenvectors), extracting the subspace by symmetric
Lanczos on matvecs.

Build order, each with a kill-gate. (P0) Expose `L(u) = u − K_Hxc[χ₀ u]` as an explicit linear operator
over the existing `apply_chi0` / `apply_k_hxc` matvecs, plus a power-iteration softness diagnostic (the
dominant eigenvalue of `K_Hxc χ₀`); gate on a benign insulator predicting Anderson's observed
contraction rate. (P1) Symmetrize and extract the near-null Ritz pairs by short symmetric Lanczos. (P2,
the go/no-go) Deflated solve inside `solve_adjoint`; gate on a constructed near-critical case (an AFM
transition-metal oxide near its instability, or the xc kernel scaled toward gain→1) converging in ~10
iters where Anderson takes hundreds or diverges, bit-identical to a converged Anderson solve where
Anderson still works. (P3) Recycle the subspace across a sweep. (P4) Confirm autograd
forces/phonons/Born charges are unaffected against `test_dielectric_vs_qe` / `test_phonons`, and land a
near-critical regression test — the missing NiO-lesson test. Metals need the window-pair
partial-occupation χ₀ path to compose with the deflation. Effort: P0-P2 core ~3-4 weeks, P3-P4 another
1-2.

## Plane-wave Green's-function defect embedding (CirculantCore)

**Status: open, planned (2026-08-06). Multi-month research build — a new resolvent / contour /
impurity-SCF solver stack. Front-load the de-risking spike (P0-P1, ~4-5 weeks) as an explicit go/no-go
before committing the full build. Tracked in #258.**

A dilute point defect (vacancy, substitutional dopant, colour centre, qubit host) is solved today only
via a 200-500-atom periodic supercell — at or past the size cliff, requiring size extrapolation against
spurious image interactions, and re-solving perfect bulk hundreds of times to study one local
perturbation. The KKR impurity method avoids this: dress the perfect-host Green's function `G₀` with a
Dyson correction `G = (1 − G₀ ΔV)⁻¹ G₀` restricted to the defect support, where `ΔV` is spatially local
so Woodbury collapses the inverse to an `O(defect-rank³)` solve. Cost is the primitive-cell host solve
(done once, reusable for every defect in that host) plus a small correction, no supercell. Building this
in a plane-wave basis AND keeping it autograd-differentiable is the novel part: the defect formation
energy then carries exact forces and analytic gradients w.r.t. dopant identity / host composition —
differentiable defect design, which no KKR code offers. It removes the size wall exactly (no
truncation-accuracy floor) for the gapped-host defect class.

Two constraints are structural. Metals are excluded — the Friedel screening tail `cos(2k_F r)/r³` is
long-ranged and defeats the local-`ΔV` / low-rank premise, so this is insulators and semiconductors only
(which is exactly the defect / qubit-host class). And a truncated-band plane-wave Green's function loses
the high-energy continuum tail that matters for the local `G₀`, a documented PW-GF pitfall needing a
completeness correction.

Build order is front-loaded to de-risk. (P0, make-or-break) Build `G₀(E)` for the perfect host from a
converged primitive-cell spectral sum and recover the bulk density by contour integration around the
occupied states; gate on the contour density and DOS matching the ordinary SCF density — this validates
the Green's-function + contour machinery and the tail correction BEFORE any defect work. (P1) A
complex-shift resolvent `(E − H)⁻¹` apply, generalising the existing `cg_sternheimer` / `projected_cg`
real-shift solves to complex energy; gate against a dense inverse on a tiny cell. (P2) Woodbury/Dyson
dressing on a frozen local `ΔV`, `δρ` by contour integration; gate against a defect in a large explicit
supercell inside the defect region. (P3) Impurity self-consistency `ΔV ← ΔV[δρ]`; gate on the embedded
formation energy matching a converged supercell reference for a gapped host (vacancy / substitutional in
MgO / diamond / Si). (P4) Autograd through the dressing → `dE/d(dopant)` vs finite difference, then the
differentiable-design demo. gradwave is pure Davidson + density mixing today, so the resolvent solver,
contour machinery, and impurity loop are all new; `scf/implicit.py` (χ₀/K_Hxc, the first-order Dyson
kernel, the IFT adjoint) and the k-mesh machinery are the substrate. Effort: P0-P1 spike ~4-5 weeks (the
go/no-go), full build P2-P4 ~3-4 months.

# Done and resolved

Kept for the reasoning. Each is either landed in the code or settled as a measured negative.

## D3(BJ) dispersion correction (DONE)

Landed as an opt-in, SCF-independent Grimme D3(BJ) correction (#59, #69),
`postscf/dispersion.py`. A real-space image sum over a cutoff (reusing the Ewald image
enumeration and the shift-before-norm double-backward guard) with exponential coordination
numbers, CN-interpolated C6, C8 = 3 C6 √(Q_A Q_B), and Becke–Johnson damping; positions and
cell in Å, energy in eV. Forces are autograd on positions and stress autograd on strain, so the
correction matches the suite's differentiable-energy ethos rather than reimplementing analytic
derivatives. The BJ damping presets for 13 functionals are vendored in `postscf/_d3_params.py`
(regenerated from Grimme's reference data by `scripts/gen_d3_params.py`). Wired opt-in through
`inputs.py` (`DispersionParams`, a `dispersion:` block accepting `true`/`false` or an override
dict), `api.py` (folds E_disp into the reported total and emits a dispersion summary), the
checkpoint energy dict, and the ASE calculator (#69), so ASE-driven relaxations and MD carry
it. It degrades to a no-op on elements without C6 coverage or a missing preset. Self-oracle
tests (`tests/unit/test_dispersion.py`, `tests/unit/test_calculator_dispersion.py`), forces and
stress vs finite differences of the dispersion energy on a rattled low-symmetry cell, an
independent scalar-loop transcription of the energy, a positions gradcheck, the ΣF=0 sum rule,
and the calculator's on-minus-off shift matching the raw dispersion term. D4 was not attempted.

## NLCC core-correction forces, including meta-GGA (DONE)

Ungated the nonlinear-core-correction force term in `forces()` (#64, then #68 for meta-GGA). An
NLCC pseudo adds a frozen pseudo-core charge ρ_core(r−R_I) to the XC argument, whose force is
−∫ v_xc dρ_core/dR_I. Because the suite is autograd-based this is the gradient of E_xc(ρ +
ρ_core(pos)) w.r.t. positions with the SCF density detached, Hellmann-Feynman, stationary at
convergence, so v_xc carries the full LDA/GGA gradient correction automatically. `setup_common`
factors the core density into a position-independent |G| form factor plus a differentiable-in-
positions assembly through the structure factor, reused by both the frozen-SCF build and the
force path. The meta-GGA extension (#68) rebuilds τ_valence from the converged coefficients
(τ_core = 0, matching the SCF's E_xc assembly) and threads it through `xc.energy` and
`spin_xc_energy`; the LDA/GGA path is byte-for-byte unchanged since `needs_tau=False` leaves τ
as `None`. Self-oracle tests match central finite differences of the total energy to ~1e-8
eV/Å (`test_forces_nlcc`, `test_forces_nlcc_metagga`), the latter also guarding the τ argument
against being silently dropped. The USPP / noncollinear meta-GGA NLCC edge stays gated (τ is not
stored on the batched-k / USPP results).

## Magnetic space groups (Shubnikov symmetry) for non-collinear k-reduction (DONE)

**LANDED (2026-07-18).** Every magnetic non-collinear run previously used the full k-mesh, since
`scf_noncollinear` (and the spinor PAW loop) refused `use_symmetry` for any nonzero m⃗, the
existing spglib machinery only knowing the paramagnetic group. The safe choice, not the cheap
one, the FePt MAE runs carried 384 unreduced k-points.

The physics. A finite m⃗ changes the symmetry group. Time reversal dies (no k ↔ −k Kramers
folding, the code already handles the nonmagnetic SOC case where TR survives). And the moment
filters the point group, because m⃗ is an axial vector (transforms as det(R)·R) locked to the
lattice by SOC, so an operation survives only if it maps the magnetization field onto itself.
Operations that flip m⃗ survive only combined with time reversal, the anti-unitary half of the
magnetic (Shubnikov) group, relating band energies at Rk without being a unitary symmetry of H.
For L1_0 FePt (paramagnetic D4h, 16 ops + TR), moments along c leave the unitary C4h (8 ops,
~8× reduction), moments in-plane leave ~C2h (4 ops, ~4×). The easy-axis state is literally more
symmetric than the hard-axis one, the anisotropy seen group-theoretically.

All four phases are in. (1) `magnetic_spacegroup(sg, magmoms, cell)` in symmetry.py, the axial-
vector filter det(S)·S·m⃗ classifying each paramagnetic op as unitary / anti-unitary (op·T) /
dropped, cross-checked against spglib.get_magnetic_symmetry. (2) `reduce_mesh_magnetic`, the
shared orbit fold with unitary {W⁻ᵀ} ∪ anti-unitary {−W⁻ᵀ}, grey group (m⃗=0) reproducing the
paramagnetic+TR fold bit-for-bit. (3) `MagneticSymmetrizer` (grid ρ, m⃗, RhoSymmetrizer maps on
the combined op list + per-op axial 3×3 with s_T=−1 on the anti set) and
`MagneticBecsumSymmetrizer` (BecsumSymmetrizer D^l blocks + the same axial across the Pauli
channels + conj on anti ops). (4) `setup_system`/`setup_uspp` take `magmoms=`, both spinor loops
consume the magnetic system and re-symmetrize (ρ, m⃗[, becsum]) each iteration, and the collinear
loops reject magnetic systems. Measured folds, FePt m∥[001] (6,6,4) 144→30 k (equals the para+TR
IBZ, inversion is unitary for axial vectors), m∥[100] 144→48, bcc Fe m∥z 64→13. Validation
(`tests/unit/test_magnetic_symmetry.py`, `tests/integration/test_magnetic_ibz.py`), SOC FePt
magnetic IBZ ≡ full mesh to 5.0e-11 eV, polar (inversion-broken) FePt exercises the anti-unitary-
only fold (27→6 where unitary ops alone give 9), spinor PAW Si grey group ≡ symmetrized collinear
scf_uspp to 5.1e-11 eV.

The caveat the original plan carried, "do not reduce each orientation to its own IBZ for MAE
differences," turned out wrong for this folding. The magnetic-IBZ sum is exactly the full-mesh sum
re-weighted (measured 5e-11 eV, five orders below the meV signal), so each orientation's k-
discretization error is identical to its full-mesh value and the common-mode cancellation in
E(hard) − E(easy) survives per-orientation folding. The caveat only bites if the two orientations
use different underlying meshes. Keep the same (n1,n2,n3) mesh for both and fold each by its own
magnetic group ([001]→30 k, [100]→48 k at (6,6,4)), 3.7× on the MAE pair, exactness preserved.

## RMM-DIIS solver and whole-step CUDA graph (both TRIED, measured negatives)

Prompted by the GPU profile (dense-LA-bound, 32-46 percent launch gap), two of its three candidate
optimizations were built and measured, and neither pays for small cells.

RMM-DIIS (a `solvers/rmm_diis.py` prototype, since removed) replaces the block Davidson's growing
Rayleigh-Ritz subspace with per-band residual minimization, so it has no per-round eigh and no
m×m subspace GEMM, the 64 percent the profile flagged. It needed two fixes to converge at all, a
units-correct preconditioner (teter_b is a dimensionless filter, right for Davidson subspace
expansion but not for a direct Jacobi step) and an exact line search (Teter-Payne preconditioned
CG, a fixed step does not converge). After both it converges on a fixed operator (synthetic
batched Hermitian, err 2e-11) but in about 100 iterations to the block Davidson's 22, at two
H-applies per iteration. In the real SCF it is worse than slow, on smeared fcc Al it hit the
iteration cap without converging and returned the wrong energy (−368 vs −1828 eV), at 1,548,800
band-applies against Davidson's 10,512 (147×). The reasons are the textbook ones, subspace methods
converge in far fewer iterations, the SCF drives the solver with a loose-early tolerance schedule
a residual method handles poorly, and a metal's near-degenerate bands break the per-band tracking.
RMM-DIIS is a large-system solver (where the O(N³) subspace eigh/GEMM finally dominates) and an MD
warm-start refiner, not a small-cell Davidson replacement, and the dense LA it removes, while 64
percent of GPU time, is cheap in absolute terms at small cell size. Removed the prototype.

The whole-step CUDA graph is blocked upstream, `torch.linalg.eigh` is not CUDA-graph capturable
(it does a host-side info check) and sits in the Davidson inner loop every expansion round, so a
whole-step capture fragments into tiny pieces around each eigh rather than removing the launch gap.
It genuinely needs the eigh out of the hot loop, which was RMM-DIIS's job. So the 32-46 percent
launch gap is real but not reclaimable by either optimization without a solver that avoids eigh
altogether.

UPDATE (2026-07-28). The "fragments into tiny pieces around each eigh" concern was tested directly
rather than assumed. Building on the sync-free Davidson skeleton (branch-free per round, a
prerequisite for capture), a round's post-eigh math (Rayleigh-Ritz combination, residual, Teter
precondition, orthonormalize, restart) was captured per (subspace-dim, n_add) shape and replayed
against real cached-shape recurrences from a real SCF run. It reproduces eager output bit-for-bit,
the fragmentation itself is not the problem, but replays at 1.0× eager speed, same verdict as the
apply-only probe in docs/manual/performance.md. There is no launch gap in this part of the loop
either. The investigation was worth running anyway, isolating every op in a round found
`torch.linalg.qr` on the tall-skinny expansion-direction shape costing more than the Hamiltonian
apply itself on an RTX 3050, which a CPU-offload (mirroring this same eigh's own fix) turns into a
real, measured 1.55-1.67× end-to-end win, see "CUDA batched-QR CPU-offload" in
docs/manual/performance.md. No graphed solver, but a real fix came out of looking for one.

## Local Thomas–Fermi metal preconditioner (DONE)

Landed as opt-in `precond="local_tf"` on both `scf` and `scf_uspp` (`scf/local_tf.py`, default
`"kerker"`). The bare Kerker filter screens charge sloshing with a single length `1/q0`, right for
a bulk metal but wrong for an inhomogeneous cell, where a fixed `q0` over-screens the vacuum.
Following QE's `mixing_mode='local-TF'`, `LocalTFPrecond` lets the screening wavevector track the
local density, `q²(r)=min(q²_TF(r), q0_max²)` with `q²_TF=(4/π)k_F(r)`, capped at the bare `q0` so
a bulk metal is unchanged. It is applied by a short preconditioned-CG solve of the screened-Poisson
operator (a few box FFTs per mixing step, warm-started across iterations), acting on the ρ-total
block only.

Measured (NC, fcc Al, PBE, gaussian 0.1 eV), energies bit-identical to bare Kerker (same fixed
point). Bulk 8×8×8 neutral (9→9), Al(100) slab 21→17 (4 layers) and 27→21 (6 layers) iterations,
the margin growing with cell inhomogeneity, exactly the regime the operator targets. So the
original framing was right that a fixed Kerker is the wrong operator away from a uniform bulk, but
the win is on slabs and molecules, not on a homogeneous bulk metal, where Kerker at a sensible `q0`
is already near-optimal. The bulk-Pt 16-vs-7 iteration gap is therefore a starting-density and
Broyden-history question more than a screening-length one, and this preconditioner does not by
itself close it. Unit tests pin the three operator limits (`tests/unit/test_local_tf.py`),
integration tests gate the fixed-point invariant on NC and USPP
(`tests/integration/test_local_tf_scf.py`), and the slab iteration-count win lives in
`benchmarks/bench_precond.py`.

Two follow-ups. Building this surfaced and fixed a separate bug, `setup_uspp` sized the FFT box as
a blanket cube for any symmetric cell, so an anisotropic slab got a 105³ box instead of 20×20×105,
a 27.6× over-allocation that OOMs during setup, now fixed by porting the NC path's symmetry-coupled
axis grouping (`symmetry.coupled_axis_groups`). And the modern parameter-free successor to local-TF
is the LDOS preconditioner of Herbst and Levitt (arXiv:2009.01665, DFTK's default), which adapts
the screening to whether each region is metallic or insulating from the local density of states
rather than a Thomas–Fermi model. If local-TF ever underdelivers on a strongly mixed metal-vacuum-
insulator cell, that is the next rung, and it reuses the same reciprocal-space mixing hook.

## torch.compile for the exchange-correlation layer (DONE)

Landed as the opt-in `compile_xc` flag (`GradWave(compile_xc=True)` or `xc.enable_compile()`).
Measured 19× forward and 16× forward-plus-`v_xc` at 64³, `v_xc` bit-accurate to 3e-16, with an
eager fallback for the missing NixOS toolchain. Compiled aot_autograd cannot double-backward, so
the `f_xc` response and HVP sites wrap their `xc.energy()` in `xc_eager()` to stay eager, which
means only the forward and first-order `v_xc` legs accelerate. Details in
`docs/manual/performance.md`.

The original analysis, kept for the reasoning. The compiler is dead on the complex, FFT-bound
Hamiltonian apply, which two earlier attempts confirmed, but the real-valued XC functional was
never isolated and compiles well on a 64^3 grid. The end-to-end effect on a plain SCF is only a few
percent because XC is a minority of runtime and its FFT-based gradient assembly does not compile,
but learned-XC training, the PAW one-center angular loop, and the `f_xc` response HVPs call the XC
transcendental chain far more than once per iteration and are CPU-bound, so those are the real
targets. Insertion point is the single `XCFunctional.energy_density` choke point, opt-in with an
eager fallback for the NixOS toolchain gap.

## Dual FFT grid (DONE)

Landed as commit `71a5265`, about 2× on the USPP/PAW H-apply FFT by running the smooth
wavefunctions on a coarse grid and the augmentation on the dense grid, matching the audit spec.

## CheFSI, benchmarked no-go on the RTX 3050 (DONE)

Chebyshev-filtered subspace iteration is in `solvers/chebyshev.py`, unit-tested and wired opt-in as
`scf(..., eigensolver="chebyshev")` on the NC collinear path, bit-identical to Davidson on the real
NC SCF regression. The noncollinear spinor twin was tried but left unwired, CheFSI converges too
slowly on the dense metal spinor spectrum (100-iteration cap vs Davidson's 18). The RTX 3050
fp32-deep benchmark found it 2.5 to 5× slower than Davidson at every grid size that fits in 6 GB, up
to 35^3. The fp32 FFT advantage there is only about 3.4×, not the 12× the larger systems would need,
and CheFSI does 2 to 3× more H-applies, so the filter loses. It stays opt-in and off by default.
Revisit on a bigger card where the grid can grow into the regime where the fp32 FFT gain dominates,
the same hardware caveat the scaling section opens with.

The H100 session (#206, round 5) closes the revisit permanently. At datacenter fp64 the fp32 FFT
advantage the revisit was waiting for is gone, since fp64 tensor cores run the standard kernels
directly and the extra H-applies have nothing to gain. Chebyshev is 2x slower than davidson at 10
atoms and 13x slower at Si-128 (15.316 vs 1.165 s/iter, identical energy), and the gap widens with
cell size with no crossover anywhere in 10 to 216 atoms. Davidson stays the unconditional default on
every device class.

## Batched Davidson conditioning guard, cond-SVD removed (DONE)

The k-batched USPP/PAW generalized Davidson computed a full `linalg.cond` of the subspace overlap
every round on top of the `cholesky_ex` it already ran. Probing a low-ecut Si PAW SCF (8, 10, 12 Ry)
showed the overlap tips into non-PD, which `cholesky_ex` flags with info>0, long before its
condition number nears the 1e14 trip (max observed ~9e7), so the SVD never fired independently and
was pure cost. Removed it. Batched-vs-per-k equality (identical eigenpairs) and USPP/PAW-vs-QE
regression still pass, including nspin=2 PAW (O2 triplet, 7e-12 eV). Recorded in
`docs/manual/wisdom.md` under Eigensolvers.

## Extended-xyz trajectory output for relax (DONE)

`run_relax` accumulates an ASE frame per optimizer step with energy and forces frozen on a
`SinglePointCalculator`, and `run` writes them to `relax.xyz` (extxyz) next to the JSON, re-readable
in ovito or the ASE gui. The relax CLI returns exit 0 on normal completion, since reaching the
ionic-step limit still yields a valid trajectory, with convergence carried by the JSON
`relax.converged` flag. Regression in
`tests/integration/test_io.py::test_relax_writes_extxyz_trajectory`. MD does not have an output path
yet, so the same frame accumulation extends there once it lands.

## Atomic-orbital seeding for the initial wavefunctions (TRIED, no net gain)

The idea was to hand the first Davidson solve a superposition of pseudo-atomic orbitals instead of
bare lowest-kinetic plane waves. `scf/loop.py` builds `c0` as an identity block on the first `nb`
sphere entries, the smoothest plane waves and nothing about the atoms, poor enough that the loop
runs the first diagonalization at a loose `1e-3` tolerance before tightening. QE's default instead
projects the atomic pseudo-wavefunctions onto the plane-wave basis (`startingwfc='atomic'`). All the
pieces existed in-tree, the `upf.pswfc`/`paw.chi` orbitals and the SBT-and-Ylm projector build
shared with the KB, Hubbard, and PDOS paths.

Built `lcao_seed` (per-k atomic-orbital block, QR-orthonormalized to 8e-15, padded with plane waves
past the orbital count) and wired it at the `c0` site. It reaches the plane-wave-seeded energy to
machine precision, as it must (NC O2 gives dF = 5e-12 eV, fcc Ni gives dF = 3e-11 eV). The predicted
one-to-three iteration saving is real but small (O2 goes 28 to 26 iterations, fcc Ni 6×6×6 goes 12
to 12), and the per-k seed build costs enough that wall time came out neutral to slightly worse (Ni,
108 s to 122 s). The reason is the one the prediction named, the loop already runs the first
diagonalization at a loose 1e-3 tolerance, so a crude plane-wave start converges the cheap early
eigensolves fine, and the total SCF count is set by density mixing, not initial-orbital quality.
Reverted the wiring rather than add per-k overhead to the default path for no measured gain. Recorded
in `docs/manual/wisdom.md` under SCF and mixing.

The remaining reason to revisit is that it composes with CheFSI, whose convergence rate depends
directly on how much of the wanted subspace is already in the start. A Chebyshev filter fed atomic
orbitals needs fewer rounds than one fed smooth plane waves, so the pair should be measured together.
That is the only configuration where the seed cost might be repaid, and it is worth building
`lcao_seed` back only alongside a CheFSI-default benchmark that shows the compound win.

A separate revisit of the starting *density* seed (ρ0, not the wavefunctions) is recorded in
the next entry.

## Atomic-orbital seeding, the density channel (TRIED, no-go as default)

**Status: TRIED 2026-07-30, no-go as a default, an opt-in rescue identified. Archive branch
`research/ao-density-seed`.**

The wavefunction-seeding no-go above covered the Davidson `c0`. This revisits a different
object, the starting density ρ0 the first potential is built from. The audit came back
cleaner than expected. The seed is already a superposition of atomic densities from the UPF
`PP_RHOATOM` table, one builder shared by both formalisms (`scf/guess.py::sad_density`, wired
at `loop.py` and `uspp_loop.py`), normalized to exact electron count. So the only live lever
was the nspin=2 spin-split shape, and the charge channel has no headroom, matching the
wavefunction result.

The change under test shapes the seeded magnetization by the `PP_PSWFC` d-orbital density
instead of the uniform `(1±m)/2` split, at the same per-atom moment. It is a systematic win
on one system and a disqualifying regression on another.

- fcc Ni PAW near the Stoner boundary is the win. The default seed converged 0 of 4
  `start_mag` values at `rhotol` 1e-5, stagnating near the FM fixed point at 0.02, 0.05, and
  0.30 and landing on the wrong NM branch (+77 meV) at 0.10. The d-localized seed converged
  3 of 4, always to the FM branch.
- bcc Fe PAW is the regression. At `start_mag` 0.02 the d-localized seed collapses Fe to the
  nonmagnetic branch, +660 meV above the FM ground state the default reaches from the
  identical seeded moment, and it costs 2 to 6 extra iterations at 0.05 and 0.10. fcc Ni NC
  is neutral either way, within one iteration, and the seed build cost is neutral.

The fixed-point oracle held everywhere both seeds reached the same branch, F agreeing to
2e-11 eV or better, so the seed changes only the trajectory. The verdict is no-go for the
default, because a default cannot trade Fe robustness for Ni convergence. The d-localized
shape is worth keeping as a future opt-in rescue (a `start_mag_shape="d"` style option,
prototyped only as the archive branch's monkeypatch, not wired into inputs) for
marginal-Stoner PAW systems that stagnate under the default.

One diagnostic lead is worth its own line. The Ni PAW default-seed stagnation at 120
iterations at every `start_mag`, including the comfortable 0.30, localizes the residual floor
to the mixer's magnetization channel, not the seed or the charge channel, since a different
magnetization shape at the identical moment converges fine.

# Acceleration idea rounds: evaluated and shelved (2026-08)

Three structured idea-generation-and-vetting passes (agent-workflow ideation, then literature
plus codebase compare plus adversarial verification) ran on "what else could accelerate
gradwave." They are catalogued here so the directions are not re-litigated. The recurring lesson
is that the single-SCF fp64 small-cell axis is largely exhausted, and the live headroom is
campaign-level, not kernel-level. Verdicts are the vetting output, not commitments. Anything
that graduates to real work gets its own open entry above (the fork-server already has).

**Round 1 (single-SCF numerical / solver methods; all researched, none a speed win in this
regime).**
- fp64 emulation (double-single / Ozaki tensor-core): NOT-HELPFUL. double-single is 44-48 bits
  (fails the 1e-13 gates); Ozaki reaches fp64 but only pays at m>=8192 (gradwave npw <=6746) and
  needs custom kernels; Amdahl-capped ~1.2-1.35x wall, consumer-GPU-only, moot on CPU/H100.
- Stochastic DFT (stochastic-orbital density): NOT-HELPFUL. Crossover is thousands of atoms;
  force noise floors ~0.1 eV/A even with variance reduction, disqualifying for autograd forces
  and the sub-meV gates.
- Direct energy minimization (ensemble-CG): NOT a speed win (large-N story), MARGINAL for
  robustness (monotonic descent structurally avoids charge-sloshing / Stoner collapse; substrate
  exists in `opt/joint.py` + `opt/newton.py`; net-new is differentiable ensemble/MV occupations).
- Sketched Rayleigh-Ritz: NOT-HELPFUL. A dressed-up fp32-RR; the sketch dimension reaching 1e-9
  exceeds npw, so the savings invert.
- s-step / communication-avoiding Davidson: MARGINAL to NOT-HELPFUL. The CheFSI/LOBPCG
  more-applies-to-cheapen-RR trade minus the MPI-latency regime it was built for; also drives the
  n>32 eigh cliff.
- Proposed, not deep-researched: Fermi-operator-expansion density matrix (reuse KPM), Shirley
  optimal-basis k-interpolation, stick/pruned FFT (needs kernels), meta-learned SCF map,
  adaptive/multi-fidelity BZ integration.

**Round 2 (mixing / eigensolver / conditioning micro-opts; 0 pursue, 2 marginal, 7 dead).**
- Quasi-Newton mixer-Jacobian transfer across scan steps: MARGINAL. ~1.3-1.7x fewer iters/step on
  fixed-cell magnetic/metal scans, but collides with the "early garbage secant pairs poison the
  inverse Jacobian near Stoner" negative (wisdom.md); probe-gate before building.
- Occupation-aware adaptive band budget: MARGINAL. Only the empty-headroom per-band gate survives
  (stop stragglers holding `rn.max()` hostage); a few percent CPU; drop the pruning and buffer-ramp.
- Dead: charge-block / codensity + delta-mu preconditioner (re-runs the #232 learned-multipole
  negative, DIIS absorbs the smooth charge response); smearing / electronic-temperature
  continuation ("the mixing scheme sets the metal iteration count, the smearing kernel does not");
  FFT arena + Morton layout (the arena half already ships in `BatchedHamiltonian`); cross-SCF
  invariant-subspace deflation (destroys the uniform `(nk,nb,npw)` batch); cross-k Taylor
  subspace-transport warm start (batching-hostile, first-iteration-only); one-center becsum
  local-response preconditioner (phantom bottleneck, Johnson conditions it natively);
  Harris-Foulkes trust-region line search (step-size control, an archived negative; bare HF is
  non-variational near convergence).

**Round 3 (lateral: architectural / systems / surrogate / campaign / deliverable; 0 dead, four
live axes). These are DEFERRED / OPEN, not rejected.**
- Warm-process setup amortization (fork-server / daemon / CRIU): PROMISING, the quick win, now its
  own open entry above.
- Delta-tier (multifidelity delta-learning, cheap base + learned smooth correction, oracle-audited):
  PROMISING; the uncertainty/audit gate is the research, not the plumbing.
- GreedyManifold (reduced-basis POD-Galerkin over a sweep): PROMISING but research-project
  (nonlinear + non-affine parameter dependence needs EIM/DEIM); 5-20x campaign potential and the
  reduced model is itself a differentiable surrogate.
- Continuation Sweep (analytic drho/dlambda tangent drives EOS/elastic, read C_ij/B0/Gruneisen as
  exact derivatives) and GradTrust (exact-gradient trust-region inverse design): INTERESTING, both
  gated by the same missing piece, the metallic Fermi-surface response and strain/dR seed in
  `scf/implicit.py` (insulator-only today). That response machinery is the highest-leverage single
  enabler across the whole round.
- RESPA-SCF (multirate: hold the ACE Fock on a slow clock for hybrids): INTERESTING, 2-4x on a
  hybrid SCF at an unchanged fixed point; the multirate stride is the win, the response-certifier
  the least-justified part.
- HessNet (emulator supervised on exact Hessians / response tensors external MLIPs cannot supply),
  Density Git (content-keyed converged-state store with equivariant transport), Control-Variate /
  MLMC campaigns (only for true population expectations), Backprop-the-Budget (per-knob error
  allocation, but the repo's own per-axis attributions are unreliable), Speculative trajectory
  prefetch (capped at the line-search regime): INTERESTING to bounded.
- InverseFlow (amortized generative designer): LONGSHOT, gradwave is the worst-positioned engine to
  generate the training corpus. MigrateSCF precision-migration half: net-negative on this hardware
  (fp32 GPU draft measured 0.35x); only its warm-pool half survives.

**Round 4 (fresh lenses: cheaper-physics inner accelerators, active campaign control, compression,
exotic hardware, systems borrowings, researcher velocity). The cheaper-physics-inner-accelerator
theme largely did not pay off; the built-est ideas were campaign-layer, and the MOONSHOTS are the
ones flagged worth revisiting.**

CONSIDERED AND DEFERRED (moonshots, revisit — bold, unbuilt, wall-breaking-by-construction, not
rejected):
- QTT-Orbitals: represent Kohn-Sham orbitals as quantized tensor trains with a QTT-FFT H-apply,
  compressing storage O(npw) to O(r^2 log npw). The one idea that breaks the fp64-tax and size walls
  BY CONSTRUCTION rather than around them. Deferred: ranks explode on metals' oscillatory
  Fermi-surface states, the 1e-13 gate forces near-dense truncation tolerances, and differentiable
  complex128 TT-SVD/rounding gradients are ill-conditioned at near-degenerate singular values and
  unproven. Revisit for smooth insulating large cells.
- Wannier Courier: run downfolding in REVERSE, Slater-Koster-scaling a transported Wannier/TB
  Hamiltonian to regenerate a seed density across a geometry scan, targeting the large-jump regime
  where local density extrapolators break. Deferred: needs a full MLWF/downfolding subsystem plus a
  reverse WF->PW-density map (the Wannier interface is itself an open capability gap), and rigid-Wannier
  translation fails for metals' entangled bands.
- Conformal Scout Cascade: a cheap in-code scout (LDA / Gamma-only / tiny-ecut) emitting BOTH a warm
  density AND a conformally-calibrated deliverable band, gating the exact solver at a controlled
  abstention-error ceiling. Deferred: marginal conformal's coverage needs exchangeability, which
  chemistry-to-chemistry campaigns violate exactly on the decision-relevant hard tail; needs
  Mondrian/covariate-shift conformal, which is the research content.
- CompressedTape: low-rank/HOSVD compression of the retained adjoint state to raise the
  differentiable-regime (learned-XC / inverse-design / Hessian) cell-size ceiling. Deferred and
  premise-flagged: gradwave's adjoint is implicit-function-theorem based (one fixed-point linear
  solve), NOT differentiation through an unrolled SCF trajectory, so there is no long activation tape
  to compress; the real low-rank structure is spatial/Wannier, needing an on-the-fly localization.
- GammaPrime (the cheaper-physics-theme survivor): repurpose a converged SCC-DFTB gamma-matrix as a
  connectivity-aware dielectric PRECONDITIONER, meta-learning its Hubbard-U through downstream PW
  iteration count. Deferred: needs a whole SCC-DFTB engine + Slater-Koster tables absent for the
  metal/f-element campaign, and gamma is a long-wavelength monopole response overlapping the Kerker
  regime already shipped.

Evaluated, campaign/settings layer (mostly-built, lower novelty, deprioritized in review):
- Converge-to-Answer: goal-oriented auto-convergence driving ecut/kmesh/smearing from an a-posteriori
  error target on a DERIVED observable, emitting a per-run error certificate; the estimation half
  already ships (discretization_error.py / convergence_error.py / stress_error.py), only the
  escalation controller is missing. Real ~5-10x-fewer-exploratory-SCF potential, but the estimators
  are calibrated indicators not bounds and certify distance to E_KS_converged, never to reality.
- Hull-Chasing: per-point convex-hull distance as the SCF convergence-DEPTH knob (distinct from
  published sample-acquisition active learning). ~1.3-1.7x campaign trim; references must stay tight.
- SplitFinish: fp32-GPU bands/DOS/PDOS overlapped against the fp64-CPU SCF of the next structure. Small
  real win on property campaigns only (its phonon example is physically wrong). RussianRoulette-SCF:
  unbiased randomized SCF-tail truncation, out-competed by the repo's own deterministic geometric
  extrapolator except on multi-mode tails.

CLOSED: cheaper-physics SEED channel (Huckel Ignition). The AO predecessor (lcao_seed) was built,
measured 0-2 saved iterations at neutral-to-worse wall time, and reverted; by Rayleigh-Ritz invariance
a Huckel rotation WITHIN the AO block converges identically to the raw-AO block already tried, and the
outer count is mixing-limited not orbital-quality-limited. The seed channel is structurally closed; the
preconditioner reframing (GammaPrime) is the only live variant of the theme.

**Round 5 (MOONSHOT round: wall-breaking-by-construction reformulations). Two ideas PROMOTED to full
open-backlog entries above; the rest recorded here.**

PROMOTED (see the Performance and scaling entries above, with phased plans and spike issues):
soft-mode deflation for the near-critical response solve (best payoff-to-tractability ratio, near-term,
exact-at-convergence), and CirculantCore plane-wave Green's-function defect embedding (the boldest EXACT
size-wall break, multi-month).

CONSIDERED AND DEFERRED (credible mechanism, harder blockers):
- ShatterDFT: differentiable adaptive-local-basis Discontinuous Galerkin (DGDFT). Dissolves the size
  wall structurally (block-sparse H, tens of adaptive local basis functions/atom, metal-accurate) but
  trades the 1e-13 gate for a ~1 meV DG-truncation floor and needs a block-sparse / selected-inversion
  forward solve reconciled with eager autograd (the IFT adjoint is the escape). The boldest size-wall
  break, but a large new research code.
- Mermin thermal-annealing homotopy: track the KS fixed point as a temperature-continuation ODE with
  (1 − χ₀K) as the exact analytic tangent, so no near-singular cold dielectric is inverted (breaks the
  ~2.3x-QE metal iteration penalty). Blocker: crossing a real electronic transition on the cooling path
  can track the wrong branch; safe floor is degrade-to-the-shipped-annealing-list. Shares the soft-mode
  machinery with the promoted deflation entry.
- SeparableResponse (THC-χ): extend the shipped ISDF/THC substrate to the response manifold with an
  imaginary-frequency quadrature for a cubic-scaling, autograd-differentiable RPA correlation energy.
  Blocker is empirical (THC-χ rank vs accuracy on metals), not theoretical; an extension of validated
  code and the natural precursor to a differentiable GW/BSE.
- Longshots with real mechanisms: RiemannWave (learned adaptive-coordinate plane waves), BetaBeam (NUFFT
  nonlocal projector apply), Resolvent-Flow FEAST contour projector, DielectricNet (learned short-range
  dielectric preconditioner via the mixing hook), FieldGalerkin (neural-field coarse space with an exact
  PW corrector), Parareal-SCF (parallel-in-imaginary-time), Resonate (analog coupled-oscillator inner
  loop). Each names a hard blocker: custom kernels, PW-uncompetitive contour solves, or hardware years
  out.

REJECTED: HierFock (hierarchical / butterfly exact exchange) — redundant, the ISDF+ACE hybrid stack it
would replace is already shipped and validated to 1e-13, and butterfly has lost to ISDF in practice.

NEW (2026-08-17, moonshot-vetted — response/adjoint composition-design family):
- ResponseDesign (dε∞/dλ, dZ*/dλ): analytic mixed 3rd-order composition×field derivative via one nested
  adjoint composing the shipped alchemical χ₀ (scf/alchemical.py, PR #301) with the field DFPT
  (postscf/dielectric.py). Gradient-descend composition to a target permittivity / Born charge — a
  VCA-IMPOSSIBLE capability (Z* is a discrete-atom projector response the averaged VCA potential cannot
  represent). Both first-order solves already ship on one autograd graph; the mixed derivative composes
  existing matvecs — no new physics or solver change. De-risk: dε∞/dλ on SiC→C vs the existing FD ground
  truth to ~1e-3 rel. Flags: NO "stopgap" is being replaced (sell as new capability); DROP piezo (strain
  DFPT absent from src). Highest-ROI moonshot bet — most machinery shipped, cheapest validation.
- RelaxDesign (relax-consistent dO/dλ): dO/dλ = ∂O/∂λ − ∂O/∂R·H⁻¹·∂F/∂λ, chaining the alchemical force
  response through the inverse Γ-Hessian (postscf.phonons.gamma_hessian) for relaxation-dominated
  observables (polar-mode amplitude, adsorption geometry) the frozen-ion gap gradient gets structurally
  wrong. Needs 2 near-existing pieces: the mixed alchemical-force response ∂F/∂λ (absent) and ∂O/∂R
  (buildable from the ∂V/∂R displacement DFPT). Reuses ResponseDesign's nested-adjoint harness; build 2nd.
- MagnonSusceptibility (χ⁺⁻(q,ω) poles): magnon dispersion + differentiable spin-wave stiffness dD/dparam
  from the finite-q TRANSVERSE dynamical spin susceptibility, in the (m_x,m_y) noncollinear channel —
  builds on the new q≠0 DFPT stack (postscf/dfpt_q.py) + _fxc_hvp_noncollinear. An INDEPENDENT route to
  the spin-wave spectrum vs the shipped classical J/D/K→MC path; same-process cross-validation of the
  susceptibility- vs J-derived stiffness is the novel deliverable. Blocker: needs a NEW DYNAMICAL
  Sternheimer (complex shift (H−ε±ω−iη), non-Hermitian resolvent — TDDFT/BSE-tier build) + the classic
  LSDA Goldstone-violation pathology (spurious Γ gap unless χ₀ and the xc kernel are grid-consistent).
  GATE on a bcc-Fe/CrI3 acoustic-magnon ω→0 (Goldstone) check before any real investment. High ceiling,
  most directly on the response/adjoint edge.

**Round 6 (MOONSHOT round: "resurrect & re-cross the crossover" lens). Two survivors VALIDATED by
prove-or-kill; #1 deferred to a future PR, #2 pocketed as a niche extension.**

The lens: every settled plane-wave-DFT choice was decided at a crossover for a regime (one large
fp64 CPU cell, non-differentiable). gradwave's regime — tiny batched cells, GPU tensor cores,
dispatch-bound, autograd-native — is none of those. A dense-GEMM/Toeplitz re-cross works ONLY when
N_grid contracts out (operator output lands back on the npw sphere); operators with grid-valued
output (density build, Hartree, XC, τ-build) do NOT re-cross. The shipped local-potential Toeplitz
apply (PR #316) is the proof-of-concept; these extend it.

DEFERRED — the real frontier (insulators; pick up when returning to the response/phonon layer):
- **Resolvent-Sternheimer for small-insulator DFPT.** Resurrect 1990s-abandoned dense sum-over-states
  response: ONE per-k eigh of the fixed ground-state H (reusing the Toeplitz dense-H assembly), then
  δψ = Σ_c U_c (U_c†·rhs) / (Λ_c − ε_n) for every band/direction/perturbation — GEMM back-substitution
  replacing the unfused iterative cg_sternheimer. The response layer is simultaneously dispatch-bound
  (~1e5 launches/phonon-q), operator-reuse-rich (one H fixed across 3N×n_DFPT right-hand sides), and
  un-reformulated. VALIDATED forward (resolvent_sternheimer.py): δψ matches cg_sternheimer to 4–7e-11;
  the warm-CG kill-risk is REFUTED (CG to absolute tol 1e-8 needs ~all its iters from any warm start,
  so warm is only ~10–20% faster than cold across outer-step sizes 20%/2%/0.2%); crossover N to
  amortize the eigh is a robust 1.8/4.1/9.7 for Si npw 537/588/960, so at a realistic ~60 shared solves
  the resolvent is 20×/8.5×/4× on the Sternheimer wall → ~2–4× whole-phonon-DFPT. Scope: INSULATORS
  only (metals put the resolvent pole in the continuum) and small cells (eigh O(nk·npw³) grows N with
  npw). Remaining unbuilt piece: the differentiable backward (dω²/dparam through DFPT) needs a custom
  autograd.Function that re-applies the resolvent instead of differentiating the degenerate eigh at
  high-symmetry Γ — the non-differentiated phonon calc (the common product) does not need it. This is
  where the crossover has moved MOST: the gradient/response machinery a differentiable DFT code exists
  to run, not the forward SCF apply (already re-crossed and shipped).

POCKETED — niche extension of the shipped apply (build behind the size gate + a meta-GGA-active check
whenever an r2SCAN/SCAN campaign wants it):
- **Meta-GGA τ-operator as one weighted-Toeplitz GEMM** Ṽ[i,j] = ½·((k+G_i)·(k+G_j))·v̂_τ(G_i−G_j),
  collapsing the 6-FFT/band round-trip in core.metagga.metagga_tau_operator. VALIDATED (tau_toeplitz.py):
  Ṽ@c matches the operator to 7e-16, apply-level 10.7×/54.7×/11.0× (bigger than the local apply — 6 FFTs
  deleted vs 2). But meta-GGA-gated (niche functional) → regime-weighted ~1.03×. The τ-BUILD stays FFT
  (grid-valued output, does not re-cross).

REJECTED with data (this round and prior): dense-eigh forward-solve (O(npw³) all-band loses to warm
Davidson), truncated/banded M (v̂ decays too slowly to sparsify), GPU whole-SCF on consumer hardware
(mixed-precision regresses + non-apply fp64 FFTs dominate; crippled consumer fp64), density-build /
Hartree Toeplitz (grid-valued output — does not re-cross).
