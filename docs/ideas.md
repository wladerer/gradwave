# Ideas and future work

A running backlog for gradwave, with enough reasoning attached that each item
can be picked up cold. Not commitments, just directions worth taking. The open
backlog comes first, then a "Done and resolved" section that keeps the reasoning
for items already built or settled.

# Open backlog

## Scaling up: RI and tensor hypercontraction

Framing first, because it reframes the GPU results in the done section. The
CheFSI no-go, the EOS-batching analysis, and the 128-atom memory cliff were all
measured on a consumer RTX 3050, whose fp64 throughput is a small fraction of its
fp32 and whose 6 GB caps the grid. Those numbers bound that card, not GPUs in
general. A datacenter card with real fp64 changes the CheFSI arithmetic story on
its own, before any code change. But the more durable lever is to cut the
operation count itself, which helps on the CPU path and on any GPU, and that is
what resolution of identity and tensor hypercontraction do. They are also the
enabling substrate for exact exchange, the biggest single physics gap in the
code, so the scaling work and the accuracy work are the same work.

### Resolution of identity (RI, density fitting)

RI expands products of orbitals in an auxiliary basis so a four-center
electron-repulsion object factorizes through two- and three-center intermediates.
In a plane-wave code the Hartree term is already O(N log N) through the FFT, so
RI is not a Hartree win. Where it pays is exact exchange. The Fock term is the
O(N^4) bottleneck that keeps hybrids out of the code, and RI on the orbital pair
densities `rho_ij(r) = psi_i*(r) psi_j(r)` is the standard route to make it
affordable. So RI is not a standalone feature, it is the substrate that makes the
biggest missing physics piece, hybrid functionals, tractable.

- The auxiliary representation. A plane-wave code already carries a complete
  auxiliary basis in the dense-grid plane waves, so a pair density is exact on the
  grid. The cost problem is the number of pairs, O(N^2) of them, each needing an
  FFT to Coulomb-couple. RI proper compresses that, and its plane-wave-native form
  is the ISDF factorization below.
- The differentiable angle. The fit is a linear solve against a metric, which is
  differentiable end to end, so a learnable hybrid could carry the exchange-mixing
  fraction and the range-separation length as trained parameters on top of an
  RI-compressed Fock build.

### Tensor hypercontraction and ISDF

Tensor hypercontraction factorizes the pair-product tensor into a small set of
interpolation points and interpolation vectors, so an object that is O(N^2) in
orbital pairs and O(N_grid) in real space collapses to O(N) points times a
compact factor. The plane-wave-native form is interpolative separable density
fitting (ISDF, Lu and Ying), which writes `psi_i(r) psi_j(r)` approximately as
`sum_mu zeta_mu(r) psi_i(r_mu) psi_j(r_mu)` over a small chosen point set
`{r_mu}`. A QR-pivoted or centroidal-Voronoi point selection makes the rank grow
like N rather than N^2.

- Why it is the right scaling lever. It cuts the FLOP count of the exchange and
  correlation builds directly, so unlike the CheFSI fp32 story it does not depend
  on a particular card's fp64 throughput. It helps on the CPU path and on any GPU.
  That is the durable answer to "scale up", attack the operation count, not only
  the hardware.
- What it unlocks. ISDF is the standard enabling technique for affordable exact
  exchange and RPA correlation in plane-wave codes (Qbox, PWDFT, and the ISDF-K
  line of work). With ISDF in place a hybrid functional and an RPA correlation
  energy both become reachable, which is the jump from a very well validated GGA
  code to one that does electronic structure GGA cannot.
- Build order. Land ISDF first as a compression of the pair densities with a
  QR-pivoted point selection, validate the compressed Fock exchange energy against
  a direct plane-wave Fock build on a small molecule to milli-eV, then layer the
  learnable-hybrid parameters and, separately, the RPA correlation contraction.
  Each stage is a set of tensor contractions, so it stays inside the
  differentiable-by-construction design.
- The honest caveat. ISDF has its own accuracy knob, the interpolation-point
  count, and the point selection is the subtle part. Budget the validation against
  direct Fock, not against another approximate method, and treat the rank as a
  convergence parameter reported alongside the result.

**First cut LANDED (2026-07-20), single-k (Γ).** `postscf/isdf.py`: the
pivoted-QR interpolation-point selector (`select_interpolation_points`, exact
pair matrix for small orbital sets, randomized Khatri–Rao sketch above a
threshold), the `M S⁻¹` fit (`build_isdf`), the interpolation-vector Coulomb
coupling (`_coulomb_coupling`, reusing the `hartree.py` G=0-excluded 4πe²/G²
kernel), and the fully-contracted exchange energy
`E_x = −½ Σ_μν W_μν |D_μν|²` (`ISDFExchange.energy`). Validated against a direct
O(N²) pair-FFT Fock build (`exchange_energy_direct`) — the build-order gate
above: at a rank past the co-density space ISDF ≡ direct to machine precision
(synthetic complex orbitals, `tests/unit/test_isdf.py`; converged Γ Si,
`tests/integration/test_isdf_vs_direct.py`), and below saturation the exchange
error falls monotonically with the rank (8-atom Si measured 6.4 eV → 0.24 eV →
1e-14 as n_μ → 40 → 80 → 136, the 16·17/2 real-orbital co-density rank), so the
rank is the accuracy knob exactly as the caveat asks.

**Exchange operator + ACE LANDED (2026-07-20), single-k (Γ).**
`postscf/exchange.py` builds on the factorization to give the pieces a hybrid
SCF actually needs — the *operator*, not only the energy. The direct Fock
operator `V_x φ = −Σ_j ψ_j v[ψ_j*φ]` (`exchange_operator_direct`, the O(N_occ²)
pair-FFT reference); the ISDF-accelerated operator on the occupied set
(`exchange_operator_isdf`, `V_x ψ_t = −Σ_μ v[ζ_μ] B(r,r_μ) ψ_t(r_μ)`, one
Coulomb solve per interpolation vector instead of N_occ² pair solves); and the
adaptively-compressed exchange (`build_ace`, Lin Lin JCTC 2016): the Cholesky of
the exchange matrix gives a low-rank `V_x ≈ −Σ_k |ξ_k⟩⟨ξ_k|` that is *exact* on
the occupied subspace, the object a generalized-KS Hamiltonian would carry.
Validated on converged Γ Si and synthetic orbitals (`tests/unit/test_exchange.py`,
`tests/integration/test_exchange_operator.py`): the operator energy equals the
contracted energy build to machine precision; the ISDF operator saturates to the
direct operator (rel-err 2e-2 → 1e-13 as the rank fills); and ACE reproduces
`V_x ψ_n` on every occupied n to ~5e-14.

**Multi-k exchange + range-separated kernel + learnable hybrid slot LANDED
(2026-07-20).** `postscf/coulomb_kernel.py` and `postscf/exchange_multik.py`:
- `coulomb_kernel` — the range-separated exchange kernel K(q+G) in three modes:
  bare `full` (PBE0-style), `short_range` (erfc, HSE), `long_range` (erf), ω
  differentiable. The erfc/erf split satisfies K_sr+K_lr=K_full at q+G≠0; at
  q+G=0 the screened kernel is *finite* (π e²/ω²) while full/long-range drop the
  divergence, so **screened (HSE) exchange needs no singularity correction** —
  the reason to reach for it first (`tests/unit/test_coulomb_kernel.py`).
- `multik_exchange_energy` — the full exchange over a k-mesh through the
  co-density crystal momentum q = k′−k, kernel evaluated at |q+G|²; reduces to
  the Γ direct build at one k-point (measured exact) and gives a finite real
  exchange on a 2×2×2 full-BZ Si mesh. Requires an unreduced mesh
  (`use_symmetry=False, time_reversal=False`) and the dense grid, both enforced
  by convention (`tests/integration/test_exchange_multik.py`).
- `HybridExchangeParams` / `hybrid_exchange_energy` — the learnable slot: α
  (sigmoid) and ω (softplus) mirroring `core/xc/learnable.py`, initialized to an
  HSE-like screened hybrid. The exchange energy is differentiable in both
  (autograd dE/dω matches finite difference to 1e-9), so a learned hybrid trains
  the mixing and range end to end.

**Self-consistent hybrid SCF + ISDF-K LANDED (2026-07-20), Γ.**
- `postscf/hybrid.py` — a self-consistent PBE0-form global hybrid at Γ.
  `ScaledExchangePBE` scales the semilocal exchange by (1−α); `GammaFockExchange`
  is an SCF `fock` hook that rebuilds the ACE operator from the current orbitals
  each iteration (lagging one step like DFT+U) and adds α·V_x to
  `BatchedHamiltonian.apply` plus α·E_x as the new `EnergyBreakdown.fock` term.
  The hook is a minimal, guarded addition to `scf` (default `fock=None` leaves
  every existing path bit-identical — golden energies and the E↔H consistency
  gate still pass). Validated (`tests/integration/test_hybrid_scf.py`): α=0
  reproduces the PBE SCF exactly; PBE0 (α=0.25) converges and opens the Γ Si gap
  2.32→2.97 eV; the Fock energy matches the ACE energy on the converged orbitals.
  The spin factor (2/nspin in the energy, none in the operator) keeps energy and
  operator derivative-consistent.
- `postscf/isdf_k.py` — the multi-k ISDF acceleration. One shared interpolation
  set (points + ζ) fit across all occupied orbitals over the BZ compresses every
  co-density; the exchange contracts through a per-q Coulomb matrix V^q and per-k
  point-Grams A_k with N_μ FFTs instead of N_k²·N_occ² co-density FFTs. Reduces
  to the Γ ISDF build exactly and matches the direct `multik_exchange_energy` at
  saturated rank, with the rank as the accuracy knob and the range-separated
  kernel supported (`tests/integration/test_isdf_k.py`).

**k-mesh hybrid SCF LANDED (2026-07-20).**
- `exchange_multik.multik_exchange_operator` — the direct multi-k Fock operator:
  W_{tk} = V_x ψ̂_{tk} summing over the whole BZ through the co-density momentum
  q = k−k′ and the range-separated kernel (`coulomb_potential_q`). At one k-point
  it is exactly `exchange.exchange_operator_direct`, and its energy trace matches
  `multik_exchange_energy` on a mesh for the full and screened kernels
  (machine-precision gates in `tests/integration/test_hybrid_kmesh.py`). This is
  the O(N_k²·N_occ²) reference the per-k ACE compresses.
- `postscf/hybrid.py::MultiKFockExchange` — the k-mesh `fock` hook. `rebuild`
  extracts the occupied orbitals at every k, builds that operator, ACE-compresses
  it *per k*, and returns a per-spin callable applying α·V_x block-by-block over
  the k batch plus α·E_x. `hybrid_scf` now routes through it (reducing to the Γ
  build at a single k-point — the existing Γ tests pass unchanged). Validated:
  α=0 reduces to the PBE SCF on a mesh exactly; PBE0 on a (2,1,1) full-BZ mesh
  converges, the Fock term equals α·(2/nspin)·`multik_exchange_energy` on the
  converged orbitals (~1e-13), and the gap opens relative to PBE.
- Screened caveat. The screened Fock *operator* is exact and energy-consistent,
  but `ScaledExchangePBE` scales the *whole* PBE exchange by (1−α) — correct for
  full-range PBE0, but a complete HSE also range-separates the semilocal exchange
  (keep long-range PBE exchange, remove only the short-range fraction). That needs
  the range-separated (wPBE) enhancement, still open; use `mode="full"` for a
  physically complete SCF until then.

**Learned hybrid LANDED (2026-07-20).** The mixing α and screening ω are now
trainable end to end. `hybrid.differentiable_hybrid_energy(res, params)` turns a
converged hybrid SCF into a differentiable objective via the stationary-energy
(Hellmann–Feynman) derivative: at self-consistency the density is variational, so
dE_total/dθ = ∂E_total/∂θ — only the *explicit* θ-dependence of the exchange terms
on the frozen converged orbitals survives. It returns a scalar equal in value to
`res.energies.total` whose (α, ω) gradient is that exact derivative, so a plain
optimizer over `HybridExchangeParams` trains the hybrid. `hybrid_scf(..., params=)`
solves at the current param values (the SCF stays `no_grad`; the gradient rides the
converged result). The training loop is: converge → build the differentiable energy
→ backprop a loss → step. Validated (`tests/integration/test_learned_hybrid.py`):
the differentiable energy equals the SCF total; dE/dα matches a finite difference of
*re-converged* SCF energies to 6e-7 (rel), dE/dω to 2e-3; and a backward+optimizer
step moves (α, ω). This is the payoff no mainstream code has — see the framing below.

What remains, in build order: a Gygi–Baldereschi q+G=0 correction to complete
unscreened `full`/PBE0 at fine meshes (the divergent q+G=0 cell is dropped today,
converging slowly in N_k); the range-separated (wPBE) semilocal exchange to
complete screened HSE; a truncated Coulomb for physically isolated molecules; and
the RPA correlation contraction.

## Exact exchange and hybrid functionals

**Status: a learned PBE0-form hybrid SCF runs on a k-mesh — mixing/screening are
trainable.** The energy, operator, ACE, multi-k build, range-separated kernel,
differentiable hybrid parameters, the ISDF-K compression, the self-consistent Γ
hybrid SCF, the k-mesh lift (per-k ACE with each k's exchange summed over the whole
BZ, α·V_x acting in `BatchedHamiltonian.apply` block-by-block), and now the learned
hybrid (differentiable dE_total/dα, dE_total/dω through the stationary-energy
theorem) all landed (see the LANDED notes under the ISDF section above). gradwave
*self-consistently solves* a PBE0 hybrid on a full-BZ k-mesh (gap opening measured)
and *trains* its mixing and screening against a target. The remaining physics tails
are the Gygi–Baldereschi q+G=0 correction (fine-mesh PBE0 convergence) and the wPBE
semilocal screening (complete HSE). The paragraphs below are the original framing,
kept for the reasoning.

The biggest single physics gap, and the reason the two scaling items above are
worth building. Every energy, gap, force, and adsorbate level gradwave produces
sits on a GGA electronic structure with self-interaction error, so band gaps come
out too small and defect and adsorbate levels land in the wrong place. There is no
exact exchange anywhere in the SCF Hamiltonian today. A hybrid needs a Fock exchange operator applied
each SCF step, which is the O(N^4) object RI and ISDF exist to tame. The payoff
that no mainstream code has is a learnable hybrid, the mixing fraction and
range separation as trained parameters, which only makes sense once the Fock build
is affordable. **This is now realized** (see the Learned hybrid LANDED note above):
the ISDF/ACE Fock build is affordable, the k-mesh hybrid SCF converges, and
`differentiable_hybrid_energy` exposes the exact dE_total/dα, dE_total/dω so α and ω
train end to end against a target. The remaining reach is the same as any hybrid —
finer meshes (Gygi–Baldereschi) and a complete screened form (wPBE).

## Learned meta-GGA and the kinetic energy density

The learnable functional spans GGA form only, the two PBE parameters kappa and mu.
Every modern accurate semilocal functional (SCAN, r2SCAN) is meta-GGA, which means
it depends on the kinetic energy density `tau(r) = (1/2) Σ_i f_i |∇ψ_i(r)|²` on
top of rho and `|∇rho|²`. Without tau the learnable-XC path cannot fit, learn
against, or even compare with the functionals people actually use, so it cannot go
past GGA form. This is the natural next rung for the differentiable-XC work and the
one that lets `train_xc_paw` learn a real functional rather than only recover PBE.
It is also the cheaper stepping stone before hybrids, roughly a week against a
much larger EXX build.

- New piece, tau on the grid. Each occupied orbital's gradient is `i(k+G)` in
  reciprocal space, so `∇ψ_i` is one FFT per band per Cartesian direction, squared
  and accumulated with the occupations. This reuses the density-build FFT machinery
  with an extra factor of `i(k+G)`; the batched g-to-r path already carries the
  orbitals, so it is an added contraction, not a new solver.
- New piece, the meta-GGA potential. `v_tau = ∂e_xc/∂tau` does not act
  multiplicatively on rho. It enters the Hamiltonian as a tau-dependent
  modification of the kinetic term, `-∇·(v_tau ∇ψ)`, which makes this a generalized
  Kohn-Sham scheme and touches the H-apply, not just the functional. Autograd gives
  `∂e/∂tau` exactly the way it already gives `v_xc`, so no hand-derived kernel is
  needed, but the extra operator has to be wired into `BatchedHamiltonian.apply`
  and into the force and stress terms.
- Reuse, the functional interface. `XCFunctional.energy_density` gains a third
  argument `tau` beside rho and sigma, and the autograd `v_xc`/`f_xc` machinery, the
  spin channels, and the learnable-parameter graph all extend without new
  derivations.

Validate against QE `input_dft='scan'` (or r2SCAN) at pinned settings to the usual
milli-eV, then expose a learnable meta-GGA (an r2SCAN-form functional with
learnable parameters) and repeat the `train_xc_paw` recovery test at the meta-GGA
level. This is the item that most directly serves what makes gradwave distinct from
a very well-validated second copy of QE.

## Differentiable pseudopotential correction (learn away the Cu pseudization error)

The periodic-table Δ-gauge (`benchmarks/delta_gauge`) surfaced a concrete target:
the PseudoDojo standard UPF for Cu reproduces neither all-electron nor its own
psp8 (B0 167 vs 141, Δ 7.9 meV/atom), and gradwave matches QE on that same UPF to
0.08 meV, so the error is pseudization rather than implementation. Most transition
metals carry a smaller version of the same error (the "stiff-metal" Δ floor). One
day it would be worth using gradwave's differentiability to *address* that error
rather than only measure it.

The approach is to treat a small correction to the pseudopotential as a trained
parameter and descend it through the self-consistent solution against
all-electron reference data, the way the learned hybrid descends α and ω.
Concretely, add a differentiable δv(r) (a few-parameter radial form, or a
correction to the local channel / a KB coefficient) to the pseudo, and minimise a
multi-property loss: the EOS curve vs WIEN2k, or (better, once the all-electron
anchor exists) valence eigenvalues and the logarithmic derivatives at the
reference energies. The gradient dLoss/dθ_pp flows through the same
stationary-energy and Sternheimer machinery already used for dE/dα and the density
adjoint. The pseudopotential enters the energy through the local potential and the
nonlocal projectors, both already τ-differentiable for forces, so the parameter
graph is mostly in place. The result would be a *corrected* Cu (and, if it
generalises, a per-element learned correction that pulls the stiff-metal Δ floor
down), and a demonstration that differentiable DFT can improve a pseudopotential
against all-electron data rather than only use a fixed one. Care is needed. Keep
the correction small and norm-conserving, and validate that it does not overfit
the EOS at the expense of transferability (band structure, a second crystal
structure).

## Differentiable spintronics: spin Hamiltonians, DMI, and inverse design

The constrained non-collinear framework (`postscf/moment_config.py`,
`scf/moment_penalty.py`) is differentiable, and that is the lever that separates it
from the constrained-DFT in QE/VASP/FLEUR. The per-atom torque `dW/de_I` is
autograd-exact (validated to a finite difference at ratio 1.000), not
finite-differenced, and the magnitude-robust `vector` penalty holds an arbitrary
non-collinear texture at fixed moment instead of letting it collapse. Every item
here is a thing that is cheap *because* of AD and awkward otherwise; the framing
below is roughly in impact order.

- **The killer app: extract the spin Hamiltonian (J, D, K) by differentiating the
  torque.** The whole spintronics modeling stack — atomistic spin dynamics
  (VAMPIRE, Spirit), micromagnetics (mumax) — runs on
  `H = -Σ J_ij S_i·S_j - Σ D_ij·(S_i×S_j) - Σ K_i (S_i·n)²`, and DFT's job is to
  parametrize it. Conventionally that is finite-difference energy mapping (fragile)
  or a separate Green's-function (LKAG) machinery. Here the Heisenberg `J_ij`, the
  Dzyaloshinskii–Moriya vectors `D_ij`, and the anisotropy `K` are *derivatives of
  the torque*: `d(dW/de_I)/de_J` is the exchange/DMI coupling tensor between sites I
  and J, so a second autograd pass over the torque we already compute gives the
  couplings directly, cross-terms included. The DMI is the prize — it needs SOC and
  non-collinearity (both present), it sets skyrmion chirality and size, and it is
  notoriously noisy to compute by finite differences. A differentiable,
  magnitude-conserving DMI extractor is methodologically novel, not just a demo, and
  it is the bridge from small-cell plane-wave DFT to device-scale spin dynamics: the
  code cannot simulate a 50 nm skyrmion, but it can hand a clean spin Hamiltonian to
  a code that can. This is the highest-impact direction. Second derivatives of the
  penalty scalar are already within reach of the autograd path; the work is wiring
  the site-pair Hessian and validating J against a known magnet.
- **Chiral textures and the micromagnetic DMI from spin-spiral asymmetry.** The Fe
  spin-spiral demo (`examples/fe_spin_spiral.py`) traces `E(theta)` with inversion
  symmetry, so `E(+q) = E(-q)`. Break inversion — an interface (Co/Pt, Fe/Ir) or a
  B20 bulk (MnSi, FeGe) — and turn SOC on, and `E(+q) ≠ E(-q)`: the chiral splitting
  whose `q→0` slope *is* the micromagnetic DMI constant. Demonstrating a measured
  `E(+q) - E(-q)` is a direct extension of the committed spiral sweep (add SOC,
  rotate in the DMI-active plane, sweep signed q) and a genuine chiral-magnetism
  result.
- **Inverse design — the differentiable moonshot.** Energy is differentiable w.r.t.
  atomic positions, moment directions, and in principle strain and composition, so a
  gradient-based search can optimize *toward a target magnetic property*: the strain
  that maximizes DMI, the composition that flips the easy axis. No finite-difference
  code can do this. Higher risk — it needs gradients through the SCF, not just the
  envelope torque — but it is the purest expression of the gradwave thesis applied to
  magnetism.
- **Demonstration vehicle: 2D magnets.** CrI3, Fe3GeTe2, CrSBr maximize impact per
  core-hour: small unit cells (tractable in plane waves), large anisotropy (clears
  the ~0.2 µeV rotation-invariance precision floor that cubic Fe sits on), and open
  questions about their DMI, topological magnons, and stacking-dependent order. A
  J/D/K extraction on CrI3, with DMI as the headline, is the tractable-plus-novel
  sweet spot. The MAE map (next section) is the better *visual* deliverable and the
  natural second step once the SOC force-theorem path is in.

**The pseudopotential blocker is now cleared.** DMI and single-ion anisotropy K need
spin-orbit coupling *on a magnetic atom*. The fully-relativistic magnetic pseudos
(and iodine) are now in the fixtures — `Fe/Co/Ni/Cr/Pt_ONCV_PBE_FR-1.0.upf` and
`I_ONCV_PBE_FR-1.1.upf`, pulled from the SG15 ONCV set (quantum-simulation.org, the
same source as the scalar `_ONCV_PBE` pseudos). A magnetic FR pseudo correctly
triggers the SOC path (`system.is_fr` → j-resolved spinor projectors). The natural
order: L1_0 FePt MAE (`Fe_FR` + `Pt_FR`, ~2-3 meV/f.u., far above the 0.2 µeV floor)
→ hcp Co (~65 µeV) → CrI3 (`Cr_FR` + `I_FR`) for the full J/K/DMI story. The
Heisenberg-J machinery (`postscf/spin_exchange.py`) and the `characterize_magnetism`
routine already work without SOC.

## Magnetocrystalline anisotropy (MAE maps) and per-atom spin torques

**Status: core landed, one open tail.** The force-theorem evaluator
(`postscf/mae.py`) and per-direction magnetic-IBZ folding both landed and are
validated at production scale (the dated LANDED notes below). What remains open
is **band-resolved anisotropy** — decomposing ΔF into per-k, per-band
contributions (last subsection). The rest of this section is kept for the
reasoning.

The constrained-moment work (`postscf/moment_config.py`) already produces one half
of this for free. `constrained_moment_scf` returns a per-atom transverse torque
`-dW/de_I`, validated to a finite difference at ratio 1.000 — that *is* the
magnetic force-theorem spin torque on each atom. Without spin-orbit coupling it is
the inter-atomic exchange torque (what drives the config search and sets a spin
spiral's stiffness); with a fully-relativistic pseudo the same per-atom torque
picks up the on-site anisotropy term. So "individual spin torques" is not a future
capability, it is what the module returns today. The missing half is the *global*
anisotropy: MAE maps `E(theta, phi)` over the magnetization sphere.

The ingredients are in the tree. The SOC path exists — `core/spinor_proj.py` builds
the j = l ± ½ resolved projectors and `SpinorHamiltonian` accepts them (`q`,
`dij_so`). `NCResult.energies.free_energy` gives a total energy per direction, so
`MAE = E(n1) - E(n2)` and a full surface are directly a direction sweep. The
efficient route is the torque method: one SOC evaluation per direction yields the
anisotropy torque `-dE/dn`, and integrating it over the sphere reconstructs the
surface — and that torque is the machinery we already have, applied to the total
moment instead of a local one.

**Proof-of-physics landed (2026-07-18): the two-point MAE of L1_0 FePt by full SOC
SCF is correct** — +2.55 meV/cell (+1.28 meV/atom), easy axis [001], magnitude in
the literature band, both orientations converged to ~1e-11 eV (`examples/
fept_mae.py`, 144 k, on the asus CPU). The 48-k mesh gives the WRONG easy axis
(−1.39 meV/cell toward [100]) — the textbook sampling-error sign flip, measured
here directly. So the remaining work is exactly the cost problem below: the
force-theorem evaluator to make dense-k sweeps and full E(theta, phi) maps
affordable, plus magnetic-space-group reduction (now landed, see the Shubnikov
symmetry section under "Done and resolved") for the ~4-8x k-savings.

Three things are genuinely in the way, and the third is the only real code.

- **Precision floor.** `test_noncollinear.py` pins rotation-invariance (MAE ≡ 0
  without SOC) to ~0.2 µeV, the numerical noise floor. Cubic Fe's MAE is ~1 µeV/atom,
  sitting right on it — reproducible only with great care. Start instead on a
  high-anisotropy case that clears the floor by orders of magnitude: L1_0 FePt
  (~1 meV/atom), hcp Co (~65 µeV), or a uniaxial 2D magnet.
- **k-convergence.** Metal MAE converges painfully slowly in k (thousands of points,
  or fine-smearing / Fermi-surface-aware tricks). A cost problem, not a capability
  gap, but it is the reason the force theorem matters.
- **No force-theorem path for SOC yet.** The standard cheap recipe — converge the
  density scalar-relativistically once, add SOC *non-self-consistently*, and take
  occupied-band-energy differences per direction — is not wired. The frozen-potential
  band-solve infrastructure already exists (`postscf/uspp_bands.py`, `core/gamma.py`,
  the one-shot solve in `postscf/hubbard_u.py`); it just is not connected to the SOC
  Hamiltonian plus a directional band sum. Without it, MAE falls back to a full
  self-consistent SOC SCF per direction: affordable for FePt-class anisotropy, too
  expensive and too noisy for Fe.

What to build, all reusing what is here: (1) a global spin-axis control (rotate all
local `e_I` together — a one-line special case of the per-atom constraint, or seed
and let SOC pin it); (2) a force-theorem evaluator that freezes the converged
density, adds the SOC block, and does one non-SCF diagonalization per direction into
`dE(n)`, reusing the frozen-potential solve and the spinor projector block; (3) a
thin sweep/integrate layer over `(theta, phi)` taking energy differences or
integrating the torque into the anisotropy surface. The blocker is not the math —
the torque is already exact and autograd-derived — it is getting a fully-relativistic
pseudopotential into the fixtures and writing that force-theorem loop so the map is
affordable.

**Force-theorem evaluator LANDED (2026-07-19).** `postscf/mae.py`
`force_theorem_mae`: one converged SOC SCF along a reference axis, then per
direction a rigid rotation of (m⃗, B_xc) — exact for the locally-collinear XC,
since B_xc co-rotates with m⃗ and v_xc depends only on (ρ, |m⃗|) — one
frozen-potential spinor diagonalization seeded with the SU(2)-rotated reference
spinors, and the occupied band free energy at that direction's own Fermi level.
The anisotropy enters solely through the lattice-fixed SOC projectors, so a
scalar-relativistic system gives an exactly direction-independent band sum, and
that is the correctness gate: measured invariant to <1e-6 eV over four
directions, with the reference direction reproducing the converged SCF spectrum
(tests/integration/test_mae_force_theorem.py). At a shared small FePt mesh the
FT difference tracks the two-SCF difference within the 30% gate. At scale
(examples/fept_mae_force_theorem.py, 144 k full mesh, 70 Ry): FT MAE
[100]−[001] = +2.673 meV/cell vs the self-consistent +2.552 (4.7%), [110]
+2.713 (in-plane spread 0.04 meV), and the 45°-tilted [101] at +1.340 ≈ half of
[100] — the uniaxial K₁ sin²θ form measured directly from four one-shot solves.
Each direction costs ~11 min against ~84 min for a full SCF (7.7×), which is
what makes E(θ, φ) maps affordable. Items (1) and (3) above are subsumed: the
directions list is the sweep layer.

**Per-direction magnetic-IBZ folding LANDED (2026-07-19).**
`force_theorem_mae(..., magmoms=...)` folds each one-shot solve into its own
direction's Shubnikov IBZ: the per-atom reference moments rotate with the
direction, the magnetic group of the rotated texture folds the mesh
(`reduce_mesh_magnetic`), and — because every folded representative is a point
of the full mesh — the solve runs on a subset of the stored reference spheres
with the folded weights, and the SU(2)-rotated seeds gather straight from the
reference coefficients. The reference SCF still needs the full mesh. Only the
evaluations fold. The fold is exact for the collinear part of the frozen
magnetization (ρ and |m⃗| carry the crystal symmetry, the uniform rotated
direction transforms as an axial vector). The SOC-induced transverse textures
in m⃗(r) formally break it, but the measured folded-vs-full residual on the
small FePt system is ~4e-12 eV, far below the force-theorem error
(tests/integration/test_mae_force_theorem.py, gate at 1e-6). On the 6×6×4
mesh the folds are [001]→30/144, [100]→48, generic (010)-plane tilt→56,
compounding with the 7.7× per-direction saving.
examples/fept_mae_map.py uses this for an E(θ) scan [001]→[100] with a
K₁sin²θ + K₂sin⁴θ fit. Measured at scale (asus CPU, 2026-07-19, on the
batched spinor density path): 7 directions in 889 s (~2.1 min each, against
~11 min unfolded pre-batching), K₁ = +2.6965 meV/cell, K₂ = −0.0358, max fit
residual 0.0015 meV — FePt is uniaxial to 1%. The 45° point reproduces the
unfolded full-mesh +1.3398 meV to all printed digits, confirming the fold
exact at production scale, and [100]−[001] = +2.660 vs the self-consistent
+2.552 (4.2%). The whole 7-point map costs about an hour of CPU: one 2450 s
reference SCF plus 15 min of folded solves.

**Band-resolved anisotropy (open).** The MAE is a single number; the
diagnostic that *explains* it decomposes ΔF into per-k, per-band
contributions by differencing the spectra of two directions state by state
(each at its own Fermi level) before summing. The physics: away from ε_F the
SOC shifts cancel in the difference, so the net anisotropy lives in
near-degenerate band pairs within ~ξ_SOC of the Fermi level that split
differently per direction — in FePt the Pt-5d avoided crossings, which is
also why coarse meshes flip the sign. Everything needed is already persisted:
MAEResult.save keeps the full (nk, nb) spectra and Fermi levels per
direction. What to build: (1) a ΔE(k) map over the mesh plus a band-index
decomposition, occupation-weighted with the entropy term handled explicitly;
(2) an unfolding step, since per-direction folds put two directions on
different k-subsets — either re-run the pair of interest unfolded (minutes
now) or expand folded spectra back to the full mesh through the orbit maps
the magnetic-symmetry machinery already computes. Caveats to state in the
docs: per-k contributions are gauge-sensitive (only the k-sum is physical),
and band *indices* must be matched through crossings if the decomposition is
followed along θ. Payoff: hotspot maps on the Fermi surface, and a principled
handle on how alloying/strain will move the MAE.

## Davidson subspace Gram: conj-copy memory spike at large nk

Measured on the A100 384-k FePt run (job 14076535, 2026-07-18): with
fragmentation already fixed (expandable_segments), the run died at 32.2 GiB
allocated when `davidson_batched`'s subspace overlap
`torch.einsum("kig,kjg->kij", v.conj(), hv)` requested another 6.68 GiB —
einsum materializes `.conj()` as a full copy of the (nk, nsub, 2npw) subspace
block. Two cheap fixes when it next matters: chunk the Gram over k (the result
is only (nk, nsub, nsub), tiny), or restructure to avoid the conj copy
(`(hv @ v.mH)`-style batched matmul conjugates lazily). Deferred because the
magnetic-IBZ fold (60–100 k instead of 384) removed the pressure — but any
future dense-k run without magnetic symmetry hits the same wall at
nk·nsub·2npw·16 B ≈ 7 GiB per copy.

## One-center ddd analytic derivative

The one-center ddd is a named micro-cost from the performance audit, 5% of the PAW
profile through an autograd backward per iteration. It is already compiled when
`compile_xc=True` (the `energy_and_ddd` path is a single backward), so the remaining
question is only whether an analytic quadrature derivative beats the compiled
autograd, which is a small isolated experiment, not a feature. (The local-TF metal
preconditioner that used to head this section is now built, see the done section.)

## RAIRS and a slab dipole moment

We can do vibrational frequencies now (`postscf/phonons.py`, validated against QE
`ph.x` to 0.003% on Si) and IR intensities for insulators and molecules (Born
charges and epsilon-infinity from `postscf/dielectric.py`). Metals are the gap.

The current `dielectric_born` refuses anything but an nspin=1 insulator, because it
splits valence from conduction with a conduction projector `P_c` and a `(H - eps_v)`
solve, and that construction goes singular at a metal Fermi level. A bulk metal also
has no IR-active optical phonon in the insulator sense and no static Born charge, so
"bulk metal IR" is not a real target.

The real target is RAIRS, the reflection-absorption IR of an adsorbate on a metal
surface (CO on Pt is the textbook case). The metal surface selection rule says only
the dynamic dipole perpendicular to the surface couples, and the slab surface-normal
dipole `mu_z` is well defined despite the metal because the vacuum gap gives a clean
reference. So we sidestep the singular DFPT entirely and finite-difference.

- New piece, a slab dipole-moment function `mu_z = integral z rho_tot(r) + ionic`,
  with the standard slab caveat that it needs a vacuum gap and a dipole correction
  so the two surfaces do not talk through the cell. This is the only genuinely new
  physics. gradwave already has rho on the grid and the ion charges, so it is modest.
- Reuse, finite-difference the dynamic dipole. Displace each adsorbate atom by
  plus/minus delta, compute `mu_z`, difference to get `Ztilde_{s,zbeta} = d mu_z / d tau`.
  For CO that is 2 atoms x 3 directions x 2 = 12 SCFs. Contract with the CO-projected
  Hessian modes from `gamma_hessian` and keep the z component.

Cost is finite-difference, roughly 12 extra slab SCFs on top of the Hessian, no DFPT.
It works on the metal because it uses `mu_z`, not `Z*`. Estimate about 1 to 2 days,
almost all of it in the slab dipole routine and its validation.

Raman on a metal stays hard. The Raman tensor is `d alpha / d Q` and alpha itself is
ill-defined for a metal, and surface-enhanced Raman is dominated by electromagnetic
field enhancement rather than a clean DFT observable. Leave it.

If someone wants the metal's own far-IR response, that is a different deliverable,
the optical conductivity `sigma(omega)` (Drude plus interband). It extends the
existing E-field Sternheimer to finite frequency and adds the intraband
Fermi-surface term `dielectric.py` omits. Roughly a week, and it produces optical
conductivity, not a vibrational spectrum.

## Little and orbit groups for DFPT under symmetry-breaking perturbations

The response calculations either use the full crystal symmetry or drop to
time-reversal only. A perturbation lowers the symmetry to its little group, and we
should reduce k-sampling and irreducible displacements by that residual group
instead of discarding symmetry outright.

- For a Gamma-phonon column, the displaced-atom pattern has a little group, the site
  symmetry intersected with the displacement direction. Only symmetry-inequivalent
  columns need computing, and the rest are reconstructed by the group action (the
  `HessianSymmetry` reconstruction already does the reconstruction half for the full
  group, this generalizes it to the perturbation little group).
- For the E-field response, the little group is the subgroup that leaves the field
  vector invariant, and k reduces to that subgroup's IBZ rather than to
  time-reversal only.

Payoff is direct k-point and displacement-column savings on exactly the expensive
response runs, which is what QE `ph.x` does with its `modes_of_q` and small-group
machinery. The building blocks (`find_spacegroup`, `reduce_mesh`) exist, the work is
computing the little group of a given perturbation and threading it into the DFPT
drivers.

## Phonon band structures

`gamma_hessian` is Gamma-only. Extend to finite q to get dispersions.

- Real-space force constants from finite displacements in a supercell, or an
  analytic force response at q with a q-dependent perturbation. The supercell route
  is simpler to land first.
- Fourier interpolate `D(q)` onto a band path. Reuse the electronic bands path
  builder for the q-path and labels.
- Acoustic sum rule, and for polar insulators the nonanalytic LO-TO term at q to
  Gamma, which needs Born charges and epsilon-infinity. Both already exist in
  `dielectric.py`, so the polar correction is reachable.

With a q-mesh this also gives the phonon DOS and the harmonic thermodynamics (free
energy, entropy, heat capacity), which pairs well with the EOS work for a full
thermal equation of state.

## Full nspin=2 and PAW coverage for every feature

Coverage is uneven across the postscf features. Several are nspin=1 or NC only.
`dielectric_born` is nspin=1 insulators, the discretization-error force path
is NC (nspin=1 or 2, no USPP/PAW), and the noncollinear and SOC PDOS paths have their
own constraints.

Make an explicit matrix of feature x {NC, USPP/PAW} x {nspin=1, 2} and close the
gaps. Most of the per-channel machinery exists, so the work is threading the spin
index and the S-metric or augmentation consistently, plus tests at each new cell of
the matrix. Unglamorous, but it is what makes the code trustworthy on real systems
like magnetic surfaces and spin-polarized adsorbates. The SCF core itself is already
even here, the batched USPP/PAW eigensolve is validated at nspin=2 (O2 triplet,
batched vs per-k to 7e-12 eV, and 21 iterations to QE's 20), so the gaps are in the
postscf property layer, not the solver.

## Error estimation: the rest of the budget, and what it can and cannot reach

The discretization estimate (`postscf/discretization_error.py`) is one term in a
larger error budget, and it is the cleanest one because a plane-wave cutoff is a
variational truncation: a converged truth exists at infinite basis, the energy is
stationary there, so the error is second order and a single cheap perturbative pass
reaches it. Density, energy, force (NC nspin=1/2), and now per-band eigenvalue and
band-gap errors all fall out of the same complement correction. This section records
what the other terms are, which of them share that structure, and whether adding
them up ever gives the true error. It never does, and the reason bounds the whole
program, so it is worth writing down.

Split the budget into a numerical part and a model part. Every number sits at some
distance from reality, and that distance factors:

    E_computed - E_reality = (E_computed - E_KS_converged) + (E_KS_converged - E_reality)
                              \______ numerical ______/       \____ model (XC) ____/

`E_KS_converged` is the exact-basis, dense-k, fully self-consistent Kohn-Sham energy
for the functional you chose. The numerical term is the sum of the convergence
errors (cutoff, k-points, SCF, smearing, density grid, cell size). The model term is
the XC functional error. The two are categorically different, and only the first is
reachable from inside a calculation.

The numerical terms are trackable, roughly additive, and individually cheap.

- SCF convergence error. Stopping the iteration at finite `rhotol` leaves a density
  residual `drho = rho_out - rho_in`. Because the energy is stationary at the fixed
  point, the error is `~ (1/2) <drho | K_Hxc + chi0 | drho>`, second order in the
  residual (this is why the energy converges as the square of the density). The
  kernel is exactly the operators `scf/implicit.py` already exposes for the Dyson
  dressing, so this is a few lines and one response application, no new SCF. The
  Harris-Foulkes vs Kohn-Sham energy pair at the last step brackets it as a
  zero-machinery cross-check.
- k-point sampling error. The largest untracked term for metals and small cells, and
  the one that does not share the variational structure: BZ integration is a
  quadrature, not a truncated variational space, so the complement trick does not
  transfer. It is reachable instead by mesh extrapolation or integrand-smoothness
  estimates. Different math, higher value.
- Smearing / electronic temperature. The `E - (1/2)TS` (Methfessel-Paxton) T->0
  correction adds almost no cost: the entropy term is already computed in
  `shared_fermi_occupations`, so exposing the extrapolated energy is the whole task.
- Density-grid (`ecutrho`) and finite-size round out the list. The first is the same
  perturbative logic on the dense grid (USPP/PAW-relevant). The second is
  system-specific (Makov-Payne image corrections for charged/defect/molecular cells).

At leading order these add, but not exactly: the axes couple (the basis error depends
on the density, which depends on the k-mesh), so the true numerical error carries
cross terms the per-axis estimates omit. Summed, they estimate the distance to
`E_KS_converged` well; they do not reproduce it to machine precision.

The model term, the XC error, is categorically not internally reachable. Converge
every numerical knob and you are left with the exact answer for an approximate
functional, off from reality by the XC error, and nothing in the run measures it: it
needs external reference (CCSD(T), QMC, experiment) or the exact functional. What is
available is weaker, and worth being precise about.

- Density-corrected decomposition (DC-DFT). The XC error splits into a
  functional-driven part (wrong functional on the exact density, not recoverable
  internally) and a density-driven part (`E_xc[rho_A] - E_xc[rho_B]` for a better
  density `rho_B`, computable and correctable). The textbook `rho_B` is the
  self-interaction-free Hartree-Fock density, which needs exact exchange the code
  does not have cheaply; the available proxy is the LDA<->PBE density sensitivity,
  and the differentiable machinery carries the resulting density change to any
  observable exactly as the force estimate does.
- Functional sensitivity. The `learnable.py` slot plus autograd give
  `d(observable)/d(XC parameters)` in one pass, a linearized single-run version of
  the BEEF ensemble spread. It is a variance, not the error, and it is calibrated to
  whatever the parameters span.
- Self-interaction diagnostic. `E(N)` should be piecewise-linear in fractional
  electron number for the exact functional. The deviation is a direct, fully internal
  measure of the delocalization error that dominates gaps and charge transfer,
  needing only the fractional occupations the smearing path already supports.

Self-interaction and the density-driven error are decompositions of the XC term, not
independent channels to add on top of it; treating them as separate additive errors
would double-count.

So, does the full budget give the true error? No, for two reasons stacked. The XC
(model) term is not knowable from inside the calculation, so there is an irreducible
unknown no combination of internal estimates reaches; and even the numerical terms
only sum to leading order, missing their cross-coupling. What the numerical budget
does provide is a defensible estimate of the distance to `E_KS_converged`: you can drive
the cutoff, k-point, SCF, and smearing errors to near zero and know that you have.
The XC error is then both the largest remaining term for most production work and the
only one you cannot self-certify. That is the honest shape of the effort. The
numerical errors are a solved problem in principle, the accuracy that matters is in
the functional, and the differentiable framework's advantage on that term is
sensitivity and the density-driven half, not an absolute bar.

**Smearing and k-point terms LANDED (2026-07-18); SCF-convergence term OPEN
(formula wrong).** `postscf/convergence_error.py` holds three estimators.
`estimate_smearing_error` (the scheme-matched `E0 = (E+F)/2` extrapolation with
per-scheme caveats) and `estimate_kpoint_error` (mesh extrapolation
`E(N_k) → E_inf`, the non-variational term that needs more than one run) are
validated in `tests/integration/test_convergence_error.py`. Together with the
Ecut estimate in `discretization_error.py`, those two plus the cutoff term are
the trackable part of the numerical budget that is built.

`estimate_scf_error` is **not** validated: its formula is wrong. The exact
second-order residual energy is `1/2<x|(K_Hxc - chi0^-1)|x>` with `x` the
dielectric-dressed density error, but the code forms
`1/2<r|K_Hxc (1-chi0 K)^-1|r>`, which omits the `chi0^-1` kinetic-response term.
The two SCF-error tests are marked `xfail(strict=True)`. Fixing it needs the
exact Schur coupling, which is the same missing term as the coarse-space Dyson
refinement of δρ in [todo.md](todo.md) — pin one and both resolve. The
`chi0^-1` solve is numerically awkward by direct CG (chi0 is only known through
its forward action), so this is a real piece of work, not a typo.

Also open from this section is the model-term tooling: the fractional-charge
self-interaction probe (self-contained, no second functional) and the DC-DFT
density-sensitivity piece, plus the smaller density-grid (`ecutrho`) and
finite-size terms.

## Showcase figures: noncollinear magnetism and error estimates

The validation record is tables of meV agreements against QE, which persuades a
methods reader and no one else. A small set of figures would carry the two
capabilities that distinguish the code, autograd through the full spinor stack
and per-calculation numerical error bars. DFTK has the Cancès/Herbst error
bounds and several codes do noncollinear SOC, but the combination, and anything
built on spinor autograd, has no published counterpart to point at. Candidates
below, roughly by impact, each grounded in what exists in the tree today.

Noncollinear magnetism.

- **MAE sphere for FePt.** `examples/fept_mae_map.py` already scans E(theta)
  along [001]→[100]. The full version is E(theta, phi) − E(easy) as a heatmap
  on the sphere (Mollweide projection), easy axis marked, with a few full-SCF
  anchor points overlaid to show the force-theorem accuracy. The per-direction
  magnetic-IBZ folding plus the 7.7× one-shot saving is what makes the map
  affordable, so the figure doubles as the cost story.
- **Torque against angle.** Plot the autograd dE/dtheta of the moment
  direction as a smooth curve and overlay finite-difference slopes of the
  E(theta) scan as points. One figure shows the SOC physics and that the
  spinor stack differentiates. The per-atom torque is validated in
  `moment_config`. The global-axis torque through the SOC energy is the piece
  the "MCA dE/dtheta" line in the backlog still owes, so this figure is also
  the natural acceptance test for it.
- **Real-space magnetization texture.** A quiver plot of m⃗(r) on a plane
  through the cell, arrows colored by |m⃗|, density as a background contour.
  The strongest subject is a 120° Néel state on a triangular Mn or Cr
  lattice, because frustration forces genuine noncollinearity and a collinear
  code cannot represent the ground state at all. The Fe spiral
  (`examples/fe_spin_spiral.py`) is the already-computed fallback.
- **k-space spin texture.** `projected_dos_noncollinear` already Pauli-decomposes
  each state into (n, m_x, m_y, m_z). Coloring a band path by ⟨sigma_z⟩ with
  in-plane arrows at each k gives the spin-momentum-locking picture around
  Gamma for Bi₂Se₃, whose SOC bands are already validated
  (`examples/bi2se3_inversion.py`). Needs the per-state amplitudes routed onto
  a band path rather than binned into a DOS.
- **Magnetic-IBZ folding diagram.** The full mesh next to the Shubnikov IBZ
  for FePt m∥[001] (144→30) and m∥[100] (144→48), with the folded-vs-full
  energy residual quoted. A methods figure, narrower audience.

Error estimates.

- **Estimated against true error.** For a grid of systems and cutoffs, scatter
  the `discretization_error` estimate against the measured error to a
  converged-Ecut reference. Points on or bounded by the diagonal are the whole
  argument for trusting the estimator, and the plot the Herbst/Levitt paper
  the module follows leads with. The EOS/Δ-factor infrastructure
  already produces the reference energies.
- **EOS with error bars.** E(V) at a deliberately modest cutoff with per-point
  error bars, overlaid on the converged curve, the bars visibly containing it,
  and the fitted a₀/B₀ carrying propagated uncertainties. The force version is
  displaced-Si force components with bars against converged forces.
- **Stacked error budget.** One bar per system, stacked into basis, SCF,
  smearing, and k-sampling terms from `discretization_error.py` plus
  `convergence_error.py`. No other code's output decomposes this way, so the
  figure states the capability without a caption.

The combination, and the strongest single figure, is **MAE with numerical error
bars**. Anisotropy energies sit at tens of µeV to meV, exactly the scale where
convergence is the standing doubt, so E(n̂) − E(easy) with a shaded numerical
uncertainty band makes a scientific claim rather than a benchmark claim, namely
that the easy-axis assignment clears the error bar (or honestly, at which
Ecut/k-mesh it does not). The `discretization_error` estimator covers NC
nspin=1/2 but not the spinor/SOC path, so it needs the spinor extension first,
and this figure is the reason to build it.

Several items are half-built (the MAE scan, the spiral, the Bi₂Se₃ band data,
the EOS scans), so the marginal work is mostly plotting plus a few targeted
calculations, with the heavy SOC sweeps routed to the GPU box as usual.

## Batched multi-structure SCF, and the EOS-on-GPU question

Question, would an EOS go faster by batching several volumes on the GPU at once?

Measured on the asus RTX 3050 for the 1-atom fcc Pt EOS (40/400 Ry, 12x12x12), a
single point sits at 100% nvidia-smi util but only 24.7 W of draw (the card's TGP is
35 to 80 W) and 2.6 of 6 GB. The 100% util flag only means a kernel was in flight
during the sample. The low power and low memory say the GPU is not compute-saturated.
For a system this small it is launch and latency bound on many tiny kernels (small
matmuls, a 35^3 FFT, per-k Davidson steps), so there is real headroom. So yes,
concurrency would help here. Three ways, cheapest first.

- Run several volumes as concurrent processes sharing the GPU, either plain
  backgrounding or CUDA MPS. Zero code. Two points fit in 6 GB (2 x 2.6). Likely
  1.5 to 1.8x on a launch-bound system. The catch is that the current EOS chains the
  volumes with `start_from` warm starts, so they are serial by construction. Dropping
  the chain trades the warm-start iteration savings for the concurrency, which is
  close to a wash at N=2 but wins as the GPU empties.
- Batched multi-structure SCF, the real structural win. Stack the volumes as
  independent k-blocks in one padded generalized Davidson, the same way the batched
  Davidson already stacks k-points, so the small per-volume GEMMs become one big GEMM
  and the kernel-launch overhead amortizes. The SCF loop has to carry per-volume
  densities and potentials and mix them independently while sharing the linear
  algebra, which is a genuine feature, not a tweak. This is the version that would
  actually fill the card. It generalizes past EOS to any embarrassingly-parallel set
  of small structures (displacement stencils for phonons, rattled configs for
  training data, a k-convergence sweep).
- CUDA streams to overlap independent kernels. Hard to orchestrate from PyTorch
  eager, low priority.

Note that this only pays for small systems where a single SCF underfills the GPU. The
slab already uses more of the card, so batch structures for the cheap cases (bulk
EOS, phonon stencils) and run the heavy cases one at a time.

The cleanest first target is a spin-spiral / magnetic-dispersion sweep (see
`examples/fe_spin_spiral.py`). Every angle theta is the *identical* cell, k-mesh, and
band count -- same FFT dims, same tensor shapes -- so the batch has zero raggedness in
the data layout; only the per-point convergence count differs. That is a strictly
cleaner batching case than the EOS, where the cells (and their FFT boxes) vary slightly
with volume. The one wrinkle is the same one everywhere: the frustrated large-angle
points need many more iterations than the collinear ones, so a lockstep batched solve
either over-iterates the easy members or needs per-member convergence masking. The real
blocker is the hardware, not the workload -- on the RTX 3050 the sweep is fp64-bound and
7.8x slower than the CPU (it runs as concurrent CPU processes today, see the done
section on the measured 3050 profile). On a card with real fp64 (A100/H100, fp64 = 1/2
fp32) and tens of GB, stacking these identical independent SCFs to fill the device is
exactly where the batched-multi-structure path first pays off.

The best fit is GGA insulators. They are fixed-occupation, converge in few
iterations, and hold a small grid, so a single one badly underfills the card, which is
exactly the regime where stacking several into one padded solve wins. A batch of GGA
insulator structures is also the shape of a learned-XC training set and an EOS or
convergence sweep, so this feature and the meta-GGA training work reinforce each other.

## Gamma-only real wavefunctions for slabs and molecules

At the Gamma point the orbitals can be taken real, because time reversal makes
`ψ(-G) = ψ*(G)`, so only half the plane-wave sphere is independent. The foundation for
this is built and validated in `core/gamma.py`, gated to machine precision against the
complex path (apply 1e-13, frozen-potential eigenvalues 5e-14). It stores the half
sphere, runs the local term on `irfftn`/`rfftn`, and solves the eigenproblem as a real
symmetric one in a feature embedding where the half-sphere metric is the plain dot
product, so the standard Davidson applies unchanged.

The premise was a roughly 2x real-FFT win on the hottest kernel. That did not appear on
the available CPU. The forward-plus-inverse real transform measured 0.75x to 1.25x the
complex pair on non-power-of-two boxes at 63^3 and 72^3, so the H-apply came out 0.97x
in isolation, and the full solver ran slower still (directionally 0.6x to 0.8x) once the
per-apply overhead compounds over the Davidson iterations. The real-transform advantage
is grid-size and library dependent, and MKL did not deliver it here. The correctness is
solid, so the work that remains is measurement and integration rather than the core
representation.

- Re-measure on a GPU. cuFFT's real transform behaves differently from MKL's, and the
  memory story is also better on the GPU, so the win may exist there even though it does
  not on this CPU. This is the first thing to check before investing more.
- Wire it into the SCF loop behind a flag, for a single Gamma k-point, insulators and
  molecules first, then metals at Gamma with smeared occupations. The density build,
  mixing, and energy assembly are unchanged, only the diagonalize call swaps.
- The memory angle stands on its own. The real-space fields are half the size, so this
  pairs with the size-ceiling item below independent of any speedup.

## Raising the system-size ceiling past the dense-allocation cliff

The GPU probe found peak memory scaling roughly linearly to about 96 atoms on the 6 GB
RTX 3050, then a hard cliff at 128 atoms from a single roughly 37 GB allocation, an
O(npw²) dense step (complex128 around 7.7 GB times the eigh workspace copies) that spikes
at `npw` near 22k. So the practical ceiling is about 96 to 110 atoms at that cutoff,
and the cliff is a specific dense allocation, not gradual fill, which means it is
tileable rather than fundamental.

- Identify the O(npw²) step. It is the dense object that scales with the square of the
  plane-wave count, most likely a subspace-related workspace or the eigensolve's internal
  copies, and the first task is to confirm which allocation trips at 128 atoms with a
  memory profile.
- Tile or avoid forming it. Block the offending contraction so the peak is bounded the
  way `BatchedHamiltonian.apply` and `density_b` already band-chunk their dense-grid
  temporaries, or restructure the step to never materialize the full O(npw²) array.

This only matters if larger cells become a goal, defects, bigger slabs, or supercells for
finite-q phonons, so it is a when-you-need-it item rather than a now item. But it is the
one thing standing between the current sub-100-atom validation regime and running the
kind of system where the code would do new science, so it is worth knowing the fix is a
tiling change and not an architecture change. The ISDF work above is the complementary
lever, it lowers the operation count where this item lowers the peak memory.

## Acceleration frontier, 2024-2026 literature sweep

A focused survey of the recent literature (done after the local-TF preconditioner
landed) for levers that pass the filter "single GPU or CPU, small FFT-bound cell,
fp64". Two of the sweep's headline ideas turned out to be already implemented: the
Gong and Dal Corso trick of batching the H-apply FFTs across all bands and k-points
into one call (arXiv:2412.01695, worth 6x on their small-cell many-k H-apply) is
exactly what `core/batch.py` already does over `(nk, nb, grid)`, and the CPU FFT is
already on MKL rather than pocketfft, so the "free 1.5-2x pocketfft to MKL" swap is
not available here. What remains, ranked by how well it fits this code:

MEASURED on the RTX 3050 (2026-07-16, torch.profiler on 8 NC SCF iterations, aten-op
device time, no kernel double-count). This revises the "FFT-bound" framing for the GPU
small-cell regime, which came from CPU profiles and the molecule-in-large-box / USPP-Pt
cases. For an ordinary small crystal on the GPU the FFT is only about 12 percent:

    Si8 2x2x2 (nband 20, m~40, box 27^3): GPU-busy 2111 ms, launch/sync gap 996 ms
      = 32% of wall.  GEMM(bmm) 43%, eigh 21%, QR/ortho 14%, FFT 12%, other 10%.
    Si2 4x4x4 (nband  8, m~16, box 20^3): GPU-busy  652 ms, launch/sync gap 559 ms
      = 46% of wall.  QR/ortho 44%, GEMM 23%, FFT 12%, eigh 11%, other 10%.

Two things fall out. First, a small-cell GPU SCF is dense-linear-algebra-bound, not
FFT-bound: GEMM + eigh + QR are about 78 percent of GPU-busy time (small boxes make the
FFT cheap, and fp64 GEMM/eigh/QR pay the same 1/64 fp64 tax). Second, the launch/sync
gap is 32-46 percent of wall (profiler-inflated but consistent with the earlier finding
that eager dispatch of dozens of tiny kernels per Davidson round is the binding GPU
constraint) - that gap is exactly what a whole-step CUDA graph reclaims. The eigh cliff
is visible: eigh 11 percent at m~16 vs 21 percent at m~40 (the n>32 cusolver-batched
fallback, measured 2.5-4.5x on its own). Reprioritized by this data: (1) whole-step CUDA
graph to close the 32-46 percent launch gap, (2) cut the dense subspace LA - RMM-DIIS is
now attractive because it removes the Rayleigh-Ritz (eigh) AND the subspace
orthonormalization (QR), together 35 percent (Si8) to 54 percent (Si2) of GPU-busy - and
a c64 subspace reduction on the NC standard problem would dodge both the fp64 tax and the
eigh cliff, (3) the FFT is no longer the thing to chase on GPU small cells.



- Whole-SCF-step CUDA-graph capture of the dispatch-bound glue. The measured GPU
  negatives so far were an apply-only CUDA graph (1.0-1.1x, the back-to-back FFT
  kernels have no launch gap) and torch.compile on the XC functional in isolation.
  Neither touched the 55-65 percent of a step that is many-tiny-kernel real-valued
  glue between the FFTs (XC assembly, mixing, occupations, PAW one-center, density
  build). Capturing the whole step as one CUDA graph (the PyGraph line,
  arXiv:2503.19779, averages 1.18x and never regresses where naive reduce-overhead
  degrades up to 32 percent) removes the per-kernel launch overhead across that glue,
  which is exactly where an 8-core host plus a consumer GPU hurt most. CUDA-graph
  capture, unlike torch.compile fullgraph, tolerates the complex FFTs (the earlier
  apply probe captured them fine), so the whole step is capturable. It cannot speed
  the FFTs themselves. Estimate 1.2-1.5x on the non-FFT fraction, GPU only, needs
  measuring on the RTX 3050. Highest-value new software lever.
- The batched `eigh` size cliff (diagnostic, cheap). `davidson_batched` calls
  `torch.linalg.eigh` on the `(nk, m, m)` subspace matrix with `m` about `2*nband`.
  On CUDA the fast `cusolverXsyevBatched` path is used only for `n <= 32`; above that
  PyTorch loops per-matrix (measured about 83x slower at the boundary, pytorch#175585).
  Every real system has `m > 32`, so the subspace diagonalization is probably on the
  slow per-k loop on the 3050. It is only about 5 percent of the CPU profile, but the
  cliff can inflate it on GPU. A ten-minute microbenchmark on asus settles whether it
  matters; if it does, cap or tile the subspace or split the batched solve.
- ML density initializer, plane-wave-native. "Global Plane Waves From Local
  Gaussians" (arXiv:2601.19966) and a transferability study (arXiv:2509.25724) report
  25-33 percent fewer SCF iterations, and show a density init transfers out of
  distribution where an ML-Hamiltonian init collapses. It only cuts iteration count,
  not per-iteration FFTs, so about a 1.3x ceiling on a single point, but it stacks
  with everything and its training set is the same shape as the learned-XC data. For
  MD and relaxation the cheaper analog is wavefunction/Grassmann extrapolation across
  geometries (about 3 iterations per step, JCTC 2022 1c00751), which QE and VASP
  already do and gradwave's warm-start approximates.

Skip, from the same sweep, because they do not transfer: distributed GPU eigensolvers
(ELPA, ChASE, SIRIUS all lose on small subspace matrices), ML Hamiltonian predictors
and learned preconditioners (they need a localized basis; our kinetic preconditioner
is already analytic), tensor-core FP16 FFT (accuracy-fatal against QE-grade fp64),
FP8-emulated fp64 FFT (Blackwell-only, no FP8 on Ampere), NUFFT (our grid is uniform),
and VkFFT (wins only at large-prime grids; `good_fft_size` restricts to 2*3*5*7
radices cuFFT already handles). RMM-DIIS is the one prototype-worthy eigensolver, it
removes the Rayleigh-Ritz that CheFSI could not, but the RR is cheap at small cell
size so the win is uncertain. The through-line matches the earlier audit: on a single
small SCF the consumer-GPU fp64 tax is the wall, and the durable levers are throughput
(batch many small structures), fewer iterations (learned or extrapolated start), and a
datacenter fp64 GPU.

## Learned multi-pole density-mixing preconditioner (PROTOTYPED)

The 2024-2026 sweep above skips "learned preconditioners" on the grounds that they
need a localized basis and our kinetic preconditioner is already analytic. That
reason is about *eigensolver* preconditioners. A learned *density-mixing*
preconditioner is a different object, and this section is the prototype of it. It
lives entirely in G-space, needs no localized basis, and generalizes the Kerker
and local-TF filters the code already ships. It is also the lever
`docs/manual/wisdom.md` points at twice over: prefer a preconditioner to
step-size control, and the SCF iteration count is set by density mixing, not by
the initial wavefunction (the reason the atomic-orbital seed in the done section
saved nothing).

The mechanism (`scf/learned_precond.py`). Bare Kerker, R̃(G) = R(G)·G²/(G²+q0²),
is the single-pole long-wavelength approximation to the exact response
preconditioner ε⁻¹ = (1 − v_c χ₀)⁻¹. `MultipoleKerkerPrecond` replaces the one
pole with a learned sum, f_θ(G²) = Σ_i w_i·G²/(G²+q_i²), applied per density-sphere
component exactly where the mixer applies Kerker (wired as `scf(..., precond_op=)`
and `mixer.precond_op`, mirroring local-TF). Two Kerker properties carry over by
construction and both matter: f_θ(0) = 0, so the pinned G=0 charge is never
touched, and the fixed point is unchanged, so a bad filter can only cost
iterations, never accuracy. K=1, w=1 reproduces bare Kerker to round-off, so the
single pole is always inside the hypothesis class.

The fit is where a differentiable solver does something a non-differentiable one
cannot. The error of preconditioned linear mixing evolves component-wise as
e_{n+1}(G) = [1 − α·f_θ(G²)·d(G)]·e_n(G), with d(G) = 1 − j(G) the diagonal
response denominator. `fit_multipole` unrolls that recurrence and backpropagates
‖e_N‖ to the pole weights and positions, training the preconditioner against the
solver's own linearized response. `response_from_residuals` estimates d(G) per
|G|-shell from a short plain-mixing SCF captured through the new `scf` `mixer_hook`,
so probe, fit, and deploy all run on real solver output.

Measured (`benchmarks/bench_learned_precond.py`, `tests/unit/test_learned_precond.py`).
On a synthetic response with two length scales, where a single pole is provably the
wrong shape, the fitted three-pole filter drops the spectral radius from 0.82 to
0.50, a 3.5x iteration ratio. On real fcc Al the end-to-end loop runs and the
learned filter reaches the Kerker energy to 5e-12 eV (fixed point unchanged, as
designed), but it takes 9 iterations against Kerker's 7. That is the honest and
expected result on a homogeneous metal, and it is the same story bench_precond.py
tells for local-TF on bulk Al: Kerker is already near-optimal there, and there is
nothing for a radial filter to win.

The Al loss is also diagnostic, and names the next step. The fit minimizes the
*plain-mixing* spectral radius, but deployment runs Pulay DIIS with history 8,
which already accelerates the low-G modes the filter spent its weight on. So the
linear-model fit over-invests where DIIS is already strong. The moat's own logic
says the fix: unroll the *actual* mixer, not the plain-damped linear surrogate, so
the filter is trained to complement DIIS rather than to duplicate it. gradwave can
differentiate through the Pulay recurrence; nothing else in the field can.

Where the headroom is, and where it is not. Not bulk metals (Kerker plus DIIS is
near-optimal). The candidates are systems whose G-space response genuinely carries
more than one scale and whose extra structure survives DIIS: semicore metals
(Cu 3s3p), intermetallics (Cu₃Al), and the DIIS-limited regimes where history is
short or reset often (large cells near the charge-sloshing cliff, ferromagnetic
metals near the Stoner boundary where wisdom.md already asks for the χ₀-diagonal
operator by name). A cleaner probe would help too: the current d(G) estimate ran
with Kerker off and picked up values above one, so dividing the response out of a
converging Kerker run, or unrolling DIIS directly, is the more trustworthy path to
d(G). The prototype settles that the machinery is correct and safe and that the
fit-through-linear-model does not transfer under DIIS on easy systems; the open
question is whether a real multi-scale system plus a DIIS-aware fit clears the bar.

# Done and resolved

Kept for the reasoning. Each of these is either landed in the code or settled as a
measured negative.

## Magnetic space groups (Shubnikov symmetry) for non-collinear k-reduction

Every magnetic non-collinear run today uses the FULL k-mesh — `scf_noncollinear`
(and the spinor PAW loop) refuse `use_symmetry` for any nonzero m⃗, because the
existing spglib machinery only knows the paramagnetic group. That is the safe
choice, not the cheap one: the FePt MAE runs carry 384 unreduced k-points, and the
k-cost is the whole reason the force-theorem item above matters.

The physics: a finite m⃗ changes the symmetry group itself. Time reversal dies (no
k ↔ −k Kramers folding — the code already handles the *nonmagnetic* SOC case, where
TR survives and the IBZ test pins it). And the moment filters the point group,
because m⃗ is an axial vector (transforms as det(R)·R) locked to the lattice by SOC:
an operation survives only if it maps the magnetization field onto itself.
Operations that flip m⃗ survive only *combined with time reversal* — the anti-unitary
half of the magnetic (Shubnikov) group, which relates band energies at Rk without
being a unitary symmetry of H. Concretely for L1_0 FePt (paramagnetic D4h, 16 ops
+ TR): moments along c leave the unitary C4h (8 ops, ~8x k-reduction); moments
in-plane leave ~C2h (4 ops, ~4x). The easy-axis state is literally more symmetric
than the hard-axis one — the anisotropy, seen group-theoretically.

What to build, on top of the existing spglib path: (1) filter the space group by the
axial-vector action on the moment field (and classify the surviving anti-unitary
R·T elements); (2) symmetrize (ρ, m⃗) with m⃗ transformed as an axial vector — and
the on-site becsum's four Pauli channels likewise, mirroring `becsum_sym`; (3) fold
k with the magnetic little groups, using the anti-unitary elements for band-energy
relations only. spglib ships magnetic space-group (Shubnikov) support since 2.0, so
the group identification is available off the shelf; the work is the axial-vector
symmetrization and the k-folding bookkeeping.

**LANDED (2026-07-18).** All four phases are in: (1) `magnetic_spacegroup(sg,
magmoms, cell)` in symmetry.py — the axial-vector filter det(S)·S·m⃗ classifying
each paramagnetic op as unitary / anti-unitary (op·T) / dropped, cross-checked
against spglib.get_magnetic_symmetry; (2) `reduce_mesh_magnetic` — the shared
orbit fold with unitary {W⁻ᵀ} ∪ anti-unitary {−W⁻ᵀ}, grey group (m⃗=0)
reproducing the paramagnetic+TR fold bit-for-bit; (3) `MagneticSymmetrizer`
(grid ρ, m⃗: RhoSymmetrizer maps on the combined op list + per-op axial 3×3 with
s_T=−1 on the anti set) and `MagneticBecsumSymmetrizer` (BecsumSymmetrizer D^l
blocks + the same axial across the Pauli channels + conj on anti ops); (4)
`setup_system`/`setup_uspp` take `magmoms=`, both spinor loops consume the
magnetic system and re-symmetrize (ρ, m⃗[, becsum]) each iteration, and the
collinear loops reject magnetic systems. Measured folds: FePt m∥[001] (6,6,4)
144→30 k (equals the para+TR IBZ — inversion is unitary for axial vectors);
m∥[100] 144→48; bcc Fe m∥z 64→13. Validation (tests/unit/
test_magnetic_symmetry.py, tests/integration/test_magnetic_ibz.py): SOC FePt
magnetic IBZ ≡ full mesh to 5.0e-11 eV; polar (inversion-broken) FePt exercises
the anti-unitary-only fold (27→6 where unitary ops alone give 9); spinor PAW Si
grey group ≡ symmetrized collinear scf_uspp to 5.1e-11 eV. The force-theorem
MAE evaluator on the magnetic IBZ is the natural next stage.

The caveat the original plan carried — "do not reduce each orientation to its
own IBZ for MAE differences" — turned out to be wrong for this folding: the
magnetic-IBZ sum is exactly the full-mesh sum re-weighted (measured 5e-11 eV,
five orders below the meV signal), so each orientation's k-discretization error
is identical to its full-mesh value and the common-mode cancellation in
E(hard) − E(easy) survives per-orientation folding untouched. The caveat only
bites if the two orientations use *different underlying meshes*. Keep the same
(n1,n2,n3) mesh for both and fold each by its own magnetic group ([001]→30 k,
[100]→48 k at (6,6,4)): 3.7× on the MAE pair, exactness preserved.

## RMM-DIIS solver and whole-step CUDA graph (both TRIED, measured negatives)

Prompted by the GPU profile above (dense-LA-bound, 32-46 percent launch gap), two of
the three levers it suggested were built and measured, and neither pays for small
cells.

RMM-DIIS (a `solvers/rmm_diis.py` prototype, since removed) replaces the block
Davidson's growing Rayleigh-Ritz subspace with per-band residual minimization, so it
has no per-round eigh and no m x m subspace GEMM - the 64 percent the profile flagged.
It needed two fixes to converge at all: a units-correct preconditioner (teter_b is a
dimensionless filter, right for Davidson subspace expansion but not for a direct Jacobi
step) and an exact line search (Teter-Payne preconditioned CG - a fixed step does not
converge). After both it converges on a FIXED operator (synthetic batched Hermitian, err
2e-11) but in about 100 iterations to the block Davidson's 22, at two H-applies per
iteration. In the real SCF it is worse than slow: on smeared fcc Al it hit the iteration
cap without converging and returned the wrong energy (-368 vs -1828 eV), at 1,548,800
band-applies against Davidson's 10,512 (147x). The reasons are exactly the textbook
ones: subspace methods converge in far fewer iterations, the SCF drives the solver with
a loose-early tolerance schedule that a residual method handles poorly, and a metal's
near-degenerate bands break the per-band tracking. RMM-DIIS is a large-system solver
(where the O(N^3) subspace eigh/GEMM finally dominates) and an MD warm-start refiner, not
a small-cell Davidson replacement - and the dense LA it removes, while 64 percent of GPU
time, is cheap in absolute terms at small cell size. Removed the prototype.

The whole-step CUDA graph is blocked upstream: `torch.linalg.eigh` is not CUDA-graph
capturable (it does a host-side info check), and it sits in the Davidson inner loop every
expansion round, so a whole-step capture fragments into tiny pieces around each eigh
rather than removing the launch gap. It genuinely needs the eigh out of the hot loop,
which was RMM-DIIS's job, and RMM-DIIS is not viable here. So the 32-46 percent launch
gap is real but not reclaimable by either lever without a solver that avoids eigh
altogether. Net: the measured GPU bottleneck resists these fixes because the eigh is both
cliff-hit and non-capturable and removing the subspace method costs convergence speed;
the durable levers stay throughput batching and a datacenter fp64 GPU.

## Local Thomas–Fermi metal preconditioner (DONE)

Landed as opt-in `precond="local_tf"` on both `scf` and `scf_uspp`
(`scf/local_tf.py`, default `"kerker"`). The bare Kerker filter screens charge
sloshing with a single length `1/q0`, right for a bulk metal but wrong for an
inhomogeneous cell, where a fixed `q0` over-screens the vacuum. Following QE's
`mixing_mode='local-TF'`, `LocalTFPrecond` lets the screening wavevector track the
local density, `q²(r)=min(q²_TF(r), q0_max²)` with `q²_TF=(4/π)k_F(r)`, capped at
the bare `q0` so a bulk metal is unchanged. It is applied by a short
preconditioned-CG solve of the screened-Poisson operator (a few box FFTs per
mixing step, warm-started across iterations), acting on the ρ-total block only.

Measured (NC, fcc Al, PBE, gaussian 0.1 eV): energies bit-identical to bare Kerker
(same fixed point). Bulk 8×8×8 neutral (9→9), Al(100) slab 21→17 (4 layers) and
27→21 (6 layers) iterations, the margin growing with cell inhomogeneity, exactly the
inhomogeneous regime the operator targets. So the original framing was right that a
fixed Kerker is the wrong operator away from a uniform bulk, but the win is on slabs
and molecules, not on a homogeneous bulk metal, where Kerker at a sensible `q0` is
already near-optimal. The bulk-Pt 16-vs-7 iteration gap is therefore a
starting-density and Broyden-history question more than a screening-length one, and
this preconditioner does not by itself close it. Unit tests pin the three operator
limits (`tests/unit/test_local_tf.py`), integration tests gate the fixed-point
invariant on NC and USPP (`tests/integration/test_local_tf_scf.py`), and the
slab iteration-count win lives in `benchmarks/bench_precond.py`.

Two follow-ups worth noting. First, building this surfaced and fixed a separate
bug: `setup_uspp` sized the FFT box as a blanket cube for any symmetric cell, so an
anisotropic slab got a 105³ box instead of 20×20×105, a 27.6× over-allocation that
OOMs during setup, now fixed by porting the NC path's symmetry-coupled axis grouping
(`symmetry.coupled_axis_groups`). Second, the modern parameter-free successor to
local-TF is the LDOS preconditioner of Herbst and Levitt (arXiv:2009.01665, DFTK's
default), which adapts the screening to whether each region is metallic or
insulating from the local density of states rather than a Thomas–Fermi model. If
local-TF ever underdelivers on a strongly mixed metal-vacuum-insulator cell, that is
the next rung, and it reuses the same reciprocal-space mixing hook.

## torch.compile for the exchange-correlation layer (DONE)

Landed as the opt-in `compile_xc` flag (`GradWave(compile_xc=True)` or
`xc.enable_compile()`). Measured 19x forward and 16x forward-plus-`v_xc` at 64³,
`v_xc` bit-accurate to 3e-16, with an eager fallback for the missing NixOS
toolchain. Compiled aot_autograd cannot double-backward, so the `f_xc` response
and HVP sites wrap their `xc.energy()` in `xc_eager()` to stay eager, which means
only the forward and first-order `v_xc` legs accelerate. Details in
`docs/manual/performance.md`.

The original analysis, kept for the reasoning. The compiler is dead on the complex,
FFT-bound Hamiltonian apply, which two earlier attempts already confirmed, but the
real-valued XC functional was never isolated and compiles well on a 64^3 grid. The
end-to-end effect on a plain SCF is only a few percent because XC is a minority of
runtime and its FFT-based gradient assembly does not compile, but learned-XC
training, the PAW one-center angular loop, and the `f_xc` response HVPs call the XC
transcendental chain far more than once per iteration and are CPU-bound, so those are
the real targets. Insertion point is the single `XCFunctional.energy_density` choke
point, opt-in with an eager fallback for the NixOS toolchain gap.

## Dual FFT grid (DONE)

Landed as commit `71a5265`, about 2x on the USPP/PAW H-apply FFT by running the
smooth wavefunctions on a coarse grid and the augmentation on the dense grid,
matching the audit spec.

## CheFSI, benchmarked no-go on the RTX 3050 (DONE)

Chebyshev-filtered subspace iteration is in `solvers/chebyshev.py`, unit-tested and
wired opt-in as `scf(..., eigensolver="chebyshev")` on the NC collinear path,
bit-identical to Davidson on the real NC SCF regression. The noncollinear spinor
twin was tried but left unwired, CheFSI converges too slowly on the dense metal
spinor spectrum (100-iteration cap vs Davidson's 18). The RTX 3050 fp32-deep
benchmark found it 2.5 to 5x slower than Davidson at every grid size that fits in 6
GB, up to 35^3. The fp32 FFT advantage there is only about 3.4x, not the 12x the
larger systems would need, and CheFSI does 2 to 3x more H-applies, so the filter
loses. It stays opt-in and off by default. Revisit on a bigger card where the grid
can grow into the regime where the fp32 FFT gain dominates, which is the same
hardware caveat the scaling section above opens with.

## Batched Davidson conditioning guard, cond-SVD removed (DONE)

The k-batched USPP/PAW generalized Davidson computed a full `linalg.cond` of the
subspace overlap every round on top of the `cholesky_ex` it already ran. Probing a
low-ecut Si PAW SCF (8, 10, 12 Ry) showed the overlap tips into non-PD, which
`cholesky_ex` flags with info>0, long before its condition number nears the 1e14
trip (max observed ~9e7), so the SVD never fired independently and was pure cost.
Removed it. Batched-vs-per-k equality (identical eigenpairs) and USPP/PAW-vs-QE
regression still pass, including nspin=2 PAW (O2 triplet, 7e-12 eV). Recorded in
`docs/manual/wisdom.md` under Eigensolvers.

## Extended-xyz trajectory output for relax (DONE)

`run_relax` accumulates an ASE frame per optimizer step with energy and forces
frozen on a `SinglePointCalculator`, and `run` writes them to `relax.xyz` (extxyz)
next to the JSON, re-readable in ovito or the ASE gui. The relax CLI now returns
exit 0 on normal completion, since reaching the ionic-step limit still yields a
valid trajectory, with convergence carried by the JSON `relax.converged` flag.
Regression in `tests/integration/test_io.py::test_relax_writes_extxyz_trajectory`.
MD does not have an output path yet, so the same frame accumulation extends there
once it lands.

## Atomic-orbital seeding for the initial wavefunctions (TRIED, no net gain)

The idea was to hand the first Davidson solve a superposition of pseudo-atomic
orbitals instead of bare lowest-kinetic plane waves. `scf/loop.py` builds `c0` as an
identity block on the first `nb` sphere entries, the smoothest plane waves and
nothing about the atoms, poor enough that the loop runs the first diagonalization at
a loose `1e-3` tolerance before tightening. QE's default instead projects the atomic
pseudo-wavefunctions onto the plane-wave basis (`startingwfc='atomic'`). All the
pieces existed in-tree, the `upf.pswfc`/`paw.chi` orbitals and the SBT-and-Ylm
projector build shared with the KB, Hubbard, and PDOS paths.

Built `lcao_seed` (per-k atomic-orbital block, QR-orthonormalized to 8e-15, padded
with plane waves past the orbital count) and wired it at the `c0` site. It reaches
the plane-wave-seeded energy to machine precision, as it must (NC O2 gives dF = 5e-12
eV, fcc Ni gives dF = 3e-11 eV). The predicted one-to-three iteration saving is real
but small (O2 goes 28 to 26 iterations, fcc Ni 6x6x6 goes 12 to 12), and the per-k
seed build costs enough that wall time came out neutral to slightly worse (Ni, 108 s
to 122 s). The reason is the one the prediction named. The loop already runs the
first diagonalization at a loose 1e-3 tolerance, so a crude plane-wave start
converges the cheap early eigensolves fine, and the total SCF count is set by density
mixing, not by initial-orbital quality. Reverted the wiring rather than add per-k
overhead to the default path for no measured gain. Recorded in
`docs/manual/wisdom.md` under SCF and mixing.

The remaining reason to revisit is that it composes with CheFSI, whose convergence
rate depends directly on how much of the wanted subspace is already in the start. A
Chebyshev filter fed atomic orbitals needs fewer rounds than one fed smooth plane
waves, so the pair should be measured together. That is the only configuration where
the seed cost might be repaid, and it is worth building `lcao_seed` back only
alongside a CheFSI-default benchmark that shows the compound win.
