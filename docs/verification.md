# Verification: correct vs accidentally correct

Working document. What it takes to establish that gradwave is *correct*, as
opposed to *in agreement with Quantum ESPRESSO*. Those are different claims,
and we have first-hand evidence of the difference: the PAW one-center `ddd`
matched QE forces to 1.3e-4 eV/Å while being 0.05–1% away from the true
derivative of our own energy, because both codes shared the same lm-truncated
integration-by-parts inexactness. Finite differences of our own energy caught
it; QE could not have.

## Framing from the V&V literature

The computational-science verification and validation literature (Roache;
Oberkampf & Trucano 2002; Oberkampf & Roy 2010) separates three activities:

- **Code verification**: is the discretized math solved correctly? A purely
  mathematical question with mathematical oracles (exact solutions,
  manufactured solutions, convergence orders, internal identities).
- **Solution verification**: how large is the numerical error of a given run
  (cutoff, k-mesh, SCF tolerance)?
- **Validation**: do the equations describe reality (functional accuracy vs
  experiment)? Not our problem when matching a fixed functional.

Comparison against another code is a "pseudo-oracle" (Kanewala & Bieman 2014)
and Oberkampf & Trucano rank it as the weakest form of verification evidence,
for exactly the shared-fault reason. The DFT community's own studies say the
same: the Δ-gauge paper (Lejaeghere et al., Science 2016) and the ACWF study
(Bosoni et al., Nat. Rev. Phys. 2024) frame cross-code agreement as measuring
*precision*, not correctness. Two details worth keeping in mind:

- Code agreement is historical, not automatic. The literature spread of the
  PBE Si lattice constant shrank from 0.05 Å to 0.01 Å only after explicit
  verification effort. Agreement at a point in time is not convergence to
  the exact answer.
- Even the two all-electron reference codes (WIEN2k, FLEUR) disagree
  residually (~0.2 meV/atom on Os), traceable to convention choices such as
  the scalar-relativistic scheme. Codes sharing a convention agree more
  closely regardless of whether the convention is right.

A test suite establishes correctness to the extent that its oracles are
independent of the implementation under test *and* of the reference code.
QE comparisons stay valuable (they catch plenty, and they pin conventions);
they are the outer ring, not the foundation.

## Tier 0: self-consistency identities (implemented)

Mathematical identities relating the code's own outputs, checked at states
where nothing cancels. These hold at any cutoff/grid/k-mesh because they are
properties of the discretized functional, not of converged physics, so the
test systems can be tiny and fast.

**Off-stationarity E↔H gate**
(`tests/unit/test_energy_hamiltonian_consistency.py`). The KS Hamiltonian is
by definition the derivative of the energy functional. At random,
non-self-consistent orbital coefficients:

    grad_c E_total  ==  2 · w_k · f_nk · (H c)_nk        (< 1e-10 relative)

with E assembled exactly as the SCF assembles it (`density_b`, `becp_b`,
`total_energy`) and H being the SCF's own `BatchedHamiltonian` at
`effective_potentials(...)` (the loop's v_eff assembly, extracted so the test
gates the identical code path). Covered: LDA, PBE (the σ chain), NLCC
(ρ_core in the XC argument), nspin=2 per channel, plus a closed-form-vs-
autograd check on v_H and v_loc. A deliberately mismatched E/H pair (LDA vs
PBE) registers at 1.8e-3, seven orders above the gate, so the test has teeth.

The USPP/PAW version of the gate uses `_HkS` at
`uspp_potentials_dscr(...)` (the USPP loop's potential + screened-D
assembly, extracted the same way): grad E == 2wf(Hc) where H's
D = dij + ∫v_eff Q̃ + ddd(becsum), with v_eff from ρ[c] including the
augmentation density and ddd from becsum[c]. No S term appears: at
unconstrained coefficients the energy's gradient IS Hc; εSc only arises
from the orthonormality constraint at stationarity. The PAW variant is the
exact term class of the original `ddd` bug — it passes only because ddd is
autograd of `e1c_t` at the same becsum, and the gate composes that with the
full Q̃/phase/ρ_aug orbital chain.

Why off-stationarity matters: at a converged SCF point, an E/H inconsistency
is second-order in the error and invisible; against QE it can be exactly
invisible when shared. At a random state it is first-order and glaring. The
`ddd` bug and the spin-GGA factor-2 vector-field bug were both of this class.
Energy-only tests cannot see them.

**Existing tests in this tier** (grown case by case; keep the policy that
every new energy term lands with one):

- forces vs FD of our own energy (M2 gates; PAW `hubbard_force` 4.85e-8;
  O₂ PAW spin 5e-4)
- stress vs strain-FD and vs QE (0.006 kbar NC, 0.13 kbar PAW)
- NVE drift (secular ~1 µeV/atom: the force is the exact gradient of the
  SCF surface)
- becsum-FD of the one-center energy vs `ddd` (5e-9, post-rewrite)
- dE/dU by Hellmann-Feynman vs FD SCF re-runs (7.4e-7)
- gradcheck on unit-level autograd paths
- volumetric export identities (`tests/integration/test_volumetric.py`)
- D3(BJ) dispersion (`tests/unit/test_dispersion.py`): forces and stress vs
  FD of the dispersion energy itself on a rattled, low-symmetry, mixed-element
  (H/C/N/O) cell (both at the 1e-6-relative FD floor), plus an independent
  scalar-loop transcription of the reference D3(BJ) expression (matches the
  vectorized+autograd energy to 1e-10), a positions gradcheck, and the
  Σ F = 0 / translation-invariance sum rules. The one external anchor is the
  simple-dftd3 tutorial water–peptide dimer (PBE0-D3(BJ), two-body):
  reproduced to 4e-11 Ha — Tier-3 cross-code, but here it is machine-level
  because both read the same published reference C6 tables.

**PARCHG↔CHGCAR sum identity**
(`tests/integration/test_volumetric.py`). The band-decomposed density and the
total density come from the same coefficients, so their occupation-weighted sum
is an identity of the discretized functional, not converged physics:

    Σ_k w_k Σ_n f_nk |ψ_nk(r)|²  ==  ρ(r)        (< 1e-10, elementwise on the grid)

The test reconstructs each |ψ_nk(r)|² through the export path (`g_to_r` on the
stored coefficients) and compares against `res.rho` from the SCF's own `density_b`.
It held at 5e-16 on a Si mesh. The companion normalizations — ∫ρ dr = n_electrons and
∫|ψ_nk|² dr = 1 — are the same class, and ELF is bounded to [0,1] by construction.
The identity catches a wrong G-sphere index, a spinor up/down mixup, or a missing
1/Ω in the export before any file reaches VESTA.

The charge-response field carries its own identity: ∫ ∂n(r)/∂R_I dr = 0, because
moving an atom conserves the electron count. The finite-difference response
(`density_response_fd`) returns this residual with the field and it lands at 1e-11 on
a rattled Si cell, an end-to-end check that the two displaced SCFs converged to the
same electron count on the same grid.

Rules learned the hard way, now policy:

- Gate on **low-symmetry, rattled, mixed-occupation** states. Symmetric
  fixtures let error terms cancel (displaced-Si passed while `ddd` was
  wrong; O₂ exposed it).
- Match the reference's occupation scheme when comparing forces (smeared vs
  fixed differ legitimately at 7e-3 eV/Å).
- A derivative test against QE is not a derivative test against our own
  energy. Do both.

## Tier 1: metamorphic invariance battery (started)

Exact identities of the theory under input transformations. The metamorphic
testing literature (Kanewala; MorphQ found 14 confirmed Qiskit bugs with no
oracle at all) says to impose them per layer, not only end-to-end, so
compensating errors cannot hide.

- **Supercell identity** (implemented,
  `tests/integration/test_supercell_identity.py`): E(2×1×1 supercell at Γ)
  == 2·E(primitive at k=(2,1,1)) on a rattled P1 geometry, with the
  supercell FFT grid pinned to 2× the primitive grid so the quadratures
  sample identical points. Same ecut means the supercell basis is exactly
  the union of the folded primitive bases, so this is an identity at solver
  tolerance, not a convergence statement (passes at <2e-6 eV/atom).
  Eigenvalues fold (Γ supercell == sorted union over folded k) and forces
  map rigidly onto the copies. One test exercises k-weights, Fermi filling,
  Hartree G=0 ownership, nonlocal phases, and the density assembly at once.
  The USPP/PAW + smeared variant is implemented too (kjpaw Si, gaussian
  smearing): per-copy becsums/one-center energies replicate, the
  augmentation density folds through the supercell phases, and the
  shared Fermi level fills the folded spectrum identically (marked
  slow). Extensions: N along other axes.
- Permutation + translation (implemented,
  `tests/integration/test_metamorphic_invariance.py`): atom relabeling is
  exact to SCF tolerance (tested tight on heteropolar GaAs, <5e-8
  eV/atom); rigid translation is invariant up to the XC-quadrature
  aliasing (egg-box) floor, which the test MEASURED rather than assumed:
  Si 5.6e-7 eV/atom at 14 Ry and 4.6e-6 at 20 Ry (non-monotonic — the
  minimal FFT box changes shape with ecut), GaAs semicore 9.1e-5 at 25 Ry
  → 2.5e-6 at 60 Ry. A useful solution-verification number in its own
  right: egg-box forces at 20 Ry Si are ~1.5e-4 eV/Å, a floor for
  relaxation convergence criteria.
- k → −k (implemented, same file): on a shifted MP mesh over a rattled P1
  cell only time reversal relates the ±k pairs; the TR-halved mesh
  reproduces the full mesh to <5e-8 eV/atom and 1e-6 eV/Å — gates the
  H(−k) = H(k)* conjugation/phase conventions end to end.
- Cell re-parameterization (implemented, same file): the same crystal in a
  rigidly rotated Cartesian frame — every Cartesian intermediate (g_cart,
  the σ chain, Ewald, projector Ylm's) re-indexes while E is invariant and
  forces co-rotate (<5e-8 eV/atom, 1e-6 eV/Å). Still to do: the
  same-lattice different-primitive-vectors variant (Niggli); the
  reciprocal lattice and G-sphere are identical, but a fractional k-mesh
  maps onto different k-points, so it needs explicit k-point control.
- U(N) gauge rotation (implemented, `tests/unit/test_gauge_invariance.py`):
  a random unitary inside an equal-occupation block at each k leaves ρ
  (<1e-12) and every energy term (<1e-10 eV) invariant at random
  off-stationary coefficients, with an unequal-occupation control proving
  the machinery would see a violation.
- Already present in this spirit: IBZ == full mesh (+0.0000 meV), rotation
  invariance of noncollinear (0.19 µeV), collinear limit, ζ=0 limit.
- Upgrade path: property-based randomization (seeded random low-symmetry
  structures per CI run) instead of fixed fixtures.

## Tier 2: exact solutions and convergence orders (started)

- Empty lattice (implemented, `tests/unit/test_exact_limits.py`): V=0 on a
  triclinic cell at a generic k gives free-electron bands from the
  production BatchedHamiltonian + Davidson stack to <1e-9 eV. Isolates
  kinetic + sphere + FFT machinery; no lattice symmetry to hide
  indexing/convention errors.
- Cosine potential (implemented, same file): V0·cos(2πx/a) maps to
  Mathieu's equation; band edges at Γ (a_0, b_2, a_2, ...) and X
  (b_1, a_1, b_3, ...) match scipy's characteristic values to <1e-7 eV.
  A nontrivial analytic band structure through the full 3D solver — the
  single-harmonic potential makes the basis error superexponentially
  small, so the comparison is at solver tolerance.
- Isolated pseudo-atom in a box vs a radial atomic solver (the SAD path):
  needs a 1D radial KS solver, which the repo does not currently have —
  future work.
- Order-of-accuracy assertions: measure the convergence exponent of each
  quadrature (Simpson O(h⁴) radial, angular XC grid, SBT) and assert the
  *rate*, not just smallness. Salari & Knupp's blind study: planted bugs
  that answer-comparison missed were caught by wrong observed order.
- Sum rules as measured diagnostics, never silent fixes: ASR and Born
  neutrality residuals (we already treat the raw 4×4×4 violation as
  physics), f-sum rule for ε∞.

## Tier 3: external anchors that are not QE

- **Periodic-table Δ-gauge vs WIEN2k** (`benchmarks/delta_gauge`, implemented):
  22 cubic elements across s/p/d valence, PseudoDojo NC-SR PBE, Δ vs the
  all-electron WIEN2k reference of the Δ-factor benchmark. Median Δ = 0.8
  meV/atom, 21/22 at V0 < 0.6 % and B0 < 4.4 %. Two lessons confirm the notes
  below: (1) the elevated transition-metal Δ (Pt 2.7, Ir 1.9) is the metric's
  **B0 sensitivity**, not error — their fractional V0/B0 are excellent, so the
  ε/ν metrics are the fairer read; (2) Cu is the two-axis exemplar — bad vs
  all-electron (Δ 7.9) but gradwave reproduces QE on the *same* UPF to 0.08 meV,
  isolating the fault to the pseudopotential file (`results/cu_anomaly.md`).
- **ACWF all-electron reference** (acwf-verification.materialscloud.org):
  960 EOS averaged from two independent all-electron codes, with archived
  per-code pseudopotential datasets. Comparing (gradwave − QE-same-pseudo)
  vs (QE-pseudo-dataset − all-electron) separates implementation error from
  the shared pseudopotential error, which our Δ ≤ 0.17 meV/atom vs QE
  cannot do. Use their ε and ν metrics (Δ is overly bulk-modulus
  sensitive; "excellent" is ε < 0.06, ν < 0.1, Δ < 0.3 meV/atom).
- XC exact conditions on `core/xc` (uniform/spin scaling, Lieb-Oxford,
  known limits) in the XCVerifier spirit (arXiv:2408.05316).

## Tier 4: score the suite itself

- **Mutation testing** (`mutmut`/`cosmic-ray` over `core/`, fast suite
  only): surviving mutants localize the code whose only coverage is
  QE-agreement. Restrict to arithmetic/sign/index mutations to bound the
  floating-point equivalent-mutant noise.
- Differentiable-path checks: dot-product (tangent/adjoint) test on
  Sternheimer/implicit-diff solves; implicit vs unrolled vs FD agreement as
  SCF tolerance tightens (the DFTK and DQC methodology).

## Follow-ups for the Tier-0 gate

Extend `test_energy_hamiltonian_consistency.py` to the remaining
Hamiltonian terms, same pattern:

All Hamiltonian term classes are now gated: the +U gate (Dudarev V_U vs
autograd of E_U through the occupation matrices, including the nspin=1
half-occupation bookkeeping), nspin=2 USPP/PAW (per-channel ρ_aug/becsum,
∫v_eff_σ Q screening, spin ddd from energy_and_ddd([ρ↑,ρ↓]) as the exact
channel-derivative of e1c_t), and SOC spinors (the Pauli-decomposition
m⃗-chain vs the B⃗_xc·σ⃗ apply, and the j-resolved nonlocal dij_so
contraction vs E_NL, on the doubled plane-wave axis). Remaining:

- Smeared occupations as functions of ε: done as a Legendre-consistency
  gate (`tests/unit/test_smearing_consistency.py`) — dF/dε_nk = w_k f_nk
  for every smearing scheme through the SCF's own
  `shared_fermi_occupations` (incl. the nspin=2 shared-μ bookkeeping),
  perturbing the most fractional state where the entropy chain is
  maximally active; a deliberately mismatched occupation/entropy pair
  breaks it at first order. This is the identity that makes smeared
  forces Hellmann-Feynman forces of F (Mermin) and the classic trap for
  Methfessel-Paxton / Marzari-Vanderbilt entropy expressions.

## Citations

- Oberkampf, Trucano, Prog. Aerosp. Sci. 38, 209 (2002).
- Oberkampf, Roy, *Verification and Validation in Scientific Computing*,
  Cambridge (2010).
- Roy, J. Comput. Phys. 205, 131 (2005). Roache, *V&V in Computational
  Science and Engineering* (1998).
- Salari, Knupp, SAND2000-1444 (2000).
- Lejaeghere et al., Science 351, aad3000 (2016).
- Bosoni et al., Nat. Rev. Phys. 6, 45 (2024); arXiv:2305.17274;
  acwf-verification.materialscloud.org.
- Prandini et al. (SSSP), npj Comput. Mater. 4, 72 (2018).
- Kanewala, Bieman, Inf. Softw. Technol. 56, 1219 (2014); arXiv:1804.01954.
- Deng, Pradel (MorphQ), ICSE 2023; arXiv:2206.01111.
- XCVerifier, arXiv:2408.05316.
- DFTK algorithmic differentiation, npj Comput. Mater. (2025),
  doi:10.1038/s41524-025-01880-3.
- Kasim et al. (DQC), J. Chem. Phys. 156, 084801 (2022); Zhang, Chan
  (PySCFAD), arXiv:2207.13836.
- BOUT++ MMS, arXiv:1602.06747.
