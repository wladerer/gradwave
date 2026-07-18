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

Rules learned the hard way, now policy:

- Gate on **low-symmetry, rattled, mixed-occupation** states. Symmetric
  fixtures let error terms cancel (displaced-Si passed while `ddd` was
  wrong; O₂ exposed it).
- Match the reference's occupation scheme when comparing forces (smeared vs
  fixed differ legitimately at 7e-3 eV/Å).
- A derivative test against QE is not a derivative test against our own
  energy. Do both.

## Tier 1: metamorphic invariance battery (next)

Exact identities of the theory under input transformations. The metamorphic
testing literature (Kanewala; MorphQ found 14 confirmed Qiskit bugs with no
oracle at all) says to impose them per layer, not only end-to-end, so
compensating errors cannot hide.

- **Supercell identity**: E(N×1×1 supercell at folded k) == N·E(primitive)
  to solver tolerance. Same ecut means the supercell basis is exactly the
  union of the folded primitive bases, so this is an identity, not a
  convergence statement. One test exercises k-weights, Fermi level, Hartree
  G=0, smearing entropy, and symmetrization at once.
- Rigid translation (E invariant, forces mapped rigidly), atom permutation,
  equivalent-cell re-parameterization (Niggli/rotated frame), k → −k,
  random U(N) rotation inside degenerate/occupied subspaces (ρ, E, forces
  invariant).
- Already present in this spirit: IBZ == full mesh (+0.0000 meV), rotation
  invariance of noncollinear (0.19 µeV), collinear limit, ζ=0 limit.
- Upgrade path: property-based randomization (seeded random low-symmetry
  structures per CI run) instead of fixed fixtures.

## Tier 2: exact solutions and convergence orders

- Empty lattice (V=0): free-electron bands to machine precision. Isolates
  kinetic + sphere + FFT machinery.
- 1D cosine potential: analytic Mathieu band structure through the full 3D
  solver.
- Isolated pseudo-atom in a box vs our own radial atomic solver (the SAD
  path): 3D machinery against 1D quadrature.
- Order-of-accuracy assertions: measure the convergence exponent of each
  quadrature (Simpson O(h⁴) radial, angular XC grid, SBT) and assert the
  *rate*, not just smallness. Salari & Knupp's blind study: planted bugs
  that answer-comparison missed were caught by wrong observed order.
- Sum rules as measured diagnostics, never silent fixes: ASR and Born
  neutrality residuals (we already treat the raw 4×4×4 violation as
  physics), f-sum rule for ε∞.

## Tier 3: external anchors that are not QE

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

- +U (V_U from Dudarev D at random occupation matrices vs grad of E_U
  through `occupation_matrices`)
- USPP/PAW (generalized eigenproblem: grad E == 2wf(Hc − εSc) needs the
  S-side treated explicitly; screened D and `ddd_paw` from the previous
  becsum lag one iteration in the SCF by design, so gate the consistent
  pair, not the lagged one)
- SOC spinors (doubled pw axis; same contraction)
- smeared occupations as functions of ε (adds the entropy chain; the fixed-f
  identity is what the SCF actually iterates, so this is optional)

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
