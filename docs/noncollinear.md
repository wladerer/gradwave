# Non-collinear magnetism + spin-orbit coupling: design

Status: XC leg implemented (`core/xc/noncollinear.py`, validated in the
collinear limit). Spinor SCF and SOC are the next implementation phase.

## Architecture (decided)

1. **Spinors.** Wavefunctions become 2-component: coefficients
   `(nk, nb, 2, npw_max)`. The batched machinery generalizes by folding the
   spinor index into the plane-wave axis for FFT/scatter (two boxes per
   band) and into the Gram matrices for Davidson (subspace dim doubles).
   Time-reversal k-reduction is LOST once SOC or non-collinear m breaks it —
   meshes fall back to full BZ (the k-batched solver absorbs the 2× cost).

2. **Density matrix.** ρ_αβ(r) → (ρ, m⃗) via Pauli decomposition:
   ρ = tr ρ̂, m_i = tr(σ_i ρ̂). Four real fields mixed jointly
   (Kerker on ρ only — the collinear lesson generalizes: never Kerker the
   magnetization channels).

3. **Potential.** V̂(r) = [v_H + v_loc + v_xc]·1 + B⃗_xc·σ⃗, with
   (v_xc, B⃗_xc) from autograd (already implemented). H apply: two extra
   grid multiplies for the off-diagonal B_x ± iB_y coupling.

4. **SOC.** Fully-relativistic UPFs carry j-resolved KB projectors
   (PP_SPIN_ORB block: j = l ± ½). The parser's `has_so` rejection lifts;
   projectors gain 2×2 spin structure via Clebsch–Gordan coefficients
   coupling (l, m, σ) to |j, m_j⟩. D_ij stays real; the spin structure
   lives in the projector spinors. Purely additive to the nonlocal apply.

5. **Symmetry.** Magnetic groups deferred; non-collinear runs start with
   `use_symmetry=False` (TR-only is already invalid with SOC anyway).

6. **The payoff plumbing** (why this phase exists):
   - torques dE/dê_i for gradient-based configuration search
     (constrained-moment Lagrange fields),
   - transverse exchange J_ij via the implicit-diff machinery
     (cross-check for `postscf/exchange.py` energy mapping),
   - magnetocrystalline anisotropy dE/dθ with SOC.

## Validation ladder

1. Collinear limit: m⃗ = (0, 0, m_z) reproduces nspin=2 exactly (XC leg: done).
2. Global spin rotation invariance without SOC: E independent of rotating
   all moments together.
3. Fe FM with moments along x̂ vs ẑ: identical without SOC.
4. With SOC: Fe magnetocrystalline anisotropy energy (μeV scale — the
   precision stress test), Pt/Au band splittings vs QE fully-relativistic.
