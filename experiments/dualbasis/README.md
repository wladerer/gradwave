# DualBasis — learned Gaussian-dressed Bloch waves (moonshot #1)

Idea: carry the sharp near-core valence structure in a handful of per-element Gaussians so the
plane-wave ecut (hence npw, hence the O(npw²) memory/GEMM cost) can drop at fixed accuracy — a
mixed Gaussian + plane-wave pseudopotential solve that stays a real autograd DFT calculation.

## GATE 1 — representation payoff + overlap conditioning (`oxygen_2p_representation.py`)

A decisive, self-contained (no-SCF) measurement of both halves of the idea, on the **real
oxygen 2p pseudo-orbital** (PD_O_PBE.upf, PseudoDojo) — not an idealized cusp, so the payoff is
not overstated. Everything is done in the l=1 radial channel with gradwave's own `sbt`; the PW
basis up to ecut spans exactly the |G|<Gcut shells, so its best-approx error is the beyond-Gcut
tail, and the Gaussians least-squares-fit that tail with their own beyond-Gcut content.

Run: `uv run python experiments/dualbasis/oxygen_2p_representation.py` (also verified on asus).

### Findings

**Payoff is real but MODEST at meaningful accuracy.** Energy error ≈ (relL2)², so sub-meV-ish
sits at relL2 ≈ 1e-2..3e-3. There:

| target relL2 | PW-only ecut | PW+Gauss ecut | ecut ratio | ~npw ratio |
|---|---|---|---|---|
| 3e-2 (loose) | 42 Ry | 2.5 Ry | 16.7× | 68× |
| **1e-2** | 56 Ry | 32 Ry | **1.73×** | **2.3×** |
| 3e-3 | 69 Ry | 49 Ry | 1.40× | 1.7× |

So DualBasis buys ~**2× npw** at sub-meV (→ ~4× on the O(npw²) blocks), large only at loose
accuracy — **not** the 3–8× npw / 10–60× the moonshot hoped.

**The blocker is confirmed and is a hard SQUEEZE.** The combined overlap S is singular exactly
when a Gaussian lies in the PW span; the diagnostic is the smallest eigenvalue of the Gaussians'
beyond-Gcut ("new-direction") Gram, normalized. With well-spaced exponents the *intrinsic*
Gaussian conditioning is fine (cond 1.5–15), but:

- at the best-conditioned ecut in range (2.5 Ry) the accuracy is only relL2 ≈ 1e-1;
- where PW+G reaches relL2 < 1.2e-2, the new-direction min-eig ≈ 5e-10 → **cond(S) ~ 1e7**.

Accuracy and conditioning **do not co-exist**: the Gaussians go linearly dependent with the PWs
precisely in the accuracy-relevant ecut range, forcing canonical orthogonalization (which, per
the substrate deep-dive, caps SCF accuracy near ~1e-8). And *learning* the exponents to lower
energy — the moonshot's own mechanism — pushes **harder** toward this singular manifold, not away.

### Bottom line

DualBasis has a **modest ~2×-npw ceiling with a real conditioning tax**, not a breakthrough. It is
not killed — ~2× npw at controlled accuracy is still a genuine lever for medium cells — but the
optimistic framing is refuted with data. The right next de-risk, **before any learning build**, is
to measure the *canonical-orthogonalization accuracy floor in a real SCF* (does dropping the
near-null combined directions still land sub-meV total energy?), since the representation payoff
ceiling and the conditioning squeeze are now both quantified here.

Compare to AutoAPW (#2, `experiments/autoapw/`): PAW/LAPW augmentation reallocates near-atom
resolution *without* a linear-dependence overlap because the augmentation channels are
constructed orthogonal to the smooth part — which is exactly the pathology DualBasis runs into.
