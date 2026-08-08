# 1D DtN jellium-slab prototype

Step-1 proof-of-concept for the Round-5 moonshot idea **"The Vacuum That Isn't
There"** — exact open / Dirichlet-to-Neumann (DtN) boundaries for surfaces, so a
plane-wave slab lives in a slab-sized box with no image leakage, *by construction*.

This is the cheap go/no-go, in the reduced geometry the moonshot report prescribed:
the `G∥ = 0` channel of a jellium slab as a 1D Kohn-Sham problem in z, with the
**only** difference between runs being the boundary treatment.

## The thesis

Plane-wave DFT pays a "vacuum tax": a periodic box makes a surface slab interact
with its images, so observables **drift with the vacuum thickness**. Open-BC
electrostatics + a DtN evanescent-tail boundary impose the *exact* isolated-slab
problem, so the same observables become **box-size independent** — not by using a
bigger box, but by construction.

## Result

`uv run python experiments/dtn_1d/jellium_slab.py` (rs = 4 bohr, 10 Å slab):

```
  vac[A] |  periodic  |    open    |    dtn
  -------------------------------------------
     3   |   +1.72    |   +2.32    |   +1.92
     4   |   +2.22    |   +2.62    |   +2.38
     6   |   +2.75    |   +2.85    |   +2.78
    10   |   +3.25    |   +2.94    |   +2.93
    16   |   +3.95    |   +2.95    |   +2.95     <- periodic still climbing
```
(work function Φ = V_vac − E_F [eV] vs vacuum-per-side)

**Verdict — the electrostatics wall is broken.** `periodic` drifts monotonically
and unboundedly (1.72 → 3.95 and still rising: the spurious vacuum-level / image
artifact), while `open` and `dtn` **plateau at ~2.95 eV** — the correct,
box-independent work function. The drift over the last two boxes is ~0 for
open/dtn and ~0.7 eV (and growing) for periodic.

**Honest nuance.** For a *symmetric* jellium slab (no net surface dipole) the
dominant box artifact is the periodic **electrostatics**, which open-BC Poisson
removes — not the wavefunction tail. So DtN here mainly *recovers the same correct
plateau* rather than dramatically beating a hard wall. DtN's real edge shows up on
(a) smaller boxes / systems where the wavefunction tail is the limiter, and (b)
**asymmetric / dipolar slabs**, where periodic images create a genuine dipole
artifact — the natural next test (see below). And this is still the fixed-κ
*honest-approximation* branch; the energy-**exact** DtN needs a Green's-function
density path (the report's hard blocker).

## What's real gradwave

The XC potential comes straight from gradwave's `LDA_PW92.energy_density` via
autograd (exactly as in the 3D SCF), and all units/prefactors come from
`gradwave.constants` (`HBAR2_2M`, `E2`). Everything is in gradwave units (eV, Å).

## The three modes (only the boundary differs)

| mode       | Poisson              | kinetic BC                                  |
|------------|----------------------|---------------------------------------------|
| `periodic` | FFT (periodic images)| periodic (wrap)                             |
| `open`     | open 1D (isolated)   | hard wall (ψ = 0)                           |
| `dtn`      | open 1D (isolated)   | Robin ψ′ = −κψ (fixed reference-energy κ)   |

`open` = electrostatics fix only; `dtn` = electrostatics + exact evanescent tail.

## Notes on the numerics

- Density-mixed SCF with `alpha = 0.05` (jellium is metallic → charge sloshing;
  larger α does not converge). Subbands filled by the 2D in-plane free-electron gas.
- The DtN Robin BC **must** use the symmetric weak-form discretization (Neumann
  base + boundary self-energy `+HBAR2_2M·κ·|ψ_edge|²`). A naive ghost-point
  elimination gives a non-Hermitian matrix and `eigh` returns garbage — that bug
  cost a debugging round and is called out in the code.

## Next steps (the POC ladder)

1. **Asymmetric / dipolar slab** — the case where the periodic artifact is
   *dramatic* and DtN clearly beats a hard wall (bilayer of two `rs`, or a dipole
   layer). This is the most convincing next experiment.
2. **Step 2 — differentiability**: put a surface parameter (e.g. the slab edge or
   an adsorbate position) into the model and check `autograd dE/dparam` through the
   nested (SCF + V_vac→κ) fixed point matches finite difference. This validates the
   moonshot's actual superpower — exact, differentiable surface forces.
3. **Energy-exact DtN** (multi-quarter): replace the fixed-κ eigensolver path with
   a Green's-function / contour-integration density so `κ(E)` is exact. This is the
   real blocker and the ambitious 3D version.
