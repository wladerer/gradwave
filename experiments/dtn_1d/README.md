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
plateau* rather than dramatically beating a hard wall. And this is still the
fixed-κ *honest-approximation* branch; the energy-**exact** DtN needs a
Green's-function density path (the report's hard blocker).

## The dramatic case — asymmetric (bilayer) slab

Where the boundary treatment stops being a refinement and becomes right-vs-wrong:
a bilayer jellium (two `r_s` across the midplane) has a real charge-transfer
**surface dipole**, so its two faces have *different* work functions. From
`asym_sweep()` (rs = 3 / 5 bohr, 12 Å slab), Φ_left / Φ_right [eV]:

```
  vac[A] |    periodic    |      open       |       dtn
  ------------------------------------------------------------
     8   |  +3.00 / +2.96 |  +3.43 / +2.57  |  +3.41 / +2.55
    14   |  +3.03 / +3.00 |  +3.47 / +2.60  |  +3.47 / +2.60
    20   |  +3.03 / +3.01 |  +3.48 / +2.60  |  +3.48 / +2.60
```

**Periodic collapses the dipole** — it forces one common vacuum level, so
Φ_left ≈ Φ_right (~3.0, *wrong*): a periodic box structurally cannot represent two
different work functions, which is exactly why periodic surface calculations need a
dipole correction. **Open/DtN resolve the true dipole**: Φ_left = 3.48 ≠
Φ_right = 2.60 (a 0.88 eV difference), both **box-independent** — no dipole
correction, by construction. This is the compelling demonstration; the symmetric
slab above is the conservative one.

## Step 2 — differentiable surface force (the superpower)

The whole reason to do this *in* gradwave. `force_check()` validates that autograd
gives the correct surface force **through the self-consistent DtN fixed point**. A
weak gaussian "adsorbate" proxy is placed just outside the surface; the force on it
via Hellmann-Feynman (autograd of the explicit z₀-dependence at the fixed converged
density) is compared against a finite-difference of the total energy:

```
  Hellmann-Feynman force (autograd, fixed rho*):  -0.01783 eV/A
  finite-difference force (dE_total / dz0):       -0.01796 eV/A
  agreement: 0.74%
```

The ~1% residual is SCF tolerance + grid + the fixed-κ approximation. This is the
thing a non-AD code (QE/VASP) structurally cannot do: **exact, differentiable
surface forces through an open boundary** — the basis for inverse surface design
and learned embedding operators trained through the exact vacuum.

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

## POC ladder — status

- [x] **Step 1 — box-independence** (`box_sweep`): open-BC Poisson removes the
      periodic vacuum-level drift; open/DtN plateau, periodic diverges.
- [x] **Asymmetric / dipolar slab** (`asym_sweep`): the dramatic case — periodic
      collapses the surface dipole (both faces ~equal, wrong), open/DtN resolve the
      true 0.88 eV dipole, box-independent.
- [x] **Step 2 — differentiable surface force** (`force_check`): autograd
      Hellmann-Feynman force matches finite difference to 0.74% — differentiates
      correctly through the nested SCF+DtN fixed point.
- [ ] **Inverse design demo**: optimize an adsorbate position / surface parameter
      by gradient descent on `force_check`'s autograd force (a few lines on top of
      Step 2) — the "only differentiability unlocks this" money shot.
- [ ] **Energy-exact DtN** (multi-quarter): replace the fixed-κ eigensolver path
      with a Green's-function / contour-integration density so `κ(E)` is exact.
      This is the report's real blocker and the ambitious 3D version.
