# Lejaeghere Δ-benchmark subset (PAW)

E(V) equations of state for a five-element subset of the Lejaeghere
reproducibility benchmark (Science 351, aad3000, 2016), computed with
gradwave's PAW implementation (psl 1.0.0 kjpaw pseudopotentials) and
compared against the all-electron WIEN2k v13.1 reference shipped with
calcDelta 3.0.

The subset covers the feature matrix, not the periodic table. Si is a
fixed-occupation insulator, Ge a near-zero-gap semiconductor with semicore
d states, Al a simple metal, Cu a noble metal, and Ni a ferromagnetic
metal run with nspin=2. All five are cubic (diamond or fcc primitive
cells), so the reference V0 fixes the geometry and no CIF inputs are
needed. Carbon is deliberately absent — the reference structure is
graphite, whose PBE interlayer spacing makes the E(V) curve a soft-mode
benchmark rather than a PAW one.

## Protocol

Seven volumes at 94-106% of the WIEN2k equilibrium volume, one fixed FFT
grid per element (elementwise max over the seven volumes, so E(V) is not
stepped by grid changes), IBZ k-meshes, third-order Birch-Murnaghan fit
per atom. Δ is the RMS difference of the two fitted curves, each shifted
to its own minimum, over the ±6% window around the average equilibrium
volume. Free energies are compared, matching QE's printed total energy
for smeared systems.

## Running

```sh
uv run python benchmarks/lejaeghere/run_gw.py dims          # once per case
uv run python benchmarks/lejaeghere/run_gw.py run si ge     # CPU-sized
GW_DEVICE=cuda uv run python benchmarks/lejaeghere/run_gw.py run al cu ni
uv run python benchmarks/lejaeghere/fit_delta.py
uv run python benchmarks/lejaeghere/gen_qe.py               # QE cross-check
```

Results merge into `results/eos_gw.json`, so cases can run on different
machines. Cutoffs sit at or above the psl-suggested values. The metal
k-meshes (16^3 for Al/Cu, 12^3 for Ni) are the main remaining convergence
knob; the reference-grade Δ protocol used denser meshes still, so treat
sub-meV agreement on metals as k-mesh-limited rather than exact.

The pinned-grid QE inputs in `qe_inputs/` separate implementation error
from convergence error. Δ vs WIEN2k mixes both (pseudization plus
settings), and the published QE/psl Δ values (~0.3-0.6 meV/atom for these
elements) are the fair target. Δ(gradwave vs QE) at identical settings
isolates the implementation and should sit at the 0.01 meV/atom scale
seen in `benchmarks/delta_factor/` for norm-conserving pseudos.

## Results (2026-07)

Δ vs WIEN2k (meV/atom): Si 0.061, Ge 0.096, Al 0.049, Cu 0.638,
Ni 2.909. Δ vs QE at pinned identical settings: 0.014-0.034 across all
five — the implementation floor. Cu and Ni sit at their pseudization
limits (gradwave and QE agree to 0.02-0.03 while both differ from the
all-electron curve identically), so the WIEN2k gap is a property of the
psl 1.0.0 pseudos, not of either implementation. Ni ran FM at every
volume (m 0.64-0.70 μB, matching QE's 0.68 at V0).

FM Ni convergence notes, learned the hard way: the default mixing
(alpha 0.7) collapses the moment to the NM branch — the damped alpha 0.3
of the validated ni_paw_spin config is required; the density residual
plateaus at metallic occupation noise, so the driver gates the energy
tail (last-10 spread < 1e-5 eV) and a surviving moment instead of the
converged flag; one volume (1.06) still landed NM from the 0.6-seed
trajectory and needed start_mag 0.8.
