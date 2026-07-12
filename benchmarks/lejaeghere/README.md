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
