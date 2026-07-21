# Periodic-table Δ-gauge (norm-conserving)

Equation-of-state reproducibility across the cubic elements, gradwave against
the all-electron WIEN2k reference of the Δ-factor benchmark (Lejaeghere et al.,
*Science* **351**, aad3000, 2016), using PseudoDojo NC-SR-v0.4 PBE (standard)
pseudopotentials. This is the breadth companion to `benchmarks/lejaeghere/`,
which runs a five-element PAW feature-matrix subset; here the goal is coverage —
alkali, alkaline-earth, simple, refractory, noble, and Pt-group metals plus the
group-IV semiconductors, spanning s, p, and d valence.

## What is measured

For each element, E(V) at seven volumes from 94 % to 106 % of the all-electron
equilibrium volume, a third-order Birch-Murnaghan fit, and the calcDelta-3.0 Δ:
the RMS difference of the two fitted curves (each shifted to its own minimum)
over the ±6 % window around the average equilibrium volume, per atom.

Two reference axes (see `docs/manual/wisdom.md`, "Process and validation"):

- **Δ vs WIEN2k** mixes the pseudopotential's own pseudization error with any
  gradwave implementation error.
- **Δ_pdojo** is PseudoDojo's published Δ-factor for the *same* standard
  pseudopotential, computed in ABINIT against the same all-electron reference.
  gradwave's Δ vs WIEN2k should track it element by element; that tracking is
  the validation claim, since both codes use the identical pseudopotential.

## Structures

Cubic ground states only, because the reference V0 alone fixes the geometry:
`fcc` (noble/Pt-group, fcc alkaline-earth, Al), `bcc` (alkali, group-5/6
refractory), and `diamond` (group IV). Non-cubic ground states (hcp Mg/Zn/Ti,
rhombohedral As/Bi, graphite C) are out of scope. Magnetic Fe/Ni (ferromagnetic)
and Cr (antiferromagnetic) are defined in `cases.py` but gated behind the
`johnson`-mixer work — the norm-conserving `scf` mixer risks a moment collapse to
the nonmagnetic branch on the itinerant 3d metals.

## Settings

`ecut` is 2× the PseudoDojo "high" hint per element (a well-converged EOS, so
the residual Δ is pseudization, not basis truncation); `ecutrho = 4·ecut` for
norm-conserving. Metals use a 16³ Γ-centred mesh reduced to the irreducible
wedge and a 0.136 eV gaussian smearing; the group-IV semiconductors use 8³, and
Ge/Sn a whisker of smearing because PBE closes their gap. One FFT grid per
element, the elementwise max over the seven volumes, so E(V) is not stepped by a
grid change. Volumes warm-start along the chain.

## Running

The cells are 1–2 atom primitives; on the RTX 3050 the many-k norm-conserving
Hamiltonian is FFT+eigh-bound and batches over k, so the GPU beats the 22-core
CPU here (2.6–5.6× on Al/Cu/Si), unlike the low-k PAW-metal case in
`docs/manual/performance.md`. Each element writes its own JSON, so elements
parallelize across processes without a shared-file race.

```sh
GW_DEVICE=cuda uv run python benchmarks/delta_gauge/run_gw.py run          # all
GW_DEVICE=cuda uv run python benchmarks/delta_gauge/run_gw.py run Cu Ag Au # subset
uv run python benchmarks/delta_gauge/fit_delta.py                          # table
uv run python benchmarks/delta_gauge/make_fig.py                           # figure
```

Results merge per element into `results/eos_<el>.json`; `fit_delta.py` fits
whatever subset is present and writes `results/delta_summary.json`.

## Results

22 cubic elements (spanning s, p, d valence), Δ vs the WIEN2k all-electron
reference. **Median Δ = 0.8 meV/atom, and 21 of 22 sit at V0 to <0.6 % and B0 to
<4.4 % of all-electron** — inside the mature-code reproducibility range, and
below PseudoDojo's own published Δ on roughly a third of the set (Nb 0.45 vs
1.29, Mo 0.62 vs 1.41, Ag 0.08 vs 0.32, Rh 1.35 vs 2.56). The figure is
`results/delta_gauge.png`; the table is `results/delta_summary.json`.

Two features of the data are worth reading correctly.

- **The elevated transition-metal Δ (Pt 2.7, Pd 2.1, Ir 1.9) is the stiff-metal
  floor of the metric, not error.** The calcDelta Δ scales with B0, so a hard
  metal inherits a larger Δ for the same fractional accuracy: Ir (B0 = 348 GPa,
  |ΔV0| = 0.17 %) gives Δ 1.9 while Ag (B0 = 90 GPa, |ΔV0| = 0.00 %) gives 0.08.
  Their fractional V0/B0 errors are excellent. PseudoDojo's own dfacts show the
  same B0 ordering.

- **Cu is a defective pseudopotential file, not a gradwave error** — see
  `results/cu_anomaly.md`. A matched QE run on the same PseudoDojo `Cu.upf`
  reproduces gradwave's stiff EOS to 0.08 meV (B0 167 in both codes vs 141
  all-electron), and SG15 Cu in the same harness gives a normal Δ = 1.33. The
  standard-tarball UPF disagrees with all-electron *and* with PseudoDojo's own
  psp8 dfact (0.53), and it fails identically in QE. This is the two-axis check
  of `docs/manual/wisdom.md`: bad on the all-electron axis (pseudization),
  exact on the QE axis (implementation).

### Practical notes recorded during the campaign (`results/timing.json`)

- **Device.** For these norm-conserving many-k cells the RTX 3050 beats the
  22-core CPU 2.6–5.6× (Al/Cu/Si), because 145 irreducible k-points batch into
  the fp64 units — the opposite of the low-k PAW-metal guidance in
  `docs/manual/performance.md`. Energies are bit-identical across devices. An
  8-core laptop CPU, by contrast, ran Ge at 435 s/vol vs ~8 s/vol on the GPU;
  offload to a slow CPU only when the GPU cannot fit the job.
- **Memory.** The batched apply holds `n_k^IBZ · n_band · ∏nᵢ` complex128, and
  the grid is `nᵢ ∝ |aᵢ|·√E_cut`, so big-cell low-Z elements (K, 54³) overflow
  6 GB before compact high-Z ones (Cu, 36³). The fix is k-density scaling: K/Sr
  use 12³ (matching the reciprocal-space density of the ~3.6 Å cells at 16³),
  which is both more correct and fits the card.
- **Cutoff.** `ecut = the PseudoDojo high hint` is EOS-converged here — Cu at
  140 Ry reproduces 104 Ry to the digit, so the outlier is not basis error.
