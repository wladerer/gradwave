# All-electron FLAPW and NMR/EFG/XPS spectroscopy

gradwave is a pseudopotential plane-wave code, but it also carries a parallel
*all-electron* engine: a full-potential linearized-augmented-plane-wave (FLAPW)
stack in `gradwave.flapw` that keeps the bare nuclear $-Z/r$ potential and
augments it inside muffin-tin spheres, with no pseudopotential anywhere. It
exists for the observables a pseudopotential smooths away — the near-nucleus
quantities that solid-state NMR and X-ray photoelectron spectroscopy measure:
the electric field gradient (EFG), the magnetic shielding, and core-level
binding-energy shifts.

Two things are worth stating before any input file. First, FLAPW here is a dense
$O(N^3)$ all-electron **oracle**: it is for small cells and code-to-code
validation, not a production large-cell engine, and there is no iterative escape
from the dense generalized eigenproblem. Second, the realistic-material property
path is *not* FLAPW — it is the plane-wave/PAW ground state plus GIPAW
reconstruction (`task: nmr`, `nmr.task=shielding`). FLAPW is the all-electron
reference; PAW+GIPAW is the production property path. The two are cross-validated
against each other and against [Elk 11.0.2](bibliography.md).

!!! warning "FLAPW eigenvalues wander — compare splittings, not absolute levels"
    FLAPW eigenvalues are referenced to the interstitial-region zero, which is an
    arbitrary offset that drifts with cell size, muffin-tin radius, and basis. The
    absolute levels are therefore *not* physical. Only energy **differences** are:
    eigenvalue splittings, band widths, within-cell core-level shifts, and
    site-to-site EFG contrasts. Every gradwave FLAPW validation compares
    splittings, and so should you.

## Three entry points

| Task | Engine | Observable |
|---|---|---|
| `task: flapw` | all-electron FLAPW SCF | Γ eigenvalues, convergence, initial-state core levels / XPS shifts |
| `task: nmr`, `nmr.task: efg` | all-electron FLAPW | electric field gradient $V_{zz}$, $\eta$, quadrupolar coupling $C_Q$ |
| `task: nmr`, `nmr.task: shielding` | plane-wave/PAW + GIPAW | magnetic shielding $\sigma$, chemical shift $\delta_{\mathrm{iso}}$ (+ optional PAW EFG and powder spectrum) |

The FLAPW geometry comes from the top-level `structure` (Å, converted to Bohr
internally) and the k-mesh from `kpoints`. Because FLAPW is all-electron, the
plane-wave `pseudopotentials`, the top-level `ecut`, and the top-level `smearing`
block do **not** apply to a `task: flapw` or `nmr.task=efg` run — the muffin-tin
stack carries its own interstitial cutoff (`flapw.ecut`, in FLAPW units, distinct
from the eV plane-wave cutoff) and its own Fermi width (`flapw.smearing`).

## All-electron FLAPW SCF (`task: flapw`)

The only required FLAPW input is a muffin-tin radius (Å) for every element.
Everything else has a default.

```yaml
task: flapw
structure: Ne.cif          # cell → Bohr internally
kpoints:
  mesh: [2, 2, 2]
flapw:
  radii: {Ne: 0.74}        # muffin-tin R_MT per species, Å (required)
  ecut: 200.0              # interstitial plane-wave cutoff (FLAPW units)
  lmax: 2                  # augmentation angular-momentum cutoff
  smearing: 0.0            # Fermi-Dirac width [eV]; 0 = exact insulator fill
  iters: 40                # SCF iteration cap
  tol: 3.0e-3              # SCF residual convergence gate
```

The full `flapw` block (`inputs.models.FlapwParams`,
`src/gradwave/inputs/models.py`):

| key | default | meaning |
|---|---|---|
| `radii` | `{}` (required) | `{species: R_MT}` muffin-tin radius, Å; must cover every element |
| `ecut` | `200.0` | interstitial plane-wave cutoff (FLAPW units, **not** eV) |
| `lmax` | `2` | augmentation angular-momentum cutoff |
| `fullpot` | `false` | self-consistent non-spherical (full) potential |
| `fullpot_lmax` | `2` | non-spherical potential $L$ cutoff |
| `iters` | `40` | SCF iteration cap |
| `tol` | `3.0e-3` | SCF residual convergence gate |
| `smearing` | `0.0` | Fermi-Dirac width [eV]; `0` = exact insulator fill |
| `kerker` | `null` | interstitial Kerker screen [Å⁻¹] (a full-potential convergence aid) |
| `kworkers` | `1` | k-point process-pool size |
| `los`, `val_e`, `core`, `el_override` | `null` | per-species LAPW+LO basis / valence-count / frozen-core / linearization-energy overrides |
| `efg_anion_basis` | `null` | opt named anion species into the validated EFG anion recipe (see below) |

Through the API the same run is:

```python
from gradwave import api
from gradwave.inputs import Input, FlapwParams, KPointsParams

inp = Input(atoms=ne_atoms, pseudo_dir=".", pseudo_map={}, ecut=1.0,  # ecut unused by FLAPW
            task="flapw", kpoints=KPointsParams(mesh=(2, 2, 2)),
            flapw=FlapwParams(radii={"Ne": 0.74}, ecut=200.0, lmax=2))
summary = api.run(inp)
flapw = summary["flapw"]
```

`summary["flapw"]` (assembled in
`src/gradwave/api/flapw.py`,
`_flapw_meta`) carries:

| key | contents |
|---|---|
| `eigenvalues_eV` | Γ eigenvalues (referenced to the interstitial zero — splittings only) |
| `band_span_eV` | occupied-band span |
| `e_fermi_eV` | Fermi level |
| `n_bands` | band count |
| `converged` | bool; also gates the aspherical residual when `fullpot` |
| `convergence` | the recorder summary (residual span etc.) |
| `muffin_tin_radii_ang`, `flapw_ecut`, `lmax`, `fullpot`, `fullpot_lmax`, `kmesh`, `smearing_eV` | the knobs the run actually used |
| `core_levels`, `core_level_shifts` | initial-state core levels and same-element shifts (see XPS below) |

**Validation.** The single-shot LAPW $2s$–$2p$ splitting of simple-cubic Ne
matches Elk 11.0.2 to under 0.5 eV (a 0.15 eV residual), and the $3s$–$3p$
splitting of simple-cubic Ar is 13.51 eV versus Elk's 13.998 eV — a 0.49 eV /
3.5 % cross-code agreement, with the larger residual attributable to the
single-shot atomic potential and scalar-relativistic contraction
(`tests/integration/test_flapw_scf.py`).
In the dilute limit the self-consistent crystal splitting recovers the
isolated-atom splitting to under 0.15 eV. The worked example
`examples/flapw_neon_crystal.py`
runs the whole atomic → empty-lattice → self-consistent-crystal pipeline and
shows autograd through the muffin-tin radial match.

### XPS core-level shifts

A FLAPW run also re-solves the frozen core in the converged crystal potential,
so `summary["flapw"]["core_levels"]` and `["core_level_shifts"]` carry
initial-state (Koopmans) core binding energies and the same-element,
within-cell shifts (`gradwave.flapw.core_levels`). Because absolute core levels
wander with the interstitial zero, only the shift is physical — equivalent sites
must come out exactly equal, and an inequivalent site's shift carries the sign
and magnitude. `core_level_shifts` is a list of per-pair records
(`species`, `orbital`, `site`, `ref_site`, `delta_eV`).

**Validation** (`tests/unit/test_flapw_core_levels.py`):
the isolated-atom core solve recovers the NIST-LDA levels (Ne 1s to ~1 eV);
symmetry-equivalent sites give a null shift to machine precision; and the O-1s
Madelung-referenced shift for oxygen nearer a cation (the corundum/TiO₂-style
contrast) is $\sim +4.3$ eV against an Elk interpolation of $\sim +4.4$ eV.
Cross-cell binding-energy differences (e.g. Si-2p in Si vs SiO₂) are wander-invariant
by construction (`cross_cell_binding_shift`).

## All-electron EFG (`task: nmr`, `nmr.task: efg`)

The electric field gradient is the anisotropy of the electrostatic potential at
the nucleus. It vanishes for a closed shell and grows with an aspherical valence
density, and its quadrupolar coupling
$C_Q = 2.4180 \cdot Q[\text{barn}] \cdot V_{zz}[\text{eV/Å}^2]$ is what a
quadrupolar-nucleus NMR experiment reads. gradwave takes it all-electron through
the same FLAPW stack (the aspherical sphere Poisson solve in
`gradwave.flapw.efg`), so it needs a `flapw` block, not pseudopotentials.

```yaml
task: nmr
structure: rutile.cif
kpoints:
  mesh: [3, 3, 3]
flapw:
  radii: {Ti: 0.95, O: 0.80}
  ecut: 150.0
  fullpot: true            # non-spherical potential — needed for converged V_zz
  fullpot_lmax: 4
  kerker: 0.7              # full-potential convergence aid
nmr:
  task: efg
  isotopes: {Ti: 49Ti, O: 17O}   # which isotope's Q sets C_Q per species
```

```python
from gradwave.inputs import NmrParams
inp = Input(atoms=rutile, pseudo_dir=".", pseudo_map={}, ecut=1.0, task="nmr",
            kpoints=KPointsParams(mesh=(3, 3, 3)),
            flapw=FlapwParams(radii={"Ti": 0.95, "O": 0.80}, ecut=150.0,
                              fullpot=True, fullpot_lmax=4, kerker=0.7),
            nmr=NmrParams(task="efg", isotopes={"Ti": "49Ti", "O": "17O"}))
sites = api.run(inp)["nmr"]["sites"]
```

`summary["nmr"]` is the shared FLAPW-meta block plus `observable: "efg"`,
`n_sites`, and a `sites` list. Each site carries (`_run_efg` in
`src/gradwave/api/flapw.py`):

| key | meaning |
|---|---|
| `site`, `species` | atom index and element |
| `V_zz_eV_ang2`, `eta` | largest-magnitude principal component and asymmetry $\eta$ |
| `V_zz_valence_eV_ang2`, `eta_valence` | the valence-only contribution |
| `sphere_charge` | electrons inside the muffin-tin sphere |
| `tensor_eV_ang2` | the full $3\times3$ EFG tensor |
| `isotope`, `C_Q_MHz`, `abs_C_Q_MHz`, `nu_Q_MHz`, `Q_barn`, `spin` | present only for a species mapped to a tabulated isotope |

`nmr.isotopes` maps a species to the isotope whose quadrupole moment sets $C_Q$;
a species left unmapped reports $V_{zz}/\eta$ only, and `isotopes: null`
auto-selects the first tabulated isotope for each element. For the anion whose
aspherical near-core density carries the EFG, `flapw.efg_anion_basis: [O]` opts
that species into the validated EFG anion basis recipe (an unconfined $l{=}1$
HELO plus an $l{=}0$ $2s\to2p$ semicore local orbital); any explicit
`los`/`el_override` for the same species overrides it.

**Validation.** The strongest committed anchor is the ²⁷Al site of α-Al₂O₃
corundum against Elk 11.0.2 (LDA, matched spheres and cutoffs): Elk gives
$V_{zz} = -5.90$ eV/Å², $\eta \approx 0$ (axial ∥ *c*); gradwave reproduces
$V_{zz} \approx -7$ eV/Å² — same sign, $\eta \approx 0$, magnitude $\sim 118\%$
of Elk — asserted as a defensible $\pm35\%$ band rather than a tight pin, since
the fully-converged number needs the minutes-long staged converger (torture tier,
`tests/integration/test_flapw_efg_vs_elk.py`).
Symmetry-equivalent Al sites agree in $V_{zz}$ and $\eta$ to $<10^{-2}$
relative. The
`examples/flapw_oxygen_nmr.py`
example tracks the ¹⁷O coupling through a tetragonal crystal-field distortion.

!!! note
    The standard-tier end-to-end test uses a cheap Γ-point, spherical
    (`fullpot: false`) smoke configuration — it pins the api → FLAPW → EFG → $C_Q$
    wiring, and its $V_{zz}$ values are *not* the physical EFG. A converged
    magnitude needs `fullpot: true`, a Kerker screen, Newton polishing, and a
    k-mesh of at least $3\times3\times3$ — tens of minutes per cell.

## Magnetic shielding, PAW EFG, and powder spectra (`task: nmr`, `nmr.task: shielding`)

The magnetic shielding is the production property path, and it runs through the
plane-wave/PAW ground state with GIPAW reconstruction — not FLAPW. It needs
`pseudopotentials`, a plane-wave `ecut`, and a k-mesh with at least two axes of
length $>1$ (the shielding derivative is a $\partial/\partial q$ over the
periodic mesh).

```yaml
task: nmr
structure: SiO2.cif
ecut: 60.0
pseudopotentials:
  Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF
  O:  O.pbe-n-kjpaw_psl.1.0.0.UPF
kpoints:
  mesh: [2, 2, 2]
nmr:
  task: shielding
  shielding_level: auto      # auto | bare | gipaw
  sigma_ref: {Si: 328.4}     # absolute σ_ref of a reference solid → δ_iso
  efg: auto                  # also compute the PW/PAW EFG (auto = on for PAW)
```

`nmr.shielding_level` (`inputs.models.NmrParams`,
`src/gradwave/inputs/models.py`)
picks the assembly against the ground state produced:

- `bare` — the smooth valence term alone, the analytic $q\to0$ route
  (`kgeometry_nmr.sigma_shielding_dq`); requires a **norm-conserving** ground state.
- `gipaw` — the full absolute
  $\sigma = \sigma_{\text{bare}} + \sigma_{\text{core}} + \sigma_{\text{dia,aug}} + \sigma_{\text{para,aug}}$
  (`kgeometry_nmr.sigma_shielding_gipaw`); requires an **all-PAW** ground state.
- `auto` (default) — `gipaw` for PAW pseudopotentials, `bare` otherwise.

`summary["nmr"]` carries `observable: "shielding"`, `method`
(`bare_analytic_dq` or `gipaw_absolute`), `shielding_level`, `n_sites`, and a
`sites` list. Each site reports the Haeberlen quantities of the shielding tensor:

| key | meaning |
|---|---|
| `site`, `species` | atom index and element |
| `sigma_iso_ppm` | isotropic shielding $\sigma_{\mathrm{iso}}$ |
| `sigma_aniso_ppm`, `sigma_eta` | shielding anisotropy and asymmetry |
| `span_ppm`, `tensor_ppm` | span and the full $3\times3$ tensor |
| `sigma_bare/core/dia_aug/para_aug_ppm` | per-term breakdown (`gipaw` only) |
| `delta_iso_ppm` | chemical shift $\delta_{\mathrm{iso}} = \sigma_{\mathrm{ref}} - \sigma_{\mathrm{iso}}$, when `sigma_ref` covers the site |

`nmr.sigma_ref` maps a species or isotope to a reference absolute shielding
(ppm); `api.reference_sigma_iso(inp, species)` computes one by running the
same-level shielding on a reference solid and averaging over its sites (use the
*same* `shielding_level` for reference and sample).

**k-mesh symmetry matters.** A Γ-centred mesh with unequal subdivisions on
symmetry-equivalent axes (e.g. `[3, 3, 2]` on a cubic crystal) breaks the point
group, and symmetry-equivalent sites then sample the BZ inequivalently and split.
The driver warns both when the mesh is unsuitable and — post hoc — when
equivalent sites disagree in $\sigma_{\mathrm{iso}}$ by more than 1 ppm, which
flags an unconverged or broken result. Use equal subdivisions on equivalent axes.

### PAW EFG alongside the shielding

With `nmr.efg` on (`auto` = on for a PAW ground state), the same SCF also yields
the plane-wave/PAW EFG (Petrilli–Blöchl, `postscf.efg_paw`) — a separate
`summary["nmr"]["efg"]` sub-block with per-site $V_{zz}/\eta$ and, for the mapped
`isotopes`, $C_Q$. This is the PW-side counterpart to the all-electron FLAPW EFG,
cross-validated against it, and it feeds the quadrupolar spectrum path. `efg: true`
on a norm-conserving run is a clear error (the PAW EFG reconstructs the on-site
$l{=}2$ field gradient from `PAWData`).

### Powder lineshapes

`nmr.spectrum` (`inputs.models.NmrSpectrumParams`) synthesizes a powder NMR
lineshape from the referenced sites; it is off by default, so an unreferenced
shielding run stays byte-identical.

```yaml
nmr:
  task: shielding
  sigma_ref: {Si: 328.4}     # a spectrum needs δ_iso, so a reference is required
  spectrum:
    enabled: true
    mode: mas                # static | mas
    spin_rate_hz: 10000.0    # required for mas
    larmor_mhz: 79.5         # observed-nucleus Larmor frequency (MHz)
    broadening_ppm: 2.0
    lineshape: gauss         # gauss | lorentz
    n_orientations: 2000
    n_points: 2048
```

A spin-½ observed nucleus uses the CSA lineshape (the shielding alone sets it);
a half-integer $I \ge 3/2$ nucleus uses the second-order central-transition
lineshape and pulls $C_Q/\eta_Q$ from the `efg` sub-block, so it needs
`nmr.efg` on and a positive `larmor_mhz`. The `spectrum` sub-block returns the
config echo plus `peak_ppm`, `ppm_range`, and the `(ppm_axis, intensity)` arrays.

**Validation** (`tests/unit/test_gipaw.py`):
the frozen-core (Lamb) shielding term agrees with the FLAPW all-electron core to
a few percent, and the diamagnetic augmentation of a free atom recovers the
expected magnitudes ($\sim 14.8$ ppm for ¹⁷O, an order of magnitude above ²⁹Si's
$\sim 0.9$ ppm, matching Yates–Pickard–Mauri). The absolute GIPAW $\sigma$ is
anchored to published diamond-Si ²⁹Si and other GIPAW references.

## Scope and limitations

- **FLAPW is an $O(N^3)$ all-electron oracle**, not a production large-cell
  engine: dense generalized eigensolves, small cells, minutes per iteration on a
  converged cell. It is for code-to-code validation and near-nucleus reference
  numbers. For a realistic material, take the property through the PW/PAW +
  GIPAW path instead.
- **Absolute eigenvalues wander** (interstitial zero) — trust only splittings,
  within-cell core-level shifts, and site contrasts.
- **Converged EFG magnitudes need the full potential** (`fullpot: true`,
  `fullpot_lmax` raised, a Kerker screen, and Newton polishing at a converged
  k-mesh). The default spherical potential is a fast approximation; the
  smoke-tier defaults are for wiring, not physics. The core is frozen (as in
  Elk), and the anion aspherical basis recipe (`efg_anion_basis`) governs the
  near-core accuracy.
- **Shielding formalism gates.** `bare` is norm-conserving only; `gipaw` is
  all-PAW only; `auto` picks the right one. The shielding needs a k-mesh with at
  least two axes of length $>1$, and unequal subdivisions on symmetry-equivalent
  axes break equivalent sites.
- **A spectrum needs a reference** (`sigma_ref`, so each observed site carries
  $\delta_{\mathrm{iso}}$) and, for quadrupolar nuclei, the `efg` sub-block and a
  Larmor frequency.

## See also

- [Post-SCF analysis](postscf-analysis.md) — the plane-wave post-processing catalog.
- [Non-collinear magnetism and SOC](noncollinear-soc.md) — the fully-relativistic ground states behind magnetic observables.
- [Output schema](output-schema.md) — the JSON summary key list.
