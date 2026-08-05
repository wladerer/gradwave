# First-principles phase diagrams (QHA + cluster expansion)

Design note for a phase-diagram builder that assembles free energies from the
pieces gradwave already produces, so a binary temperature-composition diagram
comes out of the differentiable stack. It reuses the harmonic thermodynamics
(`postscf.thermo`), the supercell phonons (`postscf.phonons_supercell`), the
equation of state (`postscf.eos`), and the composition surrogate
(`postscf.composition_design`).

## Status

All four phases below have landed with unit tests. Each maps to a module in
`postscf` and its test.

| phase | module | key symbols | test |
|---|---|---|---|
| 1. Quasi-harmonic single phase | `postscf.qha` | `qha`, `QHAResult` | `tests/unit/test_qha.py` |
| 2. Zero-temperature hull | `postscf.convex_hull` | `formation_energy`, `lower_convex_hull`, `hull_distance`, `ground_states`, `leave_one_out_rmse` | `tests/unit/test_convex_hull.py` |
| 3. Finite-temperature configurational | `postscf.lattice_mc` | `simulate`, `scan_temperature`, `order_disorder_temperature`, `configurational_free_energy` | `tests/unit/test_lattice_mc.py` |
| 4. Full temperature-composition diagram | `postscf.phase_diagram` | `two_phase_regions`, `binodal`, `critical_temperature` | `tests/unit/test_phase_diagram.py` |

Remaining beyond the four phases is what the scope limits already name. Strongly
anharmonic phases and the liquid free energy have no harmonic reference, so they
sit outside the quasi-harmonic term and wait on the Boltzmann-generator frontier
below. The full-binary validation against a CALPHAD or experimental reference is
the Phase 4 oracle rather than a shipped diagram, and cluster-expansion
convergence stays its own art.

## What a phase diagram decomposes into

A phase boundary is where two competing phases share a free energy. For each
phase the Gibbs free energy G(T, P, x) splits into three parts that are computed
separately and added.

- Configurational. For a substitutional alloy, which atom occupies which site.
  This is discrete and combinatorial, so it is a cluster expansion sampled by
  Monte Carlo, not a coordinate model.
- Vibrational. The lattice free energy, taken to the quasi-harmonic level so it
  carries thermal expansion.
- Electronic. The Sommerfeld free energy for metals, from the electronic density
  of states at the Fermi level.

The boundaries then come from a common-tangent construction on the G(x) curves
of the competing phases at each temperature, which reduces to the convex hull of
formation energies at T=0.

## The quasi-harmonic vibrational free energy

Harmonic phonons give a free energy at a fixed volume, which misses thermal
expansion and the temperature dependence of the elastic and vibrational
properties. Instead the quasi-harmonic approximation makes the phonon
frequencies volume-dependent and minimizes over volume at each temperature.

For one ordered structure the builder evaluates the static energy E(V) on a small
volume grid and fits it with the Birch-Murnaghan form (`postscf.eos`), then
computes a phonon density of states g(omega, V) at each volume
(`postscf.phonons_supercell`) and the harmonic vibrational free energy
F_vib(T, V) from it (`postscf.thermo.free_energy_vib`). The Gibbs free energy is

    G(T, P) = min_V [ E(V) + F_vib(T, V) + P V ].

The minimizing volume is the thermal-expanded V(T, P), so the same pass yields
the thermal expansion, the constant-pressure heat capacity, and the
temperature-dependent bulk modulus. The electronic term
(`postscf.thermo.electronic_heat_capacity`, integrated to a free energy) adds on
for metals.

## The configurational free energy

The configurational entropy of an alloy is the hard part, since the number of
site occupations grows exponentially. A cluster expansion writes the energy as a
polynomial in the site-occupation variables, which is exactly the surrogate the
composition work already fits (`postscf.composition_design.fit_surrogate`).

The fit is data-efficient here for the same reason it was in the composition
work. Each reference structure supplies its energy and its per-site alchemical
gradient (`scf.alchemical.alchemical_energy_gradient`), so one calculation
carries 1 + n_atom constraints on the cluster coefficients rather than one. A
Monte Carlo or mean-field pass over the fitted expansion then gives the
configurational free energy F_config(T, x) and the order-disorder transition
temperatures.

## Assembling the diagram

The total free energy of a phase is the sum

    G_phase(T, P, x) = E_static + F_vib^QHA + F_config^CE + F_el.

At a fixed temperature and pressure the stable phases and the two-phase regions
come from the convex hull of G against composition, and the tie-lines are the
common tangents that equalize the chemical potentials of the shared species.
Sweeping temperature traces the temperature-composition diagram, and sweeping
pressure traces the pressure-temperature diagram. At T=0 the construction is the
convex hull of formation energies, the ground-state line.

## Phases

1. Quasi-harmonic single phase. E(V) plus phonons on a volume grid for one
   ordered structure, giving G(T, P), the thermal expansion, and Cp. The oracle
   is the thermal expansion and Cp of Si or MgO against experiment, and a
   positive, physical Grueneisen parameter.
2. Zero-temperature hull. Formation energies of a set of ordered alloy
   structures, fit to a cluster expansion, giving the ground-state line. The
   oracle is that the hull reproduces the known ground states and the
   cross-validation error is small.
3. Finite-temperature configurational. Monte Carlo on the cluster expansion for
   the order-disorder transition temperature. The oracle is a classic binary
   (for example Cu-Au) against a literature transition temperature.
4. Full temperature-composition diagram. The quasi-harmonic vibrational term and
   the configurational term combined through the common-tangent construction,
   demonstrated on one binary against a CALPHAD or experimental reference.

## Scope limits

- The quasi-harmonic approximation fails near melting and for strongly
  anharmonic phases, where the frequencies are no longer a function of volume
  alone. That regime, and the liquid free energy that has no harmonic reference,
  is the anharmonic frontier a Boltzmann generator or a stochastic normalizing
  flow trained on gradwave forces would address, on top of this foundation.
- Cluster-expansion convergence is its own art, since the cluster set and the
  cross-validation govern the accuracy.
- Phonons at several volumes across several ordered structures are expensive, so
  the volume and structure sweeps route to the queue and the GPU where they help.
- The construction is equilibrium only and does not capture metastable or
  kinetically trapped phases.

## The differentiable payoff

Every free-energy piece sits on the same autograd stack, the static energy
through forces and stress, the vibrational term through the phonon free energy,
and the configurational term through the differentiable cluster expansion. The
phase boundary is therefore differentiable in composition, pressure, and the
functional or pseudopotential parameters, so a boundary can be driven toward a
target. This is the composition-gradient method applied at the thermodynamic
level, the design of a phase diagram rather than only its calculation.
