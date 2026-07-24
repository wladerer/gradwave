"""Starter input templates emitted by `gradwave init <name>`.

Each template is a complete, schema-valid YAML input with an inline example
structure and placeholder pseudopotential paths (the `# EDIT` markers). They are
teaching artifacts: sensible key combinations for one kind of calculation, not a
dump of every default. `tests/unit/test_templates.py` runs each one through
`load_input`, so a schema change that breaks a template fails the test rather
than shipping a stale example.

Add a template by adding an entry to ``_TEMPLATES``; the first tuple field is the
one-line description shown by `gradwave init` with no argument.
"""

from __future__ import annotations

_SCF = """\
# Single-point SCF of an insulator (silicon in the diamond structure).
# Run:  gradwave input.yaml -o out/
# Check before running:  gradwave validate input.yaml

structure:
  # EDIT: your cell (Å, rows are lattice vectors) and atoms.
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]   # or use `cart:` for Å
  species: [Si, Si]

pseudopotentials:
  dir: ./pseudos                 # EDIT: folder holding your UPF files
  map:
    Si: Si_ONCV_PBE-1.2.upf      # EDIT: element -> UPF filename

ecut: 500.0                      # eV, plane-wave cutoff (converge this)
xc: pbe                          # lda | pbe
kpoints:
  mesh: [6, 6, 6]                # Monkhorst-Pack grid

scf:
  etol: 1.0e-8                   # eV
  rhotol: 1.0e-7

output:
  dir: ./out
"""

_METAL = """\
# SCF of a metal (fcc aluminium). Metals need smearing (a finite electronic
# temperature) and a denser k-mesh than an insulator to converge.
# Run:  gradwave input.yaml -o out/

structure:
  cell: [[0.0, 2.025, 2.025], [2.025, 0.0, 2.025], [2.025, 2.025, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0]]
  species: [Al]

pseudopotentials:
  dir: ./pseudos
  map:
    Al: Al_ONCV_PBE-1.2.upf

ecut: 400.0
xc: pbe
kpoints:
  mesh: [12, 12, 12]             # metals want a dense mesh

smearing:
  type: mp1                      # methfessel-paxton; also cold | gaussian | fermi-dirac
  width: 0.1                     # eV

scf:
  etol: 1.0e-8
  rhotol: 1.0e-7

output:
  dir: ./out
"""

_RELAX = """\
# Ionic relaxation: move the atoms to the nearest force minimum at a fixed cell.
# Run:  gradwave input.yaml -o out/
# Writes out/relax.xyz (one frame per step) alongside out/relax.json.

structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    # EDIT: a displaced or guessed geometry to relax.
    cart: [[0.0, 0.0, 0.0], [1.40, 1.30, 1.42]]
  species: [Si, Si]

pseudopotentials:
  dir: ./pseudos
  map:
    Si: Si_ONCV_PBE-1.2.upf

ecut: 500.0
xc: pbe
kpoints:
  mesh: [6, 6, 6]

task: relax
relax:
  optimizer: bfgs                # bfgs (default) | fire
  fmax: 0.01                     # eV/Å convergence criterion
  max_steps: 100

output:
  dir: ./out
"""

_RELAX_CELL = """\
# Variable-cell relaxation: relax the atoms and the lattice together (stress).
# Converge `ecut` first, or re-relax at the new cell: at fixed ecut the cell
# carries a Pulay (basis-incompleteness) stress.
# Run:  gradwave input.yaml -o out/

structure:
  # EDIT: start from a strained cell to see it relax back.
  cell: [[0.0, 2.63, 2.63], [2.63, 0.0, 2.63], [2.63, 2.63, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]

pseudopotentials:
  dir: ./pseudos
  map:
    Si: Si_ONCV_PBE-1.2.upf

ecut: 600.0                      # higher cutoff: cell relaxation is Pulay-sensitive
xc: pbe
kpoints:
  mesh: [6, 6, 6]

task: relax
relax:
  optimizer: bfgs
  fmax: 0.01                     # eV/Å; also gates the stress
  max_steps: 100
  cell: true                     # relax the lattice too
  pressure: 0.0                  # GPa external hydrostatic pressure

output:
  dir: ./out
"""

_BANDS = """\
# Band structure: an SCF, then a non-self-consistent solve along a k-path.
# Run:  gradwave input.yaml -o out/
# Plot: gradwave plot out/bands.json

structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]

pseudopotentials:
  dir: ./pseudos
  map:
    Si: Si_ONCV_PBE-1.2.upf

ecut: 500.0
xc: pbe
kpoints:
  mesh: [6, 6, 6]                # the SCF mesh

task: bands
bands:
  path: "GXWKGLUWLK"             # EDIT: ASE band-path string; empty = lattice default
  npoints: 120                   # points along the whole path
  irreps: false                  # label bands at high-symmetry points

output:
  dir: ./out
"""

_BANDS_SOC = """\
# Spin-orbit band structure (fcc platinum). Needs a fully-relativistic (FR)
# pseudopotential and the noncollinear (spinor) solver; SOC splits the bands.
# Run:  gradwave input.yaml -o out/

structure:
  cell: [[0.0, 1.96, 1.96], [1.96, 0.0, 1.96], [1.96, 1.96, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0]]
  species: [Pt]

pseudopotentials:
  dir: ./pseudos
  map:
    Pt: Pt_ONCV_PBE_FR-1.0.upf   # EDIT: must be a fully-relativistic (FR) UPF

ecut: 500.0
xc: pbe
kpoints:
  mesh: [10, 10, 10]

noncollinear: true               # spinor solver; required for spin-orbit coupling
nonmagnetic: true                # spin-orbit only (m ≡ 0): Pt is nonmagnetic, so
                                 # this pins the moment to zero and keeps symmetry
smearing:
  type: mp1
  width: 0.1

task: bands
bands:
  path: "GXWLGK"
  npoints: 160

output:
  dir: ./out
"""

_PDOS = """\
# Projected density of states: an SCF with atomic-orbital projections.
# Needs a pseudopotential that carries atomic orbitals (PP_PSWFC).
# Run:  gradwave input.yaml -o out/
# Plot: gradwave plot out/scf.json --kind pdos

structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]

pseudopotentials:
  dir: ./pseudos
  map:
    Si: Si_ONCV_PBE-1.2.upf

ecut: 500.0
xc: pbe
kpoints:
  mesh: [8, 8, 8]

projections:
  enabled: true
  group_by: l                    # atom | l | lm | total
  width: 0.1                     # eV gaussian broadening

output:
  dir: ./out
"""

_MAGNETISM = """\
# Collinear magnetism (bcc iron): a spin-polarized SCF plus extraction of the
# Heisenberg exchange couplings from the magnetic torque (task: magnetism).
# Run:  gradwave input.yaml -o out/

structure:
  cell: [[2.87, 0.0, 0.0], [0.0, 2.87, 0.0], [0.0, 0.0, 2.87]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
  species: [Fe, Fe]

pseudopotentials:
  dir: ./pseudos
  map:
    Fe: Fe_ONCV_PBE-1.2.upf

ecut: 800.0                      # iron is hard; converge this carefully
xc: pbe
kpoints:
  mesh: [10, 10, 10]

nspin: 2                         # collinear spin
start_mag:
  Fe: 0.6                        # initial moment fraction in [-1, 1] per element
smearing:
  type: gaussian
  width: 0.1
symmetry: false                  # required: the exchange scan tilts a moment
                                 # (noncollinear), which IBZ reduction cannot do

task: magnetism
magnetism:
  exchange: true                 # extract J from the torque (adds a few SCFs)
  ref_atom: 0                    # atom whose moment is tilted for the scan

output:
  dir: ./out
"""

_NONCOLLINEAR = """\
# Noncollinear (spinor) SCF with spin-orbit coupling (bcc iron). Use this for
# canted moments, spin spirals, or magnetocrystalline anisotropy. Needs a
# fully-relativistic (FR) pseudopotential.
# Run:  gradwave input.yaml -o out/

structure:
  cell: [[2.87, 0.0, 0.0], [0.0, 2.87, 0.0], [0.0, 0.0, 2.87]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
  species: [Fe, Fe]

pseudopotentials:
  dir: ./pseudos
  map:
    Fe: Fe_ONCV_PBE_FR-1.0.upf   # EDIT: fully-relativistic (FR) UPF for SOC

ecut: 800.0
xc: pbe
kpoints:
  mesh: [10, 10, 10]

noncollinear: true               # spinor SCF; moments are 3-vectors
start_mag:
  Fe: 0.6                        # seed moment magnitude (direction defaults to +z)
smearing:
  type: gaussian
  width: 0.1
symmetry: false                  # required: a magnetic spinor SCF cannot use IBZ
                                 # reduction (symmetry acts on the moment vector)

task: scf

output:
  dir: ./out
"""

_EOS = """\
# Equation of state: isotropic volume scan + Birch-Murnaghan fit (V0, B0, B0').
# Run:  gradwave input.yaml -o out/   ->  reports V0 [Å³/atom], B0 [GPa], B0'
# Start from the RELAXED cell: the fit is only meaningful when V0 sits inside
# the scanned window.

structure:
  # EDIT: your (ideally relaxed) cell and atoms.
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]

pseudopotentials:
  dir: ./pseudos                 # EDIT: folder holding your UPF files
  map:
    Si: Si_ONCV_PBE-1.2.upf      # EDIT: element -> UPF filename

ecut: 500.0                      # eV, plane-wave cutoff (converge this)
xc: pbe                          # lda | pbe
kpoints:
  mesh: [8, 8, 8]                # denser than a single SCF: B0 needs a smooth E(V)

task: eos
eos:
  # volume factors relative to the input cell; the default is the calcDelta
  # seven-point 94-106% window. Needs >=4 points for the four-parameter fit.
  scales: [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]
  energy: free_energy            # free_energy | total | e0 (quantity fitted vs V)

output:
  dir: ./out
"""

_ELASTIC = """\
# Elastic constants: FD of the analytic stress over the six Voigt strains ->
# the 6x6 stiffness C plus Voigt-Reuss-Hill bulk/shear/Young/Poisson moduli.
# Run:  gradwave input.yaml -o out/
# Clamped-ion: the cell is strained with fractional coordinates fixed. This is
# exact for the bulk modulus and for rocksalt (MgO, NaCl); for the SHEAR
# constants of diamond/zincblende (Si, C, GaAs) it overestimates C44 (no
# internal sublattice relaxation). Start from the RELAXED cell (residual stress
# is reported).

structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]

pseudopotentials:
  dir: ./pseudos                 # EDIT: folder holding your UPF files
  map:
    Si: Si_ONCV_PBE-1.2.upf      # EDIT: element -> UPF filename

ecut: 500.0                      # eV, plane-wave cutoff (converge this)
xc: pbe                          # lda | pbe
kpoints:
  mesh: [8, 8, 8]                # elastic constants need a well-converged stress

task: elastic
elastic:
  strain: 0.005                  # Voigt strain magnitude for the central difference

output:
  dir: ./out
"""

# name -> (one-line description, template body). Order is the listing order.
_PHONONS = """\
# Supercell finite-displacement phonons: dispersion along a q-path + phonon DOS.
# Run:  gradwave input.yaml -o out/  &&  gradwave plot out/phonons.json
# Norm-conserving, nspin=1. Displaces only the primitive home-cell atoms, so the
# cost is 6·N_prim SCFs regardless of supercell size. Start from a RELAXED cell
# (residual stress/forces shift the frequencies).

structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]

pseudopotentials:
  dir: ./pseudos                 # EDIT: folder holding your UPF files
  map:
    Si: Si_ONCV_PBE-1.2.upf      # EDIT: element -> UPF filename

ecut: 500.0                      # eV, plane-wave cutoff (converge this)
xc: pbe
kpoints:
  mesh: [8, 8, 8]                # primitive k-mesh; folded by the supercell size

task: phonons
phonons:
  supercell: [2, 2, 2]           # diagonal supercell for the force constants
  displacement: 0.01             # atomic displacement h [Å] for the central FD
  path: ""                       # ASE bandpath string (e.g. GXWKGL); "" = default
  npoints: 120                   # q-points along the dispersion
  dos_mesh: [8, 8, 8]            # MP q-mesh for the DOS ([0,0,0] to skip)

output:
  dir: ./out
"""

_TEMPLATES: dict[str, tuple[str, str]] = {
    "scf": ("Single-point SCF of an insulator.", _SCF),
    "metal": ("SCF of a metal (smearing + dense k-mesh).", _METAL),
    "relax": ("Ionic relaxation at a fixed cell.", _RELAX),
    "relax-cell": ("Variable-cell relaxation (atoms + lattice).", _RELAX_CELL),
    "bands": ("Band structure along a k-path.", _BANDS),
    "bands-soc": ("Spin-orbit band structure (noncollinear, FR pseudo).", _BANDS_SOC),
    "pdos": ("Projected density of states.", _PDOS),
    "magnetism": ("Collinear magnetism + exchange couplings.", _MAGNETISM),
    "noncollinear": ("Noncollinear SCF with spin-orbit coupling.", _NONCOLLINEAR),
    "eos": ("Equation of state (bulk modulus via Birch-Murnaghan).", _EOS),
    "elastic": ("Elastic constants (6×6 stiffness + VRH moduli).", _ELASTIC),
    "phonons": ("Supercell phonon dispersion + DOS.", _PHONONS),
}


def names() -> list[str]:
    """Template names in listing order."""
    return list(_TEMPLATES)


def summaries() -> dict[str, str]:
    """name -> one-line description, for `gradwave init` with no argument."""
    return {name: desc for name, (desc, _) in _TEMPLATES.items()}


def render(name: str) -> str:
    """The template body for `name`. Raises KeyError if there is no such template."""
    return _TEMPLATES[name][1]
