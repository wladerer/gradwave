"""YAML input schema → validated frozen dataclasses (Layer C).

Geometry goes through ASE, so `structure:` accepts any format ASE reads
(cif, POSCAR, xyz, ...) or an inline cell/positions/species block.
All energies in eV, lengths in Å (the package-wide convention).
"""

from __future__ import annotations

import dataclasses
import difflib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import read as ase_read


class InputError(ValueError):
    """A malformed input file. Carries the input path (prepended by
    ``load_input``) so the message points at the file the user edited."""


@dataclass(frozen=True)
class MixingParams:
    scheme: str = "pulay"  # pulay | broyden | johnson (USPP/PAW path)
    alpha: float = 0.7
    history: int | None = None  # None → per-scheme default (johnson 12, else 8)
    kerker: str | bool = "auto"  # auto: on iff smearing enabled


@dataclass(frozen=True)
class SCFParams:
    max_iter: int = 100
    etol: float = 1.0e-8
    rhotol: float = 1.0e-7
    mixing: MixingParams = field(default_factory=MixingParams)
    diago_tol: float = 1.0e-9


@dataclass(frozen=True)
class SmearingParams:
    type: str = "none"  # none | fermi-dirac | gaussian | mp1 | cold
    width: float = 0.1  # eV


@dataclass(frozen=True)
class KPointsParams:
    mesh: tuple[int, int, int] = (1, 1, 1)
    shift: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class RelaxParams:
    # bfgs is the default: on displaced diamond it needs 3 steps where
    # fire needs 25 (measured 2026-07-15); fire remains available for
    # far-from-minimum or noisy-force cases
    optimizer: str = "bfgs"  # bfgs | fire
    fmax: float = 0.01  # eV/Å (also gates stress under cell relaxation)
    max_steps: int = 200
    cell: bool = False  # variable-cell: relax the lattice with the atoms (stress)
    pressure: float = 0.0  # external hydrostatic pressure [GPa]; cell relaxation only


@dataclass(frozen=True)
class BandsParams:
    path: str = ""  # ASE bandpath string, e.g. "LGXUG"; empty = ASE default
    npoints: int = 120
    nbands: int | None = None
    irreps: bool = False  # label bands at special points with Mulliken symbols


@dataclass(frozen=True)
class ProjectionsParams:
    enabled: bool = False
    group_by: str = "l"      # atom | l | lm | total (j | jmj for FR)
    width: float = 0.1       # gaussian broadening [eV]
    npoints: int = 800


@dataclass(frozen=True)
class VolumetricParams:
    """Volumetric fields to export after an SCF, as .cube/.xsf for VESTA/Ovito."""

    density: bool = False        # ρ(r), the CHGCAR analog
    elf: bool = False            # electron localization function ELF(r)
    magnetization: bool = False  # |m(r)|, noncollinear/SOC runs only
    bands: tuple = ()            # (band, kpoint) pairs → PARCHG |ψ_nk(r)|²
    format: str = "cube"         # "cube" or "xsf"

    def any(self) -> bool:
        return bool(self.density or self.elf or self.magnetization or self.bands)


@dataclass(frozen=True)
class EOSParams:
    """Isotropic volume scan → 3rd-order Birch-Murnaghan fit (V0, B0, B0')."""

    # volume factors relative to the input cell; the default is the calcDelta /
    # Lejaeghere seven-point window (94–106% of V0). Needs ≥4 points to fit.
    scales: tuple = (0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06)
    energy: str = "free_energy"  # free_energy | total | e0 — quantity fitted vs V

    def __post_init__(self):
        # coerce a YAML list to a tuple (frozen dataclass hashability) and
        # validate at parse time rather than deep in the driver
        object.__setattr__(self, "scales", tuple(float(s) for s in self.scales))
        if len(self.scales) < 4:
            raise InputError(
                f"eos.scales needs >=4 volume factors for a Birch-Murnaghan "
                f"fit, got {len(self.scales)}")
        if self.energy not in ("free_energy", "total", "e0"):
            raise InputError(
                f"unknown eos.energy {self.energy!r} (free_energy | total | e0)")


@dataclass(frozen=True)
class ElasticParams:
    """Clamped-ion elastic constants: FD of the analytic stress over the six
    Voigt strains → the 6×6 stiffness C (and Voigt–Reuss–Hill moduli)."""

    strain: float = 0.005  # Voigt strain magnitude h for the central difference

    def __post_init__(self):
        if not 0.0 < self.strain < 0.1:
            raise InputError(
                f"elastic.strain must be in (0, 0.1), got {self.strain}")


@dataclass(frozen=True)
class MagnetismParams:
    exchange: bool = True      # extract J/D from the torque (adds ~3 constrained SCFs)
    lam: float = 8.0           # constraint penalty strength [eV/μB²]
    delta: float = 0.08        # moment-tilt step for the torque derivative [rad]
    seed_scale: float = 1.5    # high-spin seed for the reference SCF (multi-stable)
    ref_atom: int = 0          # atom whose moment is tilted for the exchange scan


@dataclass(frozen=True)
class Input:
    atoms: Atoms
    pseudo_dir: Path
    pseudo_map: dict[str, str]
    ecut: float
    ecutrho: float | None = None  # USPP/PAW density cutoff; None → 4×ecut
    xc: str = "pbe"  # lda | pbe
    kpoints: KPointsParams = field(default_factory=KPointsParams)
    smearing: SmearingParams = field(default_factory=SmearingParams)
    nbands: int | None = None
    scf: SCFParams = field(default_factory=SCFParams)
    symmetry: bool = True  # IBZ reduction + density symmetrization
    nspin: int = 1  # 1 | 2 (collinear)
    noncollinear: bool = False  # spinor (non-collinear) SCF for task: scf
    nonmagnetic: bool = False  # with noncollinear: pin m⃗ ≡ 0 (spin-orbit only, keeps symmetry)
    start_mag: dict | None = None  # element -> initial moment fraction (nspin=2/NC seed)
    task: str = "scf"  # scf | relax | bands | magnetism | eos | elastic
    relax: RelaxParams = field(default_factory=RelaxParams)
    bands: BandsParams = field(default_factory=BandsParams)
    magnetism: MagnetismParams = field(default_factory=MagnetismParams)
    eos: EOSParams = field(default_factory=EOSParams)
    elastic: ElasticParams = field(default_factory=ElasticParams)
    projections: ProjectionsParams = field(default_factory=ProjectionsParams)
    device: str = "cpu"
    verbose: bool = True  # per-iteration SCF chatter; CLI --quiet overrides
    output_dir: Path = Path("./out")
    output_checkpoint: bool = True  # write checkpoint.pt after SCF tasks
    output_wavefunctions: bool = False  # include coeffs in the checkpoint
    output_volumetric: VolumetricParams = field(default_factory=VolumetricParams)
    # post-SCF numerical-error estimates (basis set, SCF, smearing) in the
    # output — on by default; every estimate is derived from the finished
    # run (no extra SCF) and out-of-coverage runs degrade to available: false
    error_estimate: bool = True
    restart: Path | None = None  # checkpoint.pt to warm-start from (USPP/PAW)


def _check_keys(label: str, got, allowed) -> None:
    """Reject unknown keys in a mapping with a did-you-mean hint, so a typo
    like `optimzer:` fails loudly at parse time instead of being silently
    dropped (a dropped key means the default is used and the result is quietly
    wrong)."""
    if not isinstance(got, dict):
        raise InputError(f"{label} must be a mapping, got {type(got).__name__}")
    allowed = set(allowed)
    unknown = [k for k in got if k not in allowed]
    if not unknown:
        return
    parts = []
    for k in unknown:
        near = difflib.get_close_matches(str(k), [str(a) for a in allowed], n=1)
        parts.append(f"{k!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
    raise InputError(
        f"unknown key(s) in {label}: {', '.join(parts)}. "
        f"valid keys: {', '.join(sorted(str(a) for a in allowed))}")


def _build(cls, raw, label):
    """Construct a frozen params dataclass from a mapping, rejecting unknown
    keys first so `RelaxParams(**{'optimzer': ...})` reports the typo by name
    rather than raising a bare TypeError from the constructor."""
    _check_keys(label, raw, {f.name for f in dataclasses.fields(cls)})
    return cls(**raw)


def _build_volumetric(raw) -> VolumetricParams:
    """Parse the `output.volumetric` block. `true` is shorthand for the density
    alone; a mapping selects fields and the file format."""
    if isinstance(raw, bool):
        return VolumetricParams(density=raw)
    _check_keys("output.volumetric", raw,
                {"density", "elf", "magnetization", "bands", "format"})
    fmt = str(raw.get("format", "cube"))
    if fmt not in ("cube", "xsf", "chgcar"):
        raise InputError(
            f"output.volumetric.format must be 'cube', 'xsf' or 'chgcar', got {fmt!r}")
    try:
        bands = tuple((int(b), int(k)) for b, k in raw.get("bands", ()))
    except (TypeError, ValueError) as exc:
        raise InputError(
            "output.volumetric.bands must be a list of [band, kpoint] pairs") from exc
    return VolumetricParams(
        density=bool(raw.get("density", False)),
        elf=bool(raw.get("elf", False)),
        magnetization=bool(raw.get("magnetization", False)),
        bands=bands,
        format=fmt,
    )


def _read_atoms(path: Path, fmt=None, index=-1) -> Atoms:
    """Read a geometry through ASE and enforce the plane-wave prerequisites.

    ASE guesses the format from the extension/content; `fmt` overrides that
    when the guess misfires. Multi-image files (trajectories, multi-frame xyz)
    default to the last frame (`index=-1`) rather than silently — pass a
    `structure.index` to choose. A structure with no 3D cell cannot be run by a
    plane-wave code, so that fails here with a clear message rather than deep in
    grid construction."""
    try:
        atoms = ase_read(str(path), format=fmt, index=index)
    except FileNotFoundError:
        raise FileNotFoundError(f"structure file not found: {path}") from None
    except Exception as e:  # ASE raises a grab-bag of parse errors
        hint = "" if fmt else " (try setting structure.format)"
        raise InputError(f"could not read structure {path}: {e}{hint}") from None
    if isinstance(atoms, list):
        raise InputError(
            f"structure.index {index!r} selected {len(atoms)} frames; "
            f"give a single integer index (e.g. 0 or -1)")
    if atoms.cell.rank < 3:
        raise InputError(
            f"structure {path} has no 3D cell (cell rank {atoms.cell.rank}); "
            f"plane-wave DFT requires a periodic cell. If this is a molecule, "
            f"put it in a box (e.g. a POSCAR/cif with a lattice).")
    atoms.pbc = True
    return atoms


def _normalize_kerker(value):
    """MixingParams.kerker accepts "auto", a bool, or the on/off/true/false
    string spellings (a bare `kerker: off` is already a YAML bool); anything
    else is a user error rather than a silent truthy string."""
    if value == "auto" or isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "auto":
            return "auto"
        if s in ("on", "true"):
            return True
        if s in ("off", "false"):
            return False
    raise InputError(
        f"invalid mixing.kerker {value!r} (auto | on | off | true | false)")


def _load_structure(spec, base: Path) -> Atoms:
    """Three spellings, all reaching the same Atoms:

      structure: geometry.cif                     # bare filename (any ASE format)
      structure: {file: t.xyz, format: extxyz, index: 0}   # file + read controls
      structure: {cell: ..., positions: ..., species: ...} # inline block
    """
    if isinstance(spec, str):
        return _read_atoms(base / spec)
    if not isinstance(spec, dict):
        raise InputError(
            f"structure must be a filename or a mapping, got {type(spec).__name__}")
    if "file" in spec:
        _check_keys("structure", spec, {"file", "format", "index"})
        return _read_atoms(base / spec["file"], fmt=spec.get("format"),
                           index=spec.get("index", -1))
    _check_keys("structure", spec, {"cell", "positions", "species"})
    for req in ("cell", "positions", "species"):
        if req not in spec:
            raise InputError(f"inline structure is missing required key {req!r}")
    cell = np.asarray(spec["cell"], dtype=float)
    posblock = spec["positions"]
    _check_keys("structure.positions", posblock, {"cart", "frac"})
    species = spec["species"]
    if "frac" in posblock:
        atoms = Atoms(species, scaled_positions=posblock["frac"], cell=cell, pbc=True)
    elif "cart" in posblock:
        atoms = Atoms(species, positions=posblock["cart"], cell=cell, pbc=True)
    else:
        raise InputError("structure.positions needs a 'cart' or 'frac' block")
    return atoms


# Every top-level key the schema understands; anything else is a typo. Kept
# beside the Input fields it feeds so the two do not drift.
_ALLOWED_TOP = {
    "structure", "pseudopotentials", "ecut", "ecutrho", "xc", "kpoints",
    "smearing", "nbands", "symmetry", "nspin", "noncollinear", "nonmagnetic",
    "start_mag",
    "scf", "task", "relax", "bands", "magnetism", "eos", "elastic",
    "projections", "device",
    "verbose", "output", "error_estimate", "restart",
}


def load_input(path: str | Path) -> Input:
    """Parse a YAML input into the frozen `Input` schema. Every `InputError`
    is re-raised with the file path prepended so the message names the file the
    user edited."""
    path = Path(path)
    try:
        return _load_input(path)
    except InputError as e:
        raise InputError(f"{path}: {e}") from None


def _resolve_pseudopotentials(pp, base: Path, symbols) -> tuple[Path, dict]:
    """Validate the pseudopotentials block, resolve its directory and per-element
    map, and check every element in ``symbols`` has an existing UPF file."""
    _check_keys("pseudopotentials", pp, {"dir", "map"})
    for req in ("dir", "map"):
        if req not in pp:
            raise InputError(f"pseudopotentials is missing required key {req!r}")
    pseudo_dir = (base / pp["dir"]).resolve()
    pseudo_map = dict(pp["map"])
    for sym in set(symbols):
        if sym not in pseudo_map:
            raise InputError(f"no pseudopotential mapped for element {sym}")
        if not (pseudo_dir / pseudo_map[sym]).exists():
            raise FileNotFoundError(pseudo_dir / pseudo_map[sym])
    return pseudo_dir, pseudo_map


def _resolve_symmetry(raw, task: str) -> tuple[bool, bool, bool]:
    """Resolve (noncollinear, nonmagnetic, symmetry) from the raw input.

    A magnetic spinor SCF (a magnetic noncollinear run, and the magnetism
    task's constrained tilt scans) cannot use IBZ symmetry reduction: time
    reversal and the space group act on the moment vector, so the driver
    rejects a symmetrized density. Default symmetry off for these modes and
    reject an explicit ``symmetry: true`` here, where the message can point at
    the fix, rather than letting it surface as a ValueError deep in the SCF.
    A spin-orbit-only run (nonmagnetic: true) pins m⃗ ≡ 0, so Kramers keeps
    the full crystal symmetry and it behaves like a plain SCF for symmetry.
    """
    noncollinear = bool(raw.get("noncollinear", False))
    nonmagnetic = bool(raw.get("nonmagnetic", False))
    if nonmagnetic and not noncollinear:
        raise InputError(
            "nonmagnetic requires noncollinear: true — it pins the spinor "
            "moment to zero for a spin-orbit-only run")
    magnetic_spinor = (noncollinear and not nonmagnetic) or task == "magnetism"
    sym_raw = raw.get("symmetry")
    if magnetic_spinor:
        if sym_raw is True:
            mode = "magnetism" if task == "magnetism" else "noncollinear"
            raise InputError(
                f"symmetry: true is invalid for a {mode} run — time reversal "
                f"and the space group act on the moment vector, so IBZ "
                f"reduction is rejected. Set symmetry: false (the default for "
                f"these modes), or for spin-orbit without magnetism add "
                f"nonmagnetic: true, which keeps symmetry.")
        symmetry = False
    else:
        symmetry = True if sym_raw is None else bool(sym_raw)
    return noncollinear, nonmagnetic, symmetry


def _validate_mixing(mix_raw: dict) -> None:
    """Validate the `scf.mixing` block in place: reject unknown keys and unknown
    schemes, and normalize the `kerker` shorthand. Mutates `mix_raw['kerker']`."""
    _check_keys("scf.mixing", mix_raw,
                {f.name for f in dataclasses.fields(MixingParams)})
    mix_scheme = str(mix_raw.get("scheme", "pulay"))
    if mix_scheme not in ("pulay", "broyden", "johnson"):
        raise InputError(f"unknown mixing scheme {mix_scheme!r}")
    if "kerker" in mix_raw:
        mix_raw["kerker"] = _normalize_kerker(mix_raw["kerker"])


def _build_projections(proj_raw) -> ProjectionsParams:
    """Parse the `projections` block. `true`/`false` is the enabled shorthand;
    a mapping selects the grouping and broadening."""
    if isinstance(proj_raw, bool):
        return ProjectionsParams(enabled=proj_raw)
    _check_keys("projections", proj_raw,
                {"enabled", "group_by", "width", "npoints"})
    return ProjectionsParams(
        enabled=bool(proj_raw.get("enabled", True)),
        group_by=str(proj_raw.get("group_by", "l")),
        width=float(proj_raw.get("width", 0.1)),
        npoints=int(proj_raw.get("npoints", 800)),
    )


def _load_input(path: Path) -> Input:
    raw = yaml.safe_load(path.read_text())
    base = path.parent

    if not isinstance(raw, dict):
        raise InputError("input must be a YAML mapping of keywords")
    _check_keys("input", raw, _ALLOWED_TOP)
    for req in ("structure", "pseudopotentials", "ecut"):
        if req not in raw:
            raise InputError(f"missing required key {req!r}")

    atoms = _load_structure(raw["structure"], base)
    pseudo_dir, pseudo_map = _resolve_pseudopotentials(
        raw["pseudopotentials"], base, atoms.get_chemical_symbols())

    kp = raw.get("kpoints", {})
    _check_keys("kpoints", kp, {"mesh", "shift"})
    sm = raw.get("smearing", {})
    _check_keys("smearing", sm, {"type", "width"})
    scf_raw = dict(raw.get("scf", {}))
    _check_keys("scf", scf_raw, {"max_iter", "etol", "rhotol", "mixing", "diago"})
    mix_raw = dict(scf_raw.pop("mixing", {}))
    diago = scf_raw.pop("diago", {})
    _check_keys("scf.diago", diago, {"tol"})

    xc = str(raw.get("xc", "pbe")).lower()
    if xc not in ("lda", "pbe", "r2scan"):
        raise InputError(f"unknown xc {xc!r} (lda | pbe | r2scan)")
    task = raw.get("task", "scf")
    if task not in ("scf", "relax", "bands", "magnetism", "eos", "elastic"):
        raise InputError(
            f"unknown task {task!r} "
            f"(scf | relax | bands | magnetism | eos | elastic)")
    nspin = int(raw.get("nspin", 1))
    if nspin not in (1, 2):
        raise InputError(f"nspin must be 1 or 2, got {nspin}")

    noncollinear, nonmagnetic, symmetry = _resolve_symmetry(raw, task)

    mesh = tuple(kp.get("mesh", (1, 1, 1)))
    if len(mesh) != 3:
        raise InputError(f"kpoints.mesh must have 3 entries, got {list(mesh)}")
    smtype = sm.get("type", "none")
    if smtype not in ("none", "fermi-dirac", "gaussian", "mp1", "cold"):
        raise InputError(f"unknown smearing type {smtype!r}")

    _validate_mixing(mix_raw)

    out_raw = raw.get("output", {})
    _check_keys("output", out_raw,
                {"dir", "checkpoint", "wavefunctions", "error_estimate", "volumetric"})
    volumetric = _build_volumetric(out_raw.get("volumetric", False))
    restart = raw.get("restart")

    nbands = raw.get("nbands", "auto")
    ecutrho = raw.get("ecutrho")
    projections = _build_projections(raw.get("projections", False))
    return Input(
        atoms=atoms,
        pseudo_dir=pseudo_dir,
        pseudo_map=pseudo_map,
        ecut=float(raw["ecut"]),
        ecutrho=None if ecutrho is None else float(ecutrho),
        xc=xc,
        kpoints=KPointsParams(
            mesh=mesh, shift=tuple(kp.get("shift", (0, 0, 0)))
        ),
        smearing=SmearingParams(type=smtype, width=float(sm.get("width", 0.1))),
        nbands=None if nbands == "auto" else int(nbands),
        symmetry=symmetry,
        nspin=nspin,
        noncollinear=noncollinear,
        nonmagnetic=nonmagnetic,
        start_mag=raw.get("start_mag"),
        scf=SCFParams(
            max_iter=int(scf_raw.get("max_iter", 100)),
            etol=float(scf_raw.get("etol", 1e-8)),
            rhotol=float(scf_raw.get("rhotol", 1e-7)),
            mixing=MixingParams(**mix_raw) if mix_raw else MixingParams(),
            diago_tol=float(diago.get("tol", 1e-9)),
        ),
        task=task,
        relax=_build(RelaxParams, raw.get("relax", {}), "relax"),
        bands=_build(BandsParams, raw.get("bands", {}), "bands"),
        magnetism=_build(MagnetismParams, raw.get("magnetism", {}), "magnetism"),
        eos=_build(EOSParams, raw.get("eos", {}), "eos"),
        elastic=_build(ElasticParams, raw.get("elastic", {}), "elastic"),
        projections=projections,
        device=raw.get("device", "cpu"),
        verbose=bool(raw.get("verbose", True)),
        output_dir=base / out_raw.get("dir", "./out"),
        output_checkpoint=bool(out_raw.get("checkpoint", True)),
        output_wavefunctions=bool(out_raw.get("wavefunctions", False)),
        output_volumetric=volumetric,
        error_estimate=bool(out_raw.get("error_estimate",
                                        raw.get("error_estimate", True))),
        restart=None if restart is None else (base / restart),
    )
