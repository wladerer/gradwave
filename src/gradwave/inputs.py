"""YAML input schema → validated frozen dataclasses (Layer C).

Geometry goes through ASE, so `structure:` accepts any format ASE reads
(cif, POSCAR, xyz, ...) or an inline cell/positions/species block.
All energies in eV, lengths in Å (the package-wide convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import read as ase_read


@dataclass(frozen=True)
class MixingParams:
    scheme: str = "pulay"  # pulay | linear
    alpha: float = 0.7
    history: int = 8
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
    optimizer: str = "fire"  # fire | bfgs
    fmax: float = 0.01  # eV/Å
    max_steps: int = 200


@dataclass(frozen=True)
class BandsParams:
    path: str = ""  # ASE bandpath string, e.g. "LGXUG"; empty = ASE default
    npoints: int = 120
    nbands: int | None = None
    irreps: bool = False  # label bands at special points with Mulliken symbols


@dataclass(frozen=True)
class Input:
    atoms: Atoms
    pseudo_dir: Path
    pseudo_map: dict[str, str]
    ecut: float
    xc: str = "pbe"  # lda | pbe
    kpoints: KPointsParams = field(default_factory=KPointsParams)
    smearing: SmearingParams = field(default_factory=SmearingParams)
    nbands: int | None = None
    scf: SCFParams = field(default_factory=SCFParams)
    symmetry: bool = True  # IBZ reduction + density symmetrization
    task: str = "scf"  # scf | relax | bands
    relax: RelaxParams = field(default_factory=RelaxParams)
    bands: BandsParams = field(default_factory=BandsParams)
    device: str = "cpu"
    output_dir: Path = Path("./out")


def _load_structure(spec, base: Path) -> Atoms:
    if isinstance(spec, str):
        return ase_read(base / spec)
    cell = np.asarray(spec["cell"], dtype=float)
    posblock = spec["positions"]
    species = spec["species"]
    if "frac" in posblock:
        atoms = Atoms(species, scaled_positions=posblock["frac"], cell=cell, pbc=True)
    else:
        atoms = Atoms(species, positions=posblock["cart"], cell=cell, pbc=True)
    return atoms


def load_input(path: str | Path) -> Input:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base = path.parent

    atoms = _load_structure(raw["structure"], base)
    pp = raw["pseudopotentials"]
    pseudo_dir = (base / pp["dir"]).resolve()
    pseudo_map = dict(pp["map"])
    for sym in set(atoms.get_chemical_symbols()):
        if sym not in pseudo_map:
            raise ValueError(f"no pseudopotential mapped for element {sym}")
        if not (pseudo_dir / pseudo_map[sym]).exists():
            raise FileNotFoundError(pseudo_dir / pseudo_map[sym])

    kp = raw.get("kpoints", {})
    sm = raw.get("smearing", {})
    scf_raw = dict(raw.get("scf", {}))
    mix_raw = scf_raw.pop("mixing", {})
    diago = scf_raw.pop("diago", {})

    xc = str(raw.get("xc", "pbe")).lower()
    if xc not in ("lda", "pbe"):
        raise ValueError(f"unknown xc {xc!r} (lda | pbe)")
    task = raw.get("task", "scf")
    if task not in ("scf", "relax", "bands"):
        raise ValueError(f"unknown task {task!r}")
    smtype = sm.get("type", "none")
    if smtype not in ("none", "fermi-dirac", "gaussian", "mp1", "cold"):
        raise ValueError(f"unknown smearing type {smtype!r}")

    nbands = raw.get("nbands", "auto")
    return Input(
        atoms=atoms,
        pseudo_dir=pseudo_dir,
        pseudo_map=pseudo_map,
        ecut=float(raw["ecut"]),
        xc=xc,
        kpoints=KPointsParams(
            mesh=tuple(kp.get("mesh", (1, 1, 1))), shift=tuple(kp.get("shift", (0, 0, 0)))
        ),
        smearing=SmearingParams(type=smtype, width=float(sm.get("width", 0.1))),
        nbands=None if nbands == "auto" else int(nbands),
        symmetry=bool(raw.get("symmetry", True)),
        scf=SCFParams(
            max_iter=int(scf_raw.get("max_iter", 100)),
            etol=float(scf_raw.get("etol", 1e-8)),
            rhotol=float(scf_raw.get("rhotol", 1e-7)),
            mixing=MixingParams(**mix_raw) if mix_raw else MixingParams(),
            diago_tol=float(diago.get("tol", 1e-9)),
        ),
        task=task,
        relax=RelaxParams(**raw.get("relax", {})),
        bands=BandsParams(**raw.get("bands", {})),
        device=raw.get("device", "cpu"),
        output_dir=base / raw.get("output", {}).get("dir", "./out"),
    )
