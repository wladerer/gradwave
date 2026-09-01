"""Input-file loading and validation: load_input and its builders."""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import yaml
from ase import Atoms

from gradwave.inputs.models import (
    BandsParams,
    CohpParams,
    DispersionParams,
    ElasticParams,
    EOSParams,
    FlapwParams,
    HubbardManifoldSpec,
    HubbardParams,
    HybridParams,
    Input,
    InputError,
    KPointsParams,
    MagneticParams,
    MagnetismParams,
    MixingParams,
    NebParams,
    NmrParams,
    NmrSpectrumParams,
    PhononParams,
    ProjectionsParams,
    RelaxParams,
    SCFParams,
    SlabParams,
    SmearingParams,
    VolumetricParams,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_T = TypeVar("_T", bound="DataclassInstance")


def _check_keys(label: str, got: object, allowed: Iterable[str]) -> None:
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


def _build(cls: type[_T], raw: dict[str, Any], label: str) -> _T:
    """Construct a frozen params dataclass from a mapping, rejecting unknown
    keys first so `RelaxParams(**{'optimzer': ...})` reports the typo by name
    rather than raising a bare TypeError from the constructor."""
    _check_keys(label, raw, {f.name for f in dataclasses.fields(cls)})
    return cls(**raw)


def _build_volumetric(raw: bool | dict[str, Any]) -> VolumetricParams:
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


def _read_atoms(path: Path, fmt: str | None = None, index: int = -1) -> Atoms:
    """Read a geometry through ASE and enforce the plane-wave prerequisites.

    ASE guesses the format from the extension/content; `fmt` overrides that
    when the guess misfires. Multi-image files (trajectories, multi-frame xyz)
    default to the last frame (`index=-1`) rather than silently — pass a
    `structure.index` to choose. A structure with no 3D cell cannot be run by a
    plane-wave code, so that fails here with a clear message rather than deep in
    grid construction."""
    # Deferred: `ase.io.read` drags in ASE's format-plugin registry (scipy.integrate,
    # ase.spacegroup, ...), ~0.6 s that every `import gradwave` would otherwise pay for
    # a geometry-read that happens once per run, off the SCF/autograd path.
    from ase.io import read as ase_read

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


def _normalize_kerker(value: object) -> str | bool:
    """MixingParams.kerker accepts "auto", a bool, or the on/off/true/false
    string spellings (a bare `kerker: off` is already a YAML bool); anything
    else is a user error rather than a silent truthy string."""
    if value == "auto":
        return "auto"
    if isinstance(value, bool):
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


def _parse_fixed(raw_fixed: Any, natoms: int) -> np.ndarray:
    """Parse ``structure.fixed`` into a ``(natoms, 3)`` boolean selective-dynamics
    mask where ``True`` means that fractional axis is held fixed in a relax.

    Two spellings, validated against the atom count at parse time:

      fixed: [0, 3, 4]                        # 0-based indices, atoms fully fixed
      fixed: [[true, true, true], [false, false, false], ...]  # per-atom [x,y,z]

    The index form fixes all three axes of each listed atom; the per-atom form
    needs one ``[x, y, z]`` boolean row per atom (length == natoms). A YAML bool
    is a Python ``int`` subclass, so the index form rejects booleans to keep the
    two spellings distinct.
    """
    if not isinstance(raw_fixed, (list, tuple)) or len(raw_fixed) == 0:
        raise InputError(
            "structure.fixed must be a non-empty list of atom indices or a "
            "per-atom list of [x, y, z] booleans")
    mask = np.zeros((natoms, 3), dtype=bool)
    if all(isinstance(e, (list, tuple)) for e in raw_fixed):
        # per-atom [x, y, z] boolean rows
        if len(raw_fixed) != natoms:
            raise InputError(
                f"structure.fixed per-atom mask has {len(raw_fixed)} rows but "
                f"the structure has {natoms} atoms")
        for i, row in enumerate(raw_fixed):
            if len(row) != 3:
                raise InputError(
                    f"structure.fixed[{i}] must be 3 booleans [x, y, z], got "
                    f"{len(row)}")
            for j, v in enumerate(row):
                if not isinstance(v, bool):
                    raise InputError(
                        f"structure.fixed[{i}][{j}] must be a boolean "
                        f"(true/false), got {v!r}")
                mask[i, j] = v
        if not mask.any():
            raise InputError(
                "structure.fixed pins no axes (every entry false); omit it to "
                "relax every atom")
        return mask
    if all(isinstance(e, int) and not isinstance(e, bool) for e in raw_fixed):
        # flat list of 0-based atom indices, fully fixed
        idx = list(raw_fixed)
        for i in idx:
            if not 0 <= i < natoms:
                raise InputError(
                    f"structure.fixed index {i} out of range for {natoms} atoms "
                    f"(valid 0..{natoms - 1})")
        if len(set(idx)) != len(idx):
            raise InputError(f"structure.fixed has duplicate indices: {idx}")
        mask[idx, :] = True
        return mask
    raise InputError(
        "structure.fixed must be either a flat list of integer atom indices or "
        "a per-atom list of [x, y, z] boolean rows, not a mix of the two")


def _load_structure(
    spec: str | dict[str, Any], base: Path
) -> tuple[Atoms, np.ndarray | None]:
    """Three spellings, all reaching the same Atoms (plus an optional
    selective-dynamics mask parsed from ``fixed`` in either mapping form):

      structure: geometry.cif                     # bare filename (any ASE format)
      structure: {file: t.xyz, format: extxyz, index: 0}   # file + read controls
      structure: {cell: ..., positions: ..., species: ...} # inline block

    Returns ``(atoms, fixed)`` where ``fixed`` is the ``(natoms, 3)`` boolean
    mask or ``None`` when no ``fixed`` key is given (the bare-filename form
    cannot carry one — use a mapping to fix atoms with a file geometry).
    """
    if isinstance(spec, str):
        return _read_atoms(base / spec), None
    if not isinstance(spec, dict):
        raise InputError(
            f"structure must be a filename or a mapping, got {type(spec).__name__}")
    if "file" in spec:
        _check_keys("structure", spec, {"file", "format", "index", "fixed"})
        atoms = _read_atoms(base / spec["file"], fmt=spec.get("format"),
                            index=spec.get("index", -1))
    else:
        _check_keys("structure", spec, {"cell", "positions", "species", "fixed"})
        for req in ("cell", "positions", "species"):
            if req not in spec:
                raise InputError(f"inline structure is missing required key {req!r}")
        cell = np.asarray(spec["cell"], dtype=float)
        posblock = spec["positions"]
        _check_keys("structure.positions", posblock, {"cart", "frac"})
        species = spec["species"]
        if "frac" in posblock:
            atoms = Atoms(species, scaled_positions=posblock["frac"], cell=cell,
                          pbc=True)
        elif "cart" in posblock:
            atoms = Atoms(species, positions=posblock["cart"], cell=cell, pbc=True)
        else:
            raise InputError("structure.positions needs a 'cart' or 'frac' block")
    fixed = None if "fixed" not in spec else _parse_fixed(spec["fixed"], len(atoms))
    return atoms, fixed


# Every top-level key the schema understands; anything else is a typo. Kept
# beside the Input fields it feeds so the two do not drift.
_ALLOWED_TOP = {
    "structure", "pseudopotentials", "ecut", "ecutrho", "xc", "hybrid", "hubbard",
    "kpoints", "smearing", "nbands", "symmetry", "nspin", "noncollinear",
    "nonmagnetic", "start_mag", "tot_magnetization",
    "scf", "slab", "task", "relax", "neb", "bands", "magnetism", "eos", "elastic",
    "phonons", "flapw", "nmr",
    "projections", "dispersion", "device", "distributed",
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


def _resolve_pseudopotentials(
    pp: dict[str, Any], base: Path, symbols: Iterable[str]
) -> tuple[Path, dict[str, str]]:
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
        resolved = pseudo_dir / pseudo_map[sym]
        if not resolved.exists():
            raise InputError(
                f"pseudopotential for element {sym!r} not found at {resolved} — "
                f"check pseudopotentials.dir and pseudopotentials.map")
    return pseudo_dir, pseudo_map


def _resolve_symmetry(raw: dict[str, Any], task: str) -> tuple[bool, bool, bool]:
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


def _validate_mixing(mix_raw: dict[str, Any]) -> None:
    """Validate the `scf.mixing` block in place: reject unknown keys and unknown
    schemes, and normalize the `kerker` shorthand. Mutates `mix_raw['kerker']`."""
    _check_keys("scf.mixing", mix_raw,
                {f.name for f in dataclasses.fields(MixingParams)})
    mix_scheme = str(mix_raw.get("scheme", "auto"))
    if mix_scheme not in ("auto", "pulay", "broyden", "johnson"):
        raise InputError(f"unknown mixing scheme {mix_scheme!r}")
    if "precond" in mix_raw:
        precond = str(mix_raw["precond"])
        if precond not in ("kerker", "local_tf"):
            raise InputError(
                f"unknown mixing precond {precond!r} (kerker | local_tf)")
    if "kerker" in mix_raw:
        mix_raw["kerker"] = _normalize_kerker(mix_raw["kerker"])


def _build_cohp(cohp_raw: bool | dict[str, Any]) -> CohpParams:
    """Parse the `projections.cohp` sub-block. `true`/`false` is the enabled
    shorthand; a mapping selects the atom pairs, cutoff and broadening."""
    if isinstance(cohp_raw, bool):
        return CohpParams(enabled=cohp_raw)
    _check_keys("projections.cohp", cohp_raw,
                {"enabled", "pairs", "rcut", "width", "npoints"})
    # pairs is passed through raw; CohpParams.__post_init__ coerces and validates
    # it (so a malformed pair reports the InputError, not a bare TypeError here).
    return CohpParams(
        enabled=bool(cohp_raw.get("enabled", True)),
        pairs=cohp_raw.get("pairs"),
        rcut=float(cohp_raw.get("rcut", 3.0)),
        width=float(cohp_raw.get("width", 0.1)),
        npoints=int(cohp_raw.get("npoints", 800)),
    )


def _build_projections(proj_raw: bool | dict[str, Any]) -> ProjectionsParams:
    """Parse the `projections` block. `true`/`false` is the enabled shorthand;
    a mapping selects the grouping and broadening, plus an optional `cohp`
    sub-block computed alongside the PDOS."""
    if isinstance(proj_raw, bool):
        return ProjectionsParams(enabled=proj_raw)
    _check_keys("projections", proj_raw,
                {"enabled", "group_by", "width", "npoints", "cohp"})
    return ProjectionsParams(
        enabled=bool(proj_raw.get("enabled", True)),
        group_by=str(proj_raw.get("group_by", "l")),
        width=float(proj_raw.get("width", 0.1)),
        npoints=int(proj_raw.get("npoints", 800)),
        cohp=_build_cohp(proj_raw.get("cohp", False)),
    )


def _build_dispersion(disp_raw: bool | dict[str, Any]) -> DispersionParams:
    """Parse the `dispersion` block. `true`/`false` is the enabled shorthand
    (BJ damping then taken from the SCF functional); a mapping overrides the
    functional, cutoffs, and the four damping constants."""
    if isinstance(disp_raw, bool):
        return DispersionParams(enabled=disp_raw)
    _check_keys("dispersion", disp_raw,
                {"enabled", "method", "functional", "cutoff", "cn_cutoff",
                 "s6", "s8", "a1", "a2", "charge"})
    def _optf(key: str) -> float | None:
        v = disp_raw.get(key)
        return None if v is None else float(v)
    return DispersionParams(
        enabled=bool(disp_raw.get("enabled", True)),
        method=str(disp_raw.get("method", "d3")),
        functional=disp_raw.get("functional"),
        cutoff=float(disp_raw.get("cutoff", 21.2)),
        cn_cutoff=float(disp_raw.get("cn_cutoff", 10.6)),
        s6=_optf("s6"), s8=_optf("s8"), a1=_optf("a1"), a2=_optf("a2"),
        charge=float(disp_raw.get("charge", 0.0)),
    )


# xc: pbe0 / hse select a hybrid; the value is (mode, default omega). The
# semilocal base is always PBE (the hybrid machinery scales PBE exchange).
_HYBRID_PRESETS = {
    "pbe0": ("full", None),
    "hse": ("short_range", 0.2),
}


def _resolve_xc(raw: dict[str, Any]) -> tuple[str, HybridParams]:
    """Resolve (semilocal base xc, HybridParams) from the raw `xc` value and an
    optional `hybrid` override block. A plain functional (lda|pbe|r2scan) returns
    a disabled HybridParams and rejects a stray `hybrid` block; a hybrid label
    (pbe0|hse) sets the base to PBE and folds the preset with any overrides."""
    xc = str(raw.get("xc", "pbe")).lower()
    hyb_raw = raw.get("hybrid", {})
    if xc not in _HYBRID_PRESETS:
        if xc not in ("lda", "pbe", "r2scan", "spin_adapted_pbe"):
            raise InputError(
                f"unknown xc {xc!r} "
                "(lda | pbe | r2scan | spin_adapted_pbe | pbe0 | hse)")
        if hyb_raw:
            raise InputError(
                "a hybrid block needs a hybrid functional (xc: pbe0 or hse)")
        return xc, HybridParams()
    mode, omega_def = _HYBRID_PRESETS[xc]
    _check_keys("hybrid", hyb_raw, {"alpha", "omega", "mode"})
    return "pbe", HybridParams(
        enabled=True, name=xc,
        mode=str(hyb_raw.get("mode", mode)),
        alpha=float(hyb_raw.get("alpha", 0.25)),
        omega=float(hyb_raw.get("omega", omega_def if omega_def is not None else 0.2)),
        base="pbe",
    )


def _build_hubbard(
    hub_raw: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any] | None,
    symbols: list[str],
) -> HubbardParams:
    """Parse the `hubbard` block: a list of per-species +U manifolds. Each entry
    names an element `species` present in the structure, the correlated shell
    `l`, and `u` (+ optional `j`) in eV. One manifold per species (Dudarev is a
    per-species correction); a species absent from the cell is a typo, not a
    silent no-op, so it is rejected.

    The block may also be a mapping ``{manifolds: [...], occ_mix, u_ramp_iters}``
    to carry the two convergence aids alongside the manifold list: ``occ_mix``
    (occupation-matrix damping β in (0, 1], default 1.0) and ``u_ramp_iters``
    (linear U-ramp length, non-negative integer, default 0 = off)."""
    if hub_raw is None:
        return HubbardParams()
    occ_mix, u_ramp_iters = 1.0, 0
    if isinstance(hub_raw, dict):
        # mapping form: manifolds list plus the convergence-aid knobs. Require
        # `manifolds` before validating keys so a bare single-manifold mapping
        # (no `manifolds` key) still fails with the list-form guidance.
        if "manifolds" not in hub_raw:
            raise InputError(
                "hubbard must be a list of {species, l, u[, j]} manifolds, or a "
                "mapping {manifolds: [...], occ_mix, u_ramp_iters}")
        _check_keys("hubbard", hub_raw, {"manifolds", "occ_mix", "u_ramp_iters"})
        occ_mix = float(hub_raw.get("occ_mix", 1.0))
        u_ramp_iters = int(hub_raw.get("u_ramp_iters", 0))
        if not (0.0 < occ_mix <= 1.0):
            raise InputError(
                f"hubbard.occ_mix (occupation-matrix damping) must be in "
                f"(0, 1], got {occ_mix}")
        if u_ramp_iters < 0:
            raise InputError(
                f"hubbard.u_ramp_iters must be a non-negative integer, "
                f"got {u_ramp_iters}")
        hub_raw = hub_raw["manifolds"]
    if not isinstance(hub_raw, (list, tuple)):
        raise InputError(
            "hubbard must be a list of {species, l, u[, j]} manifolds")
    present = set(symbols)
    seen: set[str] = set()
    manifolds: list[HubbardManifoldSpec] = []
    for i, entry in enumerate(hub_raw):
        _check_keys(f"hubbard[{i}]", entry, {"species", "l", "u", "j"})
        if "species" not in entry or "l" not in entry or "u" not in entry:
            raise InputError(
                f"hubbard[{i}] needs species, l and u (u in eV; l the shell)")
        sp = str(entry["species"])
        if sp not in present:
            raise InputError(
                f"hubbard[{i}]: species {sp!r} is not in the structure "
                f"(elements present: {', '.join(sorted(present))})")
        if sp in seen:
            raise InputError(
                f"hubbard: species {sp!r} appears twice (one manifold per "
                f"species — Dudarev +U applies to every atom of the element)")
        seen.add(sp)
        manifolds.append(HubbardManifoldSpec(
            species=sp, l=int(entry["l"]), u=float(entry["u"]),
            j=float(entry.get("j", 0.0))))
    return HubbardParams(enabled=bool(manifolds), manifolds=tuple(manifolds),
                         occ_mix=occ_mix, u_ramp_iters=u_ramp_iters)


def _build_flapw(raw: dict[str, Any], symbols: Iterable[str]) -> FlapwParams:
    """Parse the ``flapw`` block (all-electron muffin-tin controls). ``radii`` is
    a per-species muffin-tin radius map (Å) and must cover every element in the
    structure for an EFG/FLAPW run; the LO/basis overrides pass through opaque."""
    _check_keys("flapw", raw, {f.name for f in dataclasses.fields(FlapwParams)})
    radii = {str(k): float(v) for k, v in dict(raw.get("radii", {})).items()}
    present = set(symbols)
    missing = present - set(radii)
    if radii and missing:
        raise InputError(
            f"flapw.radii is missing a muffin-tin radius for {sorted(missing)} "
            f"(every element needs one)")
    kw = {k: raw[k] for k in raw if k not in ("radii",)}
    return FlapwParams(radii=radii, **kw)


def _build_nmr_spectrum(raw: dict[str, Any]) -> NmrSpectrumParams:
    """Parse the ``nmr.spectrum`` block (powder-lineshape synthesis). ``larmor_mhz``
    is either a scalar or a species/isotope → MHz map; the rest are simple scalars."""
    _check_keys(
        "nmr.spectrum", raw, {f.name for f in dataclasses.fields(NmrSpectrumParams)})
    larmor: float | dict[str, float] | None = raw.get("larmor_mhz")
    if isinstance(larmor, dict):
        larmor = {str(k): float(v) for k, v in larmor.items()}
    elif larmor is not None:
        larmor = float(larmor)
    return NmrSpectrumParams(
        enabled=bool(raw.get("enabled", False)),
        mode=str(raw.get("mode", "static")),
        spin_rate_hz=float(raw.get("spin_rate_hz", 0.0)),
        larmor_mhz=larmor,
        broadening_ppm=float(raw.get("broadening_ppm", 0.0)),
        lineshape=str(raw.get("lineshape", "gauss")),
        n_orientations=int(raw.get("n_orientations", 2000)),
        n_points=int(raw.get("n_points", 2048)))


def _build_nmr(raw: dict[str, Any]) -> NmrParams:
    """Parse the ``nmr`` block: the observable ``task`` (efg | shielding), an
    optional per-species ``isotopes`` map for the EFG quadrupolar coupling, the
    shielding assembly ``shielding_level`` (auto | bare | gipaw), an optional
    ``sigma_ref`` species/isotope → σ_ref [ppm] map for chemical shifts, the
    ``efg`` gate (PW/PAW EFG within the shielding task), and the ``spectrum``
    lineshape-synthesis block."""
    _check_keys("nmr", raw, {f.name for f in dataclasses.fields(NmrParams)})
    iso = raw.get("isotopes")
    if iso is not None:
        iso = {str(k): str(v) for k, v in dict(iso).items()}
    sref = raw.get("sigma_ref")
    if sref is not None:
        sref = {str(k): float(v) for k, v in dict(sref).items()}
    efg_raw = raw.get("efg", False)
    efg: bool | str = efg_raw if isinstance(efg_raw, str) else bool(efg_raw)
    spec_raw = raw.get("spectrum")
    spectrum = (_build_nmr_spectrum(dict(spec_raw))
                if isinstance(spec_raw, dict) else NmrSpectrumParams())
    ck = raw.get("chunk_k")
    return NmrParams(
        task=str(raw.get("task", "efg")), isotopes=iso,
        shielding_level=str(raw.get("shielding_level", "auto")), sigma_ref=sref,
        efg=efg, spectrum=spectrum, chunk_k=None if ck is None else int(ck))


def _build_neb(raw: dict[str, Any], base: Path, task: str) -> NebParams:
    """Parse the ``neb`` block, resolving ``final`` (the final-state geometry
    file) relative to the input file's directory. ``final`` is required for a
    ``neb`` task and rejected for any other task (a stray block is a typo)."""
    _check_keys("neb", raw, {f.name for f in dataclasses.fields(NebParams)})
    raw = dict(raw)
    final = raw.get("final")
    if task == "neb":
        if final is None:
            raise InputError(
                "task: neb requires neb.final (the final-state structure file; "
                "the top-level `structure` is the initial state)")
        raw["final"] = base / str(final)
    elif final is not None:
        raise InputError("neb.final is only valid for task: neb")
    return _build(NebParams, raw, "neb")


def _load_input(path: Path) -> Input:
    raw_yaml: Any = yaml.safe_load(path.read_text())
    base = path.parent

    if not isinstance(raw_yaml, dict):
        raise InputError("input must be a YAML mapping of keywords")
    raw: dict[str, Any] = raw_yaml
    _check_keys("input", raw, _ALLOWED_TOP)

    # All-electron FLAPW tasks (task: flapw, and task: nmr with the EFG
    # observable) carry no plane-wave pseudopotentials and no top-level ecut —
    # the muffin-tin stack has its own interstitial cutoff in the `flapw` block.
    # The plane-wave shielding NMR observable (task: nmr, nmr.task: shielding)
    # runs the ordinary PW SCF, so it DOES need pseudopotentials and ecut.
    task = raw.get("task", "scf")
    if task == "transition_state":  # friendly alias for the CI-NEB task
        task = "neb"
    nmr_raw = dict(raw.get("nmr", {}))
    nmr_task = str(nmr_raw.get("task", "efg"))
    all_electron = task == "flapw" or (task == "nmr" and nmr_task == "efg")

    if "structure" not in raw:
        raise InputError("missing required key 'structure'")
    for req in ("pseudopotentials", "ecut"):
        if req not in raw and not all_electron:
            raise InputError(f"missing required key {req!r}")

    atoms, fixed = _load_structure(raw["structure"], base)
    if all_electron and "pseudopotentials" not in raw:
        pseudo_dir, pseudo_map = base, {}
    else:
        pseudo_dir, pseudo_map = _resolve_pseudopotentials(
            raw["pseudopotentials"], base, atoms.get_chemical_symbols())

    kp = raw.get("kpoints", {})
    _check_keys("kpoints", kp, {"mesh", "shift", "kspacing"})
    sm = raw.get("smearing", {})
    _check_keys("smearing", sm, {"type", "width"})
    scf_raw = dict(raw.get("scf", {}))
    _check_keys("scf", scf_raw,
                {"max_iter", "etol", "rhotol", "mixing", "diago", "trace",
                 "convergence", "entol", "eigensolver", "magnetic",
                 "boundary", "esm_bias", "target_mu"})
    mix_raw = dict(scf_raw.pop("mixing", {}))
    mag_raw = dict(scf_raw.pop("magnetic", {}))
    _check_keys("scf.magnetic", mag_raw,
                {"mixer", "spin_precond", "mixing_alpha", "diago_schedule"})
    diago = scf_raw.pop("diago", {})
    _check_keys("scf.diago", diago, {"tol"})

    xc, hybrid = _resolve_xc(raw)
    if task not in ("scf", "relax", "neb", "bands", "magnetism", "eos", "elastic",
                    "phonons", "flapw", "nmr"):
        raise InputError(
            f"unknown task {task!r} "
            f"(scf | relax | neb | bands | magnetism | eos | elastic | phonons | "
            f"flapw | nmr)")
    nspin = int(raw.get("nspin", 1))
    if nspin not in (1, 2):
        raise InputError(f"nspin must be 1 or 2, got {nspin}")

    noncollinear, nonmagnetic, symmetry = _resolve_symmetry(raw, task)

    # selective dynamics lowers the crystal symmetry: held atoms and their free
    # symmetry partners no longer transform into each other, so IBZ reduction and
    # force symmetrization would couple a fixed atom's force back onto a partner
    # that is meant to relax. Default symmetry off when atoms are fixed and reject
    # an explicit symmetry: true here, where the message names the fix (mirrors
    # the noncollinear handling in _resolve_symmetry).
    if fixed is not None:
        if raw.get("symmetry") is True:
            raise InputError(
                "symmetry: true is invalid with selective dynamics "
                "(structure.fixed) — fixed atoms lower the crystal symmetry, so "
                "IBZ reduction and force symmetrization would couple a held atom "
                "to its free symmetry partners. Set symmetry: false (the default "
                "when atoms are fixed).")
        symmetry = False

    # a hybrid SCF is norm-conserving and spin-unpolarized (the Fock hook builds
    # on PBE exchange, nspin=1); reject the combinations the driver cannot run
    # here, where the message points at the fix rather than deep in the SCF.
    if hybrid.enabled and (nspin != 1 or noncollinear):
        raise InputError(
            f"hybrid functionals (xc: {hybrid.name}) are spin-unpolarized "
            f"(nspin=1, collinear) only")

    # distributed (k-point-sharded) SCF is a torchrun launch; the authoritative
    # gates live in api.run_scf / api.run, but they only fire at run time inside
    # torchrun. Mirror them here so `gradwave validate` rejects an unsupported
    # combination up front, with a message that names the fix. USPP/PAW is a
    # SUPPORTED formalism (this layer cannot tell it from norm-conserving anyway),
    # so nothing keys on the pseudopotential kind.
    distributed = bool(raw.get("distributed", False))
    if distributed:
        if noncollinear:
            raise InputError(
                "distributed: true is not supported with a noncollinear/SOC SCF "
                "— the k-point-sharded SCF is wired for the norm-conserving and "
                "USPP/PAW collinear paths only (see docs/manual/distributed.md)")
        if hybrid.enabled:
            raise InputError(
                f"distributed: true is not supported with a hybrid functional "
                f"(xc: {hybrid.name}) — the k-point-sharded SCF is wired for the "
                f"collinear semilocal paths only (see docs/manual/distributed.md)")
        if task not in ("scf", "bands", "relax", "eos"):
            raise InputError(
                f"distributed: true is only wired for task: scf | bands | "
                f"relax | eos (got task: {task!r}) — elastic/phonons/magnetism "
                f"don't route through the k-point-sharded SCF path yet (see "
                f"docs/manual/distributed.md)")

    # fixed spin moment: a collinear nspin=2 pin. Without smearing it is the
    # integer-occupation fill; with smearing the SCF solves two per-channel
    # Fermi levels (the smeared FSM mode for E(M) curves).
    tot_mag = raw.get("tot_magnetization")
    if tot_mag is not None:
        tot_mag = float(tot_mag)
        if nspin != 2:
            raise InputError(
                "tot_magnetization (fixed spin moment M=N↑−N↓) applies only to "
                "a collinear spin run (nspin: 2)")

    # DFT+U: a per-species Dudarev correction inside the SCF. The collinear
    # (nspin 1/2) and noncollinear/spin-orbit norm-conserving spinor paths are
    # both wired (the latter via the 2×2 spin-block occupation matrix,
    # core.hubbard.occupation_matrices_noncollinear) — note this input layer
    # cannot distinguish NC from USPP/PAW pseudopotentials (that requires
    # parsing the UPF files, done later in api.build_system), but `api.run_scf`
    # only ever routes `noncollinear: true` to the norm-conserving driver
    # regardless of pseudopotential kind, so no combination reachable through
    # this Input/api.run path is left silently wrong; the USPP/PAW-noncollinear
    # SCF (reached only by calling scf.uspp_noncollinear.scf_uspp_noncollinear
    # directly, not through Input) rejects a hubbard argument explicitly there.
    # A hybrid's Fock SCF has no +U hook, so that combination is still rejected
    # at load.
    hubbard = _build_hubbard(raw.get("hubbard"), atoms.get_chemical_symbols())
    if hubbard.enabled and hybrid.enabled:
        raise InputError(
            "DFT+U (hubbard) and a hybrid functional (xc: pbe0/hse) cannot be "
            "combined; the hybrid Fock SCF has no +U hook")

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
    # An all-electron FLAPW/EFG run has no plane-wave ecut (its cutoff lives in
    # the `flapw` block); a placeholder keeps the Input field populated and
    # unused. Every other task requires a positive ecut.
    ecut = float(raw["ecut"]) if "ecut" in raw else (1.0 if all_electron else 0.0)
    if ecut <= 0.0:
        raise InputError(f"ecut must be > 0 eV, got {ecut}")
    ecutrho = raw.get("ecutrho")
    if ecutrho is not None:
        ecutrho = float(ecutrho)
        if ecutrho <= ecut:
            raise InputError(
                f"ecutrho must exceed ecut (the density cutoff is a finer grid "
                f"than the wavefunction cutoff), got ecutrho={ecutrho} eV, "
                f"ecut={ecut} eV")
    projections = _build_projections(raw.get("projections", False))
    dispersion = _build_dispersion(raw.get("dispersion", False))
    return Input(
        atoms=atoms,
        pseudo_dir=pseudo_dir,
        pseudo_map=pseudo_map,
        ecut=ecut,
        ecutrho=ecutrho,
        xc=xc,
        hybrid=hybrid,
        hubbard=hubbard,
        kpoints=KPointsParams(
            mesh=mesh, shift=tuple(kp.get("shift", (0, 0, 0))),
            kspacing=(None if kp.get("kspacing") is None
                      else float(kp["kspacing"])),
        ),
        smearing=SmearingParams(type=smtype, width=float(sm.get("width", 0.1))),
        nbands=None if nbands == "auto" else int(nbands),
        symmetry=symmetry,
        nspin=nspin,
        noncollinear=noncollinear,
        nonmagnetic=nonmagnetic,
        start_mag=raw.get("start_mag"),
        tot_magnetization=tot_mag,
        scf=SCFParams(
            max_iter=int(scf_raw.get("max_iter", 100)),
            etol=float(scf_raw.get("etol", 1e-7)),
            rhotol=float(scf_raw.get("rhotol", 1e-7)),
            mixing=MixingParams(**mix_raw) if mix_raw else MixingParams(),
            diago_tol=float(diago.get("tol", 1e-9)),
            trace=bool(scf_raw.get("trace", False)),
            convergence=str(scf_raw.get("convergence", "density")),
            entol=float(scf_raw.get("entol", 1e-6)),
            eigensolver=str(scf_raw.get("eigensolver", "davidson")),
            boundary=str(scf_raw.get("boundary", "periodic")),
            esm_bias=float(scf_raw.get("esm_bias", 0.0)),
            target_mu=(None if scf_raw.get("target_mu") is None
                       else float(scf_raw["target_mu"])),
            magnetic=MagneticParams(**mag_raw) if mag_raw else MagneticParams(),
        ),
        slab=_build(SlabParams, raw.get("slab", {}), "slab"),
        task=task,
        relax=_build(RelaxParams, raw.get("relax", {}), "relax"),
        neb=_build_neb(dict(raw.get("neb", {})), base, task),
        bands=_build(BandsParams, raw.get("bands", {}), "bands"),
        magnetism=_build(MagnetismParams, raw.get("magnetism", {}), "magnetism"),
        eos=_build(EOSParams, raw.get("eos", {}), "eos"),
        elastic=_build(ElasticParams, raw.get("elastic", {}), "elastic"),
        phonons=_build(PhononParams, raw.get("phonons", {}), "phonons"),
        flapw=_build_flapw(dict(raw.get("flapw", {})),
                           atoms.get_chemical_symbols()),
        nmr=_build_nmr(nmr_raw),
        projections=projections,
        dispersion=dispersion,
        device=raw.get("device", "cpu"),
        distributed=distributed,
        verbose=bool(raw.get("verbose", True)),
        output_dir=base / out_raw.get("dir", "./out"),
        output_checkpoint=bool(out_raw.get("checkpoint", True)),
        output_wavefunctions=bool(out_raw.get("wavefunctions", False)),
        output_volumetric=volumetric,
        error_estimate=bool(out_raw.get("error_estimate",
                                        raw.get("error_estimate", True))),
        restart=None if restart is None else (base / restart),
        fixed=fixed,
    )
