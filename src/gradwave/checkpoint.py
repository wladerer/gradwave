"""SCF checkpoints: save a converged state to disk, restart from it.

The file is a torch.save archive of plain CPU tensors plus metadata —
no live System object, so it loads anywhere the code runs and stays
readable across sessions. Wavefunctions are EXCLUDED by default (they
dominate the file size and the restart path only consumes the density
and becsum); pass wavefunctions=True to archive them.

Restart consumes exactly what the solvers' start_from reads — the FFT
grid shape and volume, ρ (per spin), and for USPP/PAW the becsum — so a
checkpoint restarts either formalism on the same grid.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

FORMAT = "gradwave-checkpoint"
VERSION = 1


def energies_eV_dict(e) -> dict:
    """The 11-term energy breakdown (eV) shared by the checkpoint payload and
    the api summary. build_summary adds the derived e0 on top."""
    return {
        "kinetic": float(e.kinetic), "hartree": float(e.hartree),
        "xc": float(e.xc), "local": float(e.local),
        "nonlocal": float(e.nonlocal_), "ewald": float(e.ewald),
        "smearing": float(e.smearing), "hubbard": float(e.hubbard),
        "onecenter": float(e.onecenter),
        "total": float(e.total), "free_energy": float(e.free_energy),
    }


def _cpu(t):
    return t.detach().cpu() if isinstance(t, torch.Tensor) else t


def _cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return _cpu(obj)
    if isinstance(obj, (list, tuple)):
        return [_cpu_tree(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _cpu_tree(v) for k, v in obj.items()}
    return obj


def save_checkpoint(res, path, *, wavefunctions: bool = False) -> Path:
    """Write a checkpoint for an SCF result (SCFResult, NCResult or
    USPPResult). Returns the written path."""
    import numpy as np

    from gradwave import __version__

    get = (res.get if isinstance(res, dict)
           else lambda k, d=None: getattr(res, k, d))
    # The checkpoint kind is the result's formalism tag ("nc" |
    # "noncollinear" | "uspp"). Legacy shims (plain dicts / duck-typed
    # namespaces predating the result dataclasses) carry no tag: a dict is
    # the old USPP/PAW shape, an object with the m⃗ field and the integrated
    # moment vector is an NCResult stand-in, anything else is "nc".
    kind = get("formalism")
    if kind is None:
        if isinstance(res, dict):
            kind = "uspp"
        elif get("mag_vec") is not None and get("m") is not None:
            kind = "noncollinear"
        else:
            kind = "nc"
    if kind == "uspp_noncollinear":
        raise NotImplementedError(
            "checkpointing a scf_uspp_noncollinear result is not supported "
            "(no restart path consumes its (ρ, m⃗, 4-channel becsum) state)")
    system = get("system")
    grid = system.grid
    e = get("energies")

    payload = {
        "format": FORMAT,
        "version": VERSION,
        "code_version": __version__,
        "kind": kind,
        # geometry + discretization (enough to validate a restart)
        "cell_ang": np.asarray(grid.cell, dtype=float),
        "positions_ang": _cpu(system.positions).numpy(),
        "species_of_atom": list(system.species_of_atom),
        "n_electrons": float(system.n_electrons),
        "ecut_eV": float(system.ecut),
        "grid_shape": tuple(grid.shape),
        "volume_ang3": float(grid.volume),
        "kweights": _cpu(system.kweights),
        # state
        "nspin": int(get("nspin", 1) or 1),
        "converged": bool(get("converged")),
        "n_iter": int(get("n_iter")),
        "fermi_eV": None if get("fermi") is None else float(get("fermi")),
        "smearing": get("smearing", "none"),
        "width_eV": float(get("width", 0.0) or 0.0),
        "energies_eV": energies_eV_dict(e),
        "eigenvalues_eV": _cpu(get("eigenvalues")),
        "occupations": _cpu(get("occupations")),
        "rho": _cpu(get("rho")),
        "rho_spin": _cpu_tree(get("rho_spin")),
        "history": get("history", []),
    }
    if kind == "uspp":
        payload["rho_ij_atoms"] = _cpu_tree(get("rho_ij_atoms"))
        if get("hub_occ") is not None:
            payload["hub_occ"] = _cpu_tree(get("hub_occ"))
            payload["hub_sites"] = _cpu_tree(get("hub_sites"))
        for key in ("mag_total", "mag_abs"):
            if get(key) is not None:
                payload[key] = float(get(key))
    if kind == "noncollinear":
        # the full spinor state: total density ρ, magnetization field m⃗
        # (3,*grid) and the integrated moment. Restart re-seeds the atomic
        # moments from m⃗ (see nc_mag_seed), so the field is the load-bearing
        # quantity — always archived regardless of the wavefunctions flag.
        payload["m"] = _cpu(get("m"))
        payload["mag_vec"] = [float(x) for x in get("mag_vec")]
        payload["mag_abs"] = float(get("mag_abs", 0.0) or 0.0)
    if wavefunctions:
        payload["coeffs"] = _cpu_tree(get("coeffs"))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _allow_numpy_globals() -> None:
    """Allowlist the numpy reconstruction globals so our own checkpoints load
    under weights_only=True. The payload carries a couple of numpy arrays (cell,
    positions); weights_only never executes arbitrary pickles, so this stays
    safe while covering the array reconstruct path."""
    import numpy as np
    from numpy.core.multiarray import _reconstruct

    dtype_classes = [getattr(np.dtypes, n) for n in dir(np.dtypes)
                     if n.endswith("DType")]
    torch.serialization.add_safe_globals(
        [_reconstruct, np.ndarray, np.dtype, *dtype_classes])


def load_checkpoint(path) -> dict:
    """Load a checkpoint payload: a plain dict of CPU tensors + metadata,
    with payload["kind"] the saved result's formalism tag. This is the
    archive view, not a live result object — feed it to as_start_from /
    nc_mag_seed to restart."""
    _allow_numpy_globals()
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a gradwave checkpoint")
    if payload.get("version", 0) > VERSION:
        raise ValueError(
            f"checkpoint version {payload['version']} is newer than this "
            f"code understands ({VERSION})")
    return payload


def nc_mag_seed(payload: dict, system) -> "torch.Tensor":
    """Per-atom moment seed (na, 3) [μB] for warm-starting a non-collinear
    SCF from a checkpoint.

    scf_noncollinear has no density-level warm-start hook (its only seed is
    mag_vec_init, the per-atom moment fraction·direction). We recover a
    faithful magnetic seed by decomposing the checkpoint's m⃗ field onto the
    atoms with the same Hirshfeld weights the magnetism task uses. The FFT
    grid must match, since m⃗ is stored on it."""
    if payload.get("kind") != "noncollinear":
        raise ValueError(f"checkpoint kind {payload.get('kind')!r} is not a "
                         "non-collinear SCF result")
    m = payload.get("m")
    if m is None:
        raise ValueError("non-collinear checkpoint has no m⃗ field to restart from")
    from gradwave.postscf.moment_config import atomic_weights

    grid = system.grid
    if tuple(m.shape[-3:]) != tuple(grid.shape):
        raise ValueError(
            "non-collinear restart requires the same FFT grid "
            f"({tuple(m.shape[-3:])} vs {tuple(grid.shape)})")
    m = m.to(system.positions.device)
    w = atomic_weights(system)
    cf = grid.volume / grid.n_points
    return torch.einsum("axyz,ixyz->ai", w, m) * cf  # (na, 3) [μB]


def as_start_from(payload: dict) -> dict:
    """The start_from view of a loaded checkpoint: a shim dict carrying
    grid shape/volume, densities and (USPP/PAW) the becsum. The solvers
    validate grid compatibility and rescale ρ by the volume ratio.

    Non-collinear checkpoints have no collinear start_from view — restart a
    non-collinear SCF via nc_mag_seed instead."""
    if payload.get("kind") == "noncollinear":
        raise ValueError(
            "non-collinear checkpoint cannot seed a collinear SCF; use "
            "nc_mag_seed to warm-start a non-collinear run")
    shim_grid = SimpleNamespace(shape=tuple(payload["grid_shape"]),
                                volume=float(payload["volume_ang3"]))
    out = {
        "system": SimpleNamespace(grid=shim_grid),
        "nspin": payload["nspin"],
        "rho": payload["rho"],
        "rho_spin": payload.get("rho_spin"),
    }
    if payload["kind"] == "uspp":
        out["rho_ij_atoms"] = payload["rho_ij_atoms"]
    elif payload.get("coeffs") is not None:
        out["coeffs"] = payload["coeffs"]  # NC orbital reuse when archived
    return out
