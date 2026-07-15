"""SCF checkpoints: save a converged state to disk, restart from it.

The file is a torch.save archive of plain CPU tensors plus metadata —
no live System object, so it loads anywhere the code runs and stays
readable across sessions. Wavefunctions are EXCLUDED by default (they
dominate the file size and the restart path only consumes the density
and becsum); pass wavefunctions=True to archive them.

Restart consumes exactly what scf_uspp(start_from=...) reads — the FFT
grid shape and volume, ρ (per spin), and the becsum — so a checkpoint
restarts any USPP/PAW run on the same grid. NC results can be saved and
analyzed but not yet warm-restarted (the NC loop has no start_from).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

FORMAT = "gradwave-checkpoint"
VERSION = 1


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
    """Write a checkpoint for an SCF result (USPP/PAW dict or NC
    SCFResult). Returns the written path."""
    import numpy as np

    from gradwave import __version__

    is_uspp = isinstance(res, dict)
    get = (res.get if is_uspp
           else lambda k, d=None: getattr(res, k, d))
    system = get("system")
    grid = system.grid
    e = get("energies")

    payload = {
        "format": FORMAT,
        "version": VERSION,
        "code_version": __version__,
        "kind": "uspp" if is_uspp else "nc",
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
        "energies_eV": {
            "kinetic": float(e.kinetic), "hartree": float(e.hartree),
            "xc": float(e.xc), "local": float(e.local),
            "nonlocal": float(e.nonlocal_), "ewald": float(e.ewald),
            "smearing": float(e.smearing), "hubbard": float(e.hubbard),
            "onecenter": float(e.onecenter),
            "total": float(e.total), "free_energy": float(e.free_energy),
        },
        "eigenvalues_eV": _cpu(get("eigenvalues")),
        "occupations": _cpu(get("occupations")),
        "rho": _cpu(get("rho")),
        "rho_spin": _cpu_tree(get("rho_spin")),
        "history": get("history", []),
    }
    if is_uspp:
        payload["rho_ij_atoms"] = _cpu_tree(res["rho_ij_atoms"])
        if "hub_occ" in res:
            payload["hub_occ"] = _cpu_tree(res["hub_occ"])
            payload["hub_sites"] = _cpu_tree(res["hub_sites"])
        for key in ("mag_total", "mag_abs"):
            if key in res:
                payload[key] = float(res[key])
    if wavefunctions:
        payload["coeffs"] = _cpu_tree(get("coeffs"))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a gradwave checkpoint")
    if payload.get("version", 0) > VERSION:
        raise ValueError(
            f"checkpoint version {payload['version']} is newer than this "
            f"code understands ({VERSION})")
    return payload


def as_start_from(payload: dict) -> dict:
    """The scf_uspp(start_from=...) view of a loaded checkpoint: a shim
    dict carrying grid shape/volume, densities and becsum. The solver
    validates grid compatibility and rescales ρ by the volume ratio."""
    if payload["kind"] != "uspp":
        raise ValueError("restart is only supported for USPP/PAW "
                         "checkpoints (the NC loop has no start_from)")
    shim_grid = SimpleNamespace(shape=tuple(payload["grid_shape"]),
                                volume=float(payload["volume_ang3"]))
    return {
        "system": SimpleNamespace(grid=shim_grid),
        "nspin": payload["nspin"],
        "rho": payload["rho"],
        "rho_spin": payload.get("rho_spin"),
        "rho_ij_atoms": payload["rho_ij_atoms"],
    }
