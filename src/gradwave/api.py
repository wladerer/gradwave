"""Python API mirroring the YAML input (Layer C).

run() executes a task and writes three files into the output directory:
<task>.json (the machine-readable summary — the parsing target),
<task>.out (the human-readable report) and, for SCF tasks, checkpoint.pt
(restartable state, wavefunctions excluded unless requested). The same
summary dict feeds gradwave.analysis for pandas/matplotlib work.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.inputs import Input

XC_REGISTRY: dict[str, type[XCFunctional]] = {"lda": LDA_PW92, "pbe": PBE}
_OCC_TOL = 1e-6


def _load_upf(path):
    """Parse a UPF of either family (NC via upf.py, USPP/PAW via
    upf_paw.py — same detection the ASE calculator uses)."""
    from gradwave.pseudo.upf import parse_upf

    try:
        return parse_upf(path)
    except ValueError as err:
        if "norm-conserving" not in str(err):
            raise
        from gradwave.pseudo.upf_paw import parse_upf_paw

        return parse_upf_paw(path)


def _species_upfs(inp: Input):
    symbols = inp.atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    upfs = [_load_upf(inp.pseudo_dir / inp.pseudo_map[s]) for s in species]
    species_of_atom = [species.index(s) for s in symbols]
    return species, upfs, species_of_atom


def _is_uspp(upfs) -> bool:
    from gradwave.pseudo.upf_paw import PAWData

    kinds = {isinstance(u, PAWData) for u in upfs}
    if len(kinds) > 1:
        raise ValueError("mixing NC and USPP/PAW pseudopotentials is not "
                         "supported")
    return kinds.pop()


def build_system(inp: Input):
    """The Layer-B system for this input, NC or USPP/PAW by UPF kind."""
    species, upfs, species_of_atom = _species_upfs(inp)
    if _is_uspp(upfs):
        from gradwave.scf.uspp import setup_uspp

        return setup_uspp(
            inp.atoms.cell.array, inp.atoms.get_positions(), species_of_atom,
            upfs, ecut=inp.ecut, kmesh=inp.kpoints.mesh,
            ecutrho=inp.ecutrho, nbands=inp.nbands,
            use_symmetry=inp.symmetry,
        )
    from gradwave.scf.loop import setup_system

    return setup_system(
        cell=inp.atoms.cell.array,
        positions=inp.atoms.get_positions(),
        species_of_atom=species_of_atom,
        upfs=upfs,
        ecut=inp.ecut,
        kmesh=inp.kpoints.mesh,
        kshift=inp.kpoints.shift,
        nbands=inp.nbands,
        use_symmetry=inp.symmetry,
    )


def _spin_setup(inp: Input):
    from gradwave.core.xc.spin import LSDA_PW92, SpinPBE

    xc = {"lda": LSDA_PW92, "pbe": SpinPBE}[inp.xc]()
    symbols = inp.atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    mags = [float((inp.start_mag or {}).get(s, 0.5)) for s in species]
    return xc, mags


def run_scf(inp: Input, system=None, verbose: bool = True):
    """Run the SCF for either formalism. Returns the native result
    (SCFResult for NC, dict for USPP/PAW)."""
    _species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    system = system or build_system(inp)
    if inp.device != "cpu":
        system = system.to(inp.device)
    if inp.nspin == 2:
        xc, mags = _spin_setup(inp)
    else:
        xc = XC_REGISTRY[inp.xc]()
        mags = None

    start_from = None
    if inp.restart is not None:
        from gradwave.checkpoint import as_start_from, load_checkpoint

        start_from = as_start_from(load_checkpoint(inp.restart))

    kerker = inp.scf.mixing.kerker
    kerker = None if kerker == "auto" else bool(kerker)
    common = dict(
        nspin=inp.nspin, start_mag=mags,
        smearing=inp.smearing.type, width=inp.smearing.width,
        max_iter=inp.scf.max_iter, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
        mixing_alpha=inp.scf.mixing.alpha,
        diago_tol=inp.scf.diago_tol, verbose=verbose,
    )
    if uspp:
        from gradwave.scf.uspp import scf_uspp

        # history=None keeps the per-scheme default (johnson 12, else 8)
        return scf_uspp(system, xc, mixing_scheme=inp.scf.mixing.scheme,
                        mixing_history=inp.scf.mixing.history,
                        mixing_kerker=kerker, start_from=start_from, **common)
    from gradwave.scf.loop import scf

    return scf(system, xc, kerker=kerker, start_from=start_from,
               mixing_history=inp.scf.mixing.history or 8, **common)


def _get(res, key, default=None):
    return res.get(key, default) if isinstance(res, dict) else getattr(
        res, key, default)


def _gap(eigenvalues, occupations, nspin) -> float | None:
    """HOMO-LUMO gap over all k and spins, None when any occupation is
    fractional (metals/smeared systems have no meaningful scalar gap)."""
    import numpy as np

    e = np.asarray(eigenvalues, dtype=float).reshape(-1)
    f = np.asarray(occupations, dtype=float).reshape(-1)
    f_full = 2.0 / nspin
    frac = (f > _OCC_TOL) & (np.abs(f - f_full) > _OCC_TOL)
    if frac.any() or not (f > _OCC_TOL).any() or not (f <= _OCC_TOL).any():
        return None
    homo = e[f > _OCC_TOL].max()
    lumo = e[f <= _OCC_TOL].min()
    return float(lumo - homo) if lumo > homo else 0.0


def build_summary(res, inp: Input, task: str, runtime_s: float | None = None,
                  extra: dict | None = None) -> dict:
    """The unified machine-readable summary for a task run."""
    from gradwave import __version__

    system = _get(res, "system")
    e = _get(res, "energies")
    nspin = int(_get(res, "nspin", 1) or 1)
    eig = _get(res, "eigenvalues")
    occ = _get(res, "occupations")
    species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)

    import math

    def _finite(x):
        # the first iteration records dE = inf; bare Infinity is not
        # valid strict JSON, so non-finite maps to null
        return None if x is None or not math.isfinite(x) else float(x)

    trace = [
        {"iter": h["iter"], "free_energy_eV": float(h["free_energy"]),
         "dE_eV": _finite(h["dE"]), "drho": float(h["res"])}
        for h in (_get(res, "history") or [])
    ]
    scf_block = {
        "converged": bool(_get(res, "converged")),
        "n_iter": int(_get(res, "n_iter")),
        "fermi_eV": None if _get(res, "fermi") is None
        else float(_get(res, "fermi")),
        "gap_eV": _gap(eig.tolist(), occ.tolist(), nspin),
        "energies_eV": {
            "kinetic": float(e.kinetic), "hartree": float(e.hartree),
            "xc": float(e.xc), "local": float(e.local),
            "nonlocal": float(e.nonlocal_), "ewald": float(e.ewald),
            "smearing": float(e.smearing), "hubbard": float(e.hubbard),
            "onecenter": float(e.onecenter), "total": float(e.total),
            "free_energy": float(e.free_energy),
            "e0": float(0.5 * (e.total + e.free_energy)),
        },
        "trace": trace,
    }
    if nspin == 2:
        scf_block["total_magnetization_muB"] = float(_get(res, "mag_total", 0.0))
        scf_block["absolute_magnetization_muB"] = float(_get(res, "mag_abs", 0.0))

    summary = {
        "code": {"name": "gradwave", "version": __version__,
                 "created": datetime.datetime.now().isoformat(timespec="seconds")},
        "task": task,
        "structure": _structure_block(inp),
        "parameters": {
            "formalism": "uspp/paw" if uspp else "nc",
            "xc": inp.xc,
            "ecut_eV": float(inp.ecut),
            "ecutrho_eV": float(inp.ecutrho) if (uspp and inp.ecutrho) else None,
            "kmesh": list(inp.kpoints.mesh),
            "nk": len(system.kweights),
            "kweights": [float(w) for w in system.kweights],
            "nspin": nspin,
            "smearing": inp.smearing.type,
            "width_eV": float(inp.smearing.width),
            "symmetry": bool(inp.symmetry),
            "n_electrons": float(system.n_electrons),
            "nbands": int(system.nbands),
            "fft_grid": list(system.grid.shape),
            "npw": int(system.spheres[0].npw),
            "pseudos": {s: inp.pseudo_map[s] for s in species},
        },
        "scf": scf_block,
        "eigenvalues_eV": eig.tolist(),
        "occupations": occ.tolist(),
    }
    if runtime_s is not None:
        summary["runtime_s"] = round(float(runtime_s), 2)
    if extra:
        summary.update(extra)
    return summary


def result_summary(res) -> dict:
    """Backward-compatible flat summary of a bare SCF result (no Input
    context — prefer build_summary for file output)."""
    e = _get(res, "energies")
    return {
        "converged": bool(_get(res, "converged")),
        "n_iter": int(_get(res, "n_iter")),
        "energy_eV": float(e.total),
        "free_energy_eV": float(e.free_energy),
        "e0_eV": float(0.5 * (e.total + e.free_energy)),
        "fermi_eV": None if _get(res, "fermi") is None
        else float(_get(res, "fermi")),
        "nspin": int(_get(res, "nspin", 1) or 1),
        "total_magnetization_muB": float(_get(res, "mag_total", 0.0) or 0.0),
        "absolute_magnetization_muB": float(_get(res, "mag_abs", 0.0) or 0.0),
        "terms_eV": {
            "kinetic": float(e.kinetic), "hartree": float(e.hartree),
            "xc": float(e.xc), "local": float(e.local),
            "nonlocal": float(e.nonlocal_), "ewald": float(e.ewald),
            "smearing": float(e.smearing),
        },
        "eigenvalues_eV": _get(res, "eigenvalues").tolist(),
        "occupations": _get(res, "occupations").tolist(),
    }


def run_relax(inp: Input, verbose: bool = True) -> tuple[dict, object]:
    """Relax with ASE; returns (relax block, final atoms)."""
    from ase.optimize import BFGS, FIRE

    from gradwave.calculator import GradWave

    atoms = inp.atoms.copy()
    atoms.calc = GradWave(
        ecut=inp.ecut,
        pseudopotentials={s: str(inp.pseudo_dir / f)
                          for s, f in inp.pseudo_map.items()},
        xc=inp.xc,
        kpts=inp.kpoints.mesh,
        kshift=inp.kpoints.shift,
        smearing=inp.smearing.type,
        width=inp.smearing.width,
        nbands=inp.nbands,
        etol=inp.scf.etol,
        rhotol=inp.scf.rhotol,
        verbose=False,
    )
    opt_cls = {"fire": FIRE, "bfgs": BFGS}[inp.relax.optimizer]
    opt = opt_cls(atoms, logfile="-" if verbose else None)
    trajectory = []

    def _record():
        import numpy as np

        forces = atoms.get_forces()
        trajectory.append({
            "step": opt.nsteps,
            "energy_eV": float(atoms.get_potential_energy()),
            "fmax_eV_ang": float(np.linalg.norm(forces, axis=1).max()),
            "positions_ang": atoms.get_positions().tolist(),
        })

    opt.attach(_record)
    converged = opt.run(fmax=inp.relax.fmax, steps=inp.relax.max_steps)
    import numpy as np

    relax = {
        "converged": bool(converged),
        "n_steps": opt.nsteps,
        "optimizer": inp.relax.optimizer,
        "fmax_target_eV_ang": inp.relax.fmax,
        "energy_eV": float(atoms.get_potential_energy()),
        "fmax_eV_ang": float(
            np.linalg.norm(atoms.get_forces(), axis=1).max()),
        "max_displacement_ang": float(np.linalg.norm(
            atoms.get_positions() - inp.atoms.get_positions(),
            axis=1).max()),
        "species": atoms.get_chemical_symbols(),
        "positions_ang": atoms.get_positions().tolist(),
        "cell_ang": atoms.cell.array.tolist(),
        "trajectory": trajectory,
    }
    return relax, atoms


def _bands_extra(inp: Input, res, verbose: bool) -> dict:
    from gradwave.postscf.bands import bands_along_ase_path

    bs = bands_along_ase_path(
        res, inp.atoms, path=inp.bands.path, npoints=inp.bands.npoints,
        nbands=inp.bands.nbands, verbose=verbose,
    )
    bands = {
        "kpts_frac": bs.kpts_frac.tolist(),
        "x": bs.x.tolist(),
        "labels": bs.labels,
        "eigenvalues_eV": bs.eigenvalues.tolist(),
        "reference_eV": bs.reference,
    }
    if inp.bands.irreps:
        import numpy as np

        from gradwave.postscf.irreps import band_irreps

        cache, ann = {}, []
        for xt, lab in bs.labels:
            idx = int(np.argmin(np.abs(np.asarray(bs.x) - xt)))
            kf_exact = bs.kpts_frac[idx]  # full precision — rounding here
            # shrinks the little group at threshold (1/3 vs 0.33333333)
            key = tuple(np.round(kf_exact, 8))
            if key not in cache:
                cache[key] = band_irreps(res, kf_exact, nbands=inp.bands.nbands)
            ann.append({
                "x": float(xt), "name": lab,
                "clusters": [
                    {"e": float(np.mean(c.energies)), "label": c.label,
                     "dim": c.dim, "warning": c.warning}
                    for c in cache[key].clusters
                ],
            })
        bands["irreps"] = ann
    return {"bands": bands}


def _error_estimate_block(res, inp) -> dict:
    """Post-SCF plane-wave (Ecut) discretization-error estimate for the output.

    Cheap post-processing (no larger SCF): the first-order complement correction
    of Cancès et al. gives the estimated basis-set error in the energy (a
    definite lowering), the density, and, for norm-conserving nspin=1, the
    Hellmann-Feynman forces. Reported as an indicator, not a rigorous bound.
    Degrades gracefully when the run's formalism/settings are outside coverage.
    """
    from gradwave.postscf.discretization_error import (
        estimate_density_error,
        estimate_force_error,
    )

    _species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    xc = _spin_setup(inp)[0] if inp.nspin == 2 else XC_REGISTRY[inp.xc]()
    system = _get(res, "system")
    nspin = int(_get(res, "nspin", 1) or 1)
    natom = len(system.positions)
    grid = system.grid
    vol, npts = grid.volume, grid.n_points
    nelec = float(system.n_electrons)
    try:
        err = estimate_density_error(res, xc=xc)
    except NotImplementedError as e:
        return {"available": False, "reason": str(e)}

    drho = err.drho
    free_e = float(_get(res, "energies").free_energy)
    block = {
        "available": True,
        "method": "Cances first-order complement (post-SCF)",
        "ecut_eV": err.ecut,
        "ecut_large_eV": err.ecut_large,
        "denergy_eV": float(err.denergy),
        "denergy_meV_per_atom": float(err.denergy) / natom * 1e3,
        "free_energy_extrapolated_eV": free_e + float(err.denergy),
        "drho_L1_per_electron": float(drho.abs().sum()) * vol / npts / nelec,
        "int_drho": float(drho.sum()) * vol / npts,
        "note": "first-order estimate, indicative not a rigorous bound",
    }
    force_ok = (not uspp and nspin == 1
                and getattr(system, "rho_core", None) is None)
    if force_ok:
        try:
            fe = estimate_force_error(res, err).norm(dim=1)
            block["force_error_max_eV_ang"] = float(fe.max())
            block["force_error_rms_eV_ang"] = float((fe ** 2).mean().sqrt())
        except NotImplementedError:
            pass
    return block


def run(inp: Input, verbose: bool = True) -> dict:
    """Execute inp.task and write <task>.json, <task>.out and (for SCF
    state) checkpoint.pt into inp.output_dir."""
    from gradwave.output import write_output

    t0 = time.time()
    res = None
    if inp.task == "scf":
        res = run_scf(inp, verbose=verbose)
        summary = build_summary(res, inp, "scf", runtime_s=time.time() - t0)
        if inp.error_estimate:
            summary["error_estimate"] = _error_estimate_block(res, inp)
    elif inp.task == "relax":
        relax, _atoms = run_relax(inp, verbose=verbose)
        from gradwave import __version__

        summary = {
            "code": {"name": "gradwave", "version": __version__,
                     "created": datetime.datetime.now().isoformat(
                         timespec="seconds")},
            "task": "relax",
            "structure": _structure_block(inp),
            "parameters": _relax_parameters(inp),
            "relax": relax,
            "runtime_s": round(time.time() - t0, 2),
        }
    elif inp.task == "bands":
        res = run_scf(inp, verbose=verbose)
        summary = build_summary(res, inp, "bands",
                                extra=_bands_extra(inp, res, verbose),
                                runtime_s=time.time() - t0)
    else:
        raise ValueError(inp.task)

    outdir = Path(inp.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if res is not None and inp.output_checkpoint:
        from gradwave.checkpoint import save_checkpoint

        ck = save_checkpoint(res, outdir / "checkpoint.pt",
                             wavefunctions=inp.output_wavefunctions)
        outputs["checkpoint"] = ck.name
    summary["outputs"] = {**outputs, "json": f"{inp.task}.json",
                          "report": f"{inp.task}.out"}
    (outdir / f"{inp.task}.json").write_text(json.dumps(summary, indent=1))
    write_output(summary, outdir / f"{inp.task}.out")
    if verbose:
        print(f"wrote {outdir / inp.task}.json / .out"
              + (" / checkpoint.pt" if "checkpoint" in outputs else ""))
    return summary


def _structure_block(inp: Input) -> dict:
    import numpy as np

    block = {
        "cell_ang": inp.atoms.cell.array.tolist(),
        "positions_ang": inp.atoms.get_positions().tolist(),
        "species": inp.atoms.get_chemical_symbols(),
        "volume_ang3": float(abs(np.linalg.det(inp.atoms.cell.array))),
    }
    try:
        import spglib

        ds = spglib.get_symmetry_dataset(
            (inp.atoms.cell.array, inp.atoms.get_scaled_positions(),
             inp.atoms.get_atomic_numbers()), symprec=1e-5)
        block["spacegroup"] = f"{ds.international} ({ds.number})"
    except Exception:
        pass
    return block


def _relax_parameters(inp: Input) -> dict:
    species, upfs, _soa = _species_upfs(inp)
    return {
        "formalism": "uspp/paw" if _is_uspp(upfs) else "nc",
        "xc": inp.xc,
        "ecut_eV": float(inp.ecut),
        "ecutrho_eV": None,
        "kmesh": list(inp.kpoints.mesh),
        "nk": None,
        "kweights": None,
        "nspin": inp.nspin,
        "smearing": inp.smearing.type,
        "width_eV": float(inp.smearing.width),
        "symmetry": bool(inp.symmetry),
        "pseudos": {s: inp.pseudo_map[s] for s in species},
    }
