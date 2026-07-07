"""Python API mirroring the YAML input (Layer C)."""

from __future__ import annotations

import json
from pathlib import Path

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.inputs import Input
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import SCFResult, scf, setup_system

XC_REGISTRY: dict[str, type[XCFunctional]] = {"lda": LDA_PW92, "pbe": PBE}


def build_system(inp: Input):
    symbols = inp.atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    upfs = [parse_upf(inp.pseudo_dir / inp.pseudo_map[s]) for s in species]
    species_of_atom = [species.index(s) for s in symbols]
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


def run_scf(inp: Input, system=None, verbose: bool = True) -> SCFResult:
    system = system or build_system(inp)
    xc = XC_REGISTRY[inp.xc]()
    kerker = inp.scf.mixing.kerker
    return scf(
        system,
        xc,
        smearing=inp.smearing.type if inp.smearing.type != "none" else "none",
        width=inp.smearing.width,
        max_iter=inp.scf.max_iter,
        etol=inp.scf.etol,
        rhotol=inp.scf.rhotol,
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history,
        kerker=None if kerker == "auto" else bool(kerker),
        diago_tol=inp.scf.diago_tol,
        verbose=verbose,
    )


def result_summary(res: SCFResult) -> dict:
    e = res.energies
    return {
        "converged": res.converged,
        "n_iter": res.n_iter,
        "energy_eV": float(e.total),
        "free_energy_eV": float(e.free_energy),
        "e0_eV": float(e.e0),
        "fermi_eV": res.fermi,
        "terms_eV": {
            "kinetic": float(e.kinetic),
            "hartree": float(e.hartree),
            "xc": float(e.xc),
            "local": float(e.local),
            "nonlocal": float(e.nonlocal_),
            "ewald": float(e.ewald),
            "smearing": float(e.smearing),
        },
        "eigenvalues_eV": res.eigenvalues.tolist(),
        "occupations": res.occupations.tolist(),
    }


def run_relax(inp: Input, verbose: bool = True) -> dict:
    from ase.optimize import BFGS, FIRE

    from gradwave.calculator import GradWave

    atoms = inp.atoms.copy()
    atoms.calc = GradWave(
        ecut=inp.ecut,
        pseudopotentials={s: str(inp.pseudo_dir / f) for s, f in inp.pseudo_map.items()},
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
    converged = opt.run(fmax=inp.relax.fmax, steps=inp.relax.max_steps)
    return {
        "converged": bool(converged),
        "n_steps": opt.nsteps,
        "energy_eV": float(atoms.get_potential_energy()),
        "fmax_eV_ang": float(abs(atoms.get_forces()).max()),
        "positions_ang": atoms.get_positions().tolist(),
        "cell_ang": atoms.cell.array.tolist(),
    }


def run(inp: Input, verbose: bool = True) -> dict:
    if inp.task == "scf":
        res = run_scf(inp, verbose=verbose)
        summary = result_summary(res)
    elif inp.task == "relax":
        summary = run_relax(inp, verbose=verbose)
    elif inp.task == "bands":
        from gradwave.postscf.bands import bands_along_ase_path

        res = run_scf(inp, verbose=verbose)
        bs = bands_along_ase_path(
            res, inp.atoms, path=inp.bands.path, npoints=inp.bands.npoints,
            nbands=inp.bands.nbands, verbose=verbose,
        )
        summary = {
            "scf": result_summary(res),
            "kpts_frac": bs.kpts_frac.tolist(),
            "x": bs.x.tolist(),
            "labels": bs.labels,
            "eigenvalues_eV": bs.eigenvalues.tolist(),
            "reference_eV": bs.reference,
        }
    else:
        raise ValueError(inp.task)

    inp.output_dir.mkdir(parents=True, exist_ok=True)
    out = Path(inp.output_dir) / f"{inp.task}.json"
    out.write_text(json.dumps(summary, indent=1))
    if verbose:
        print(f"wrote {out}")
    return summary
