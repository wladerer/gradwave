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
    if inp.noncollinear:
        return _run_scf_noncollinear(inp, system, verbose)
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


def _run_scf_noncollinear(inp: Input, system, verbose: bool):
    """A plain non-collinear (spinor) SCF for task: scf with
    noncollinear: true. Builds a NoncollinearXC from inp.xc (as run_magnetism
    does), seeds the atomic moments along +z from start_mag (or warm-starts
    from a checkpoint's m⃗ field), and returns the NCResult."""
    import torch

    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
    from gradwave.scf.noncollinear import scf_noncollinear

    xc = NoncollinearXC({"lda": LSDA_PW92, "pbe": SpinPBE}[inp.xc]())

    if inp.restart is not None:
        from gradwave.checkpoint import load_checkpoint, nc_mag_seed

        mag_vec_init = nc_mag_seed(load_checkpoint(inp.restart), system)
    else:
        # high-spin seed along +z; per-species magnitude from start_mag (a
        # moment fraction ~ scale on the SAD magnetization), default 1.0
        symbols = inp.atoms.get_chemical_symbols()
        z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        scales = [float((inp.start_mag or {}).get(s, 1.0)) for s in symbols]
        mag_vec_init = torch.stack([s * z for s in scales])  # (na, 3)

    # NC SCF requires a real smearing scheme (spinor bands hold one electron)
    smtype = inp.smearing.type if inp.smearing.type != "none" else "gaussian"
    return scf_noncollinear(
        system, xc, mag_vec_init=mag_vec_init,
        smearing=smtype, width=inp.smearing.width,
        max_iter=inp.scf.max_iter, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history or 8,
        diago_tol=inp.scf.diago_tol, verbose=verbose,
    )


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
    # a non-collinear NCResult carries an integrated moment vector but no
    # occupations (spinor bands each hold one electron); the gap/occupations
    # blocks degrade gracefully below.
    mag_vec = _get(res, "mag_vec")
    is_ncmag = mag_vec is not None and not isinstance(res, dict)

    import math

    def _finite(x):
        # the first iteration records dE = inf; bare Infinity is not
        # valid strict JSON, so non-finite maps to null
        return None if x is None or not math.isfinite(x) else float(x)

    trace = [
        {"iter": h["iter"], "free_energy_eV": float(h["free_energy"]),
         "dE_eV": _finite(h["dE"]), "drho": float(h["res"]),
         **({"t_s": round(float(h["t"]), 3)} if "t" in h else {})}
        for h in (_get(res, "history") or [])
    ]
    scf_block = {
        "converged": bool(_get(res, "converged")),
        "n_iter": int(_get(res, "n_iter")),
        "fermi_eV": None if _get(res, "fermi") is None
        else float(_get(res, "fermi")),
        "gap_eV": None if occ is None else _gap(eig.tolist(), occ.tolist(), nspin),
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
    if is_ncmag:
        mv = [float(x) for x in mag_vec]
        scf_block["magnetization_vector_muB"] = mv
        scf_block["total_magnetization_muB"] = float((sum(x * x for x in mv)) ** 0.5)
        scf_block["absolute_magnetization_muB"] = float(_get(res, "mag_abs", 0.0))

    summary = {
        "code": {"name": "gradwave", "version": __version__,
                 "created": datetime.datetime.now().isoformat(timespec="seconds")},
        "task": task,
        "structure": _structure_block(inp),
        "parameters": {
            "formalism": "noncollinear" if is_ncmag else (
                "uspp/paw" if uspp else "nc"),
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
        "occupations": [] if occ is None else occ.tolist(),
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


def run_relax(inp: Input, verbose: bool = True) -> tuple[dict, object, list]:
    """Relax with ASE, returning (relax block, final atoms, per-step ASE frames).

    The frames carry energy and forces (SinglePointCalculator) so the caller can
    write an extxyz trajectory next to the JSON."""
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
    target = atoms
    if inp.relax.cell:
        from ase.filters import FrechetCellFilter

        # ASE stress/pressure is eV/Å³; the user knob is GPa. The filter adds
        # the cell degrees of freedom, so opt.run(fmax) then gates BOTH the
        # atomic forces and the stress (external pressure subtracted).
        gpa_to_ev_a3 = 1.0 / 160.21766208
        target = FrechetCellFilter(
            atoms, scalar_pressure=inp.relax.pressure * gpa_to_ev_a3)
    opt = opt_cls(target, logfile="-" if verbose else None)
    trajectory = []
    frames = []  # ASE Atoms per step, energy+forces frozen for extxyz output

    def _record():
        import numpy as np
        from ase.calculators.singlepoint import SinglePointCalculator

        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        trajectory.append({
            "step": opt.nsteps,
            "energy_eV": energy,
            "fmax_eV_ang": float(np.linalg.norm(forces, axis=1).max()),
            "positions_ang": atoms.get_positions().tolist(),
            "cell_ang": atoms.cell.array.tolist(),
        })
        frame = atoms.copy()
        sp_kw = {"energy": energy, "forces": forces}
        if inp.relax.cell:
            sp_kw["stress"] = atoms.get_stress()
        frame.calc = SinglePointCalculator(frame, **sp_kw)
        frame.info["step"] = opt.nsteps
        frames.append(frame)

    opt.attach(_record)
    converged = opt.run(fmax=inp.relax.fmax, steps=inp.relax.max_steps)
    import numpy as np

    relax = {
        "converged": bool(converged),
        "n_steps": opt.nsteps,
        "optimizer": inp.relax.optimizer,
        "cell_relaxed": bool(inp.relax.cell),
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
        "volume_ang3": float(atoms.get_volume()),
        "trajectory": trajectory,
    }
    if inp.relax.cell:
        relax["max_stress_eV_ang3"] = float(np.abs(atoms.get_stress()).max())
        relax["pressure_GPa"] = inp.relax.pressure
    return relax, atoms, frames


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


def _error_estimate_xc(inp):
    """The functional object the post-SCF estimators need to rebuild operators.

    Non-collinear runs need a ``NoncollinearXC`` (the exchange field enters the
    spinor Hamiltonian); collinear nspin=2 needs the spin functional; nspin=1 the
    plain one.
    """
    if inp.noncollinear:
        from gradwave.core.xc.noncollinear import NoncollinearXC
        from gradwave.core.xc.spin import LSDA_PW92, SpinPBE

        return NoncollinearXC({"lda": LSDA_PW92, "pbe": SpinPBE}[inp.xc]())
    return _spin_setup(inp)[0] if inp.nspin == 2 else XC_REGISTRY[inp.xc]()


def _error_estimate_block(res, inp) -> dict:
    """Post-SCF plane-wave (Ecut) discretization-error estimate for the output.

    Cheap post-processing (no larger SCF): the first-order complement correction
    of Cancès et al. gives the estimated basis-set error in the energy (a
    definite lowering), the density, the Kohn-Sham eigenvalues / band gap, and --
    for norm-conserving collinear runs -- the Hellmann-Feynman forces. Covers
    norm-conserving and USPP/PAW (nspin=1, 2) and the non-collinear/SOC spinor
    formalism. Reported as an indicator, not a rigorous bound. Degrades
    gracefully when the run's formalism/settings are outside coverage.
    """
    from gradwave.postscf.discretization_error import (
        estimate_density_error,
        estimate_eigenvalue_error,
        estimate_force_error,
        estimate_gap_error,
    )

    _species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    is_nc = bool(inp.noncollinear)
    xc = _error_estimate_xc(inp)
    system = _get(res, "system")
    nspin = int(_get(res, "nspin", 1) or 1)
    natom = len(system.positions)
    grid = system.grid
    vol, npts = grid.volume, grid.n_points
    nelec = float(system.n_electrons)
    # a non-collinear SCF always runs with a real smearing scheme (spinor bands
    # hold one electron); a "none" request maps to gaussian, as the run does.
    nc_scheme = ("gaussian" if inp.smearing.type == "none" else inp.smearing.type)
    dens_kw = dict(smearing=nc_scheme, width=inp.smearing.width) if is_nc else {}
    try:
        err = estimate_density_error(res, xc=xc, **dens_kw)
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
    # force error: norm-conserving collinear only (USPP augmentation/one-center
    # and the spinor force terms in P(eps) are not assembled).
    force_ok = (not uspp and not is_nc and nspin in (1, 2)
                and getattr(system, "rho_core", None) is None)
    if force_ok:
        try:
            fe = estimate_force_error(res, err).norm(dim=1)
            block["force_error_max_eV_ang"] = float(fe.max())
            block["force_error_rms_eV_ang"] = float((fe ** 2).mean().sqrt())
        except NotImplementedError:
            pass
    # band-gap error (insulators; NC/USPP/PAW now covered, skipped for metals).
    try:
        eig_kw = dict(smearing=nc_scheme, width=inp.smearing.width) if is_nc else {}
        eige = estimate_eigenvalue_error(res, ecut_large=err.ecut_large, xc=xc,
                                         **eig_kw)
        gap_kw = {}
        if is_nc:
            # NCResult carries no occupations; recompute (degeneracy 1) and set
            # the metal/insulator threshold to half of one-electron filling.
            gap_kw = dict(occupations=_nc_occupations(res, nc_scheme,
                                                      inp.smearing.width),
                          occ_threshold=0.5)
        gap = estimate_gap_error(res, eige, **gap_kw)
        block["gap_eV"] = gap["gap_eV"]
        block["gap_extrapolated_eV"] = gap["gap_extrapolated_eV"]
        block["dgap_eV"] = gap["dgap_eV"]
    except (NotImplementedError, ValueError):
        pass
    # other numerical convergence errors (SCF self-consistency, smearing). These
    # are separate axes from the basis-set error; k-point sampling needs a mesh
    # sweep (estimate_kpoint_error) and is not reachable from one run.
    from gradwave.postscf.convergence_error import (
        estimate_scf_error,
        estimate_smearing_error,
    )
    # SCF self-consistency error uses the collinear response kernel (K_Hxc/chi0);
    # USPP/PAW and the spinor formalism have no such primitive exposed yet.
    if not uspp and not is_nc:
        try:
            scfe = estimate_scf_error(res, xc)
            block["scf_convergence"] = {
                "denergy_eV": scfe.denergy,
                "denergy_meV_per_atom": scfe.denergy / natom * 1e3,
                "residual_L1_per_electron": scfe.residual_norm,
                "screened": scfe.screened,
                "energy_converged_estimate_eV": scfe.energy_converged_estimate,
            }
        except (NotImplementedError, ValueError):
            pass
    try:
        sme = estimate_smearing_error(
            res, scheme=nc_scheme if is_nc else inp.smearing.type,
            width=inp.smearing.width)
        block["smearing"] = {
            "scheme": sme.scheme,
            "dsmearing_eV": sme.dsmearing,
            "energy_extrapolated_eV": sme.energy_extrapolated,
            "residual_bound_eV": sme.half_width,
            "note": sme.note,
        }
    except (NotImplementedError, ValueError):
        pass
    return block


def _nc_occupations(res, scheme: str, width: float):
    """Per-k occupations of a spinor (NCResult) run, recomputed for the gap tool.

    NCResult stores neither the occupations nor the smearing width, so rebuild
    them from the stored eigenvalues at degeneracy 1.0 (one electron per spinor
    band), the same recipe the SCF uses.
    """
    from gradwave.core.occupations import (
        SCHEMES,
        find_fermi,
        occupations_and_entropy,
    )

    system = res.system
    eps = res.eigenvalues
    mu = find_fermi(eps, system.kweights, SCHEMES[scheme], width,
                    system.n_electrons, degeneracy=1.0)
    occ, _ = occupations_and_entropy(eps, mu, SCHEMES[scheme], width,
                                     degeneracy=1.0)
    return [occ[ik] for ik in range(eps.shape[0])]


def _pdos_summary_block(res, inp: Input) -> dict:
    """Löwdin projected-DOS block for the summary JSON. Returns a graceful
    ``{'available': False, ...}`` when the pseudopotentials omit PP_PSWFC."""
    from gradwave.postscf.pdos import projected_dos
    p = inp.projections
    try:
        block = projected_dos(res, group_by=p.group_by, width=p.width,
                              npoints=p.npoints).to_dict()
        return block
    except (ValueError, NotImplementedError) as err:
        return {"available": False, "reason": str(err)}


def run_magnetism(inp: Input, verbose: bool = True):
    """Characterize the magnetism of the input system (task: magnetism). Builds a
    non-collinear XC from inp.xc, runs `characterize_magnetism`, and returns the
    MagneticReport."""
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
    from gradwave.postscf.magnetism import characterize_magnetism

    system = build_system(inp)
    if inp.device != "cpu":
        system = system.to(inp.device)
    xc = NoncollinearXC({"lda": LSDA_PW92, "pbe": SpinPBE}[inp.xc]())
    m = inp.magnetism
    smtype = inp.smearing.type if inp.smearing.type != "none" else "gaussian"
    return characterize_magnetism(
        system, xc, exchange=m.exchange, ref_atom=m.ref_atom, lam=m.lam,
        delta=m.delta, seed_scale=m.seed_scale, smearing=smtype,
        width=inp.smearing.width, max_iter=inp.scf.max_iter, etol=inp.scf.etol,
        rhotol=inp.scf.rhotol, mixing_alpha=inp.scf.mixing.alpha, verbose=verbose)


def run(inp: Input, verbose: bool = True) -> dict:
    """Execute inp.task and write <task>.json, <task>.out and (for SCF
    state) checkpoint.pt into inp.output_dir."""
    from gradwave.output import write_output
    from gradwave.runinfo import ProcessMeter, machine_snapshot, provenance_block

    snap = machine_snapshot()
    meter = ProcessMeter()
    t0 = time.time()
    res = None
    _frames = None
    if inp.task == "scf":
        res = run_scf(inp, verbose=verbose)
        summary = build_summary(res, inp, "scf", runtime_s=time.time() - t0)
        if inp.error_estimate:
            summary["error_estimate"] = _error_estimate_block(res, inp)
        if inp.projections.enabled:
            summary["pdos"] = _pdos_summary_block(res, inp)
    elif inp.task == "relax":
        relax, _atoms, _frames = run_relax(inp, verbose=verbose)
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
        # error estimate at the FINAL geometry: the calculator caches its
        # last converged SCF, so the estimate describes the relaxed state
        res_final = getattr(_atoms.calc, "last_result", None)
        if inp.error_estimate and res_final is not None:
            summary["error_estimate"] = _error_estimate_block(res_final, inp)
    elif inp.task == "bands":
        res = run_scf(inp, verbose=verbose)
        summary = build_summary(res, inp, "bands",
                                extra=_bands_extra(inp, res, verbose),
                                runtime_s=time.time() - t0)
        if inp.error_estimate:
            summary["error_estimate"] = _error_estimate_block(res, inp)
    elif inp.task == "magnetism":
        # no error_estimate block here: magnetism runs are spinor SCFs,
        # outside every estimator's coverage (it would always be
        # available: false)
        report = run_magnetism(inp, verbose=verbose)
        if verbose:
            print(report.summary())
        from gradwave import __version__

        summary = {
            "code": {"name": "gradwave", "version": __version__,
                     "created": datetime.datetime.now().isoformat(
                         timespec="seconds")},
            "task": "magnetism",
            "structure": _structure_block(inp),
            "parameters": _relax_parameters(inp),
            "magnetism": {
                "ordering": report.ordering,
                "total_moment_muB": round(report.total_moment, 4),
                "atomic_moments_muB": report.moment_magnitudes,
                "moment_vectors_muB": report.moment_vectors,
                "exchange_J_meV": None if report.exchange_J is None else
                {str(i): round(J * 1000, 3) for i, J in report.exchange_J.items()},
                "dmi_meV": None if report.dmi is None else
                {str(i): round(d * 1000, 4) for i, d in report.dmi.items()},
                "curie_temperature_mfa_K": None if report.curie_temperature_mfa is None
                else round(report.curie_temperature_mfa),
            },
            "runtime_s": round(time.time() - t0, 2),
        }
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
    if inp.task == "relax" and _frames:
        from ase.io import write as ase_write

        ase_write(str(outdir / "relax.xyz"), _frames, format="extxyz")
        outputs["trajectory"] = "relax.xyz"
    summary["provenance"] = provenance_block(snap, meter)
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
