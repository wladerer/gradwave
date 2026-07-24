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
import logging
import time
from pathlib import Path

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.inputs import Input

XC_REGISTRY: dict[str, type[XCFunctional]] = {"lda": LDA_PW92, "pbe": PBE,
                                              "r2scan": R2SCAN}
SPIN_XC_REGISTRY: dict[str, type] = {"lda": LSDA_PW92, "pbe": SpinPBE,
                                     "r2scan": SpinR2SCAN}
_OCC_TOL = 1e-6
# the collinear/NC solvers build a fixed-length Pulay history and need an int
# (None is not accepted, unlike the USPP path); this names the default the api
# forwards, matching the per-scheme default those solvers use internally
_DEFAULT_MIXING_HISTORY = 8
# UPFs are static for a run; cache by path so build_summary / run_scf /
# _error_estimate_block / _parameters_block parse each pseudo once, not 3-4×
_UPF_CACHE: dict[str, object] = {}

logger = logging.getLogger(__name__)


def _load_upf(path):
    """Parse a UPF of either family (NC via upf.py, USPP/PAW via
    upf_paw.py — same detection the ASE calculator uses), cached by path."""
    key = str(path)
    cached = _UPF_CACHE.get(key)
    if cached is not None:
        return cached
    from gradwave.pseudo.upf import parse_upf

    try:
        upf = parse_upf(path)
    except ValueError as err:
        if "norm-conserving" not in str(err):
            raise
        from gradwave.pseudo.upf_paw import parse_upf_paw

        upf = parse_upf_paw(path)
    if logger.isEnabledFor(logging.DEBUG):
        # projection orbitals (NC: pswfc, US/PAW: chi) gate COHP/PDOS analysis —
        # SG15 ONCV fixtures ship none, PseudoDojo/PAW do
        orbitals = getattr(upf, "pswfc", None)
        if orbitals is None:
            orbitals = getattr(upf, "chi", [])
        logger.debug(
            "loaded pseudo %s: %s element=%s, n_proj=%s, projection_orbitals=%d, "
            "core_correction=%s", key, type(upf).__name__,
            getattr(upf, "element", "?"), getattr(upf, "n_proj", "?"),
            len(orbitals), getattr(upf, "core_correction", "?"))
    _UPF_CACHE[key] = upf
    return upf


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
        # a magnetic spinor breaks k ≡ −k (TR flips m⃗); a nonmagnetic spinor
        # (SOC only) keeps Kramers, so TR reduction stays valid there
        time_reversal=not (inp.noncollinear and not inp.nonmagnetic),
    )


def _spin_setup(inp: Input):
    xc = SPIN_XC_REGISTRY[inp.xc]()
    symbols = inp.atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    mags = [float((inp.start_mag or {}).get(s, 0.5)) for s in species]
    return xc, mags


def run_scf(inp: Input, system=None, verbose: bool = True, start_from=None):
    """Run the SCF for either formalism. Returns the native result
    (SCFResult for NC, USPPResult for USPP/PAW).

    ``start_from`` warm-starts the density from a previous converged result
    (the volume-scan chain in ``run_eos`` uses it); when None the checkpoint in
    ``inp.restart`` is used instead, if any."""
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

    if start_from is None and inp.restart is not None:
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
               mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
               **common)


def _run_scf_noncollinear(inp: Input, system, verbose: bool):
    """A plain non-collinear (spinor) SCF for task: scf with
    noncollinear: true. Builds a NoncollinearXC from inp.xc (as run_magnetism
    does), seeds the atomic moments along +z from start_mag (or warm-starts
    from a checkpoint's m⃗ field), and returns the NCResult."""
    import torch

    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.scf.noncollinear import scf_noncollinear

    xc = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())

    if inp.nonmagnetic:
        # spin-orbit only: pin m⃗ ≡ 0 (no seed, no spurious moment)
        mag_vec_init = torch.zeros((len(inp.atoms), 3), dtype=torch.float64)
    elif inp.restart is not None:
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
        mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
        diago_tol=inp.scf.diago_tol, verbose=verbose,
        nonmagnetic=inp.nonmagnetic,
    )


def _get(res, key, default=None):
    """Attribute read with a default: every SCF driver returns a result
    dataclass, but the field sets differ (e.g. NCResult has no nspin)."""
    return getattr(res, key, default)


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
    from gradwave.checkpoint import energies_eV_dict

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
    is_ncmag = _get(res, "formalism") == "noncollinear"

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
            **energies_eV_dict(e),
            "e0": float(0.5 * (e.total + e.free_energy)),
        },
        "free_energy_per_atom_eV": float(e.free_energy) / len(system.positions),
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

    # convergence diagnostics: final residuals against the thresholds, the
    # geometric decay rate q of the energy residual (small q = fast, clean
    # convergence), and whether the run warm-started from a checkpoint
    _des = [abs(h["dE_eV"]) for h in trace
            if h.get("dE_eV") is not None and h["dE_eV"] != 0.0]
    _ratios = [_des[i] / _des[i - 1] for i in range(1, len(_des)) if _des[i - 1] > 0]
    q = None
    if _ratios:
        _tail = sorted(_ratios[-4:])
        q = float(_tail[len(_tail) // 2])  # median of the last few ratios
    _final = trace[-1] if trace else {}
    scf_block["convergence"] = {
        "final_dE_eV": _final.get("dE_eV"),
        "final_drho": _final.get("drho"),
        "etol_eV": float(inp.scf.etol),
        "rhotol": float(inp.scf.rhotol),
        "ratio_q": q,
        "warm_started": inp.restart is not None,
    }

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
            "nk_total": int(math.prod(inp.kpoints.mesh)),
            "kweights": [float(w) for w in system.kweights],
            "nspin": nspin,
            "smearing": inp.smearing.type,
            "width_eV": float(inp.smearing.width),
            "symmetry": bool(inp.symmetry),
            "mixing": {
                "scheme": inp.scf.mixing.scheme,
                "alpha": float(inp.scf.mixing.alpha),
                "history": inp.scf.mixing.history,
                "kerker": inp.scf.mixing.kerker,
                "kerker_used": _get(res, "kerker_used"),
            },
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


def run_relax(inp: Input, verbose: bool = True) -> tuple[dict, object, list]:
    """Relax with ASE, returning (relax block, final atoms, per-step ASE frames).

    The frames carry energy and forces (SinglePointCalculator) so the caller can
    write an extxyz trajectory next to the JSON."""
    from ase.optimize import BFGS, FIRE

    from gradwave.calculator import GradWave

    atoms = inp.atoms.copy()
    kerker = inp.scf.mixing.kerker
    kerker = None if kerker == "auto" else bool(kerker)
    atoms.calc = GradWave(
        ecut=inp.ecut,
        pseudopotentials={s: str(inp.pseudo_dir / f)
                          for s, f in inp.pseudo_map.items()},
        xc=inp.xc,
        ecutrho=inp.ecutrho,
        kpts=inp.kpoints.mesh,
        kshift=inp.kpoints.shift,
        smearing=inp.smearing.type,
        width=inp.smearing.width,
        nbands=inp.nbands,
        use_symmetry=inp.symmetry,
        nspin=inp.nspin,
        max_iter=inp.scf.max_iter,
        etol=inp.scf.etol,
        rhotol=inp.scf.rhotol,
        diago_tol=inp.scf.diago_tol,
        mixing_scheme=inp.scf.mixing.scheme,
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history,
        mixing_kerker=kerker,
        device=inp.device,
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
    opt = opt_cls(target, logfile=None)  # we print our own richer per-step line
    trajectory = []
    frames = []  # ASE Atoms per step, energy+forces frozen for extxyz output

    def _record():
        import numpy as np
        from ase.calculators.singlepoint import SinglePointCalculator

        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        fmax = float(np.linalg.norm(forces, axis=1).max())
        entry = {
            "step": opt.nsteps,
            "energy_eV": energy,
            "fmax_eV_ang": fmax,
            "positions_ang": atoms.get_positions().tolist(),
            "cell_ang": atoms.cell.array.tolist(),
        }
        # the calculator caches its last SCF, so each relax step records how
        # many SCF iterations it took and whether it converged
        scf_res = getattr(atoms.calc, "last_result", None)
        if scf_res is not None:
            entry["scf_iter"] = int(getattr(scf_res, "n_iter", 0))
            entry["scf_converged"] = bool(getattr(scf_res, "converged", True))
        trajectory.append(entry)
        if verbose:
            sc = ((f" · SCF {entry['scf_iter']} it"
                   + ("" if entry.get("scf_converged", True) else " (NOT conv.)"))
                  if "scf_iter" in entry else "")
            print(f"  relax step {opt.nsteps:>3d} · E = {energy:+.8f} eV"
                  f" · fmax = {fmax:.5f} eV/Å{sc}", flush=True)
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
    scf_iters = [s["scf_iter"] for s in trajectory if "scf_iter" in s]
    if scf_iters:
        relax["scf_iter_per_step"] = scf_iters
        relax["scf_total_iter"] = int(sum(scf_iters))
        relax["scf_all_converged"] = all(
            s.get("scf_converged", True) for s in trajectory)
    if trajectory:
        relax["energy_change_eV"] = (
            float(atoms.get_potential_energy()) - trajectory[0]["energy_eV"])
        relax["volume_change_ang3"] = (
            float(atoms.get_volume()) - float(abs(np.linalg.det(
                np.asarray(trajectory[0]["cell_ang"])))))
    last = getattr(atoms.calc, "last_result", None)
    if last is not None and getattr(last, "system", None) is not None:
        relax["nk_ibz"] = len(last.system.kweights)
    if inp.relax.cell:
        relax["max_stress_eV_ang3"] = float(np.abs(atoms.get_stress()).max())
        relax["pressure_GPa"] = inp.relax.pressure
    return relax, atoms, frames


def run_eos(inp: Input, verbose: bool = True) -> dict:
    """Isotropic volume scan + 3rd-order Birch-Murnaghan fit → V0, B0, B0'.

    For each factor in ``inp.eos.scales`` the cell is scaled isotropically
    (a → a·s^(1/3), fractional coordinates fixed) and the SCF re-converged. All
    volumes share ONE FFT grid, pinned to the elementwise max over the scan, so
    E(V) carries no grid-discontinuity steps; each volume warm-starts from the
    previous converged density (the cheap, branch-stable EOS chain). Returns the
    ``eos`` summary block."""
    import numpy as np

    from gradwave.postscf.eos import EV_A3_TO_GPA, fit_bm3

    _species, upfs, species_of_atom = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    scales = list(inp.eos.scales)
    cell0 = np.asarray(inp.atoms.cell.array, dtype=float)
    frac = inp.atoms.get_scaled_positions()
    natoms = len(inp.atoms)

    def _build_at(scale, fft_shape):
        cell = cell0 * scale ** (1.0 / 3.0)
        pos = frac @ cell
        if uspp:
            from gradwave.scf.uspp import setup_uspp

            return setup_uspp(
                cell, pos, species_of_atom, upfs, ecut=inp.ecut,
                kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho, nbands=inp.nbands,
                use_symmetry=inp.symmetry, fft_shape=fft_shape), cell
        from gradwave.scf.loop import setup_system

        return setup_system(
            cell=cell, positions=pos, species_of_atom=species_of_atom,
            upfs=upfs, ecut=inp.ecut, kmesh=inp.kpoints.mesh,
            kshift=inp.kpoints.shift, nbands=inp.nbands,
            use_symmetry=inp.symmetry, fft_shape=fft_shape), cell

    # pass 1: natural FFT grid per volume, then pin the elementwise max so
    # every volume shares one grid (larger cells otherwise pick a finer grid)
    dims = [tuple(_build_at(s, None)[0].grid.shape) for s in scales]
    fixed = tuple(max(d[i] for d in dims) for i in range(3))
    if verbose:
        print(f"eos: {len(scales)} volumes on fixed FFT grid {fixed}", flush=True)

    prev = None
    volumes, energies, converged = [], [], []
    ekind = inp.eos.energy
    for s in scales:
        sysd, cell = _build_at(s, fixed)
        res = run_scf(inp, system=sysd, verbose=False, start_from=prev)
        prev = res
        e = float(getattr(res.energies, ekind))
        vol = float(abs(np.linalg.det(cell)))
        conv = bool(getattr(res, "converged", True))
        volumes.append(vol)
        energies.append(e)
        converged.append(conv)
        if verbose:
            tag = "" if conv else "  (NOT converged)"
            print(f"  s={s:.3f}  V={vol / natoms:8.4f} Å³/at  "
                  f"E={e / natoms:+.6f} eV/at{tag}", flush=True)

    v_at = np.array(volumes) / natoms
    e_at = np.array(energies) / natoms
    fit = fit_bm3(v_at, e_at)
    block = {
        "scales": scales,
        "energy_kind": ekind,
        "n_atoms": natoms,
        "volumes_ang3_per_atom": v_at.tolist(),
        "energies_eV_per_atom": e_at.tolist(),
        "fft_grid": list(fixed),
        "v0_ang3_per_atom": fit.v0,
        "b0_GPa": fit.b0_GPa,
        "b0_prime": fit.b0_prime,
        "e0_eV_per_atom": fit.e0,
        "rms_residual_eV_per_atom": fit.rms_residual_eV,
        "b0_eV_ang3": fit.b0,
        "ev_a3_to_gpa": EV_A3_TO_GPA,
        "all_converged": all(converged),
    }
    if verbose:
        print(f"eos: V0={fit.v0:.4f} Å³/at  B0={fit.b0_GPa:.2f} GPa  "
              f"B0'={fit.b0_prime:.3f}", flush=True)
    return block


def run_elastic(inp: Input, verbose: bool = True) -> dict:
    """Clamped-ion elastic constants: FD of the analytic stress over the six
    Voigt strains → the 6×6 stiffness C and Voigt–Reuss–Hill moduli.

    Each strain deforms the cell (fractional coordinates fixed) on one FFT grid
    pinned across the scan; every strained SCF warm-starts from the unstrained
    reference. Norm-conserving (``postscf.stress``) and USPP/PAW
    (``postscf.paw_stress``) are both handled; nspin=2 needs PAW (the
    norm-conserving stress is nspin=1 only)."""
    import numpy as np

    from gradwave.postscf.elastic import (
        elastic_tensor,
        is_mechanically_stable,
        moduli_from_cij,
    )

    if inp.noncollinear:
        raise NotImplementedError(
            "elastic constants for noncollinear/spin-orbit runs are not supported")
    _species, upfs, species_of_atom = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    if inp.nspin != 1 and not uspp:
        raise NotImplementedError(
            "elastic constants for nspin=2 need PAW/USPP pseudos "
            "(norm-conserving stress is nspin=1 only)")

    xc = SPIN_XC_REGISTRY[inp.xc]() if inp.nspin == 2 else XC_REGISTRY[inp.xc]()
    if uspp:
        from gradwave.postscf.paw_stress import stress_uspp as _stress
    else:
        from gradwave.postscf.stress import stress as _stress

    cell0 = np.asarray(inp.atoms.cell.array, dtype=float)
    frac = inp.atoms.get_scaled_positions()
    natoms = len(inp.atoms)
    h = inp.elastic.strain

    def _build(cell, fft_shape):
        pos = frac @ cell
        if uspp:
            from gradwave.scf.uspp import setup_uspp

            return setup_uspp(
                cell, pos, species_of_atom, upfs, ecut=inp.ecut,
                kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho, nbands=inp.nbands,
                use_symmetry=inp.symmetry, fft_shape=fft_shape)
        from gradwave.scf.loop import setup_system

        return setup_system(
            cell=cell, positions=pos, species_of_atom=species_of_atom,
            upfs=upfs, ecut=inp.ecut, kmesh=inp.kpoints.mesh,
            kshift=inp.kpoints.shift, nbands=inp.nbands,
            use_symmetry=inp.symmetry, fft_shape=fft_shape)

    # pin one FFT grid: the +h strains give the largest cells / finest grids
    from gradwave.postscf.elastic import voigt_strain_tensor

    probe = [cell0] + [cell0 @ (np.eye(3) + voigt_strain_tensor(j, h)).T
                       for j in range(6)]
    fixed = tuple(max(int(_build(c, None).grid.shape[i]) for c in probe)
                  for i in range(3))
    if verbose:
        print(f"elastic: strain h={h}, fixed FFT grid {fixed}", flush=True)

    # reference SCF once — warm-start seed and residual-stress readout
    ref = run_scf(inp, system=_build(cell0, fixed), verbose=False)
    converged = [bool(getattr(ref, "converged", True))]
    sigma_ref = _stress(ref, xc).detach().cpu().numpy()

    def _stress_at(eps):
        cell = cell0 @ (np.eye(3) + eps).T
        res = run_scf(inp, system=_build(cell, fixed), verbose=False,
                      start_from=ref)
        converged.append(bool(getattr(res, "converged", True)))
        return _stress(res, xc).detach().cpu().numpy()

    c = elastic_tensor(_stress_at, h=h)
    mod = moduli_from_cij(c)
    resid_gpa = float(np.abs(sigma_ref).max()) * 160.2176634
    block = {
        "strain": h,
        "n_atoms": natoms,
        "formalism": "uspp/paw" if uspp else "nc",
        "c_GPa": c.tolist(),
        "bulk_modulus_GPa": {"voigt": mod.bulk_voigt, "reuss": mod.bulk_reuss,
                             "hill": mod.bulk_hill},
        "shear_modulus_GPa": {"voigt": mod.shear_voigt, "reuss": mod.shear_reuss,
                              "hill": mod.shear_hill},
        "young_modulus_GPa": mod.young,
        "poisson_ratio": mod.poisson,
        "mechanically_stable": is_mechanically_stable(c),
        "residual_stress_GPa": resid_gpa,
        "all_converged": all(converged),
    }
    if verbose:
        print(f"elastic: K={mod.bulk_hill:.1f}  G={mod.shear_hill:.1f}  "
              f"E={mod.young:.1f} GPa  ν={mod.poisson:.3f}  "
              f"stable={block['mechanically_stable']}", flush=True)
    return block


def run_phonons(inp: Input, verbose: bool = True) -> dict:
    """Supercell finite-displacement phonons: dispersion along a q-path + a
    phonon DOS on a q-mesh.

    Builds the diagonal supercell, displaces only the primitive home-cell atoms
    (the translational reduction — 6·N_prim SCFs regardless of supercell size,
    each warm-started from the undisplaced reference), folds the force constants
    to D(q) and diagonalizes. Norm-conserving, nspin=1 (the forces path)."""
    import numpy as np

    from gradwave.postscf.phonons_supercell import (
        build_supercell,
        dispersion,
        force_constants_home,
        phonon_dos,
    )

    if inp.noncollinear or inp.nspin != 1:
        raise NotImplementedError(
            "supercell phonons are norm-conserving, nspin=1 only "
            "(the forces path does not support nspin=2 / spinors)")
    _species, upfs, species_of_atom = _species_upfs(inp)
    if _is_uspp(upfs):
        raise NotImplementedError(
            "supercell phonons need norm-conserving pseudopotentials (NC forces)")

    xc = XC_REGISTRY[inp.xc]()
    cell = np.asarray(inp.atoms.cell.array, dtype=float)
    positions = inp.atoms.get_positions()
    masses = inp.atoms.get_masses()  # primitive-atom masses [amu]
    n = inp.phonons.supercell
    scmap = build_supercell(cell, positions, species_of_atom, n)
    # fold the primitive k-mesh by the supercell size (equivalent BZ sampling)
    ksuper = tuple(max(1, inp.kpoints.mesh[i] // n[i]) for i in range(3))

    def make_scf(pos_sc, start_from=None):
        from gradwave.scf.loop import scf, setup_system

        system = setup_system(
            scmap.cell_super, pos_sc, scmap.species_super, upfs,
            ecut=inp.ecut, kmesh=ksuper, kshift=inp.kpoints.shift,
            use_symmetry=False)
        if inp.device != "cpu":
            system = system.to(inp.device)
        return scf(system, xc, smearing=inp.smearing.type, width=inp.smearing.width,
                   etol=inp.scf.etol, rhotol=inp.scf.rhotol,
                   mixing_alpha=inp.scf.mixing.alpha,
                   mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
                   diago_tol=inp.scf.diago_tol, start_from=start_from, verbose=False)

    if verbose:
        print(f"phonons: {tuple(n)} supercell ({scmap.n_sc} atoms), displacing "
              f"{scmap.n_prim} home atoms → {6 * scmap.n_prim} SCFs, k-mesh {ksuper}",
              flush=True)
    phi = force_constants_home(make_scf, scmap, h=inp.phonons.displacement,
                               verbose=verbose)

    bp = inp.atoms.cell.bandpath(path=inp.phonons.path or None,
                                 npoints=inp.phonons.npoints)
    qpts = np.asarray(bp.kpts)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    freqs = dispersion(phi, scmap, masses, qpts)  # (nq, 3·N_prim) [cm⁻¹]
    block = {
        "supercell": list(n),
        "n_atoms_supercell": scmap.n_sc,
        "displacement_ang": inp.phonons.displacement,
        "kmesh_supercell": list(ksuper),
        "qpts_frac": qpts.tolist(),
        "x": np.asarray(x).tolist(),
        "labels": list(zip(xticks.tolist(), list(xlabels), strict=True)),
        "frequencies_cm1": freqs.tolist(),
        "min_frequency_cm1": float(freqs.min()),
    }
    if min(inp.phonons.dos_mesh) > 0:
        from gradwave.kpoints import monkhorst_pack

        qmesh, weights = monkhorst_pack(inp.phonons.dos_mesh)
        grid, dos = phonon_dos(phi, scmap, masses, qmesh, weights,
                               width=inp.phonons.dos_width)
        block["dos"] = {"frequency_cm1": grid.tolist(), "dos": dos.tolist(),
                        "mesh": list(inp.phonons.dos_mesh)}
    if verbose:
        print(f"phonons: min frequency {freqs.min():.1f} cm⁻¹ "
              f"({'has imaginary modes' if freqs.min() < -1.0 else 'all real'})",
              flush=True)
    return block


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

        return NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())
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
    # force error: norm-conserving collinear (no NLCC) or USPP/PAW (nspin=1, 2,
    # incl. NLCC, no +U). The spinor force terms in P(eps) are not assembled.
    if uspp:
        force_ok = not is_nc and _get(res, "hub_sites") is None
    else:
        force_ok = (not is_nc and nspin in (1, 2)
                    and getattr(system, "rho_core", None) is None)
    if force_ok:
        try:
            fe = estimate_force_error(res, err, xc=xc).norm(dim=1)
            block["force_error_max_eV_ang"] = float(fe.max())
            block["force_error_rms_eV_ang"] = float((fe ** 2).mean().sqrt())
        except NotImplementedError as exc:
            block["force_error"] = {"available": False, "reason": str(exc)}
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
    except (NotImplementedError, ValueError) as exc:
        block["gap_error"] = {"available": False, "reason": str(exc)}
    # other numerical convergence errors (SCF self-consistency, smearing). These
    # are separate axes from the basis-set error; k-point sampling needs a mesh
    # sweep (estimate_kpoint_error) and is not reachable from one run.
    from gradwave.postscf.convergence_error import (
        estimate_scf_error,
        estimate_smearing_error,
    )
    # SCF self-consistency error is extrapolated from the energy trajectory, so
    # it is available for every system. The collinear response kernel
    # (K_Hxc/chi0) adds an optional second-order diagnostic where it applies
    # (norm-conserving collinear); USPP/PAW and the spinor formalism have no such
    # primitive exposed yet, so xc is only passed on the supported path.
    scf_xc = xc if (not uspp and not is_nc) else None
    try:
        scfe = estimate_scf_error(res, scf_xc)
        sc = {
            "denergy_eV": scfe.denergy,
            "denergy_meV_per_atom": scfe.denergy / natom * 1e3,
            "residual_L1_per_electron": scfe.residual_norm,
            "reliable": scfe.reliable,
            "ratio": scfe.ratio,
            "energy_converged_estimate_eV": scfe.energy_converged_estimate,
            "method": "energy-trajectory extrapolation",
        }
        if scfe.denergy_response is not None:
            sc["denergy_response_eV"] = scfe.denergy_response
            sc["screened"] = scfe.screened
            sc["note"] = ("response diagnostic is not sign-definite; the "
                          "headline denergy is the trajectory extrapolation")
        block["scf_convergence"] = sc
    except (NotImplementedError, ValueError, AttributeError) as exc:
        logger.debug("scf_convergence estimate skipped: %r", exc)
    try:
        # estimate_smearing_error reads res.energies — an attribute on every
        # result dataclass, USPP/PAW included
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
    except (NotImplementedError, ValueError) as exc:
        # record the reason under a distinct key: the human report reads
        # block["smearing"] eagerly, so an available:False entry there would
        # break it (a fixed-occupation run raises here every time)
        block["smearing_error"] = {"available": False, "reason": str(exc)}
    # rolled-up numerical-energy error: the reachable terms (basis-set Ecut, SCF
    # self-consistency, smearing) add only to leading order because the axes
    # couple, so this is an indicative sum rather than a rigorous total. k-point
    # sampling is not reachable from a single run and is excluded.
    terms = {"ecut": abs(float(block["denergy_eV"]))}
    if isinstance(block.get("scf_convergence"), dict):
        terms["scf"] = abs(float(block["scf_convergence"]["denergy_eV"]))
    if isinstance(block.get("smearing"), dict):
        terms["smearing"] = abs(float(block["smearing"]["dsmearing_eV"]))
    total = sum(terms.values())
    block["numerical_energy_error"] = {
        "total_eV": total,
        "total_meV_per_atom": total / natom * 1e3,
        "terms_eV": terms,
        "note": "leading-order sum of the reachable terms; k-point sampling excluded",
    }
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


def _apply_dispersion(res, inp: Input) -> dict:
    """Compute the D3(BJ) correction, fold its energy into ``res.energies`` (so
    the reported total/free energy include it), and return the summary block
    (energy, forces, stress, resolved damping). Degrades to
    ``{'available': False}`` when the element set is uncovered or no BJ preset
    exists for the functional."""
    import numpy as np
    import torch

    from gradwave.postscf.dispersion import (
        D3Config,
        dispersion_energy,
        dispersion_forces,
        dispersion_stress,
    )

    dp = inp.dispersion
    system = _get(res, "system")
    positions = system.positions.detach().to(torch.float64)
    cell = np.asarray(system.grid.cell, dtype=np.float64)
    z = [int(v) for v in inp.atoms.get_atomic_numbers()]
    try:
        cfg = D3Config.resolve(
            dp.functional or inp.xc,
            cutoff_ang=dp.cutoff, cn_cutoff_ang=dp.cn_cutoff,
            s6=dp.s6, s8=dp.s8, a1=dp.a1, a2=dp.a2,
        )
        cell_t = torch.as_tensor(cell, dtype=torch.float64, device=positions.device)
        e = dispersion_energy(positions, cell_t, z, cfg)
        forces = dispersion_forces(positions, cell, z, cfg)
        stress = dispersion_stress(positions, cell, z, cfg)
    except (ValueError, NotImplementedError) as err:
        return {"available": False, "reason": str(err)}

    # fold the energy into the breakdown; total/free_energy pick it up
    res.energies.dispersion = e.detach().to(positions.device)
    return {
        "available": True,
        "method": "d3-bj",
        "functional": (dp.functional or inp.xc).lower(),
        "damping": {"s6": cfg.s6, "s8": cfg.s8, "a1": cfg.a1, "a2_bohr": cfg.a2},
        "energy_eV": float(e),
        "energy_per_atom_eV": float(e) / len(z),
        "forces_eV_ang": forces.detach().cpu().tolist(),
        "stress_eV_ang3": stress.detach().cpu().tolist(),
    }


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
    from gradwave.postscf.magnetism import characterize_magnetism

    system = build_system(inp)
    if inp.device != "cpu":
        system = system.to(inp.device)
    xc = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())
    m = inp.magnetism
    smtype = inp.smearing.type if inp.smearing.type != "none" else "gaussian"
    return characterize_magnetism(
        system, xc, exchange=m.exchange, ref_atom=m.ref_atom, lam=m.lam,
        delta=m.delta, seed_scale=m.seed_scale, smearing=smtype,
        width=inp.smearing.width, max_iter=inp.scf.max_iter, etol=inp.scf.etol,
        rhotol=inp.scf.rhotol, mixing_alpha=inp.scf.mixing.alpha, verbose=verbose)


def _write_volumetric(res, spec, outdir, verbose) -> dict:
    """Write the requested volumetric fields (.cube/.xsf/CHGCAR) and return an
    {label: filename} map for summary["outputs"]. A field that the result type
    does not support (e.g. ELF on a noncollinear run) is skipped with a warning
    rather than losing the finished run."""
    from gradwave.postscf import volumetric as vol

    ext = "." + spec.format
    jobs = []
    if spec.density:
        jobs.append(("density", f"density{ext}", lambda p: vol.write_density(res, p)))
    if spec.elf:
        jobs.append(("elf", f"elf{ext}", lambda p: vol.write_elf(res, p)))
    if spec.magnetization:
        jobs.append(("magnetization", f"magnetization{ext}",
                     lambda p: vol.write_magnetization(res, p)))
    for band, kpt in spec.bands:
        label = f"parchg_b{band}_k{kpt}"
        jobs.append((label, f"{label}{ext}",
                     lambda p, b=band, k=kpt: vol.write_band_density(res, p, band=b, kpoint=k)))

    written = {}
    for label, name, write in jobs:
        try:
            produced = write(outdir / name)
            # a writer may emit several files (e.g. spin-resolved ELF → up/dn);
            # record their actual names, else the single fixed name
            written[label] = ([Path(p).name for p in produced]
                              if isinstance(produced, (list, tuple)) else name)
        except (NotImplementedError, ValueError) as exc:
            if verbose:
                print(f"skipped {label}: {exc}")
    return written


def _base_summary(inp: Input, task: str) -> dict:
    """The lightweight summary scaffold shared by tasks that carry no
    SCFResult (relax, magnetism, eos, elastic, phonons). SCF-derived tasks
    use build_summary() instead. Callers append their per-task result block
    and a trailing "runtime_s" so the serialized key order stays
    code/task/structure/parameters/<block>/runtime_s."""
    from gradwave import __version__

    return {
        "code": {"name": "gradwave", "version": __version__,
                 "created": datetime.datetime.now().isoformat(
                     timespec="seconds")},
        "task": task,
        "structure": _structure_block(inp),
        "parameters": _parameters_block(inp),
    }


# post-SCF tasks whose run() branch is a bare "run it, wrap the result" —
# collapsed into one data-driven branch below
_POSTSCF_RUNNERS = {"eos": run_eos, "elastic": run_elastic,
                    "phonons": run_phonons}


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
        disp_block = _apply_dispersion(res, inp) if inp.dispersion.enabled else None
        summary = build_summary(res, inp, "scf", runtime_s=time.time() - t0)
        if disp_block is not None:
            summary["dispersion"] = disp_block
        if inp.error_estimate:
            summary["error_estimate"] = _error_estimate_block(res, inp)
        if inp.projections.enabled:
            summary["pdos"] = _pdos_summary_block(res, inp)
    elif inp.task == "relax":
        relax, _atoms, _frames = run_relax(inp, verbose=verbose)
        summary = _base_summary(inp, "relax")
        summary["relax"] = relax
        summary["runtime_s"] = round(time.time() - t0, 2)
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
        summary = _base_summary(inp, "magnetism")
        summary["magnetism"] = {
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
        }
        summary["runtime_s"] = round(time.time() - t0, 2)
    elif inp.task in _POSTSCF_RUNNERS:
        block = _POSTSCF_RUNNERS[inp.task](inp, verbose=verbose)
        summary = _base_summary(inp, inp.task)
        summary[inp.task] = block
        summary["runtime_s"] = round(time.time() - t0, 2)
    else:
        raise ValueError(
            f"unknown task {inp.task!r} "
            f"(scf | relax | bands | magnetism | eos | elastic | phonons)")

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
    if res is not None and inp.output_volumetric.any():
        outputs.update(_write_volumetric(res, inp.output_volumetric, outdir, verbose))
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

    vol = float(abs(np.linalg.det(inp.atoms.cell.array)))
    block = {
        "cell_ang": inp.atoms.cell.array.tolist(),
        "positions_ang": inp.atoms.get_positions().tolist(),
        "species": inp.atoms.get_chemical_symbols(),
        "n_atoms": len(inp.atoms),
        "volume_ang3": vol,
        # 1 amu/Å³ = 1.66053906660 g/cm³
        "density_g_cm3": float(inp.atoms.get_masses().sum() * 1.66053906660 / vol),
    }
    try:
        import spglib
    except ImportError:
        return block
    try:
        ds = spglib.get_symmetry_dataset(
            (inp.atoms.cell.array, inp.atoms.get_scaled_positions(),
             inp.atoms.get_atomic_numbers()), symprec=1e-5)
        block["spacegroup"] = f"{ds.international} ({ds.number})"
        block["pointgroup"] = ds.pointgroup
        block["n_symops"] = len(ds.rotations)
    except (TypeError, AttributeError, spglib.SpglibError):
        # a degenerate/near-singular cell makes spglib raise or return None
        # (AttributeError on the None dataset); drop the field, don't swallow
        # unrelated bugs
        pass
    return block


def _parameters_block(inp: Input) -> dict:
    """Parameters block for the tasks written without a materialized System
    (relax, magnetism). The magnetism run is always the non-collinear/spinor
    formalism (matching build_summary's convention); relax follows the
    pseudopotential family."""
    import math

    species, upfs, _soa = _species_upfs(inp)
    if inp.task == "magnetism":
        formalism = "noncollinear"
    else:
        formalism = "uspp/paw" if _is_uspp(upfs) else "nc"
    return {
        "formalism": formalism,
        "xc": inp.xc,
        "ecut_eV": float(inp.ecut),
        "ecutrho_eV": None,
        "kmesh": list(inp.kpoints.mesh),
        "nk": None,
        "nk_total": int(math.prod(inp.kpoints.mesh)),
        "kweights": None,
        "nspin": inp.nspin,
        "smearing": inp.smearing.type,
        "width_eV": float(inp.smearing.width),
        "symmetry": bool(inp.symmetry),
        "mixing": {
            "scheme": inp.scf.mixing.scheme,
            "alpha": float(inp.scf.mixing.alpha),
            "history": inp.scf.mixing.history,
            "kerker": inp.scf.mixing.kerker,
            "kerker_used": None,
        },
        "pseudos": {s: inp.pseudo_map[s] for s in species},
    }
