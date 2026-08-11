"""run_relax and the nested/joint/Newton relaxation drivers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gradwave.api._common import XC_REGISTRY, _mixing_scheme
from gradwave.api.system import _is_uspp, _species_upfs
from gradwave.inputs import Input, InputError

if TYPE_CHECKING:
    from ase import Atoms

    from gradwave.calculator import GradWave
    from gradwave.pseudo.upf import UPFData
    from gradwave.pseudo.upf_paw import PAWData

logger = logging.getLogger(__name__)


# Above this estimated Pulay pressure a vc-relax is running on a severely
# underconverged basis: the (under-estimating) correction can no longer be
# trusted to reach the energy-consistent cell, so the driver warns loudly.
# Scale: converged runs sit ≪1 GPa; 12 Ry silicon ~1-2 GPa; 40 Ry ONCV-oxygen
# quartz (the #217 collapse) ~40 GPa.
_PULAY_WARN_GPA = 5.0


def _pulay_correction_unsupported(inp: Input) -> str | None:
    """Reason the Pulay stress correction cannot run this input, or None.

    Mirrors ``estimate_pressure_error``'s contract (and the calculator's
    shifted-mesh guard): norm-conserving, full (unreduced) Γ-centered k-set,
    no DFT+U, scalar-relativistic. Checked only for a variable-cell relax —
    a fixed-cell relax never applies the correction."""
    _species, upfs, _soa = _species_upfs(inp)
    if _is_uspp(upfs):
        return "USPP/PAW pseudopotentials (the stress-error estimator is NC-only)"
    if inp.symmetry:
        return ("symmetry: true (the estimator's frozen strained rebuild needs "
                "the full k-point set; set symmetry: false to enable)")
    if tuple(inp.kpoints.shift) != (0, 0, 0):
        return "shifted k-mesh (the estimator rebuilds a Γ-centered mesh only)"
    if inp.hubbard.enabled:
        return "DFT+U (pressure error with +U not implemented)"
    if inp.noncollinear:
        return "noncollinear/SOC (no calculator path)"
    if any(b.j is not None for u in upfs for b in u.betas):
        return "fully-relativistic pseudopotentials"
    return None


def _resolve_pulay_correction(inp: Input, verbose: bool = True) -> bool:
    """Whether the vc-relax calculator applies the Pulay pressure correction.

    ``relax.pulay_correction`` None (auto) → on whenever supported, with a
    visible warning naming the reason when it is not (the #217 trap — a
    vc-relax whose stress silently carries the full basis-incompleteness
    bias should never again look routine); True → required, InputError when
    unsupported; False → off. Fixed-cell relax: always False (the stress
    drives nothing)."""
    if not inp.relax.cell or inp.relax.pulay_correction is False:
        return False
    reason = _pulay_correction_unsupported(inp)
    if reason is None:
        return True
    if inp.relax.pulay_correction is True:
        raise InputError(f"relax.pulay_correction: true is unsupported here: {reason}")
    logger.warning("vc-relax without Pulay stress correction: %s", reason)
    if verbose:
        print(f"  relax: Pulay stress correction unavailable ({reason}) — the "
              "reported stress carries uncorrected basis-incompleteness "
              "(Pulay) pressure; converge ecut before trusting the relaxed "
              "cell", flush=True)
    return False


def _build_relax_calc(
    inp: Input,
    verbose: bool = True,
    scf_step_hook: Callable[[], None] | None = None,
) -> GradWave:
    """The GradWave calculator a relaxation drives — shared by the nested
    engine and by ``joint``'s final consistent-energy/forces/stress SCF at the
    relaxed geometry (so both report ASE-calculator numbers, not the joint
    functional's fixed-basis energy).

    ``verbose`` streams each ionic step's SCF trace (VASP OSZICAR-style);
    ``scf_step_hook`` (nested engine only) prints the per-step header above it."""
    from gradwave.calculator import GradWave

    kerker = inp.scf.mixing.kerker
    kerker = None if kerker == "auto" else bool(kerker)
    return GradWave(
        pulay_stress_correction=_resolve_pulay_correction(inp, verbose),
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
        tot_magnetization=inp.tot_magnetization,
        max_iter=inp.scf.max_iter,
        etol=inp.scf.etol,
        rhotol=inp.scf.rhotol,
        diago_tol=inp.scf.diago_tol,
        mixing_scheme=_mixing_scheme(inp),
        mixing_alpha=inp.scf.mixing.alpha,
        mixing_history=inp.scf.mixing.history,
        mixing_kerker=kerker,
        eigensolver=inp.scf.eigensolver,
        precond=inp.scf.mixing.precond,
        hubbard=list(inp.hubbard.manifolds) if inp.hubbard.enabled else None,
        hub_occ_mix=inp.hubbard.occ_mix,
        hub_u_ramp_iters=inp.hubbard.u_ramp_iters,
        extrapolation=inp.relax.extrapolation,
        pulay_solver=inp.relax.pulay_solver,
        # k-point sharding for every ionic step's SCF (a no-op outside a
        # torchrun launch); the calculator reassembles a full-mesh result so
        # forces/stress and the BFGS/FIRE step stay identical on every rank.
        distributed=inp.distributed,
        device=inp.device,
        verbose=verbose,
        scf_step_hook=scf_step_hook,
        # Outer-SCF tolerance ladder — nested engine only (the joint/newton
        # engines and every non-relax driver leave it None, so their SCFs run at
        # the constant configured rhotol). The driver (_relax_nested) threads the
        # previous step's fmax in and runs the exactness re-solve.
        tol_ladder=(
            {"c": inp.relax.tol_ladder_c, "p": inp.relax.tol_ladder_p,
             "rhotol_start": inp.relax.tol_ladder_rhotol_start,
             "rhotol_final": inp.relax.tol_ladder_rhotol_final,
             "first_step": inp.relax.tol_ladder_first_step}
            if inp.relax.tol_ladder and inp.relax.method == "nested" else None),
    )


def run_relax(
    inp: Input, verbose: bool = True
) -> tuple[dict[str, Any], Atoms, list[Atoms]]:
    """Relax with ASE, returning (relax block, final atoms, per-step ASE frames).

    ``relax.method`` selects the engine. ``"nested"`` (default) runs a full SCF
    inside every BFGS/FIRE geometry step — robust for every formalism, spin, and
    metals. ``"joint"`` (opt-in) descends on (strain, positions, orbitals) at
    once via ``opt.joint.joint_relax`` for far fewer Hamiltonian applications;
    it is norm-conserving-insulator only and transparently falls back to the
    nested engine on an unsupported system (USPP/PAW, spin, smearing) or if the
    joint descent does not converge. The frames carry energy and forces
    (SinglePointCalculator) so the caller can write an extxyz trajectory.

    Selective dynamics (``inp.fixed``) only the nested engine honours — the joint
    and newton engines relax all degrees of freedom — so a ``joint``/``newton``
    request with fixed atoms falls back to nested with the reason recorded in the
    ``relax`` block (``requested_method``/``fallback_reason``).

    Under ``distributed: true`` only the nested engine routes through k-point
    sharding (its per-step SCF runs on the calculator, which shards). The joint
    and newton engines drive ``opt.joint``/``opt.newton`` on their own
    unsharded systems, so a ``joint``/``newton`` request under distribution
    falls back to nested with the reason recorded, rather than silently running
    the whole mesh on every rank."""
    if inp.distributed:
        # every rank runs the identical (full-mesh-reduced) optimization, so
        # quiet all but rank 0 to avoid world_size-fold duplicated step lines
        from gradwave.distributed import current_rank

        verbose = verbose and current_rank() == 0
    if inp.distributed and inp.relax.method in ("joint", "newton"):
        if verbose:
            print(f"  relax: {inp.relax.method} engine does not route through "
                  f"k-point sharding — using nested under distributed: true",
                  flush=True)
        relax, atoms, frames = _relax_nested(inp, verbose)
        relax["requested_method"] = inp.relax.method
        relax["fallback_reason"] = (
            f"distributed: true routes only the nested engine (the "
            f"{inp.relax.method} engine has no k-point-sharded path)")
        return relax, atoms, frames
    if inp.relax.method in ("joint", "newton"):
        if inp.fixed is not None:
            if verbose:
                print(f"  relax: selective dynamics (structure.fixed) is not "
                      f"supported by the {inp.relax.method} engine — using "
                      f"nested", flush=True)
            relax, atoms, frames = _relax_nested(inp, verbose)
            relax["requested_method"] = inp.relax.method
            relax["fallback_reason"] = (
                "selective dynamics (structure.fixed) requires the nested engine")
            return relax, atoms, frames
        out = (_relax_newton(inp, verbose) if inp.relax.method == "newton"
               else _relax_joint(inp, verbose))
        if out is not None:
            return out
        if verbose:
            print(f"  relax: {inp.relax.method} engine unavailable or "
                  "non-converged — falling back to nested SCF+BFGS", flush=True)
    return _relax_nested(inp, verbose)


def _relax_nested(
    inp: Input, verbose: bool = True
) -> tuple[dict[str, Any], Atoms, list[Atoms]]:
    from ase.optimize import BFGS, FIRE

    atoms = inp.atoms.copy()
    if inp.fixed is not None:
        # Selective dynamics: fix each atom's masked axes in FRACTIONAL space, so
        # held atoms ride the cell under a variable-cell relax. ASE applies the
        # constraint inside get_forces (zeroing the held components before the
        # optimizer sees them and before its fmax gate) and inside the position
        # update (so a fully-fixed atom never moves). FixScaled masks the same
        # axes for a fixed cell too, where fractional and Cartesian coincide.
        from ase.constraints import FixScaled

        atoms.set_constraint([
            FixScaled(i, mask=tuple(bool(b) for b in inp.fixed[i]))
            for i in range(len(atoms)) if bool(inp.fixed[i].any())
        ])
    # VASP OSZICAR-style output: a per-ionic-step header above that step's SCF
    # trace. The calculator fires _scf_header once per FRESH SCF (one per
    # geometry — cached property fetches don't re-trigger it), so ion_step["n"]
    # counts ionic steps 1, 2, 3 … and the post-step summary below reuses it, so
    # the header and its summary line always carry the same number.
    ion_step: dict[str, Any] = {"n": 0, "t0": None}

    def _scf_header() -> None:
        ion_step["n"] += 1
        ion_step["t0"] = time.perf_counter()  # stamped at the SCF start (see _record)
        if verbose:
            print(f"\n── ionic step {ion_step['n']} ──", flush=True)

    atoms.calc = _build_relax_calc(inp, verbose, scf_step_hook=_scf_header)
    opt_cls = {"fire": FIRE, "bfgs": BFGS}[inp.relax.optimizer]
    target: Atoms | FrechetCellFilter = atoms
    if inp.relax.cell:
        from ase.filters import FrechetCellFilter

        # ASE stress/pressure is eV/Å³; the user knob is GPa. The filter adds
        # the cell degrees of freedom, so opt.run(fmax) then gates BOTH the
        # atomic forces and the stress (external pressure subtracted).
        gpa_to_ev_a3 = 1.0 / 160.21766208
        target = FrechetCellFilter(
            atoms, scalar_pressure=inp.relax.pressure * gpa_to_ev_a3)
    # Parallel/speculative line search (opt-in): a positions-only bfgs relax may
    # replace each step's fixed length with one parallel round (see
    # opt/line_search.py). A cell/fire request keeps the serial step and records
    # why, so the default and unsupported paths stay byte-identical to today.
    ls_mode = inp.relax.line_search
    ls_opt: Any = None
    ls_fallback_reason: str | None = None
    if ls_mode != "off":
        if inp.relax.cell:
            ls_fallback_reason = "cell relaxation (line search is positions-only)"
        elif inp.relax.optimizer != "bfgs":
            ls_fallback_reason = (
                f"optimizer={inp.relax.optimizer!r} (line search needs bfgs)")
        else:
            from gradwave.opt.line_search import make_line_search_bfgs

            ls_opt = make_line_search_bfgs(target, inp=inp, verbose=verbose)
            if verbose:
                print(f"  relax: {ls_mode} line search "
                      f"({inp.relax.line_search_n_samples} samples, "
                      f"{inp.relax.line_search_n_workers} workers) — forward-only",
                      flush=True)
        if ls_fallback_reason is not None and verbose:
            print(f"  relax: line_search={ls_mode!r} ignored — "
                  f"{ls_fallback_reason}; serial step", flush=True)
    # ASE's Optimizer.__init__ declares atoms: Atoms, but at runtime accepts
    # any Atoms-like object implementing get_positions/get_forces/etc. —
    # every ase.filters wrapper (FrechetCellFilter included) is meant to be
    # passed here; the stub just doesn't spell out that duck-typed contract.
    opt = (ls_opt if ls_opt is not None
           else opt_cls(cast("Atoms", target), logfile=None))  # per-step line below
    trajectory: list[dict[str, Any]] = []
    frames: list[Atoms] = []  # ASE Atoms per step, energy+forces frozen for extxyz output
    # incremental trajectory: each step's frame is appended to this file as it
    # completes, so an interrupted relax keeps its progress. write_output re-emits
    # the identical full file at the end. Under distribution only rank 0 writes
    # (matching write_output's gating), else every run writes — independent of -q.
    traj_path = Path(inp.output_dir) / "relax.xyz"
    _write_traj = True
    if inp.distributed:
        from gradwave.distributed import current_rank

        _write_traj = current_rank() == 0
    if _write_traj:
        try:
            traj_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("could not prepare %s (%s); incremental trajectory "
                           "off, final write still attempted", traj_path, exc)
            _write_traj = False

    def _record() -> None:
        import numpy as np
        from ase.calculators.singlepoint import SinglePointCalculator

        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        # fmax the optimizer actually gates on: for a variable-cell relax `target`
        # is the FrechetCellFilter, whose force array appends the stress-derived
        # rows, so this matches the convergence criterion. Reporting atomic-only
        # forces there can read ~0 (symmetric cell) while stress still drives the
        # cell — a "fmax = 0.00000" next to "NOT CONVERGED". Positions-only relax:
        # `target` is `atoms`, so this is identical to the atomic fmax.
        fmax = float(np.linalg.norm(target.get_forces(), axis=1).max())
        # wall time for this ionic step: SCF start (stamped in _scf_header, just
        # before the fresh SCF) to here (forces/step done). ~VASP's LOOP+ time.
        wall_s = (time.perf_counter() - ion_step["t0"]) if ion_step["t0"] else 0.0
        entry: dict[str, Any] = {
            "step": opt.nsteps,
            "energy_eV": energy,
            "fmax_eV_ang": fmax,
            "wall_s": round(wall_s, 2),
            "positions_ang": atoms.get_positions().tolist(),
            "cell_ang": atoms.cell.array.tolist(),
        }
        # the calculator caches its last SCF, so each relax step records how
        # many SCF iterations it took and whether it converged
        scf_res = getattr(atoms.calc, "last_result", None)
        if scf_res is not None:
            entry["scf_iter"] = int(getattr(scf_res, "n_iter", 0))
            entry["scf_converged"] = bool(getattr(scf_res, "converged", True))
        if inp.relax.tol_ladder:
            # thread THIS accepted geometry's optimizer fmax (target's, so it
            # folds in the stress under a cell relax) into the calculator so the
            # NEXT ionic step's SCF tol is scheduled from it; record the tol this
            # step actually ran at for the trace.
            atoms.calc._last_fmax = fmax
            rt = getattr(atoms.calc, "last_rhotol_used", None)
            if rt is not None:
                entry["rhotol_used"] = float(rt)
        pulay_gpa = getattr(atoms.calc, "last_pulay_pressure_gpa", None)
        if pulay_gpa is not None:
            entry["pulay_pressure_GPa"] = round(float(pulay_gpa), 4)
        trajectory.append(entry)
        if verbose:
            sc = ((f" · SCF {entry['scf_iter']} it"
                   + ("" if entry.get("scf_converged", True) else " (NOT conv.)"))
                  if "scf_iter" in entry else "")
            pl = (f" · Pulay {pulay_gpa:+.2f} GPa" if pulay_gpa is not None
                  else "")
            print(f"  ionic step {ion_step['n']:>3d} · E = {energy:+.8f} eV"
                  f" · fmax = {fmax:.5f} eV/Å{sc}{pl} · {wall_s:.1f}s", flush=True)
        frame = atoms.copy()
        sp_kw: dict[str, Any] = {"energy": energy, "forces": forces}
        if inp.relax.cell:
            sp_kw["stress"] = atoms.get_stress()
        # total magnetic moment for a spin-polarized run (the calculator sets
        # results["magmom"] only for nspin=2); rides the frame into the extxyz
        mag = atoms.calc.results.get("magmom") if atoms.calc is not None else None
        if mag is not None:
            sp_kw["magmom"] = float(mag)
        frame.calc = SinglePointCalculator(frame, **sp_kw)
        frame.info["step"] = opt.nsteps
        frames.append(frame)
        if _write_traj:
            from ase.io import write as ase_write

            # First frame truncates (append=False) — cleanly overwriting any
            # stale/corrupt/leftover relax.xyz; the rest append. append mode
            # doesn't read the file, so a corrupt existing file can't break it,
            # and a missing file/dir is (re)created. Best-effort regardless: a
            # write failure (disk full, permissions, races) must not abort a
            # multi-hour relax, so warn and carry on — write_output re-emits the
            # full trajectory at the end.
            try:
                ase_write(str(traj_path), frame, format="extxyz",
                          append=len(frames) > 1)
            except Exception as exc:  # noqa: BLE001 — best-effort progress dump
                logger.warning("could not append relax step to %s: %s",
                               traj_path, exc)

    if inp.relax.initial_hessian == "lindh":
        # seed a curvature-aware model Hessian so early steps don't overshoot the
        # stiff directions (atomic DOFs only — a cell relax keeps the default)
        if inp.relax.cell:
            if verbose:
                print("  relax: initial_hessian=lindh ignored under a cell relax "
                      "(atomic model Hessian only)", flush=True)
        else:
            from gradwave.opt.model_hessian import lindh_hessian, seed_bfgs_hessian

            h0 = lindh_hessian(atoms.get_positions(),
                               list(atoms.get_atomic_numbers()))
            applied = seed_bfgs_hessian(opt, h0)
            if verbose:
                print("  relax: initial_hessian=lindh "
                      f"({'applied' if applied else 'skipped — DOF mismatch'})",
                      flush=True)

    opt.attach(_record)
    try:
        converged = opt.run(fmax=inp.relax.fmax, steps=inp.relax.max_steps)
    finally:
        # tear down the line search's persistent candidate pool (no-op otherwise)
        _close_pool = getattr(opt, "_ls_close_pool", None)
        if callable(_close_pool):
            _close_pool()
    import numpy as np

    # Exactness gate for the tolerance ladder: the trajectory above converged the
    # SCFs to a schedule of loosened rhotol, so re-solve the final geometry ONCE
    # at the full rhotol (warm-started from the loose state) and report THAT
    # energy/forces/stress. A baseline (ladder-off) relax skips this entirely.
    final_resolve_iter: int | None = None
    if inp.relax.tol_ladder and atoms.calc is not None:
        if verbose:
            print("  tol_ladder: re-solving the converged geometry at full "
                  "rhotol (exactness gate) …", flush=True)
        final_resolve_iter = atoms.calc.resolve_full_tol()

    relax: dict[str, Any] = {
        "converged": bool(converged),
        "method": "nested",
        "n_steps": opt.nsteps,
        "optimizer": inp.relax.optimizer,
        "cell_relaxed": bool(inp.relax.cell),
        "fmax_target_eV_ang": inp.relax.fmax,
        "energy_eV": float(atoms.get_potential_energy()),
        # the optimizer's convergence quantity (target = FrechetCellFilter under
        # a cell relax, so this includes stress), matching the per-step fmax and
        # the fmax_target gate — not the atomic-only forces
        "fmax_eV_ang": float(
            np.linalg.norm(target.get_forces(), axis=1).max()),
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
    if inp.relax.tol_ladder:
        relax["tol_ladder"] = True
        if final_resolve_iter is not None:
            # the exactness re-solve's SCF iterations are part of the total cost
            relax["final_resolve_scf_iter"] = int(final_resolve_iter)
            if "scf_total_iter" in relax:
                relax["scf_total_iter"] += int(final_resolve_iter)
    relax["extrapolation"] = inp.relax.extrapolation
    if ls_mode != "off":
        # document the parallel line search: whether it engaged, and (adaptive)
        # how many ionic steps actually fanned candidates out vs took the serial
        # step — so an easy relax shows a dormant trigger and a soft/overshoot
        # relax shows it firing.
        relax["line_search"] = ls_mode
        if ls_fallback_reason is not None:
            relax["line_search_active"] = False
            relax["line_search_fallback_reason"] = ls_fallback_reason
        else:
            events = list(getattr(opt, "_ls_events", []))
            relax["line_search_active"] = True
            relax["line_search_n_samples"] = inp.relax.line_search_n_samples
            relax["line_search_n_workers"] = inp.relax.line_search_n_workers
            relax["line_search_steps_searched"] = int(
                sum(1 for e in events if e.get("searched")))
            relax["line_search_events"] = events
    if getattr(atoms.calc, "_density_clamped", False):
        # the extrapolated density dipped negative on at least one step and was
        # clamped to zero then renormalized to N_e (a benign, recorded fallback)
        relax["extrapolation_density_clamped"] = True
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
        pulay_final = getattr(atoms.calc, "last_pulay_pressure_gpa", None)
        relax["pulay_correction"] = pulay_final is not None
        if pulay_final is not None:
            relax["pulay_pressure_GPa_final"] = round(float(pulay_final), 4)
            if abs(pulay_final) > _PULAY_WARN_GPA and verbose:
                print(f"  relax: WARNING — estimated Pulay pressure "
                      f"{pulay_final:+.1f} GPa exceeds {_PULAY_WARN_GPA:.0f} GPa: "
                      "the stress at this ecut is severely underconverged. The "
                      "correction reduces but does not eliminate the bias "
                      "(first-order estimate, ~0.5-0.75x); increase ecut before "
                      "trusting the relaxed cell.", flush=True)
    if inp.fixed is not None:
        # record the selective-dynamics mask (True = axis held fixed) so the
        # output documents which degrees of freedom were frozen
        relax["fixed"] = inp.fixed.tolist()
        relax["n_fixed_atoms"] = int(inp.fixed.any(axis=1).sum())
    return relax, atoms, frames


def _joint_supported(inp: Input, upfs: list[UPFData | PAWData]) -> str | None:
    """Reason the joint engine cannot run this input, or None if it can.

    Joint descent (opt/joint.py) is the norm-conserving, nspin=1, fixed-integer-
    occupation prototype: insulators only. Everything outside that contract
    (USPP/PAW overlap, spin, smeared/metallic occupations, external pressure)
    routes to the nested engine instead — see the coverage matrix in
    docs/design/joint-geometry-electronic.md."""
    if _is_uspp(upfs):
        return "USPP/PAW pseudopotential (generalized S-overlap not on the joint graph)"
    if inp.nspin == 2 or inp.noncollinear:
        return "spin-polarized / noncollinear (joint prototype is nspin=1)"
    if inp.smearing.type != "none":
        return f"smearing={inp.smearing.type!r} (metals reintroduce level crossings)"
    if inp.relax.pressure != 0.0:
        return "external pressure (joint functional minimizes E, not enthalpy)"
    if inp.fixed is not None:
        return ("selective dynamics (structure.fixed) — the joint/newton engines "
                "relax all degrees of freedom; use method=nested to hold atoms")
    return None


def _relax_joint(
    inp: Input, verbose: bool = True
) -> tuple[dict[str, Any], Atoms, list[Atoms]] | None:
    """Joint (strain, positions, orbitals) descent as a relax engine.

    Returns the same ``(relax, atoms, frames)`` tuple as the nested engine, or
    ``None`` to signal the caller should fall back to nested — either because
    the system is outside the joint contract or because the descent did not
    converge. Convergence gates are mapped from the ASE calculator's: the force
    tolerance is ``relax.fmax`` and (variable cell) the stress tolerance is
    ``fmax/Ω`` — the FrechetCellFilter treats σ·Ω as a generalized force gated
    by the same scalar. The final energy/forces/stress are recomputed with one
    calculator SCF at the relaxed geometry so they are ASE-consistent (not the
    joint functional's fixed-basis value) and ``last_result`` is populated for
    downstream error estimates."""
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator

    from gradwave.opt.joint import joint_relax

    species, upfs, species_of_atom = _species_upfs(inp)
    reason = _joint_supported(inp, upfs)
    if reason is not None:
        logger.info("relax method=joint not applicable: %s", reason)
        return None

    cell0 = inp.atoms.cell.array.copy()
    pos0 = inp.atoms.get_positions().copy()
    fix_cell = not inp.relax.cell
    omega = float(abs(np.linalg.det(cell0)))
    smax = inp.relax.fmax / omega  # σ·Ω is the FrechetCellFilter cell "force"
    try:
        res = joint_relax(
            cell0, pos0, species_of_atom, upfs, XC_REGISTRY[inp.xc](),
            ecut=inp.ecut, kmesh=inp.kpoints.mesh, fmax=inp.relax.fmax,
            smax=smax, max_closures=40 * inp.relax.max_steps,
            fix_cell=fix_cell, device=inp.device, verbose=verbose,
        )
    except (ValueError, RuntimeError) as exc:  # torch LinAlgError ⊂ RuntimeError
        logger.warning("joint relax failed (%s); falling back to nested", exc)
        return None
    if not res.converged:
        logger.info("joint relax did not converge in %d closures; falling back",
                    res.n_closures)
        return None

    # final ASE-consistent energy/forces/stress at the relaxed geometry
    atoms = inp.atoms.copy()
    atoms.set_cell(res.cell, scale_atoms=False)
    atoms.set_positions(res.positions)
    atoms.calc = _build_relax_calc(inp, verbose)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float(np.linalg.norm(forces, axis=1).max())
    sp_kw: dict[str, Any] = {"energy": energy, "forces": forces}
    if inp.relax.cell:
        sp_kw["stress"] = atoms.get_stress()
    frame = atoms.copy()
    frame.calc = SinglePointCalculator(frame, **sp_kw)
    frame.info["step"] = res.n_closures
    last = getattr(atoms.calc, "last_result", None)

    relax: dict[str, Any] = {
        "converged": True,
        "method": "joint",
        "n_steps": res.n_cycles,             # basis-rebuild cycles (outer loop)
        "n_closures": res.n_closures,        # L-BFGS energy+grad evaluations
        "optimizer": "lbfgs",
        "cell_relaxed": bool(inp.relax.cell),
        "fmax_target_eV_ang": inp.relax.fmax,
        "energy_eV": energy,
        "fmax_eV_ang": fmax_final,
        "max_displacement_ang": float(np.linalg.norm(
            atoms.get_positions() - inp.atoms.get_positions(), axis=1).max()),
        "species": atoms.get_chemical_symbols(),
        "positions_ang": atoms.get_positions().tolist(),
        "cell_ang": atoms.cell.array.tolist(),
        "volume_ang3": float(atoms.get_volume()),
        # H-apply provenance — the reason to use this engine
        "h_applies": int(res.h_equiv),
        "h_seed": int(res.h_seed),
    }
    if last is not None:
        relax["scf_iter_final"] = int(getattr(last, "n_iter", 0))
        if getattr(last, "system", None) is not None:
            relax["nk_ibz"] = len(last.system.kweights)
    if inp.relax.cell:
        relax["max_stress_eV_ang3"] = float(np.abs(atoms.get_stress()).max())
        relax["pressure_GPa"] = inp.relax.pressure
    if verbose:
        print(f"  relax: joint engine converged — E = {energy:+.8f} eV · "
              f"fmax = {fmax_final:.5f} eV/Å · {res.n_cycles} cycles / "
              f"{res.n_closures} closures · {res.h_equiv} H-applies", flush=True)
    return relax, atoms, [frame]


def _relax_newton(
    inp: Input, verbose: bool = True
) -> tuple[dict[str, Any], Atoms, list[Atoms]] | None:
    """Exact-Hvp Steihaug trust-region Newton-CG as a relax engine.

    Same contract, guard, and nested fallback as ``_relax_joint`` (returns
    ``None`` to fall back); the only difference is the inner optimizer
    (``opt.newton.newton_cg_relax``) and the H-apply provenance it reports
    (grad/Hvp/trial counts instead of L-BFGS closures). Final energy/forces/
    stress are recomputed with one calculator SCF at the relaxed geometry so
    the reported numbers are ASE-consistent."""
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator

    from gradwave.opt.newton import newton_cg_relax

    species, upfs, species_of_atom = _species_upfs(inp)
    reason = _joint_supported(inp, upfs)
    if reason is not None:
        logger.info("relax method=newton not applicable: %s", reason)
        return None

    cell0 = inp.atoms.cell.array.copy()
    pos0 = inp.atoms.get_positions().copy()
    fix_cell = not inp.relax.cell
    omega = float(abs(np.linalg.det(cell0)))
    smax = inp.relax.fmax / omega
    try:
        res = newton_cg_relax(
            cell0, pos0, species_of_atom, upfs, XC_REGISTRY[inp.xc](),
            ecut=inp.ecut, kmesh=inp.kpoints.mesh, fmax=inp.relax.fmax,
            smax=smax, max_newton=inp.relax.max_steps, fix_cell=fix_cell,
            device=inp.device, verbose=verbose,
        )
    except (ValueError, RuntimeError) as exc:  # torch LinAlgError ⊂ RuntimeError
        logger.warning("newton relax failed (%s); falling back to nested", exc)
        return None
    if not res.converged:
        logger.info("newton relax did not converge (%d Newton steps); falling "
                    "back", res.n_newton)
        return None

    atoms = inp.atoms.copy()
    atoms.set_cell(res.cell, scale_atoms=False)
    atoms.set_positions(res.positions)
    atoms.calc = _build_relax_calc(inp, verbose)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float(np.linalg.norm(forces, axis=1).max())
    sp_kw: dict[str, Any] = {"energy": energy, "forces": forces}
    if inp.relax.cell:
        sp_kw["stress"] = atoms.get_stress()
    frame = atoms.copy()
    frame.calc = SinglePointCalculator(frame, **sp_kw)
    frame.info["step"] = res.n_newton
    last = getattr(atoms.calc, "last_result", None)

    relax: dict[str, Any] = {
        "converged": True,
        "method": "newton",
        "n_steps": res.n_cycles,
        "n_newton": res.n_newton,
        "n_grad": res.n_grad,
        "n_hvp": res.n_hvp,
        "optimizer": "steihaug-newton-cg",
        "cell_relaxed": bool(inp.relax.cell),
        "fmax_target_eV_ang": inp.relax.fmax,
        "energy_eV": energy,
        "fmax_eV_ang": fmax_final,
        "max_displacement_ang": float(np.linalg.norm(
            atoms.get_positions() - inp.atoms.get_positions(), axis=1).max()),
        "species": atoms.get_chemical_symbols(),
        "positions_ang": atoms.get_positions().tolist(),
        "cell_ang": atoms.cell.array.tolist(),
        "volume_ang3": float(atoms.get_volume()),
        "h_applies": int(res.h_equiv),
        "h_seed": int(res.h_seed),
    }
    if last is not None:
        relax["scf_iter_final"] = int(getattr(last, "n_iter", 0))
        if getattr(last, "system", None) is not None:
            relax["nk_ibz"] = len(last.system.kweights)
    if inp.relax.cell:
        relax["max_stress_eV_ang3"] = float(np.abs(atoms.get_stress()).max())
        relax["pressure_GPa"] = inp.relax.pressure
    if verbose:
        print(f"  relax: newton-cg engine converged — E = {energy:+.8f} eV · "
              f"fmax = {fmax_final:.5f} eV/Å · {res.n_newton} steps · "
              f"{res.n_hvp} Hvp · {res.h_equiv} H-applies", flush=True)
    return relax, atoms, [frame]
