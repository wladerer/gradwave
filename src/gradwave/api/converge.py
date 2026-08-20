"""recommend_convergence: target precision -> (ecut, kmesh) with evidence.

The automated half of the Janssen et al. protocol ("Automated optimization and
uncertainty quantification of convergence parameters in plane wave DFT
calculations", npj Comput. Mater. 10 (2024), DOI 10.1038/s41524-024-01388-2):
convergence parameters are chosen against BOTH error families —

  systematic: the ecut error curve, scored for every candidate cutoff from ONE
    high-ecut run (``postscf.discretization_error.estimate_ecut_error_curve``,
    the reversed-annulus trick), and the k-mesh quadrature error from a short
    mesh sweep (``postscf.convergence_error.estimate_kpoint_error``);
  statistical: the grid-discreteness noise floor measured by rigid-translation
    probes (``postscf.convergence_noise.estimate_noise_floor``).

The recommendation rule is the paper's phase-diagram insight: the smallest
parameters whose systematic errors sit below max(target, noise floor). A
target below the noise floor is UNACHIEVABLE at these parameters — tightening
ecut/k past that point resolves noise, not physics — and is reported as such.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from gradwave.api._common import XC_REGISTRY
from gradwave.api.scf import run_scf
from gradwave.api.system import _as_upfs, _is_uspp, _species_upfs
from gradwave.inputs import Input

if TYPE_CHECKING:

    from gradwave.scf.loop import SCFResult

logger = logging.getLogger(__name__)

# default candidate cutoffs, as fractions of the high reference run's ecut
_ECUT_FRACTIONS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# default k-mesh sweep, as multipliers on the input mesh
_KMESH_FACTORS = (1.0, 1.5, 2.0)
# uniform scale factors searched when no sampled mesh meets the target
_KMESH_EXTRAP_FACTORS = tuple(0.5 * n for n in range(1, 17))


def _scaled_mesh(base: tuple[int, ...], f: float) -> tuple[int, ...]:
    return tuple(max(1, round(f * m)) for m in base)


def _apply_rule(
    *,
    target: float,
    floor: float,
    ecuts: list[float],
    ecut_err_at: list[float],
    tail_at: float,
    ecut_run: float,
    base_ecut: float,
    meshes: list[tuple[int, ...]],
    kmesh_err_at: list[float],
    c_k: float,
    p: float,
    base_mesh: tuple[int, ...],
) -> tuple[bool, float, float | None, tuple[int, ...] | None, bool, list[str]]:
    """The paper's recommendation rule on plain numbers (unit-testable).

    Smallest ecut / mesh whose estimated systematic error (eV/atom) sits below
    the effective target max(target, floor); a target below the statistical
    noise floor is flagged unachievable. Returns ``(achievable, effective,
    rec_ecut, rec_mesh, k_extrapolated, notes)``.
    """
    import numpy as np

    notes: list[str] = []
    achievable = target >= floor
    effective = max(target, floor)
    if not achievable:
        notes.append(
            f"target {target:.2e} eV/atom sits below the measured noise floor "
            f"{floor:.2e} eV/atom at (ecut={base_ecut:.0f} eV, mesh "
            f"{tuple(base_mesh)}): the target is unachievable by refining "
            "ecut/k alone; recommending against the floor instead")

    rec_ecut: float | None = None
    for ec, err in zip(ecuts, ecut_err_at, strict=True):
        if err <= effective:
            rec_ecut = ec
            break
    if rec_ecut is None:
        if tail_at <= effective:
            rec_ecut = ecut_run
            notes.append(
                "no candidate cutoff met the target; the reference cutoff "
                f"itself does (tail {tail_at:.2e} eV/atom)")
        else:
            notes.append(
                f"even the reference cutoff's residual error ({tail_at:.2e} "
                "eV/atom) exceeds the effective target; raise ecut_run")

    rec_mesh: tuple[int, ...] | None = None
    k_extrapolated = False
    for m, err in zip(meshes, kmesh_err_at, strict=True):
        if err <= effective:
            rec_mesh = m
            break
    if rec_mesh is None:
        for f in _KMESH_EXTRAP_FACTORS:
            m = _scaled_mesh(base_mesh, f)
            if abs(c_k) * int(np.prod(m)) ** (-p) <= effective:
                rec_mesh = m
                k_extrapolated = True
                notes.append(
                    f"no sampled mesh met the target; {m} is extrapolated "
                    f"from the fitted error law c*N^-p (p={p:.2f})")
                break
        if rec_mesh is None:
            notes.append(
                "no mesh within the extrapolation search met the effective "
                "target; densify the sweep or relax the target")
    return achievable, effective, rec_ecut, rec_mesh, k_extrapolated, notes


def recommend_convergence(
    inp: Input,
    *,
    target: float,
    ecut_run: float | None = None,
    ecut_candidates: list[float] | None = None,
    kmeshes: list[tuple[int, ...]] | None = None,
    kpoint_exponent: float | None = None,
    n_translations: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Recommend (ecut, kmesh) meeting a target energy precision, with evidence.

    Runs three cheap campaigns on the input's system and combines them:

    1. ONE SCF at a high reference cutoff ``ecut_run`` (default 2x ``inp.ecut``)
       scores every candidate lower cutoff via the reversed-annulus estimator —
       no per-candidate runs.
    2. A short k-mesh sweep at ``inp.ecut`` (default 1x/1.5x/2x the input mesh,
       warm-started along the chain) fit by the quadrature extrapolation
       E(N_k) = E_inf + c N^-p.
    3. ``n_translations`` rigid-translation probes at (``inp.ecut``, input
       mesh) measuring the statistical noise floor sigma_E.

    ``target`` is the acceptable energy error in eV/atom. The recommendation
    is the smallest candidate ecut and smallest mesh whose estimated
    systematic errors fall below ``max(target, noise floor)``; when the target
    sits below the measured noise floor it is flagged unachievable
    (``achievable: False``) and the floor becomes the effective target.

    Norm-conserving collinear (nspin=1) inputs only. Returns a summary block
    (JSON-serializable dict); see the ``notes`` list inside it for caveats,
    e.g. when the k recommendation had to extrapolate beyond the sampled
    meshes. The noise floor is measured at the BASE parameters — it grows
    mildly as ecut/k shrink, so treat a recommendation sitting exactly at the
    floor as optimistic.
    """
    import numpy as np

    from gradwave.postscf.convergence_error import estimate_kpoint_error
    from gradwave.postscf.convergence_noise import estimate_noise_floor
    from gradwave.postscf.discretization_error import estimate_ecut_error_curve
    from gradwave.scf.loop import setup_system

    if target <= 0.0:
        raise ValueError("target must be a positive energy precision [eV/atom]")
    _species, upfs, species_of_atom = _species_upfs(inp)
    if _is_uspp(upfs):
        raise NotImplementedError(
            "recommend_convergence is norm-conserving only (the one-run ecut "
            "curve and the noise probes are NC nspin=1)")
    if inp.noncollinear or inp.nspin != 1 or inp.hybrid.enabled:
        raise NotImplementedError(
            "recommend_convergence supports the plain collinear nspin=1 SCF")

    cell = np.asarray(inp.atoms.cell.array, dtype=float)
    frac = inp.atoms.get_scaled_positions()
    pos = frac @ cell
    natoms = len(inp.atoms)
    base_mesh = tuple(inp.kpoints.mesh)
    kshift = tuple(inp.kpoints.shift)
    notes: list[str] = []

    def _build(ecut: float, kmesh: tuple[int, ...]):
        return setup_system(
            cell=cell, positions=pos, species_of_atom=species_of_atom,
            upfs=_as_upfs(upfs), ecut=float(ecut), kmesh=kmesh, kshift=kshift,
            nbands=inp.nbands, use_symmetry=inp.symmetry)

    # ---- 1. systematic ecut curve from ONE high-ecut run ------------------ #
    ecut_run = float(ecut_run) if ecut_run is not None else 2.0 * float(inp.ecut)
    if ecut_run <= float(inp.ecut):
        raise ValueError(
            f"ecut_run={ecut_run:.1f} eV must exceed the input ecut "
            f"({float(inp.ecut):.1f} eV)")
    if ecut_candidates is None:
        ecut_candidates = [f * ecut_run for f in _ECUT_FRACTIONS]
    if verbose:
        print(f"converge: reference run at ecut={ecut_run:.1f} eV, "
              f"mesh {base_mesh}", flush=True)
    # the USPP/noncollinear/hybrid formalisms are all rejected above, so every
    # run_scf below returns a plain collinear SCFResult
    res_high = cast("SCFResult",
                    run_scf(inp, system=_build(ecut_run, base_mesh),
                            verbose=False))
    curve = estimate_ecut_error_curve(res_high, list(ecut_candidates))
    ecut_err_at = [e / natoms for e in curve.error]
    tail_at = curve.tail / natoms

    # ---- 2. k-mesh quadrature error from a short sweep -------------------- #
    if kmeshes is None:
        kmeshes = [_scaled_mesh(base_mesh, f) for f in _KMESH_FACTORS]
    meshes = sorted({tuple(int(x) for x in m) for m in kmeshes},
                    key=lambda m: int(np.prod(m)))
    if base_mesh not in meshes:
        meshes.insert(0, base_mesh)
        meshes.sort(key=lambda m: int(np.prod(m)))
    prev = None
    k_energies: list[float] = []
    res_base = None
    for m in meshes:
        r = cast("SCFResult",
                 run_scf(inp, system=_build(float(inp.ecut), m),
                         verbose=False, start_from=prev))
        prev = r
        k_energies.append(float(r.energies.free_energy))
        if verbose:
            print(f"converge: mesh {m}: E={k_energies[-1]:+.6f} eV", flush=True)
        if m == base_mesh:
            res_base = r
    nkpts = [int(np.prod(m)) for m in meshes]
    kp = estimate_kpoint_error(nkpts, k_energies, exponent=kpoint_exponent)
    p = float(kp["exponent"])
    # amplitude of the fitted error law, err(N) ~= c * N^-p
    c_k = float(kp["error_eV"]) * nkpts[-1] ** p

    # ---- 3. statistical noise floor at the base parameters ---------------- #
    assert res_base is not None  # base_mesh is always in `meshes`
    xc = XC_REGISTRY[inp.xc]()
    noise = estimate_noise_floor(
        res_base, xc, kmesh=base_mesh, kshift=kshift,
        n_translations=n_translations, seed=seed, verbose=verbose)

    # ---- recommendation rule --------------------------------------------- #
    floor = noise.sigma_e_per_atom
    achievable, effective, rec_ecut, rec_mesh, k_extrapolated, rule_notes = (
        _apply_rule(
            target=target, floor=floor, ecuts=curve.ecuts,
            ecut_err_at=ecut_err_at, tail_at=tail_at, ecut_run=ecut_run,
            base_ecut=float(inp.ecut), meshes=meshes,
            kmesh_err_at=[abs(per["error_eV"]) / natoms
                          for per in kp["per_mesh"]],
            c_k=c_k / natoms, p=p, base_mesh=base_mesh))
    notes.extend(rule_notes)

    block: dict[str, Any] = {
        "target_eV_per_atom": float(target),
        "noise_floor_eV_per_atom": floor,
        "achievable": achievable,
        "effective_target_eV_per_atom": effective,
        "recommended_ecut_eV": rec_ecut,
        "recommended_kmesh": list(rec_mesh) if rec_mesh is not None else None,
        "kmesh_extrapolated": k_extrapolated,
        "n_atoms": natoms,
        "ecut_run_eV": ecut_run,
        "ecut_curve": [
            {"ecut_eV": ec, "error_eV_per_atom": err}
            for ec, err in zip(curve.ecuts, ecut_err_at, strict=True)],
        "ecut_tail_eV_per_atom": tail_at,
        "kpoint": {
            "meshes": [list(m) for m in meshes],
            "energies_eV": k_energies,
            **{k: v for k, v in kp.items()},
        },
        "noise": {
            "sigma_e_eV": noise.sigma_e,
            "sigma_e_eV_per_atom": noise.sigma_e_per_atom,
            "spread_e_eV": noise.spread_e,
            "n_translations": noise.n_translations,
            "energies_eV": noise.energies,
            "all_converged": noise.all_converged,
            "measured_at": {"ecut_eV": float(inp.ecut),
                            "kmesh": list(base_mesh)},
        },
        "notes": notes,
    }
    if verbose:
        ach = "achievable" if achievable else "UNACHIEVABLE (noise floor)"
        ec_s = f"{rec_ecut:.1f} eV" if rec_ecut is not None else "not reached"
        km_s = str(list(rec_mesh)) if rec_mesh is not None else "not reached"
        print(f"converge: target {target:.1e} eV/atom is {ach}; "
              f"noise floor {floor:.1e} eV/atom", flush=True)
        print(f"converge: recommend ecut={ec_s}, kmesh={km_s}", flush=True)
    return block
