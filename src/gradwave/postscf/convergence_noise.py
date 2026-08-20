"""Statistical noise floor of the total energy at fixed (ecut, k-mesh).

The systematic estimators (``postscf.discretization_error``,
``postscf.convergence_error``) bound how far a converged run sits from the
infinite-basis / dense-k / fully self-consistent limit. Below those systematic
curves sits a STATISTICAL floor: at fixed convergence parameters the energy is
not a smooth function of geometry — the finite FFT grid and the sharp-Ecut
G-sphere break exact translation invariance (the eggbox effect), so physically
equivalent configurations scatter by a small, effectively random amount.
Convergence curves below this floor are discreteness noise, not signal
(Janssen, Makarov, Hickel, Shapeev, Neugebauer, "Automated optimization and
uncertainty quantification of convergence parameters in plane wave DFT
calculations", npj Comput. Mater. 10 (2024), DOI 10.1038/s41524-024-01388-2).

Probe. Rigid translations of ALL atoms by incommensurate fractions of a grid
spacing leave the physics exactly invariant (the invariance
``tests/integration/test_metamorphic_invariance.py`` exercises); the sample
standard deviation of the converged energy across a handful of such
translations IS the grid-discreteness noise at these parameters. Each probe
warm-starts from the base run (density and, when the meshes match, orbitals),
so it converges in a few SCF iterations. An optional secondary probe jitters
the volume by a few tenths of a percent on the SAME (pinned) FFT grid — the
situation an EOS scan samples — and reads the noise off the residual of a
quadratic fit in V, since E(V) carries real curvature the translation probe
does not.

Run the probes at SCF tolerances well below the floor being measured (defaults
etol=1e-9 eV, rhotol=1e-8): otherwise the estimate conflates the SCF tail with
grid noise. ``postscf.convergence_error.estimate_scf_error`` bounds that tail
if in doubt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gradwave.core.xc.base import XCFunctional
from gradwave.scf.loop import SCFResult, System, scf, setup_system

__all__ = ["NoiseFloor", "estimate_noise_floor", "translation_shifts"]


# Additive low-discrepancy (Kronecker/R3) generator vector: (1/g, 1/g^2,
# 1/g^3) with g the real root of x^4 = x + 1 — the standard 3-D quasirandom
# sequence. Irrational per axis, so shifts are grid-incommensurate fractions.
_R3 = (0.8191725133961645, 0.6710436067037893, 0.5497004779019703)


@dataclass
class NoiseFloor:
    """Measured statistical (discreteness) noise floor at fixed (ecut, kmesh).

    ``sigma_e`` is the headline: the sample standard deviation (ddof=1) of the
    converged energy over the rigid-translation probes, in eV per cell;
    ``sigma_e_per_atom`` divides by the atom count. A convergence target below
    ``sigma_e_per_atom`` is unachievable at these parameters — refining
    tolerances only resolves the noise, not the physics.
    """

    sigma_e: float                     # eV/cell, std of E over translation probes
    sigma_e_per_atom: float            # eV/atom
    spread_e: float                    # eV/cell, max-min over the probes
    e_base: float                      # eV/cell, base-run energy (same kind)
    energies: list[float]              # eV/cell, per-probe converged energies
    shifts_frac: list[list[float]]     # per-probe rigid shift, cell-fractional
    n_translations: int
    energy_kind: str                   # EnergyBreakdown attribute sampled
    all_converged: bool
    sigma_f: float | None = None       # eV/Å, RMS per-component force scatter
    sigma_e_volume: float | None = None  # eV/cell, quadratic-fit residual std
    volume_scales: list[float] | None = None
    volume_energies: list[float] | None = None


def translation_shifts(n: int, grid_shape: tuple[int, int, int],
                       seed: int = 0) -> np.ndarray:
    """(n, 3) rigid shifts, cell-fractional, each an incommensurate fraction of
    one grid spacing per axis.

    Shift ``j`` is ``frac(offset + (j+1) * R3) / grid_shape`` — a seeded random
    offset plus the plastic-number Kronecker sequence, scaled so every
    component stays inside one grid cell (the eggbox period). Deterministic for
    a given ``(n, grid_shape, seed)``.
    """
    rng = np.random.default_rng(seed)
    offset = rng.random(3)
    j = np.arange(1, n + 1, dtype=float)[:, None]
    alpha = np.mod(offset[None, :] + j * np.asarray(_R3)[None, :], 1.0)
    return alpha / np.asarray(grid_shape, dtype=float)[None, :]


def _probe_scf_kwargs(res: SCFResult,
                      scf_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """SCF kwargs for the probes: base run's smearing, tight tolerances,
    quiet; user overrides win."""
    kw: dict[str, Any] = dict(
        smearing=res.smearing, width=res.width,
        etol=1e-9, rhotol=1e-8, verbose=False,
    )
    if scf_kwargs:
        kw.update(scf_kwargs)
    return kw


def estimate_noise_floor(
    res: SCFResult,
    xc: XCFunctional,
    *,
    kmesh: tuple[int, ...],
    kshift: tuple[int, ...] = (0, 0, 0),
    n_translations: int = 4,
    seed: int = 0,
    energy: str = "free_energy",
    forces: bool = False,
    n_volumes: int = 0,
    volume_jitter: float = 0.003,
    scf_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
) -> NoiseFloor:
    """Measure the statistical energy-noise floor of a converged SCF.

    Re-converges the SAME system under ``n_translations`` small rigid
    translations of all atoms (incommensurate fractions of a grid spacing, see
    ``translation_shifts``) on the base run's FFT grid, warm-starting each
    probe from ``res``. The scatter of the converged energies is the
    grid-discreteness noise at this (ecut, kmesh).

    Parameters
    ----------
    res : SCFResult
        The converged base run (norm-conserving collinear, nspin=1).
    xc : XCFunctional
        The functional the base run used.
    kmesh, kshift : tuple
        The base run's k-mesh spec — ``System`` does not retain it, so the
        caller must supply it to rebuild the translated systems.
    n_translations : int
        Number of rigid-translation probes (4-6 is enough: the estimate is a
        scale, not a tail quantile).
    energy : str
        ``EnergyBreakdown`` attribute to sample (default ``free_energy``).
    forces : bool
        Also report ``sigma_f``, the RMS per-component scatter of the analytic
        forces across the probes (rigid translation leaves forces invariant,
        so the scatter is pure eggbox force noise).
    n_volumes : int
        If >= 5, run the secondary volume-jitter probe: ``n_volumes`` scales
        evenly spaced in ``1 ± volume_jitter`` on the pinned grid, quadratic
        fit in V, residual std (ddof = n-3) reported as ``sigma_e_volume``.
    scf_kwargs : dict, optional
        Overrides for the probe SCF calls (tolerances, mixing, ...). Defaults
        run the base smearing at etol=1e-9 / rhotol=1e-8.
    """
    if getattr(res, "formalism", "nc") != "nc" or int(getattr(res, "nspin", 1)) != 1:
        raise NotImplementedError(
            "noise-floor probes are implemented for the norm-conserving "
            "collinear (nspin=1) SCF only")
    if n_translations < 2:
        raise ValueError("need at least 2 translation probes for a std")
    if 0 < n_volumes < 5:
        raise ValueError("the volume probe needs n_volumes >= 5 (quadratic "
                         "fit leaves n-3 residual dof)")

    system = res.system
    grid = system.grid
    cell = np.asarray(grid.cell, dtype=float)
    pos0 = system.positions.detach().cpu().numpy()
    natoms = pos0.shape[0]
    tot_charge = float(system.charges.sum()) - float(system.n_electrons)
    use_symmetry = system.sym is not None
    kw = _probe_scf_kwargs(res, scf_kwargs)

    def _build(positions: np.ndarray, cell_p: np.ndarray) -> System:
        return setup_system(
            cell_p, positions, system.species_of_atom, system.upfs,
            ecut=float(system.ecut), kmesh=tuple(kmesh), kshift=tuple(kshift),
            nbands=system.nbands, use_symmetry=use_symmetry,
            fft_shape=tuple(grid.shape), tot_charge=tot_charge)

    shifts = translation_shifts(n_translations, tuple(grid.shape), seed=seed)
    energies: list[float] = []
    f_probes: list[np.ndarray] = []
    all_conv = True
    for i, frac in enumerate(shifts):
        sys_i = _build(pos0 + frac @ cell, cell)
        res_i = scf(sys_i, xc, start_from=res, **kw)
        all_conv = all_conv and bool(res_i.converged)
        e_i = float(getattr(res_i.energies, energy))
        energies.append(e_i)
        if verbose:
            print(f"  noise probe {i + 1}/{n_translations}: "
                  f"E={e_i:+.8f} eV ({res_i.n_iter} iters)", flush=True)
        if forces:
            # lazy: the force probe is opt-in, and the import is cached after
            # the first iteration
            from gradwave.postscf.forces import forces as compute_forces

            fxc = xc if system.rho_core is not None else None
            f_probes.append(compute_forces(res_i, xc=fxc).detach().cpu().numpy())

    e_arr = np.asarray(energies)
    sigma_e = float(np.std(e_arr, ddof=1))
    sigma_f = None
    if forces:
        # per-component std across probes, RMS over atoms/components
        sigma_f = float(np.sqrt(np.mean(np.var(np.stack(f_probes), axis=0,
                                               ddof=1))))

    sigma_e_volume = None
    v_scales = v_energies = None
    if n_volumes:
        scales = 1.0 + volume_jitter * np.linspace(-1.0, 1.0, n_volumes)
        frac0 = pos0 @ np.linalg.inv(cell)
        vols, ev = [], []
        for s in scales:
            cell_s = cell * s ** (1.0 / 3.0)
            sys_s = _build(frac0 @ cell_s, cell_s)
            res_s = scf(sys_s, xc, start_from=res, **kw)
            all_conv = all_conv and bool(res_s.converged)
            vols.append(float(abs(np.linalg.det(cell_s))))
            ev.append(float(getattr(res_s.energies, energy)))
        resid = np.polyval(np.polyfit(vols, ev, 2), vols) - np.asarray(ev)
        sigma_e_volume = float(np.sqrt(np.sum(resid ** 2) / (n_volumes - 3)))
        v_scales, v_energies = [float(s) for s in scales], ev

    return NoiseFloor(
        sigma_e=sigma_e,
        sigma_e_per_atom=sigma_e / natoms,
        spread_e=max(energies) - min(energies),
        e_base=float(getattr(res.energies, energy)),
        energies=energies,
        shifts_frac=[[float(x) for x in row] for row in shifts],
        n_translations=n_translations,
        energy_kind=energy,
        all_converged=all_conv,
        sigma_f=sigma_f,
        sigma_e_volume=sigma_e_volume,
        volume_scales=v_scales,
        volume_energies=v_energies,
    )
