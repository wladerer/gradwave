"""build_summary and the per-feature summary blocks."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gradwave.api._common import _OCC_TOL, SPIN_XC_REGISTRY, XC_REGISTRY, _gap, _get, build_xc
from gradwave.api.system import _is_uspp, _species_upfs
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.inputs import Input, VolumetricParams

if TYPE_CHECKING:

    from gradwave.api._common import SCFLike
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.scf.loop import SCFResult
    from gradwave.scf.noncollinear import NCResult
    from gradwave.scf.results import USPPResult

logger = logging.getLogger(__name__)


def _xc_label(inp: Input) -> str:
    """The functional name for the report: the hybrid label (pbe0/hse) when a
    hybrid is enabled, else the plain semilocal xc."""
    return inp.hybrid.name if inp.hybrid.enabled else inp.xc


def build_summary(res: SCFLike, inp: Input, task: str,
                  runtime_s: float | None = None,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The unified machine-readable summary for a task run."""
    from gradwave._version import __version__
    from gradwave.io.checkpoint import energies_eV_dict

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

    def _finite(x: Any) -> float | None:
        # the first iteration records dE = inf; bare Infinity is not
        # valid strict JSON, so non-finite maps to null
        return None if x is None or not math.isfinite(x) else float(x)

    trace: list[dict[str, Any]] = [
        {"iter": h["iter"], "free_energy_eV": float(h["free_energy"]),
         "dE_eV": _finite(h["dE"]), "drho": float(h["res"]),
         **({"t_s": round(float(h["t"]), 3)} if "t" in h else {}),
         # spinor energy-metric gate (scf.convergence: energy): the per-iteration
         # estimate and its charge / longitudinal / transverse decomposition,
         # recorded only on the noncollinear driver's own history (it has no
         # SCFRecorder). Absent on the density-gate path and the other formalisms.
         **({"energy_metric_eV": float(h["energy_metric_eV"]),
             "energy_metric_charge_eV": float(h["energy_metric_charge_eV"]),
             "energy_metric_longitudinal_eV": float(h["energy_metric_longitudinal_eV"]),
             "energy_metric_transverse_eV": float(h["energy_metric_transverse_eV"])}
            if h.get("energy_metric_eV") is not None else {})}
        for h in (_get(res, "history") or [])
    ]
    scf_block: dict[str, Any] = {
        "converged": bool(_get(res, "converged")),
        "n_iter": int(_get(res, "n_iter")),
        "fermi_eV": None if _get(res, "fermi") is None
        else float(_get(res, "fermi")),
        # smeared fixed-spin-moment runs carry the per-channel Fermi pair;
        # the constraining field is h = (μ↑ − μ↓)/2 = ∂F/∂M
        **({"fermi_spin_eV": [float(m) for m in _get(res, "fermi_spin")]}
           if _get(res, "fermi_spin") is not None else {}),
        "gap_eV": None if occ is None else _gap(eig.tolist(), occ.tolist(), nspin),
        "energies_eV": {
            **energies_eV_dict(e),
            "fock": float(getattr(e, "fock", 0.0)),
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
        "criterion": inp.scf.convergence,
        "final_dE_eV": _final.get("dE_eV"),
        "final_drho": _final.get("drho"),
        "etol_eV": float(inp.scf.etol),
        "rhotol": float(inp.scf.rhotol),
        "ratio_q": q,
        "warm_started": inp.restart is not None,
        # the per-iteration energy-metric value is surfaced under
        # scf_diagnostics.energy_metric_eV (the recorder block) on the collinear
        # and USPP/PAW paths; here we record the active criterion and threshold,
        # plus the final estimate for the spinor path (which has no recorder).
        **({"entol_eV": float(inp.scf.entol)}
           if inp.scf.convergence == "energy" else {}),
        **({"final_energy_metric_eV": _final.get("energy_metric_eV"),
            "final_energy_metric_charge_eV": _final.get("energy_metric_charge_eV"),
            "final_energy_metric_longitudinal_eV":
                _final.get("energy_metric_longitudinal_eV"),
            "final_energy_metric_transverse_eV":
                _final.get("energy_metric_transverse_eV")}
           if _final.get("energy_metric_eV") is not None else {}),
    }

    summary = {
        "code": {"name": "gradwave", "version": __version__,
                 "created": datetime.datetime.now().isoformat(timespec="seconds")},
        "task": task,
        "structure": _structure_block(inp),
        "parameters": {
            "formalism": "noncollinear" if is_ncmag else (
                "uspp/paw" if uspp else "nc"),
            "xc": _xc_label(inp),
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
                "precond": inp.scf.mixing.precond,
            },
            "n_electrons": float(system.n_electrons),
            "nbands": int(system.nbands),
            "fft_grid": list(system.grid.shape),
            "npw": int(system.spheres[0].npw),
            "pseudos": {s: inp.pseudo_map[s] for s in species},
            **({"hubbard": [
                {"species": m.species, "l": m.l, "U_eV": m.u, "J_eV": m.j}
                for m in inp.hubbard.manifolds]} if inp.hubbard.enabled else {}),
        },
        "scf": scf_block,
        "eigenvalues_eV": eig.tolist(),
        "occupations": [] if occ is None else occ.tolist(),
    }
    # SCF flight-recorder diagnostics (scf.recorder): a compact block of the
    # per-iteration convergence health — heuristic tags, long-wavelength
    # (sloshing) residual fraction, band-reordering count, per-iteration wall
    # time. Present for the recording drivers (NC / USPP-PAW collinear); absent
    # for the noncollinear path, which does not record.
    _rec = _get(res, "recorder")
    if _rec is not None and getattr(_rec, "iters", None):
        summary["scf_diagnostics"] = _rec.summarize()
    if runtime_s is not None:
        summary["runtime_s"] = round(float(runtime_s), 2)
    if extra:
        summary.update(extra)
    return summary


def _bands_reference(res: SCFLike) -> float:
    """Band-plot energy zero: the Fermi level for a metal (any partially filled
    state), else the valence-band maximum. Mirrors postscf.bands.band_structure
    so the USPP path reports the same reference the NC path does."""
    import numpy as np

    occ = np.asarray(_get(res, "occupations"), dtype=float)
    eig = np.asarray(_get(res, "eigenvalues"), dtype=float)
    nspin = int(_get(res, "nspin", 1) or 1)
    g = 2.0 if nspin == 1 else 1.0
    is_metal = bool(((occ > _OCC_TOL) & (occ < g - _OCC_TOL)).any())
    if is_metal:
        return float(_get(res, "fermi"))
    return float(eig[occ > _OCC_TOL].max())


def _bands_uspp_block(inp: Input, res: SCFLike, verbose: bool) -> dict[str, Any]:
    """USPP/PAW band structure along an ASE k-path via postscf.uspp_bands. The NC
    ``bands_along_ase_path`` builds the path and returns a BandStructure; the
    USPP solver instead takes an explicit k-list and returns bare eigenvalues, so
    the ASE path (k-points, axis, special-point labels) and the energy reference
    are assembled here to match the NC bands block shape."""
    import numpy as np

    from gradwave.postscf.uspp_bands import bands_uspp

    bp = inp.atoms.cell.bandpath(path=inp.bands.path or None,
                                 npoints=inp.bands.npoints)
    kpts = np.asarray(bp.kpts, dtype=float)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    xc = SPIN_XC_REGISTRY[inp.xc]() if inp.nspin == 2 else XC_REGISTRY[inp.xc]()
    if verbose:
        print(f"bands (USPP/PAW): {len(kpts)} k-points along the path", flush=True)
    # this block only ever runs for a USPP/PAW result (see _bands_uspp_block's
    # caller); res's static SCFLike type is wider than what's true here.
    eig = bands_uspp(cast("USPPResult", res), xc, kpts,
                     nbands=inp.bands.nbands).detach().cpu().numpy()
    bands: dict[str, Any] = {
        "kpts_frac": kpts.tolist(),
        "x": np.asarray(x).tolist(),
        "labels": list(zip(xticks.tolist(), list(xlabels), strict=True)),
        "eigenvalues_eV": eig.tolist(),
        "reference_eV": _bands_reference(res),
    }
    return {"bands": bands}


def _bands_extra(inp: Input, res: SCFLike, verbose: bool) -> dict[str, Any]:
    from gradwave.postscf.bands import bands_along_ase_path

    _species, upfs, _soa = _species_upfs(inp)
    if _is_uspp(upfs):
        return _bands_uspp_block(inp, res, verbose)

    # bands_along_ase_path declares res: SCFResult; this branch is reached
    # only when _is_uspp(upfs) is False above, so res is the norm-conserving
    # SCFResult (the USPP/PAW and noncollinear formalisms route to
    # _bands_uspp_block / a different task entirely).
    bs = bands_along_ase_path(
        cast("SCFResult", res), inp.atoms, path=inp.bands.path,
        npoints=inp.bands.npoints, nbands=inp.bands.nbands, verbose=verbose,
    )
    # bands_along_ase_path always populates x/labels (the BandStructure
    # dataclass is shared with the lower-level band_structure(), which
    # leaves them unset — see postscf/bands.py).
    assert bs.x is not None
    assert bs.labels is not None
    bands: dict[str, Any] = {
        "kpts_frac": bs.kpts_frac.tolist(),
        "x": bs.x.tolist(),
        "labels": bs.labels,
        "eigenvalues_eV": bs.eigenvalues.tolist(),
        "reference_eV": bs.reference,
    }
    if inp.bands.irreps:
        import numpy as np

        from gradwave.postscf.irreps import band_irreps

        cache: dict[Any, Any] = {}
        ann: list[dict[str, Any]] = []
        for xt, lab in bs.labels:
            idx = int(np.argmin(np.abs(np.asarray(bs.x) - xt)))
            kf_exact = bs.kpts_frac[idx]  # full precision — rounding here
            # shrinks the little group at threshold (1/3 vs 0.33333333)
            key = tuple(np.round(kf_exact, 8))
            if key not in cache:
                # same "reached only for the norm-conserving SCFResult" guard
                # as the bands_along_ase_path call above.
                cache[key] = band_irreps(
                    cast("SCFResult", res), kf_exact, nbands=inp.bands.nbands)
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


def _optics_extra(inp: Input, res: SCFLike, verbose: bool) -> dict[str, Any]:
    from gradwave.postscf.optics import optical_epsilon

    _species, upfs, _soa = _species_upfs(inp)
    if _is_uspp(upfs):
        raise NotImplementedError("task: optics is norm-conserving only")

    om, eps1, eps2, alpha, info = optical_epsilon(
        cast("SCFResult", res),
        omega_max=inp.optics.omega_max, n_omega=inp.optics.n_omega,
        eta=inp.optics.eta, n_extra_bands=inp.optics.n_extra_bands,
        velocity=inp.optics.velocity, local_fields=inp.optics.local_fields,
        dk=inp.optics.dk, verbose=verbose,
    )
    optics: dict[str, Any] = {
        "omega_eV": om.tolist(),
        "eps1": eps1.tolist(),
        "eps2": eps2.tolist(),
        "absorption_inv_cm": alpha.tolist(),
        "eta_eV": inp.optics.eta,
        "n_bands": info["n_bands"],
        "n_occ": info["n_occ"],
        "eps_static": info["eps_static"],
        "velocity": info["velocity"],
        "local_fields": info["local_fields"],
        "eps2_tensor": info["eps2_tensor"],  # (xx, yy, zz) diagonal components
        "eps1_tensor": info["eps1_tensor"],
    }
    if info["local_fields"]:
        optics["eps1_ip"] = info["eps1_ip"]  # IP values for comparison
        optics["eps2_ip"] = info["eps2_ip"]
    return {"optics": optics}


def _error_estimate_xc(inp: Input) -> NoncollinearXC | SpinXC | XCFunctional:
    """The functional object the post-SCF estimators need to rebuild operators.

    Non-collinear runs need a ``NoncollinearXC`` (the exchange field enters the
    spinor Hamiltonian); collinear nspin=2 needs the spin functional; nspin=1 the
    plain one.
    """
    return build_xc(inp)


def _error_estimate_block(res: SCFLike, inp: Input) -> dict[str, Any]:
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
    # postscf.discretization_error / convergence_error declare `res:
    # SCFResult`, but (per this function's own docstring) duck-type over
    # every formalism via getattr, exactly like `_get` above — the
    # annotations there are simply narrower than the runtime contract.
    res_nc = cast("SCFResult", res)
    # a non-collinear SCF always runs with a real smearing scheme (spinor bands
    # hold one electron); a "none" request maps to gaussian, as the run does.
    nc_scheme = ("gaussian" if inp.smearing.type == "none" else inp.smearing.type)
    dens_kw: dict[str, Any] = (
        dict(smearing=nc_scheme, width=inp.smearing.width) if is_nc else {})
    try:
        err = estimate_density_error(res_nc, xc=xc, **dens_kw)
    except NotImplementedError as e:
        return {"available": False, "reason": str(e)}

    drho = err.drho
    free_e = float(_get(res, "energies").free_energy)
    block: dict[str, Any] = {
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
            # force_ok is False whenever is_nc, so xc is never a NoncollinearXC
            # here — same "narrower annotation than the force_ok-guarded
            # runtime contract" seam as res_nc above.
            fe = estimate_force_error(
                res_nc, err, xc=cast("XCFunctional | SpinXC | None", xc)
            ).norm(dim=1)
            block["force_error_max_eV_ang"] = float(fe.max())
            block["force_error_rms_eV_ang"] = float((fe ** 2).mean().sqrt())
        except NotImplementedError as exc:
            block["force_error"] = {"available": False, "reason": str(exc)}
    # band-gap error (insulators; NC/USPP/PAW now covered, skipped for metals).
    try:
        eig_kw: dict[str, Any] = (
            dict(smearing=nc_scheme, width=inp.smearing.width) if is_nc else {})
        eige = estimate_eigenvalue_error(res_nc, ecut_large=err.ecut_large, xc=xc,
                                         **eig_kw)
        gap_kw: dict[str, Any] = {}
        if is_nc:
            # NCResult carries no occupations; recompute (degeneracy 1) and set
            # the metal/insulator threshold to half of one-electron filling.
            gap_kw = dict(occupations=_nc_occupations(cast("NCResult", res),
                                                      nc_scheme,
                                                      inp.smearing.width),
                          occ_threshold=0.5)
        gap = estimate_gap_error(res_nc, eige, **gap_kw)
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
        # scf_xc is never a NoncollinearXC (it's None whenever is_nc) —
        # narrower annotation than the guarded runtime contract, as above.
        scfe = estimate_scf_error(res_nc, cast("XCFunctional | None", scf_xc))
        sc: dict[str, Any] = {
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
            res_nc, scheme=nc_scheme if is_nc else inp.smearing.type,
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


def _nc_occupations(res: NCResult, scheme: str, width: float) -> list[Any]:
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


def _pdos_summary_block(res: SCFLike, inp: Input) -> dict[str, Any]:
    """Löwdin projected-DOS block for the summary JSON. Returns a graceful
    ``{'available': False, ...}`` when the pseudopotentials omit PP_PSWFC."""
    from gradwave.postscf.pdos import projected_dos
    p = inp.projections
    try:
        # projected_dos declares SCFResult | USPPResult and raises
        # NotImplementedError for anything else (NCResult/USPPNCResult) via
        # its own _unpack_result — caught below, so the wider SCFLike here
        # is a safe runtime seam, not a real type mismatch.
        return projected_dos(cast("SCFResult | USPPResult", res),
                              group_by=p.group_by, width=p.width,
                              npoints=p.npoints).to_dict()
    except (ValueError, NotImplementedError) as err:
        return {"available": False, "reason": str(err)}


def _cohp_summary_block(res: SCFLike, inp: Input) -> dict[str, Any]:
    """Crystal Orbital Hamilton Population block for the summary JSON, computed
    alongside the PDOS when ``projections.cohp`` is enabled. Returns a graceful
    ``{'available': False, ...}`` when the pseudopotentials omit PP_PSWFC or the
    formalism is out of coverage."""
    from gradwave.postscf.cohp import cohp
    c = inp.projections.cohp
    pairs = None if c.pairs is None else [tuple(p) for p in c.pairs]
    try:
        # cohp declares SCFResult | USPPResult and raises NotImplementedError
        # for anything else via its own _unpack_result — caught below (the
        # noncollinear/SOC cohp_noncollinear/cohp_soc entry points aren't
        # wired into this summary path yet), so the wider SCFLike here is a
        # safe runtime seam, not a real type mismatch.
        block = cohp(cast("SCFResult | USPPResult", res), pairs=pairs,
                     rcut=c.rcut, width=c.width, npoints=c.npoints).to_dict()
        block["available"] = True
        return block
    except (ValueError, NotImplementedError) as err:
        return {"available": False, "reason": str(err)}


def _write_volumetric(
    res: SCFLike, spec: VolumetricParams, outdir: Path, verbose: bool
) -> dict[str, Any]:
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


def _base_summary(inp: Input, task: str) -> dict[str, Any]:
    """The lightweight summary scaffold shared by tasks that carry no
    SCFResult (relax, magnetism, eos, elastic, phonons). SCF-derived tasks
    use build_summary() instead. Callers append their per-task result block
    and a trailing "runtime_s" so the serialized key order stays
    code/task/structure/parameters/<block>/runtime_s."""
    from gradwave._version import __version__

    return {
        "code": {"name": "gradwave", "version": __version__,
                 "created": datetime.datetime.now().isoformat(
                     timespec="seconds")},
        "task": task,
        "structure": _structure_block(inp),
        "parameters": _parameters_block(inp),
    }


def _flapw_base_summary(inp: Input, task: str) -> dict[str, Any]:
    """Summary scaffold for the all-electron FLAPW / NMR tasks (flapw, nmr). Unlike
    _base_summary it does NOT resolve pseudopotentials (FLAPW/EFG is all-electron
    and carries none), so it never touches _species_upfs. The driver appends its
    result block under summary[task] and a trailing runtime_s."""
    from gradwave._version import __version__

    return {
        "code": {"name": "gradwave", "version": __version__,
                 "created": datetime.datetime.now().isoformat(timespec="seconds")},
        "task": task,
        "structure": _structure_block(inp),
        "parameters": _flapw_parameters_block(inp),
    }


def _flapw_parameters_block(inp: Input) -> dict[str, Any]:
    """Parameters block for the FLAPW/NMR tasks. The all-electron EFG/flapw branch
    reports the muffin-tin/basis knobs (LDA internally); the plane-wave shielding
    branch reports the PW SCF knobs."""
    import math

    block: dict[str, Any] = {
        "kmesh": list(inp.kpoints.mesh),
        "nk_total": int(math.prod(inp.kpoints.mesh)),
        "symmetry": bool(inp.symmetry),
    }
    if inp.task == "nmr" and inp.nmr.task == "shielding":
        block["formalism"] = "plane-wave GIPAW"
        block["xc"] = _xc_label(inp)
        block["ecut_eV"] = float(inp.ecut)
        block["nmr_observable"] = "shielding"
    else:
        block["formalism"] = "all-electron FLAPW"
        block["xc"] = "lda"  # the FLAPW stack is LDA (Slater + PW92) internally
        block["flapw_ecut"] = float(inp.flapw.ecut)
        block["lmax"] = int(inp.flapw.lmax)
        block["fullpot"] = bool(inp.flapw.fullpot)
        block["fullpot_lmax"] = int(inp.flapw.fullpot_lmax)
        block["muffin_tin_radii_ang"] = dict(inp.flapw.radii)
        block["smearing_eV"] = float(inp.flapw.smearing)
        if inp.task == "nmr":
            block["nmr_observable"] = "efg"
    return block


def _structure_block(inp: Input) -> dict[str, Any]:
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
        if ds is None:
            # a degenerate/near-singular cell; drop the field rather than
            # swallow an unrelated bug below
            return block
        block["spacegroup"] = f"{ds.international} ({ds.number})"
        block["pointgroup"] = ds.pointgroup
        block["n_symops"] = len(ds.rotations)
    except (TypeError, spglib.SpglibError):
        pass
    return block


def _parameters_block(inp: Input) -> dict[str, Any]:
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
        "xc": _xc_label(inp),
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
            "precond": inp.scf.mixing.precond,
        },
        "pseudos": {s: inp.pseudo_map[s] for s in species},
    }
