"""run_flapw and run_nmr — the all-electron FLAPW / NMR drivers.

``run_flapw`` runs the muffin-tin FLAPW SCF (``flapw.crystal_scf_multi``) from an
``Input`` and reports the Γ eigenvalues + convergence. ``run_nmr`` computes an NMR
observable: ``nmr.task='efg'`` takes the all-electron electric field gradient per
site (and, per isotope, the quadrupolar coupling C_Q) through the same FLAPW
stack; ``nmr.task='shielding'`` takes the bare magnetic shielding tensor per site
through the plane-wave GIPAW analytic q→0 route (``kgeometry_nmr.sigma_shielding_dq``).

The FLAPW geometry is all-electron and takes a different form from the plane-wave
System: a cell in Bohr (rows = lattice vectors), fractional (coord, symbol) atoms,
and per-species muffin-tin radii in Å. The cell/positions come from ``inp.atoms``
(Å → Bohr), the k-mesh from ``inp.kpoints``, and everything else from the ``flapw``
block. FLAPW eigenvalues are referenced to the interstitial zero — compare
splittings, not absolute levels.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from gradwave.inputs import Input

if TYPE_CHECKING:
    import numpy as np


def _flapw_geometry(inp: Input) -> tuple[np.ndarray, list[tuple[tuple[float, float, float], str]]]:
    """Convert ``inp.atoms`` (Å) to the FLAPW geometry: a 3×3 cell in Bohr (rows
    = lattice vectors) and ``[(frac, symbol), ...]``."""
    import numpy as np

    from gradwave.constants import BOHR_ANG

    cell_bohr = np.asarray(inp.atoms.cell.array, dtype=float) / BOHR_ANG
    frac = inp.atoms.get_scaled_positions()
    syms = inp.atoms.get_chemical_symbols()
    atoms_list = [
        ((float(frac[i][0]), float(frac[i][1]), float(frac[i][2])), syms[i])
        for i in range(len(syms))
    ]
    return cell_bohr, atoms_list


def _run_crystal_scf(inp: Input, *, efg: bool, verbose: bool) -> tuple[Any, dict[str, Any]]:
    """Drive ``flapw.crystal_scf_multi`` from the Input, returning ``(bands, info)``."""
    from gradwave.flapw import crystal_scf_multi

    fp = inp.flapw
    cell_bohr, atoms_list = _flapw_geometry(inp)
    present = set(inp.atoms.get_chemical_symbols())
    missing = present - set(fp.radii)
    if missing:
        raise ValueError(
            f"flapw.radii is missing a muffin-tin radius (Å) for {sorted(missing)}; "
            f"give one per element under flapw.radii")
    kw: dict[str, Any] = {}
    if fp.los is not None:
        kw["los"] = fp.los
    if fp.val_e is not None:
        kw["val_e"] = fp.val_e
    if fp.core is not None:
        kw["core"] = fp.core
    if fp.el_override is not None:
        kw["el_override"] = fp.el_override
    if fp.kerker is not None:
        kw["kerker"] = fp.kerker
    return crystal_scf_multi(
        cell_bohr, atoms_list, dict(fp.radii), ecut=fp.ecut, lmax=fp.lmax,
        iters=fp.iters, tol=fp.tol, kmesh=tuple(inp.kpoints.mesh),
        smearing=fp.smearing, efg=efg, fullpot=fp.fullpot,
        use_symmetry=bool(inp.symmetry), fullpot_lmax=fp.fullpot_lmax,
        kworkers=fp.kworkers, verbose=verbose, **kw)


def _flapw_meta(inp: Input, bands: Any, info: dict[str, Any]) -> dict[str, Any]:
    """FLAPW SCF metadata shared by run_flapw and the EFG block: Γ eigenvalues,
    Fermi level, convergence story, and the muffin-tin/basis knobs."""
    rec = info.get("recorder")
    rsum = rec.summarize() if rec is not None else {}
    tol = float(inp.flapw.tol)
    d_span = rsum.get("d_span_eV")
    converged = d_span is not None and float(d_span) < tol
    if inp.flapw.fullpot and rsum.get("r_nsph") is not None:
        # the aspherical loop must also be gated (see flapw/scf.py convergence)
        converged = converged and float(rsum["r_nsph"]) < 0.05
    return {
        "eigenvalues_eV": bands.get("ev"),
        "band_span_eV": bands.get("span"),
        "e_fermi_eV": info.get("e_fermi"),
        "n_bands": info.get("nbands"),
        "converged": bool(converged),
        "convergence": rsum,
        "muffin_tin_radii_ang": dict(inp.flapw.radii),
        "fullpot": bool(inp.flapw.fullpot),
        "lmax": int(inp.flapw.lmax),
        "fullpot_lmax": int(inp.flapw.fullpot_lmax),
        "flapw_ecut": float(inp.flapw.ecut),
        "kmesh": list(inp.kpoints.mesh),
        "smearing_eV": float(inp.flapw.smearing),
    }


def run_flapw(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Run the all-electron muffin-tin FLAPW SCF (task: flapw) and return the
    ``flapw`` summary block (Γ eigenvalues, Fermi level, convergence)."""
    bands, info = _run_crystal_scf(inp, efg=False, verbose=verbose)
    return _flapw_meta(inp, bands, info)


def _resolve_isotopes(explicit: dict[str, str] | None,
                      symbols: list[str]) -> dict[str, str]:
    """species → isotope for the EFG C_Q. An explicit map is validated against
    the tabulated quadrupole moments; None auto-selects the first tabulated
    isotope for each element present (elements without one report V_zz/η only)."""
    from gradwave.flapw.nmr import NUCLEAR_Q

    if explicit is not None:
        for sp, iso in explicit.items():
            if iso not in NUCLEAR_Q:
                raise ValueError(
                    f"nmr.isotopes: no tabulated quadrupole moment for {iso!r} "
                    f"(species {sp}); known: {sorted(NUCLEAR_Q)}")
        return dict(explicit)
    by_elem: dict[str, str] = {}
    for iso in NUCLEAR_Q:
        el = re.sub(r"^\d+", "", iso)
        by_elem.setdefault(el, iso)
    return {sym: by_elem[sym] for sym in set(symbols) if sym in by_elem}


def _run_efg(inp: Input, verbose: bool) -> dict[str, Any]:
    from gradwave.flapw.nmr import quadrupolar_coupling

    bands, info = _run_crystal_scf(inp, efg=True, verbose=verbose)
    efg = info["efg"]
    symbols = inp.atoms.get_chemical_symbols()
    iso_map = _resolve_isotopes(inp.nmr.isotopes, symbols)
    sites: list[dict[str, Any]] = []
    for i, sym in enumerate(symbols):
        e = efg[f"a{i}"]
        v_zz, eta = float(e["V_zz"]), float(e["eta"])
        tensor = e["tensor"]
        entry: dict[str, Any] = {
            "site": i,
            "species": sym,
            "V_zz_eV_ang2": v_zz,
            "eta": eta,
            "V_zz_valence_eV_ang2": float(e["V_zz_valence"]),
            "eta_valence": float(e["eta_valence"]),
            "sphere_charge": float(e["sphere_charge"]),
            "tensor_eV_ang2": tensor.tolist() if hasattr(tensor, "tolist") else tensor,
        }
        iso = iso_map.get(sym)
        if iso is not None:
            cq = quadrupolar_coupling(v_zz, eta, iso)
            entry.update({
                "isotope": iso,
                "C_Q_MHz": cq["C_Q_MHz"],
                "abs_C_Q_MHz": cq["abs_C_Q_MHz"],
                "nu_Q_MHz": cq["nu_Q_MHz"],
                "Q_barn": cq["Q_barn"],
                "spin": cq["spin"],
            })
        sites.append(entry)
    block = _flapw_meta(inp, bands, info)
    block["observable"] = "efg"
    block["n_sites"] = len(sites)
    block["sites"] = sites
    return block


def _run_shielding(inp: Input, verbose: bool) -> dict[str, Any]:
    import numpy as np

    from gradwave.api.scf import run_scf
    from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq
    from gradwave.scf.loop import SCFResult

    res = run_scf(inp, verbose=verbose)
    if not isinstance(res, SCFResult):
        raise NotImplementedError(
            "nmr shielding (sigma_shielding_dq) is wired for the norm-conserving "
            "SCF only; noncollinear / USPP-PAW shielding is not yet plumbed here")
    sig = sigma_shielding_dq(res).detach().cpu().numpy()  # (nsite, 3, 3) ppm
    symbols = inp.atoms.get_chemical_symbols()
    sites: list[dict[str, Any]] = []
    for i in range(sig.shape[0]):
        s = np.asarray(sig[i], dtype=float)
        s_sym = 0.5 * (s + s.T)
        iso = float(np.trace(s_sym) / 3.0)
        eig = np.linalg.eigvalsh(s_sym)
        # Haeberlen ordering by ascending |σ_ii − σ_iso|: xx, yy, zz
        order = np.argsort(np.abs(eig - iso))
        s_xx, s_yy, s_zz = (float(eig[order[0]]), float(eig[order[1]]),
                            float(eig[order[2]]))
        denom = s_zz - iso
        sites.append({
            "site": i,
            "species": symbols[i] if i < len(symbols) else None,
            "sigma_iso_ppm": iso,
            "sigma_aniso_ppm": float(s_zz - 0.5 * (s_xx + s_yy)),
            "sigma_eta": float((s_yy - s_xx) / denom) if abs(denom) > 1e-12 else 0.0,
            "span_ppm": float(eig.max() - eig.min()),
            "tensor_ppm": s.tolist(),
        })
    return {
        "observable": "shielding",
        "method": "bare_analytic_dq",
        "n_sites": len(sites),
        "sites": sites,
        "ecut_eV": float(inp.ecut),
        "kmesh": list(inp.kpoints.mesh),
    }


def run_nmr(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Compute the NMR observable selected by ``inp.nmr.task`` (task: nmr).

    ``efg`` returns the all-electron FLAPW electric field gradient per site (V_zz,
    η, tensor) plus the quadrupolar coupling C_Q for the selected isotopes.
    ``shielding`` returns the bare plane-wave GIPAW shielding tensor per site
    (σ_iso, anisotropy, η). Returns the ``nmr`` summary block."""
    if inp.nmr.task == "efg":
        return _run_efg(inp, verbose)
    return _run_shielding(inp, verbose)
