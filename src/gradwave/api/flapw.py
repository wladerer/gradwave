"""run_flapw and run_nmr — the all-electron FLAPW / NMR drivers.

``run_flapw`` runs the muffin-tin FLAPW SCF (``flapw.crystal_scf_multi``) from an
``Input`` and reports the Γ eigenvalues + convergence. ``run_nmr`` computes an NMR
observable: ``nmr.task='efg'`` takes the all-electron electric field gradient per
site (and, per isotope, the quadrupolar coupling C_Q) through the same FLAPW
stack; ``nmr.task='shielding'`` takes the magnetic shielding tensor per site
through the plane-wave GIPAW route — either the bare valence term alone
(``kgeometry_nmr.sigma_shielding_dq``, norm-conserving) or the full absolute
σ = σ_bare + σ_core + σ_dia_aug + σ_para_aug (``sigma_shielding_gipaw``, all-PAW),
selected by ``nmr.shielding_level`` (default ``auto``: gipaw for PAW pseudos).
With ``nmr.sigma_ref`` each site also reports the chemical shift δ_iso.

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
    # The validated EFG anion recipe (l=1 HELO + l=0 2s→2p) as the base, with any
    # explicit per-species los/el_override overriding it (flapw.basis.merge_basis).
    if fp.efg_anion_basis is not None:
        from gradwave.flapw.basis import efg_anion_basis, merge_basis
        base = efg_anion_basis(list(fp.efg_anion_basis))
        merged = merge_basis(base, {"los": fp.los, "el_override": fp.el_override})
        kw["los"] = merged["los"]
        if merged.get("el_override"):
            kw["el_override"] = merged["el_override"]
    else:
        if fp.los is not None:
            kw["los"] = fp.los
        if fp.el_override is not None:
            kw["el_override"] = fp.el_override
    if fp.val_e is not None:
        kw["val_e"] = fp.val_e
    if fp.core is not None:
        kw["core"] = fp.core
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
        # initial-state core levels (eV, referenced to the interstitial zero —
        # only the within-cell shifts are physical) + the same-element shifts
        "core_levels": info.get("core_levels"),
        "core_level_shifts": info.get("core_level_shifts"),
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


def _kmesh_symmetry_broken(atoms: Any, mesh: Any) -> tuple[int, int]:
    """Count crystal point-group ops that a Γ-centred MP mesh ``mesh`` does NOT respect.

    A Γ-centred grid with subdivisions ``n`` is invariant under a reciprocal rotation ``g``
    iff ``n_i·g_ij/n_j ∈ ℤ`` ∀ i,j — so an *un-equal* subdivision on symmetry-equivalent axes
    (e.g. [3,3,2] on a cubic crystal, [2,2,4] on a hexagonal one) breaks the point group, and
    symmetry-equivalent sites then sample the BZ inequivalently → split NMR parameters. Returns
    ``(n_broken, n_total)``; ``(0, ·)`` = suitable. Requires spglib; returns ``(0, 0)`` without it.
    """
    try:
        import numpy as np
        import spglib
    except ImportError:
        return (0, 0)
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    sym = spglib.get_symmetry(cell, symprec=1e-4)
    if sym is None:
        return (0, 0)
    n = np.asarray(mesh, dtype=int)
    broken = 0
    for R in sym["rotations"]:
        g = np.rint(np.linalg.inv(R).T).astype(int)              # reciprocal rotation
        if not np.all((n[:, None] * g) % n[None, :] == 0):
            broken += 1
    return (broken, len(sym["rotations"]))


def _equivalent_site_groups(atoms: Any) -> list[list[int]]:
    """Symmetry-equivalent atom orbits (list of index lists) via spglib; [] if unavailable."""
    try:
        import spglib
    except ImportError:
        return []
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell, symprec=1e-4)
    if ds is None:
        return []
    eq = ds["equivalent_atoms"] if isinstance(ds, dict) else ds.equivalent_atoms
    groups: dict[int, list[int]] = {}
    for i, e in enumerate(eq):
        groups.setdefault(int(e), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _haeberlen_site(tensor: np.ndarray) -> dict[str, Any]:
    """The Haeberlen CSA quantities of one shielding tensor (3×3, ppm): σ_iso,
    anisotropy, asymmetry η, span, and the raw tensor. Symmetrized before the
    eigenanalysis; the principal values are ordered by ascending |σ_ii − σ_iso|
    (xx, yy, zz)."""
    import numpy as np

    s = np.asarray(tensor, dtype=float)
    s_sym = 0.5 * (s + s.T)
    iso = float(np.trace(s_sym) / 3.0)
    eig = np.linalg.eigvalsh(s_sym)
    order = np.argsort(np.abs(eig - iso))
    s_xx, s_yy, s_zz = (float(eig[order[0]]), float(eig[order[1]]),
                        float(eig[order[2]]))
    denom = s_zz - iso
    return {
        "sigma_iso_ppm": iso,
        "sigma_aniso_ppm": float(s_zz - 0.5 * (s_xx + s_yy)),
        "sigma_eta": float((s_yy - s_xx) / denom) if abs(denom) > 1e-12 else 0.0,
        "span_ppm": float(eig.max() - eig.min()),
        "tensor_ppm": s.tolist(),
    }


def _warn_kmesh_symmetry(inp: Input) -> None:
    import warnings

    broken, total = _kmesh_symmetry_broken(inp.atoms, inp.kpoints.mesh)
    if broken:
        warnings.warn(
            f"NMR shielding: the k-mesh {list(inp.kpoints.mesh)} breaks {broken} of {total} "
            f"crystal point-group operations, so symmetry-equivalent sites can come out "
            f"inequivalent (split) — use equal subdivisions on symmetry-equivalent axes "
            f"(e.g. a cubic n×n×n, a hexagonal n_a=n_b). ",
            stacklevel=2)


def _warn_equivalent_split(inp: Input, sites: list[dict[str, Any]]) -> None:
    """Post-hoc red flag: symmetry-equivalent sites must give equal σ_iso; a
    split (unsuitable/coarse k-mesh, unconverged ecut, or a broken structure)
    means the numbers are not trustworthy."""
    import warnings

    for grp in _equivalent_site_groups(inp.atoms):
        vals = [sites[i]["sigma_iso_ppm"] for i in grp]
        spread = max(vals) - min(vals)
        if spread > 1.0:                    # ppm; equivalent sites should agree to well under this
            sp = sites[grp[0]]["species"]
            warnings.warn(
                f"NMR shielding: symmetry-equivalent {sp} sites {grp} give σ_iso spread "
                f"{spread:.1f} ppm (should be ~0) — the result is not converged/consistent; "
                f"check the structure (bond lengths), the k-mesh suitability, and ecut/k "
                f"convergence before trusting the numbers.",
                stacklevel=2)


def _apply_shift_reference(
    block: dict[str, Any], sigma_ref: dict[str, float] | None
) -> None:
    """Add ``delta_iso_ppm = σ_ref − σ_iso`` to each site whose species or
    isotope is keyed in ``sigma_ref`` (ppm). A no-op when ``sigma_ref`` is None
    or a site has no matching key, so unreferenced runs stay byte-identical."""
    if not sigma_ref:
        return
    for site in block["sites"]:
        ref: float | None = None
        for key in (site.get("isotope"), site.get("species")):
            if key is not None and key in sigma_ref:
                ref = float(sigma_ref[key])
                break
        if ref is not None:
            site["delta_iso_ppm"] = ref - float(site["sigma_iso_ppm"])
    block["sigma_ref_ppm"] = dict(sigma_ref)


def _shielding_bare_block(inp: Input, res: Any) -> dict[str, Any]:
    from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq

    sig = sigma_shielding_dq(
        res, chunk_k=inp.nmr.chunk_k).detach().cpu().numpy()  # (nsite, 3, 3) ppm
    symbols = inp.atoms.get_chemical_symbols()
    sites: list[dict[str, Any]] = [
        {
            "site": i,
            "species": symbols[i] if i < len(symbols) else None,
            **_haeberlen_site(sig[i]),
        }
        for i in range(sig.shape[0])
    ]
    _warn_equivalent_split(inp, sites)
    return {
        "observable": "shielding",
        "method": "bare_analytic_dq",
        "n_sites": len(sites),
        "sites": sites,
        "ecut_eV": float(inp.ecut),
        "kmesh": list(inp.kpoints.mesh),
    }


def _shielding_gipaw_block(inp: Input, res: Any) -> dict[str, Any]:
    import numpy as np

    from gradwave.api._common import build_xc
    from gradwave.api.system import _as_paws, _species_upfs
    from gradwave.postscf.kgeometry_nmr import (
        build_uspp_response_ctx,
        sigma_shielding_gipaw,
    )

    _species, upfs, _soa = _species_upfs(inp)
    paws = _as_paws(upfs)
    ctx = build_uspp_response_ctx(res, build_xc(inp))
    out = {k: v.detach().cpu().numpy()
           for k, v in sigma_shielding_gipaw(
               res, ctx, paws, chunk_k=inp.nmr.chunk_k).items()}
    symbols = inp.atoms.get_chemical_symbols()
    sites: list[dict[str, Any]] = []
    for i in range(out["total"].shape[0]):
        entry: dict[str, Any] = {
            "site": i,
            "species": symbols[i] if i < len(symbols) else None,
            "sigma_bare_ppm": float(np.trace(out["bare"][i]) / 3.0),
            "sigma_core_ppm": float(np.trace(out["core"][i]) / 3.0),
            "sigma_dia_aug_ppm": float(np.trace(out["dia_aug"][i]) / 3.0),
            "sigma_para_aug_ppm": float(np.trace(out["para_aug"][i]) / 3.0),
        }
        entry.update(_haeberlen_site(out["total"][i]))
        sites.append(entry)
    _warn_equivalent_split(inp, sites)
    return {
        "observable": "shielding",
        "method": "gipaw_absolute",
        "n_sites": len(sites),
        "sites": sites,
        "ecut_eV": float(inp.ecut),
        "kmesh": list(inp.kpoints.mesh),
    }


def _resolve_pw_efg(want: bool | str, is_paw: bool) -> bool:
    """Whether to compute the PW/PAW EFG in the shielding task. ``'auto'`` → on for
    a PAW ground state; ``True`` requires PAW (else a clear error); ``False`` off."""
    if want == "auto":
        return is_paw
    if want and not is_paw:
        raise ValueError(
            "nmr.efg=true needs an all-PAW ground state (the Petrilli–Blöchl PAW "
            "EFG reconstructs the on-site l=2 field gradient from PAWData); this "
            "run is norm-conserving — use false or 'auto'.")
    return bool(want)


def _efg_pw_block(inp: Input, res: Any, iso_map: dict[str, str]) -> dict[str, Any]:
    """Per-site plane-wave/PAW EFG (``postscf.efg_paw``) on the shielding ground
    state: V_zz/η and, for the ``iso_map`` isotopes, the quadrupolar coupling C_Q.
    Every value is JSON-native (tensors → nested lists)."""
    from gradwave.postscf.efg_paw import efg_paw

    entries = efg_paw(res, isotopes=iso_map or None)
    sites: list[dict[str, Any]] = []
    for e in entries:
        v = e["V"]
        entry: dict[str, Any] = {
            "site": int(e["site"]),
            "species": e["element"],
            "V_zz_eV_ang2": float(e["V_zz"]),
            "eta": float(e["eta"]),
            "tensor_eV_ang2": v.detach().cpu().numpy().tolist()
            if hasattr(v, "detach") else v,
        }
        cq = e.get("C_Q")
        if cq is not None:
            entry.update({
                "isotope": cq["isotope"],
                "C_Q_MHz": float(cq["C_Q_MHz"]),
                "abs_C_Q_MHz": float(cq["abs_C_Q_MHz"]),
                "nu_Q_MHz": float(cq["nu_Q_MHz"]),
                "spin": float(cq["spin"]),
                "Q_barn": float(cq["Q_barn"]),
            })
        sites.append(entry)
    return {"method": "paw_petrilli_blochl", "n_sites": len(sites), "sites": sites}


def _assemble_nmr_sites(block: dict[str, Any]) -> list[Any]:
    """Map the REFERENCED shielding sites (those carrying ``delta_iso_ppm``) to
    ``postscf.nmr_spectrum.NMRSite`` records: δ_iso, the Haeberlen CSA (δ_aniso,
    η_csa) from the shielding tensor, and — when the ``block['efg']`` block covers
    the site with a quadrupolar isotope — C_Q (Hz)/η_Q/spin. Unreferenced sites
    (no δ_iso) are skipped: a spectrum is a chemical-shift axis."""
    from gradwave.postscf.nmr_spectrum import NMRSite

    efg_by_site: dict[int, dict[str, Any]] = {}
    efg_block = block.get("efg")
    if efg_block is not None:
        for e in efg_block["sites"]:
            efg_by_site[int(e["site"])] = e

    sites: list[Any] = []
    for s in block["sites"]:
        if "delta_iso_ppm" not in s:
            continue
        # Haeberlen reduced anisotropy in the SHIFT convention:
        #   δ_aniso = δ_zz − δ_iso = −(σ_zz − σ_iso) = −(2/3)·(σ_zz − ½(σ_xx+σ_yy)),
        # i.e. minus two-thirds of the stored full shielding anisotropy. η is
        # convention-invariant under δ = σ_ref − σ (numerator and denominator both
        # flip sign), so it carries over unchanged.
        delta_aniso = -(2.0 / 3.0) * float(s["sigma_aniso_ppm"])
        c_q, eta_q, spin = 0.0, 0.0, 0.5
        e = efg_by_site.get(int(s["site"]))
        if e is not None and "spin" in e:
            c_q = float(e["C_Q_MHz"]) * 1.0e6   # MHz → Hz
            eta_q = float(e["eta"])
            spin = float(e["spin"])
        sites.append(NMRSite(
            delta_iso=float(s["delta_iso_ppm"]),
            delta_aniso=delta_aniso,
            eta_csa=float(s["sigma_eta"]),
            c_q=c_q, eta_q=eta_q, spin=spin, weight=1.0,
            label=f"{s.get('species') or ''}{s['site']}"))
    return sites


def _spectrum_kind(mode: str, sites: list[Any]) -> str:
    """Lineshape family for the assembled ``sites`` at ``mode`` (static | mas).
    Spin-½ → CSA (``csa_static`` | ``mas``); half-integer I ≥ 3/2 → second-order
    central transition (``quad2_ct_static`` | ``quad2_ct_mas``). A mix of nuclei
    (different spins), or an unsupported integer spin, is a clear error."""
    spins = {s.spin for s in sites}
    if len(spins) != 1:
        raise ValueError(
            f"nmr.spectrum: the referenced sites carry mixed nuclear spins {sorted(spins)}; "
            "a single powder spectrum observes one nucleus — reference one element at a time.")
    spin = next(iter(spins))
    if spin == 0.5:
        return "mas" if mode == "mas" else "csa_static"
    if spin < 1.5 or int(round(2.0 * spin)) % 2 != 1:
        raise ValueError(
            f"nmr.spectrum: unsupported nuclear spin I={spin} (only spin-½ CSA and "
            "half-integer I ≥ 3/2 central-transition lineshapes are wired).")
    return "quad2_ct_mas" if mode == "mas" else "quad2_ct_static"


def _resolve_nu0_hz(
    larmor: float | dict[str, float] | None,
    species: list[str],
    isotopes: list[str],
) -> float | None:
    """Observed-nucleus Larmor frequency (Hz) from ``nmr.spectrum.larmor_mhz``: a
    scalar applies directly; a map is keyed by isotope first, then species."""
    if larmor is None:
        return None
    if not isinstance(larmor, dict):
        return float(larmor) * 1.0e6
    for key in (*isotopes, *species):
        if key in larmor:
            return float(larmor[key]) * 1.0e6
    raise ValueError(
        f"nmr.spectrum.larmor_mhz has no entry for the observed nucleus "
        f"(species {species}, isotopes {isotopes}); keys present: {sorted(larmor)}.")


def _spectrum_block(cfg: Any, block: dict[str, Any]) -> dict[str, Any]:
    """Synthesize the powder lineshape from the referenced shielding sites and,
    for quadrupolar nuclei, the EFG block. Returns a JSON-native block carrying
    the config echo and the ``(ppm_axis, intensity)`` arrays as lists."""
    import numpy as np

    from gradwave.postscf.nmr_spectrum import spectrum

    sites = _assemble_nmr_sites(block)
    if not sites:
        raise ValueError(
            "nmr.spectrum.enabled needs referenced sites: set nmr.sigma_ref so each "
            "observed site carries δ_iso (the spectrum axis is a chemical-shift scale). "
            "No site had delta_iso_ppm.")
    kind = _spectrum_kind(cfg.mode, sites)
    ref_species = sorted({s["species"] for s in block["sites"]
                          if "delta_iso_ppm" in s and s.get("species")})
    efg_sites = block.get("efg", {}).get("sites", []) if block.get("efg") else []
    isotopes = sorted({e["isotope"] for e in efg_sites
                       if "isotope" in e and e.get("species") in ref_species})
    nu0_hz = _resolve_nu0_hz(cfg.larmor_mhz, ref_species, isotopes)
    if kind != "csa_static" and (nu0_hz is None or nu0_hz <= 0.0):
        raise ValueError(
            f"nmr.spectrum: the {kind!r} lineshape needs a positive "
            "nmr.spectrum.larmor_mhz (the observed-nucleus Larmor frequency, MHz).")
    fwhm_g = cfg.broadening_ppm if cfg.lineshape == "gauss" else 0.0
    fwhm_l = cfg.broadening_ppm if cfg.lineshape == "lorentz" else 0.0
    ppm_axis, intensity = spectrum(
        sites, kind=kind, nu0_hz=nu0_hz or 0.0, nu_r_hz=cfg.spin_rate_hz,
        n_orientations=cfg.n_orientations, n_points=cfg.n_points,
        fwhm_gauss=fwhm_g, fwhm_lorentz=fwhm_l)
    return {
        "mode": cfg.mode,
        "kind": kind,
        "nucleus": isotopes or ref_species,
        "larmor_mhz": (nu0_hz / 1.0e6) if nu0_hz else None,
        "spin_rate_hz": cfg.spin_rate_hz if cfg.mode == "mas" else None,
        "broadening_ppm": cfg.broadening_ppm,
        "lineshape": cfg.lineshape,
        "n_orientations": int(cfg.n_orientations),
        "n_sites": len(sites),
        "peak_ppm": float(ppm_axis[int(np.argmax(intensity))]),
        "ppm_range": [float(np.min(ppm_axis)), float(np.max(ppm_axis))],
        "ppm_axis": ppm_axis.tolist(),
        "intensity": intensity.tolist(),
    }


def _run_shielding(inp: Input, verbose: bool) -> dict[str, Any]:
    """Plane-wave GIPAW shielding. The concrete assembly is chosen by
    ``inp.nmr.shielding_level`` against the SCF ground state actually produced:
    ``auto`` → 'gipaw' for a USPP/PAW ground state, 'bare' for norm-conserving;
    an explicit level is validated against that ground state. When ``nmr.efg`` is
    on (auto for PAW) the same ground state also yields the PW/PAW EFG, and
    ``nmr.spectrum`` synthesizes a powder lineshape from the referenced sites."""
    from gradwave.api.scf import run_scf
    from gradwave.scf.results import USPPResult

    req = inp.nmr.shielding_level
    _warn_kmesh_symmetry(inp)
    res = run_scf(inp, verbose=verbose)
    is_paw = isinstance(res, USPPResult)
    if req == "gipaw" and not is_paw:
        raise ValueError(
            "nmr.shielding_level='gipaw' needs an all-PAW ground state (the "
            "absolute σ = σ_bare + σ_core + σ_dia_aug + σ_para_aug assembles the "
            "on-site augmentation from PAWData); this run is norm-conserving — "
            "use 'bare' or 'auto'.")
    if req == "bare" and is_paw:
        raise NotImplementedError(
            "nmr.shielding_level='bare' on a USPP/PAW ground state is not wired "
            "(the smooth-only sigma_shielding_dq bare path is norm-conserving) — "
            "use 'gipaw' or 'auto' for the absolute GIPAW σ on PAW.")
    level = req if req != "auto" else ("gipaw" if is_paw else "bare")
    block = (_shielding_gipaw_block(inp, res) if level == "gipaw"
             else _shielding_bare_block(inp, res))
    block["shielding_level"] = level
    _apply_shift_reference(block, inp.nmr.sigma_ref)

    iso_map = _resolve_isotopes(inp.nmr.isotopes, inp.atoms.get_chemical_symbols())
    if _resolve_pw_efg(inp.nmr.efg, is_paw):
        block["efg"] = _efg_pw_block(inp, res, iso_map)
    if inp.nmr.spectrum.enabled:
        block["spectrum"] = _spectrum_block(inp.nmr.spectrum, block)
    return block


def reference_sigma_iso(
    inp: Input, species: str, *, verbose: bool = False
) -> float:
    """Absolute isotropic shielding σ_iso (ppm) of ``species`` in a reference
    solid, for use as ``nmr.sigma_ref`` in δ_iso = σ_ref − σ_iso.

    Runs the shielding task on the reference ``Input`` at its own
    ``nmr.shielding_level`` (use the SAME level as the sample) and averages
    σ_iso over the sites of ``species`` (crystallographically equivalent sites
    agree; the mean cancels residual mesh noise). The caller supplies the
    reference structure/pseudopotentials in ``inp`` — e.g. the primary IUPAC
    reference for the nucleus, or a well-characterized secondary standard."""
    block = _run_shielding(inp, verbose)
    vals = [float(s["sigma_iso_ppm"]) for s in block["sites"]
            if s["species"] == species]
    if not vals:
        raise ValueError(
            f"reference_sigma_iso: no {species!r} site in the reference input "
            f"(species present: {sorted({s['species'] for s in block['sites']})})")
    return sum(vals) / len(vals)


def run_nmr(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Compute the NMR observable selected by ``inp.nmr.task`` (task: nmr).

    ``efg`` returns the all-electron FLAPW electric field gradient per site (V_zz,
    η, tensor) plus the quadrupolar coupling C_Q for the selected isotopes.
    ``shielding`` returns the plane-wave GIPAW shielding tensor per site (σ_iso,
    anisotropy, η); with ``nmr.shielding_level='gipaw'`` (auto for PAW pseudos)
    it is the full absolute σ with the per-term breakdown, and with
    ``nmr.sigma_ref`` each site carries the chemical shift δ_iso. Returns the
    ``nmr`` summary block."""
    if inp.nmr.task == "efg":
        return _run_efg(inp, verbose)
    return _run_shielding(inp, verbose)
