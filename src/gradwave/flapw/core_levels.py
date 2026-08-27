"""Initial-state core-level (XPS) chemical shifts from all-electron FLAPW.

After the muffin-tin SCF converges, each atom's core states are re-solved in
that atom's self-consistent spherical potential ``V_{l=0}(r)`` (``v_by_key`` —
the full radial Hartree + nuclear −Z·e²/r + XC inside R_MT, flat = interstitial
zero outside). This is the SAME radial eigenproblem the SCF already solves every
iteration to build ρ_core (``scf._multi_core_density`` → ``radial_eigs_tridiag``);
it just discards the eigenvalues, which this module recovers.

FLAPW absolute eigenvalues are referenced to the flat interstitial zero and
"wander" between cells, so only the WITHIN-CELL SHIFT between sites is physical
— the initial-state XPS chemical shift. Within one cell every muffin tin shares
that single interstitial reference, and the systematic radial-mesh error is the
same at every site, so both cancel in the difference: the equivalent-site shift
is exactly zero and the inequivalent-site shift is the observable.

Validation (vs Elk 11.0.2 and experiment; see ``experiments/autoapw/xps_validation.md``):
equivalent-site nulls are exact in both codes (rutile TiO₂ O/Ti, β-cristobalite SiO₂ O).
The cross-cell Si-2p Si-vs-SiO₂ shift comes out +4.4 eV (gradwave) / +5.5 eV (Elk) vs
~+4 eV experiment when the cells are aligned on the interstitial zero (average-potential
reference; ``cross_cell_binding_shift`` with ``e_ref=0``) — the physical reference for a
core-level chemical shift. NOTE the reliability boundary: gradwave's flat interstitial
zero discards inter-site potential differences that live BETWEEN the muffin tins, so the
shift is trustworthy only when the site-distinguishing potential is ON-SITE (close-packed
crystals, oxidation-state changes). A within-cell shift sourced purely by a long-range
interstitial Madelung term (two far-apart O in a vacuum box) comes out wrong-signed vs Elk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gradwave.flapw.radial import radial_eigs_tridiag

_ORB = "spdf"


def core_levels_from_state(v_by_key: dict[str, Any], syms: list[str], keys: list[str],
                           core_map: dict[str, Any], r, dx: float) -> dict[str, Any]:
    """Per-site all-electron core eigenvalues (eV).

    ``v_by_key[key]`` is the converged spherical MT potential on the radial mesh
    ``(r, dx)``; ``core_map[element]`` the ``(l, n_radial_index, occ)`` core list
    (``scf._CORE`` merged with any per-run override). Site ``keys[i]`` carries
    element ``syms[i]``. Returns ``{key: {"symbol": s, "levels": {"1s": eV, …}}}``;
    the orbital label is ``n = l + n_radial_index`` (1s: l=0/nidx=1, 2s: l=0/nidx=2,
    2p: l=1/nidx=1)."""
    rt = torch.as_tensor(np.asarray(r), dtype=torch.float64)
    dxf = float(dx)
    out: dict[str, Any] = {}
    for i, s in enumerate(syms):
        k = keys[i]
        v = torch.as_tensor(np.asarray(v_by_key[k]), dtype=torch.float64)
        levels: dict[str, float] = {}
        for (l, nidx, _occ) in core_map.get(s, []):
            e, _ = radial_eigs_tridiag(int(l), rt, dxf, v, int(nidx))
            levels[f"{int(l) + int(nidx)}{_ORB[int(l)]}"] = float(e[int(nidx) - 1])
        out[k] = {"symbol": s, "levels": levels}
    return out


def referenced_binding_energies(levels: dict[str, Any], e_ref: float) -> dict[str, Any]:
    """Per-site core binding energies referenced to a per-cell energy ``e_ref`` (eV).

    Cross-cell comparison of absolute FLAPW core eigenvalues needs a common energy
    zero because each cell's levels float on its own interstitial zero. ``BE = e_ref −
    e_core`` picks the reference: any interstitial shift ``c`` moves the cores and a
    same-frame ``e_ref`` together, so ``BE`` is invariant under ``c``. ``BE > 0`` is the
    initial-state XPS binding energy.

    Choice of ``e_ref`` matters (see ``experiments/autoapw/xps_validation.md``): for a
    core-level CHEMICAL SHIFT the physical reference is the **average electrostatic
    potential**, for which the FLAPW **interstitial zero itself** is the proxy — pass
    ``e_ref = 0`` (potential alignment). This reproduces the ~4 eV Si-2p Si-vs-SiO₂
    shift. Referencing to each cell's **VBM/E_F** instead folds the (large, material-
    dependent) valence-band/gap difference into the number and over-corrects — useful
    only as an uncertainty bracket, not the physical shift. Returns
    ``{key: {"symbol": s, "binding": {"1s": BE, …}}}``."""
    out: dict[str, Any] = {}
    for k, rec in levels.items():
        binding = {orb: float(e_ref) - float(e) for orb, e in rec["levels"].items()}
        out[k] = {"symbol": rec["symbol"], "binding": binding}
    return out


def cross_cell_binding_shift(levels_a: dict[str, Any], e_ref_a: float,
                             levels_b: dict[str, Any], e_ref_b: float,
                             symbol: str, orbital: str) -> dict[str, Any]:
    """Cross-cell initial-state binding-energy shift ΔBE(b − a) for ``symbol``/``orbital``.

    Each cell's core level is referenced to its own ``e_ref`` (VBM/E_F), then averaged
    over the sites of ``symbol`` (equivalent sites agree; inequivalent sites are meaned).
    ``ΔBE = BE_b − BE_a > 0`` means the core is MORE bound in cell b (the textbook
    Si-2p-in-SiO₂ chemical shift, ≈ +4 eV vs bulk Si). This is the initial-state
    (Koopmans) estimate — it omits final-state core-hole screening, which for Si/SiO₂
    adds ≲1 eV of the observed shift. Returns ``{species, orbital, delta_BE_eV,
    BE_a_eV, BE_b_eV, n_sites_a, n_sites_b}``."""
    def _mean_be(levels: dict[str, Any], e_ref: float) -> tuple[float, int]:
        be = referenced_binding_energies(levels, e_ref)
        vals = [rec["binding"][orbital] for rec in be.values()
                if rec["symbol"] == symbol and orbital in rec["binding"]]
        if not vals:
            raise ValueError(f"no {symbol} site carries orbital {orbital}")
        return float(np.mean(vals)), len(vals)

    be_a, na = _mean_be(levels_a, e_ref_a)
    be_b, nb = _mean_be(levels_b, e_ref_b)
    return {"species": symbol, "orbital": orbital, "delta_BE_eV": be_b - be_a,
            "BE_a_eV": be_a, "BE_b_eV": be_b, "n_sites_a": na, "n_sites_b": nb}


def core_level_shifts(levels: dict[str, Any]) -> list[dict[str, Any]]:
    """Within-cell same-element, same-orbital core-level shifts (eV).

    For every element present on ≥2 sites, Δε(site_j − site_ref) per shared
    orbital, with ``site_ref`` the first site of that element (both share the
    interstitial reference, so the shift is physical). Returns a list of
    ``{species, orbital, site, ref_site, delta_eV, e_site_eV, e_ref_eV}``."""
    by_species: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for k, rec in levels.items():
        by_species.setdefault(rec["symbol"], []).append((k, rec["levels"]))
    shifts: list[dict[str, Any]] = []
    for s, sites in by_species.items():
        if len(sites) < 2:
            continue
        ref_k, ref_lv = sites[0]
        for k, lv in sites[1:]:
            for orb in sorted(ref_lv):
                if orb in lv:
                    shifts.append({
                        "species": s, "orbital": orb, "site": k, "ref_site": ref_k,
                        "delta_eV": lv[orb] - ref_lv[orb],
                        "e_site_eV": lv[orb], "e_ref_eV": ref_lv[orb],
                    })
    return shifts
