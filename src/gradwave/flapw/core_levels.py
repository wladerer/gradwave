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
