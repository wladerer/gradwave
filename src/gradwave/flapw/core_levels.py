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
core-level chemical shift.

RELIABILITY BOUNDARY of the raw muffin-tin eigenvalue shift (``delta_eig_eV``).
The spherical muffin-tin core eigenvalue DOES carry the on-site external (Madelung)
field: the sphere's l=0 potential is matched at R_MT to the Weinert-reconstructed
interstitial Coulomb (``scf._weinert_multi``), which adds the external constant
``C0_ext = v_bc(R) − v_own(R)`` inside the sphere, so the eigenvalue = (in-sphere own
term) + C0_ext. The failure mode for a within-cell same-element shift whose ONLY source
is an inter-site Madelung difference (two far-apart O in a vacuum box) is NOT a discarded
reference — it is that the muffin-tin SPHERICAL in-sphere own term picks up a large,
wrong-signed spurious difference between the two inequivalent sites (their
spherically-averaged in-sphere densities differ) that SWAMPS the correct C0_ext. Measured
on the Ti+2O demo (O R_MT 1.32 Bohr): eigenvalue Δ = −1.67 eV = (in-sphere −3.81) +
(C0_ext +2.19), while Elk gives +2.5 eV, essentially all in C0_ext (its in-sphere own
difference ≈ 0 — verified: at R_MT 1.40 Bohr gradwave's C0_ext +2.81 matches Elk's TOTAL
+2.86). So for a same-element inter-site (Madelung) shift the physically-correct
initial-state quantity is the on-site electrostatic-potential difference
``delta_madelung_eV``, computed from C0_ext (``onsite_madelung_potentials``); it reproduces
Elk's sign, magnitude (+2.19 vs +2.5), AND its growth with R_MT — but is itself reliable
only for R_MT ≳ 1.0 Bohr (the coarse interstitial FFT under-resolves the field at tiny
spheres: C0_ext goes wrong-signed at R_MT 0.70 Bohr). The eigenvalue shift stays right for
ON-SITE (oxidation-state) changes captured inside the muffin tin (Si→SiO₂ +4.4). Both
numbers are reported per within-cell pair; see ``experiments/autoapw/xps_validation.md``
for the full decomposition.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.flapw.radial import radial_eigs_tridiag

_ORB = "spdf"


def onsite_madelung_potentials(v_hart, spheres: list[dict[str, Any]], keys: list[str],
                               cell) -> dict[str, float]:
    """On-site external (Madelung) electrostatic potential ``C0_ext`` at each site (eV).

    The l=0 external Coulomb constant the full potential adds INSIDE a muffin tin — the
    Weinert-reconstructed interstitial Hartree surface-average at ``R_MT`` minus the
    sphere's own monopole field::

        C0_ext = v_bc(R) − E2·(q_sph − Z)/R ,  v_bc(R) = ⟨V_int⟩_{|r−τ|=R} (l=0).

    This is the site-distinguishing INITIAL-STATE reference for a same-element inter-site
    (Madelung) core-level shift: it matches Elk's within-cell O1s shift in sign, magnitude,
    and R_MT trend for R_MT ≳ 1.0 Bohr (below that the coarse interstitial FFT under-resolves
    the field at the tiny sphere and it goes wrong-signed). It is NOT a substitute for the
    eigenvalue on an ON-SITE (oxidation-state) shift, which lives in the in-sphere charge.
    ``spheres`` is the per-key list from ``scf._weinert_multi`` (each a dict with
    ``tau``/``R``/``rr``/``dx``/``rho_sph``/``Z``), order-aligned with ``keys``; ``v_hart``
    its Coulomb-only interstitial grid; ``cell`` the cell (Å, any of scalar/(3,)/(3,3))."""
    from gradwave.flapw.efg import interstitial_boundary_multi

    out: dict[str, float] = {}
    for k, sp in zip(keys, spheres, strict=True):
        R = float(sp["R"])
        rr = np.asarray(sp["rr"], dtype=float)
        drw = rr * float(sp["dx"])
        q_sph = float(np.sum(4 * math.pi * np.asarray(sp["rho_sph"], dtype=float) * rr**2 * drw))
        vbc0 = float(interstitial_boundary_multi(v_hart, sp["tau"], R, cell, [(0, 0)])[(0, 0)].real
                     / math.sqrt(4 * math.pi))
        out[k] = vbc0 - E2 * (q_sph - float(sp["Z"])) / R
    return out


def core_levels_from_state(v_by_key: dict[str, Any], syms: list[str], keys: list[str],
                           core_map: dict[str, Any], r, dx: float,
                           v_madelung: dict[str, float] | None = None) -> dict[str, Any]:
    """Per-site all-electron core eigenvalues (eV).

    ``v_by_key[key]`` is the converged spherical MT potential on the radial mesh
    ``(r, dx)``; ``core_map[element]`` the ``(l, n_radial_index, occ)`` core list
    (``scf._CORE`` merged with any per-run override). Site ``keys[i]`` carries
    element ``syms[i]``. Returns ``{key: {"symbol": s, "levels": {"1s": eV, …}}}``;
    the orbital label is ``n = l + n_radial_index`` (1s: l=0/nidx=1, 2s: l=0/nidx=2,
    2p: l=1/nidx=1). When ``v_madelung`` (from :func:`onsite_madelung_potentials`) is
    supplied, each record also carries ``"v_madelung_eV"`` — the on-site external
    electrostatic reference used by :func:`core_level_shifts` for the initial-state
    same-element inter-site (Madelung) shift."""
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
        if v_madelung is not None and k in v_madelung:
            out[k]["v_madelung_eV"] = float(v_madelung[k])
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

    For every element present on ≥2 sites, Δε(site_j − site_ref) per shared orbital,
    with ``site_ref`` the first site of that element (both share the interstitial
    reference). Each entry carries TWO numbers (see the module docstring's reliability
    boundary):

    * ``delta_eV`` — the raw muffin-tin core-eigenvalue difference. Right for ON-SITE
      (oxidation-state) shifts. For a shift sourced purely by an inter-site Madelung field
      its sign depends on two upstream choices that used to be wrong (``xps_madelung_stage2a.md``):
      the interstitial density fed to the Weinert Hartree continuation, and the cation
      core/valence partition. With ``crystal_scf_multi(mask_interstitial=True)`` (band-limited
      interstitial ρ_I mask, Elk ``rhoir`` — kills the small-sphere catastrophic-cancellation
      corruption of ``v_hart``) AND the cation semicore in the valence (e.g. Ti 3s as a
      local orbital, ``los``/``val_e``/``core``, so the in-sphere charge matches Elk's), the raw
      ``delta_eV`` recovers the physically-correct POSITIVE sign and lands near Elk (Ti+2O demo:
      −1.19 → +1.19 at O R_MT 1.40 Bohr). Left at the defaults (unmasked ρ_I, cation semicore
      frozen) it stays wrong-signed — the spherical in-sphere own term swamps the (correct)
      external reference.
    * ``delta_madelung_eV`` — the on-site external electrostatic (Madelung) potential
      difference (from each record's ``v_madelung_eV``; ``None`` if not supplied). The
      physically-correct initial-state quantity for a same-element inter-site shift; matches
      Elk for R_MT ≳ 1.0 Bohr once the interstitial ρ_I is masked and the cation semicore is
      in the valence (Ti+2O demo, O R_MT 1.00/1.40 Bohr: +3.09/+5.07 vs Elk +2.90/+4.79).

    Returns a list of ``{species, orbital, site, ref_site, delta_eV, delta_madelung_eV,
    e_site_eV, e_ref_eV}``."""
    by_species: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for k, rec in levels.items():
        by_species.setdefault(rec["symbol"], []).append((k, rec))
    shifts: list[dict[str, Any]] = []
    for s, sites in by_species.items():
        if len(sites) < 2:
            continue
        ref_k, ref_rec = sites[0]
        ref_lv = ref_rec["levels"]
        for k, rec in sites[1:]:
            lv = rec["levels"]
            d_mad = (None if "v_madelung_eV" not in rec or "v_madelung_eV" not in ref_rec
                     else float(rec["v_madelung_eV"]) - float(ref_rec["v_madelung_eV"]))
            for orb in sorted(ref_lv):
                if orb in lv:
                    shifts.append({
                        "species": s, "orbital": orb, "site": k, "ref_site": ref_k,
                        "delta_eV": lv[orb] - ref_lv[orb],
                        "delta_madelung_eV": d_mad,
                        "e_site_eV": lv[orb], "e_ref_eV": ref_lv[orb],
                    })
    return shifts
