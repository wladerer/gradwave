"""EFG basis-recipe helpers for FLAPW.

The muffin-tin EFG accuracy for anion sites needs two validated basis levers on top
of the plain LAPW basis (``experiments/autoapw/efg_helo_l1_fix.md``,
``efg_multimaterial_validation.md``):

* an **unconfined l=1 HELO** (high-energy local orbital, ``E₂≈90 eV``, ``confine=False``)
  — supplies the genuinely-distinct in-sphere p radial the on-site aspherical (l=2, p×p)
  EFG density needs. It moves the anion on-site ``V_zz`` from ~0.66 to ~0.94 of the Elk
  all-electron reference and recovers the correct biaxial principal frame (rutile O:
  η 0.17 on the wrong [001] axis → 0.65 on Elk's [110] axis, correct sign). It gates at
  the production ``kerker=0.7`` where the *confined* l=1 LO diverged.

* the **l=0 semicore LO with its energy moved off the valence 2s** (``el_override`` to the
  2p label — the #344 recipe) — conditions the s channel so the LO adds a distinct radial
  rather than a near-null one.

These are correct for the period-2 anions O, F, N (valence 2s/2p). They are *not* a blanket
cation lever — the HELO helps anions and Al-type semicore-frozen cations but over-enriches a
light main-group cation (h-BN B gets worse). So the recipe is opt-in per named anion species,
not auto-applied by element: the caller states which species are anions (a compound-dependent
physical choice), exactly as the campaign scripts do by hand. This helper only removes the
copy-paste, it does not decide chemistry.
"""
from __future__ import annotations

from typing import Any

__all__ = ["efg_anion_basis", "merge_basis"]


def efg_anion_basis(anions: list[str], *, helo_e: float = 90.0, semicore: bool = True,
                    s_label: str = "2s", s_energy_label: str = "2p"
                    ) -> dict[str, dict[str, Any]]:
    """Build the validated EFG anion basis recipe for the named ``anions``.

    Returns ``{"los": ..., "el_override": ...}`` ready to merge into
    ``FlapwParams``/``crystal_scf_multi``: for each anion species an unconfined l=1
    HELO at ``helo_e`` eV and (``semicore``) an l=0 LO on ``s_label`` with its
    linearization energy moved to ``s_energy_label``. Defaults are the period-2
    (O/F/N) recipe; pass ``s_label``/``s_energy_label`` for a different period, or
    ``semicore=False`` for the HELO alone.

    Raises ``ValueError`` on an empty/duplicate anion list.
    """
    if not anions:
        raise ValueError("efg_anion_basis needs at least one anion species")
    if len(set(anions)) != len(anions):
        raise ValueError(f"duplicate anion species in {anions}")
    los: dict[str, list[Any]] = {}
    el_override: dict[str, dict[int, Any]] = {}
    for sp in anions:
        specs: list[Any] = []
        if semicore:
            specs.append((0, s_label))
            el_override[sp] = {0: s_energy_label}
        specs.append((1, {"e": float(helo_e), "confine": False}))
        los[sp] = specs
    out: dict[str, dict[str, Any]] = {"los": los}
    if el_override:
        out["el_override"] = el_override
    return out


def merge_basis(*specs: dict[str, Any] | None) -> dict[str, Any]:
    """Merge basis recipes (``{"los": ..., "el_override": ..., "val_e": ...}``) left→right.

    Later specs win per species. ``los`` is replaced per species (not concatenated —
    "one LO per l per atom" is enforced downstream, so a caller who wants to add to an
    anion's list must build the combined list themselves). ``None`` entries are skipped.
    """
    merged: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if not spec:
            continue
        for field_name, by_species in spec.items():
            if by_species is None:
                continue
            merged.setdefault(field_name, {}).update(by_species)
    return merged
