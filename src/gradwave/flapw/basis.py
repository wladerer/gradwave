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


def efg_anion_basis(anions: list[str], *, helo_e: float | dict[str, float] = 90.0,
                    semicore: bool = True, s_label: str = "2s", s_energy_label: str = "2p"
                    ) -> dict[str, dict[str, Any]]:
    """Build the validated EFG anion basis recipe for the named ``anions``.

    Returns ``{"los": ..., "el_override": ...}`` ready to merge into
    ``FlapwParams``/``crystal_scf_multi``: for each anion species an unconfined l=1
    HELO at ``helo_e`` eV and (``semicore``) an l=0 LO on ``s_label`` with its
    linearization energy moved to ``s_energy_label``. Defaults are the period-2
    (O/F/N) recipe; pass ``s_label``/``s_energy_label`` for a different period, or
    ``semicore=False`` for the HELO alone.

    ``helo_e`` — the HELO energy E₂ (eV) — is a **structure-specific η lever**, measured on
    rutile-TiO₂ O and corundum-Al₂O₃ O (``experiments/autoapw/efg_eta_anion.md``). It may be a
    single float (applied to every anion) or a ``{species: eV}`` mapping (per-species; species
    absent from the mapping fall back to 90 eV). Measured guidance:

    * **Default 90 eV is the robust, validated value** — keep it for the general case and for
      well-separated / already-biaxially-correct anion sites, where E₂ is a **no-op** (corundum O:
      η 0.460 → 0.462 over E₂ 90 → 120, magnitude flat at ~96 %; raising E₂ neither helps nor
      overshoots).
    * On a **stringent biaxial anion whose muffin-tin spheres nearly touch** (rutile-class O, where
      the standard basis leaves η badly low), E₂ tunes η **monotonically upward**: rutile O
      η 0.654 → 0.724 over E₂ 90 → 120, crossing the Elk reference (0.74) near E₂ ≈ 123. The cost
      is a ~3 % drop in |V_zz| (the magnitude↔η trade-off) and a fullpot SCF that stops gating
      cleanly above E₂ ≈ 120 — so raise E₂ deliberately, per hard species, when a correct
      second-order MAS *lineshape* (η-driven) matters more than the last few % of C_Q magnitude.

    Raises ``ValueError`` on an empty/duplicate anion list.
    """
    if not anions:
        raise ValueError("efg_anion_basis needs at least one anion species")
    if len(set(anions)) != len(anions):
        raise ValueError(f"duplicate anion species in {anions}")
    los: dict[str, list[Any]] = {}
    el_override: dict[str, dict[int, Any]] = {}
    for sp in anions:
        e2 = helo_e.get(sp, 90.0) if isinstance(helo_e, dict) else helo_e
        specs: list[Any] = []
        if semicore:
            specs.append((0, s_label))
            el_override[sp] = {0: s_energy_label}
        specs.append((1, {"e": float(e2), "confine": False}))
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
