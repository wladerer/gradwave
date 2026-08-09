"""Computational hydrogen electrode (CHE) thermodynamics for *H and *CO.

Given the relaxed DFT energies, form adsorption energies and CHE free energies:

    *H  :  * + ½H2(g) → H*    ΔE_H  = E(H*) − E(*) − ½E(H2)
    *CO :  * + CO(g)  → CO*   ΔE_CO = E(CO*) − E(*) − E(CO)

ΔG adds ZPE + enthalpy − TS corrections. The gas-phase / adsorbate corrections
below are standard 298 K literature aggregates (Nørskov CHE); for production
replace them with values from computed vibrational frequencies (vib.py hook).

CHE: ½H2(g) ⇌ H⁺ + e⁻ at 0 V vs RHE, so ΔG_H* (referenced to ½H2) is the
potential-independent HER descriptor — |ΔG_H*| ≈ 0 is optimal. The electron
potential dependence (+eU) enters the *steps* that transfer (H⁺+e⁻), and the
grand-canonical constant-µ path (differentiable.py) captures the charge response
directly.
"""

from __future__ import annotations

from dataclasses import dataclass

# Aggregate G − E_DFT corrections at 298 K, 1 bar [eV] (ZPE + ∫Cp dT − TS).
# Adsorbate values are relative to the clean slab (frustrated modes only).
# LITERATURE DEFAULTS — swap for frequency-derived values for quantitative work.
DG_CORR = {
    "H": 0.24,    # ½H2 → H*   (ZPE + TS aggregate)
    "CO": 0.10,   # CO(g) → CO* (loses gas trans/rot entropy; frustrated modes)
}


@dataclass
class AdsResult:
    metal: str
    ads: str            # "H" | "CO"
    best_site: str
    e_ads: float        # ΔE [eV], electronic
    dg_ads: float       # ΔG [eV], CHE (298 K)
    per_site: dict      # site → ΔE [eV]


def adsorption_energy(e_slab_ads: float, e_clean: float, e_gas: float) -> float:
    """ΔE = E(slab+ads) − E(slab) − E(gas_reference) [eV].

    Reference gas is ½H2 for *H and CO for *CO — pass e_gas already scaled
    (0.5·E_H2 for H, E_CO for CO)."""
    return e_slab_ads - e_clean - e_gas


def summarize(metal: str, ads: str, per_site: dict[str, float],
              e_clean: float, e_gas_ref: float) -> AdsResult:
    """per_site: site → E(slab+ads) [eV]; e_gas_ref: the scaled gas reference."""
    de = {s: adsorption_energy(e, e_clean, e_gas_ref) for s, e in per_site.items()}
    best = min(de, key=de.get)
    return AdsResult(metal=metal, ads=ads, best_site=best,
                     e_ads=de[best], dg_ads=de[best] + DG_CORR[ads],
                     per_site=de)


def her_limiting_potential(dg_h: float) -> float:
    """HER limiting potential [V vs RHE] from ΔG_H* (the potential at which the
    least-favorable H step becomes downhill): U_L = −|ΔG_H*|/e."""
    return -abs(dg_h)


def report(res: AdsResult) -> str:
    sites = "  ".join(f"{s}:{v:+.3f}" for s, v in sorted(res.per_site.items(),
                                                         key=lambda kv: kv[1]))
    lines = [
        f"*{res.ads} on {res.metal}(111):",
        f"  per-site ΔE [eV]: {sites}",
        f"  best site: {res.best_site}   ΔE={res.e_ads:+.3f} eV   "
        f"ΔG={res.dg_ads:+.3f} eV",
    ]
    if res.ads == "H":
        lines.append(f"  HER descriptor |ΔG_H*|={abs(res.dg_ads):.3f} eV  "
                     f"(U_L={her_limiting_potential(res.dg_ads):+.3f} V vs RHE)")
    return "\n".join(lines)
