"""Runtime species injection + shared reporting for the main-group quadrupolar-NMR study.

Extends the multi-material EFG validation (efg_multimaterial_validation.md) from the ²⁷Al/¹⁷O/
¹⁹F set to the light closed-shell main-group cations ²³Na, ⁷Li, ¹¹B (plus a 2nd ²⁷Al site). Each
injected atomic species (B, N, Na, Li — Al/O ship or are injected by the corundum runner) is added
at module import time so the spawned k-worker pool (spawn re-imports the driver, which imports this
module) re-applies it. NO src file is modified.

Every injected species is validated against NIST-LSD atomic eigenvalues by
``validate_mgroup_species.py`` (gradwave ``atomic_scf`` vs reference, valence levels within ~1-2 %;
the deep 1s gap is the expected scalar-nonrelativistic shift). Core partitions mirror the corundum
recipe: freeze the noble-gas core (B/N freeze 1s; Na freezes [Ne]; Li freezes 1s), valence = the
rest. Na thus FREEZES its 2p semicore — the same choice that reached experiment for Al³⁺ but failed
(wrong sign) for Mg²⁺; whether Na⁺ behaves like Al³⁺ or Mg²⁺ is one of the questions here.
"""
import numpy as np

from gradwave.flapw import atom as _atom
from gradwave.flapw import scf as _scf
from gradwave.flapw.nmr import quadrupolar_coupling

BOHR = 0.5291772109      # Å per Bohr


def _inject(sym, Z, occ, core, val_e, n_val_bands, valence_nl, nist_ha):
    _atom.CONFIG[sym] = (float(Z), occ)
    _scf._CORE[sym] = core
    _scf._VAL_E[sym] = val_e
    _scf._N_VAL_BANDS[sym] = n_val_bands
    _scf._VALENCE_NL[sym] = valence_nl
    _atom.NIST_LDA_EV[sym] = {k: v * 27.211386 for k, v in nist_ha.items()}


# ---- B (Z=5): freeze 1s, valence 2s²2p¹ (all valence, no semicore — like a light O) ----
_inject("B", 5, [(1, 0, 2), (2, 0, 2), (2, 1, 1)], [(0, 1, 2)], 3, 2,
        {0: "2s", 1: "2p"}, {"1s": -6.564, "2s": -0.3448, "2p": -0.1363})

# ---- N (Z=7): freeze 1s, valence 2s²2p³ ----
_inject("N", 7, [(1, 0, 2), (2, 0, 2), (2, 1, 3)], [(0, 1, 2)], 5, 3,
        {0: "2s", 1: "2p"}, {"1s": -13.976, "2s": -0.6763, "2p": -0.2658})

# ---- Na (Z=11): freeze [Ne] (1s2s2p), valence 3s¹ (Na⁺ donates its 3s). 2p SEMICORE FROZEN. ----
_inject("Na", 11, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 1)],
        [(0, 1, 2), (0, 2, 2), (1, 1, 6)], 1, 1,
        {0: "3s"}, {"1s": -37.68, "2s": -2.062, "2p": -1.052, "3s": -0.1056})

# ---- Li (Z=3): freeze 1s, valence 2s¹ (no p semicore; EFG = lattice + tiny 1s Sternheimer) ----
_inject("Li", 3, [(1, 0, 2), (2, 0, 1)], [(0, 1, 2)], 1, 1,
        {0: "2s"}, {"1s": -1.8785, "2s": -0.1063})

# ---- Al (Z=13): the corundum recipe, re-injected here so AlN can reuse it (frozen [Ne]) ----
_inject("Al", 13, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 1)],
        [(0, 1, 2), (0, 2, 2), (1, 1, 6)], 3, 2,
        {0: "3s", 1: "3p"}, {"1s": -55.5, "2s": -3.93, "2p": -2.56, "3s": -0.286, "3p": -0.10})


def hex_cell_bohr(a_ang, c_ang):
    """Hexagonal lattice vectors (rows, Bohr): a1=a x̂, a2=a(-½,√3/2), a3=c ẑ."""
    a, c = a_ang / BOHR, c_ang / BOHR
    return np.array([[a, 0.0, 0.0],
                     [-0.5 * a, np.sqrt(3) / 2 * a, 0.0],
                     [0.0, 0.0, c]])


def report_site(site, name, isotope, axes, extra=""):
    """Axis-resolved EFG report for one site. ``axes`` = {label: unit-vector}. ``isotope`` may be
    None (no C_Q, e.g. a spin-½ nucleus)."""
    t = np.array(site["tensor"])
    w, vecs = np.linalg.eigh(t)
    order = np.argsort(-np.abs(w))
    w, vecs = w[order], vecs[:, order]
    proj = {ax: float(n @ t @ n) for ax, n in axes.items()}
    cq = ""
    if isotope is not None:
        c = quadrupolar_coupling(site["V_zz"], site["eta"], isotope)
        cq = f" C_Q({isotope})={c['abs_C_Q_MHz']:.4f} MHz"
    print(f"{name} FULL: eig=[{w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f}] eta={site['eta']:.3f} "
          f"V_zz={site['V_zz']:+.3f}{cq}", flush=True)
    pstr = "  ".join(f"{k}={v:+.3f}" for k, v in proj.items())
    print(f"     proj {pstr}  V_zz eigvec=[{vecs[0,0]:+.2f},{vecs[1,0]:+.2f},{vecs[2,0]:+.2f}]",
          flush=True)
    print(f"     ONSITE(valence): V_zz={site['V_zz_valence']:+.3f} eta={site['eta_valence']:.3f} "
          f"{extra}", flush=True)
