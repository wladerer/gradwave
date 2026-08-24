"""Scope prototype: initial-state core-level (XPS) shifts from all-electron FLAPW.

GO/NO-GO probe for computing environment-dependent core eigenvalues by re-solving the
core radial equation in the CONVERGED self-consistent spherical muffin-tin potential
V_{l=0}(r) that ``crystal_scf_multi`` already exposes as ``info["v_by_key"][key]``.

Physics: FLAPW absolute eigenvalues are referenced to the flat interstitial zero and
"wander" between cells, but WITHIN one cell every muffin tin shares that single
interstitial reference ``v_i0``, so the core-level SHIFT between two sites,
    Δε = ε_core(site A) − ε_core(site B),
is reference-independent and physically meaningful (an initial-state XPS chemical shift).

The spherical potential ``v_by_key[k]`` is, inside R_MT, the full radial potential
(Hartree of the sphere charge + nuclear −Z·e²/r + LDA XC), and flat = v_i0 outside — see
gradwave/flapw/scf.py:_weinert_multi (line ``-Z*E2/rr``) and _multi_iterate (line
``vnew_np[mask] = vnew_sph``). So it is exactly a true all-electron core potential; no
nuclear term needs to be re-added. The core solver ``radial_eigs_tridiag`` is the same
routine the SCF already calls at every iteration to build ρ_core (scf.py line 1520) —
it just discards the eigenvalues, which is all this probe recovers.

Run:  uv run python experiments/flapw/core_levels_probe.py [tier]
      tier in {atom, crystal, all}   (default: atom)
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import HARTREE_EV
from gradwave.flapw.atom import CONFIG, NIST_LDA_EV, atomic_scf
from gradwave.flapw.radial import log_mesh, radial_eigs_tridiag
from gradwave.flapw.scf import _CORE

# The FLAPW radial mesh is fixed in _multi_setup; v_by_key arrays live on exactly this mesh.
_R, _DX = log_mesh(1e-5, 28.0, 2500)


def core_levels_from_vbk(vbk: dict, syms: list[str],
                         extra_core: dict | None = None) -> dict:
    """Re-solve the core states of every site in its converged spherical MT potential.

    ``vbk`` = ``info["v_by_key"]`` from ``crystal_scf_multi``; ``syms`` the per-site
    element list (site i ↔ key ``a{i}``). Returns ``{key: (symbol, {"1s": eV, ...})}``.
    ``extra_core`` optionally overrides the (l, n_radial_index, occ) core list per element
    (e.g. to also re-solve semicore states not in scf._CORE).
    """
    core_tab = dict(_CORE)
    if extra_core:
        core_tab.update(extra_core)
    out: dict = {}
    for i, s in enumerate(syms):
        k = f"a{i}"
        v = torch.as_tensor(np.asarray(vbk[k]), dtype=torch.float64)
        levels: dict[str, float] = {}
        for (l, nidx, _occ) in core_tab.get(s, []):
            E, _ = radial_eigs_tridiag(l, _R, _DX, v, nidx)
            n = l + nidx                       # 1s: l=0,nidx=1 ; 2p: l=1,nidx=1 ; 2s: l=0,nidx=2
            levels[f"{n}{'spdf'[l]}"] = float(E[nidx - 1])
        out[k] = (s, levels)
    return out


def _pairwise_shifts(levels: dict) -> list[str]:
    """Format every within-cell same-element, same-orbital core-level shift."""
    lines = []
    by_species: dict[str, list[tuple[str, dict]]] = {}
    for k, (s, lv) in levels.items():
        by_species.setdefault(s, []).append((k, lv))
    for s, sites in by_species.items():
        if len(sites) < 2:
            continue
        orbs = sorted(sites[0][1])
        for orb in orbs:
            base_k, base = sites[0]
            for k, lv in sites[1:]:
                if orb in lv and orb in base:
                    d = lv[orb] - base[orb]
                    lines.append(f"    Δε({s} {orb}) {k}−{base_k} = {d:+.3f} eV "
                                 f"[{lv[orb]:.2f} vs {base[orb]:.2f}]")
    return lines


def tier_atom() -> None:
    """Anchor: re-solve cores in the isolated-ATOM converged potential; check units/sign
    against the module's NIST-LDA reference eigenvalues and known atomic core levels."""
    print("=" * 72)
    print("TIER 1 — atomic anchor (units/sign; isolated-atom V from atomic_scf)")
    print("=" * 72)
    # Known atomic 1s core levels (LDA, eV) for order-of-magnitude sanity.
    known_1s = {"Ne": -30.305 * HARTREE_EV, "O": None, "Ti": None}
    for sym in ("O", "Ne", "Ti"):
        eigs, v = atomic_scf(sym, _R, _DX)
        # Re-solve the SAME cores from the converged v via our extraction path.
        vbk = {"a0": v.numpy()}
        lv = core_levels_from_vbk(vbk, [sym])["a0"][1]
        print(f"\n  {sym} (Z={CONFIG[sym][0]:.0f}):")
        for orb, e in sorted(lv.items()):
            ref_int = eigs.get(orb)
            nist = NIST_LDA_EV.get(sym, {}).get(orb)
            msg = f"    {orb} = {e:11.3f} eV"
            if ref_int is not None:
                msg += f"   (atomic_scf internal: {ref_int:11.3f} eV, Δ={e-ref_int:+.2e})"
            if nist is not None:
                msg += f"   (NIST-LDA: {nist:11.3f}, Δ={e-nist:+.3f})"
            print(msg)
        if known_1s.get(sym):
            print(f"    known atomic {sym} 1s ≈ {known_1s[sym]:.1f} eV")


def _forsterite_atoms():
    from ase.spacegroup import crystal
    return crystal(
        ["Mg", "Mg", "Si", "O", "O", "O"],
        basis=[(0, 0, 0), (0.27751, 0.25, 0.99143), (0.09400, 0.25, 0.42627),
               (0.09133, 0.25, 0.76575), (0.44736, 0.25, 0.22150),
               (0.16333, 0.03317, 0.27751)],
        spacegroup=62, cellpar=[10.1971, 5.9806, 4.7540, 90, 90, 90])


def _run_crystal(cell_bohr, atoms_list, radii, **kw):
    from gradwave.flapw import crystal_scf_multi
    return crystal_scf_multi(cell_bohr, atoms_list, radii, **kw)


def tier_crystal(which: str = "twone") -> None:
    """Crystal plumbing + null test, then (optionally) a real inequivalent-site shift."""
    print("=" * 72)
    print(f"TIER 2/3 — crystal ({which})")
    print("=" * 72)
    if which == "twone":
        # Two symmetry-EQUIVALENT Ne in a cubic cell: null test (Δε ≈ 0), and a real
        # post-SCF v_by_key extraction on a converging crystal.
        cell = 12.0
        atoms = [((0.25, 0.25, 0.25), "Ne"), ((0.75, 0.75, 0.75), "Ne")]
        radii = {"Ne": 2.0}
        kw = dict(ecut=120.0, iters=20, kmesh=(1, 1, 1), use_symmetry=False)
    elif which == "neo":
        # Ne + O in one cell: two DIFFERENT elements share the interstitial reference; not a
        # same-element shift but proves distinct sites get distinct, well-separated core levels.
        cell = [10.0, 10.0, 10.0]
        atoms = [((0.2, 0.2, 0.2), "Ne"), ((0.7, 0.7, 0.7), "O")]
        radii = {"Ne": 1.3, "O": 1.3}
        kw = dict(ecut=140.0, iters=30, kmesh=(1, 1, 1), use_symmetry=False)
    elif which == "tio2":
        # An asymmetric TiO2 unit in a box: Ti(d0) with two O at DIFFERENT Ti-O distances, so the
        # two O are chemically inequivalent (a real initial-state O 1s chemical shift). Ti4+ + 2 O2-
        # is charge-neutral and non-magnetic. Both O share the one interstitial reference -> the
        # O1s(short)-O1s(long) shift is a clean within-cell difference. Bohr cell/coords.
        import numpy as _np
        L = 11.0
        BOHR = 0.529177210903
        ti = _np.array([0.5, 0.5, 0.5]) * L
        o1 = ti + _np.array([1.75 / BOHR, 0.0, 0.0])           # short Ti-O 1.75 A
        o2 = ti + _np.array([0.0, 2.30 / BOHR, 0.0])           # long  Ti-O 2.30 A
        atoms = [(tuple(ti / L), "Ti"), (tuple(o1 / L), "O"), (tuple(o2 / L), "O")]
        cell = float(L)
        radii = {"Ti": 0.90, "O": 0.70}
        kw = dict(ecut=140.0, iters=30, kmesh=(1, 1, 1), use_symmetry=False,
                  smearing=0.10, verbose=True)
    elif which == "betio3":
        # Tetragonally-distorted BeTiO3 perovskite (c/a>1): DENSE (fast), neutral
        # (Be2+ + Ti4+ + 3 O2-), non-magnetic (Ti4+ d0). The tetragonal distortion splits the
        # 3 perovskite O into 1 APICAL O (along c, at (1/2,1/2,0) rel. to Ti) + 2 EQUATORIAL O:
        # inequivalent same-element sites sharing one interstitial reference -> a real O 1s
        # initial-state chemical shift (apical vs equatorial). Elements all configured.
        import numpy as _np
        BOHR = 0.529177210903
        a, c = 3.90 / BOHR, 4.20 / BOHR                 # c/a = 1.077 (Bohr)
        cell = _np.diag([a, a, c])
        atoms = [((0.0, 0.0, 0.0), "Be"),               # A-site
                 ((0.5, 0.5, 0.5), "Ti"),               # B-site
                 ((0.5, 0.5, 0.0), "O"),                # apical O (Ti-O along c)
                 ((0.5, 0.0, 0.5), "O"),                # equatorial O
                 ((0.0, 0.5, 0.5), "O")]                # equatorial O
        radii = {"Be": 0.75, "Ti": 0.95, "O": 0.75}
        kw = dict(ecut=200.0, iters=40, kmesh=(2, 2, 2), use_symmetry=True,
                  smearing=0.05, verbose=True)
    else:
        raise ValueError(which)
    bands, info = _run_crystal(cell, atoms, radii, **kw)
    syms = [s for _, s in atoms]
    rec = info.get("recorder")
    rsum = rec.summarize() if rec is not None else {}
    print(f"  converged span d={rsum.get('d_span_eV')}  nbands={info['nbands']}")
    levels = core_levels_from_vbk(info["v_by_key"], syms)
    for k, (s, lv) in levels.items():
        print(f"  {k} {s}: " + ", ".join(f"{o}={e:.3f}" for o, e in sorted(lv.items())))
    shifts = _pairwise_shifts(levels)
    if shifts:
        print("  within-cell same-element core-level shifts:")
        print("\n".join(shifts))
    else:
        print("  (no two same-element sites in this cell)")


if __name__ == "__main__":
    tier = sys.argv[1] if len(sys.argv) > 1 else "atom"
    if tier in ("atom", "all"):
        tier_atom()
    if tier in ("crystal", "all"):
        tier_crystal("twone")
    if tier in ("neo", "all"):
        tier_crystal("neo")
    if tier in ("tio2", "all"):
        tier_crystal("tio2")
    if tier in ("betio3", "all"):
        tier_crystal("betio3")
