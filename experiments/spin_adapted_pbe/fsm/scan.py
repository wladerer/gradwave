"""Fixed-spin-moment E(M) scans for the spin-adapted-PBE training campaign.

For each system and each spin-adaptation strength μ₁ ∈ {0, −0.06, −0.10}
(κ₁ = 0 throughout — dead handle, LO-clamped, established in #408):

  1. an unconstrained smeared SCF (the free moment m_free and free energy), then
  2. a fixed-spin-moment E(M) sweep over the system's M grid using the smeared
     FSM mode (tot_magnetization with smearing → two Fermi levels, PR branch),
     warm-start-chained outward from the free solution.

Each point records the free energy F(M), the pinned moment, both Fermi levels
(μ↑ − μ↓ = 2·∂F/∂M is the constraining field), iterations, and convergence.
Raw per-(system, μ₁) JSON goes to --out; fit.py assembles curves → m*(μ₁)
interpolants → the μ₁ refit + LOO.

Settings mirror #408's converged canon: ecut 60 Ry, mp1 smearing 0.1 eV,
14³ MP for the 1-atom elements; FeCo (B2, 2 atoms) and Co2MnSi (L2₁, 4 atoms)
carry their own meshes with a --kconv probe to justify them.

Run per system (parallelize across systems, NOT within one):
  uv run python experiments/spin_adapted_pbe/fsm/scan.py --system Fe
  uv run python experiments/spin_adapted_pbe/fsm/scan.py --system Co2MnSi --kconv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from gradwave.api.system import build_system
from gradwave.constants import RY_EV
from gradwave.core.xc.learnable import LearnableSpinXZeta
from gradwave.inputs import Input
from gradwave.inputs.models import KPointsParams, SmearingParams
from gradwave.scf.loop import scf

ECUT_EV = 60.0 * RY_EV
PSEUDO_DIR = Path("tests/fixtures/qe/pseudos")
MAX_ITER = 80  # hard cap: fail fast, never hang
SMEAR = "mp1"
WIDTH = 0.1  # eV
MU1_GRID = [0.0, -0.06, -0.10]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _heusler_co2mnsi(a: float) -> Atoms:
    """L2₁ full-Heusler Co₂MnSi: fcc lattice, 4-atom primitive cell.
    Cubic-cell fractions: Si (0,0,0), Co (¼,¼,¼) & (¾,¾,¾), Mn (½,½,½)."""
    prim = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    frac_cubic = np.array([
        [0.25, 0.25, 0.25],  # Co
        [0.75, 0.75, 0.75],  # Co
        [0.50, 0.50, 0.50],  # Mn
        [0.00, 0.00, 0.00],  # Si
    ])
    return Atoms("CoCoMnSi", positions=frac_cubic * a, cell=prim, pbc=True)


# exp_moment sources (per CELL): Fe/Ni/Co as in #408 (spontaneous moments,
# Landolt-Börnstein; fcc-Co 1.60). FeCo B2 (equiatomic, ordered): 4.60 μB/cell
# adopted — neutron site moments μ_Fe → ~3.0, μ_Co ≈ 1.7–1.8 (Collins &
# Forsyth, Phil. Mag. 8, 401 (1963)) and mean-magnetization ~2.3 μB/atom
# (Bardos, J. Appl. Phys. 40, 1371 (1969)); literature spread 4.5–4.7 noted in
# the results note. Co2MnSi: 5.00 μB/f.u. (Slater-Pauling integer; polarized
# neutron 5.0, Brown et al., J. Phys.: Condens. Matter 12, 1827 (2000)).
SPECS: dict[str, dict] = {
    "Fe": dict(atoms=lambda: bulk("Fe", "bcc", a=2.87),
               pseudos={"Fe": "Fe_ONCV_PBE-1.2.upf"}, exp=2.22,
               start_mag={"Fe": 0.8}, kmesh=(14, 14, 14),
               mgrid=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]),
    "Ni": dict(atoms=lambda: bulk("Ni", "fcc", a=3.52),
               pseudos={"Ni": "PD_Ni_PBE.upf"}, exp=0.61,
               start_mag={"Ni": 0.4}, kmesh=(14, 14, 14),
               mgrid=[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]),
    "Co": dict(atoms=lambda: bulk("Co", "fcc", a=3.54),
               pseudos={"Co": "Co_ONCV_PBE-1.0.upf"}, exp=1.60,
               start_mag={"Co": 0.6}, kmesh=(14, 14, 14),
               mgrid=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]),
    "FeCo": dict(atoms=lambda: bulk("FeCo", "cesiumchloride", a=2.857),
                 pseudos={"Fe": "Fe_ONCV_PBE-1.2.upf",
                          "Co": "Co_ONCV_PBE-1.0.upf"}, exp=4.60,
                 start_mag={"Fe": 0.8, "Co": 0.6}, kmesh=(12, 12, 12),
                 kconv=[(12, 12, 12), (14, 14, 14)],
                 mgrid=[3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.1, 5.2]),
    "Co2MnSi": dict(atoms=lambda: _heusler_co2mnsi(5.654),
                    pseudos={"Co": "Co_ONCV_PBE-1.0.upf",
                             "Mn": "PD_Mn_PBE.upf",
                             "Si": "Si_ONCV_PBE-1.2.upf"}, exp=5.00,
                    # physical-scale seeds (start_mag is a fraction of Z_val:
                    # Mn 0.25·15 ≈ 3.8 μB, Co 0.1·17 ≈ 1.7 μB) — the elemental
                    # overspin-hard trick seeds ~12 μB here and can land the
                    # free run in a wrong (ferri) basin
                    start_mag={"Co": 0.1, "Mn": 0.25, "Si": 0.0},
                    kmesh=(10, 10, 10), kconv=[(10, 10, 10), (12, 12, 12)],
                    # dense near the Slater-Pauling integer (half-metal kink)
                    mgrid=[4.4, 4.7, 4.9, 4.95, 5.0, 5.05, 5.1, 5.3]),
}


def build(key: str, kmesh=None):
    spec = SPECS[key]
    atoms = spec["atoms"]()
    n_val = 0.0
    from gradwave.pseudo.upf import parse_upf
    zmap = {el: parse_upf(str(PSEUDO_DIR / up)).z_valence
            for el, up in spec["pseudos"].items()}
    n_val = sum(zmap[s] for s in atoms.get_chemical_symbols())
    mmax = max(abs(m) for m in spec["mgrid"])
    nbands = int(np.ceil((n_val + mmax) / 2.0)) + 8
    inp = Input(
        atoms=atoms, pseudo_dir=PSEUDO_DIR, pseudo_map=spec["pseudos"],
        ecut=ECUT_EV, xc="pbe", nspin=2, nbands=nbands,
        kpoints=KPointsParams(mesh=tuple(kmesh or spec["kmesh"])),
        smearing=SmearingParams(type=SMEAR, width=WIDTH),
        start_mag=spec["start_mag"],
    )
    system = build_system(inp)
    start_mag = [spec["start_mag"][s] for s in atoms.get_chemical_symbols()]
    _log(f"{key}: {len(atoms)} atoms, N_e={n_val:.0f}, nbands={nbands}, "
         f"k={kmesh or spec['kmesh']}, {len(system.kweights)} IBZ k")
    return system, start_mag


def _point(res) -> dict:
    mu_up, mu_dn = (res.fermi_spin if res.fermi_spin is not None
                    else (res.fermi, res.fermi))
    return {
        "converged": bool(res.converged), "n_iter": int(res.n_iter),
        "F": float(res.energies.free_energy),
        "E_total": float(res.energies.total),
        "m": float(res.mag_total), "mag_abs": float(res.mag_abs),
        "mu_up": float(mu_up), "mu_dn": float(mu_dn),
        "fermi": float(res.fermi),
    }


def run_system(key: str, out_dir: Path, mu1_list: list[float],
               kmesh=None, tag: str = "") -> None:
    spec = SPECS[key]
    system, start_mag = build(key, kmesh=kmesh)
    base = dict(nspin=2, smearing=SMEAR, width=WIDTH, start_mag=start_mag,
                max_iter=MAX_ITER, verbose=False)
    free_prev = None
    for mu1 in mu1_list:
        t0 = time.time()
        xc = LearnableSpinXZeta(kappa1=0.0, mu1=float(mu1))
        rec: dict = {"system": key, "mu1": mu1, "exp": spec["exp"],
                     "settings": {"ecut_Ry": 60.0, "smearing": SMEAR,
                                  "width_eV": WIDTH,
                                  "kmesh": list(kmesh or spec["kmesh"]),
                                  "max_iter": MAX_ITER}}
        # 1. unconstrained SCF (warm-started along the mu1 chain)
        r_free = scf(system, xc, start_from=free_prev, **base)
        rec["free"] = _point(r_free)
        free_prev = r_free
        m0 = float(r_free.mag_total)
        _log(f"{key} mu1={mu1:+.2f} FREE: m={m0:.4f} "
             f"F={rec['free']['F']:.6f} conv={r_free.converged} "
             f"it={r_free.n_iter}")
        # 2. E(M) sweep, chained outward from the free moment
        ms = sorted(spec["mgrid"])
        i0 = int(np.argmin([abs(m - abs(m0)) for m in ms]))
        order = ms[i0:] + ms[:i0][::-1]  # rightward from nearest, then leftward
        warm_right = warm_left = r_free
        pts = {}
        for m in order:
            warm = warm_right if m >= ms[i0] else warm_left
            r = scf(system, xc, tot_magnetization=float(m), start_from=warm,
                    **base)
            pts[f"{m:.3f}"] = _point(r)
            if m >= ms[i0]:
                warm_right = r
            else:
                warm_left = r
            p = pts[f"{m:.3f}"]
            _log(f"  {key} mu1={mu1:+.2f} M={m:.3f}: F={p['F']:.6f} "
                 f"dmu={p['mu_up'] - p['mu_dn']:+.4f} conv={p['converged']} "
                 f"it={p['n_iter']}")
        rec["curve"] = pts
        rec["wall_s"] = time.time() - t0
        out = out_dir / f"{key}{tag}_mu1_{mu1:+.3f}.json"
        out.write_text(json.dumps(rec, indent=2))
        _log(f"{key} mu1={mu1:+.2f} DONE ({rec['wall_s']:.0f}s) -> {out}")


def run_refine(key: str, out_dir: Path, mu1_list: list[float],
               delta: float = 0.05) -> None:
    """Densify existing curves near the minimum: two extra FSM points at
    M = m_free ± delta per (system, mu1), merged into the existing raw JSON.
    The 0.15–0.3 μB production grids limit the field-root interpolation to
    ~0.05 μB on the stiffer curves; these two warm points pin the crossing."""
    spec = SPECS[key]  # noqa: F841  (validates the key)
    system, start_mag = build(key)
    base = dict(nspin=2, smearing=SMEAR, width=WIDTH, start_mag=start_mag,
                max_iter=MAX_ITER, verbose=False)
    warm = None
    for mu1 in mu1_list:
        path = out_dir / f"{key}_mu1_{mu1:+.3f}.json"
        rec = json.loads(path.read_text())
        m0 = abs(rec["free"]["m"])
        xc = LearnableSpinXZeta(kappa1=0.0, mu1=float(mu1))
        for m in (m0 - delta, m0 + delta):
            r = scf(system, xc, tot_magnetization=float(m), start_from=warm,
                    **base)
            warm = r
            rec["curve"][f"{m:.3f}"] = _point(r)
            p = rec["curve"][f"{m:.3f}"]
            _log(f"  {key} mu1={mu1:+.2f} REFINE M={m:.3f}: F={p['F']:.6f} "
                 f"dmu={p['mu_up'] - p['mu_dn']:+.4f} conv={p['converged']} "
                 f"it={p['n_iter']}")
        path.write_text(json.dumps(rec, indent=2))
        _log(f"{key} mu1={mu1:+.2f} REFINED -> {path}")


def run_kconv(key: str, out_dir: Path) -> None:
    """Unconstrained PBE moment at the candidate meshes (moment stability)."""
    spec = SPECS[key]
    xc = LearnableSpinXZeta(kappa1=0.0, mu1=0.0)
    rec = {}
    for km in spec.get("kconv", [spec["kmesh"]]):
        system, start_mag = build(key, kmesh=km)
        r = scf(system, xc, nspin=2, smearing=SMEAR, width=WIDTH,
                start_mag=start_mag, max_iter=MAX_ITER, verbose=False)
        rec[str(tuple(km))] = _point(r)
        _log(f"{key} kconv {km}: m={float(r.mag_total):.4f} "
             f"conv={r.converged} it={r.n_iter}")
    out = out_dir / f"{key}_kconv.json"
    out.write_text(json.dumps(rec, indent=2))
    _log(f"{key} kconv DONE -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SPECS))
    ap.add_argument("--out", default="experiments/spin_adapted_pbe/fsm/raw")
    ap.add_argument("--kconv", action="store_true",
                    help="run only the k-mesh moment-stability probe")
    ap.add_argument("--refine", action="store_true",
                    help="add two FSM points at m_free±0.05 to existing curves")
    ap.add_argument("--mu1", type=float, nargs="*", default=None,
                    help="override the mu1 list (default 0, -0.06, -0.10)")
    ap.add_argument("--kmesh", type=int, default=None,
                    help="override the mesh (n,n,n) — smoke tests only")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "5")))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    km = (args.kmesh,) * 3 if args.kmesh else None
    if args.kconv:
        run_kconv(args.system, out_dir)
    elif args.refine:
        run_refine(args.system, out_dir, args.mu1 or MU1_GRID)
    else:
        run_system(args.system, out_dir, args.mu1 or MU1_GRID, kmesh=km,
                   tag=f"_k{args.kmesh}" if args.kmesh else "")
    _log("EXIT=0")


if __name__ == "__main__":
    main()
