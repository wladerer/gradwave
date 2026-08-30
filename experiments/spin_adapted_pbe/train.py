"""Train the ζ-dependent spin-adaptation parameters (κ₁, μ₁) of
``LearnableSpinXZeta`` to reproduce the experimental magnetic moments of the
itinerant 3d ferromagnets Fe, Ni, Co.

Closed-shell systems have ζ≡0 everywhere so the ζ² term vanishes and they are
EXACTLY PBE for any (κ₁, μ₁): the fit can only touch magnetic systems.

Phases (select with --phase; default runs the full campaign in one process so
converged states warm-start each other):

  kconv     k-mesh convergence of the plain-PBE (κ₁=μ₁=0) moment per system
  baseline  plain-PBE moment vs experiment at the converged mesh
  fit       least_squares (κ₁, μ₁) on residuals (m_i − m_i^exp), FD Jacobian
  loo       leave-one-out: refit on 2 systems, predict the 3rd
  sanity    Si/Al total energy PBE vs trained — must be bit-identical (ζ≡0)

All heavy SCFs run on asus. Every SCF has a hard iteration cap so a
non-converging metal fails fast rather than hanging.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.build import bulk
from scipy.optimize import minimize_scalar

from gradwave.api.system import build_system
from gradwave.constants import RY_EV
from gradwave.core.xc.learnable import LearnableSpinXZeta
from gradwave.inputs import Input
from gradwave.inputs.models import KPointsParams, SmearingParams
from gradwave.scf.loop import SCFResult, scf

ECUT_EV = 60.0 * RY_EV  # 60 Ry
PSEUDO_DIR = Path("tests/fixtures/qe/pseudos")
MAX_ITER = 80  # hard cap: a non-converging metal fails fast
SMEAR_TYPE = "mp1"  # Methfessel-Paxton order 1 (cleaner than gaussian per study)
SMEAR_WIDTH = 0.1  # eV


@dataclass
class SystemSpec:
    key: str
    symbol: str
    crystal: str  # 'bcc' | 'fcc'
    a: float  # Angstrom
    pseudo: str
    exp_moment: float  # muB
    start_mag: float  # seed moment fraction
    nbands: int
    kmesh: tuple[int, int, int] = (10, 10, 10)  # converged mesh (set by kconv)
    # populated at runtime
    _system: Any = field(default=None, repr=False)
    _warm: SCFResult | None = field(default=None, repr=False)


# Canonical converged setting (see kconv_probe / results note): 14³ Monkhorst-
# Pack, mp1 smearing, width 0.1 eV, ecut 60 Ry. At 14³ the plain-PBE moment is
# stable against an adjacent mesh to <=0.03 muB per system (Fe 12³/14³ agree to
# 1e-4, Ni 14³/16³ to 2e-3, Co 12³/14³ to 1e-3), so the fit corrects the
# functional, not a k-sampling artifact.
_KMESH = (14, 14, 14)
SYSTEMS: dict[str, SystemSpec] = {
    "Fe": SystemSpec("Fe", "Fe", "bcc", 2.87, "Fe_ONCV_PBE-1.2.upf", 2.22, 0.8, 16, _KMESH),
    "Ni": SystemSpec("Ni", "Ni", "fcc", 3.52, "PD_Ni_PBE.upf", 0.61, 0.4, 18, _KMESH),
    # fcc-Co, experimental moment ~1.60 muB (hcp is ~1.72; fcc chosen for the
    # 1-atom cubic cell). SG15 scalar-relativistic ONCV PBE pseudo.
    "Co": SystemSpec("Co", "Co", "fcc", 3.54, "Co_ONCV_PBE-1.0.upf", 1.60, 0.6, 17, _KMESH),
}

NONMAG = {  # closed-shell sanity systems (Si diamond, Al fcc)
    "Si": ("Si", "diamond", 5.43, "Si_ONCV_PBE-1.2.upf", (8, 8, 8)),
    "Al": ("Al", "fcc", 4.05, "Al_ONCV_PBE-1.2.upf", (10, 10, 10)),
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(spec: SystemSpec, kmesh: tuple[int, int, int] | None = None):
    km = kmesh or spec.kmesh
    atoms = bulk(spec.symbol, spec.crystal, a=spec.a)
    inp = Input(
        atoms=atoms,
        pseudo_dir=PSEUDO_DIR,
        pseudo_map={spec.symbol: spec.pseudo},
        ecut=ECUT_EV,
        xc="pbe",
        nspin=2,
        nbands=spec.nbands,
        kpoints=KPointsParams(mesh=km),
        smearing=SmearingParams(type=SMEAR_TYPE, width=SMEAR_WIDTH),
        start_mag={spec.symbol: spec.start_mag},
    )
    return build_system(inp)


def moment(
    spec: SystemSpec,
    kappa1: float,
    mu1: float,
    *,
    system=None,
    warm: SCFResult | None = None,
    verbose: bool = False,
) -> tuple[float, SCFResult]:
    """Converged |m_tot| [muB] for LearnableSpinXZeta(κ₁, μ₁) on `spec`.
    Returns (moment, converged result) so callers can warm-start the next eval.
    """
    sys_ = system if system is not None else spec._system
    xc = LearnableSpinXZeta(kappa1=float(kappa1), mu1=float(mu1))
    res = scf(
        sys_, xc, nspin=2,
        smearing=SMEAR_TYPE, width=SMEAR_WIDTH,
        start_mag=[spec.start_mag], max_iter=MAX_ITER,
        start_from=warm, verbose=verbose,
    )
    return abs(float(res.mag_total)), res


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def phase_kconv(out: dict) -> None:
    _log("PHASE kconv: k-mesh convergence of plain-PBE moment")
    meshes = {"Fe": [(8, 8, 8), (10, 10, 10), (12, 12, 12)],
              "Ni": [(8, 8, 8), (10, 10, 10), (12, 12, 12)],
              "Co": [(8, 8, 8), (10, 10, 10), (12, 12, 12)]}
    res: dict[str, dict[str, float]] = {}
    for key, spec in SYSTEMS.items():
        res[key] = {}
        for km in meshes[key]:
            sysm = build(spec, km)
            m, r = moment(spec, 0.0, 0.0, system=sysm, warm=None)
            res[key][str(km)] = m
            _log(f"  {key} k={km}: |m|={m:.4f} muB")
        out.setdefault("kconv", {})[key] = res[key]


def _prepare(keys: list[str]) -> None:
    """Build systems (at their converged mesh) and cache a plain-PBE warm state."""
    for key in keys:
        spec = SYSTEMS[key]
        if spec._system is None:
            _log(f"  building {key} at k={spec.kmesh} ...")
            spec._system = build(spec)
        if spec._warm is None:
            m, r = moment(spec, 0.0, 0.0)
            spec._warm = r
            _log(f"  {key} PBE baseline |m|={m:.4f} (exp {spec.exp_moment})")


def phase_baseline(out: dict, keys: list[str]) -> None:
    _log("PHASE baseline: plain-PBE moment vs experiment")
    _prepare(keys)
    tab = {}
    for key in keys:
        spec = SYSTEMS[key]
        m = abs(float(spec._warm.mag_total))
        tab[key] = {"pbe": m, "exp": spec.exp_moment, "err": m - spec.exp_moment}
        _log(f"  {key}: PBE={m:.4f}  exp={spec.exp_moment:.4f}  err={m - spec.exp_moment:+.4f}")
    out["baseline"] = tab


# The fit is effectively one-dimensional in μ₁: κ₁ is inert because PBE already
# saturates the Lieb-Oxford bound (κ₀=0.804), so κ(ζ)=κ₀+κ₁ζ² is clamped for
# κ₁>0 and only weakly (dm/dκ₁≈0.03, ~60× below μ₁) affects the moment for κ₁<0
# (verified in PR #407 and by _KAPPA1_PROBE below). We therefore pin κ₁=0 and
# fit μ₁ against a per-system RESPONSE SURFACE m_i(μ₁): sample each system's
# moment on a μ₁ grid ONCE (warm-started along the chain), fit a smooth quadratic
# m_i(μ₁), then solve the full fit and every leave-one-out fit ANALYTICALLY from
# the same samples — noise-robust (the quadratic averages SCF moment jitter) and
# cheap (LOO needs no new SCFs).
_MU1_GRID = [0.0, -0.02, -0.04, -0.06, -0.08, -0.10]
_KAPPA1_PROBE = [-0.10, -0.20]  # at μ₁=-0.06, to record κ₁'s (weak) sensitivity


def sample_response(keys: list[str]) -> dict[str, dict[str, float]]:
    """m_i(μ₁) on _MU1_GRID (κ₁=0) for each system, warm-started along the grid.
    Returns {key: {str(mu1): moment}}."""
    _prepare(keys)
    samples: dict[str, dict[str, float]] = {}
    for key in keys:
        spec = SYSTEMS[key]
        samples[key] = {}
        warm = spec._warm  # μ₁=0 converged PBE state
        for mu1 in _MU1_GRID:
            if mu1 == 0.0:
                m = abs(float(spec._warm.mag_total))
                warm = spec._warm
            else:
                m, warm = moment(spec, 0.0, mu1, warm=warm)
            samples[key][f"{mu1:.3f}"] = m
        _log(f"  {key} m(μ1): "
             + " ".join(f"{mu1:+.2f}:{samples[key][f'{mu1:.3f}']:.4f}"
                        for mu1 in _MU1_GRID))
    return samples


def _response_model(sample: dict[str, float]):
    """Quadratic m(μ₁) fit to one system's grid samples; returns a poly1d."""
    xs = np.array([float(k) for k in sample])
    ys = np.array([sample[k] for k in sample])
    return np.poly1d(np.polyfit(xs, ys, 2))


def fit_from_samples(keys, samples, weights=None) -> dict:
    """Minimize Σ w_i (m_i(μ₁) − exp_i)² over μ₁ (κ₁=0) using the per-system
    quadratic response models — analytic, no SCFs."""
    weights = weights or dict.fromkeys(keys, 1.0)
    models = {k: _response_model(samples[k]) for k in keys}
    lo, hi = min(_MU1_GRID), max(_MU1_GRID)

    def loss(mu1):
        return sum(weights[k] * (models[k](mu1) - SYSTEMS[k].exp_moment) ** 2
                   for k in keys)

    sol = minimize_scalar(loss, bounds=(lo, hi), method="bounded")
    mu1 = float(sol.x)
    return {"kappa1": 0.0, "mu1": mu1, "loss": float(sol.fun), "keys": keys,
            "predicted": {k: float(models[k](mu1)) for k in keys}}


def phase_fit(out: dict, keys: list[str]) -> None:
    _log("PHASE fit: sample response surface m(μ1), then solve (κ1=0, μ1)")
    samples = sample_response(keys)
    out["response_samples"] = samples
    # record κ₁'s (weak) sensitivity on Fe to justify pinning κ₁=0
    kprobe = {}
    spec = SYSTEMS["Fe"]
    m0, warm = moment(spec, 0.0, -0.06, warm=spec._warm)
    kprobe["0.00"] = m0
    for k1 in _KAPPA1_PROBE:
        mk, _ = moment(spec, k1, -0.06, warm=warm)
        kprobe[f"{k1:.2f}"] = mk
    out["kappa1_probe_Fe_at_mu1_-0.06"] = kprobe
    _log("  Fe κ1 probe @ μ1=-0.06: "
         + " ".join(f"κ1={k}:{v:.4f}" for k, v in kprobe.items()))

    res = fit_from_samples(keys, samples)
    _log(f"  FIT DONE κ1={res['kappa1']:+.5f} μ1={res['mu1']:+.5f} "
         f"loss={res['loss']:.3e}")
    # confirm the analytic solution with a real SCF per system at μ1*
    tab = {}
    for key in keys:
        spec = SYSTEMS[key]
        m, _ = moment(spec, res["kappa1"], res["mu1"], warm=spec._warm)
        pbe = abs(float(spec._warm.mag_total))
        tab[key] = {"pbe": pbe, "trained": m, "model": res["predicted"][key],
                    "exp": spec.exp_moment}
        _log(f"  {key}: PBE={pbe:.4f}  trained(SCF)={m:.4f}  "
             f"model={res['predicted'][key]:.4f}  exp={spec.exp_moment:.4f}")
    res["moments"] = tab
    out["fit"] = res


def phase_loo(out: dict) -> None:
    _log("PHASE loo: leave-one-out transferability (from the same samples)")
    keys = list(SYSTEMS)
    samples = out.get("response_samples")
    if samples is None:
        samples = sample_response(keys)
        out["response_samples"] = samples
    loo = {}
    for held in keys:
        train_keys = [k for k in keys if k != held]
        res = fit_from_samples(train_keys, samples)
        # predict held-out from ITS response model at the trained μ1, and confirm
        # with a real SCF
        model = _response_model(samples[held])
        pred_model = float(model(res["mu1"]))
        spec = SYSTEMS[held]
        pred_scf, _ = moment(spec, res["kappa1"], res["mu1"], warm=spec._warm)
        loo[held] = {
            "trained_on": train_keys, "kappa1": res["kappa1"], "mu1": res["mu1"],
            "predicted_scf": pred_scf, "predicted_model": pred_model,
            "exp": spec.exp_moment, "err": pred_scf - spec.exp_moment,
        }
        _log(f"  held-out {held}: predict(SCF)={pred_scf:.4f} "
             f"model={pred_model:.4f}  exp={spec.exp_moment:.4f}  "
             f"err={pred_scf - spec.exp_moment:+.4f}  (μ1={res['mu1']:+.4f})")
    out["loo"] = loo


def phase_sanity(out: dict, params: dict) -> None:
    _log("PHASE sanity: closed-shell Si/Al PBE vs trained (must be identical)")
    from gradwave.core.xc.pbe import PBE

    k1, m1 = params["kappa1"], params["mu1"]
    tab = {}
    for key, (sym, cryst, a, pseudo, km) in NONMAG.items():
        atoms = bulk(sym, cryst, a=a)
        inp = Input(atoms=atoms, pseudo_dir=PSEUDO_DIR, pseudo_map={sym: pseudo},
                    ecut=ECUT_EV, xc="pbe", nspin=1, kpoints=KPointsParams(mesh=km),
                    smearing=SmearingParams(type="mp1", width=0.1))
        sysm = build_system(inp)
        # plain PBE (charge-only) and the trained spin functional at nspin=1.
        # nspin=1 -> ζ=0 identically, so LearnableSpinXZeta must equal PBE.
        e_pbe = float(scf(sysm, PBE(), smearing="mp1", width=0.1,
                          max_iter=MAX_ITER, verbose=False).energies.free_energy)
        # trained functional forced through nspin=1: closed shell -> PBE exactly
        from gradwave.core.xc.spin import SpinPBE
        e_spin_pbe = float(scf(sysm, SpinPBE(), nspin=2, smearing="mp1",
                               width=0.1, start_mag=[0.0], max_iter=MAX_ITER,
                               verbose=False).energies.free_energy)
        e_trained = float(scf(sysm, LearnableSpinXZeta(kappa1=k1, mu1=m1), nspin=2,
                              smearing="mp1", width=0.1, start_mag=[0.0],
                              max_iter=MAX_ITER,
                              verbose=False).energies.free_energy)
        tab[key] = {"pbe_nspin1": e_pbe, "spinpbe_nspin2": e_spin_pbe,
                    "trained_nspin2": e_trained,
                    "d_trained_vs_spinpbe": abs(e_trained - e_spin_pbe)}
        _log(f"  {key}: E_SpinPBE={e_spin_pbe:.10f}  E_trained={e_trained:.10f}  "
             f"|Δ|={abs(e_trained - e_spin_pbe):.2e} eV")
    out["sanity"] = tab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "main", "kconv", "baseline", "fit", "loo",
                             "sanity"])
    ap.add_argument("--out", default="experiments/spin_adapted_pbe/results.json")
    args = ap.parse_args()

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "8")))
    out: dict[str, Any] = {}
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists():
        out = json.loads(outpath.read_text())

    def save():
        outpath.write_text(json.dumps(out, indent=2))

    keys = list(SYSTEMS)
    if args.phase in ("all", "kconv"):
        phase_kconv(out)
        save()
    if args.phase in ("all", "main", "baseline"):
        phase_baseline(out, keys)
        save()
    if args.phase in ("all", "main", "fit"):
        phase_fit(out, keys)
        save()
    if args.phase in ("all", "main", "loo"):
        phase_loo(out)
        save()
    if args.phase in ("all", "main", "sanity"):
        params = out.get("fit", {"kappa1": 0.0, "mu1": -0.05})
        phase_sanity(out, params)
        save()
    _log("DONE")
    save()


if __name__ == "__main__":
    main()
