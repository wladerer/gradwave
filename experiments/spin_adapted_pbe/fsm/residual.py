"""Phase-3 residual analysis: is the leftover per-system moment error
correlated with a local ingredient a 2nd parameter could exploit?

For each system at its (unconstrained) equilibrium state under the FITTED
functional, compute on the SCF density:

  - ζ(r)  spin polarization,
  - s(r)  reduced total-density gradient,
  - α(r)  iso-orbital indicator (τ − τ_W)/τ_unif from the converged orbitals'
          kinetic-energy density (core.metagga.tau_b),
  - Δe_x(r) = e_xc^fit(r) − e_xc^PBE(r) evaluated on the SAME density — where
          the ζ² term actually acts,

and report means of s, ζ, α over the magnetization-carrying regions, weighted
by |m(r)| = |ρ↑−ρ↓| and by |Δe_xc(r)|. fit.py's per-system residuals are then
correlated against these in the results note (2nd-parameter verdict).

Run on asus (one PBE + one fitted SCF per system):
  uv run python experiments/spin_adapted_pbe/fsm/residual.py --mu1 <fitted>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# runnable as a plain script from the repo root (sys.path[0] is this file's
# directory, and `experiments` is not an installed package)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.spin_adapted_pbe.fsm.scan import SPECS, _log, build  # noqa: E402
from gradwave.constants import BOHR_ANG
from gradwave.core.metagga import tau_b
from gradwave.core.xc.learnable import LearnableSpinXZeta
from gradwave.dtypes import CDTYPE
from gradwave.scf.common import spin_sigmas
from gradwave.scf.loop import scf

SMEAR, WIDTH, MAX_ITER = "mp1", 0.1, 80


def _batched_coeffs(res, sp: int) -> torch.Tensor:
    """Pad the per-k coefficient list back into (nk, nb, npw_max)."""
    bk = res.system.batch
    ch = res.coeffs[sp]
    nk = len(ch)
    nb = max(c.shape[0] for c in ch)
    npw_max = int(bk.npw.max())
    out = torch.zeros(nk, nb, npw_max, dtype=CDTYPE)
    for ik, c in enumerate(ch):
        out[ik, : c.shape[0], : c.shape[1]] = c
    return out


def _wmean(x: torch.Tensor, w: torch.Tensor) -> float:
    return float((x * w).sum() / w.sum())


def analyze(key: str, mu1_fit: float) -> dict:
    system, start_mag = build(key)
    base = dict(nspin=2, smearing=SMEAR, width=WIDTH, start_mag=start_mag,
                max_iter=MAX_ITER, verbose=False)
    xc_pbe = LearnableSpinXZeta(kappa1=0.0, mu1=0.0)
    xc_fit = LearnableSpinXZeta(kappa1=0.0, mu1=mu1_fit)
    r_pbe = scf(system, xc_pbe, **base)
    r_fit = scf(system, xc_fit, start_from=r_pbe, **base)
    assert r_pbe.converged and r_fit.converged

    grid = system.grid
    vol = float(grid.volume)
    ru, rd = r_fit.rho_spin[0].detach(), r_fit.rho_spin[1].detach()
    c2 = 0.0 if system.rho_core is None else 0.5 * system.rho_core
    ru_x, rd_x = ru + c2, rd + c2  # NLCC split, as in the SCF energy assembly
    s_uu, s_dd, s_tt = spin_sigmas(ru_x, rd_x, xc_pbe, grid.g_cart)

    with torch.no_grad():
        e_pbe = xc_pbe.energy_density(ru_x, rd_x, s_uu, s_dd, s_tt)
        e_fit = xc_fit.energy_density(ru_x, rd_x, s_uu, s_dd, s_tt)
    de = (e_fit - e_pbe).abs()

    # local fields (atomic units where dimensionless)
    rho_au = torch.clamp((ru_x + rd_x) * BOHR_ANG**3, min=1e-12)
    zeta = ((ru_x - rd_x) / torch.clamp(ru_x + rd_x, min=1e-12)).clamp(-1, 1)
    grad_au = torch.sqrt(torch.clamp(s_tt, min=0.0) * BOHR_ANG**8)
    kf = (3.0 * math.pi**2 * rho_au) ** (1.0 / 3.0)
    s_red = grad_au / (2.0 * kf * rho_au)

    # iso-orbital indicator from the converged orbitals' tau
    tau = torch.zeros_like(ru)
    for sp in range(2):
        occ = r_fit.occupations[sp].detach()
        tau = tau + tau_b(_batched_coeffs(r_fit, sp), occ, system.kweights,
                          system.batch, grid.shape, vol)
    tau_au = torch.clamp(tau * BOHR_ANG**5, min=0.0)  # e/A^5 -> a.u.
    tau_w = grad_au**2 / (8.0 * rho_au)
    zc = zeta.abs()
    ds = 0.5 * ((1 + zc) ** (5.0 / 3.0) + (1 - zc) ** (5.0 / 3.0))
    tau_unif = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0) * rho_au ** (5.0 / 3.0) * ds
    alpha = torch.clamp(tau_au - tau_w, min=0.0) / torch.clamp(tau_unif, min=1e-12)

    m_abs = (ru - rd).abs()
    fields = {"s": s_red, "zeta_abs": zeta.abs(), "alpha": alpha}
    stats: dict = {
        "m_pbe": abs(float(r_pbe.mag_total)),
        "m_fit": abs(float(r_fit.mag_total)),
        "dExc_total_eV": float((e_fit - e_pbe).mean()) * vol,
    }
    for wname, w in (("m", m_abs), ("dexc", de)):
        for fname, f in fields.items():
            stats[f"{fname}_w{wname}"] = _wmean(f, w)
    # where does the zeta^2 correction act: fraction of |dExc| at high zeta
    tot = float(de.sum())
    stats["dexc_frac_zeta_gt_0.5"] = float(de[zeta.abs() > 0.5].sum()) / tot
    stats["dexc_frac_s_gt_1"] = float(de[s_red > 1.0].sum()) / tot
    stats["dexc_frac_alpha_gt_1.5"] = float(de[alpha > 1.5].sum()) / tot
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu1", type=float, required=True,
                    help="the fitted mu1 to analyze at")
    ap.add_argument("--systems", nargs="*", default=list(SPECS))
    ap.add_argument("--out",
                    default="experiments/spin_adapted_pbe/fsm/raw/residual.json")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "5")))
    out = {"mu1": args.mu1, "systems": {}}
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists():
        out = json.loads(outpath.read_text())
        out["mu1"] = args.mu1
    for key in args.systems:
        t0 = time.time()
        stats = analyze(key, args.mu1)
        out["systems"][key] = stats
        outpath.write_text(json.dumps(out, indent=2))
        _log(f"{key}: m_fit={stats['m_fit']:.4f} s_wm={stats['s_wm']:.3f} "
             f"zeta_wm={stats['zeta_abs_wm']:.3f} alpha_wm={stats['alpha_wm']:.3f} "
             f"({time.time() - t0:.0f}s)")
    _log("EXIT=0")


if __name__ == "__main__":
    main()
