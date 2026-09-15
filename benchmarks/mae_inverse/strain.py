"""Inverse design of FePt magnetocrystalline anisotropy by tetragonal strain —
GRADIENT-DRIVEN.

L1₀ FePt is the workhorse high-anisotropy magnet; its MAE is a spin-orbit effect
that depends sharply on the tetragonal distortion c/a, so strain — epitaxial or
applied — is a design knob. This finds the c/a that MAXIMIZES MAE at fixed volume
by Newton root-finding on the ANALYTIC gradient dMAE/d(c/a)
(`postscf.mae.mae_strain_gradient`), rather than scanning c/a and fitting a
parabola to the samples.

Each step is one spinor SOC ground state, then the magnetic force theorem AND its
strain gradient in one shot: freeze (ρ, m⃗), diagonalize the two directions once
(postscf/mae.py), and differentiate the frozen-density band energy w.r.t. a
volume-conserving tetragonal strain by autograd — the exact Hellmann-Feynman
gradient, no spinor χ₀ needed (the energy is stationary at the SCF point). A
secant-Newton step on dMAE/d(c/a)=0 then converges in ~3 SCFs, versus the ~5-point
scan the parabola fit needed. The density converges on a coarse k-mesh (cheap) and
the force theorem + gradient are evaluated on a dense mesh (the MAE sign is
mesh-sensitive, so the anisotropy needs the fine quadrature). One FFT box per
volume — pinned across ratios so the frozen density transfers.

The gradient IS the inverse-design engine: this is the differentiability the code
advertises, doing real work rather than a scan dressed up as optimization.

    GW_DEVICE=cpu uv run python benchmarks/mae_inverse/strain.py
    GW_CHEAP=1     uv run python benchmarks/mae_inverse/strain.py   # fast smoke
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.mae import force_theorem_mae, mae_strain_gradient
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear

torch.set_num_threads(int(os.environ.get("GW_THREADS", "22")))
sys.stdout.reconfigure(line_buffering=True)

RY = 13.605693122994
PSE = Path(os.environ.get("GW_PSE",
           str(Path(__file__).parents[2] / "tests/fixtures/qe/pseudos")))
SP = Path(__file__).parent
FE = parse_upf(str(PSE / "Fe_ONCV_PBE_FR-1.0.upf"))
PT = parse_upf(str(PSE / "Pt_ONCV_PBE_FR-1.0.upf"))
FRAC = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]])

A0, C0 = 2.723, 3.712                 # L1₀ FePt reference [Å]; c/a = 1.363
V0 = A0 * A0 * C0                     # fixed-volume tetragonal distortion
INIT = [[0, 0, 3.0], [0, 0, 0.4]]     # Fe ~3 μB, induced Pt ~0.4, along c

CHEAP = os.environ.get("GW_CHEAP") == "1"
ECUT = (30 if CHEAP else 70) * RY
KMESH = (4, 4, 2) if CHEAP else (6, 6, 4)   # MAE sign needs the dense mesh
CA_START = 1.363                      # start at the reference c/a
GTOL = 1.0e-3                         # |dMAE/d(c/a)| convergence [meV per c/a]
STEP0 = 0.02                          # first (gradient-ascent) c/a step
MAX_STEPS = 8
xc = NoncollinearXC(LSDA_PW92())


def cell_of(ratio):
    a = (V0 / ratio) ** (1.0 / 3.0)
    return np.diag([a, a, ratio * a]), a, ratio * a


DEV = os.environ.get("GW_DEVICE", "cpu")
# one FFT box for every ratio (pin from the largest a = smallest ratio explored)
FFT = tuple(setup_system(cell_of(1.10)[0], FRAC @ cell_of(1.10)[0],
                         [0, 1], [FE, PT], ecut=ECUT, kmesh=KMESH).grid.shape)


def build(ratio):
    cell, _, _ = cell_of(ratio)
    s = setup_system(cell, FRAC @ cell, [0, 1], [FE, PT], ecut=ECUT,
                     kmesh=KMESH, nbands=30, use_symmetry=False,
                     time_reversal=False, fft_shape=FFT)
    return s.to(DEV) if DEV != "cpu" else s


def scf_at(ratio):
    """Converged spinor SOC ground state at tetragonal ratio ``ratio``.

    energy_metric=True converges on the residual's second-order energy error
    rather than the density residual: FePt is a magnetic metal whose
    magnetization-channel density residual floors above any reachable rhotol,
    so the plain rhotol gate can strand a fully-settled SCF at max_iter
    (see scf.noncollinear / docs). The energy error settles far below it."""
    res = scf_noncollinear(build(ratio), xc, mag_vec_init=INIT,
                           smearing="gaussian", width=0.1, etol=1e-9,
                           rhotol=1e-7, energy_metric=True, entol=1e-6,
                           max_iter=400, mixing_alpha=0.3, mixing_history=12,
                           mag_mixer="johnson", spin_precond=True,
                           verbose=False)
    assert res.converged, (ratio, "SCF not converged")
    return res


def mae_and_grad(ratio):
    """(MAE [meV], dMAE/d(c/a) [meV per c/a unit]) at fixed volume, plus |M|.

    One SCF, then the force-theorem MAE gradient. For the volume-conserving
    tetragonal strain ε = η·diag(-1,-1,2), c/a → c/a·(1+3η), so
    dMAE/d(c/a) = (2·g_zz − g_xx − g_yy)/(3·c/a) with g = ∂MAE/∂ε."""
    res = scf_at(ratio)
    mae_val, g = mae_strain_gradient(res, xc, hard_dir=(1, 0, 0),
                                     easy_dir=(0, 0, 1))
    dmae_deta = float(2 * g[2, 2] - g[0, 0] - g[1, 1])
    dmae_dca = dmae_deta / (3.0 * ratio)
    mag = float(np.linalg.norm(np.array(res.mag_vec)))
    return float(mae_val) * 1000.0, dmae_dca * 1000.0, mag


def main():
    print(f"FePt MAE inverse design by GRADIENT ascent (fixed V={V0:.3f} Å³, "
          f"FFT {FFT}, mesh {KMESH}, ecut {ECUT/RY:.0f} Ry, device {DEV})",
          flush=True)
    ca = CA_START
    ca_prev = g_prev = None
    rows = []
    t0 = time.time()
    for step in range(MAX_STEPS):
        try:
            mae, g, mag = mae_and_grad(ca)
        except AssertionError:
            # A hard magnetic-metal SCF at this strained cell did not converge.
            # Backtrack halfway toward the last good c/a and retry — an
            # inverse-design loop must tolerate a stray stiff cell rather than
            # abort. (The dense physical mesh is far more robust; this mainly
            # guards the coarse smoke run.)
            if not rows:
                raise
            print(f"  SCF stalled at c/a={ca:.4f}; backtracking", flush=True)
            ca = float(0.5 * (ca + rows[-1]["ca"]))
            continue
        rows.append(dict(step=step, ca=ca, mae=mae, dmae_dca=g, mag=mag))
        print(f"step {step}  c/a={ca:.4f}  MAE={mae:+.4f} meV  "
              f"dMAE/d(c/a)={g:+.3f} meV  |M|={mag:.3f} μB", flush=True)
        if abs(g) < GTOL:
            print(f"converged: |dMAE/d(c/a)| < {GTOL} meV", flush=True)
            break
        if g_prev is None:                       # gradient-ascent seed step
            ca_new = ca + STEP0 * np.sign(g)
        else:                                    # secant-Newton on g(c/a)=0
            gp = (g - g_prev) / (ca - ca_prev)
            ca_new = ca - g / gp if abs(gp) > 1e-12 else ca + STEP0 * np.sign(g)
        ca_prev, g_prev = ca, g
        ca = float(np.clip(ca_new, 1.10, 1.70))

    out = SP / "strain.json"
    out.write_text(json.dumps(dict(V0=V0, a0=A0, c0=C0, mesh=list(KMESH),
                                   ecut_ry=ECUT / RY, rows=rows), indent=1))
    ca_opt = rows[-1]["ca"]
    print(f"\nMAE-maximizing c/a ≈ {ca_opt:.4f} in {len(rows)} SCFs "
          f"(reference L1₀ c/a = {C0/A0:.3f}); {time.time()-t0:.0f}s", flush=True)
    if not CHEAP:
        # cross-check the located optimum against the force theorem at ±Δ: MAE
        # should be stationary (a scan would need many points to show this; the
        # gradient pins it in one triple). Skipped in the cheap smoke run.
        d = 0.03
        m_lo = force_theorem_mae(
            scf_at(ca_opt - d), xc, [[0, 0, 1.0], [1.0, 0, 0]], verbose=False)
        m_hi = force_theorem_mae(
            scf_at(ca_opt + d), xc, [[0, 0, 1.0], [1.0, 0, 0]], verbose=False)
        print(f"stationarity check  MAE(c/a-{d})={float(m_lo.mae[1])*1e3:+.4f}  "
              f"MAE(c/a+{d})={float(m_hi.mae[1])*1e3:+.4f} meV", flush=True)
    print("MAE_STRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
