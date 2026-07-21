"""Inverse design of FePt magnetocrystalline anisotropy by tetragonal strain.

L1₀ FePt is the workhorse high-anisotropy magnet; its MAE is what decides whether
a grain holds its magnetization. The anisotropy is a spin-orbit effect that
depends sharply on the tetragonal distortion c/a, so strain — epitaxial or
applied — is a design knob. This maps MAE(c/a) at fixed volume and finds the c/a
that maximizes it, i.e. the strain a substrate should impose to make the hardest
magnet.

Each point is one spinor SOC ground state and then the magnetic force theorem for
the two directions (postscf/mae.py): freeze (ρ, m⃗) of the [001] reference, rotate
the moment rigidly to [100], and difference the occupied band energies —
MAE = F[100] − F[001], positive = easy axis along c. The density converges on a
coarse k-mesh (cheap) and the force theorem is evaluated on a dense mesh (the MAE
sign is mesh-sensitive, so the anisotropy itself needs the fine quadrature). One
FFT box per volume — pinned across ratios so the frozen density transfers.

Runs on CPU: the 6 GB GPU cannot hold the 144-k spinor Hamiltonian at 70 Ry.

    GW_DEVICE=cpu uv run python benchmarks/mae_inverse/strain.py
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
from gradwave.postscf.mae import force_theorem_mae
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
ECUT = 70 * RY
KMESH = (6, 6, 4)                     # MAE sign needs the dense mesh (mae.md)
INIT = [[0, 0, 3.0], [0, 0, 0.4]]     # Fe ~3 μB, induced Pt ~0.4, along c
RATIOS = [1.20, 1.28, 1.363, 1.45, 1.55]
xc = NoncollinearXC(LSDA_PW92())


def cell_of(ratio):
    a = (V0 / ratio) ** (1.0 / 3.0)
    return np.diag([a, a, ratio * a]), a, ratio * a


DEV = os.environ.get("GW_DEVICE", "cpu")
# one FFT box for every ratio (pin from the largest a = smallest ratio)
FFT = tuple(setup_system(cell_of(min(RATIOS))[0], FRAC @ cell_of(min(RATIOS))[0],
                         [0, 1], [FE, PT], ecut=ECUT, kmesh=KMESH).grid.shape)


def build(ratio):
    cell, a, c = cell_of(ratio)
    s = setup_system(cell, FRAC @ cell, [0, 1], [FE, PT], ecut=ECUT,
                     kmesh=KMESH, nbands=30, use_symmetry=False,
                     time_reversal=False, fft_shape=FFT)
    return s.to(DEV) if DEV != "cpu" else s


def mae_at(ratio):
    t0 = time.time()
    res = scf_noncollinear(build(ratio), xc, mag_vec_init=INIT,
                           smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-7,
                           max_iter=300, mixing_alpha=0.3, mixing_history=12,
                           verbose=False)
    assert res.converged, (ratio, "SCF not converged")
    # force theorem on the same mesh; magmoms= folds each direction into its
    # own magnetic IBZ so the two solves are cheap
    ft = force_theorem_mae(res, xc, [[0, 0, 1.0], [1.0, 0, 0]],
                           magmoms=INIT, verbose=False)
    mae = float(ft.mae[1]) * 1000.0                       # F[100]−F[001], meV/cell
    m = float(np.linalg.norm(np.array(res.mag_vec)))
    dt = time.time() - t0
    print(f"c/a={ratio:.3f}  MAE={mae:+.4f} meV  |M|={m:.3f} μB  "
          f"it={res.n_iter}  ({dt:.0f}s)", flush=True)
    return dict(ratio=ratio, a=cell_of(ratio)[1], c=cell_of(ratio)[2],
                mae=mae, mag=m, sec=round(dt))


def main():
    print(f"FePt MAE vs tetragonal c/a (fixed V={V0:.3f} Å³, FFT {FFT}, "
          f"mesh {KMESH}, device {DEV})", flush=True)
    out = SP / "strain.json"
    prev = (json.loads(out.read_text())["rows"] if out.exists() else [])
    done = {round(r["ratio"], 3) for r in prev}
    rows = list(prev)
    for r in RATIOS:                                     # resumable per point
        if round(r, 3) in done:
            print(f"c/a={r:.3f} done (resume)", flush=True)
            continue
        rows.append(mae_at(r))
        out.write_text(json.dumps(dict(V0=V0, a0=A0, c0=C0, rows=rows), indent=1))
    # locate the MAE maximum from a quadratic fit (the inverse-design answer)
    r = np.array([x["ratio"] for x in rows])
    mae = np.array([x["mae"] for x in rows])
    p = np.polyfit(r, mae, 2)
    r_opt = -p[1] / (2 * p[0]) if p[0] < 0 else r[int(mae.argmax())]
    print(f"\nMAE-maximizing c/a ≈ {float(r_opt):.3f} "
          f"(reference L1₀ c/a = {C0/A0:.3f})", flush=True)
    print("MAE_STRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
