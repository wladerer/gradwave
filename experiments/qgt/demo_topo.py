"""Chern numbers and Wannier-charge-center flow via link overlaps.

Milestone-2 demo: Fukui–Hatsugai–Suzuki plaquette fluxes and Wilson-loop WCC
flow (postscf.kgeometry_topo) on the QWZ Chern insulator across its phase
diagram, plus the trivial null on real Si (plane-wave states from a small
SCF, BZ-boundary links embedded by the Miller shift).

Run:  uv run python experiments/qgt/demo_topo.py   (~20 s, CPU)
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.postscf.kgeometry_topo import (  # noqa: E402
    BlochLinkStates,
    ModelLinkStates,
    chern_fhs,
    wcc_flow,
)
from gradwave.scf.loop import scf, setup_system  # noqa: E402
from tests.helpers import RY, si_fcc, si_upf  # noqa: E402

SX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


def qwz(u):
    def h(k):
        a, b = 2.0 * np.pi * k[0], 2.0 * np.pi * k[1]
        return (
            torch.sin(a).to(torch.complex128) * SX
            + torch.sin(b).to(torch.complex128) * SY
            + (u + torch.cos(a) + torch.cos(b)).to(torch.complex128) * SZ
        )

    return h


def wcc_ascii(flow, width=48):
    """One line per k_perp; '|' marks each Wannier center on [0,1)."""
    lines = []
    for kp, xs in zip(flow.k_perp, flow.wcc, strict=True):
        row = [" "] * width
        for x in xs:
            row[int(x * width) % width] = "|"
        lines.append(f"  k_perp={kp:4.2f}  [{''.join(row)}]")
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(2)

    print("QWZ model, lower band (C convention: C = (1/2π)∫Ω, Ω = −2 Im Q):")
    for u in (1.0, -1.0, 3.0):
        prov = ModelLinkStates(qwz(u), [0])
        c = chern_fhs(prov, 12, e1=(1, 0), e2=(0, 1), origin=(0, 0))
        f = wcc_flow(prov, e_loop=(1, 0), e_perp=(0, 1), origin=(0, 0),
                     n_loop=12, n_perp=12)
        phase = "topological" if c.chern else "trivial"
        print(f"  u = {u:+.1f}: C_FHS = {c.chern:+d} (residual {c.residual:.1e}), "
              f"C_WCC = {f.chern:+d} (residual {f.residual:.1e})  [{phase}]")

    print("\nWCC flow, u = +1 (winding −1: centers cross the cell once):")
    f = wcc_flow(ModelLinkStates(qwz(1.0), [0]), e_loop=(1, 0), e_perp=(0, 1),
                 origin=(0, 0), n_loop=12, n_perp=12)
    print(wcc_ascii(f))

    t0 = time.time()
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(1, 1, 1), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    print(f"\nSi SCF converged ({time.time() - t0:.1f} s); valence-group nulls:")

    prov = BlochLinkStates(res, [0, 1, 2, 3])
    c = chern_fhs(prov, 6, e1=(1, 0, 0), e2=(0, 1, 0), origin=(0.03, 0.06, 0.17))
    print(f"  FHS slice k3=0.17:          C = {c.chern} (residual {c.residual:.1e})")
    c2 = chern_fhs(BlochLinkStates(res, [0, 1, 2, 3]), 6, e1=(1, 0, 0),
                   e2=(0, 1, 0), origin=(0.63, 0.71, 0.17))
    print(f"  FHS straddling BZ boundary: C = {c2.chern} (residual {c2.residual:.1e})")
    f = wcc_flow(prov, e_loop=(1, 0, 0), e_perp=(0, 1, 0),
                 origin=(0.03, 0.06, 0.17), n_loop=6, n_perp=6)
    print(f"  WCC winding:                C = {f.chern} (residual {f.residual:.1e})")
    print("\nSi WCC flow (windingless — centers never cross):")
    print(wcc_ascii(f))


if __name__ == "__main__":
    main()
