"""Magnetocrystalline anisotropy of L1_0 FePt from two non-collinear SOC SCFs.

Measured (asus CPU, LSDA, 70 Ry, fully-relativistic SG15 Fe/Pt, gaussian 0.1 eV):
    kmesh (4,4,3) =  48 k:  MAE = -1.392 meV/cell -> easy axis [100]  (WRONG:
                            Fermi-surface sampling error flips the sign at coarse k)
    kmesh (6,6,4) = 144 k:  MAE = +2.552 meV/cell (+1.28 meV/atom) -> easy axis
                            [001], correct for FePt, magnitude in the literature
                            band (~1-3 meV/f.u.)
Both orientations converge to |dE| ~ 1e-11 eV (five orders below the signal) in
~26-28 iterations; |M| = 3.22-3.25 muB. The sign flip from 48 -> 144 k is the
textbook MAE k-convergence lesson: never trust an anisotropy sign from a coarse
mesh, and keep the SAME full mesh for both orientations so the k-error cancels
in the difference (see the magnetic-space-groups note in docs/ideas.md).

Magnetic-IBZ update (asus CPU, same physics, measured 2026-07-18): with
setup_system(..., use_symmetry=True, magmoms=...) each orientation folds by its
OWN magnetic (Shubnikov) group — [001] keeps C4h + 8 anti-unitary ops (144 -> 30 k),
[100] keeps C2h + 4 (144 -> 48 k) — and reproduces MAE = +2.5520 meV/cell to all
printed digits in 486 s + 815 s (~22 min vs ~78 min full-mesh, 3.6x). The fold is
exact (the magnetic-IBZ sum IS the full-mesh sum re-weighted, validated to 5e-11 eV
in tests/integration/test_magnetic_ibz.py), so per-orientation reduction preserves
the common-mode k-error cancellation as long as both use the same underlying mesh.

L1_0 FePt (alternating Fe/Pt layers along c) is the textbook high-MAE magnet: easy
axis along c, MAE ~ 2-3 meV/f.u. in DFT — orders of magnitude above the ~0.2 ueV
rotation-invariance precision floor, so it is the right first validation of the
spin-orbit path on a magnetic system. MAE = E([100]) - E([001]); positive = easy
axis c. The two orientations share the underlying k-mesh, so the k-sampling error
cancels in the difference.

SOC metals converge slowly and the density residual floors at occupation noise, so
we gate on the free-energy tail: the run prints the last few energies and the final
|dE| per orientation. The MAE is only trustworthy if both energies are settled well
below the meV signal and the two |M| magnitudes match (equal-magnitude moments,
different direction)."""
import time

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear

PSE = "tests/fixtures/qe/pseudos"
RY = 13.605693122994
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)
fe = parse_upf(f"{PSE}/Fe_ONCV_PBE_FR-1.0.upf")
pt = parse_upf(f"{PSE}/Pt_ONCV_PBE_FR-1.0.upf")
a, c = 2.723, 3.712                         # L1_0 FePt tetragonal [Å]
cell = np.diag([a, a, c])
pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
KMESH = (6, 6, 4)


def energy(axis, tag):
    ax = np.array(axis, float)
    init = [(3.0 * ax).tolist(), (0.4 * ax).tolist()]   # Fe ~3, Pt induced ~0.4
    # magnetic (Shubnikov) symmetry: fold k into the magnetic IBZ of THIS
    # orientation ([001] -> 30 k, [100] -> 48 k from the 144 full mesh) and
    # re-symmetrize (rho, m) each iteration. Exact vs full mesh to ~5e-11 eV.
    system = setup_system(cell, pos, [0, 1], [fe, pt], ecut=70 * RY, kmesh=KMESH,
                          nbands=30, use_symmetry=True, magmoms=init)
    if dev != "cpu":
        system = system.to(dev)
    print(f"[{tag}] nk={len(system.spheres)} (full={np.prod(KMESH)})", flush=True)
    t = time.time()
    res = scf_noncollinear(system, NoncollinearXC(LSDA_PW92()), mag_vec_init=init,
                           smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-7,
                           max_iter=300, mixing_alpha=0.3, mixing_history=12,
                           verbose=True)
    E = float(res.energies.free_energy)
    tail = [round(h["free_energy"], 9) for h in res.history[-5:]]
    dE = abs(res.history[-1]["free_energy"] - res.history[-2]["free_energy"])
    mv = np.array(res.mag_vec)
    print(f"[{tag}] conv={res.converged} n_it={res.n_iter} {time.time()-t:.0f}s "
          f"|M|={np.linalg.norm(mv):.4f}", flush=True)
    print(f"   E_tail={tail}  |dE_last|={dE:.2e} eV  M={[round(float(x),3) for x in mv]}",
          flush=True)
    return E, dE


print(f"device={dev}  L1_0 FePt  kmesh={KMESH}  ecut=70Ry", flush=True)
E001, d001 = energy([0, 0, 1.0], "001 c-axis")
E100, d100 = energy([1.0, 0, 0], "100 in-plane")
mae = (E100 - E001) * 1000
print(f"\nMAE = E[100]-E[001] = {mae:+.4f} meV/cell  ({mae/2:+.4f} meV/atom)", flush=True)
print(f"energy settle: |dE| {d001:.1e}, {d100:.1e} eV  (must be << |MAE|={abs(mae):.3f} meV)",
      flush=True)
print(f"easy axis: {'c-axis [001] (correct for FePt)' if mae > 0 else '[100] (!?)'}",
      flush=True)
print("FEPT_MAE_DONE", flush=True)
