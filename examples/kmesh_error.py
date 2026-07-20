"""Estimate the k-point (Brillouin-zone) sampling error from a mesh sweep.

The plane-wave cutoff is not the only numerical knob. Brillouin-zone integration
is a quadrature, and a finite k-mesh leaves a residual that the single-shot
basis-set estimator (examples/error-estimation) cannot reach -- the
complement/second-order trick that estimates the Ecut error is variational, and a
sampling error is not. It is reached instead by extrapolation: run the same cell
at a few rising meshes and fit

    E(N_k) = E_inf + c * N_k^(-p),

reporting the dense-k limit E_inf and the residual of the finest mesh.
``estimate_kpoint_error`` does the fit; with three or more meshes it fits the
exponent p from the data too.

This script sweeps silicon over 4x4x4, 6x6x6, and 8x8x8 at a fixed cutoff and
smearing, then prints the extrapolated energy and the per-mesh error. Keep the
meshes in the asymptotic regime: a too-coarse mesh sits off the power law and
skews the fit (a 2x2x2 Si run is ~2.5 eV off and drags E_inf the wrong way). For
a metal, extrapolate at a *fixed* smearing width -- the Fermi-surface
discontinuity sets the convergence rate, so a width change is a separate axis.

    uv run python examples/kmesh_error.py

Measured (CPU): E_inf ~= -214.483 eV, the 8x8x8 residual ~7 meV, and the fitted
exponent ~1.6; the per-mesh error falls 193 -> 27 -> 7 meV across the sweep.
"""
import numpy as np

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.convergence_error import estimate_kpoint_error
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"

a = 5.43
cell = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
pos = np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
upf = parse_upf(f"{PSE}/Si_ONCV_PBE-1.2.upf")

MESHES = [4, 6, 8]
nkpts, energies = [], []
for n in MESHES:
    system = setup_system(cell, pos, [0, 0], [upf], ecut=25 * RY, kmesh=(n, n, n),
                          nbands=8)
    res = scf(system, PBE(), smearing="gaussian", width=0.1,
              etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged
    e = float(res.energies.free_energy)
    nkpts.append(n ** 3)          # total BZ point count; the fit only sees ratios
    energies.append(e)
    print(f"  {n}x{n}x{n}  ({n**3:3d} k)  F = {e:+.6f} eV", flush=True)

kp = estimate_kpoint_error(nkpts, energies)
print(f"\nextrapolated E_inf = {kp['e_infinity_eV']:+.6f} eV")
print(f"finest-mesh residual = {kp['error_eV'] * 1e3:+.3f} meV  "
      f"(fitted exponent p = {kp['exponent']:.2f})")
print("per-mesh signed error [meV]:")
for m in kp["per_mesh"]:
    print(f"  {m['nk']:3d} k : {m['error_eV'] * 1e3:+8.3f}")
