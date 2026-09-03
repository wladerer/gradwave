import time, numpy as np, torch, sys
from pathlib import Path
from gradwave.api import build_system, run_scf
from gradwave.grids import build_gsphere
from gradwave.core.hamiltonian import projectors
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.inputs import load_input
HERE = Path("experiments/symbolic")
inp = load_input(str(HERE/"si_abs.yaml")); system = build_system(inp)
run_scf(inp, system=system, verbose=False)
grid = system.grid; k = np.array([0.13,0.21,0.34])
sph = build_gsphere(grid, system.ecut, k)
beta_ls, dij = species_projector_tables(system.upfs, None)
def build():
    pd = projector_data_at_k(sph, system.species_of_atom, system.upfs, beta_ls, dij, grid.volume, None)
    return projectors(pd, system.positions)
p = build()
ts = []
for _ in range(20):
    t = time.perf_counter(); build(); ts.append(time.perf_counter()-t)
t = float(np.median(ts))
print(f"projector rebuild at one k: {t*1e3:.2f} ms  (npw={sph.npw}, nproj={p.shape[0]})")
print(f"finite-diff ∂H/∂k (3 dirs × k±δk = 6 rebuilds): ~{6*t*1e3:.1f} ms")
print(f"analytic ∂H/∂k (1 assembly + spline-deriv F'(q)): ~{1.5*t*1e3:.1f} ms")
print(f"→ per-k saving ~{4.5*t*1e3:.1f} ms; matters only if ∂H/∂k is taken over a DENSE k-mesh")
