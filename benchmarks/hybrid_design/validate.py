"""Validate the differentiable hybrid gap: E_x self-check, value, and the
frozen-orbital dgap/d(α,ω) against finite difference of re-converged hybrids.

    uv run python benchmarks/hybrid_design/validate.py
"""
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from gap import differentiable_hybrid_gap, exx_energy_from_diagonal, vbm_cbm  # noqa: E402

from gradwave.postscf.exchange_multik import (  # noqa: E402
    HybridExchangeParams,
    multik_exchange_energy,
    occupied_periodic_orbitals,
)
from gradwave.postscf.hybrid import hybrid_scf  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import setup_system  # noqa: E402

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"
a = 5.43
cell = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
pos = np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
upf = parse_upf(f"{PSE}/Si_ONCV_PBE-1.2.upf")
MODE, OMEGA = "short_range", 0.2


def system():
    return setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY, kmesh=(1, 1, 1),
                        nbands=8)


def converge(alpha, omega=OMEGA):
    r = hybrid_scf(system(), alpha=float(alpha), mode=MODE, omega=float(omega),
                   smearing="none", etol=1e-9, rhotol=1e-8, max_iter=80,
                   verbose=False)
    assert r.converged
    return r


def gap_of(res):
    (_, _, ev), (_, _, ec) = vbm_cbm(res)
    return ec - ev


print("converging reference hybrid (α=0.25, ω=0.2)...")
res = converge(0.25)

# 1) orbital normalization + E_x convention self-check
u_occ, kcart, kw = occupied_periodic_orbitals(res, res.system)
nrm = float((u_occ[0][0].abs() ** 2).sum().real) / u_occ[0][0].numel()
e_ref = float(2.0 * multik_exchange_energy(u_occ, kcart, kw,
              res.system.grid.g_cart, res.system.grid.volume, mode=MODE, omega=OMEGA))
e_diag = float(2.0 * exx_energy_from_diagonal(res, mode=MODE, omega=OMEGA))
print(f"  <i|i> = {nrm:.6f} (want 1)")
print(f"  E_x:  multik {e_ref:.6f}  vs  ½Σ<i|Vx|i> {e_diag:.6f}  "
      f"(Δ {abs(e_ref-e_diag)*1e3:.2e} meV)")

# 2) gap value matches
params = HybridExchangeParams(alpha=0.25, omega=OMEGA, mode=MODE)
g = differentiable_hybrid_gap(res, params)
print(f"  gap value: differentiable {float(g):.6f}  vs  converged {gap_of(res):.6f} eV")

# 3) frozen-orbital dgap/dα, dgap/dω  vs  FD of re-converged hybrids
params.zero_grad(set_to_none=True)
differentiable_hybrid_gap(res, params).backward()
al = float(params.alpha.detach())
om = float(params.omega.detach())
dga_frozen = float(params.raw_alpha.grad) / (al * (1 - al))
dgo_frozen = float(params.raw_omega.grad) / (1 - np.exp(-om))

d = 0.02
dga_fd = (gap_of(converge(0.25 + d)) - gap_of(converge(0.25 - d))) / (2 * d)
do = 0.03
dgo_fd = (gap_of(converge(0.25, OMEGA + do)) - gap_of(converge(0.25, OMEGA - do))) / (2 * do)
print(f"  dgap/dα:  frozen {dga_frozen:+.4f}   FD {dga_fd:+.4f}   "
      f"(SCF-response {abs(dga_frozen-dga_fd)/abs(dga_fd)*100:.0f}% of FD)")
print(f"  dgap/dω:  frozen {dgo_frozen:+.4f}   FD {dgo_fd:+.4f}")
