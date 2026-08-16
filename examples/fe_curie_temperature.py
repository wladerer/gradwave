"""Curie temperature of bcc Fe, end to end: relax -> extract J -> Monte Carlo.

The one place a differentiable small-cell DFT code has a real edge is handing a
clean spin Hamiltonian to a spin-model solver. This reproduces the experimental
Curie temperature of iron (1043 K) with no empirical input:

  1. RELAX the bcc lattice constant by a spin-polarised (FM) Birch-Murnaghan EOS.
  2. EXTRACT the Heisenberg exchange J_01 at the relaxed geometry from the
     autograd-exact constrained-moment torque (postscf.spin_exchange), no SOC
     needed for the isotropic J.
  3. Two T_c estimates on that J: mean field (the repo's only estimator, known to
     overshoot ~30 %) and a classical Heisenberg Monte Carlo (postscf.heisenberg_mc)
     that includes the transverse spin fluctuations mean field ignores.

Mean field lands near 1390 K; the Monte Carlo brings it down to ~1050 K — the
fluctuation correction is the difference between "33 % too high" and experiment.

    uv run python examples/fe_curie_temperature.py
"""

from pathlib import Path

import numpy as np

from gradwave.constants import KB_EV
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.eos import fit_bm3
from gradwave.postscf.heisenberg_mc import (
    bcc_lattice,
    curie_temperature,
    heisenberg_mc,
    mean_field_tc,
)
from gradwave.postscf.magnetism import characterize_magnetism
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear

RY = 13.605693122994
Z = 8              # bcc nearest-neighbour coordination
EXP_TC = 1043.0    # experimental Curie temperature of iron [K]
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qe" / "pseudos"
fe = parse_upf(FIX / "Fe_ONCV_PBE-1.2.upf")     # scalar-relativistic, 16 e-
xc = NoncollinearXC(LSDA_PW92())
SCF_KW = dict(smearing="gaussian", width=0.1, etol=1e-7, rhotol=1e-6,
              max_iter=200, mixing_alpha=0.4, verbose=False)


def _system(a):
    cell = a * np.eye(3)
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) * a
    return setup_system(cell, pos, [0, 0], [fe], ecut=60 * RY, kmesh=(3, 3, 3),
                        nbands=24, time_reversal=False)


def relax_lattice():
    """Spin-polarised BM3 EOS → equilibrium cubic lattice constant [Å]."""
    a_grid = np.array([2.76, 2.79, 2.82, 2.85, 2.88])
    seed = [[0.0, 0, 1.5], [0.0, 0, 1.5]]        # ferromagnetic along z
    energies = []
    for a in a_grid:
        res = scf_noncollinear(_system(float(a)), xc, seed, **SCF_KW)
        energies.append(float(res.energies.free_energy))
        print(f"  a={a:.2f} Å  E={energies[-1]:.5f} eV  |M|/atom={res.mag_abs / 2:.2f} μB")
    fit = fit_bm3(a_grid ** 3, np.array(energies))
    a_eq = float(fit.v0) ** (1 / 3)
    return a_eq


def main():
    print("1. relax bcc Fe (spin-polarised EOS)")
    a_eq = relax_lattice()
    print(f"   relaxed a = {a_eq:.4f} Å   (experiment 2.87 Å)\n")

    print("2. extract Heisenberg exchange at the relaxed geometry")
    rep = characterize_magnetism(_system(a_eq), xc, ref_atom=0, lam=8.0,
                                 delta=0.08, **SCF_KW)
    j01 = rep.exchange_J[1]                       # {i: J_0i}, ref_atom 0 -> i=1
    tc_mfa = rep.curie_temperature_mfa
    print(f"   moment = {rep.moment_magnitudes[0]:.2f} μB   "
          f"J_01 = {j01 * 1e3:.1f} meV (per-bond ≈ {j01 / Z * 1e3:.1f} meV)   "
          f"mean-field T_c = {tc_mfa:.0f} K\n")

    print("3. classical Heisenberg Monte Carlo on the extracted J")
    # calibrate the per-bond K so the MC's mean-field limit (zK/3) equals the DFT
    # mean-field T_c; the MC then applies the geometry-exact fluctuation reduction.
    k_bond = 3.0 * KB_EV * tc_mfa / Z
    nbr, sub, _ = bcc_lattice(10)
    temps = np.linspace(0.60, 0.88, 9) * tc_mfa * KB_EV      # eV, around 0.77·T_c^MFA
    r = heisenberg_mc(nbr, sub, k_bond, temps, n_equil=500, n_sample=1000, seed=0)
    tc_mc = curie_temperature(r["temp"], r["chi"]) / KB_EV
    mfa_check = mean_field_tc(k_bond, Z, KB_EV)

    print(f"   mean-field T_c   = {tc_mfa:.0f} K   (MC self-check {mfa_check:.0f} K)")
    print(f"   Monte Carlo T_c  = {tc_mc:.0f} K")
    print(f"   experiment       = {EXP_TC:.0f} K")
    print(f"   → MC error {abs(tc_mc - EXP_TC) / EXP_TC * 100:.0f}% vs mean-field "
          f"{abs(tc_mfa - EXP_TC) / EXP_TC * 100:.0f}%")


if __name__ == "__main__":
    main()
