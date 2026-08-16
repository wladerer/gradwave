"""Stretch: fcc Co Curie temperature (exp 1388 K) — the general (fcc) MC path.

Second magnet after bcc Fe, exercising the non-bipartite fcc Monte Carlo. The
isotropic Heisenberg J needs only the constrained-moment torque, *not* SOC, so
this uses a scalar-relativistic Co pseudo (same as Fe) — a fully-relativistic
(FR) pseudo makes every noncollinear SCF a full SOC-spinor solve, ~100× slower,
for zero benefit to the isotropic J. Set ``CO_UPF`` to override the pseudo path.
Conventional 4-atom fcc cell; relax then extract at experimental a=3.54 Å.
"""

import os
from pathlib import Path

import numpy as np

from gradwave.constants import KB_EV
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.eos import fit_bm3
from gradwave.postscf.heisenberg_mc import curie_temperature, fcc_lattice, heisenberg_mc
from gradwave.postscf.magnetism import characterize_magnetism
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear

RY = 13.605693122994
Z = 12
EXP_TC = 1388.0
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qe" / "pseudos"
CO_UPF = os.environ.get("CO_UPF", str(FIX / "Co_ONCV_PBE_FR-1.0.upf"))
co = parse_upf(CO_UPF)
xc = NoncollinearXC(LSDA_PW92())
SCF_KW = dict(smearing="gaussian", width=0.1, etol=1e-6, rhotol=1e-5,
              max_iter=250, mixing_alpha=0.3, verbose=False)
BASIS = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def _system(a):
    return setup_system(a * np.eye(3), BASIS * a, [0, 0, 0, 0], [co],
                        ecut=55 * RY, kmesh=(3, 3, 3), nbands=44, time_reversal=False)


def tc_at(a, label):
    rep = characterize_magnetism(_system(a), xc, ref_atom=0, lam=8.0,
                                 delta=0.08, **SCF_KW)
    j0 = sum(rep.exchange_J.values())            # shell sum over the 12 fcc nn
    tc_mfa = rep.curie_temperature_mfa
    k_bond = 3.0 * KB_EV * tc_mfa / Z
    nbr, _ = fcc_lattice(6)
    temps = np.linspace(0.60, 0.88, 9) * tc_mfa * KB_EV
    r = heisenberg_mc(nbr, k_bond, temps, n_equil=500, n_sample=1000, seed=0)
    tc_mc = curie_temperature(r["temp"], r["chi"]) / KB_EV
    print(f"  [{label}] a={a:.3f} Å  M={rep.moment_magnitudes[0]:.2f} μB  "
          f"J_0={j0 * 1e3:.0f} meV  MFA T_c={tc_mfa:.0f} K  MC T_c={tc_mc:.0f} K", flush=True)
    return tc_mfa, tc_mc


def main():
    print("relax fcc Co (spin-polarised EOS)", flush=True)
    a_grid = np.array([3.44, 3.48, 3.52, 3.56, 3.60])
    seed = [[0, 0, 1.2]] * 4
    E = []
    for a in a_grid:
        res = scf_noncollinear(_system(float(a)), xc, seed, **{**SCF_KW, "width": 0.05})
        E.append(float(res.energies.free_energy))
        print(f"  a={a:.2f} Å  E={E[-1]:.4f} eV  |M|/atom={res.mag_abs / 4:.2f} μB", flush=True)
    a_eq = float(fit_bm3(a_grid ** 3, np.array(E)).v0) ** (1 / 3)
    print(f"  PBE-relaxed a = {a_eq:.3f} Å (exp 3.54)\n", flush=True)

    mfa, mc = tc_at(3.54, "experimental")
    tc_at(a_eq, "PBE-relaxed ")
    print(f"\nfcc Co: MFA {mfa:.0f} K, MC {mc:.0f} K, experiment {EXP_TC:.0f} K "
          f"(MC {abs(mc - EXP_TC) / EXP_TC * 100:.0f}% off)", flush=True)


if __name__ == "__main__":
    main()
