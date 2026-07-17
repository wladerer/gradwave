"""Benchmark: Heisenberg exchange of bcc Fe from the AD constrained-moment torque.

Extracts the effective inter-sublattice exchange of bcc Fe (2-atom cubic cell) by
tilting the body-center moment at the ferromagnetic reference and reading the
induced torque on the corner atom (postscf/spin_exchange.py). bcc Fe is
centrosymmetric with no SOC here, so the DMI must vanish and only the isotropic
Heisenberg J survives.

Validation targets:
  * sign: J > 0 (ferromagnetic) — bcc Fe is a ferromagnet.
  * nearest-neighbor J1: the extracted J_01 folds the whole first shell (8 nn are
    periodic images of the body-center atom), so J1 ~ J_01 / 8. Reference LKAG value
    for bcc Fe: J1 ~ 15-19 meV [Pajda, Kudrnovsky, Turek, Drchal, Bruno,
    Phys. Rev. B 64, 174402 (2001)].
  * Curie temperature: nearest-neighbor mean-field estimate
    k_B T_c = (2/3) J_01, compare to experiment T_c = 1043 K. (Same-sublattice 2nn
    are not captured by a 2-atom cell; a supercell or the reciprocal-space J(q)
    route resolves individual shells — see the module docstring.)

Heavy: three constrained non-collinear SCFs at kmesh (3,3,3). ~25 min on CPU,
~5 min on an A100. Run:
    PYTHONPATH=src python examples/fe_exchange.py

Measured (A100, LSDA, 60 Ry, (3,3,3), lam=8, delta=0.08):
    effective J_01     +179.5 meV      (inter-sublattice sum)
    implied nn J1      +22.4 meV       (Pajda 2001 LKAG: ~15-19 meV)
    DMI along z        +0.001 meV      (numerical zero, as symmetry requires)
    mean-field T_c     1388 K          (Pajda MFA ~1414 K; experiment 1043 K)
The sign is ferromagnetic, DMI is zero by centrosymmetry, J1 sits just above the
LKAG value (J01/8 folds in further same-sublattice shells), and the nn mean-field
T_c reproduces Pajda's MFA -- both overshoot experiment by the textbook mean-field
margin (RPA/Monte Carlo pull it toward ~950 K). The extraction reproduces the
established magnetism of bcc Fe.
"""
import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.spin_exchange import decompose, exchange_from_atom
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

torch.set_num_threads(8)
RY = 13.605693122994
KB = 8.617333e-5  # eV/K
PSE = "tests/fixtures/qe/pseudos"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    fe = parse_upf(f"{PSE}/Fe_ONCV_PBE-1.2.upf")
    a = 2.87
    cell = a * np.eye(3)
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) * a
    system = setup_system(cell, pos, [0, 0], [fe, fe], ecut=60 * RY,
                          kmesh=(3, 3, 3), nbands=24, time_reversal=False)
    if DEV != "cpu":
        system = system.to(DEV)
    xc = NoncollinearXC(LSDA_PW92())
    m0 = torch.tensor([2.222, 2.222], dtype=torch.float64,
                      device=system.positions.device)  # FM |M| per atom

    tensors, _ = exchange_from_atom(
        system, xc, j=1, m0=m0, ref_dir=(0, 0, 1), delta=0.08, lam=8.0,
        smearing="gaussian", width=0.1, etol=1e-7, rhotol=1e-6, max_iter=200,
        mixing_alpha=0.4, verbose=False)
    J_iso, D_ref, gamma = decompose(tensors[0])

    z1 = 8                      # bcc first-shell coordination
    j1 = J_iso / z1
    tc_mfa = (2.0 / 3.0) * J_iso / KB
    print(f"device                 : {DEV}")
    print(f"effective J_01         : {J_iso*1000:+.1f} meV  (inter-sublattice sum)")
    print(f"implied nn J1 ~ J01/8  : {j1*1000:+.1f} meV  (ref Pajda 2001: ~15-19 meV)")
    print(f"DMI along z            : {D_ref*1000:+.3f} meV  (must be ~0: centrosym, no SOC)")
    print(f"mean-field T_c (nn)    : {tc_mfa:.0f} K       (experiment: 1043 K)")
    print(f"sign check             : {'PASS (FM)' if J_iso > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
