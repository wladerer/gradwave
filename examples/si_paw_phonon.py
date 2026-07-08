"""Gamma optical phonon of Si (PAW) from finite differences of the ANALYTIC
autograd forces, against QE ph.x DFPT at identical settings.

Result (psl kjpaw, PBE, 45/180 Ry, 2x2x2, 32^3): 586.11 cm-1 vs ph.x
586.09 cm-1 (0.003%). The point: postscf/paw_forces.py assembles the full
USPP/PAW force (augmentation, S-orthogonality, one-center chain, NLCC) in
one backward pass, so second derivatives come from a two-point stencil per
displacement instead of a DFPT implementation.

Run from the repo root with the psl UPF in tests/fixtures/qe/pseudos."""

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.paw_forces import forces_uspp
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

torch.set_num_threads(4)
RY = 13.605693122994
paw = parse_upf_paw("tests/fixtures/qe/pseudos/Si.pbe-n-kjpaw_psl.1.0.0.UPF")
cell = 5.43/2*np.array([[0.,1,1],[1,0,1],[1,1,0]])
pos0 = np.array([[0.,0,0],[5.43/4]*3])
delta = 0.005

def force_at(disp):
    pos = pos0.copy()
    pos[1, 0] += disp
    system = setup_uspp(cell, pos, [0,0], [paw], ecut=45*RY, kmesh=(2,2,2),
                        ecutrho=180*RY, fft_shape=(32,32,32))
    res = scf_uspp(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
                   verbose=False, max_iter=40)
    assert res["converged"]
    return forces_uspp(res, PBE()).numpy()

fp = force_at(+delta)
fm = force_at(-delta)
phi = -(fp[1,0] - fm[1,0]) / (2*delta)  # eV/A^2, Phi_2x,2x
m_si = 28.0855
# omega^2 = 2*phi/m for the diamond Gamma optical mode
# eV/A^2/amu -> (rad/s)^2: 1 eV = 1.602176634e-19 J, 1 A = 1e-10 m, 1 amu = 1.66053906660e-27 kg
w2_SI = 2*phi * 1.602176634e-19 / 1e-20 / (m_si * 1.66053906660e-27)
freq_cm = np.sqrt(w2_SI) / (2*np.pi * 2.99792458e10)
print(f"Phi_2x2x = {phi:.4f} eV/A^2 -> Gamma optical = {freq_cm:.2f} cm^-1")
print("QE ph.x reference: 586.093457 cm^-1 (acoustic 17.8 = no-ASR mesh artifact)")
