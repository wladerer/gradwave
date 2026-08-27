"""MgO (rocksalt) validation: Berry-phase Z*, ASR, k-mesh convergence, IR spectrum.

Norm-conserving ONCV (Mg, O). Prints:
  - Z* k-mesh convergence (diagonal component of Mg),
  - full Z* tensors + ASR residual at the production mesh,
  - end-to-end run_phonons IR: Gamma TO frequencies + IR-active modes + spectrum peak.

Literature: Z*_Mg ~ +1.96 (Z*_O ~ -1.96), isotropic; TO(Gamma) ~ 400 cm^-1 (exp).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from gradwave.api import run_phonons
from gradwave.inputs import Input, KPointsParams, PhononParams, SmearingParams
from gradwave.pseudo.upf import parse_upf
from gradwave.postscf.born import born_effective_charges
from gradwave.postscf.polarization import berry_phase_polarization
from gradwave.scf.loop import scf, setup_system
from gradwave.core.xc.pbe import PBE
from tests.helpers import PSEUDOS

torch.set_num_threads(8)
RY = 13.605693122994

A = 4.24  # PBE-ish MgO lattice constant [Angstrom]
CELL = A / 2.0 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL  # Mg, O
SPECIES = [0, 1]  # 0=Mg, 1=O
ECUT = 50 * RY

upf_mg = parse_upf(str(Path(PSEUDOS) / "Mg_ONCV_PBE-1.2.upf"))
upf_o = parse_upf(str(Path(PSEUDOS) / "O_ONCV_PBE-1.2.upf"))
UPFS = [upf_mg, upf_o]


def scf_fn(mesh):
    def _fn(positions):
        system = setup_system(CELL, positions, SPECIES, UPFS, ecut=ECUT,
                              kmesh=mesh, kshift=(0, 0, 0), use_symmetry=False,
                              time_reversal=False)
        return scf(system, PBE(), nspin=1, smearing="none",
                   etol=1e-9, rhotol=1e-8, diago_tol=1e-10, verbose=False)
    return _fn


def zstar_convergence():
    print("=== Z* k-mesh convergence (MgO, Berry-phase FD) ===", flush=True)
    for nk in (4, 6):
        mesh = (nk, nk, nk)
        res = born_effective_charges(scf_fn(mesh), POS, mesh, step=2e-3)
        z = res["born"].numpy()
        zmg = np.trace(z[0]) / 3.0
        zo = np.trace(z[1]) / 3.0
        asr = float(res["asr_max"])
        print(f"  mesh {nk}^3: Z*_Mg(iso)={zmg:+.4f}  Z*_O(iso)={zo:+.4f}  "
              f"|ASR|max={asr:.3e}", flush=True)
        if nk == 6:
            np.set_printoptions(precision=4, suppress=True)
            print("  Z*_Mg tensor:\n", z[0], flush=True)
            print("  Z*_O  tensor:\n", z[1], flush=True)


def polarization_sanity():
    print("=== polarization sanity (undisplaced, 6^3) ===", flush=True)
    res = scf_fn((6, 6, 6))(POS)
    pol = berry_phase_polarization(res, (6, 6, 6))
    print("  reduced p_ion:", pol.reduced_ionic.numpy(),
          " p_el:", pol.reduced_electronic.numpy(), flush=True)
    print("  P/e vector [1/Ang^2]:", pol.vector().numpy(), flush=True)


def ir_end_to_end():
    print("=== run_phonons end-to-end IR (MgO) ===", flush=True)
    atoms = Atoms("MgO", positions=POS, cell=CELL, pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Mg": "Mg_ONCV_PBE-1.2.upf", "O": "O_ONCV_PBE-1.2.upf"},
        ecut=ECUT, xc="pbe",
        kpoints=KPointsParams(mesh=(6, 6, 6)),
        smearing=SmearingParams(type="none"),
        phonons=PhononParams(supercell=(2, 2, 2), displacement=0.01,
                             npoints=40, dos_mesh=(0, 0, 0),
                             ir=True, ir_kmesh=(6, 6, 6),
                             born_displacement=2e-3, ir_broadening=8.0))
    ph = run_phonons(inp, verbose=True)
    ir = ph["ir"]
    freqs = np.array(ir["frequency_cm1"])
    inten = np.array(ir["intensity"])
    active = np.array(ir["ir_active"])
    print("  Gamma modes [cm^-1]:", np.round(freqs, 1).tolist(), flush=True)
    print("  IR intensities     :", np.round(inten, 4).tolist(), flush=True)
    print("  IR-active          :", active.tolist(), flush=True)
    print("  ASR |ZZ*| max      :", ir["born_asr_max"], flush=True)
    to = freqs[active]
    if to.size:
        print(f"  TO(active) modes ~ {np.round(to,1).tolist()} cm^-1 "
              f"(exp MgO TO ~400)", flush=True)
    spec = ir["spectrum"]
    g = np.array(spec["frequency_cm1"])
    s = np.array(spec["intensity"])
    print(f"  spectrum peak at {g[int(np.argmax(s))]:.1f} cm^-1", flush=True)


if __name__ == "__main__":
    polarization_sanity()
    zstar_convergence()
    ir_end_to_end()
    print("DONE", flush=True)
