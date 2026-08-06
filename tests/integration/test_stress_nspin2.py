"""Fixed-basis stress for collinear spin (nspin=2 unblock).

The stress path now sums the kinetic and nonlocal terms over both spin
channels and evaluates E_xc from the per-spin densities (the NLCC core split
half/half), mirroring the SCF energy's spin assembly in
scf.common.assemble_pw_energies / spin_xc_energy. Two checks:

- nonmagnetic limit: nspin=2 (start_mag=0) stress on a sheared, low-symmetry
  Si cell equals the spin-restricted stress — already validated vs QE and
  finite differences — so the two-channel occupation/coeff bookkeeping
  reconstructs the nspin=1 tensor to SCF-convergence precision.
- genuinely magnetic (V↑ ≠ V↓): the mandated self-oracle. On a rattled,
  low-symmetry ferromagnetic bcc-Fe cell with mixed (smeared) occupations,
  the ε=0 strained expression reproduces the SCF total energy, and the
  analytic nspin=2 stress equals a central finite difference of that same
  energy w.r.t. strain — the check that actually exercises the per-spin
  kinetic/nonlocal sums and the spin-resolved XC strain derivative.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.postscf.stress import _energy_strained, stress
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

# Sheared, low-symmetry Si cell + displaced atom (from test_stress_vs_qe): a
# full anisotropic tensor, so every stress component is exercised.
SI_CELL = np.array([
    [0.0108600000, 2.7041400000, 2.7285750000],
    [2.7421500000, 0.0162900000, 2.7231450000],
    [2.7530100000, 2.7095700000, 0.0054300000],
])
SI_POS = np.array([[0.0, 0.0, 0.0], [1.4265050000, 1.3275000000, 1.3842875000]])


@pytest.mark.standard
def test_stress_nspin2_matches_spin_restricted():
    """nspin=2 (start_mag=0) stress reproduces the spin-restricted stress on a
    sheared, low-symmetry Si cell to SCF-convergence precision."""
    torch.set_num_threads(4)
    upf = parse_upf(PSEUDOS / "Si_ONCV_PBE-1.2.upf")

    def make():
        return setup_system(SI_CELL, SI_POS, [0, 0], [upf],
                            ecut=20 * RY, kmesh=(2, 2, 2))

    r1 = scf(make(), LDA_PW92(), smearing="none",
             etol=1e-10, rhotol=1e-9, verbose=False)
    r2 = scf(make(), LSDA_PW92(), smearing="none", nspin=2, tot_magnetization=0.0,
             start_mag=[0.0, 0.0], etol=1e-10, rhotol=1e-9, verbose=False)
    assert r1.converged and r2.converged
    s1 = stress(r1, LDA_PW92()).cpu().numpy()
    s2 = stress(r2, LSDA_PW92()).cpu().numpy()
    assert np.abs(s2 - s1).max() < 1e-8, f"\nnspin1:\n{s1}\nnspin2:\n{s2}"


@pytest.mark.slow
def test_stress_nspin2_autograd_vs_fd_magnetic():
    """Ferromagnetic bcc Fe, rattled to a low-symmetry 2-atom cell (mixed,
    smeared occupations, V↑ ≠ V↓): the ε=0 strained expression reproduces the
    SCF total, and the analytic nspin=2 stress matches a central finite
    difference of that expression w.r.t. strain. Fe ONCV has no NLCC, so the
    (spin-agnostic) core-stress term is not the thing under test here — the
    per-spin kinetic/nonlocal sums and spin-resolved XC strain derivative are.
    """
    torch.set_num_threads(8)
    fe = parse_upf(PSEUDOS / "Fe_ONCV_PBE-1.2.upf")
    a = 2.87  # bcc Fe lattice constant (Å); 60 Ry for the magnetic state
    # rattle the cell off cubic so the tensor is anisotropic and low-symmetry
    cell = a * np.array([[1.00, 0.02, 0.01],
                         [0.0, 0.99, 0.015],
                         [0.0, 0.0, 1.01]])
    frac = np.array([[0.0, 0.0, 0.0], [0.51, 0.49, 0.505]])
    pos = frac @ cell
    system = setup_system(cell, pos, [0, 0], [fe], ecut=60 * RY,
                          kmesh=(2, 2, 2), nbands=20, use_symmetry=False)
    res = scf(system, SpinPBE(), smearing="gaussian", width=0.1, nspin=2,
              start_mag=[0.4, 0.4], etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged and res.mag_total > 5.0  # genuinely FM (~3.6 μB/atom)

    # the ε=0 spin-resolved expression must reproduce the SCF total energy
    e0 = _energy_strained(res, SpinPBE(), torch.zeros(3, 3, dtype=torch.float64))
    assert abs(float(e0) - float(res.energies.total)) < 1e-6, (
        float(e0), float(res.energies.total))

    sig = stress(res, SpinPBE(), symmetrize=False).cpu().numpy()
    d = 1e-6

    def fd_component(i, j):
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[i, j] = d
        return (float(_energy_strained(res, SpinPBE(), ep))
                - float(_energy_strained(res, SpinPBE(), -ep))) / (2 * d)

    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (2, 1)]:
        fd_sym = 0.5 * (fd_component(i, j) + fd_component(j, i)) / system.grid.volume
        assert abs(sig[i, j] - fd_sym) < 1e-7, (i, j, sig[i, j], fd_sym)


@pytest.mark.slow
def test_stress_nspin2_matches_spin_restricted_pbe():
    """GGA cross-check of the nonmagnetic limit: PBE stress equals SpinPBE
    (start_mag=0) stress on the sheared Si cell, exercising the spin-resolved
    σ_uu/σ_dd/σ_tot gradient path against the restricted GGA stress."""
    torch.set_num_threads(4)
    upf = parse_upf(PSEUDOS / "Si_ONCV_PBE-1.2.upf")

    def make():
        return setup_system(SI_CELL, SI_POS, [0, 0], [upf],
                            ecut=20 * RY, kmesh=(2, 2, 2))

    r1 = scf(make(), PBE(), smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)
    r2 = scf(make(), SpinPBE(), smearing="none", nspin=2, tot_magnetization=0.0,
             start_mag=[0.0, 0.0], etol=1e-10, rhotol=1e-9, verbose=False)
    assert r1.converged and r2.converged
    s1 = stress(r1, PBE()).cpu().numpy()
    s2 = stress(r2, SpinPBE()).cpu().numpy()
    assert np.abs(s2 - s1).max() < 1e-8, f"\nnspin1:\n{s1}\nnspin2:\n{s2}"


@pytest.mark.standard
def test_run_elastic_nspin2_nc_matches_nspin1():
    """Driver-level ungate: run_elastic now accepts collinear nspin=2 with
    norm-conserving pseudos (was gated to PAW/USPP-only). Decisive self-oracle:
    in the nonmagnetic limit the full 6×6 clamped-ion stiffness from the nspin=2
    (start_mag=0) elastic driver must reproduce the nspin=1 tensor computed with
    identical settings — the whole strain scan (reference + 12 warm-started
    strained SCFs) runs through the newly-reachable spin-resolved stress path.
    Coarse Si (ecut 9 Ry, 2×2×2 k) keeps this a self-consistency check where the
    absolute C is irrelevant, only nspin=2 == nspin=1: the two routes share the
    basis/k/grid, so they agree to SCF-convergence precision (validated to
    0.0000 GPa) at any ecut — the low cutoff just makes it cheap for every-push
    CI."""
    from gradwave.api import run_elastic
    from gradwave.inputs import (
        ElasticParams,
        Input,
        KPointsParams,
        SmearingParams,
    )

    torch.set_num_threads(4)
    a = 5.47
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)

    def _inp(nspin):
        return Input(
            atoms=atoms, pseudo_dir=Path(PSEUDOS),
            pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"}, ecut=9 * RY, xc="pbe",
            kpoints=KPointsParams(mesh=(2, 2, 2)),
            smearing=SmearingParams(type="none"),
            elastic=ElasticParams(strain=0.01),
            nspin=nspin,
            start_mag=({"Si": 0.0} if nspin == 2 else None),
            tot_magnetization=(0.0 if nspin == 2 else None))

    e2 = run_elastic(_inp(2), verbose=False)
    assert e2["formalism"] == "nc" and e2["all_converged"]
    e1 = run_elastic(_inp(1), verbose=False)
    c1 = np.array(e1["c_GPa"])
    c2 = np.array(e2["c_GPa"])
    # identical basis/k/grid → the two tensors agree to SCF-convergence level;
    # 0.2 GPa is a generous band over the warm-start residual differences.
    assert np.abs(c2 - c1).max() < 0.2, (
        f"\nmax|C2-C1|={np.abs(c2 - c1).max():.4f} GPa\nnspin1:\n{c1}\nnspin2:\n{c2}")
