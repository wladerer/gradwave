"""MAE regression gate: L1_0 FePt easy axis and magnitude by full SOC SCF.

Transcribes examples/fept_mae.py into a gate. The physics (measured on the
asus CPU, 2026-07-18, and recorded in the example header): at the converged
(6, 6, 4) mesh the MAE = E[100] - E[001] is +2.552 meV/cell -- easy axis [001],
magnitude in the literature band (~1-3 meV/f.u.). The 48-k mesh gives the
WRONG easy axis (-1.39 meV/cell), so this gate deliberately runs the dense
mesh; a cheap-mesh version would pin the sampling artifact, not the physics.

Each orientation folds by its own magnetic (Shubnikov) group over the SAME
underlying mesh ([001] -> 30 k, [100] -> 48 k from 144), which preserves the
common-mode cancellation of the k-discretization error in the difference
(validated to 5e-11 eV in test_magnetic_ibz.py) at 3.6x less cost.

torture tier: two SOC metal SCFs at 70 Ry, ~20-60 min on an 8-core CPU. Run it
when the SOC path, the spinor SCF, or the magnetic symmetry machinery changes.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"

A, C = 2.723, 3.712                         # L1_0 FePt tetragonal [A]
KMESH = (6, 6, 4)


def _fept_energy(axis):
    fe = parse_upf(f"{PSE}/Fe_ONCV_PBE_FR-1.0.upf")
    pt = parse_upf(f"{PSE}/Pt_ONCV_PBE_FR-1.0.upf")
    cell = np.diag([A, A, C])
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    ax = np.array(axis, float)
    init = [(3.0 * ax).tolist(), (0.4 * ax).tolist()]   # Fe ~3, Pt induced ~0.4
    system = setup_system(cell, pos, [0, 1], [fe, pt], ecut=70 * RY, kmesh=KMESH,
                          nbands=30, use_symmetry=True, magmoms=init)
    res = scf_noncollinear(system, NoncollinearXC(LSDA_PW92()), mag_vec_init=init,
                           smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-7,
                           max_iter=300, mixing_alpha=0.3, mixing_history=12,
                           verbose=False)
    assert res.converged
    d_last = abs(res.history[-1]["free_energy"] - res.history[-2]["free_energy"])
    mag = float(np.linalg.norm(np.array(res.mag_vec)))
    return float(res.energies.free_energy), d_last, mag, len(system.spheres)


@pytest.mark.torture
def test_fept_easy_axis_and_mae_magnitude():
    torch.set_num_threads(8)
    e001, d001, m001, nk001 = _fept_energy([0, 0, 1.0])
    e100, d100, m100, nk100 = _fept_energy([1.0, 0, 0])

    # magnetic-IBZ folds of the 144-point mesh (regression on the Shubnikov path)
    assert nk001 == 30 and nk100 == 48, f"folds {nk001}, {nk100}"

    mae = (e100 - e001) * 1000.0            # meV/cell
    # both orientations settled far below the signal, equal-magnitude moments
    assert d001 < 1e-7 and d100 < 1e-7, f"energy tails {d001:.1e}, {d100:.1e} eV"
    assert abs(m001 - m100) < 0.1, f"|M| = {m001:.3f} vs {m100:.3f} muB"

    # easy axis [001] with the measured +2.55 meV/cell inside a generous band
    assert mae > 0, f"MAE = {mae:+.3f} meV/cell: wrong easy axis"
    assert 1.0 < mae < 4.0, f"MAE = {mae:+.3f} meV/cell outside the FePt band"
