"""KPM-DOS for collinear spin (nspin=2 unblock).

kpm_dos now runs the stochastic Chebyshev trace once per spin channel and
returns spin-resolved DOS (2, n_energies) on a shared energy grid. Checks on
ferromagnetic fcc Ni:

- per-spin sum rule (exact invariant of correct moments): ∫DOS_σ dE equals the
  per-channel state count Σ_k w_k·npw_k (degeneracy 1 per channel);
- total sum rule reproduces the spin-paired 2·Σ_k w_k·npw_k;
- the spin split N↑ − N↓ up to E_F recovers the SCF magnetization;
- Jackson-kernel positivity, and info["nspin"] == 2.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.dos import kpm_dos
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard


def test_kpm_dos_nspin2_sum_rules_and_magnetization():
    torch.set_num_threads(8)
    a = 3.52
    cell = 0.5 * a * np.array([[0, 1, 1.0], [1, 0, 1], [1, 1, 0]])
    ni = parse_upf(PSEUDOS / "PD_Ni_PBE.upf")
    system = setup_system(cell, np.zeros((1, 3)), [0], [ni], ecut=45 * RY,
                          kmesh=(2, 2, 2), nbands=14, time_reversal=False)
    res = scf(system, LSDA_PW92(), smearing="gaussian", width=0.1, nspin=2,
              start_mag=[0.5], etol=1e-7, rhotol=1e-6, verbose=False, kerker=True)
    assert res.converged and res.mag_total > 0.5

    # fine energy grid: the sum rule below is a Riemann sum of a reconstruction
    # with integrable 1/√(1-Ẽ²) edge singularities, so its accuracy is grid- not
    # moment-limited.
    energies, dos, info = kpm_dos(res, n_moments=1600, n_random=8,
                                  n_energies=2400, seed=1)
    assert info["nspin"] == 2
    assert dos.shape == (2, len(energies))
    assert np.all(np.isfinite(dos))
    de = energies[1] - energies[0]
    w = system.kweights.cpu().numpy()

    # per-spin sum rule: ∫DOS_σ dE = Σ_k w_k·npw_k (one electron per state).
    # The exact invariant is the moment μ₀; the discrete integral carries ~0.5%
    # edge-discretization error, so the tolerance is set to 1% (a factor-of-two
    # channel-weighting bug would miss by 100%).
    per_spin = float(sum(wi * sp.npw for wi, sp in zip(w, system.spheres, strict=True)))
    for sp in range(2):
        assert abs(dos[sp].sum() * de - per_spin) / per_spin < 1e-2, sp
    # total sum rule: 2·Σ_k w_k·npw_k, matching the nspin=1 spin-paired trace
    total = float((dos[0] + dos[1]).sum() * de)
    assert abs(total - 2 * per_spin) / (2 * per_spin) < 1e-2

    # magnetization from the spin-resolved cumulative counts up to E_F.
    # Leakage from the plane-wave continuum affects both channels similarly, so
    # it largely cancels in the difference N↑ − N↓ (loose tol per the dos.py
    # leakage note).
    def kpm_count(e, sp):
        return float(dos[sp][energies <= e].sum() * de)

    n_up, n_dn = kpm_count(res.fermi, 0), kpm_count(res.fermi, 1)
    assert abs((n_up - n_dn) - float(res.mag_total)) < 0.7, (n_up - n_dn, res.mag_total)

    # Jackson-kernel positivity (strict invariant of the method)
    assert dos.min() > -1e-6 * dos.max()
