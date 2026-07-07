"""KPM-DOS validation: the stochastic Chebyshev DOS must reproduce the
explicitly diagonalized spectrum — cumulative state counts in the window
where explicit eigenvalues exist (all bands below the highest computed one)."""

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.dos import kpm_dos
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0, 0], [A / 4] * 3])


def test_kpm_dos_matches_explicit_spectrum():
    torch.set_num_threads(8)
    upf = parse_upf("tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=15 * RY,
                          kmesh=(2, 2, 2), nbands=12)
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-8, rhotol=1e-7,
              verbose=False)
    assert res.converged

    energies, dos, info = kpm_dos(res, n_moments=1600, n_random=8, seed=1)
    assert np.all(np.isfinite(dos))
    de = energies[1] - energies[0]
    sigma = info["resolution_eV"]  # compare against the SAME broadening

    # explicit spectrum: valid comparison window = below the lowest 12th band
    eigs = res.eigenvalues.cpu().numpy()
    w = system.kweights.cpu().numpy()
    e_top = eigs[:, -1].min() - 1.0

    from scipy.special import erf

    def explicit_count(e):
        # Gaussian-CDF-broadened cumulative count at the KPM resolution
        return float(sum(
            2.0 * wi * (0.5 * (1.0 + erf((e - ek) / (sigma * np.sqrt(2))))).sum()
            for wi, ek in zip(w, eigs, strict=True)
        ))

    def kpm_count(e):
        m = energies <= e
        return float(dos[m].sum() * de)

    # 1) the total trace is an EXACT invariant of correct moments:
    #    ∫DOS dE = 2·Σ_k w_k·npw_k
    exact_total = float(sum(
        2.0 * wi * sp.npw for wi, sp in zip(w, system.spheres, strict=True)))
    kpm_total = float(dos.sum() * de)
    assert abs(kpm_total - exact_total) / exact_total < 5e-3

    # 2) cumulative counts: leakage-limited (~0.5 states from the conduction
    #    continuum's kernel tails — see dos.py docstring; grows near the
    #    continuum itself, so probes stay at/below the gap), tolerance 0.7
    assert e_top > res.fermi  # window sanity
    for e_probe in (res.fermi - 6.0, res.fermi - 2.0, res.fermi + 0.036):
        n_exp = explicit_count(e_probe)
        n_kpm = kpm_count(e_probe)
        assert abs(n_kpm - n_exp) < 0.7, (e_probe, n_exp, n_kpm)

    # 3) Jackson-kernel positivity (a strict invariant of the method); the
    #    0.6 eV Si gap itself is not resolvable at M=1600 against the
    #    continuum leakage baseline — resolving it needs ~4x the moments
    assert dos.min() > -1e-6 * dos.max()
