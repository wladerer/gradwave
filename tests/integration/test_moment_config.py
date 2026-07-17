"""Constrained non-collinear magnetism: local-moment constraint, torque, and
the moment-direction config search (postscf/moment_config.py).

The system is a triplet O2 molecule at Gamma — small (single k-point) but
genuinely magnetic (|m| = 2 muB, ferromagnetic: the two O moments want to be
parallel). Constraining the moments away from parallel and reading the torque
lets us both validate dW/de against a finite difference and watch the search
drive the configuration back to the ferromagnetic ground state.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.moment_config import (
    atomic_weights,
    constrained_moment_scf,
    relax_moment_directions,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

RY = 13.605693122994
PSEUDO = "tests/fixtures/qe/pseudos/O_ONCV_PBE-1.2.upf"


def _o2_system(L=6.0, d=1.21):
    o = parse_upf(PSEUDO)
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    return setup_system(cell, pos, [0, 0], [o, o], ecut=30 * RY, kmesh=(1, 1, 1),
                        nbands=8, time_reversal=False)


def _relangle(d):
    d = np.asarray(d)
    c = np.dot(d[0], d[1]) / (np.linalg.norm(d[0]) * np.linalg.norm(d[1]))
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


@pytest.mark.slow
def test_constraint_holds_moment_and_torque_matches_fd():
    """The penalty holds each atomic moment near its target (magnitude intact),
    and the analytic gradient dW/de matches a finite difference of the
    constrained functional W = E_KS + lambda|M_perp|^2."""
    torch.set_num_threads(8)
    system = _o2_system()
    xc = NoncollinearXC(LSDA_PW92())
    w = atomic_weights(system)
    assert bool((w >= 0).all()) and float(w.sum(0).max()) <= 1.0 + 1e-9

    lam, phi0, dlt = 4.0, 30.0, 3.0
    scf_kw = dict(smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8,
                  max_iter=200, verbose=False)

    def run(phi):
        th = np.deg2rad(phi)
        dirs = [[0.0, 0, 1], [float(np.sin(th)), 0, float(np.cos(th))]]
        _, info = constrained_moment_scf(system, xc, dirs, lam=lam, weights=w,
                                         **scf_kw)
        return info

    im, ip, imn = run(phi0), run(phi0 + dlt), run(phi0 - dlt)

    # moment magnitude survives (no demagnetization) and stays near its target
    assert torch.linalg.norm(im["M"], dim=-1).min() > 0.8
    cos = ((im["M"] * im["directions"]).sum(-1)
           / torch.linalg.norm(im["M"], dim=-1)).clamp(-1, 1)
    ang = torch.rad2deg(torch.arccos(cos))
    assert float(ang.max()) < 20.0  # held within 20 deg of target

    # analytic dW/dphi vs central finite difference of W
    th = np.deg2rad(phi0)
    de1 = torch.tensor([np.cos(th), 0.0, -np.sin(th)], dtype=torch.float64)
    grad_analytic = float((im["energy_grad"][1] * de1).sum()) * (np.pi / 180)
    grad_fd = (ip["W_eV"] - imn["W_eV"]) / (2 * dlt)
    assert abs(grad_analytic - grad_fd) < 0.05 * abs(grad_fd) + 1e-4, \
        f"analytic {grad_analytic:.5f} vs FD {grad_fd:.5f} eV/deg"


@pytest.mark.slow
def test_vector_penalty_holds_magnitude_where_perp_demagnetizes():
    """The magnitude problem: force O2's two moments antiparallel (180 deg). The
    strong ferromagnetic coupling makes demagnetization the cheap escape, so the
    direction-only 'perp' penalty collapses the moments (|M| -> 0) to satisfy
    |M_perp|^2 for free. The magnitude-robust 'vector' penalty charges for it and
    holds a genuine antiferromagnetic state at near-full magnitude."""
    torch.set_num_threads(8)
    system = _o2_system()
    xc = NoncollinearXC(LSDA_PW92())
    w = atomic_weights(system)
    # both runs are seeded high-spin (mag_init_scale=1.5) so the demagnetization
    # is the SCF choosing to collapse the moment, not a weak initial guess. The
    # target magnitude for "vector" is O2's known atomic moment (~1 muB); the
    # unconstrained auto-reference is deliberately not used here because the bare
    # non-collinear O2 SCF is multi-stable, which is orthogonal to this test.
    scf_kw = dict(smearing="gaussian", width=0.1, etol=1e-7, rhotol=1e-6,
                  max_iter=120, mag_init_scale=1.5, verbose=False)
    afm = [[0.0, 0, 1], [0.0, 0, -1]]
    m0 = torch.tensor([1.0, 1.0], dtype=torch.float64)

    _, perp = constrained_moment_scf(system, xc, afm, lam=8.0, weights=w,
                                     mode="perp", **scf_kw)
    _, vec = constrained_moment_scf(system, xc, afm, lam=8.0, weights=w,
                                    mode="vector", target_mag=m0, **scf_kw)

    perp_mag = torch.linalg.norm(perp["M"], dim=-1)
    vec_mag = torch.linalg.norm(vec["M"], dim=-1)
    assert float(perp_mag.max()) < 0.3          # perp demagnetized
    assert float(vec_mag.min()) > 0.6           # vector held the moment
    # and the held state is genuinely antiferromagnetic: moments antiparallel
    mz = vec["M"][:, 2]
    assert float(mz[0]) > 0.5 and float(mz[1]) < -0.5


@pytest.mark.torture
def test_config_search_finds_ferromagnet():
    """Two O moments started 45 deg apart relax, under the torque, to the
    ferromagnetic (parallel) ground state, with the energy decreasing to the
    unconstrained value."""
    torch.set_num_threads(8)
    system = _o2_system()
    xc = NoncollinearXC(LSDA_PW92())
    w = atomic_weights(system)
    th = np.deg2rad(45.0)
    dirs0 = [[0.0, 0, 1], [float(np.sin(th)), 0, float(np.cos(th))]]

    final, hist = relax_moment_directions(
        system, xc, dirs0, lam=2.0, step=0.5, tol=1e-2, max_sweeps=25, weights=w,
        smearing="gaussian", width=0.1, etol=1e-7, rhotol=1e-6, max_iter=120,
        verbose=False)

    energies = [h["energy_eV"] for h in hist]
    assert energies == sorted(energies, reverse=True)   # monotone descent
    assert _relangle(final.tolist()) < 2.0              # parallel = ferromagnet
    assert hist[-1]["misalign_muB"] < 1e-2              # no constraint needed
