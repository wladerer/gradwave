"""Phase 2 FLAPW-DFPT: the augmented-basis chi_0 density-response calculus.

Fast, pure-primitive (no SCF) gates for :mod:`gradwave.flapw.dfpt` — the pieces that turn a
converged augmented eigenstate + a secular-matrix perturbation into the l=2 muffin-tin density
response and, through the nonlinear tensor eigenvalue, ``dV_zz``:

* ``eig_response_occupied`` — the first-order eigenvector response of the GENERALIZED secular
  problem (``dH != 0`` and, decisively, ``dS != 0`` — the moving-muffin-tin overlap derivative)
  reproduces the finite difference of the occupied-projector density matrix. This is the exact
  test of the metric/normalisation term the plane-wave DFPT never sees;
* ``dvzz_du_jvp`` — the nonlinear ``(v0,v1,v2) -> V_zz`` (largest-|eigenvalue|) Jacobian is
  autograd-exact vs a central finite difference;
* ``dmats_response`` / ``multipoles_from_dmats`` — the density-matrix -> rho_LM contraction is
  the exact derivative of the base multipole build under a first-order amplitude perturbation.

The converged-rutile numbers (bare chi_0 dV_zz/du = Ti +2469 within 2% of the screened FD
reference, O +1143 with the Phase-1 explicit part reproducing -156) live in
``experiments/autoapw/dfpt_validate.py`` (asus).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh as scipy_eigh

from gradwave.flapw import dfpt


def _herm(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _gapped_pencil(rng, n, nocc):
    """A Hermitian (H0, SPD S0) generalized pencil with a prescribed occupied/empty gap.

    Build an S0-orthonormal eigenbasis C0 (from an auxiliary generalized eigensolve) and a
    spectrum with a wide gap after ``nocc``, then reconstruct ``H0 = S0 C0 diag(lam) C0^H S0`` so
    ``(H0, S0)`` has eigenvectors C0 and eigenvalues lam — no fragile random-gap dependence."""
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    S0 = B @ B.conj().T + n * np.eye(n)
    _, C0 = scipy_eigh(_herm(rng, n), S0)                # C0^H S0 C0 = I
    lam = np.array([float(i) for i in range(nocc)] + [10.0 + i for i in range(n - nocc)])
    M = S0 @ C0
    H0 = M @ np.diag(lam) @ M.conj().T
    H0 = 0.5 * (H0 + H0.conj().T)
    return H0, S0


def _occ_dmat(ev, C, occ):
    """Occupied-projector density matrix D = sum_n occ_n c_n c_n^H (coefficient space)."""
    D = np.zeros((C.shape[0], C.shape[0]), dtype=complex)
    for n in range(len(occ)):
        if occ[n] > 0:
            D += occ[n] * np.outer(C[:, n], C[:, n].conj())
    return D


def test_eig_response_generalized_with_ds():
    """dc_n from eig_response_occupied reproduces the FD of the occupied density matrix, with a
    nonzero dS (the moving-boundary overlap derivative) — the gauge-invariant test."""
    rng = np.random.default_rng(1)
    n = 8
    nocc = 3
    H0, S0 = _gapped_pencil(rng, n, nocc)
    dH = _herm(rng, n)
    dS = 0.3 * _herm(rng, n)
    occ = np.array([2.0] * nocc + [0.0] * (n - nocc))

    def solve(s):
        ev, C = scipy_eigh(H0 + s * dH, S0 + s * dS)   # C^H S C = I, ascending ev
        return ev, C

    ev0, C0 = solve(0.0)
    # gap check: occupied manifold isolated
    assert ev0[nocc] - ev0[nocc - 1] > 0.5
    dC, occ_idx = dfpt.eig_response_occupied(ev0, C0, dH, dS, occ)
    # analytic dD = sum_occ (dc_n c_n^H + c_n dc_n^H)
    dD_an = np.zeros((n, n), dtype=complex)
    for j, nidx in enumerate(occ_idx):
        dD_an += occ[nidx] * (np.outer(dC[:, j], C0[:, nidx].conj())
                              + np.outer(C0[:, nidx], dC[:, j].conj()))
    h = 1e-5
    evp, Cp = solve(h)
    evm, Cm = solve(-h)
    dD_fd = (_occ_dmat(evp, Cp, occ) - _occ_dmat(evm, Cm, occ)) / (2 * h)
    assert np.abs(dD_an - dD_fd).max() < 1e-6 * max(np.abs(dD_fd).max(), 1e-30)


def test_eig_response_pure_dh():
    """With dS = 0 the response reduces to plain sum-over-states; still matches the FD density."""
    rng = np.random.default_rng(2)
    n = 6
    nocc = 2
    H0, S0 = _gapped_pencil(rng, n, nocc)
    dH = _herm(rng, n)
    dS = np.zeros((n, n), dtype=complex)
    occ = np.array([2.0] * nocc + [0.0] * (n - nocc))
    ev0, C0 = scipy_eigh(H0, S0)
    dC, occ_idx = dfpt.eig_response_occupied(ev0, C0, dH, dS, occ)
    dD_an = np.zeros((n, n), dtype=complex)
    for j, nidx in enumerate(occ_idx):
        dD_an += occ[nidx] * (np.outer(dC[:, j], C0[:, nidx].conj())
                              + np.outer(C0[:, nidx], dC[:, j].conj()))
    h = 1e-5
    evp, Cp = scipy_eigh(H0 + h * dH, S0)
    evm, Cm = scipy_eigh(H0 - h * dH, S0)
    dD_fd = (_occ_dmat(evp, Cp, occ) - _occ_dmat(evm, Cm, occ)) / (2 * h)
    assert np.abs(dD_an - dD_fd).max() < 1e-6 * max(np.abs(dD_fd).max(), 1e-30)


def test_dvzz_jvp_matches_fd():
    """The nonlinear (v0,v1,v2) -> V_zz Jacobian (autograd) equals a central finite difference."""
    rng = np.random.default_rng(3)
    vb = {m: complex(rng.standard_normal(), rng.standard_normal()) for m in range(3)}
    vb[0] = complex(vb[0].real, 0.0)                   # v0 enters as its real part
    dv = {m: complex(rng.standard_normal(), rng.standard_normal()) for m in range(3)}
    dv[0] = complex(dv[0].real, 0.0)
    grad, _ = dfpt.dvzz_du_jvp(vb, dv)

    def vzz_of(s):
        _, vz = dfpt.dvzz_du_jvp({m: vb[m] + s * dv[m] for m in range(3)}, {0: 0j, 1: 0j, 2: 0j})
        return vz

    h = 1e-6
    fd = (vzz_of(h) - vzz_of(-h)) / (2 * h)
    assert abs(grad - fd) < 1e-5 * max(abs(fd), 1.0)


def test_multipole_response_is_derivative():
    """dmats_response + multipoles_from_dmats is the exact derivative of the base multipole build
    (dmats_from_amps + multipoles_from_dmats) under amps(s) = amps0 + s*damps."""
    rng = np.random.default_rng(4)
    lmax = 2
    nb, nr = 4, 20
    rr = np.linspace(0.05, 1.0, nr)
    us = {l: [rng.standard_normal(nr), rng.standard_normal(nr)] for l in range(lmax + 1)}
    occ = np.array([2.0, 2.0, 0.0, 0.0])

    def rand_amps():
        return {l: [rng.standard_normal((nb, 2 * l + 1)) + 1j * rng.standard_normal((nb, 2 * l + 1))
                    for _ in range(2)] for l in range(lmax + 1)}

    amps = rand_amps()
    damps = rand_amps()
    lset = [(2, m) for m in range(-2, 3)]

    def rho_of(s):
        a = {l: [amps[l][p] + s * damps[l][p] for p in range(2)] for l in range(lmax + 1)}
        dm = dfpt.dmats_from_amps(a, occ, lmax)
        return dfpt.multipoles_from_dmats(dm, us, rr, lmax, lset)

    dd = dfpt.dmats_response(amps, damps, occ, lmax)
    dr_an = dfpt.multipoles_from_dmats(dd, us, rr, lmax, lset)
    h = 1e-6
    for lm in lset:
        dr_fd = (rho_of(h)[lm] - rho_of(-h)[lm]) / (2 * h)
        assert np.abs(dr_an[lm] - dr_fd).max() < 1e-6 * max(np.abs(dr_fd).max(), 1e-30)


def test_fxc_lda_matches_fd_of_vxc():
    """The LDA XC kernel f_xc = dV_xc/drho (functionals.fxc_lda) equals a central finite
    difference of vxc_lda, elementwise, over a wide density range — the K_Hxc XC building block."""
    import torch

    from gradwave.flapw.functionals import fxc_lda, vxc_lda
    rho = torch.tensor([0.02, 0.1, 0.5, 2.0, 10.0, 50.0], dtype=torch.float64)
    an = fxc_lda(rho)
    h = 1e-6
    fd = (vxc_lda(rho * (1 + h)) - vxc_lda(rho * (1 - h))) / (2 * h * rho)
    assert torch.abs(an - fd).max() < 1e-7 * float(torch.abs(fd).max())


def test_khxc_response_matches_fd_of_nonspherical_potential():
    """K_Hxc (dfpt.khxc_response) is the exact JVP of efg.nonspherical_potential: the induced
    aspherical potential response dV_LM to a density response drho_LM reproduces a central finite
    difference of the forward density->potential map, for every l=2 channel."""
    from gradwave.flapw.dfpt import khxc_response
    from gradwave.flapw.efg import nonspherical_potential
    rng = np.random.default_rng(11)
    nr = 160
    rr = np.linspace(0.02, 1.0, nr)
    drw = np.gradient(rr)
    rho_sph = 3.0 * np.exp(-2.0 * rr) + 0.2                # positive spherical background
    lset = [(2, m) for m in range(-2, 3)]
    rho_2m = {lm: (0.15 * np.exp(-1.5 * rr) * rng.standard_normal()
                   + 0.05j * np.exp(-1.5 * rr) * rng.standard_normal()) for lm in lset}
    rho_2m[(2, 0)] = 0.15 * np.exp(-1.5 * rr)             # real m=0
    drho = {lm: 0.1 * np.exp(-1.8 * rr) * (rng.standard_normal() + 1j * rng.standard_normal())
            for lm in lset}
    drho[(2, 0)] = 0.1 * np.exp(-1.8 * rr) * rng.standard_normal()
    an = khxc_response(drho, rho_sph, rho_2m, rr, drw, lset)
    h = 1e-6
    vp = nonspherical_potential(rho_sph, {lm: rho_2m[lm] + h * drho[lm] for lm in lset},
                                rr, drw, lset=lset)
    vm = nonspherical_potential(rho_sph, {lm: rho_2m[lm] - h * drho[lm] for lm in lset},
                                rr, drw, lset=lset)
    for lm in lset:
        fd = (vp[lm] - vm[lm]) / (2 * h)
        assert np.abs(an[lm] - fd).max() < 1e-5 * max(np.abs(fd).max(), 1e-30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
