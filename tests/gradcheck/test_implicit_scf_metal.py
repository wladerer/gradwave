"""χ₀ for metals: the partial-occupation (Fermi-surface) independent-particle
response.

The metallic path in ``scf/implicit.py`` (``_chi0_channel_metal``) is validated
three ways, all on the real operator:

1. Against a finite difference of the bare (non-self-consistent) density. χ₀ w
   is by definition the first-order density response to a static local potential
   w at FIXED electron number: perturb the converged KS potential by ±h·w,
   dense-diagonalize every k, refill at the same smearing with the Fermi level
   floating, and difference the density. This is the ground truth for every
   piece at once — the above-window Sternheimer, the in-window divided-difference
   band pairs, and the rank-one δμ Fermi-level shift.
2. Charge conservation ∫ χ₀ w = 0, the defining property of the δμ term (the
   response cannot change the electron count).
3. The insulator limit: forced down the metallic path, a gapped (smeared but
   integer-filled) Si reproduces the exact insulator ``_chi0_channel`` — the
   "same code, not a special case" property, the metallic response reducing
   smoothly to the insulator one as the Fermi-surface weight occ' → 0.

Slow tier: a metal SCF plus dense per-k diagonalizations for the FD reference.
"""

import numpy as np
import pytest
import torch

from gradwave.core.fftbox import g_to_r
from gradwave.core.hamiltonian import HamiltonianK, projectors
from gradwave.core.occupations import (
    SCHEMES,
    find_fermi,
    occupations_and_entropy,
)
from gradwave.core.xc.learnable import LearnableX
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.implicit import (
    _chi0_channel,
    _chi0_channel_metal,
    _hamiltonians,
    apply_chi0,
)
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo, si_fcc

pytestmark = pytest.mark.slow

FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _al_metal():
    """fcc Al, gaussian-smeared — a nearly-free-electron metal with genuine
    fractional occupations at every k (a 2×2×2 mesh needs the wide 0.5 eV
    smearing before anything is fractional). ``use_symmetry=False``: a local
    perturbation breaks the crystal symmetry, so the response uses the full
    mesh."""
    torch.set_num_threads(4)
    upf = parse_upf(pseudo("Al_ONCV_PBE-1.2.upf"))
    cell = 4.05 / 2.0 * FCC
    system = setup_system(cell, np.zeros((1, 3)), [0], [upf], ecut=16 * RY,
                          kmesh=(2, 2, 2), nbands=12, use_symmetry=False)
    res = scf(system, PBE(), smearing="gaussian", width=0.5, etol=1e-10,
              rhotol=1e-9, verbose=False, max_iter=200)
    assert res.converged
    return res


def _density_from_potential(res, vloc, nb_keep=16):
    """Bare one-shot density in the local potential ``vloc``: dense-diagonalize
    each k, find the Fermi level for the converged electron count, and fill with
    the run's own smearing. NOT self-consistent — exactly the independent-
    particle response χ₀ integrates."""
    grid = res.system.grid
    kw = res.system.kweights
    scheme = SCHEMES[res.smearing]
    n_elec = float((kw[:, None] * res.occupations).sum())
    evals, evecs = [], []
    for ik, sph in enumerate(res.system.spheres):
        p = projectors(res.system.proj_data[ik], res.system.positions)
        h = HamiltonianK(sph, grid.shape, vloc, res.system.proj_data[ik], p)
        npw = res.coeffs[ik].shape[1]
        h_mat = h.apply(torch.eye(npw, dtype=CDTYPE)).transpose(-1, -2)
        h_mat = 0.5 * (h_mat + h_mat.conj().transpose(-1, -2))  # Hermitize
        ev, v = torch.linalg.eigh(h_mat)
        evals.append(ev[:nb_keep])
        evecs.append(v[:, :nb_keep])
    e = torch.stack(evals)
    mu = find_fermi(e, kw, scheme, res.width, n_elec, degeneracy=2.0)
    f, _ = occupations_and_entropy(e, mu, scheme, res.width, degeneracy=2.0)
    rho = torch.zeros(grid.shape, dtype=RDTYPE)
    ax = (-1, *([1] * len(grid.shape)))
    for ik, sph in enumerate(res.system.spheres):
        psi_r = g_to_r(evecs[ik].transpose(-1, -2), sph.flat_idx, grid.shape)
        rho += float(kw[ik]) * (f[ik].view(*ax) * (psi_r.conj() * psi_r).real).sum(0)
    return rho / grid.volume


def test_metal_chi0_matches_finite_difference():
    res = _al_metal()
    # a real metal: fractional occupations exist (else there is nothing to test)
    frac = (res.occupations > 1e-6) & (res.occupations < 2.0 - 1e-6)
    assert int(frac.sum()) > 0

    # the FD reference reproduces the converged density unperturbed
    rho0 = _density_from_potential(res, res.v_eff)
    assert float(torch.linalg.vector_norm(rho0 - res.rho)
                 / torch.linalg.vector_norm(res.rho)) < 1e-8

    g = torch.Generator().manual_seed(1)
    w = torch.randn(res.rho.shape, dtype=res.rho.dtype, generator=g)
    w = w - w.mean()  # a zero-mean local perturbation

    h = 1e-3
    fd = (_density_from_potential(res, res.v_eff + h * w)
          - _density_from_potential(res, res.v_eff - h * w)) / (2.0 * h)
    an = apply_chi0(res, w, tol=1e-8, max_iter=400)

    rel = float(torch.linalg.vector_norm(an - fd)
                / torch.linalg.vector_norm(fd))
    assert rel < 1e-5, rel
    # charge conservation: the δμ Fermi-shift keeps ∫χ₀w = 0 to machine zero
    assert abs(float(an.sum())) < 1e-9, float(an.sum())


def test_metal_chi0_reduces_to_insulator_limit():
    """Forced down the metallic path, a gapped smeared Si reproduces the exact
    insulator response — the smooth occ' → 0 limit, verified on the real
    operator (the two paths partition the Sternheimer projection differently:
    occupied-only vs whole-window, yet integrate to the same density)."""
    torch.set_num_threads(4)
    cell, pos = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    # a narrow smearing far below Si's gap → integer occupations, μ in the gap,
    # occ' ≈ 0 everywhere, but the metallic code path is fully exercisable
    system = setup_system(cell, pos, [0, 0], [upf], ecut=12 * RY,
                          kmesh=(1, 1, 1), nbands=8, use_symmetry=False)
    xc = LearnableX(kappa=0.70, mu=0.20)
    res = scf(system, xc, smearing="gaussian", width=0.05, etol=1e-11,
              rhotol=1e-10, verbose=False)
    assert res.converged
    # integer filled: the metallic path here is a limit test, not a metal
    assert bool(((res.occupations < 1e-6) | (res.occupations > 2.0 - 1e-6)).all())

    hs = _hamiltonians(res)
    g = torch.Generator().manual_seed(3)
    w = torch.randn(res.rho.shape, dtype=res.rho.dtype, generator=g)
    w = w - w.mean()

    ins = _chi0_channel(res, hs, 0, w, 2.0, tol=1e-9, max_iter=400)
    met = _chi0_channel_metal(res, hs, 0, w, 2.0, float(res.fermi),
                              SCHEMES[res.smearing], res.width,
                              tol=1e-9, max_iter=400)
    rel = float(torch.linalg.vector_norm(met - ins)
                / torch.linalg.vector_norm(ins))
    assert rel < 1e-6, rel
