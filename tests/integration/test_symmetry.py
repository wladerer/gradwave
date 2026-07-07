"""Symmetry validation. The decisive test: IBZ-reduced + density-symmetrized
SCF must equal the full-mesh (TR-only) SCF to near machine precision — Si
exercises non-symmorphic glide phases (diamond, Fd-3̄m), Al the symmorphic
metallic path. IBZ counts are cross-checked against QE's pw.out.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.symmetry import RhoSymmetrizer, find_spacegroup, reduce_mesh

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
A = 5.43
SI_CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0, 0], [A / 4] * 3])
AL_CELL = 4.05 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def test_si_spacegroup_and_ibz_counts():
    frac = SI_POS @ np.linalg.inv(SI_CELL)
    sg = find_spacegroup(SI_CELL, frac, [0, 0])
    assert sg.international == "Fd-3m"
    assert sg.n_ops == 48
    # non-symmorphic: some operations carry fractional translations
    assert np.abs(sg.translations).max() > 0.1

    k, w = reduce_mesh((4, 4, 4), (0, 0, 0), sg)
    assert len(k) == 8  # matches QE "number of k points= 8" for this mesh
    assert abs(w.sum() - 1.0) < 1e-12
    k2, _ = reduce_mesh((2, 2, 2), (0, 0, 0), sg)
    assert len(k2) == 3


def test_symmetrizer_idempotent_and_invariant():
    from gradwave.grids import build_fft_grid

    frac = SI_POS @ np.linalg.inv(SI_CELL)
    sg = find_spacegroup(SI_CELL, frac, [0, 0])
    grid = build_fft_grid(SI_CELL, 15 * RY, equal_dims=True)
    sym = RhoSymmetrizer(grid.shape, sg, dens_mask=grid.dens_mask)
    gen = torch.Generator().manual_seed(3)
    raw = torch.randn(*grid.shape, generator=gen, dtype=torch.float64)
    rho_g = torch.fft.fftn(raw).to(torch.complex128) / raw.numel()
    once = sym.apply(rho_g)
    twice = sym.apply(once)
    assert torch.allclose(once, twice, atol=1e-14)  # exact projector on the sphere
    # hermiticity survives (ρ(r) stays real)
    rho_r = torch.fft.ifftn(once * once.numel(), dim=(-3, -2, -1))
    assert float(rho_r.imag.abs().max()) < 1e-12 * float(rho_r.real.abs().max())


@pytest.mark.parametrize(
    ("name", "cell", "pos", "nat", "pseudo", "xc", "smearing", "nbands"),
    [
        ("si", SI_CELL, SI_POS, 2, "Si_ONCV_PBE-1.2.upf", LDA_PW92, "none", None),
        ("al", AL_CELL, np.zeros((1, 3)), 1, "Al_ONCV_PBE-1.2.upf", PBE, "gaussian", 10),
    ],
)
def test_ibz_scf_equals_full_mesh(name, cell, pos, nat, pseudo, xc, smearing, nbands):
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / pseudo)
    results = {}
    for use_sym in (False, True):
        system = setup_system(
            cell, pos, [0] * nat, [upf], ecut=15 * RY, kmesh=(2, 2, 2),
            nbands=nbands, use_symmetry=use_sym,
        )
        res = scf(system, xc(), smearing=smearing, width=0.1,
                  etol=1e-10, rhotol=1e-9, verbose=False)
        assert res.converged
        results[use_sym] = res

    e_full = float(results[False].energies.free_energy)
    e_ibz = float(results[True].energies.free_energy)
    assert abs(e_full - e_ibz) < 5e-7, f"{name}: {e_full} vs {e_ibz}"
    assert len(results[True].system.spheres) < len(results[False].system.spheres)


def test_symmetrized_forces_on_displaced_si():
    # Two identical atoms ALWAYS have midpoint inversion symmetry (P-1), so
    # the symmetric run enforces F1 = -F2 exactly; the unsymmetrized run
    # differs at the XC egg-box level (~5e-5 eV/Å at this cutoff).
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    pos = np.array([[0.0, 0, 0], [0.24, 0.26, 0.255]]) @ SI_CELL
    fvals = {}
    for use_sym in (False, True):
        system = setup_system(SI_CELL, pos, [0, 0], [upf], ecut=15 * RY,
                              kmesh=(2, 2, 2), use_symmetry=use_sym)
        res = scf(system, LDA_PW92(), smearing="none",
                  etol=1e-10, rhotol=1e-9, verbose=False)
        fvals[use_sym] = forces(res)
    assert torch.allclose(fvals[False], fvals[True], atol=1e-4)
    f = fvals[True]
    assert torch.allclose(f[0], -f[1], atol=1e-10)  # exact by symmetrization
