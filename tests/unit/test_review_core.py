"""Regression tests for the core code-review fixes.

Each test pins one review finding so the fix cannot silently regress:
projector fallback dtypes, becp reuse, non-collinear rho_core, the Hartree
G=0 mask, and the removal of the dead density_from_orbitals path.
"""

from types import SimpleNamespace

import torch

from gradwave.dtypes import CDTYPE, RDTYPE


def test_projector_fallback_dtype_device():
    """Empty-projector ProjectorData matches the populated path's dtype/device
    (finding 1/2): complex128 columns, float64 dij, all on the sphere device."""
    from gradwave.core.hamiltonian import build_projector_data

    npw = 5
    kpg = torch.randn(npw, 3, dtype=RDTYPE)
    sphere = SimpleNamespace(kpg=kpg, npw=npw)

    pd = build_projector_data(
        sphere, species_of_atom=[], beta_tables=[], beta_ls=[],
        dij_species=[], volume=10.0,
    )
    assert pd.f_ylm_phase_free.dtype == CDTYPE
    assert pd.f_ylm_phase_free.shape == (0, npw)
    assert pd.dij_full.dtype == RDTYPE
    assert pd.f_ylm_phase_free.device == kpg.device
    assert pd.dij_full.device == kpg.device


def test_becp_b_matches_inline_einsum():
    """becp_b with a cached conjugate reproduces the old inline contraction
    used in BatchedHamiltonian.apply (finding 10), to the bit."""
    from gradwave.core.batch import becp_b

    gen = torch.Generator().manual_seed(0)
    nk, nb, npw, nproj = 2, 3, 7, 4
    p = torch.randn(nk, nproj, npw, dtype=CDTYPE, generator=gen)
    c = torch.randn(nk, nb, npw, dtype=CDTYPE, generator=gen)
    p_conj = p.conj().resolve_conj()

    ref = torch.einsum("kpg,kbg->kbp", p_conj, c)
    assert torch.equal(becp_b(p, c, p_conj), ref)
    assert torch.equal(becp_b(p, c), ref)  # None path recomputes the conjugate


def test_noncollinear_energy_honours_rho_core():
    """NoncollinearXC.energy now folds rho_core into the rho± split exactly as
    energy_with_grid does (finding 6), so the two entry points agree."""
    from gradwave.core.xc.noncollinear import NoncollinearXC, energy_with_grid
    from gradwave.core.xc.spin import LSDA_PW92

    gen = torch.Generator().manual_seed(3)
    rho = 0.05 + 0.3 * torch.rand(6, generator=gen, dtype=RDTYPE)
    mz = 0.4 * rho * torch.rand(6, generator=gen, dtype=RDTYPE)
    rho_core = 0.02 + 0.1 * torch.rand(6, generator=gen, dtype=RDTYPE)
    m_vec = torch.stack([torch.zeros_like(mz), torch.zeros_like(mz), mz])

    nc = NoncollinearXC(LSDA_PW92())  # LSDA: no GGA sigma, grid only carries volume
    grid = SimpleNamespace(volume=1.0, n_points=6, g_cart=None)

    e_energy = nc.energy(rho, m_vec, volume=1.0, rho_core=rho_core)
    e_grid = energy_with_grid(nc, rho, m_vec, grid, rho_core=rho_core)
    assert torch.allclose(e_energy, e_grid, rtol=1e-12)

    # rho_core actually shifts the result: hand-computed collinear reference
    m_norm = mz.abs()
    r_up = 0.5 * (rho + rho_core + m_norm)
    r_dn = 0.5 * (rho + rho_core - m_norm)
    e_ref = LSDA_PW92().energy(r_up, r_dn, volume=1.0)
    assert torch.allclose(e_energy, e_ref, rtol=1e-12)
    e_no_core = nc.energy(rho, m_vec, volume=1.0)
    assert not torch.allclose(e_energy, e_no_core)


def test_hartree_g0_masked():
    """The Hartree masked inverse zeros the G=0 component (finding 8)."""
    from gradwave.core.energies.hartree import _inv_g2_masked

    g2 = torch.tensor([0.0, 1e-13, 2.0, 8.0], dtype=RDTYPE)
    inv = _inv_g2_masked(g2)
    assert inv[0] == 0.0
    assert inv[1] == 0.0  # below tolerance → excluded
    assert torch.allclose(inv[2:], 1.0 / g2[2:])


def test_density_from_orbitals_removed():
    """The dead density_from_orbitals entry point is gone (finding 3); the live
    path is core.batch.density_b."""
    from gradwave.core import density

    assert not hasattr(density, "density_from_orbitals")
    assert hasattr(density, "sigma_from_rho")
