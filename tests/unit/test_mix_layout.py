"""MixLayout: pack/unpack must round-trip exactly for physical (real
densities, Hermitian becsum) inputs, and the derived vectors must have
the documented block structure."""

import types

import torch

from gradwave.scf.layout import MixLayout


def _mock_grid(shape=(6, 6, 6)):
    """Inversion-symmetric density sphere (like every real FFT grid) —
    an asymmetric mask would drop half of a real field's Hermitian
    spectrum and the round trip would rightly fail."""
    n = shape[0] * shape[1] * shape[2]
    fx = torch.fft.fftfreq(shape[0]) * shape[0]
    fy = torch.fft.fftfreq(shape[1]) * shape[1]
    fz = torch.fft.fftfreq(shape[2]) * shape[2]
    gx, gy, gz = torch.meshgrid(fx, fy, fz, indexing="ij")
    g2 = (gx**2 + gy**2 + gz**2).to(torch.float64)
    mask = g2 <= 4.0  # includes G=0, inversion-symmetric
    return types.SimpleNamespace(shape=shape, n_points=n, g2=g2,
                                 dens_mask=mask)


def _roundtrip(nspin):
    torch.manual_seed(5)
    grid = _mock_grid()
    slices = [(0, 3), (3, 5)]
    lay = MixLayout(grid, nspin, slices)

    # physical densities: real fields band-limited to the sphere
    rho_spin = []
    for _ in range(nspin):
        # Hermitian-symmetric spectrum → real field: build from a real field
        f = torch.rand(*grid.shape, dtype=torch.float64)
        c = torch.fft.fftn(f) / grid.n_points
        c.reshape(-1)[~grid.dens_mask.reshape(-1)] = 0.0
        rho_spin.append(torch.fft.ifftn(
            c.reshape(grid.shape) * grid.n_points).real)
    becs = [[torch.rand(s1 - s0, s1 - s0, dtype=torch.complex128)
             for (s0, s1) in slices] for _ in range(nspin)]

    v = lay.pack(rho_spin, becs)
    assert v.shape[0] == lay.size
    rho2, becs2 = lay.unpack(v)
    for a, b in zip(rho_spin, rho2, strict=True):
        assert float((a - b).abs().max()) < 1e-12
    for isp in range(nspin):
        for a, b in zip(becs[isp], becs2[isp], strict=True):
            assert float((a - b).abs().max()) < 1e-12


def test_roundtrip_nspin1():
    _roundtrip(1)


def test_roundtrip_nspin2():
    _roundtrip(2)


def test_derived_vectors_block_structure():
    grid = _mock_grid()
    lay = MixLayout(grid, 2, [(0, 2)])
    ng, nbec = lay.ng, lay.nbec
    assert lay.g2_full.shape[0] == 2 * ng + 2 * nbec
    assert bool(lay.kerker_mask[:ng].all())
    assert not bool(lay.kerker_mask[ng:].any())
    assert float(lay.step_scale[: 2 * ng].min()) == 1.0
    assert float(lay.step_scale[2 * ng:].max()) == 0.4
    assert lay.block_ids[:ng].eq(0).all()
    assert lay.block_ids[ng:2 * ng].eq(1).all()
    assert lay.block_ids[2 * ng:].eq(2).all()
