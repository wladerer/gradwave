import torch

from gradwave.core.occupations import (
    SCHEMES,
    find_fermi,
    fixed_occupations,
    occupations_and_entropy,
)


def fake_bands(nk=6, nb=10, seed=1):
    gen = torch.Generator().manual_seed(seed)
    eigs = torch.sort(torch.randn(nk, nb, generator=gen, dtype=torch.float64) * 3.0, dim=1).values
    w = torch.full((nk,), 1.0 / nk, dtype=torch.float64)
    return eigs, w


def test_electron_count_conserved():
    eigs, w = fake_bands()
    for scheme in SCHEMES.values():
        for ne in (4.0, 7.0, 11.0):
            mu = find_fermi(eigs, w, scheme, width=0.2, n_electrons=ne)
            f, _ = occupations_and_entropy(eigs, mu, scheme, width=0.2)
            n = (w[:, None] * f).sum().item()
            assert abs(n - ne) < 1e-9


def test_zero_width_limit_recovers_integers():
    eigs, w = fake_bands()
    ne = 8.0
    scheme = SCHEMES["gaussian"]
    mu = find_fermi(eigs, w, scheme, width=1e-6, n_electrons=ne)
    f, s = occupations_and_entropy(eigs, mu, scheme, width=1e-6)
    # far from any eigenvalue crossing, occupations are 0 or 2 and entropy ~ 0
    frac = torch.minimum(f, 2.0 - f)
    assert (frac < 1e-6).float().mean() > 0.9
    assert s.sum() < 1e-3


def test_fd_entropy_matches_definition():
    x = torch.linspace(-8, 8, 41, dtype=torch.float64)
    fd = SCHEMES["fermi-dirac"]
    f = fd.occupation(x)
    inner = torch.clamp(f, 1e-300) .log() * f + torch.clamp(1 - f, 1e-300).log() * (1 - f)
    assert torch.allclose(fd.entropy(x), -inner, atol=1e-12)


def test_entropy_positive_and_peaked_at_mu():
    x = torch.linspace(-5, 5, 101, dtype=torch.float64)
    for scheme in SCHEMES.values():
        s = scheme.entropy(x)
        assert torch.all(s >= 0)
        assert s.argmax() == 50  # peak at x = 0 (ε = μ)


def test_fixed_occupations():
    eigs, _ = fake_bands()
    f = fixed_occupations(eigs, 8.0)
    assert torch.all(f[:, :4] == 2.0) and torch.all(f[:, 4:] == 0.0)
