"""Physics EDGE-CASE robustness — each catches a failure class the mainstream
cubic-Si tests cannot see.

1. Single-electron H atom (nspin=2, one occupied ↑ state): the moment must be
   exactly 1 μB, the energy finite/sane, and the LDA self-interaction must be
   *present and finite* (E_H > 0, not spuriously cancelled) — a broken
   single-occupation nspin=2 path or a SIC-sign slip breaks one of these.
2. Near-Stoner ferromagnetic moment with the shipped ``SpinAdaptedPBE`` preset —
   the full-SCF moment shift its unit-test docstring names as "the standard-tier
   validation" but that was never committed. A ζ²-exchange sign/scale bug shows
   up as a moment outside the physical ferromagnetic band.
3. ecut extremes / grid-edge stability: build the FFT grid + one density→XC
   evaluation at a very small and a very large ecut for a fixed cell — no NaN/Inf
   at either extreme and a monotone-ish convergence of the XC energy in between
   (catches aliasing at tiny ecut / FFT-size-edge bugs at large ecut).
"""

import math

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.learnable import SpinAdaptedPBE
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.grids import build_fft_grid
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY, pseudo


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(8)


# --- 1. single-electron H atom ----------------------------------------------

@pytest.mark.standard
def test_single_electron_hydrogen_atom():
    """One H atom, one electron, spin-polarized in a vacuum box.

    tot_magnetization=1 fixes N↑=1, N↓=0 (integer occupations, no smearing), so
    the converged state is a single spin-up electron. Asserts:
      * the moment is exactly 1 μB (broken nspin=2 single-occupation path → not 1),
      * the total energy is finite and bounded (not diverged/NaN),
      * the LDA self-interaction error is PRESENT and finite — a one-electron
        density still feels its own Hartree repulsion (E_H > 0), which LDA XC only
        partially cancels (E_xc < 0). A SIC sign flip or a zeroed self-Hartree
        would break the E_H > 0 / E_xc < 0 structure.
    """
    h = parse_upf(pseudo("H_ONCV_PBE-1.2.upf"))
    a = 8.0  # Å cubic box, ~8 Å vacuum around the single atom
    cell = np.diag([a, a, a])
    pos = np.array([[a / 2, a / 2, a / 2]])  # centered
    system = setup_system(cell, pos, [0], [h], ecut=25 * RY,
                          kmesh=(1, 1, 1), use_symmetry=False, nbands=4)
    r = scf(system, LSDA_PW92(), nspin=2, tot_magnetization=1.0,
            smearing="none", etol=1e-8, rhotol=1e-7, max_iter=100, verbose=False)
    assert r.converged, "single-electron H atom SCF did not converge"

    # (a) moment is exactly one Bohr magneton
    assert abs(float(r.mag_total) - 1.0) < 1e-6, (
        f"H-atom moment {float(r.mag_total):.6f} μB != 1")

    # (b) finite, bounded total energy (the pseudo's energy zero makes the
    # absolute value pseudo-dependent, so this is a sanity band, not a reference)
    etot = float(r.energies.total)
    assert math.isfinite(etot)
    assert -60.0 < etot < 20.0, f"H-atom total energy {etot:.4f} eV out of sane band"

    # (c) the self-interaction is present and finite: a single electron still has
    # a positive Hartree self-energy, only partially cancelled by LDA exchange-
    # correlation. This is the diagnostic edge case — the SIE must not be zero.
    e_h = float(r.energies.hartree)
    e_xc = float(r.energies.xc)
    assert math.isfinite(e_h) and math.isfinite(e_xc)
    assert e_h > 1e-3, f"one-electron self-Hartree E_H={e_h:.4f} eV should be > 0 (SIE)"
    assert e_xc < 0.0, f"LDA E_xc={e_xc:.4f} eV should be negative"


# --- 2. near-Stoner moment with SpinAdaptedPBE -------------------------------

def _fe_bcc(ecut_ry=30.0, kmesh=(3, 3, 3), nbands=14):
    """1-atom primitive bcc Fe — the same coarse, fast cell the committed
    ``test_fsm_smeared`` FSM gate uses (there it self-consistently magnetizes to
    m₀ > 1 with SpinPBE), so it is a known-magnetic, known-cheap fixture."""
    a = 2.87
    cell = 0.5 * a * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    pos = np.zeros((1, 3))
    fe = parse_upf(str(PSEUDOS / "Fe_ONCV_PBE-1.2.upf"))
    return setup_system(cell, pos, [0], [fe], ecut=ecut_ry * RY,
                        kmesh=kmesh, nbands=nbands)


@pytest.mark.standard
def test_spin_adapted_pbe_ferromagnetic_moment_fe():
    """The full-SCF moment shift SpinAdaptedPBE's unit test calls "the standard-
    tier validation": an unconstrained collinear SCF on bcc Fe with the shipped
    ``SpinAdaptedPBE`` preset must converge to a stable moment inside the physical
    ferromagnetic band. bcc Fe is ~2.2 μB experimentally; the ζ² adaptation
    (μ₁ = −0.0475) shifts it modestly, and this coarse 30 Ry / 3³ cell resolves
    the moment approximately, so the band is generous [1.4, 3.2] μB — wide enough
    to survive the coarse cell, tight enough that a ζ²-exchange SIGN error (which
    quenches or explodes the moment) fails it.

    Also runs plain SpinPBE on the same cell as a companion so a regression can
    tell a preset-specific bug from a shared-loop bug: both must land in the band.
    """
    kw = dict(nspin=2, smearing="gaussian", width=0.1, start_mag=[0.5],
              mixing_scheme="pulay", max_iter=150, etol=1e-8, rhotol=1e-7,
              verbose=False)
    r_pbe = scf(_fe_bcc(), SpinPBE(), **kw)
    r_sa = scf(_fe_bcc(), SpinAdaptedPBE(), **kw)
    assert r_pbe.converged, "SpinPBE Fe SCF did not converge"
    assert r_sa.converged, "SpinAdaptedPBE Fe SCF did not converge"

    m_pbe = float(r_pbe.mag_total)
    m_sa = float(r_sa.mag_total)
    assert 1.4 < m_pbe < 3.2, f"SpinPBE Fe moment {m_pbe:.3f} μB out of band"
    assert 1.4 < m_sa < 3.2, f"SpinAdaptedPBE Fe moment {m_sa:.3f} μB out of band"
    # both energies finite (stability)
    assert math.isfinite(float(r_sa.energies.free_energy))


# --- 3. ecut extremes / grid-edge stability ----------------------------------

def _gaussian_density(grid, n_electrons=8.0, width_frac=0.18):
    """A smooth normalized Gaussian charge packet sampled on the real grid of
    ``grid`` (centered in the box). Analytic and band-limited-ish, so its XC
    energy is well-defined at any resolution and converges as the grid refines."""
    n1, n2, n3 = grid.shape
    cell = torch.as_tensor(grid.cell, dtype=torch.float64)
    # fractional grid coordinates in [0,1)
    f = [torch.arange(n, dtype=torch.float64) / n for n in (n1, n2, n3)]
    ff = torch.stack(torch.meshgrid(*f, indexing="ij"), dim=-1)  # (n1,n2,n3,3)
    # minimum-image displacement from the box center, in Cartesian Å
    d = ff - 0.5
    r = d @ cell  # (n1,n2,n3,3)
    r2 = (r * r).sum(dim=-1)
    sigma = width_frac * float(np.linalg.norm(grid.cell, axis=1).min())
    rho = torch.exp(-r2 / (2.0 * sigma**2))
    dvol = grid.volume / grid.n_points
    return rho * (n_electrons / (rho.sum() * dvol))  # normalize ∫ρ = N_e


@pytest.mark.standard
def test_xc_energy_stable_across_ecut_extremes():
    """Fixed cell, one fixed analytic density, swept across a very small and a
    very large ecut: the FFT grid build and a single XC evaluation must stay
    finite (no NaN/Inf) at both extremes, and the XC energy must converge
    monotone-ish as the grid refines. Catches aliasing at tiny ecut and
    FFT-size-edge bugs (e.g. a bad ``good_fft_size`` at large ecut).
    """
    cell = np.diag([6.0, 6.0, 6.0])
    xc = LDA_PW92()
    ecuts = [3.0, 6.0, 12.0, 30.0, 80.0, 200.0]  # Ry, tiny → very large
    energies = []
    shapes = []
    for e_ry in ecuts:
        grid = build_fft_grid(cell, e_ry * RY)
        shapes.append(grid.shape)
        rho = _gaussian_density(grid)
        e_xc = float(xc.energy(rho, grid.volume))
        assert math.isfinite(e_xc), f"E_xc not finite at ecut={e_ry} Ry"
        assert e_xc < 0.0, f"LDA E_xc should be negative, got {e_xc} at {e_ry} Ry"
        energies.append(e_xc)

    # grids must actually grow with ecut (edge-case: a stuck good_fft_size)
    npts = [s[0] * s[1] * s[2] for s in shapes]
    assert npts[-1] > npts[0], f"grid did not grow with ecut: {shapes}"

    # convergence: the tail (well-resolved) values are close, and successive
    # steps shrink — a monotone-ish approach to the continuum XC energy. The
    # coarsest grids alias the Gaussian, so only require the refined tail to
    # settle rather than strict monotonicity from the very first point.
    e = np.array(energies)
    assert abs(e[-1] - e[-2]) < abs(e[1] - e[0]), (
        f"XC energy not converging across ecut: {energies}")
    assert abs(e[-1] - e[-2]) < 0.05, (
        f"XC energy not settled at large ecut: {energies}")
