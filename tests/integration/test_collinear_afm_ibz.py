"""Collinear magnetic-IBZ SCF equivalence (FM/AFM, nspin=2).

A collinear antiferromagnet forfeits the ordinary space-group k-reduction: the
op that swaps the two spin sublattices is a symmetry of the crystal but not of
either spin channel, so a per-channel RhoSymmetrizer would fold ρ↑ and ρ↓
together. The system therefore "drops to time-reversal-only" and runs a
near-full mesh (bcc Cr AFM: 260 k for a 2-atom cell at 8×8×8).

The magnetic (Shubnikov) group keeps that swap as an ANTI-UNITARY op g·T: it
folds k as −W⁻ᵀ and, in the density, folds the spatial op with a spin-channel
SWAP (CollinearMagneticSymmetrizer). Folding into the magnetic IBZ must
reproduce the unreduced (use_symmetry=False) collinear SCF exactly — the fold
is a symmetry statement, not an approximation.

  bcc Cr AFM (Im-3m chemical, m∥z): 8 unitary + 8 anti-unitary. At (4,4,4) the
  correct collinear reference (TR-only per channel) is 36 k; the magnetic IBZ
  is 18 (2.0×). Larger meshes reclaim more (8×8×8: 260 → 75, 3.5×).
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY


def _cr_afm_run(mesh, ecut, **setup_kw):
    cr = parse_upf(PSEUDOS / "Cr_ONCV_PBE-1.2.upf")
    a = 2.88
    cell = np.diag([a, a, a])
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    system = setup_system(cell, pos, [0, 0], [cr], ecut=ecut, kmesh=mesh,
                          nbands=22, **setup_kw)
    # PBE (matching the PBE pseudo) sustains bcc Cr's itinerant AFM moment;
    # LDA quenches it here (m→0), which would silently degrade the fold to the
    # ordinary non-magnetic one (ρ↑=ρ↓) and never exercise the spin swap.
    res = scf(system, SpinPBE(), smearing="gaussian", width=0.1, nspin=2,
              start_mag=[0.6, -0.6], etol=1e-10, rhotol=1e-9, max_iter=250,
              verbose=False)
    assert res.converged
    return (float(res.energies.free_energy), float(res.mag_abs),
            len(system.spheres))


@pytest.mark.slow
def test_cr_afm_collinear_magnetic_ibz():
    """bcc Cr AFM: the collinear magnetic IBZ (spin-channel-swapping fold)
    reproduces the full time-reversal-only SCF energy and moment, on fewer k."""
    torch.set_num_threads(8)
    mesh, ecut = (4, 4, 4), 45 * RY
    e_full, m_full, nk_full = _cr_afm_run(mesh, ecut, use_symmetry=False)
    e_mag, m_mag, nk_mag = _cr_afm_run(
        mesh, ecut, use_symmetry=True,
        magmoms=[[0, 0, 1.0], [0, 0, -1.0]], collinear_magnetic=True)

    # (a) the irreducible k-count drops meaningfully vs TR-only
    assert (nk_full, nk_mag) == (36, 18)
    # (b) energy per atom and the AFM order parameter match the full mesh
    assert abs(e_full - e_mag) / 2 < 1e-6, f"|dE/atom| = {abs(e_full-e_mag)/2:.2e} eV"
    assert abs(m_full - m_mag) < 1e-4, f"|dM| = {abs(m_full-m_mag):.2e} µB"
    # genuinely magnetic (PBE bcc Cr AFM sustains a large itinerant moment),
    # not a spurious null result that would trivialize the spin-swap fold
    assert m_mag > 0.5


@pytest.mark.slow
def test_nonmagnetic_symmetry_unchanged():
    """A non-magnetic run built without magmoms is untouched: the ordinary
    space-group fold and the collinear-magnetic path must not interfere."""
    torch.set_num_threads(8)
    from gradwave.core.xc.lda_pw92 import LDA_PW92

    si = parse_upf(PSEUDOS / "Si_ONCV_PBE-1.2.upf")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])

    def run(**kw):
        system = setup_system(cell, pos, [0, 0], [si], ecut=20 * RY,
                              kmesh=(4, 4, 4), nbands=8, **kw)
        res = scf(system, LDA_PW92(), smearing="gaussian", width=0.05,
                  etol=1e-10, rhotol=1e-9, verbose=False)
        assert res.converged
        return float(res.energies.free_energy), len(system.spheres)

    e_nosym, nk_nosym = run(use_symmetry=False)
    e_sym, nk_sym = run(use_symmetry=True)
    assert nk_sym < nk_nosym  # ordinary IBZ reduction still active
    assert abs(e_nosym - e_sym) < 1e-6
