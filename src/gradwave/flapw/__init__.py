"""Differentiable all-electron FLAPW (full-potential linearized augmented plane waves) for gradwave.

A self-contained augmented-basis all-electron stack, in gradwave eV/Å units, built and verified
against independent references (NIST LDA atomic data; Elk 11 all-electron FLAPW):

- ``radial`` — log-mesh radial Schrödinger solvers (Numerov for u_l(E); tridiagonal for the bands)
- ``functionals`` — LDA XC (Slater exchange + PW92 correlation)
- ``coulomb`` — radial Hartree, interstitial FFT Poisson, and Weinert's pseudocharge/sphere matching
- ``mixing`` — Anderson mixing
- ``atom`` — the self-consistent spherical KS atom (the muffin-tin reference)
- ``lapw`` — the (L)APW secular equation (single- and multi-atom, differentiable primitives)
- ``scf`` — the self-consistent crystal FLAPW loop (muffin-tin): ``crystal_scf`` (single atom) and
  ``crystal_scf_multi`` (several atoms/species, a Monkhorst-Pack k-mesh, cubic or orthorhombic
  cells, and Fermi smearing for metals)

Example — a self-consistent simple-cubic Ne crystal, and its atomic limit::

    from gradwave.flapw import crystal_scf
    bands, atomic = crystal_scf(6.0, "Ne")          # a = 6 Bohr
    print(bands["2p"] - bands["2s"])                # 2s-2p splitting (eV)

Example — a multi-atom cell with a BZ mesh (and Fermi smearing for a metal)::

    from gradwave.flapw import crystal_scf_multi
    bands, info = crystal_scf_multi(6.0, [((0.0, 0.0, 0.0), "Ne")], {"Ne": 1.4},
                                    kmesh=(2, 2, 2), smearing=0.1)

The muffin-tin ``scf`` is a numerical (LAPACK) forward solver; the ``lapw`` and ``atom`` primitives
are autograd-differentiable in the potential/basis parameters (see the gate-D atomic oracle and the
differentiable-band demonstrations in ``experiments/autoapw``).
"""

from gradwave.flapw.atom import CONFIG, NIST_LDA_EV, atomic_scf
from gradwave.flapw.efg import (
    efg_tensor,
    l2_sphere_poisson,
    sphere_density_multipoles,
    valence_efg_moments,
)
from gradwave.flapw.lapw import build_matrices, build_matrices_multi, solve_geneig
from gradwave.flapw.radial import (
    log_mesh,
    numerov_log,
    numerov_log_batch,
    radial_eigs_tridiag,
)
from gradwave.flapw.scf import crystal_scf, crystal_scf_multi

__all__ = [
    "CONFIG",
    "NIST_LDA_EV",
    "atomic_scf",
    "build_matrices",
    "build_matrices_multi",
    "crystal_scf",
    "crystal_scf_multi",
    "efg_tensor",
    "l2_sphere_poisson",
    "log_mesh",
    "numerov_log",
    "numerov_log_batch",
    "radial_eigs_tridiag",
    "solve_geneig",
    "sphere_density_multipoles",
    "valence_efg_moments",
]
