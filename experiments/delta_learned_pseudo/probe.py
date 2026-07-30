"""Shared machinery for the differentiable-pseudopotential-correction probe.

Builds a Si diamond system with a theta-corrected local channel threaded through
System.vloc_atom, runs the plain NC SCF on it, and exposes a differentiable
E(theta) at the converged density (Hellmann-Feynman: v_loc enters the energy
linearly through the density, so at the SCF fixed point dE_total/dtheta =
dE_local/dtheta at frozen rho, by the envelope theorem).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch

SP = Path(__file__).parent
sys.path.insert(0, str(SP.parent.parent / "benchmarks" / "delta_gauge"))
from correction import corrected_vloc_atom  # noqa: E402
from lattices import geometry  # noqa: E402

from gradwave.core.energies.local_pp import local_energy, local_potential_g  # noqa: E402
from gradwave.core.fftbox import r_to_g  # noqa: E402
from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.dtypes import RDTYPE  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402

RY = 13.605693122994
V0_SI = 20.4530  # WIEN2k all-electron equilibrium volume, Ang^3/atom
PSEUDO = SP.parent.parent / "benchmarks" / "delta_gauge" / "pseudos" / "Si.upf"

_upf = None


def _get_upf():
    global _upf
    if _upf is None:
        _upf = parse_upf(PSEUDO)
    return _upf


def build_si(scale: float, ecut_ry: float, k: int,
             fft_shape=None):
    """Base (uncorrected) Si diamond System at volume factor `scale`."""
    cell, pos, elems = geometry("diamond", "Si", V0_SI, scale)
    return setup_system(cell, pos, [0] * len(elems), [_get_upf()],
                        ecut=ecut_ry * RY, kmesh=(k, k, k),
                        use_symmetry=True, fft_shape=fft_shape)


def build_si_displaced(scale, ecut_ry, k, disp):
    """Si diamond at `scale` with atom 1 displaced by `disp` (Cartesian Ang),
    built with use_symmetry=False so the reduced k-mesh and density symmetrizer
    match the broken symmetry (a force check needs the true low-symmetry cell)."""
    import numpy as np

    cell, pos, elems = geometry("diamond", "Si", V0_SI, scale)
    pos = np.asarray(pos, dtype=float).copy()
    pos[1] = pos[1] + np.asarray(disp, dtype=float)
    return setup_system(cell, pos, [0] * len(elems), [_get_upf()],
                        ecut=ecut_ry * RY, kmesh=(k, k, k),
                        use_symmetry=False)


def fixed_grid(scales, ecut_ry, k):
    """Elementwise-max FFT grid over the volume chain, so E(V) is not stepped by
    a grid change (matches benchmarks/delta_gauge/run_gw)."""
    dims = [build_si(s, ecut_ry, k).grid.shape for s in scales]
    return tuple(max(d[i] for d in dims) for i in range(3))


def apply_correction(system, theta, centers, species=0):
    """Return a copy of `system` with theta-corrected vloc_atom."""
    va = corrected_vloc_atom(system, theta, centers, species=species)
    return dataclasses.replace(system, vloc_atom=va)


def run_scf(system, start_from=None, etol=1e-9, rhotol=1e-8, max_iter=200):
    return scf(system, PBE(), etol=etol, rhotol=rhotol, max_iter=max_iter,
               verbose=False, start_from=start_from)


def energy_of_theta(res, system, theta, centers, theta_ref, species=0):
    """Differentiable total energy as a function of theta at the converged,
    detached density of `res`. `theta_ref` is the theta that produced `res`;
    the value equals res.energies.total at theta == theta_ref, and autograd wrt
    theta is the exact Hellmann-Feynman dE_total/dtheta.

    Only the local term carries theta, so E(theta) = E_total_converged
    - E_local(theta_ref) + E_local(theta), all local terms at the frozen rho.
    """
    grid = system.grid
    rho_g = r_to_g(res.rho.detach().to(torch.complex128))
    # local energy at theta_ref (the value the converged total already contains)
    va0 = corrected_vloc_atom(system, theta_ref.detach(), centers, species)
    vloc0 = local_potential_g(system.positions, system.species_index,
                              system.vloc_tables, grid.g_cart, grid.volume,
                              vloc_atom=va0.detach())
    e_loc0 = local_energy(rho_g, vloc0, grid.volume).detach()
    # theta-dependent local energy at the same frozen density
    va = corrected_vloc_atom(system, theta, centers, species)
    vloc = local_potential_g(system.positions, system.species_index,
                             system.vloc_tables, grid.g_cart, grid.volume,
                             vloc_atom=va)
    e_loc = local_energy(rho_g, vloc, grid.volume)
    e_total_ref = torch.as_tensor(float(res.energies.total), dtype=RDTYPE)
    return e_total_ref - e_loc0 + e_loc


def zeros_theta(n):
    return torch.zeros(n, dtype=RDTYPE)
