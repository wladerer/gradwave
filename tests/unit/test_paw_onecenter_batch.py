"""Batched PAW one-center vs the per-atom reference path (task: make the
one-center correction batched over atoms and device-aware).

energy_and_ddd_batch must reproduce the per-atom energy_and_ddd loop — the
path the SCF iterated before batching — to ≤1e-10 per element (energies AND
ddd matrices), for a multi-atom species plus a second species, nspin=1 and
spin. The SCF assembly (uspp_potentials_dscr) must place the batched ddd
blocks exactly where the per-atom loop placed them.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.paw_onsite import OneCenter, onecenters

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
TOL = 1e-10


def _rho(paw, seed, scale=0.03):
    """Physically-shaped random becsum: atomic occupations + symmetric noise."""
    nm = sum(2 * b.l + 1 for b in paw.betas)
    m0 = torch.zeros(nm, nm, dtype=torch.float64)
    col = 0
    for i, b in enumerate(paw.betas):
        for _m in range(2 * b.l + 1):
            m0[col, col] = paw.paw_occ[i] / (2 * b.l + 1)
            col += 1
    gen = torch.Generator().manual_seed(seed)
    p = scale * torch.randn(nm, nm, generator=gen, dtype=torch.float64)
    return m0 + (p + p.T) / 2


@pytest.mark.parametrize("upf_name, n_atoms", [
    ("Si.pbe-n-kjpaw_psl.1.0.0.UPF", 3),   # multi-atom species
    ("C.pbe-n-kjpaw_psl.1.0.0.UPF", 1),    # second species, single atom
])
def test_batched_matches_per_atom(upf_name, n_atoms):
    paw = parse_upf_paw(FIX / "pseudos" / upf_name)
    oc = OneCenter(paw, PBE())
    becs = [_rho(paw, 100 + 7 * a) for a in range(n_atoms)]

    e_ref, ddd_ref = zip(*[oc.energy_and_ddd(b) for b in becs], strict=True)
    e_bat, ddd_bat = oc.energy_and_ddd_batch(torch.stack(becs))

    assert e_bat.shape == (n_atoms,)
    for a in range(n_atoms):
        assert abs(float(e_bat[a]) - e_ref[a]) <= TOL
        assert float((ddd_bat[a] - ddd_ref[a]).abs().max()) <= TOL


def test_batched_matches_per_atom_spin():
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    oc = OneCenter(paw, SpinPBE())
    n_atoms = 2
    ups = [0.55 * _rho(paw, 200 + a) for a in range(n_atoms)]
    dns = [0.45 * _rho(paw, 300 + a) for a in range(n_atoms)]

    refs = [oc.energy_and_ddd([ups[a], dns[a]]) for a in range(n_atoms)]
    e_bat, ddd_bat = oc.energy_and_ddd_batch(
        [torch.stack(ups), torch.stack(dns)])

    for a in range(n_atoms):
        e_ref, ddd_ref = refs[a]
        assert abs(float(e_bat[a]) - e_ref) <= TOL
        for isp in range(2):
            assert float((ddd_bat[isp][a] - ddd_ref[isp]).abs().max()) <= TOL


def test_scf_assembly_places_batched_blocks_like_per_atom_loop():
    """uspp_potentials_dscr (batched one-center) vs the per-atom reference
    assembly on a 3-atom, two-species PAW cell: e_onec and every dscr block
    agree to ≤1e-10."""
    from gradwave.core.energies.local_pp import local_potential_g
    from gradwave.core.fftbox import g_to_r_box
    from gradwave.scf.uspp import setup_uspp
    from gradwave.scf.uspp_loop import uspp_potentials_dscr
    from tests.helpers import RY

    xc = PBE()
    a = 5.43
    lattice = a * np.eye(3)
    pos = np.array([[0.0, 0.0, 0.0], [2.1, 2.0, 1.9], [1.0, 3.2, 2.7]])
    paws = [parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF"),
            parse_upf_paw(FIX / "pseudos" / "C.pbe-n-kjpaw_psl.1.0.0.UPF")]
    system = setup_uspp(lattice, pos, [0, 0, 1], paws, ecut=12 * RY,
                        kmesh=(1, 1, 1))
    grid, vol = system.grid, system.grid.volume
    dev = system.positions.device

    gen = torch.Generator().manual_seed(11)
    rho = torch.rand(*grid.shape, generator=gen, dtype=torch.float64)
    rho = (rho / rho.sum() * system.n_electrons * grid.n_points / vol).to(dev)
    rho_ij = [[_rho(paws[sp], 400 + 13 * at).to(dev)
               for at, sp in enumerate(system.species_of_atom)]]
    vloc_g = local_potential_g(
        system.positions, torch.tensor(system.species_of_atom, device=dev),
        system.vloc_tables, grid.g_cart, vol)
    vloc_r = g_to_r_box(vloc_g, real=True)
    phase_arg = system.g_sphere @ system.positions.T
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))

    onec = onecenters(system, xc, device=dev)
    veff_s, dscr_s, e_onec = uspp_potentials_dscr(
        system, xc, [rho], rho_ij, vloc_r, phases, onec)
    # reference: the pre-batching per-atom assembly
    _, dscr_ref, _ = uspp_potentials_dscr(
        system, xc, [rho], rho_ij, vloc_r, phases, None)
    dscr_ref = [d.clone() for d in dscr_ref]
    e_ref = 0.0
    for at, sp in enumerate(system.species_of_atom):
        s0, s1 = system.atom_slices[at]
        e1c, ddd = onec[sp].energy_and_ddd(rho_ij[0][at])
        e_ref += e1c
        dscr_ref[0][s0:s1, s0:s1] += ddd.to(dev)

    assert abs(float(e_onec) - e_ref) <= TOL
    assert float((dscr_s[0] - dscr_ref[0]).abs().max()) <= TOL


@pytest.mark.gpu
def test_batched_gpu_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    becs = torch.stack([_rho(paw, 500 + a) for a in range(2)])
    e_cpu, ddd_cpu = OneCenter(paw, PBE()).energy_and_ddd_batch(becs)
    oc_gpu = OneCenter(paw, PBE(), device="cuda")
    e_gpu, ddd_gpu = oc_gpu.energy_and_ddd_batch(becs.cuda())
    assert float((e_gpu.cpu() - e_cpu).abs().max()) <= 1e-9
    assert float((ddd_gpu.cpu() - ddd_cpu).abs().max()) <= 1e-9
