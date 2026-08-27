"""D3(BJ) dispersion: self-oracle verification.

Tier-0 gate for the dispersion correction (docs/verification.md): forces and
stress against finite differences of the dispersion energy itself, plus an
independent scalar transcription of the reference D3(BJ) expression and a
gradcheck. No external code in the oracle — the FD is of our own energy, and
the scalar reference is a separate (loop-based) code path over the same
vendored reference tables, so a vectorization or autograd-graph bug shows up.

Systems are deliberately low-symmetry, rattled, mixed-element cells (the
verification.md rule: symmetric fixtures let error terms cancel).
"""

import math

import numpy as np
import pytest
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.postscf._d3_params import C6AB, K1, K3, MXC, R2R4, RCOV
from gradwave.postscf.dispersion import (
    D3Config,
    _image_labels,
    dispersion_energy,
    dispersion_forces,
    dispersion_stress,
)


def _rattled_hcno():
    """A rattled, low-symmetry triclinic cell of H, C, N, O (Å)."""
    rng = np.random.default_rng(20260724)
    cell = np.array([[5.9, 0.4, -0.3], [0.2, 6.3, 0.5], [-0.4, 0.3, 6.1]])
    frac = np.array([[0.05, 0.10, 0.08], [0.52, 0.48, 0.13],
                     [0.20, 0.71, 0.55], [0.83, 0.27, 0.79]])
    frac = frac + 0.02 * rng.standard_normal(frac.shape)
    pos = frac @ cell
    Z = [6, 7, 8, 1]  # C, N, O, H
    return torch.tensor(pos), cell, Z


# --------------------------------------------------------------------------
# independent scalar reference (loop-based transcription of dftd3 edisp, BJ)
# --------------------------------------------------------------------------

def _scalar_c6(za, zb, cna, cnb):
    rsum = csum = 0.0
    for _i, _j, c6, cn1, cn2 in C6AB[(za, zb)]:
        w = math.exp(K3 * ((cn1 - cna) ** 2 + (cn2 - cnb) ** 2))
        rsum += w
        csum += w * c6
    return csum / rsum


def _scalar_d3bj(pos_ang, cell_ang, Z, cfg: D3Config):
    """E_disp [eV] by explicit triple loops over atoms and lattice images."""
    pos = pos_ang.detach().cpu().numpy() / BOHR_ANG
    na = len(Z)
    if cell_ang is None:
        cell = None
        cn_lab = e_lab = np.zeros((1, 3))
    else:
        cell = np.asarray(cell_ang) / BOHR_ANG
        cn_lab = _image_labels(cell_ang, cfg.cn_cutoff)
        e_lab = _image_labels(cell_ang, cfg.cutoff)

    def imgs(lab):
        return lab @ cell if cell is not None else np.zeros((1, 3))

    # coordination numbers
    cn = np.zeros(na)
    for a in range(na):
        for b in range(na):
            for L in imgs(cn_lab):
                if a == b and np.linalg.norm(L) < 1e-12:
                    continue
                r = np.linalg.norm(pos[a] - pos[b] + L)
                if r > cfg.cn_cutoff:
                    continue
                rco = RCOV[Z[a] - 1] + RCOV[Z[b] - 1]
                cn[a] += 1.0 / (1.0 + math.exp(-K1 * (rco / r - 1.0)))

    e = 0.0
    for a in range(na):
        for b in range(na):
            c6 = _scalar_c6(Z[a], Z[b], cn[a], cn[b])
            c8 = 3.0 * c6 * R2R4[Z[a] - 1] * R2R4[Z[b] - 1]
            f = cfg.a1 * math.sqrt(c8 / c6) + cfg.a2
            for L in imgs(e_lab):
                if a == b and np.linalg.norm(L) < 1e-12:
                    continue
                r = np.linalg.norm(pos[a] - pos[b] + L)
                if r > cfg.cutoff:
                    continue
                r6 = r ** 6
                r8 = r ** 8
                e += -0.5 * (cfg.s6 * c6 / (r6 + f ** 6) + cfg.s8 * c8 / (r8 + f ** 8))
    return e * HARTREE_EV


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_matches_scalar_reference_periodic():
    pos, cell, Z = _rattled_hcno()
    cfg = D3Config.from_functional("pbe", cutoff_ang=14.0, cn_cutoff_ang=11.0)
    e = dispersion_energy(pos, torch.tensor(cell), Z, cfg).item()
    ref = _scalar_d3bj(pos, cell, Z, cfg)
    assert abs(e - ref) <= 1e-10 * abs(ref)


def test_matches_scalar_reference_molecule():
    # water molecule, no cell (CN interpolation active on O and H)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
                       dtype=torch.float64)
    Z = [8, 1, 1]
    cfg = D3Config.from_functional("b3lyp")
    e = dispersion_energy(pos, None, Z, cfg).item()
    ref = _scalar_d3bj(pos, None, Z, cfg)
    assert abs(e - ref) <= 1e-11 * abs(ref)


def test_forces_match_finite_differences():
    pos, cell, Z = _rattled_hcno()
    cfg = D3Config.from_functional("pbe", cutoff_ang=13.0, cn_cutoff_ang=10.0)
    cell_t = torch.tensor(cell)
    f = dispersion_forces(pos, cell, Z, cfg)  # (na,3) eV/Å, = -dE/dτ

    h = 1e-5
    base = pos.clone()
    scale = f.abs().max().item()
    for a in range(len(Z)):
        for comp in range(3):
            dp = torch.zeros_like(base)
            dp[a, comp] = h
            ep = dispersion_energy(base + dp, cell_t, Z, cfg).item()
            em = dispersion_energy(base - dp, cell_t, Z, cfg).item()
            fd = -(ep - em) / (2 * h)
            assert abs(fd - f[a, comp].item()) <= 1e-6 * scale + 1e-10


def test_stress_matches_finite_differences():
    pos, cell, Z = _rattled_hcno()
    cfg = D3Config.from_functional("pbe", cutoff_ang=13.0, cn_cutoff_ang=10.0)
    cell0 = np.asarray(cell)
    omega0 = abs(np.linalg.det(cell0))
    pos0 = pos.detach()
    a0 = torch.tensor(cell0)
    sigma = dispersion_stress(pos, cell, Z, cfg)  # (3,3) eV/Å³

    def energy_at(eps_np):
        f_map = torch.eye(3, dtype=torch.float64) + torch.tensor(eps_np)
        pos_e = pos0 @ f_map.T
        cell_e = a0 @ f_map.T
        return dispersion_energy(pos_e, cell_e, Z, cfg, ref_cell=cell0).item()

    h = 1e-6
    grad_fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ep = np.zeros((3, 3))
            em = np.zeros((3, 3))
            ep[i, j] = h
            em[i, j] = -h
            grad_fd[i, j] = (energy_at(ep) - energy_at(em)) / (2 * h)
    sigma_fd = 0.5 * (grad_fd + grad_fd.T) / omega0
    scale = np.abs(sigma.detach().numpy()).max()
    assert np.allclose(sigma.detach().numpy(), sigma_fd, atol=1e-6 * scale + 1e-10)


def test_translation_invariance_and_force_sum_rule():
    pos, cell, Z = _rattled_hcno()
    cfg = D3Config.from_functional("pbe", cutoff_ang=13.0, cn_cutoff_ang=10.0)
    cell_t = torch.tensor(cell)
    shift = torch.tensor([0.41, -0.23, 0.62], dtype=torch.float64)
    e1 = dispersion_energy(pos, cell_t, Z, cfg).item()
    e2 = dispersion_energy(pos + shift, cell_t, Z, cfg).item()
    assert abs(e1 - e2) <= 1e-10 * abs(e1)

    f = dispersion_forces(pos, cell, Z, cfg)
    assert f.sum(dim=0).abs().max().item() < 1e-9  # Σ_a F_a = 0


def test_energy_gradcheck_positions():
    # small molecule so double-backward gradcheck is cheap; guards the r→0
    # self-pair shift and the CN/interpolation graph
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [0.3, 1.3, 0.4]],
                       dtype=torch.float64, requires_grad=True)
    Z = [6, 8, 1]
    cfg = D3Config.from_functional("pbe")
    assert torch.autograd.gradcheck(
        lambda p: dispersion_energy(p, None, Z, cfg), (pos,), eps=1e-6, atol=1e-6
    )


def test_stress_symmetry_cubic():
    # a cubic cell must give an isotropic (diagonal, equal) stress
    a = 3.9
    cell = np.eye(3) * a
    pos = torch.tensor([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]], dtype=torch.float64)
    Z = [6, 6]
    cfg = D3Config.from_functional("pbe", cutoff_ang=12.0)
    s = dispersion_stress(pos, cell, Z, cfg).detach().numpy()
    diag = np.diag(s)
    assert np.allclose(diag, diag.mean(), rtol=1e-8)
    assert abs(s[0, 1] - s[0, 2]) < 1e-8 * abs(diag.mean())


def test_unknown_element_raises():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    cfg = D3Config.from_functional("pbe")
    with pytest.raises(NotImplementedError, match="not vendored"):
        dispersion_energy(pos, None, [1, 37], cfg)  # Rb (Z=37) beyond the H..Kr subset


def test_covered_range_is_h_through_kr():
    # the widened vendored set is exactly Z=1..36 (periods 1-4 main-group + 3d)
    assert sorted(MXC) == list(range(1, 37))


def test_all_vendored_elements_self_pair_present():
    # every covered element must have its homonuclear reference block
    for z in MXC:
        assert (z, z) in C6AB


# The one external anchor (Tier-3): the simple-dftd3 CLI tutorial's water–peptide
# dimer, PBE0-D3(BJ), two-body only. Independent code, exact published geometry
# and energy — https://dftd3.readthedocs.io/en/latest/tutorial/first-steps-cli.html
_DIMER_SYMBOLS = ["O", "H", "H", "C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "H", "H"]
_DIMER_XYZ = [
    [-3.2939688, 0.4402024, 0.1621802], [-3.8134112, 1.2387332, 0.2637577],
    [-2.3770466, 0.7564365, 0.1766203], [-0.6611637, -1.4159110, -0.1449409],
    [-0.0112009, -2.2770229, -0.2778563], [-1.3421397, -1.3384389, -0.9888061],
    [-1.2741806, -1.5547070, 0.7420675], [0.0935684, -0.1178981, -0.0123474],
    [-0.4831471, 0.9573968, 0.1442414], [1.4442015, -0.2154008, -0.0769653],
    [1.8451531, -1.1259348, -0.2064804], [2.3124436, 0.9365697, 0.0324778],
    [1.6759495, 1.8048701, 0.1672624], [2.9780331, 0.8451145, 0.8885706],
    [2.9069093, 1.0659902, -0.8697814],
]
_DIMER_EDISP_HARTREE = -8.2944752821052e-3  # s-dftd3 --bj PBE0, two-body (s9=0)


def test_external_reference_simple_dftd3():
    from gradwave.constants import HARTREE_EV as _H

    z = {"H": 1, "C": 6, "N": 7, "O": 8}
    pos = torch.tensor(_DIMER_XYZ, dtype=torch.float64)
    Z = [z[s] for s in _DIMER_SYMBOLS]
    cfg = D3Config.from_functional("pbe0")
    e_hartree = dispersion_energy(pos, None, Z, cfg).item() / _H
    assert abs(e_hartree - _DIMER_EDISP_HARTREE) < 1e-7  # machine-level agreement


# --------------------------------------------------------------------------
# widened element set (Z=1..36): new-element molecules vs simple-dftd3.
# References are total two-body D3(BJ) energies (Hartree) from simple-dftd3's
# RationalDampingParam(method, atm=False) — an independent C implementation over
# the same Grimme reference data. Geometry is Å converted to Bohr with BOHR_ANG,
# so agreement is at the Bohr-conversion floor (~1e-11 Ha, well under 1e-9).
# Systems deliberately use elements outside the original 11-element subset:
# Al, Si (period 3) and K, Ca, Fe (period 4, incl. the 3d metal Fe).
# --------------------------------------------------------------------------
_ALSIO_Z = [13, 14, 8, 8, 8, 1]  # Al Si O O O H
_ALSIO_XYZ = [
    [0.0102, 0.0231, -0.0157], [1.7321, 0.0412, 0.0233],
    [-0.9124, 1.2842, 0.3319], [2.6013, 1.2201, -0.4471],
    [0.8442, -1.3122, 0.7215], [-1.8331, 1.0113, 0.5528],
]
_KCAFE_Z = [19, 20, 26, 8, 8]  # K Ca Fe O O
_KCAFE_XYZ = [
    [0.0311, -0.0212, 0.0431], [2.9124, 0.1201, -0.1123],
    [1.4432, 2.3841, 0.2214], [1.5013, 0.0113, 1.4528],
    [1.4221, -1.9902, -0.3315],
]
# (system, functional) -> total two-body D3(BJ) energy [Hartree], s-dftd3 atm=False
_NEW_ELEMENT_D3 = {
    ("alsio", "pbe"): -0.004777489306177081,
    ("alsio", "pbe0"): -0.004406848542979641,
    ("kcafe", "pbe"): -0.004904060769491955,
    ("kcafe", "pbe0"): -0.004962852390056221,
}


@pytest.mark.parametrize("system,func", list(_NEW_ELEMENT_D3))
def test_external_reference_new_elements(system, func):
    Z, xyz = {"alsio": (_ALSIO_Z, _ALSIO_XYZ),
              "kcafe": (_KCAFE_Z, _KCAFE_XYZ)}[system]
    pos = torch.tensor(xyz, dtype=torch.float64)
    cfg = D3Config.from_functional(func)
    e_hartree = dispersion_energy(pos, None, Z, cfg).item() / HARTREE_EV
    assert abs(e_hartree - _NEW_ELEMENT_D3[(system, func)]) < 1e-9


def test_new_element_forces_match_finite_differences():
    # autograd force vs central FD on a 3d-metal-bearing molecule (K, Ca, Fe)
    Z = _KCAFE_Z
    pos = torch.tensor(_KCAFE_XYZ, dtype=torch.float64)
    cfg = D3Config.from_functional("pbe0")
    f = dispersion_forces(pos, None, Z, cfg)  # (na,3) eV/Å = -dE/dτ
    h = 1e-5
    scale = f.abs().max().item()
    for a in range(len(Z)):
        for comp in range(3):
            dp = torch.zeros_like(pos)
            dp[a, comp] = h
            ep = dispersion_energy(pos + dp, None, Z, cfg).item()
            em = dispersion_energy(pos - dp, None, Z, cfg).item()
            fd = -(ep - em) / (2 * h)
            assert abs(fd - f[a, comp].item()) <= 1e-6 * scale + 1e-10
    assert f.sum(dim=0).abs().max().item() < 1e-9  # Σ_a F_a = 0
