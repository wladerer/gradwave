"""D4(BJ) dispersion: self-oracle + external-reference verification.

Tier-0 gate for the D4 correction (docs/verification.md): forces against finite
differences of the D4 energy itself, and a gradcheck of the whole EEQ→ζ→C6→BJ
graph. Tier-3 external anchor: total two-body energies *and* EEQ partial charges
from the reference implementation dftd4 (via tad-dftd4 0.8.0), transcribed here
as fixed numbers — independent code, so a formula or table-transcription bug
shows up. Reference energies use ``s9=0`` (two-body only, matching this module's
scope). Positions are in Å; the reference was produced in Bohr, so agreement is
at the ~1e-11 Hartree level (the Bohr↔Å constant differs in its last digits).

Systems are deliberately low-symmetry and rattled (the verification.md rule).
``cluster11`` carries one atom of every vendored element, exercising each
reference block and every C6REF pair.
"""

import numpy as np
import pytest
import torch

from gradwave.constants import HARTREE_EV
from gradwave.postscf._d4_params import BJ_PARAMS, C6REF, NREF
from gradwave.postscf.dispersion_d4 import (
    D4Config,
    dispersion_energy,
    dispersion_forces,
    dispersion_stress,
    eeq_charges,
)

# --------------------------------------------------------------------------
# fixtures (Å); reference numbers from tad-dftd4 0.8.0, two-body (s9=0)
# --------------------------------------------------------------------------

_WATER_Z = [8, 1, 1]
_WATER_XYZ = [[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
              [0.0, -0.763239, -0.477047]]
_WATER_E = -1.576174374352937e-4  # pbe0
_WATER_Q = [-0.5863907989, 0.2931953994, 0.2931953994]

_RATTLED_Z = [6, 7, 8, 1, 9, 16]  # C N O H F S
_RATTLED_XYZ = [
    [0.0001492810909738512, -0.048029417440517676, -0.06259935236118629],
    [1.2729109464204844, 0.1248947623561021, 0.023429360173618553],
    [2.103481912360382, 1.1849005751233879, -0.12369443976134006],
    [-0.8777788127712205, 0.6955770114285955, 0.24133433259778658],
    [0.6406358365237567, -1.1942181344486613, 0.7032187484847978],
    [3.264723731403542, -0.4079075360981219, 0.6356289602559031],
]
_RATTLED_E = -0.0037772102634709686  # pbe0

# water–peptide dimer (same geometry as the D3 external-reference test)
_DIMER_SYMBOLS = "O H H C H H H C O N H C H H H".split()
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
_DIMER_E = -0.007591533425342104  # pbe0

_OH_Z = [8, 1]
_OH_XYZ = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]
_OH_E = -0.00017066628134750867  # pbe, charge -1
_OH_Q = [-1.1351115137, 0.1351115137]

# one atom of every vendored element (H C N O F Mg P S Cl Cu Zn), rattled
_CLUSTER_Z = [1, 6, 7, 8, 9, 12, 15, 16, 17, 29, 30]
_CLUSTER_XYZ = [
    [-0.0705481833, -0.2028551816, 0.1495700839],
    [0.0764630801, -0.014549416, 2.7318474821],
    [-0.1498795837, 0.0340532247, 5.2352482751],
    [-0.0828683508, 2.8915797376, 0.2321849111],
    [-0.1169936022, 2.7087239548, 2.785779228],
    [-0.0099992654, 2.717771625, 5.6893266586],
    [-0.0737086415, 5.3954295057, 0.2549576993],
    [-0.0413099366, 5.3388241985, 2.694647875],
    [0.1459341796, 5.7321059977, 5.4863610116],
    [2.8247156512, 0.1354668002, 0.2106991163],
    [2.4797595341, -0.0861717048, 2.5869887182],
]
_CLUSTER_E = -0.0146177524269614  # pbe0


def _tensor(xyz):
    return torch.tensor(xyz, dtype=torch.float64)


# --------------------------------------------------------------------------
# external reference: total two-body D4 energy vs dftd4
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "Z, xyz, func, charge, e_ref",
    [
        (_WATER_Z, _WATER_XYZ, "pbe0", 0.0, _WATER_E),
        (_RATTLED_Z, _RATTLED_XYZ, "pbe0", 0.0, _RATTLED_E),
        ([{"H": 1, "C": 6, "N": 7, "O": 8}[s] for s in _DIMER_SYMBOLS],
         _DIMER_XYZ, "pbe0", 0.0, _DIMER_E),
        (_OH_Z, _OH_XYZ, "pbe", -1.0, _OH_E),
        (_CLUSTER_Z, _CLUSTER_XYZ, "pbe0", 0.0, _CLUSTER_E),
    ],
)
def test_external_reference_energy(Z, xyz, func, charge, e_ref):
    cfg = D4Config.from_functional(func, charge=charge)
    e_hartree = dispersion_energy(_tensor(xyz), None, Z, cfg).item() / HARTREE_EV
    assert abs(e_hartree - e_ref) < 1e-9


# --------------------------------------------------------------------------
# external reference: EEQ partial charges vs the reference charge model
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "Z, xyz, charge, q_ref",
    [
        (_WATER_Z, _WATER_XYZ, 0.0, _WATER_Q),
        (_OH_Z, _OH_XYZ, -1.0, _OH_Q),
    ],
)
def test_eeq_charges_reference(Z, xyz, charge, q_ref):
    q = eeq_charges(_tensor(xyz), None, Z, charge=charge)
    assert np.allclose(q.numpy(), np.asarray(q_ref), atol=1e-8)


def test_eeq_charge_constraint():
    # the Lagrange constraint must reproduce the requested total charge exactly
    for charge in (0.0, -1.0, 1.0):
        q = eeq_charges(_tensor(_RATTLED_XYZ), None, _RATTLED_Z, charge=charge)
        assert abs(q.sum().item() - charge) < 1e-10


# --------------------------------------------------------------------------
# self-oracle: forces vs central finite differences of our own energy
# --------------------------------------------------------------------------

def test_forces_match_finite_differences():
    Z, cfg = _RATTLED_Z, D4Config.from_functional("pbe0")
    base = _tensor(_RATTLED_XYZ)
    f = dispersion_forces(base, None, Z, cfg)  # (na,3) eV/Å = -dE/dτ

    h = 1e-5
    scale = f.abs().max().item()
    for a in range(len(Z)):
        for comp in range(3):
            dp = torch.zeros_like(base)
            dp[a, comp] = h
            ep = dispersion_energy(base + dp, None, Z, cfg).item()
            em = dispersion_energy(base - dp, None, Z, cfg).item()
            fd = -(ep - em) / (2 * h)
            assert abs(fd - f[a, comp].item()) <= 1e-6 * scale + 1e-9


def test_energy_gradcheck_positions():
    # small molecule so double-backward gradcheck is cheap; exercises the full
    # EEQ solve → ζ scaling → C6 interpolation → BJ graph
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [0.3, 1.3, 0.4]],
                       dtype=torch.float64, requires_grad=True)
    Z = [6, 8, 1]
    cfg = D4Config.from_functional("pbe")
    assert torch.autograd.gradcheck(
        lambda p: dispersion_energy(p, None, Z, cfg), (pos,), eps=1e-6, atol=1e-6
    )


def test_translation_invariance_and_force_sum_rule():
    Z, cfg = _RATTLED_Z, D4Config.from_functional("pbe0")
    base = _tensor(_RATTLED_XYZ)
    shift = torch.tensor([0.41, -0.23, 0.62], dtype=torch.float64)
    e1 = dispersion_energy(base, None, Z, cfg).item()
    e2 = dispersion_energy(base + shift, None, Z, cfg).item()
    assert abs(e1 - e2) <= 1e-10 * abs(e1)

    f = dispersion_forces(base, None, Z, cfg)
    assert f.sum(dim=0).abs().max().item() < 1e-9  # Σ_a F_a = 0


# --------------------------------------------------------------------------
# coverage / guards
# --------------------------------------------------------------------------

def test_all_vendored_elements_have_reference_blocks():
    for z in NREF:
        assert (z, z) in C6REF
        assert len(C6REF[(z, z)]) == NREF[z]


def test_unknown_element_raises():
    cfg = D4Config.from_functional("pbe")
    with pytest.raises(NotImplementedError, match="not vendored"):
        dispersion_energy(_tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]), None,
                          [1, 3], cfg)  # Li (Z=3) not in subset


# --------------------------------------------------------------------------
# periodic self-oracle: Ewald-EEQ charges, forces vs FD, stress vs FD,
# and the large-vacuum reduction to the molecular result
# --------------------------------------------------------------------------

def _rattled_mgo():
    """Rattled triclinic rock-salt MgO (Z=12/8, both vendored): a bonded, ionic
    periodic cell — CN>0 everywhere and meaningful EEQ charges."""
    rng = np.random.default_rng(3)
    a = 4.2
    cell = np.array([[a, 0.0, 0.0], [0.15, a, 0.0], [-0.1, 0.2, a]])
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],  # Mg
                     [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]])  # O
    Z = [12, 12, 12, 12, 8, 8, 8, 8]
    frac = base + 0.03 * rng.standard_normal(base.shape)
    pos = torch.tensor(frac @ cell, dtype=torch.float64)
    return pos, torch.tensor(cell, dtype=torch.float64), cell, Z


def test_periodic_eeq_charge_neutrality_and_sign():
    pos, cell, _, Z = _rattled_mgo()
    cfg = D4Config.from_functional("pbe0", cn_cutoff_ang=12.0)
    q = eeq_charges(pos, cell, Z, cfg=cfg)
    assert abs(q.sum().item()) < 1e-10  # Lagrange constraint (neutral cell)
    assert (q[:4] > 0).all() and (q[4:] < 0).all()  # Mg cations, O anions


def test_periodic_forces_match_finite_differences():
    pos, cell, _, Z = _rattled_mgo()
    cfg = D4Config.from_functional("pbe0", cutoff_ang=18.0, cn_cutoff_ang=12.0)
    f = dispersion_forces(pos, cell, Z, cfg)
    h = 1e-5
    scale = f.abs().max().item()
    for i in range(len(Z)):
        for c in range(3):
            dp = torch.zeros_like(pos)
            dp[i, c] = h
            ep = dispersion_energy(pos + dp, cell, Z, cfg).item()
            em = dispersion_energy(pos - dp, cell, Z, cfg).item()
            fd = -(ep - em) / (2 * h)
            assert abs(fd - f[i, c].item()) <= 1e-6 * scale + 1e-10
    assert f.sum(dim=0).abs().max().item() < 1e-9  # Σ_a F_a = 0


def test_periodic_stress_matches_finite_differences():
    pos, cell, cell_np, Z = _rattled_mgo()
    cfg = D4Config.from_functional("pbe0", cutoff_ang=18.0, cn_cutoff_ang=12.0)
    sig = dispersion_stress(pos, cell, Z, cfg).detach().numpy()
    omega0 = abs(np.linalg.det(cell_np))
    h = 1e-6

    def e_strain(eps):
        fmap = np.eye(3) + eps
        return dispersion_energy(
            torch.tensor(pos.numpy() @ fmap.T), torch.tensor(cell_np @ fmap.T),
            Z, cfg, ref_cell=cell_np).item()

    fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            e = np.zeros((3, 3))
            e[i, j] = h
            fd[i, j] = (e_strain(e) - e_strain(-e)) / (2 * h) / omega0
    fd = 0.5 * (fd + fd.T)
    scale = np.abs(sig).max()
    assert np.allclose(sig, fd, atol=1e-6 * scale + 1e-10)


def test_periodic_reduces_to_molecular_in_large_box():
    # a molecule in a large vacuum box → periodic EEQ charges and D4 energy must
    # converge to the molecular result (the Ewald sum's isolated limit)
    Z, cfg = _WATER_Z, D4Config.from_functional("pbe0", cn_cutoff_ang=12.0)
    mol = _tensor(_WATER_XYZ)
    q_mol = eeq_charges(mol, None, Z).numpy()
    e_mol = dispersion_energy(mol, None, Z, cfg).item()

    L = 30.0
    cell = torch.eye(3, dtype=torch.float64) * L
    pos = mol - mol.mean(0) + L / 2  # center it in the box
    q_p = eeq_charges(pos, cell, Z, cfg=cfg).numpy()
    e_p = dispersion_energy(pos, cell, Z, cfg).item()
    assert np.abs(q_p - q_mol).max() < 5e-5   # dipole-tail finite-box residual
    assert abs(e_p - e_mol) < 5e-7


def test_periodic_translation_invariance():
    pos, cell, _, Z = _rattled_mgo()
    cfg = D4Config.from_functional("pbe0", cn_cutoff_ang=12.0)
    shift = torch.tensor([0.31, -0.22, 0.47], dtype=torch.float64)
    e1 = dispersion_energy(pos, cell, Z, cfg).item()
    e2 = dispersion_energy(pos + shift, cell, Z, cfg).item()
    assert abs(e1 - e2) <= 1e-9 * abs(e1)


def test_periodic_resolve_matches_from_functional():
    # the calculator/api-facing resolve() must agree with from_functional
    pos, cell, _, Z = _rattled_mgo()
    a = D4Config.resolve("pbe0", cutoff_ang=18.0, cn_cutoff_ang=12.0)
    b = D4Config.from_functional("pbe0", cutoff_ang=18.0, cn_cutoff_ang=12.0)
    ea = dispersion_energy(pos, cell, Z, a).item()
    eb = dispersion_energy(pos, cell, Z, b).item()
    assert ea == eb


def test_unknown_functional_raises():
    with pytest.raises(ValueError, match="no D4"):
        D4Config.from_functional("not-a-functional")


def test_bj_presets_have_expected_functionals():
    for fn in ("pbe", "pbe0", "b3lyp", "tpss", "scan"):
        assert fn in BJ_PARAMS and BJ_PARAMS[fn][0] == 1.0  # s6 == 1
