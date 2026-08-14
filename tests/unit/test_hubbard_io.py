"""Input surface for DFT+U (the `hubbard` block) and its end-to-end wiring.

The schema/validation tests are parse-only (fast tier). The oracle that a
`hubbard` block with every U set to zero reproduces the plain-PBE path
BIT-FOR-BIT runs a small SCF and stays in the fast tier; the U>0 end-to-end
run (SCF + forces + stress through the input surface and the calculator) is
standard tier.

DFT+U forces the full spatial Brillouin zone (an IBZ-folded mesh under-counts
the correlated occupation matrix), so the U=0 oracle compares against a plain
run with `symmetry: false` — the same k treatment, with an inert +U term.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import PSEUDOS


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def _base(extra: str = "") -> str:
    # diamond C with a PseudoDojo pseudo (carries PP_PSWFC for the p manifold),
    # small enough that a full SCF is a fast-tier cost.
    return f"""
structure:
  cell: [[0.0, 1.7835, 1.7835], [1.7835, 0.0, 1.7835], [1.7835, 1.7835, 0.0]]
  positions: {{frac: [[0, 0, 0], [0.25, 0.25, 0.25]]}}
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: PD_C_PBE_std.upf}}
ecut: 340.0
kpoints:
  mesh: [2, 2, 2]
scf:
  etol: 1.0e-9
  rhotol: 1.0e-8
{extra}"""


# ---------------------------------------------------------------- schema (fast)

def test_hubbard_block_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base(
        "hubbard:\n  - {species: C, l: 1, u: 4.5, j: 0.5}\n")))
    assert inp.hubbard.enabled
    assert len(inp.hubbard.manifolds) == 1
    m = inp.hubbard.manifolds[0]
    assert (m.species, m.l, m.u, m.j) == ("C", 1, 4.5, 0.5)


def test_no_hubbard_block_is_disabled(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base()))
    assert not inp.hubbard.enabled
    assert inp.hubbard.manifolds == ()


def test_hubbard_j_defaults_to_zero(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base("hubbard:\n  - {species: C, l: 1, u: 3.0}\n")))
    assert inp.hubbard.manifolds[0].j == 0.0


def test_noncollinear_hubbard_parses(tmp_path):
    """DFT+U on the noncollinear (spinor) path — the collinear-only gate was
    removed once the 2×2 spin-block occupation matrix was wired
    (core.hubbard.occupation_matrices_noncollinear); this is the schema-level
    ungate check (the SCF-level oracle is below)."""
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base(
        "noncollinear: true\nnonmagnetic: true\n"
        "hubbard:\n  - {species: C, l: 1, u: 4.0}\n")))
    assert inp.hubbard.enabled and inp.noncollinear


@pytest.mark.parametrize("extra, needle", [
    ("hubbard:\n  - {species: Ni, l: 2, u: 4.0}\n", "not in the structure"),
    ("hubbard:\n  - {species: C, l: 5, u: 4.0}\n", "hubbard.l must be"),
    ("hubbard:\n  - {species: C, l: 1, u: 1.0}\n  - {species: C, l: 2, u: 2.0}\n",
     "appears twice"),
    ("hubbard:\n  - {speces: C, l: 1, u: 4.0}\n", "unknown key"),
    ("hubbard:\n  - {species: C, l: 1}\n", "needs species, l and u"),
    ("hubbard: {species: C, l: 1, u: 4.0}\n", "must be a list"),
    ("xc: pbe0\nhubbard:\n  - {species: C, l: 1, u: 4.0}\n", "cannot be combined"),
])
def test_hubbard_validation_errors(tmp_path, extra, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(extra)))


# ------------------------------------------------ U=0 bit-for-bit oracle (fast)

def test_u_zero_reproduces_plain_pbe_bit_for_bit(tmp_path):
    """A `hubbard` block with U=0 must be numerically inert. +U runs the full
    spatial BZ, so the reference is a plain `symmetry: false` PBE run: same k
    treatment, identical result to the last bit."""
    import torch

    from gradwave import api
    from gradwave.inputs import load_input

    plain = api.run_scf(load_input(_write(tmp_path, _base("symmetry: false\n"))),
                        verbose=False)
    (tmp_path / "in.yaml").unlink()
    u0 = api.run_scf(load_input(_write(
        tmp_path, _base("hubbard:\n  - {species: C, l: 1, u: 0.0}\n"))),
        verbose=False)

    assert float(u0.energies.hubbard) == 0.0
    assert float(plain.energies.total) == float(u0.energies.total)
    assert float(plain.energies.free_energy) == float(u0.energies.free_energy)
    assert torch.equal(plain.eigenvalues, u0.eigenvalues)


def test_noncollinear_u_zero_reproduces_plain_bit_for_bit(tmp_path):
    """The noncollinear (spinor) +U wiring end to end through the input
    surface: at U=0 the D-matrix is exactly zero at every iteration (uj=0
    multiplies out the occupation matrix identically), so the whole SpinorH
    +U term adds an exact zero and the trajectory is bit-for-bit identical to
    the plain noncollinear SCF — mirrors the collinear oracle above.

    Like the collinear oracle, the reference run needs `symmetry: false`
    explicitly: for a nonmagnetic (m≡0) noncollinear run `symmetry` otherwise
    DEFAULTS to true (Kramers keeps the full crystal symmetry), but +U forces
    it off regardless of the input (build_system's `not hubbard` term) so the
    occupation matrix isn't built from a star-averaged IBZ mesh. Comparing a
    3-k-point IBZ-reduced `plain` against an 8-k-point full-mesh `u0` (an
    earlier version of this test did exactly that) is not a U=0 oracle at
    all — it is two different k-point treatments of the same crystal, which
    do agree physically but not to the last bit (~1e-8 eV, a different
    floating-point summation order, not a wiring bug)."""
    import torch

    from gradwave import api
    from gradwave.inputs import load_input

    nc = "noncollinear: true\nnonmagnetic: true\nsymmetry: false\n"
    plain = api.run_scf(load_input(_write(tmp_path, _base(nc))), verbose=False)
    (tmp_path / "in.yaml").unlink()
    u0 = api.run_scf(load_input(_write(
        tmp_path, _base(nc + "hubbard:\n  - {species: C, l: 1, u: 0.0}\n"))),
        verbose=False)

    assert float(u0.energies.hubbard) == 0.0
    assert float(plain.energies.total) == float(u0.energies.total)
    assert float(plain.energies.free_energy) == float(u0.energies.free_energy)
    assert torch.equal(plain.eigenvalues, u0.eigenvalues)


# --------------------------------------------- U>0 end-to-end (standard tier)

@pytest.mark.standard
def test_u_positive_end_to_end_scf(tmp_path):
    """U>0 through the input surface: SCF converges and the Dudarev energy is
    a finite, positive on-site penalty."""
    from gradwave import api
    from gradwave.inputs import load_input

    res = api.run_scf(load_input(_write(
        tmp_path, _base("hubbard:\n  - {species: C, l: 1, u: 4.0}\n"))),
        verbose=False)
    assert res.converged
    e_u = float(res.energies.hubbard)
    assert np.isfinite(e_u) and e_u > 0.0
    # per-spin, per-site occupation matrices are attached to the result
    assert res.hub_occ is not None and len(res.hub_occ[0]) == 2  # two C sites


@pytest.mark.standard
def test_u_positive_forces_and_stress_via_calculator(tmp_path):
    """The +U forces and stress come out of the ASE calculator (the relax
    path): finite, and the +U term perturbs the plain forces."""
    from ase import Atoms

    from gradwave.calculator import GradWave

    a = 3.567
    cell = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
    # displace one atom so forces are nonzero
    atoms = Atoms("C2", scaled_positions=[[0, 0, 0], [0.26, 0.25, 0.25]],
                  cell=cell, pbc=True)
    kw = dict(ecut=340.0,
              pseudopotentials={"C": str(PSEUDOS / "PD_C_PBE_std.upf")},
              kpts=(2, 2, 2), smearing="none", verbose=False)
    calc_u = GradWave(hubbard=[{"species": "C", "l": 1, "u": 4.0}], **kw)
    atoms.calc = calc_u
    f_u = atoms.get_forces()
    s_u = atoms.get_stress()
    assert np.isfinite(f_u).all() and np.isfinite(s_u).all()

    plain = atoms.copy()
    plain.calc = GradWave(use_symmetry=False, **kw)
    f0 = plain.get_forces()
    # the +U contribution is real but small relative to the KB/local force
    assert 1e-4 < np.abs(f_u - f0).max() < 1.0


@pytest.mark.standard
def test_uspp_paw_hubbard_stress_through_calculator(tmp_path):
    """USPP/PAW +U through the calculator: no longer gated (gradwave#156 added
    the +U stress term), so energy/forces/stress all run to completion. The
    physics (autograd vs. FD, U=0 inert) is self-oracled directly against
    ``postscf.paw_stress`` in ``test_paw_stress_hubbard.py``; this is the
    calculator-wiring regression guard for the ``_calculate_uspp`` gate that
    used to reject this combination outright."""
    from ase import Atoms

    from gradwave.calculator import GradWave

    a = 5.43
    cell = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
    atoms = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                  cell=cell, pbc=True)
    atoms.calc = GradWave(
        # only asserts isfinite (physics oracled in test_paw_stress_hubbard.py),
        # so run at a wiring-guard cutoff, not the 340 eV accuracy setting
        ecut=200.0,
        pseudopotentials={"Si": str(PSEUDOS / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")},
        kpts=(1, 1, 1), smearing="none",
        hubbard=[{"species": "Si", "l": 1, "u": 2.0}], verbose=False)
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()
    assert np.isfinite(e)
    assert np.isfinite(f).all()
    assert np.isfinite(s).all() and s.shape == (6,)
