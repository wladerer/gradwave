"""M2 acceptance: FIRE relaxation of rattled Si recovers the ideal geometry."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.optimize import FIRE

from gradwave.calculator import GradWave
from tests.helpers import RY, pseudo

PSEUDO = pseudo("Si_ONCV_PBE-1.2.upf")


@pytest.mark.slow
def test_fire_relax_rattled_si(tmp_path):
    torch.set_num_threads(8)
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    ideal = np.array([[0.0, 0, 0], [a / 4] * 3])
    rng = np.random.default_rng(42)
    atoms = Atoms("Si2", positions=ideal + rng.normal(0, 0.08, (2, 3)), cell=cell, pbc=True)
    atoms.calc = GradWave(
        ecut=15 * RY, pseudopotentials={"Si": PSEUDO}, xc="lda", kpts=(2, 2, 2)
    )
    opt = FIRE(atoms, logfile=None)
    assert opt.run(fmax=0.01, steps=60)
    bond = np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0])
    assert abs(bond - a * np.sqrt(3) / 4) < 1e-3
    assert np.abs(atoms.get_forces()).max() < 0.01


# --- selective dynamics (structure.fixed) --------------------------------------
#
# These drive run_relax end to end but swap the DFT calculator for an analytic
# per-atom harmonic well, so they exercise the fixed-atom wiring (constraint
# application, fmax over free components, output recording, joint fallback)
# without a real SCF and stay in the fast tier.

from ase.calculators.calculator import Calculator, all_changes  # noqa: E402


class _Harmonic(Calculator):
    """E = 1/2 k Σ|r_i − t_i|², F_i = −k(r_i − t_i): each atom pulled toward its
    own target t_i. A held atom keeps a large unconstrained force at its target
    mismatch, so convergence proves the fmax gate ignores the fixed components."""

    implemented_properties = ("energy", "forces")

    def __init__(self, targets, k=2.0, **kw):
        super().__init__(**kw)
        self._targets = np.asarray(targets, float)
        self._k = k

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        d = self.atoms.get_positions() - self._targets
        self.results["energy"] = 0.5 * self._k * float((d * d).sum())
        self.results["forces"] = -self._k * d


# An orthogonal cell so a held fractional axis coincides with the Cartesian one
# (FixScaled fixes fractional coordinates — the documented "atoms ride the cell"
# choice), which keeps the per-axis assertions transparent. atom 0 sits at the
# origin with a far target (large held force); atom 1 starts at [1,1,1] and must
# relax to [2, 2, 2].
_TARGETS = np.array([[2.5, 2.5, 2.5], [2.0, 2.0, 2.0]])


def _relax_input(tmp_path, fixed_line: str = "", method: str = "nested"):
    from gradwave.inputs import load_input
    from tests.helpers import PSEUDOS

    body = f"""
structure:
  cell: [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
  positions: {{cart: [[0, 0, 0], [1.0, 1.0, 1.0]]}}
  species: [C, C]
  {fixed_line}
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: 680.28
task: relax
relax: {{optimizer: bfgs, fmax: 0.001, max_steps: 200, method: {method}}}
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def _patch_harmonic(monkeypatch):
    # _build_relax_calc(inp, verbose, scf_step_hook=...): the extra kwargs
    # (verbose for the Pulay message #217, scf_step_hook for the ionic-step
    # header) ride along; the harmonic stand-in ignores them all
    monkeypatch.setattr("gradwave.api._build_relax_calc",
                        lambda inp, verbose=True, **_kw: _Harmonic(_TARGETS))


def test_fixed_atom_is_stationary_and_free_atom_relaxes(tmp_path, monkeypatch):
    from gradwave.api import run_relax

    _patch_harmonic(monkeypatch)
    inp = _relax_input(tmp_path, "fixed: [0]")
    pos0 = inp.atoms.get_positions().copy()
    relax, atoms, _frames = run_relax(inp, verbose=False)

    assert relax["converged"] is True
    # held atom did not move at all, bit for bit
    assert np.array_equal(atoms.get_positions()[0], pos0[0])
    # free atom reached its target
    assert np.allclose(atoms.get_positions()[1], _TARGETS[1], atol=1e-3)
    # the mask is recorded in the output
    assert relax["fixed"] == [[True, True, True], [False, False, False]]
    assert relax["n_fixed_atoms"] == 1


def test_fmax_gate_is_over_free_components_only(tmp_path, monkeypatch):
    from gradwave.api import run_relax

    _patch_harmonic(monkeypatch)
    inp = _relax_input(tmp_path, "fixed: [0]")
    relax, atoms, _frames = run_relax(inp, verbose=False)

    # converged despite the held atom carrying a large UNCONSTRAINED force
    assert relax["converged"] is True
    unconstrained = atoms.get_forces(apply_constraint=False)
    assert np.abs(unconstrained[0]).max() > 0.1          # held atom force is large
    assert np.abs(atoms.get_forces()[0]).max() == 0.0    # but masked to zero
    # reported fmax (over free components) is below the target
    assert relax["fmax_eV_ang"] < inp.relax.fmax


def test_per_axis_mask_holds_only_named_axes(tmp_path, monkeypatch):
    from gradwave.api import run_relax

    _patch_harmonic(monkeypatch)
    # fix atom 1's x only; y, z free. atom 0 free.
    inp = _relax_input(tmp_path, "fixed: [[false, false, false], [true, false, false]]")
    pos0 = inp.atoms.get_positions().copy()
    _relax, atoms, _frames = run_relax(inp, verbose=False)
    final = atoms.get_positions()

    assert final[1, 0] == pos0[1, 0]                     # x held
    assert not np.isclose(final[1, 1], pos0[1, 1])       # y moved
    assert np.allclose(final[1, 1:], _TARGETS[1, 1:], atol=1e-3)


def test_no_flags_leaves_output_and_atoms_unconstrained(tmp_path, monkeypatch):
    from gradwave.api import run_relax

    _patch_harmonic(monkeypatch)
    inp = _relax_input(tmp_path)  # no fixed
    assert inp.fixed is None
    relax, atoms, _frames = run_relax(inp, verbose=False)

    assert atoms.constraints == []                       # no constraint attached
    assert "fixed" not in relax and "n_fixed_atoms" not in relax
    # every atom relaxed to its own target
    assert np.allclose(atoms.get_positions(), _TARGETS, atol=1e-3)


def test_joint_method_falls_back_to_nested_with_reason(tmp_path, monkeypatch):
    from gradwave.api import run_relax

    _patch_harmonic(monkeypatch)
    inp = _relax_input(tmp_path, "fixed: [0]", method="joint")
    relax, atoms, _frames = run_relax(inp, verbose=False)

    assert relax["method"] == "nested"
    assert relax["requested_method"] == "joint"
    assert "selective dynamics" in relax["fallback_reason"]
    # and the fallback still honoured the mask
    assert np.array_equal(atoms.get_positions()[0], inp.atoms.get_positions()[0])
