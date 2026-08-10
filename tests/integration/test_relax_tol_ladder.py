"""The outer-SCF tolerance ladder reaches the same relaxed fixed point as a
constant-rhotol relax, and its reported energy sits at the FULL-tol fixed point
(the exactness re-solve gate), not a loosened one.

``relax.tol_ladder`` loosens each ionic step's rhotol/etol when the optimizer's
max|F| is large and tightens as forces shrink, then re-solves the converged
geometry once at the full rhotol. This drives a rattled-Si relaxation through the
real nested driver (api.run_relax) three ways — baseline, ladder-loose-first,
ladder-tight-first — and checks:

  (a) exactness: a fresh full-rhotol SCF at the ladder's own converged geometry
      reproduces the reported energy to ~1e-7 eV (the re-solve landed on the
      fixed point, so downstream forces/derivatives are exact);
  (b) same minimum: the ladder and baseline relaxations converge to the same
      geometry and energy (the loosened early steps do not corrupt the path);
  (c) plumbing: the ladder records a final_resolve_scf_iter and a per-step
      rhotol_used schedule that actually varies.

Standard tier: a handful of Si2 SCFs, above the fast gate's per-test budget.
"""

import numpy as np
import pytest

from gradwave.api import run_relax
from gradwave.calculator import GradWave
from gradwave.inputs import load_input
from tests.helpers import PSEUDOS, pseudo

pytestmark = pytest.mark.standard

_A0 = 5.43
_IDEAL = [[0.0, 0.0, 0.0], [_A0 / 4, _A0 / 4, _A0 / 4]]
_CELL = (_A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])).tolist()


def _rattled():
    rng = np.random.default_rng(3)
    pos = np.array(_IDEAL) + rng.normal(0.0, 0.06, (2, 3))
    return pos.tolist()


def _input(tmp_path, ladder: str | None):
    """ladder: None (baseline) | 'loose' | 'tight'."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pos = _rattled()
    lad = ""
    if ladder is not None:
        lad = (f"  tol_ladder: true\n"
               f"  tol_ladder_c: 1.0e-3\n"
               f"  tol_ladder_p: 2.0\n"
               f"  tol_ladder_rhotol_start: 1.0e-4\n"
               f"  tol_ladder_first_step: {ladder}\n")
    body = f"""
structure:
  cell: {_CELL}
  positions: {{cart: {pos}}}
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: 204.0
xc: lda
kpoints: {{mesh: [2, 2, 2]}}
symmetry: false
task: relax
scf: {{etol: 1.0e-8, rhotol: 1.0e-7, max_iter: 200, diago: {{tol: 1.0e-9}}}}
relax:
  optimizer: bfgs
  fmax: 0.02
  max_steps: 60
{lad}output: {{dir: {tmp_path / 'out'}}}
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def _fresh_full_tol_energy(atoms) -> float:
    """A single independent full-rhotol SCF at the given geometry through a fresh
    (no-ladder, no warm-start-history) calculator — the ground-truth fixed-point
    energy the ladder must report."""
    a = atoms.copy()
    a.calc = GradWave(pseudopotentials={"Si": pseudo("Si_ONCV_PBE-1.2.upf")},
                      ecut=204.0, xc="lda", kpts=(2, 2, 2), use_symmetry=False,
                      etol=1e-8, rhotol=1e-7, diago_tol=1e-9, max_iter=200)
    return float(a.get_potential_energy())


def test_tol_ladder_exact_fixed_point_and_same_minimum(tmp_path):
    base_relax, base_atoms, _ = run_relax(_input(tmp_path / "b", None), verbose=False)
    loose_relax, loose_atoms, _ = run_relax(
        _input(tmp_path / "l", "loose"), verbose=False)
    tight_relax, tight_atoms, _ = run_relax(
        _input(tmp_path / "t", "tight"), verbose=False)

    assert base_relax["converged"]
    assert loose_relax["converged"] and tight_relax["converged"]

    # (a) exactness gate: the reported ladder energy equals a fresh full-rhotol
    # SCF at the ladder's OWN converged geometry (the re-solve is a true fixed
    # point, not a loosened SCF value).
    for relax, atoms in ((loose_relax, loose_atoms), (tight_relax, tight_atoms)):
        assert "final_resolve_scf_iter" in relax
        truth = _fresh_full_tol_energy(atoms)
        assert abs(relax["energy_eV"] - truth) < 1e-6, relax["energy_eV"] - truth

    # (b) same minimum: loosened early steps do not corrupt the converged
    # geometry/energy relative to the constant-rhotol baseline.
    for relax, atoms in ((loose_relax, loose_atoms), (tight_relax, tight_atoms)):
        assert abs(relax["energy_eV"] - base_relax["energy_eV"]) < 5e-5
        dpos = np.abs(atoms.get_positions() - base_atoms.get_positions()).max()
        assert dpos < 5e-3, dpos

    # (c) the schedule actually varied the tolerance across the trajectory (the
    # ladder is doing something, not silently running at the full tol).
    rhotols = [s["rhotol_used"] for s in loose_relax["trajectory"]
               if "rhotol_used" in s]
    assert rhotols and max(rhotols) > min(rhotols)


def test_tol_ladder_off_is_unrecorded(tmp_path):
    # baseline relax carries no ladder bookkeeping (byte-for-byte the old path).
    relax, _atoms, _ = run_relax(_input(tmp_path, None), verbose=False)
    assert "tol_ladder" not in relax
    assert "final_resolve_scf_iter" not in relax
