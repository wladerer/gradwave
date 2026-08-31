"""ESM open-boundary knobs on the ASE ``GradWave`` calculator (the correctness
fix): ``boundary`` / ``esm_bias`` / ``target_mu`` must thread from the calculator
constructor into the SCF call, the non-orthogonal-slab geometry must be rejected
with a clear error before any expensive setup, and a bad boundary string must
fail at construction.

The threading test monkeypatches the SCF so no self-consistent iteration runs —
it builds the (tiny, Γ-only H) plane-wave system, captures the kwargs the
calculator hands ``scf()``, and aborts. That verifies routing/plumbing without a
benchmark: the physics of the ESM potential itself is covered by
test_esm_hartree.py / test_esm_correction.py, and the measured convergence win
(Pt(111)+CO: periodic fails at 80 iters with washed-out faces [4.999, 5.022];
open_z converges in 58 with correct asymmetric faces [5.144, 4.832]) is a
deferred asus/H100 benchmark, not a unit test.
"""

import numpy as np
import pytest
from ase import Atoms

from gradwave.calculator import GradWave, _validate_esm_geometry
from tests.helpers import RY, pseudo


class _StopSCF(Exception):
    """Sentinel to abort inside the monkeypatched scf() before any iteration."""


def _ortho_h(cell=None):
    """A single H in an orthogonal box (c ⊥ a,b) — valid ESM slab geometry."""
    if cell is None:
        cell = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 10.0]]
    return Atoms("H", positions=[[2.0, 2.0, 2.0]], cell=cell, pbc=True)


def _kw(**extra):
    return dict(
        ecut=20 * RY,
        pseudopotentials={"H": pseudo("H_ONCV_PBE-1.2.upf")},
        xc="pbe", kpts=(1, 1, 1), smearing="gaussian", width=0.1,
        **extra,
    )


# ---- construction-time validation (no compute) -----------------------------


def test_bad_boundary_rejected_at_construction():
    with pytest.raises(ValueError, match="boundary must be"):
        GradWave(**_kw(boundary="bogus"))


def test_validate_esm_geometry_orthogonal_ok():
    ortho = np.diag([4.0, 4.0, 10.0])
    # periodic never validates; ESM on an orthogonal cell passes
    _validate_esm_geometry(ortho, "periodic")
    _validate_esm_geometry(ortho, "open_z")
    _validate_esm_geometry(ortho, "open_z_metal")


def test_validate_esm_geometry_skewed_rejected():
    # c tilted toward a — not a slab: ESM cannot separate in-plane from z
    skew = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 0.0, 10.0]])
    with pytest.raises(ValueError, match="orthogonal slab"):
        _validate_esm_geometry(skew, "open_z")


def test_rhombic_inplane_is_valid_esm_geometry():
    # a hexagonal-surface (fcc111-style) rhombic in-plane cell is fine for ESM:
    # only the OPEN axis must be ⊥ the periodic pair, not a ⊥ b.
    rhombic = np.array([[3.0, 0.0, 0.0], [1.5, 2.6, 0.0], [0.0, 0.0, 12.0]])
    _validate_esm_geometry(rhombic, "open_z")  # must not raise


# ---- geometry guard fires through calculate() before setup (no SCF) --------


def test_calculate_rejects_skewed_esm_cell():
    skew = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.5, 0.0, 10.0]]
    atoms = _ortho_h(cell=skew)
    atoms.calc = GradWave(**_kw(boundary="open_z"))
    with pytest.raises(ValueError, match="orthogonal slab"):
        atoms.get_potential_energy()


# ---- the routing fix: boundary/esm_bias/target_mu reach scf() --------------


@pytest.mark.parametrize(
    "extra, expect",
    [
        (dict(boundary="open_z"), dict(boundary="open_z", esm_bias=0.0,
                                       target_mu=None)),
        (dict(boundary="open_z_metal", esm_bias=1.5),
         dict(boundary="open_z_metal", esm_bias=1.5, target_mu=None)),
        (dict(boundary="open_z_metal", target_mu=-3.0),
         dict(boundary="open_z_metal", esm_bias=0.0, target_mu=-3.0)),
    ],
)
def test_boundary_threaded_into_scf(monkeypatch, extra, expect):
    captured: dict = {}

    def fake_scf(*args, **kwargs):
        captured.update(kwargs)
        raise _StopSCF

    monkeypatch.setattr("gradwave.calculator.scf", fake_scf)
    atoms = _ortho_h()
    atoms.calc = GradWave(**_kw(**extra))
    with pytest.raises(_StopSCF):
        atoms.get_potential_energy()
    for k, v in expect.items():
        assert captured[k] == v, f"{k}: {captured.get(k)!r} != {v!r}"


def test_default_boundary_is_periodic(monkeypatch):
    captured: dict = {}

    def fake_scf(*args, **kwargs):
        captured.update(kwargs)
        raise _StopSCF

    monkeypatch.setattr("gradwave.calculator.scf", fake_scf)
    atoms = _ortho_h()
    atoms.calc = GradWave(**_kw())  # no boundary → default
    with pytest.raises(_StopSCF):
        atoms.get_potential_energy()
    assert captured["boundary"] == "periodic"
    assert captured["esm_bias"] == 0.0
    assert captured["target_mu"] is None
