"""The semicore-LO helper and the Ti 3s-valence default (PR #464 follow-up).

``flapw_semicore_defaults`` must reproduce the manual Ti semicore config used in
``experiments/autoapw/run_tio2.py`` byte-for-byte, and a plain (kwargs-free)
``_multi_setup`` on rutile TiO2 — Ti being default-on — must build the SAME
context (los_by_key / core_map / nbands / el_override) as the explicit manual
config. That equivalence is the acceptance test for the global default flip.
"""

import pytest

from gradwave.flapw import SEMICORE_LO, flapw_semicore_defaults
from gradwave.flapw.scf import _multi_setup

# rutile TiO2, the exact campaign cell (Bohr) and fractional atoms of run_tio2.py.
_U = 0.3048
_A_BOHR = [8.68083, 8.68083, 5.59096]
_ATOMS = [
    ((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
    ((_U, _U, 0.0), "O"), ((1 - _U, 1 - _U, 0.0), "O"),
    ((0.5 + _U, 0.5 - _U, 0.5), "O"), ((0.5 - _U, 0.5 + _U, 0.5), "O"),
]
_RADII = {"Ti": 0.95, "O": 0.80}

# The manual Ti semicore config from experiments/autoapw/run_tio2.py (the --lo branch).
_MANUAL_TI = dict(
    los={"Ti": [(0, "3s"), (1, "3p")]},
    core={"Ti": [(0, 1, 2), (0, 2, 2), (1, 1, 6)]},
    val_e={"Ti": 12},
    el_override={"Ti": {1: "3d"}},
)


def test_semicore_defaults_ti_byte_identical():
    """flapw_semicore_defaults('Ti') reproduces the manual run_tio2 config exactly."""
    los, core, val_e, el_override = flapw_semicore_defaults("Ti")
    assert los == _MANUAL_TI["los"]
    assert core == _MANUAL_TI["core"]
    assert val_e == _MANUAL_TI["val_e"]
    assert el_override == _MANUAL_TI["el_override"]
    # val_e is DERIVED (Z − frozen occ), not hard-coded: 22 − (2+2+6) = 12.
    assert val_e["Ti"] == SEMICORE_LO["Ti"]["z"] - sum(
        occ for _, _, occ in SEMICORE_LO["Ti"]["core"])


def test_untabulated_element_raises():
    with pytest.raises(KeyError):
        flapw_semicore_defaults("Fe")


@pytest.mark.standard
def test_ti_default_matches_manual_context():
    """A plain _multi_setup on rutile (Ti default-on) builds the same context as the
    explicit manual semicore config: los_by_key, core_map, nbands=24, el_override."""
    kw = dict(ecut=150.0, lmax=2, kmesh=(2, 2, 3), use_symmetry=False)
    auto = _multi_setup(_A_BOHR, _ATOMS, _RADII, **kw)              # Ti semicore by default
    manual = _multi_setup(_A_BOHR, _ATOMS, _RADII, **kw, **_MANUAL_TI)

    assert auto.los_by_key == manual.los_by_key
    # both Ti keys carry the 3s+3p LOs; neither O key does
    assert auto.los_by_key == {"a0": [(0, "3s"), (1, "3p")], "a1": [(0, "3s"), (1, "3p")]}
    assert auto.core_map["Ti"] == manual.core_map["Ti"] == [(0, 1, 2), (0, 2, 2), (1, 1, 6)]
    assert auto.el_override == manual.el_override == {"Ti": {1: "3d"}}
    # 2 Ti × 12 + 4 O × 6 = 48 e → 24 occupied bands
    assert auto.nbands == manual.nbands == 24


@pytest.mark.standard
def test_explicit_kwargs_disable_auto_layering():
    """An explicit los for Ti turns auto layering off for Ti — the caller keeps full
    control (no auto el_override is injected underneath a caller-specified basis)."""
    ctx = _multi_setup(_A_BOHR, _ATOMS, _RADII, ecut=150.0, lmax=2, kmesh=(1, 1, 1),
                       use_symmetry=False, los={"Ti": [(0, "3s")]},
                       val_e={"Ti": 12}, core={"Ti": [(0, 1, 2), (0, 2, 2), (1, 1, 6)]})
    # only the caller's 3s LO, and NO injected el_override (caller took control)
    assert ctx.los_by_key == {"a0": [(0, "3s")], "a1": [(0, "3s")]}
    assert ctx.el_override is None
