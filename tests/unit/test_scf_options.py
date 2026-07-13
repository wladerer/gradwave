"""SCFOptions.from_kwargs must route legacy flat kwargs to the right
option group and reject unknown keys loudly."""

import dataclasses

import pytest

from gradwave.scf.options import MixerOptions, SCFOptions


def test_kwargs_routing():
    o = SCFOptions.from_kwargs(etol=1e-9, mixing_scheme="johnson",
                               mixing_alpha=0.3, criterion="energy",
                               spin_precond=True)
    assert o.etol == 1e-9
    assert o.criterion == "energy"
    assert o.mixer.scheme == "johnson"
    assert o.mixer.alpha == 0.3
    assert o.mixer.spin_precond is True
    assert o.mixer.history == MixerOptions().history


def test_unknown_key_raises():
    with pytest.raises(TypeError, match="rhotoll"):
        SCFOptions.from_kwargs(rhotoll=1e-8)


def test_frozen():
    o = SCFOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.etol = 1.0
