"""Unit tests for the SCF code-review fixes (owned scope: scf/*).

Fast CPU-only checks for the input-validation and invariant fixes that can be
exercised without a full SCF run. The numerically delicate paths (Sternheimer
guard, noncollinear E_xc ordering) are covered by the integration/magnetism
suites; here we lock the guards and accessors that were silent before.
"""

from __future__ import annotations

import pytest
import torch

from gradwave import constants
from gradwave.scf import local_tf
from gradwave.scf.mixing import BroydenMixer, JohnsonMixer, PulayMixer
from gradwave.scf.uspp_loop import _build_mixer, _resolve_start_mag


# ---- fix 10: local_tf uses the shared Bohr radius ------------------------

def test_local_tf_bohr_is_shared_constant():
    assert local_tf._BOHR == constants.BOHR_ANG


# ---- fix 3: start_mag per-atom / per-species resolution ------------------

def test_resolve_start_mag_none_is_zeros():
    assert _resolve_start_mag(None, [0, 0, 1], 2) == [0.0, 0.0, 0.0]


def test_resolve_start_mag_per_species_broadcast():
    # 3 atoms, 2 species; a per-species list broadcasts onto each atom
    assert _resolve_start_mag([0.5, -0.3], [0, 0, 1], 2) == [0.5, 0.5, -0.3]


def test_resolve_start_mag_per_atom():
    # len == n_atoms and n_atoms != n_species → per-atom (AFM seed)
    assert _resolve_start_mag([0.5, -0.5, 0.2], [0, 0, 1], 2) == [0.5, -0.5, 0.2]


def test_resolve_start_mag_one_atom_per_species_is_unambiguous():
    # na == n_species (bijection): per-species broadcast == per-atom
    assert _resolve_start_mag([0.3, 0.7], [0, 1], 2) == [0.3, 0.7]


def test_resolve_start_mag_bad_length_raises():
    with pytest.raises(ValueError, match="one entry per atom or per species"):
        _resolve_start_mag([0.1, 0.2, 0.3, 0.4], [0, 0, 1], 2)


# ---- fix 4: mixing-scheme validation in _build_mixer ---------------------

def _mk(scheme):
    g2 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    return _build_mixer(scheme, g2, alpha=0.7, history=8, kerker=False,
                        kerker_mask=None, step_scale=None, metric_w=None,
                        w0=0.01, adapt_ids=None)


def test_build_mixer_valid_schemes():
    assert isinstance(_mk("pulay"), PulayMixer)
    assert isinstance(_mk("broyden"), BroydenMixer)
    assert isinstance(_mk("johnson"), JohnsonMixer)


def test_build_mixer_bad_scheme_raises():
    with pytest.raises(ValueError, match="mixing_scheme must be"):
        _mk("anderson")


# ---- fix 7: G=0 residual invariant is a raise, not a bare assert ---------

@pytest.mark.parametrize("cls", [PulayMixer, BroydenMixer, JohnsonMixer])
def test_mixer_g0_check_raises(cls):
    g2 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    mix = cls(g2, alpha=0.5, history=4, check_g0=True)
    rho_in = torch.zeros(4, dtype=torch.complex128)
    rho_out = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)  # G=0 moves
    with pytest.raises(ValueError, match="G=0 residual nonzero"):
        mix.step(rho_in, rho_out)


@pytest.mark.parametrize("cls", [PulayMixer, BroydenMixer, JohnsonMixer])
def test_mixer_g0_check_off_allows_g0_residual(cls):
    g2 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    mix = cls(g2, alpha=0.5, history=4, check_g0=False)
    rho_in = torch.zeros(4, dtype=torch.complex128)
    rho_out = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)
    out = mix.step(rho_in, rho_out)  # must not raise
    assert torch.isfinite(out).all()


# ---- fix 8: ill-conditioned Pulay drops the oldest entry, stays finite ---

def test_pulay_ill_conditioned_history_stays_finite():
    # parallel, shrinking residuals make the DIIS overlap matrix rank-deficient;
    # the drop-oldest retry must keep the step finite (no swallowed exception)
    g2 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    mix = PulayMixer(g2, alpha=0.5, history=8, check_g0=False)
    direction = torch.tensor([0.0, 1.0, 1.0j, 2.0], dtype=torch.complex128)
    rho = torch.zeros(4, dtype=torch.complex128)
    for k in range(6):
        res = (0.5 ** k) * direction
        rho = mix.step(rho, rho + res)
        assert torch.isfinite(rho).all()


# ---- fix 12: block_mult accessor replaces private _block_mult reads ------

def test_block_mult_none_without_adapt_blocks():
    g2 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    assert PulayMixer(g2).block_mult is None
    assert BroydenMixer(g2).block_mult is None
    assert JohnsonMixer(g2).block_mult is None


def test_block_mult_dict_with_adapt_blocks():
    g2 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    blocks = torch.tensor([0, 0, 1, 1])
    mix = PulayMixer(g2, adapt_blocks=blocks)
    bm = mix.block_mult
    assert bm == {0: 1.0, 1: 1.0}
    # accessor returns a copy — mutating it must not touch mixer state
    bm[0] = 0.1
    assert mix.block_mult == {0: 1.0, 1: 1.0}
