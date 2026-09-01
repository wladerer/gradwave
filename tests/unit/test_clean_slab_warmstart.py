"""Clean-slab warm-start seed for adsorbate site/coverage screens.

Pure/tiny (no SCF): the load-bearing property is electron-count exactness. A
converged clean-slab density integrates to N_slab; the adsorbate-slab system has
N_slab + N_ads electrons. Bare clean-slab reuse starts an electron short (the
"hole"); adding SAD(adsorbate only) restores the correct count. These tests pin
that fix and the shared-cell/grid contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradwave.io.checkpoint import adsorbate_warm_start_seed
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.guess import sad_density
from gradwave.scf.loop import setup_system
from tests.helpers import RY, pseudo


def _integral(rho: torch.Tensor, grid) -> float:
    """∫ρ dΩ on the FFT grid = mean·Ω."""
    return float(rho.sum()) * grid.volume / grid.n_points


def _al():
    return parse_upf(pseudo("Al_ONCV_PBE-1.2.upf"))


def _h():
    return parse_upf(pseudo("H_ONCV_PBE-1.2.upf"))


# A slab cell with vacuum along z; the clean slab is 2 Al, the adsorbate config
# adds 1 H on top IN THE SAME CELL (the contract the seed builder enforces).
_CELL = np.diag([4.05, 4.05, 15.0]).astype(float)
_SLAB_POS = np.array([[0.0, 0.0, 5.0], [2.025, 2.025, 7.0]])
_H_POS = np.array([[2.025, 2.025, 9.0]])
_ECUT = 12.0 * RY


def _clean_slab():
    return setup_system(_CELL, _SLAB_POS, [0, 0], [_al()], ecut=_ECUT, kmesh=(1, 1, 1))


def _ads_slab():
    pos = np.vstack([_SLAB_POS, _H_POS])
    return setup_system(_CELL, pos, [0, 0, 1], [_al(), _h()], ecut=_ECUT, kmesh=(1, 1, 1))


def _clean_payload(clean, *, nspin=1):
    """A checkpoint payload whose ρ integrates to N_slab exactly (SAD stands in
    for the converged clean-slab density — the electron count is what matters)."""
    rho = sad_density(clean.grid, clean.positions, clean.species_of_atom,
                      clean.upfs, clean.n_electrons)
    payload = {
        "kind": "nc",
        "nspin": nspin,
        "grid_shape": tuple(clean.grid.shape),
        "volume_ang3": float(clean.grid.volume),
        "rho": rho,
        "rho_spin": None,
    }
    if nspin == 2:
        payload["rho_spin"] = [0.5 * rho, 0.5 * rho]
    return payload


def test_seed_restores_electron_count():
    clean = _clean_slab()
    ads = _ads_slab()
    payload = _clean_payload(clean)
    n_slab = clean.n_electrons
    n_ads = float(_h().z_valence)

    # The bug the fix addresses: bare clean-slab reuse integrates to N_slab, an
    # electron short by N_ads.
    assert _integral(payload["rho"], ads.grid) == pytest.approx(n_slab, abs=1e-7)

    seed = adsorbate_warm_start_seed(payload, ads, [2])
    # The fix: ρ_clean + SAD(adsorbate) integrates to N_slab + N_ads exactly.
    assert _integral(seed["rho"], ads.grid) == pytest.approx(n_slab + n_ads, abs=1e-6)
    assert seed["rho_spin"] is None
    assert seed["nspin"] == 1


def test_seed_adds_only_adsorbate_charge():
    """The seed minus ρ_clean integrates to exactly N_ads (the SAD term alone)."""
    clean = _clean_slab()
    ads = _ads_slab()
    payload = _clean_payload(clean)
    seed = adsorbate_warm_start_seed(payload, ads, [2])
    delta = seed["rho"] - payload["rho"].to(seed["rho"])
    assert _integral(delta, ads.grid) == pytest.approx(float(_h().z_valence), abs=1e-6)


def test_seed_nspin2_splits_half_half():
    """nspin=2: each channel gains N_ads/2, so the total gains N_ads while the
    seed magnetization M = N↑ − N↓ is unchanged (adsorbate enters non-magnetic)."""
    clean = _clean_slab()
    ads = _ads_slab()
    payload = _clean_payload(clean, nspin=2)
    n_ads = float(_h().z_valence)
    seed = adsorbate_warm_start_seed(payload, ads, [2])
    up, dn = seed["rho_spin"]
    m_before = (_integral(payload["rho_spin"][0], ads.grid)
                - _integral(payload["rho_spin"][1], ads.grid))
    m_after = _integral(up, ads.grid) - _integral(dn, ads.grid)
    assert m_after == pytest.approx(m_before, abs=1e-6)
    assert _integral(up, ads.grid) + _integral(dn, ads.grid) == pytest.approx(
        clean.n_electrons + n_ads, abs=1e-6)


def test_grid_mismatch_raises():
    """A clean slab converged on a DIFFERENT cell/grid cannot seed — the builder
    rejects it rather than silently misplacing charge."""
    clean = _clean_slab()
    ads = _ads_slab()
    payload = _clean_payload(clean)
    payload["grid_shape"] = (payload["grid_shape"][0] + 2, *payload["grid_shape"][1:])
    with pytest.raises(ValueError, match="same FFT grid"):
        adsorbate_warm_start_seed(payload, ads, [2])


def test_uspp_checkpoint_rejected():
    clean = _clean_slab()
    ads = _ads_slab()
    payload = _clean_payload(clean)
    payload["kind"] = "uspp"
    with pytest.raises(ValueError, match="norm-conserving"):
        adsorbate_warm_start_seed(payload, ads, [2])


def test_empty_adsorbate_rejected():
    clean = _clean_slab()
    ads = _ads_slab()
    payload = _clean_payload(clean)
    with pytest.raises(ValueError, match="empty"):
        adsorbate_warm_start_seed(payload, ads, [])


def test_calculator_hook_consumes_seed_once():
    """warm_start_from_clean_slab stashes a one-shot request; _warm_start builds
    the seed against the adsorbate-slab system and consumes it."""
    from gradwave.calculator import GradWave

    clean = _clean_slab()
    ads = _ads_slab()
    payload = _clean_payload(clean)
    calc = GradWave(
        pseudopotentials={"Al": pseudo("Al_ONCV_PBE-1.2.upf"),
                          "H": pseudo("H_ONCV_PBE-1.2.upf")},
        ecut=_ECUT, kpts=(1, 1, 1))
    calc.warm_start_from_clean_slab(payload, [2])
    assert calc._clean_slab_seed_request is not None

    seed = calc._warm_start(ads, nspin=1)
    assert seed is not None
    assert _integral(seed["rho"], ads.grid) == pytest.approx(
        clean.n_electrons + float(_h().z_valence), abs=1e-6)
    # consumed: a second call falls through to the ordinary (no-history) path
    assert calc._clean_slab_seed_request is None
    assert calc._warm_start(ads, nspin=1) is None
