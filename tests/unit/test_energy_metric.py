"""Energy-metric SCF convergence gate (postscf._response.kernel_energy_error,
scf.common.convergence_gate, and the inputs surface).

The estimator returns the residual's exact second-order energy error
1/2 <r|K_Hxc|r>, so it is validated against a finite-difference second
difference of the actual Hartree+XC energy (the repo's oracle convention),
per-channel for the collinear spin path. No SCF here — these run fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.density import sigma_from_rho
from gradwave.core.energies.hartree import hartree_energy
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.grids import build_fft_grid
from gradwave.postscf._response import kernel_energy_error
from gradwave.scf.common import convergence_gate
from tests.helpers import PSEUDOS, RY


def _grid(a=6.0, ecut_ry=10.0):
    return build_fft_grid(np.eye(3) * a, ecut_ry * RY, equal_dims=None)


def _smooth_rho(grid, n_elec=8.0, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(grid.shape, dtype=torch.float64) * 0.1 + 0.3
    cell = grid.volume / grid.n_points
    return x / (x.sum() * cell) * n_elec


def _charge_conserving(grid, seed):
    """A random real grid field with zero G=0 component (∫δρ = 0)."""
    torch.manual_seed(seed)
    d = torch.rand(grid.shape, dtype=torch.float64) - 0.5
    dg = r_to_g(d.to(torch.complex128))
    dg.reshape(-1)[0] = 0.0
    d = g_to_r_box(dg, real=True)
    return d / d.abs().max()


def _ehxc1(grid, r, xc):
    eh = hartree_energy(r_to_g(r.to(torch.complex128)), grid.g2, grid.volume)
    sig = sigma_from_rho(r, grid.g_cart) if xc.needs_gradient else None
    return float(eh) + float(xc.energy(r, grid.volume, sig))


def _ehxc2(grid, ru, rd, xc):
    eh = hartree_energy(r_to_g((ru + rd).to(torch.complex128)), grid.g2, grid.volume)
    if xc.needs_gradient:
        s = (sigma_from_rho(ru, grid.g_cart), sigma_from_rho(rd, grid.g_cart),
             sigma_from_rho(ru + rd, grid.g_cart))
    else:
        s = (None, None, None)
    return float(eh) + float(xc.energy(ru, rd, grid.volume, *s))


# --------------------------------------------------------------------------- #
#  FD oracle: 1/2 <r|K_Hxc|r> == second difference of E_Hxc                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("xc_cls, tol", [(LDA_PW92, 1e-3), (PBE, 3e-2)])
def test_estimator_matches_second_difference_nspin1(xc_cls, tol):
    """½⟨r|K_Hxc|r⟩ equals ½[E(ρ+r) − 2E(ρ) + E(ρ−r)] to the FD truncation
    floor: exact for the quadratic Hartree part, O(r) for the XC anharmonicity
    (LDA tighter than GGA)."""
    grid = _grid()
    xc = xc_cls()
    rho = _smooth_rho(grid)
    r = _charge_conserving(grid, seed=1) * 1e-3
    est, e_charge, e_mag = kernel_energy_error(grid, xc, [r], [rho], None, 1)
    oracle = 0.5 * (_ehxc1(grid, rho + r, xc) - 2 * _ehxc1(grid, rho, xc)
                    + _ehxc1(grid, rho - r, xc))
    assert abs(est / oracle - 1.0) < tol
    # nspin=1 has no magnetization channel
    assert e_mag == 0.0
    assert e_charge == pytest.approx(est)


@pytest.mark.parametrize("xc_cls, tol", [(LSDA_PW92, 1e-3), (SpinPBE, 3e-2)])
def test_estimator_matches_second_difference_nspin2(xc_cls, tol):
    """The per-spin ½⟨r|K_Hxc|r⟩ matches the second difference of the collinear
    E_Hxc under a joint (δρ↑, δρ↓) perturbation."""
    grid = _grid()
    xc = xc_cls()
    rho = _smooth_rho(grid)
    ru, rd = rho * 0.55, rho * 0.45
    du = _charge_conserving(grid, seed=2) * 1e-3
    dd = _charge_conserving(grid, seed=3) * 1e-3
    est, e_charge, e_mag = kernel_energy_error(grid, xc, [du, dd], [ru, rd], None, 2)
    oracle = 0.5 * (_ehxc2(grid, ru + du, rd + dd, xc)
                    - 2 * _ehxc2(grid, ru, rd, xc)
                    + _ehxc2(grid, ru - du, rd - dd, xc))
    assert abs(est / oracle - 1.0) < tol


def test_charge_and_magnetization_channels_are_pure():
    """A pure-charge residual (δρ↑ = δρ↓) has ~zero magnetization energy; a
    pure-magnetization residual (δρ↑ = −δρ↓) has ~zero charge energy (its total
    charge, hence Hartree, vanishes and the f_xc charge block is small)."""
    grid = _grid()
    xc = SpinPBE()
    rho = _smooth_rho(grid)
    ru, rd = rho * 0.55, rho * 0.45
    d = _charge_conserving(grid, seed=4) * 1e-3
    # pure charge: δρ↑ = δρ↓
    _, e_charge_c, e_mag_c = kernel_energy_error(grid, xc, [d, d], [ru, rd], None, 2)
    assert abs(e_mag_c) < 1e-6 * abs(e_charge_c)
    # pure magnetization: δρ↑ = −δρ↓ (total charge residual is zero)
    _, e_charge_m, e_mag_m = kernel_energy_error(grid, xc, [d, -d], [ru, rd], None, 2)
    assert abs(e_charge_m) < 1e-6 * abs(e_mag_m)


def test_hartree_dominates_a_charge_residual():
    """A LONG-WAVELENGTH charge residual (what a real SCF leaves) gives a
    POSITIVE energy estimate — Hartree 4πe²/G² dominates at small |G|. This is
    the physical basis of the gate: the charge channel is Hartree-amplified,
    the magnetization channel is not."""
    grid = _grid()
    rho = _smooth_rho(grid)
    # keep only the lowest nonzero |G| shell: a smooth, charge-conserving field
    d = _charge_conserving(grid, seed=5)
    dg = r_to_g(d.to(torch.complex128))
    g2 = grid.g2
    gnz = g2[g2 > 1e-12]
    keep = g2 <= float(gnz.min()) * 1.01
    dg = torch.where(keep, dg, torch.zeros_like(dg))
    r = g_to_r_box(dg, real=True)
    r = r / r.abs().max() * 1e-2
    est, _, _ = kernel_energy_error(grid, PBE(), [r], [rho], None, 1)
    assert est > 0.0


def test_metagga_is_rejected():
    """The kernel omits the τ response, so a meta-GGA raises rather than return
    a silently-wrong estimate."""
    grid = _grid()
    rho = _smooth_rho(grid)
    r = _charge_conserving(grid, seed=6) * 1e-3

    class _FakeMetaGGA:
        needs_tau = True
        needs_gradient = True

    with pytest.raises(NotImplementedError, match="meta-GGA"):
        kernel_energy_error(grid, _FakeMetaGGA(), [r], [rho], None, 1)


# --------------------------------------------------------------------------- #
#  convergence_gate: the energy path vs the density path                       #
# --------------------------------------------------------------------------- #


def test_convergence_gate_density_path_unchanged():
    """With energy_error=None the gate is the original density gate: energy tail
    AND residual AND the stale-solve bound."""
    # all clauses satisfied
    assert convergence_gate(1e-9, 1e-8, 1e-9, 1e-8, 1e-7, 1e-9)
    # residual too large
    assert not convergence_gate(1e-9, 1e-5, 1e-9, 1e-8, 1e-7, 1e-9)
    # energy tail too large
    assert not convergence_gate(1e-6, 1e-8, 1e-9, 1e-8, 1e-7, 1e-9)
    # eigensolve too loose (stale-solve guard)
    assert not convergence_gate(1e-9, 1e-8, 1e-4, 1e-8, 1e-7, 1e-9)


def test_convergence_gate_energy_path():
    """With energy_error given the residual clause is replaced by |E| < entol;
    the energy-tail and stale-solve clauses still hold. A large density residual
    no longer blocks convergence when the energy error is small."""
    # small energy error + settled tail + tight solve → converged even though
    # the density residual (2e-3) is far above a typical rhotol
    assert convergence_gate(1e-9, 2e-3, 1e-9, 1e-8, 1e-7, 1e-9,
                            energy_error=1e-9, entol=1e-6)
    # energy error above threshold → not converged
    assert not convergence_gate(1e-9, 2e-3, 1e-9, 1e-8, 1e-7, 1e-9,
                                energy_error=1e-5, entol=1e-6)
    # a negative estimate is gated on magnitude
    assert convergence_gate(1e-9, 2e-3, 1e-9, 1e-8, 1e-7, 1e-9,
                            energy_error=-1e-9, entol=1e-6)
    assert not convergence_gate(1e-9, 2e-3, 1e-9, 1e-8, 1e-7, 1e-9,
                                energy_error=-1e-5, entol=1e-6)
    # the energy tail still gates
    assert not convergence_gate(1e-6, 2e-3, 1e-9, 1e-8, 1e-7, 1e-9,
                                energy_error=1e-9, entol=1e-6)


# --------------------------------------------------------------------------- #
#  Input surface: scf.convergence selector + entol                            #
# --------------------------------------------------------------------------- #


def _write(tmp_path, extra: str) -> Path:
    body = f"""
structure:
  cell: [[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]]
  positions: {{cart: [[0, 0, 0], [0.89175, 0.89175, 0.89175]]}}
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: 680.28
{extra}"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def test_convergence_default_is_density(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, ""))
    assert inp.scf.convergence == "density"
    assert inp.scf.entol == pytest.approx(1e-6)


def test_convergence_energy_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, "scf: {convergence: energy, entol: 5.0e-7}\n"))
    assert inp.scf.convergence == "energy"
    assert inp.scf.entol == pytest.approx(5e-7)


def test_convergence_unknown_value_rejected(tmp_path):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="scf.convergence must be"):
        load_input(_write(tmp_path, "scf: {convergence: bogus}\n"))


def test_convergence_unknown_key_rejected(tmp_path):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="did you mean"):
        load_input(_write(tmp_path, "scf: {convergance: energy}\n"))
