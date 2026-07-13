"""Implicit differentiation through the USPP/PAW SCF (task #58).

Milestone 1 — dE/dθ by stationarity at the converged generalized SCF point:
grid E_xc(ρ+core; θ) plus the one-center E_1c(becsum; θ) — the piece
norm-conserving never had. Observed vs FD SCF re-runs: raw_kappa 2.4e-7,
raw_mu 8.9e-9 relative (FD truncation floor).

Milestone 2 — dL/dθ for a DENSITY-dependent loss via the composite
(δρ, δbecsum) self-consistent adjoint (generalized Sternheimer + grid HVP +
one-center HVP, Anderson-mixed). Observed vs FD SCF re-runs on Si kjpaw:
raw_mu 1.2e-6, raw_kappa 2.0e-7 relative (FD floor; the NC gate was 2e-4)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.learnable import LearnableX
from gradwave.postscf.uspp_implicit import (
    uspp_density_loss_param_grads,
    uspp_energy_param_grads,
)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.slow
def test_paw_energy_param_grads_vs_fd():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    pos = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])

    def scf_at(xc):
        s = setup_uspp(SI_CELL, pos, [0, 0], [paw], ecut=20 * RY,
                       kmesh=(2, 2, 2), ecutrho=80 * RY)
        r = scf_uspp(s, xc, etol=1e-11, rhotol=1e-10, verbose=False,
                     max_iter=60)
        assert r["converged"]
        return r

    xc0 = LearnableX()
    g = uspp_energy_param_grads(scf_at(xc0), xc0)

    d = 2e-3
    es = []
    for sgn in (+1, -1):
        xc = LearnableX()
        with torch.no_grad():
            xc.raw_mu.add_(sgn * d)
        es.append(float(scf_at(xc)["energies"].free_energy))
    fd = (es[0] - es[1]) / (2 * d)
    an = float(g["raw_mu"])
    rel = abs(an - fd) / abs(fd)
    assert rel < 1e-5, f"dE/d(raw_mu) analytic {an} vs FD {fd} (rel {rel:.2e})"


@pytest.mark.slow
def test_paw_density_loss_grads_vs_fd():
    """Milestone 2: the composite (δρ, δbecsum) adjoint. A density loss has
    no stationarity shortcut — its θ-gradient carries the full
    self-consistent response (generalized Sternheimer through S, grid K_Hxc
    HVP, ∫δv Q cross term, one-center Hessian, Anderson fixed point).
    Validated against central FD of complete SCF re-runs."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    pos = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])

    def scf_at(xc):
        s = setup_uspp(SI_CELL, pos, [0, 0], [paw], ecut=15 * RY,
                       kmesh=(2, 2, 2), ecutrho=60 * RY)
        r = scf_uspp(s, xc, etol=1e-12, rhotol=1e-10, verbose=False,
                     max_iter=80)
        assert r["converged"]
        return r

    xc0 = LearnableX()
    res = scf_at(xc0)
    rho_ref = (0.95 * res["rho"]).detach().clone()

    def loss_fn(rho):
        d = rho - rho_ref
        return (d * d).sum()

    loss, grads = uspp_density_loss_param_grads(res, xc0, loss_fn)

    h = 2e-3
    vals = []
    for sgn in (+1, -1):
        xc = LearnableX()
        with torch.no_grad():
            xc.raw_mu.add_(sgn * h)
        vals.append(float(loss_fn(scf_at(xc)["rho"])))
    fd = (vals[0] - vals[1]) / (2 * h)
    an = float(grads["raw_mu"])
    rel = abs(an - fd) / abs(fd)
    # observed 1.2e-6 (FD floor); gate at the NC milestone's 2e-4 class
    assert rel < 2e-4, f"dL/d(raw_mu) adjoint {an} vs FD {fd} (rel {rel:.2e})"


@pytest.mark.slow
def test_metal_paw_density_loss_grads_vs_fd():
    """Fermi-surface response term: for a smeared metal the density adjoint
    carries three channels — Sternheimer into the uncomputed complement,
    the explicit window-pair sum with divided-difference occupation
    weights, and the diagonal δf_n = f′(δε_n − δμ) with δμ from particle
    conservation (rank-one coupling across the BZ). Validated against
    central FD of complete smeared-SCF re-runs on Al kjpaw."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 4.04
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])

    def scf_at(xc):
        s = setup_uspp(cell, pos, [0], [paw], ecut=20 * RY,
                       kmesh=(2, 2, 2), ecutrho=100 * RY, nbands=8)
        r = scf_uspp(s, xc, smearing="gaussian", width=0.5, etol=1e-12,
                     rhotol=1e-10, verbose=False, max_iter=120)
        assert r["converged"]
        return r

    xc0 = LearnableX()
    res = scf_at(xc0)
    occ = res["occupations"]
    assert bool(((occ > 1e-4) & (occ < 2.0 - 1e-4)).any()), \
        "test premise: fractional occupations must be present"
    rho_ref = (0.95 * res["rho"]).detach().clone()

    def loss_fn(rho):
        d = rho - rho_ref
        return (d * d).sum()

    loss, grads = uspp_density_loss_param_grads(res, xc0, loss_fn)

    h = 2e-3
    for name in ("raw_mu", "raw_kappa"):
        vals = []
        for sgn in (+1, -1):
            xc = LearnableX()
            with torch.no_grad():
                getattr(xc, name).add_(sgn * h)
            vals.append(float(loss_fn(scf_at(xc)["rho"])))
        fd = (vals[0] - vals[1]) / (2 * h)
        an = float(grads[name])
        rel = abs(an - fd) / max(abs(fd), 1e-30)
        assert rel < 2e-4, \
            f"dL/d({name}) adjoint {an} vs FD {fd} (rel {rel:.2e})"


@pytest.mark.slow
def test_paw_density_loss_grads_ibz_equals_full():
    """IBZ (use_symmetry=True) adjoint == full-mesh adjoint. The transposed
    symmetrized SCF map applies the self-adjoint symmetrizers to u before
    each response, and on the symmetric subspace the weighted-IBZ response
    is the full-BZ response — so the gradients must agree to solver
    tolerance, at ~n_IBZ/n_full of the Sternheimer cost (3/8 here).

    The FFT box must be commensurate with the non-symmorphic translations
    for this to gate tightly: diamond's (1/4,1/4,1/4) glide maps the
    real-space grid onto itself only when dims % 4 == 0. ecut 25/100 gives
    24^3 (commensurate: full-vs-IBZ densities agree to 2e-9); at 18^3 the
    un-projected full-mesh fixed point retains a genuine 2e-4 asymmetric
    component from pointwise XC on the glide-incommensurate grid, and the
    two adjoints correctly differ at that (state) level."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    pos = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])
    xc = LearnableX()

    def scf_at(use_sym, fft_shape=None):
        s = setup_uspp(SI_CELL, pos, [0, 0], [paw], ecut=25 * RY,
                       kmesh=(2, 2, 2), ecutrho=100 * RY,
                       use_symmetry=use_sym, fft_shape=fft_shape)
        r = scf_uspp(s, xc, etol=1e-12, rhotol=1e-10, verbose=False,
                     max_iter=80)
        assert r["converged"]
        return r

    res_sym = scf_at(True)
    shape = tuple(res_sym["system"].grid.shape)
    res_full = scf_at(False, fft_shape=shape)
    assert len(res_sym["system"].spheres) < len(res_full["system"].spheres)

    rho_ref = (0.95 * res_sym["rho"]).detach().clone()

    def loss_fn(rho):
        d = rho - rho_ref
        return (d * d).sum()

    _, g_full = uspp_density_loss_param_grads(res_full, xc, loss_fn)
    _, g_sym = uspp_density_loss_param_grads(res_sym, xc, loss_fn)
    for name in ("raw_mu", "raw_kappa"):
        a, b = float(g_full[name]), float(g_sym[name])
        rel = abs(a - b) / max(abs(a), 1e-30)
        assert rel < 1e-6, f"{name}: full {a} vs IBZ {b} (rel {rel:.2e})"
