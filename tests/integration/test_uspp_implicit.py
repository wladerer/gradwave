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
def test_spin_degenerate_density_loss_matches_nspin1():
    """nspin=2 adjoint in the degenerate limit: a broken-symmetry-free spin
    SCF on Si (m = 0) must give the same dL/dθ as the nspin=1 adjoint on
    the identical smeared configuration — the per-spin Sternheimer, spin
    K_Hxc HVP, and spin one-center HVP collapse to the unpolarized
    machinery when the channels are equal."""
    from gradwave.core.xc.learnable import LearnableSpinX

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    pos = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])
    kw = dict(smearing="gaussian", width=0.1, etol=1e-12, rhotol=1e-10,
              verbose=False, max_iter=80)

    def system():
        return setup_uspp(SI_CELL, pos, [0, 0], [paw], ecut=15 * RY,
                          kmesh=(2, 2, 2), ecutrho=60 * RY)

    xc1 = LearnableX()
    res1 = scf_uspp(system(), xc1, **kw)
    assert res1["converged"]
    xc2 = LearnableSpinX()
    res2 = scf_uspp(system(), xc2, nspin=2, start_mag=[0.0], **kw)
    assert res2["converged"]
    assert abs(res2["mag_total"]) < 1e-8
    assert abs(float(res1["energies"].free_energy)
               - float(res2["energies"].free_energy)) < 1e-6

    rho_ref = (0.95 * res1["rho"]).detach().clone()

    def loss_fn(rho):
        d = rho - rho_ref
        return (d * d).sum()

    _, g1 = uspp_density_loss_param_grads(res1, xc1, loss_fn)
    _, g2 = uspp_density_loss_param_grads(res2, xc2, loss_fn)
    for name in ("raw_mu", "raw_kappa"):
        a, b = float(g1[name]), float(g2[name])
        rel = abs(a - b) / max(abs(a), 1e-30)
        assert rel < 1e-5, f"{name}: nspin=1 {a} vs nspin=2 {b} (rel {rel:.2e})"


@pytest.mark.slow
def test_spin_o2_grads_vs_fd():
    """Real-moment spin adjoint: O₂ triplet (m = 2, integer occupations,
    Γ-only) — dE/dθ by stationarity AND density-loss dL/dθ through the
    composite spin adjoint, both against central FD of full nspin=2 SCF
    re-runs (the same two re-runs feed both gates). Exercises the per-spin
    Sternheimer, the spin f_xc HVP with genuinely different channels, and
    the cross-spin one-center Hessian blocks; the δμ Fermi-surface coupling
    stays dormant here (integer occupations) and gets its own FM-metal
    gate."""
    from gradwave.core.xc.learnable import LearnableSpinX

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "O.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell = 6.0 * np.eye(3)
    pos = np.array([[3.0, 3.0, 2.40], [3.06, 3.0, 3.75]])

    def scf_at(xc):
        # 25 Ry limit-cycles at |Δρ| ~1e-3; 35/280 converges cleanly but
        # the molecular noise floor sits at |Δρ| ~1.5e-6 and F wanders at
        # ~5e-8 eV, so the flag must come from the energy criterion at an
        # etol above that noise (the 18-iteration tail is then clean)
        s = setup_uspp(cell, pos, [0, 0], [paw], ecut=35 * RY,
                       kmesh=(1, 1, 1), ecutrho=280 * RY, nbands=10)
        r = scf_uspp(s, xc, nspin=2, start_mag=[0.5], smearing="gaussian",
                     width=0.01 * RY, etol=3e-7, criterion="energy",
                     rhotol=1e-9, verbose=False, max_iter=90)
        assert r["converged"]
        assert abs(r["mag_total"] - 2.0) < 1e-2
        return r

    xc0 = LearnableSpinX()
    res = scf_at(xc0)
    g_e = uspp_energy_param_grads(res, xc0)

    rho_ref = (0.95 * res["rho"]).detach().clone()

    def loss_fn(rho):
        d = rho - rho_ref
        return (d * d).sum()

    # history 40: the vacuum spin-f_xc broadens the Kχ̃ spectrum (max|K|
    # sits at the density-floor shell covering half the box) and
    # restarted-Anderson(8) stagnates at 1e-2; near-unrestarted Anderson
    # is GMRES-like on this LINEAR system and goes through. cg_tol must
    # be 1e-10: it is an ABSOLUTE residual norm and the kernel amplifies
    # |u| by ~1e3, so 1e-8 floors the outer loop at ~2e-6 relative.
    _, g_l = uspp_density_loss_param_grads(
        res, xc0, loss_fn, history=40, beta=0.3, max_outer=300,
        outer_tol=2e-6, cg_tol=1e-10)

    h = 5e-3  # FD signal ≫ the 5e-8 molecular F-noise (rel noise ~3e-5)
    es, ls = [], []
    for sgn in (+1, -1):
        xc = LearnableSpinX()
        with torch.no_grad():
            xc.raw_mu.add_(sgn * h)
        r = scf_at(xc)
        es.append(float(r["energies"].free_energy))
        ls.append(float(loss_fn(r["rho"])))
    fd_e = (es[0] - es[1]) / (2 * h)
    fd_l = (ls[0] - ls[1]) / (2 * h)
    rel_e = abs(float(g_e["raw_mu"]) - fd_e) / abs(fd_e)
    rel_l = abs(float(g_l["raw_mu"]) - fd_l) / abs(fd_l)
    # observed: rel_e 2.6e-7, rel_l 2.2e-5
    assert rel_e < 1e-5, f"dE/dθ {float(g_e['raw_mu'])} vs FD {fd_e} " \
                         f"(rel {rel_e:.2e})"
    assert rel_l < 2e-4, f"dL/dθ {float(g_l['raw_mu'])} vs FD {fd_l} " \
                         f"(rel {rel_l:.2e})"


@pytest.mark.torture
def test_fm_ni_density_loss_grads_vs_fd():
    """The cross-spin δμ coupling — the one piece of the spin adjoint the
    O₂ gate cannot see (integer occupations there). FM Ni is a smeared
    metal with a real Fermi surface in BOTH channels: δμ's particle-
    conservation sums run over both spins and feed back into each, so an
    FD match here validates channels (a)+(b)+(c) of the metallic response
    in the spin-polarized case. ~2 h on 8 cores; run when the spin
    response machinery changes."""
    from gradwave.core.xc.learnable import LearnableSpinX

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF")
    cell = np.array([[0.0, 1.76, 1.76], [1.76, 0.0, 1.76],
                     [1.76, 1.76, 0.0]])

    def scf_at(xc):
        # nbands 24 (not the SCF-sufficient 18): the adjoint needs the top
        # computed band free of occupation/Fermi-surface weight
        s = setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=50 * RY,
                       kmesh=(4, 4, 4), ecutrho=400 * RY, nbands=24)
        r = scf_uspp(s, xc, nspin=2, start_mag=[0.8], smearing="gaussian",
                     width=0.1, etol=1e-6, criterion="energy",
                     mixing_scheme="johnson", verbose=False, max_iter=120)
        assert r["converged"]
        assert abs(r["mag_total"]) > 0.1, "NM collapse — seed/trajectory"
        return r

    xc0 = LearnableSpinX()
    res = scf_at(xc0)
    rho_ref = (0.95 * res["rho"]).detach().clone()

    def loss_fn(rho):
        d = rho - rho_ref
        return (d * d).sum()

    _, g_l = uspp_density_loss_param_grads(
        res, xc0, loss_fn, history=40, beta=0.3, max_outer=300,
        outer_tol=1e-6, cg_tol=1e-10)

    h = 5e-3
    vals = []
    for sgn in (+1, -1):
        xc = LearnableSpinX()
        with torch.no_grad():
            xc.raw_mu.add_(sgn * h)
        vals.append(float(loss_fn(scf_at(xc)["rho"])))
    fd = (vals[0] - vals[1]) / (2 * h)
    an = float(g_l["raw_mu"])
    rel = abs(an - fd) / abs(fd)
    # gate reflects the FM-metal SCF noise floor (etol 1e-6, occupation
    # plateau); a broken δμ channel misses by orders of magnitude
    assert rel < 5e-3, f"dL/dθ adjoint {an} vs FD {fd} (rel {rel:.2e})"


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
