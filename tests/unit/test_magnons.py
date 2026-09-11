"""Linear spin-wave theory (postscf/magnons.py) against closed-form analytic
dispersions. Pure boson algebra — no SCF — so these run in the fast gate and are
the real specification of the LSWT assembly: every case below has an exact
textbook ω(q) that the general Colpa/Bogoliubov path must reproduce to machine
precision.

Convention (module docstring): H = -½ Σ_{ij,R} Ŝ_iᵀ𝒥_ij(R)Ŝ_j - Σ_i Ŝ_iᵀA_iŜ_i,
J>0 ferromagnetic, K>0 easy-axis, DM 𝒟_ab = Σ_c ε_abc D_c. Frequencies come back
in eV (same unit as the input couplings) from ``magnon_dispersion``."""

import numpy as np
import pytest
import torch

from gradwave.postscf.magnons import (
    ExchangeBond,
    HeisenbergModel,
    MagnonInstabilityError,
    magnon_bands,
    magnon_dispersion,
    spin_wave_stiffness,
)


def _fm_chain(j=0.010, s=1.0, a=1.0, k=0.0):
    """1D ferromagnetic chain: one sublattice, NN coupling J to ±â (both
    directions listed), spin S, optional easy-axis single-ion K."""
    cell = np.diag([a, 10.0, 10.0])
    bonds = [ExchangeBond(0, 0, (1, 0, 0), j), ExchangeBond(0, 0, (-1, 0, 0), j)]
    return HeisenbergModel(cell=cell, spins=[s], bonds=bonds, anisotropy_k=[k])


# (a) Goldstone mode ------------------------------------------------------------
def test_fm_goldstone_gapless_without_anisotropy():
    m = _fm_chain(k=0.0)
    # ω(q→0) → 0: sweep toward Γ and check the acoustic branch vanishes
    qs = np.array([[t, 0, 0] for t in (0.05, 0.01, 0.002)])
    w = magnon_dispersion(m, qs)[:, 0]
    assert w[0] > w[1] > w[2]
    assert w[-1] < 1e-4  # essentially zero at the smallest q


def test_fm_anisotropy_opens_a_gap():
    k = 0.002  # eV
    s = 1.0
    m = _fm_chain(k=k, s=s)
    w0 = magnon_dispersion(m, np.zeros((1, 3)))[0, 0]
    # single-ion easy-axis gap = 2 K S (linear spin-wave theory)
    assert abs(w0 - 2.0 * k * s) < 1e-9


# (b) FM closed forms -----------------------------------------------------------
def test_fm_chain_matches_analytic_cosine():
    j, s, a = 0.010, 1.5, 1.0
    m = _fm_chain(j=j, s=s, a=a)
    qs = np.array([[t, 0, 0] for t in np.linspace(0, 0.5, 9)])
    w = magnon_dispersion(m, qs)[:, 0]
    # ω(q) = 2 J S (1 − cos qa),  qa = 2π q_frac
    expect = 2.0 * j * s * (1.0 - np.cos(2 * np.pi * qs[:, 0] * a))
    assert np.allclose(w, expect, atol=1e-10)


def test_fm_simple_cubic_matches_analytic():
    j, s, a = 0.008, 1.0, 2.5
    cell = np.diag([a, a, a])
    rs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    bonds = [ExchangeBond(0, 0, r, j) for r in rs]
    m = HeisenbergModel(cell=cell, spins=[s], bonds=bonds)
    rng = np.random.default_rng(0)
    qs = rng.uniform(-0.5, 0.5, size=(20, 3))
    w = magnon_dispersion(m, qs)[:, 0]
    # ω(q) = 2 S J (3 − cos qx a − cos qy a − cos qz a),  q_i a = 2π q_frac,i
    c = np.cos(2 * np.pi * qs)
    expect = 2.0 * s * j * (3.0 - c.sum(axis=1))
    assert np.allclose(w, expect, atol=1e-10)


def test_fm_stiffness_matches_analytic_cubic():
    # small-q stiffness of the simple-cubic FM: ω ≈ (S J a²) q²  (from the
    # 2SJ(3−Σcos) expansion, isotropic), i.e. D = S J a² in meV·Å² with J in eV.
    j, s, a = 0.008, 1.0, 2.5
    cell = np.diag([a, a, a])
    rs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    bonds = [ExchangeBond(0, 0, r, j) for r in rs]
    m = HeisenbergModel(cell=cell, spins=[s], bonds=bonds)
    d = spin_wave_stiffness(m, direction=(1, 0, 0), qmax_frac=0.01)
    expect = s * j * a**2 * 1.0e3  # eV -> meV
    assert abs(d - expect) / expect < 1e-3


# (c) AFM linear dispersion -----------------------------------------------------
def test_afm_chain_linear_dispersion():
    # 1D bipartite antiferromagnet: cell = 2a, A at 0 (+z), B at a (−z), NN J<0.
    jval, s, a = -0.010, 1.0, 1.0
    cell = np.diag([2 * a, 10.0, 10.0])
    moments = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    # A(0)'s neighbours: B in same cell (+a, R=0) and B in previous cell (−a, R=-1)
    bonds = [
        ExchangeBond(0, 1, (0, 0, 0), jval), ExchangeBond(0, 1, (-1, 0, 0), jval),
        ExchangeBond(1, 0, (0, 0, 0), jval), ExchangeBond(1, 0, (1, 0, 0), jval),
    ]
    m = HeisenbergModel(cell=cell, spins=[s, s], bonds=bonds, moments=moments)
    qs = np.array([[t, 0, 0] for t in np.linspace(0.0, 0.5, 11)])
    w = magnon_dispersion(m, qs)  # (nq, 2)
    # analytic ω(q) = 2|J| S |sin(ka)| with ka = π q_frac  (cell = 2a)
    expect = 2.0 * abs(jval) * s * np.abs(np.sin(np.pi * qs[:, 0]))
    lower = w.min(axis=1)
    assert np.allclose(lower, expect, atol=1e-9)
    # linear near Γ: slope finite, ω(0) = 0 (Goldstone)
    assert lower[0] < 1e-6
    slope = (lower[1] - lower[0]) / (qs[1, 0] - qs[0, 0])
    assert slope > 1e-3  # linear, not quadratic (would be ~0 slope at Γ)


# (d) DM nonreciprocity ---------------------------------------------------------
def _dm_chain(j=0.010, s=1.0, dz=0.0, a=1.0):
    """1D FM chain (moments ∥ ẑ) with a DM vector D = dz ẑ along the bonds.
    D ∥ moments leaves the collinear FM stationary (D·(S×S)=0 classically) but
    tilts the magnon dispersion, giving nonreciprocity ω(q) ≠ ω(−q)."""
    cell = np.diag([a, 10.0, 10.0])
    bonds = [
        ExchangeBond(0, 0, (1, 0, 0), j, dm=(0.0, 0.0, dz)),
        ExchangeBond(0, 0, (-1, 0, 0), j, dm=(0.0, 0.0, -dz)),  # reversed bond: −D
    ]
    return HeisenbergModel(cell=cell, spins=[s], bonds=bonds)


def test_dm_breaks_reciprocity():
    j, s, dz = 0.010, 1.0, 0.003
    m = _dm_chain(j=j, s=s, dz=dz)
    qs = np.array([0.1, 0.2, 0.35])
    wp = magnon_dispersion(m, np.stack([qs, 0 * qs, 0 * qs], axis=1))[:, 0]
    wm = magnon_dispersion(m, np.stack([-qs, 0 * qs, 0 * qs], axis=1))[:, 0]
    # nonreciprocal: ω(q) − ω(−q) = 4 D S sin(qa) (analytic), qa = 2π q_frac
    delta = wp - wm
    expect = 4.0 * dz * s * np.sin(2 * np.pi * qs)
    assert np.allclose(delta, expect, atol=1e-9)
    assert np.abs(delta).max() > 1e-4  # genuinely nonreciprocal


def test_dm_zero_restores_reciprocity():
    m = _dm_chain(dz=0.0)
    qs = np.array([0.1, 0.2, 0.35])
    wp = magnon_dispersion(m, np.stack([qs, 0 * qs, 0 * qs], axis=1))[:, 0]
    wm = magnon_dispersion(m, np.stack([-qs, 0 * qs, 0 * qs], axis=1))[:, 0]
    assert np.allclose(wp, wm, atol=1e-12)


# (e) Colpa positive-definiteness guard ----------------------------------------
def test_unstable_state_raises_instability_error():
    # FM directions imposed on an antiferromagnetic coupling (J<0): the collinear
    # FM is a maximum, not a minimum → the grand matrix is not positive definite
    # away from Γ → MagnonInstabilityError.
    jval, s = -0.010, 1.0
    cell = np.diag([1.0, 10.0, 10.0])
    bonds = [ExchangeBond(0, 0, (1, 0, 0), jval), ExchangeBond(0, 0, (-1, 0, 0), jval)]
    m = HeisenbergModel(cell=cell, spins=[s], bonds=bonds)  # moments default +z (FM)
    with pytest.raises(MagnonInstabilityError, match="not positive definite"):
        magnon_dispersion(m, np.array([[0.5, 0, 0]]))


# from_shells + band path -------------------------------------------------------
def test_from_shells_expands_reverse_bonds():
    # one shell of the +x bond expands to both directions
    m = HeisenbergModel.from_shells(
        np.diag([1.0, 10.0, 10.0]), [1.0],
        [{"i": 0, "j": 0, "rs": [(1, 0, 0)], "j_iso": 0.01}])
    assert len(m.bonds) == 2
    assert {b.r for b in m.bonds} == {(1, 0, 0), (-1, 0, 0)}


def test_magnon_bands_dataclass_shape_and_units():
    m = _fm_chain(j=0.010, s=1.0)
    bs = magnon_bands(m, npoints=50)
    assert bs.frequencies.shape[1] == 1
    assert bs.frequencies.shape[0] == len(bs.qpts_frac)
    assert bs.labels and bs.x is not None
    # meV: 2·J·S·(1−cos) peaks at 4·J·S = 40 meV for J=10 meV, S=1
    assert bs.frequencies.max() == pytest.approx(4.0 * 10.0, abs=1e-3)


def test_magnons_task_end_to_end(tmp_path):
    # the full numbers-in task: YAML -> api.run -> magnons.json, no SCF. A simple
    # cubic FM (bcc-Fe-like lattice, J1 only) exercises the driver, summary block
    # and stiffness report.
    from gradwave.api import run
    from gradwave.inputs import load_input

    body = """
structure:
  cell: [[2.87, 0, 0], [0, 2.87, 0], [0, 0, 2.87]]
  positions: {cart: [[0, 0, 0]]}
  species: [Fe]
task: magnons
magnons:
  spins: [1.1]
  bonds:
    - {i: 0, j: 0, r: [1, 0, 0], j_iso: 15.0}
    - {i: 0, j: 0, r: [0, 1, 0], j_iso: 15.0}
    - {i: 0, j: 0, r: [0, 0, 1], j_iso: 15.0}
  npoints: 80
output:
  dir: out
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    inp = load_input(p)
    summary = run(inp, verbose=False)
    mg = summary["magnons"]
    assert mg["n_sublattices"] == 1
    assert mg["min_frequency_meV"] < 1e-2  # gapless FM Goldstone
    # stiffness of a NN simple-cubic FM: D = S J a² = 1.1·15·2.87² meV·Å²
    expect_d = 1.1 * 15.0 * 2.87**2
    assert abs(mg["stiffness_meV_A2"] - expect_d) / expect_d < 5e-3
    assert (tmp_path / "out" / "magnons.json").exists()


def test_dispersion_is_differentiable_free_of_nans():
    # a moderately complex 2-sublattice case runs clean end to end
    m = HeisenbergModel.from_shells(
        np.diag([3.0, 3.0, 6.0]), [1.2, 1.2],
        [{"i": 0, "j": 1, "rs": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "j_iso": 0.02}],
        moments=np.array([[0, 0, 1.0], [0, 0, 1.0]]))
    w = magnon_dispersion(m, np.array([[0.1, 0.1, 0.0], [0.25, 0.0, 0.0]]))
    assert np.all(np.isfinite(w)) and np.all(w >= -1e-12)
    assert not isinstance(w, torch.Tensor)
