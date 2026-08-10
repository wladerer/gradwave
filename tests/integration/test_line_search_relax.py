"""Parallel line-search relax reaches the SAME minimum as a serial BFGS relax.

The parallel/adaptive line search only changes the PATH: it evaluates several
step lengths per ionic step in worker processes and interpolates the best, but
the accepted geometry is always re-evaluated by the main calculator at full SCF
tolerance. So a ``parallel``/``adaptive`` relax must land on the same relaxed
geometry and energy as a plain serial BFGS relax, to the geometry/energy
tolerance.

Three regimes, chosen to exercise the feature where a fixed step is most likely
wrong:
  * ``bulk_metal``  — rattled fcc Al (smooth, easy; the adaptive trigger may stay
    dormant here — recorded, not required),
  * ``adsorbate``   — an H adatom rattled off-site on a small Al slab (a soft
    lateral mode → overshoot-prone; adaptive is expected to fire),
  * ``mol_crystal`` — CO₂ molecules rattled off equilibrium in a small box (soft
    intermolecular modes; adaptive is expected to fire).

MARKED SLOW and NOT run in CI or the fast gate — each is a full SCF-in-the-loop
relax plus its parallel candidates. The parent validates these on asus. Each is
kept minimal (small cell, few k-points, modest ecut, ~8 threads) to converge in
bounded time. FORWARD-ONLY: the candidates fan out over spawned worker processes.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from ase.build import fcc111, molecule

from tests.helpers import RY, pseudo

AL = pseudo("Al_ONCV_PBE-1.2.upf")
H = pseudo("H_ONCV_PBE-1.2.upf")
C = pseudo("C_ONCV_PBE-1.2.upf")
O = pseudo("O_ONCV_PBE-1.2.upf")


# --- system builders ----------------------------------------------------------


def _bulk_metal():
    """Rattled fcc-Al primitive cell (a smooth metal — the easy baseline)."""
    from ase import Atoms

    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    rng = np.random.default_rng(0)
    atoms = Atoms("Al", positions=rng.normal(0, 0.06, (1, 3)), cell=cell, pbc=True)
    pseudos = {"Al": AL}
    kpts = (4, 4, 4)
    return atoms, pseudos, kpts, dict(smearing="gaussian", width=0.1), True


def _adsorbate():
    """H adatom rattled off a hollow site on a small 2×2, 3-layer Al(111) slab
    with a modest vacuum gap — a soft lateral-mode relax path."""
    slab = fcc111("Al", size=(2, 2, 3), a=4.05, vacuum=6.0)
    # place an H above the surface and rattle it laterally off the symmetric site
    top_z = slab.get_positions()[:, 2].max()
    cell = slab.cell
    xy = 0.5 * (cell[0] + cell[1])[:2]  # roughly a high-symmetry lateral spot
    from ase import Atom

    slab.append(Atom("H", position=(xy[0] + 0.6, xy[1] + 0.4, top_z + 1.1)))
    pseudos = {"Al": AL, "H": H}
    kpts = (2, 2, 1)
    return slab, pseudos, kpts, dict(smearing="gaussian", width=0.1), False


def _mol_crystal():
    """Two CO₂ molecules rattled off equilibrium in a small box — soft
    intermolecular modes."""
    from ase import Atoms

    m = molecule("CO2")
    a = Atoms(cell=[7.0, 7.0, 7.0], pbc=True)
    for shift in ([1.5, 1.5, 1.5], [4.5, 4.0, 4.5]):
        mm = m.copy()
        mm.translate(shift)
        a += mm
    rng = np.random.default_rng(1)
    a.set_positions(a.get_positions() + rng.normal(0, 0.08, a.get_positions().shape))
    pseudos = {"C": C, "O": O}
    kpts = (1, 1, 1)
    return a, pseudos, kpts, dict(smearing="none", width=0.0), False


CASES = {
    "bulk_metal": (_bulk_metal, True),   # (builder, adaptive-may-stay-dormant)
    "adsorbate": (_adsorbate, False),
    "mol_crystal": (_mol_crystal, False),
}


# --- input plumbing -----------------------------------------------------------


def _make_input(atoms, pseudos, kpts, smearing, *, line_search, out_dir):
    """A relax ``Input`` for ``atoms`` with the given line-search mode."""
    from gradwave.inputs import load_input

    pseudo_dir = os.path.dirname(next(iter(pseudos.values())))
    pmap = "{" + ", ".join(f"{s}: {os.path.basename(p)}"
                           for s, p in pseudos.items()) + "}"
    sm = smearing.get("smearing", "none")
    body = f"""
structure:
  cell: {np.asarray(atoms.cell.array).tolist()}
  positions:
    cart: {atoms.get_positions().tolist()}
  species: {list(atoms.get_chemical_symbols())}
pseudopotentials:
  dir: {pseudo_dir}
  map: {pmap}
ecut: {18 * RY}
xc: pbe
symmetry: false
task: relax
kpoints:
  mesh: {list(kpts)}
smearing:
  type: {sm}
  width: {smearing.get("width", 0.0)}
relax:
  optimizer: bfgs
  fmax: 0.03
  max_steps: 40
  line_search: {line_search}
  line_search_n_samples: 4
  line_search_n_workers: 2
scf:
  etol: 1.0e-8
  rhotol: 1.0e-7
  max_iter: 120
output:
  dir: {out_dir}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body)
        path = f.name
    return load_input(path)


def _run(atoms, pseudos, kpts, smearing, mode, tmp_path):
    from gradwave.api import run_relax

    inp = _make_input(atoms.copy(), pseudos, kpts, smearing,
                      line_search=mode, out_dir=str(tmp_path / mode))
    relax, out_atoms, _frames = run_relax(inp, verbose=False)
    return relax, out_atoms


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["parallel", "adaptive"])
@pytest.mark.parametrize("case", list(CASES))
def test_line_search_reaches_same_minimum(case, mode, tmp_path):
    torch.set_num_threads(8)
    builder, adaptive_may_be_dormant = CASES[case]
    atoms, pseudos, kpts, smearing, compare_positions = builder()

    ref_relax, ref_atoms = _run(atoms, pseudos, kpts, smearing, "off", tmp_path)
    ls_relax, ls_atoms = _run(atoms, pseudos, kpts, smearing, mode, tmp_path)

    assert ref_relax["converged"], f"{case}: serial reference did not converge"
    assert ls_relax["converged"], f"{case}/{mode}: line-search relax did not converge"

    # SAME MINIMUM: energies agree to well under a meV (the accepted geometry is
    # re-evaluated at full SCF tol either way — the line search only changes path)
    assert ls_relax["energy_eV"] == pytest.approx(ref_relax["energy_eV"], abs=2e-3)
    assert ls_relax["fmax_eV_ang"] < 0.03 + 1e-9

    if compare_positions:
        # bulk metal: a single atom, no rotational/translational ambiguity
        dmax = float(np.abs(
            ls_atoms.get_positions() - ref_atoms.get_positions()).max())
        assert dmax < 5e-2, f"{case}/{mode}: geometry drift {dmax:.3f} Å"

    # ADAPTIVE trigger telemetry for the parent to inspect: soft cases should
    # fire (energy increases / stalls under overshoot); the easy bulk metal may
    # stay dormant. Recorded in the relax block by _relax_nested.
    if mode == "adaptive":
        searched = ls_relax.get("line_search_steps_searched", 0)
        print(f"[line-search] {case}/adaptive fired on {searched} ionic step(s) "
              f"of {ls_relax['n_steps']}")
        assert ls_relax.get("line_search_active") is True
        if not adaptive_may_be_dormant:
            assert searched >= 1, (
                f"{case}/adaptive: trigger never fired on a soft-mode relax "
                "(expected energy-increase / stall)")
    else:  # parallel: every step fans out
        assert ls_relax.get("line_search_steps_searched", 0) >= 1
