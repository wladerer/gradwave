"""Surface energy reachable from the YAML input, end to end (io catch-up).

`postscf.surface_energy.surface_energy_fm` fit γ from a slab-thickness sweep but
had no task/Input surface. This gate covers the `surface_energy` task wiring: a
list of slab geometry files with layer counts is run through the SCF, the
Fiorentini-Methfessel line is fit, and the summary/report carry γ, E_bulk and A.

Uses tiny few-atom Si slabs sharing an in-plane cell (physics is not the point —
the fit accuracy of surface_energy_fm is covered in
tests/unit/test_surface_energy.py). The mismatched-area guard is checked too.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import PSEUDOS, RY, SI_ONCV

pytestmark = pytest.mark.standard  # a small multi-slab SCF sweep


def _write_slab(path: Path, a: float, n_layers: int, d: float = 2.5,
                vac: float = 6.0):
    """A trivial n-atom Si slab: n atoms stacked along z, shared a×a in-plane
    cell, vacuum padding. Not physical — just a well-defined thickness sweep."""
    from ase import Atoms
    from ase.io import write as ase_write

    c = vac + d * n_layers
    pos = [(a / 2, a / 2, vac / 2 + d * i) for i in range(n_layers)]
    atoms = Atoms("Si" * n_layers, positions=pos,
                  cell=[a, a, c], pbc=True)
    ase_write(str(path), atoms, format="vasp")


def _sweep_input(tmp_path: Path, a: float = 3.9):
    from gradwave.inputs import load_input

    layers = [(2, "slab2.vasp"), (3, "slab3.vasp"), (4, "slab4.vasp")]
    for n, name in layers:
        _write_slab(tmp_path / name, a, n)
    body = f"""
structure:
  cell: [[{a}, 0, 0], [0, {a}, 0], [0, 0, 11.0]]
  positions:
    cart: [[{a / 2}, {a / 2}, 3.0], [{a / 2}, {a / 2}, 5.5]]
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: {SI_ONCV}}}
ecut: {12 * RY}
xc: pbe
kpoints:
  mesh: [1, 1, 1]
smearing:
  type: fermi-dirac
  width: 0.2
scf:
  max_iter: 60
  etol: 1.0e-5
  rhotol: 1.0e-4
task: surface_energy
surface_energy:
  n_surfaces: 2
  energy: free_energy
  slabs:
    - {{structure: slab2.vasp, n_layers: 2}}
    - {{structure: slab3.vasp, n_layers: 3}}
    - {{structure: slab4.vasp, n_layers: 4}}
output:
  dir: {tmp_path}
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_surface_energy_sweep_runs_and_fits(tmp_path):
    import torch

    from gradwave.api import run

    torch.set_num_threads(8)
    inp = _sweep_input(tmp_path)
    assert inp.task == "surface_energy" and len(inp.surface_energy.slabs) == 3

    summary = run(inp, verbose=False)
    se = summary["surface_energy"]
    # the fit produced finite γ / E_bulk / area on the shared in-plane cell
    assert np.isfinite(se["gamma_eV_ang2"])
    assert np.isfinite(se["e_bulk_eV_per_layer"])
    assert se["area_ang2"] == pytest.approx(3.9 * 3.9, abs=1e-3)
    assert se["n_layers"] == [2.0, 3.0, 4.0]
    # J/m² conversion is consistent with the eV/Å² value
    assert se["gamma_J_m2"] == pytest.approx(se["gamma_eV_ang2"] * 16.021766, rel=1e-4)
    # the human report carries the section
    assert "surface energy" in (tmp_path / "surface_energy.out").read_text()


def test_mismatched_in_plane_area_raises(tmp_path):
    import torch

    from gradwave.api import run
    from gradwave.inputs import load_input

    torch.set_num_threads(8)
    # two slabs with DIFFERENT in-plane areas → the driver must reject the sweep
    _write_slab(tmp_path / "slabA.vasp", 3.9, 2)
    _write_slab(tmp_path / "slabB.vasp", 4.3, 3)  # different a → different area
    body = f"""
structure:
  cell: [[3.9, 0, 0], [0, 3.9, 0], [0, 0, 11.0]]
  positions:
    cart: [[1.95, 1.95, 3.0], [1.95, 1.95, 5.5]]
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: {SI_ONCV}}}
ecut: {12 * RY}
xc: pbe
kpoints:
  mesh: [1, 1, 1]
smearing:
  type: fermi-dirac
  width: 0.2
scf:
  max_iter: 60
task: surface_energy
surface_energy:
  slabs:
    - {{structure: slabA.vasp, n_layers: 2}}
    - {{structure: slabB.vasp, n_layers: 3}}
output:
  dir: {tmp_path}
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    inp = load_input(p)
    with pytest.raises(ValueError, match="in-plane area"):
        run(inp, verbose=False)
