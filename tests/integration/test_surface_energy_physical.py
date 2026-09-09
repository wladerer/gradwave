"""Physical validation of the ``surface_energy`` task on a bonded metal slab.

``tests/integration/test_surface_energy_task.py`` guards the task *plumbing* on a
deliberately non-physical stack of isolated Si atoms (a well-defined thickness
sweep whose γ is meaningless — and in fact negative, correctly, for a stack of
non-interacting atoms). That leaves the physics of the intercept→γ map unguarded:
a sign flip in the Fiorentini-Methfessel intercept, or a wrong ``n_surfaces``
divisor, would still pass every existing test.

This gate closes that gap. It runs the task on a genuinely bonded Al(100) slab —
a real metal surface — over a thickness sweep and asserts γ is *physical*:

* γ > 0 (a stable surface costs energy to create), and
* γ lands in a broad but bounded band around the known Al(100) value.

Reference: PBE Al(100) surface energy is ≈ 0.90–1.00 J/m² in the DFT literature
(e.g. Da Silva et al., PRB 73, 205409 (2006), ~0.99 J/m²; Tran et al., Sci. Data
3, 160080 (2016) surface database, ~0.90 J/m²), i.e. ≈ 0.056–0.062 eV/Å². The
band below is deliberately wide (~±50%) so it is robust to the small cell / low
ecut / coarse k-mesh used to keep the sweep cheap, yet tight enough to reject a
sign error (γ < 0) or a factor-of-two ``n_surfaces`` mistake.

Al PAW (3 valence e⁻) is used rather than the Al ONCV fixture (11 e⁻, 2s2p
semicore) so the sweep stays cheap: fewer bands, lower ecut.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.slow  # a multi-SCF bonded-slab sweep

AL_PAW = "Al.pbe-n-kjpaw_psl.1.0.0.UPF"

# PBE Al(100): ≈ 0.90–1.00 J/m² (≈ 0.056–0.062 eV/Å²) in the literature. Keep a
# broad ~±50% band — robust to the cheap cell/ecut/k-mesh, tight enough to catch
# a sign flip or an n_surfaces factor error.
GAMMA_LO_EV_A2 = 0.030
GAMMA_HI_EV_A2 = 0.100


def _build_input(tmp_path: Path):
    from ase.build import fcc100
    from ase.io import write as ase_write

    from gradwave.inputs import load_input

    a = 4.05  # Å, Al PBE lattice constant
    layers = (3, 4, 5)
    for n in layers:
        slab = fcc100("Al", size=(1, 1, n), a=a, vacuum=8.0)
        ase_write(str(tmp_path / f"al{n}.vasp"), slab, format="vasp")

    # placeholder top-level structure (replaced per-slab by the driver); make it
    # self-consistent — one Al per position of the thinnest slab.
    ref = fcc100("Al", size=(1, 1, layers[0]), a=a, vacuum=8.0)
    cell = np.asarray(ref.cell.array).tolist()
    pos = ref.get_positions().tolist()
    species = list(ref.get_chemical_symbols())
    slab_lines = "\n".join(
        f"    - {{structure: al{n}.vasp, n_layers: {n}}}" for n in layers)
    body = f"""
structure:
  cell: {cell}
  positions:
    cart: {pos}
  species: {species}
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Al: {AL_PAW}}}
ecut: {25 * RY}
xc: pbe
kpoints:
  mesh: [8, 8, 1]
smearing:
  type: gaussian
  width: 0.1
nbands: 16
scf:
  max_iter: 100
  etol: 1.0e-6
  rhotol: 1.0e-5
task: surface_energy
surface_energy:
  n_surfaces: 2
  energy: free_energy
  slabs:
{slab_lines}
output:
  dir: {tmp_path}
"""
    (tmp_path / "in.yaml").write_text(body)
    return load_input(tmp_path / "in.yaml"), layers


def test_al100_surface_energy_is_physical(tmp_path):
    import torch

    from gradwave.api import run
    from gradwave.postscf.surface_energy import (
        surface_energy_fm,
        surface_energy_subtraction,
    )

    torch.set_num_threads(8)
    inp, layers = _build_input(tmp_path)
    assert inp.task == "surface_energy"
    assert len(inp.surface_energy.slabs) == len(layers)

    summary = run(inp, verbose=False)
    se = summary["surface_energy"]

    gamma = se["gamma_eV_ang2"]
    assert np.isfinite(gamma)
    # every slab SCF must have converged, or γ is meaningless
    assert se["all_converged"], "an Al(100) slab SCF did not converge"

    # (1) a real bonded surface costs energy to make: γ MUST be positive.
    # (The toy isolated-atom sweep in test_surface_energy_task.py is negative
    #  and correctly so; a genuinely bonded slab flips the sign.)
    assert gamma > 0.0, (
        f"Al(100) surface energy came out non-positive (γ = {gamma:.5f} "
        f"eV/Å²) — a physical bonded slab must give γ > 0; suspect a sign in "
        f"the FM intercept→γ map or the n_surfaces divisor")

    # (2) γ within a broad physical band for Al(100).
    assert GAMMA_LO_EV_A2 < gamma < GAMMA_HI_EV_A2, (
        f"Al(100) γ = {gamma:.5f} eV/Å² ({se['gamma_J_m2']:.4f} J/m²) is "
        f"outside the physical band [{GAMMA_LO_EV_A2}, {GAMMA_HI_EV_A2}] "
        f"eV/Å² (lit. ≈ 0.056–0.062); suspect a factor/sign bug in the driver")

    # (3) the summary γ is exactly the Fiorentini-Methfessel fit of the returned
    #     sweep (no silent re-scaling between fit and report).
    refit = surface_energy_fm(
        se["n_layers"], se["energies_eV"], se["area_ang2"],
        n_surfaces=se["n_surfaces"])
    assert refit.gamma == pytest.approx(gamma, rel=1e-9)
    assert refit.e_bulk == pytest.approx(se["e_bulk_eV_per_layer"], rel=1e-9)

    # (4) FM-vs-direct-subtraction identity: with the per-layer bulk energy read
    #     off the FM slope, the textbook γ = (E_slab − N·E_bulk)/(n_surf·A) at
    #     each thickness must reproduce the FM γ (up to the fit residual, which
    #     the surface term of a few-layer slab makes non-zero but small).
    e_bulk = se["e_bulk_eV_per_layer"]
    area = se["area_ang2"]
    tol = 5.0 * se["rms_residual_eV"] / (se["n_surfaces"] * area) + 1e-9
    for n, e in zip(se["n_layers"], se["energies_eV"], strict=True):
        g_sub = surface_energy_subtraction(
            e, n, e_bulk, area, n_surfaces=se["n_surfaces"])
        assert g_sub == pytest.approx(gamma, abs=tol)
