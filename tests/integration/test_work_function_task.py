"""Work function reachable from the YAML input, end to end (io catch-up).

`postscf.work_function` computed Φ = E_vac − E_F from a slab's plane-averaged
potential but had no input key or summary block. This gate covers the wiring: an
open-boundary (ESM) SCF auto-emits a `work_function` block, the summary/report
carry it, and `both_faces` resolves the two vacuum levels of a dipolar slab.

NaH is an ionic dimer with a genuine z-dipole (mirrors tests/integration/
test_esm_scf.py), so the two faces have distinct vacuum levels — a physical
check that the plane-averaging picks up the surface dipole.
"""

from pathlib import Path

import pytest

from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # a small open-boundary SCF + plane average


def _nah_input(tmp_path: Path, extra: str = ""):
    from gradwave.inputs import load_input

    body = f"""
structure:
  cell: [[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 16.0]]
  positions:
    cart: [[3.5, 3.5, 6.5], [3.5, 3.5, 9.0]]
  species: [Na, H]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Na: Na_ONCV_PBE_sr.upf, H: H_ONCV_PBE-1.2.upf}}
ecut: {24 * RY}
xc: pbe
kpoints:
  mesh: [1, 1, 1]
smearing:
  type: fermi-dirac
  width: 0.1
scf:
  boundary: open_z
  max_iter: 80
  etol: 1.0e-6
  rhotol: 1.0e-5
{extra}
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_open_boundary_auto_emits_work_function(tmp_path):
    import torch

    from gradwave.api import run

    torch.set_num_threads(8)
    inp = _nah_input(tmp_path)
    # not explicitly enabled — the open_z boundary auto-emits the block
    assert not inp.work_function.enabled and inp.scf.boundary == "open_z"

    summary = run(inp, verbose=False)
    assert summary["scf"]["converged"]
    wf = summary["work_function"]
    assert wf["available"]
    assert wf["open_axis"] == 2
    # Φ = E_vac − E_F is a finite, physically sane work function (a few eV)
    phi = wf["work_function_eV"]
    evac = wf["vacuum_level_eV"]
    efermi = wf["fermi_eV"]
    assert phi == pytest.approx(evac - efermi, abs=1e-6)
    # UNIT + MAGNITUDE pin (regression, not a literature value): NaH is not a
    # standard work-function reference, but the geometry/pseudos are fixed, so Φ is
    # a well-defined number in eV. A Ry (×13.6) or Ha (×27.2) unit slip in E_vac or
    # E_F — the classic driver bug this gate must catch — would throw Φ far outside
    # this window; the loose 0<Φ<12 admitted any material and could not. The two
    # ingredients are likewise single-digit-eV, not tens (another unit tell).
    assert phi == pytest.approx(5.13, abs=0.6)
    assert -15.0 < efermi < 5.0
    assert -10.0 < evac < 10.0
    # electrode potential on the SHE scale is reported
    assert wf["potential_vs_she_V"] == pytest.approx(
        wf["potential_vs_vacuum_V"] - wf["u_she_abs_V"], abs=1e-6)
    # the human report carries the section
    assert "work function" in (tmp_path / "scf.out").read_text()


def test_both_faces_split_for_dipolar_slab(tmp_path):
    import torch

    from gradwave.api import run

    torch.set_num_threads(8)
    inp = _nah_input(tmp_path, extra="work_function:\n  both_faces: true\n")
    assert inp.work_function.enabled and inp.work_function.both_faces

    summary = run(inp, verbose=False)
    wf = summary["work_function"]
    assert wf["available"]
    evac = wf["vacuum_level_eV"]
    assert isinstance(evac, list) and len(evac) == 2
    # NaH points an ionic dipole along z → the two vacuum levels differ. Pin both
    # the MAGNITUDE and the SIGN of the surface-dipole step, not just inequality:
    #  - positions put Na⁺ at low z (z=6.5) and H⁻ at high z (z=9.0), so the
    #    electron effective potential (v_eff) is raised on the electron-rich H⁻
    #    face → the HIGH-z face vacuum level exceeds the LOW-z face one. Ordering
    #    of vacuum_level_eV is (low-z, high-z), so evac[1] > evac[0]. A dipole-sign
    #    flip in the plane-averaging/face-splitting would invert this.
    #  - the split size (~3.1 eV) is the physical dipole step; a collapse toward 0
    #    (dipole not resolved) or a wildly different value would fail the abs band.
    split = evac[1] - evac[0]
    assert split == pytest.approx(3.14, abs=0.6)   # signed → pins sign AND size
    # per-face Φ = E_vac(face) − E_F is internally consistent, and each face is a
    # physically sane few-eV work function.
    fermi = wf["fermi_eV"]
    phi = wf["work_function_eV"]
    assert isinstance(phi, list) and len(phi) == 2
    for i in range(2):
        assert phi[i] == pytest.approx(evac[i] - fermi, abs=1e-6)
        assert 0.0 < phi[i] < 12.0


def _nah_input_lz(tmp_path: Path, lz: float):
    """The same NaH slab as `_nah_input`, but with the open-axis box length `lz`
    a free parameter (the slab atoms stay at fixed absolute z, 6.5 and 9.0 Å —
    only the vacuum above H⁻ grows). Tighter tolerances than the wiring test: the
    box-independence gate compares two Φ to the meV, so the SCF must be converged
    well below that."""
    from gradwave.inputs import load_input

    body = f"""
structure:
  cell: [[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, {lz}]]
  positions:
    cart: [[3.5, 3.5, 6.5], [3.5, 3.5, 9.0]]
  species: [Na, H]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Na: Na_ONCV_PBE_sr.upf, H: H_ONCV_PBE-1.2.upf}}
ecut: {24 * RY}
xc: pbe
kpoints:
  mesh: [1, 1, 1]
smearing:
  type: fermi-dirac
  width: 0.1
scf:
  boundary: open_z
  max_iter: 120
  etol: 1.0e-8
  rhotol: 1.0e-7
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / f"in_lz{lz}.yaml"
    p.write_text(body)
    return load_input(p)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEASURED (asus, 2026-09-14): Φ is NOT box-independent to the meV for "
        "this NaH ESM slab. Φ(Lz=16)=5.1319, Φ(Lz=18)=5.1648 eV → |Δ|=33 meV, "
        "and a wider sweep (16/18/20/24/28 Å, dz held at 1/6 Å) drifts "
        "MONOTONICALLY without plateauing: Φ = 5.132/5.165/5.209/5.258/5.309 eV "
        "(~2–13 meV/Å). E_F and BOTH per-face vacuum levels drift the same way, "
        "so this is not a single-face-blend artifact. The `postscf.work_function` "
        "docstring's 'exact because box-independent' is overstated for a dipolar / "
        "diffuse-anion slab at practical vacuum thicknesses — box-independence "
        "holds only in the field-free (ρ→0) limit, which this slab reaches slowly. "
        "This gate encodes the CORRECT few-meV identity; it is xfail(strict) so a "
        "future fix (or a convergence-adequate regime) flips it to XPASS and forces "
        "removal of this marker. Do NOT loosen the tolerance to make it pass green."),
)
def test_work_function_box_independent_across_vacuum_thickness(tmp_path):
    """Φ = E_vac − E_F should be INVARIANT to the vacuum thickness under ESM
    open-boundary electrostatics — the exactness claim in `postscf.work_function`'s
    docstring ("box-independent" vacuum level). This gates that identity directly,
    with no literature number and no convention: run the SAME NaH slab at two
    open-axis box lengths and assert the two work functions agree to the meV.

    The gotcha (prior slab work): changing the box length at fixed ecut re-samples
    the FFT grid spacing dz, which alone injects meV-scale noise into E_vac. So we
    hold dz FIXED — Lz=16 Å with nz=96 and Lz=18 Å with nz=108 both give dz=1/6 Å
    exactly (the natural grids at ecut=24 Ry; pinned here via `fft_shape` so the
    identity is tested, not a grid coincidence). Only the vacuum thickness (where
    ρ≈0) differs, so under true box-independence Φ would move only by SCF-
    convergence noise, not by the ~Å the vacuum grew.

    CURRENT VERDICT: the identity FAILS (see the xfail reason) — Φ drifts ~33 meV
    over a 2 Å box change and keeps drifting out to 28 Å. Kept as a strict-xfail
    self-consistency gate so the audit finding lives in the suite.
    """
    import torch

    from gradwave.api._slab import resolve_slab_box
    from gradwave.api.scf import run_scf
    from gradwave.api.system import _is_uspp, _species_upfs, build_scaled_system
    from gradwave.postscf.work_function import work_function

    torch.set_num_threads(6)

    def phi_at(lz: float, nz: int) -> float:
        inp = _nah_input_lz(tmp_path, lz)
        assert inp.scf.boundary == "open_z"
        _species, upfs, soa = _species_upfs(inp)
        # mirror build_system's NC path exactly, but PIN the FFT box so dz is
        # identical across the two Lz (nx=ny=42 unchanged; nz scales with Lz).
        box = resolve_slab_box(inp, upfs, soa)
        assert box.cell[2, 2] == pytest.approx(lz)
        system = build_scaled_system(
            inp, upfs, _is_uspp(upfs), soa, box.cell, box.positions,
            fft_shape=(42, 42, nz))
        # dz identical between the two boxes is the whole point of the gate
        assert system.grid.cell[2, 2] / system.grid.shape[2] == pytest.approx(
            1.0 / 6.0, abs=1e-9)
        res = run_scf(inp, system=system, verbose=False)
        assert res.converged
        return work_function(res, open_axis=2)

    phi_16 = phi_at(16.0, 96)
    phi_18 = phi_at(18.0, 108)
    # box-independence: the vacuum level (hence Φ) is the same absolute reference
    # regardless of how much vacuum sits above the slab. A few meV covers residual
    # SCF-convergence noise; a materially larger drift would mean the docstring's
    # "exact because box-independent" claim is overstated (report, don't loosen).
    assert abs(phi_16 - phi_18) < 5e-3, (
        f"Φ drifts with vacuum thickness: Φ(Lz=16)={phi_16:.6f} eV, "
        f"Φ(Lz=18)={phi_18:.6f} eV, |Δ|={abs(phi_16 - phi_18) * 1e3:.3f} meV")
