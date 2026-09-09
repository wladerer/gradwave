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
