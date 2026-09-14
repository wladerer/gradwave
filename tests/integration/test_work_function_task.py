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


def _al100_slab_input_lz(tmp_path: Path, lz: float, ecut_ry: float, kmesh: int):
    """A 4-layer Al(100) ESM open_z slab at open-axis box length `lz`, built at a
    CONVERGED setup (`ecut_ry`, dense in-plane `kmesh`×`kmesh`×1). The slab atoms
    stay at fixed absolute z (bottom vacuum pinned); only the clean vacuum above
    the top layer grows with `lz`.

    Al is a clean plateau witness: a metal's valence density decays fast, so once
    ecut and vacuum are converged Φ(Lz) flattens quickly (measured decay length
    ≈3 Å). At a crude ecut the total energy itself is box-dependent and Φ inherits
    a spurious drift — which is why this uses a converged cutoff, not the 24 Ry of
    the wiring tests above.
    """
    from ase.build import fcc100

    from gradwave.inputs import load_input

    slab = fcc100("Al", size=(1, 1, 4), a=4.05, vacuum=0.0, periodic=True)
    slab.center(axis=2, vacuum=0.0)
    cell = slab.get_cell()
    sx, sy = float(cell[0, 0]), float(cell[1, 1])
    pos = slab.get_positions()
    pos[:, 2] += 4.0 - pos[:, 2].min()  # fixed bottom vacuum; slab grows top vacuum
    cart = [[float(p[0]), float(p[1]), float(p[2])] for p in pos]

    body = f"""
structure:
  cell: [[{sx}, 0.0, 0.0], [0.0, {sy}, 0.0], [0.0, 0.0, {lz}]]
  positions:
    cart: {cart}
  species: [Al, Al, Al, Al]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Al: Al_ONCV_PBE-1.2.upf}}
ecut: {ecut_ry * RY}
xc: pbe
kpoints:
  mesh: [{kmesh}, {kmesh}, 1]
smearing:
  type: fermi-dirac
  width: 0.1
scf:
  boundary: open_z
  max_iter: 200
  etol: 1.0e-8
  rhotol: 1.0e-7
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / f"al_lz{lz}.yaml"
    p.write_text(body)
    return load_input(p)


@pytest.mark.slow
def test_work_function_plateaus_when_converged(tmp_path):
    """Φ = E_vac − E_F is box-independent under ESM open_z ONCE CONVERGED — the
    corrected claim in `postscf.work_function`'s docstring. ESM's Green's function
    forces v_H→0 as z→±∞, so a converged slab feels no z-images and Φ has no
    residual dependence on the vacuum thickness Lz. But Φ, like any observable,
    must first be converged w.r.t. ecut AND vacuum thickness: at a crude cutoff the
    total energy itself is box-dependent, and a too-thin vacuum sits on the
    exponential approach to the plateau (measured decay length ≈3 Å). The apparent
    Lz-drift seen at low ecut / thin vacuum is a convergence artifact, NOT an ESM
    electrostatics bug — `esm.py` is unchanged and correct.

    This gates the CORRECT statement directly: at a converged setup (ecut = 60 Ry,
    dense 8×8×1 in-plane k, pinned dz), Φ(Lz) PLATEAUS. We assert the far-field
    slope has collapsed — |Φ(Lz=24) − Φ(Lz=28)| < 30 meV — rather than pinning a
    literature number (Al ONCV PBE is not a Φ reference; a slab-thickness /
    relaxation study would be needed for that).

    Methodology (mirrors the well-behaved eos/elastic FFT-pinning): changing Lz at
    fixed ecut re-samples the FFT grid spacing dz, which alone injects meV-scale
    noise into E_vac. So we pin dz — derive it from the natural grid at Lz=28, then
    fix `fft_shape` at each Lz so the real-space spacing matches to <1e-3 Å. Only
    the vacuum thickness (where ρ→0) differs between the two boxes.

    MEASURED (asus, 2026-09-14, ecut=60 Ry, 8×8×1): Φ = 3.584 / 3.927 / 4.104 /
    4.192 / 4.210 eV at Lz = 16 / 18 / 20 / 24 / 28 Å — the far-field slope
    collapses 172 → 89 → 22 → 4.4 meV/Å as Lz grows, an exponential plateau
    (decay length ≈3 Å) at ≈4.21 eV. |Φ(24) − Φ(28)| ≈ 4.4 meV/Å × 4 Å ≈ 18 meV,
    comfortably inside the 30 meV gate. Contrast the un-converged 24 Ry sweep
    (3.302 / 3.641 / 3.922 / 4.251 / 4.629 eV): it does NOT plateau — ~110 meV/Å,
    |Φ(24) − Φ(28)| ≈ 378 meV — because at that cutoff the total energy itself is
    box-dependent, so the drift is a convergence artifact, not an ESM bug.
    """
    import torch

    from gradwave.api._slab import resolve_slab_box
    from gradwave.api.scf import run_scf
    from gradwave.api.system import _is_uspp, _species_upfs, build_scaled_system
    from gradwave.postscf.work_function import work_function

    torch.set_num_threads(6)
    ecut_ry, kmesh = 60.0, 8

    def build(lz: float, fft_shape=None):
        inp = _al100_slab_input_lz(tmp_path, lz, ecut_ry, kmesh)
        assert inp.scf.boundary == "open_z"
        _species, upfs, soa = _species_upfs(inp)
        box = resolve_slab_box(inp, upfs, soa)
        assert box.cell[2, 2] == pytest.approx(lz)
        system = build_scaled_system(
            inp, upfs, _is_uspp(upfs), soa, box.cell, box.positions,
            fft_shape=fft_shape)
        return inp, system

    # derive a common dz from the natural grid at the largest Lz, then pin it
    _, sys_nat = build(28.0)
    nx, ny, nz_nat = (int(s) for s in sys_nat.grid.shape)
    dz = float(sys_nat.grid.cell[2, 2]) / nz_nat

    def phi_at(lz: float) -> float:
        nz = round(lz / dz)
        inp, system = build(lz, fft_shape=(nx, ny, nz))
        # dz identical (to <1e-3 Å) between the two boxes is the point of the pin
        dz_i = float(system.grid.cell[2, 2]) / int(system.grid.shape[2])
        assert dz_i == pytest.approx(dz, abs=1e-3)
        res = run_scf(inp, system=system, verbose=False)
        assert res.converged
        return work_function(res, open_axis=2)

    phi_24 = phi_at(24.0)
    phi_28 = phi_at(28.0)
    # box-independence recovered: the far-field slope has collapsed to a few
    # meV/Å, so a 4 Å change in vacuum moves Φ by <30 meV (measured ≈18 meV). A
    # materially larger drift here would mean the setup is still un-converged
    # (raise ecut / vacuum), not that ESM electrostatics are box-dependent.
    assert abs(phi_24 - phi_28) < 0.030, (
        f"Φ has not plateaued at the converged setup: Φ(Lz=24)={phi_24:.6f} eV, "
        f"Φ(Lz=28)={phi_28:.6f} eV, |Δ|={abs(phi_24 - phi_28) * 1e3:.3f} meV "
        f"(converge ecut / vacuum thickness further)")
