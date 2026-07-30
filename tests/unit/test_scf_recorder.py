"""SCF flight recorder (scf.recorder.SCFRecorder): schema, diagnosis heuristics,
and the loop wiring for both the NC and USPP/PAW collinear drivers.

The diagnose() heuristics are exercised on hand-built synthetic trajectories
(fast, deterministic); the recorder-through-the-loop and sidecar round-trip go
through a small real SCF.
"""

import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.recorder import TRACE_VERSION, SCFRecorder
from tests.helpers import PSEUDOS, RY, si_fcc, si_upf


def _rec(nspin=1, seed_moment=0.0):
    """A recorder with a tiny nonzero g2 (shell binning only needs a grid)."""
    g2 = torch.rand(6, 6, 6, dtype=torch.float64) + 0.1
    return SCFRecorder(g2, nspin=nspin, seed_moment=seed_moment)


def _iter(it, drho, low2, *, reorder=0, mag_abs=None, n_shells=12):
    """A synthetic per-iteration record. `low2` is the fraction placed in the two
    lowest |G|-shells; the rest is spread over the remaining shells."""
    frac = [0.0] * n_shells
    frac[0] = low2 / 2
    frac[1] = low2 / 2
    rest = (1.0 - low2) / max(n_shells - 2, 1)
    for i in range(2, n_shells):
        frac[i] = rest
    return {
        "it": it,
        "free_energy": -100.0 + it,
        "dE": None if it == 1 else 1e-3 / it,
        "drho": drho,
        "t": 0.1,
        "fermi": 5.0,
        "entropy": 0.0,
        "eig_drift": None if it == 1 else 1e-3,
        "reorder": reorder,
        "shell_frac": frac,
        "mag_abs": mag_abs,
    }


# --------------------------------------------------------------------------- #
# diagnose() heuristics on synthetic trajectories                             #
# --------------------------------------------------------------------------- #


def test_no_tags_on_short_run():
    rec = _rec()
    rec.iters = [_iter(1, 1e-1, 0.7), _iter(2, 2e-1, 0.7)]
    assert rec.diagnose() == []  # 1-2 iterations never diagnose
    # summarize() must not raise on a short run either
    s = rec.summarize()
    assert s["n_iter"] == 2


def test_healthy_run_no_tags():
    rec = _rec()
    # residual falls monotonically, weight not concentrated at long wavelength
    rec.iters = [_iter(i, 10.0 ** (-i), 0.1) for i in range(1, 7)]
    assert rec.diagnose() == []


def test_sloshing_detected():
    rec = _rec()
    # >50% of the residual weight in the lowest 2 shells and a non-monotone drho
    drhos = [1e-1, 1.2e-1, 0.9e-1, 1.3e-1, 1.0e-1]
    rec.iters = [_iter(i + 1, d, 0.72) for i, d in enumerate(drhos)]
    tags = dict(rec.diagnose())
    assert "charge-sloshing" in tags
    assert rec.summarize()["sloshing_fraction_final"] > 0.5


def test_moment_collapse_detected():
    rec = _rec(nspin=2, seed_moment=1.0)
    mags = [0.9, 0.6, 0.3, 0.1, 0.02]
    drhos = [10.0 ** (-i) for i in range(1, 6)]
    rec.iters = [
        _iter(i + 1, drhos[i], 0.1, mag_abs=mags[i]) for i in range(5)
    ]
    tags = dict(rec.diagnose())
    assert "moment-collapse" in tags


def test_no_sloshing_at_converged_noise_floor():
    # Regression: a converged run whose residual has fallen to the noise floor
    # has a meaningless |G|-shell decomposition (long-wavelength-dominated by
    # roundoff) and roundoff-level non-monotonicity. It must NOT read as
    # sloshing even though frac>0.5 and the tail is non-monotone. Observed on
    # fcc Ni (NC, johnson) whose tail residual is ~1e-6.
    rec = _rec()
    # last-5 window entirely at the noise floor (< _SLOSH_RES_FLOOR), with a
    # roundoff-level non-monotone tick and a high long-wavelength fraction.
    drhos = [1e-1, 1e-2, 1e-3, 1e-5, 1e-6, 2e-6, 1e-6, 3e-6]
    rec.iters = [_iter(i + 1, drhos[i], 0.72) for i in range(len(drhos))]
    assert "charge-sloshing" not in dict(rec.diagnose())


def test_no_moment_collapse_when_seed_is_inflated_but_moment_holds():
    # Regression: the USPP/PAW seed_moment is the smooth-grid moment of the SAD
    # atomic seed, many times the converged smooth moment for a strongly-seeded
    # 3d metal (fcc Ni: seed ~10.8 μB vs converged ~0.79 μB). "Fell to <10% of
    # the seed" alone then trips on every healthy FM PAW metal. A moment that
    # settles at a clearly-magnetic value must NOT read as collapsed.
    rec = _rec(nspin=2, seed_moment=10.8)
    mags = [2.5, 1.0, 0.8, 0.79, 0.79]  # settles at a held FM moment
    rec.iters = [
        _iter(i + 1, 10.0 ** -(i + 1), 0.1, mag_abs=mags[i]) for i in range(5)
    ]
    assert "moment-collapse" not in dict(rec.diagnose())


def test_no_moment_collapse_when_moment_holds():
    rec = _rec(nspin=2, seed_moment=1.0)
    mags = [0.9, 0.92, 0.91, 0.90, 0.91]  # stable moment
    rec.iters = [
        _iter(i + 1, 10.0 ** -(i + 1), 0.1, mag_abs=mags[i]) for i in range(5)
    ]
    assert "moment-collapse" not in dict(rec.diagnose())


def test_level_crossing_detected():
    rec = _rec()
    # band reordering in a majority of the recent it>1 iterations
    reorders = [0, 2, 3, 1, 2]
    rec.iters = [
        _iter(i + 1, 10.0 ** -(i + 1), 0.1, reorder=reorders[i]) for i in range(5)
    ]
    tags = dict(rec.diagnose())
    assert "level-crossing" in tags
    assert rec.summarize()["reordering_total"] == sum(reorders)


def test_diagnose_caps_at_three_tags():
    rec = _rec(nspin=2, seed_moment=1.0)
    drhos = [1e-1, 1.2e-1, 0.9e-1, 1.3e-1, 1.0e-1]  # non-monotone
    mags = [0.9, 0.6, 0.3, 0.1, 0.02]  # collapsing
    rec.iters = [
        _iter(i + 1, drhos[i], 0.72, reorder=2, mag_abs=mags[i]) for i in range(5)
    ]
    assert len(rec.diagnose()) <= 3


# --------------------------------------------------------------------------- #
# recorder through the NC SCF loop                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def si_nc_res():
    torch.set_num_threads(4)
    upf = si_upf()
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=12 * RY,
                          kmesh=(1, 1, 1), nbands=8, use_symmetry=False)
    res = scf(system, PBE(), etol=1e-7, rhotol=1e-6, verbose=False, max_iter=50)
    assert res.converged
    return res


def test_nc_recorder_schema(si_nc_res):
    rec = si_nc_res.recorder
    assert rec is not None
    # one record per SCF iteration
    assert len(rec.iters) == si_nc_res.n_iter
    s = rec.summarize()
    assert s["n_iter"] == si_nc_res.n_iter
    for key in ("final_residual", "diagnosis", "sloshing_fraction_final",
                "reordering_total", "wall_time_mean_s"):
        assert key in s
    assert s["wall_time_mean_s"] > 0.0
    # every recorded wall time is positive
    assert all(i["t"] >= 0.0 for i in rec.iters)
    # a healthy converged insulator run fires no diagnosis tags
    assert s["diagnosis"] == []
    # mixing metadata captured once
    assert set(rec.mixing) == {"scheme", "alpha", "kerker", "history"}


def test_nc_recorder_shell_fractions_normalized(si_nc_res):
    for i in si_nc_res.recorder.iters:
        assert len(i["shell_frac"]) == si_nc_res.recorder.n_shells
        assert sum(i["shell_frac"]) == pytest.approx(1.0, abs=1e-9)


def test_trace_dict_is_versioned_and_serializable(si_nc_res):
    import json

    t = si_nc_res.recorder.to_trace_dict()
    assert t["version"] == TRACE_VERSION
    assert len(t["iterations"]) == si_nc_res.n_iter
    # fully JSON serializable
    json.loads(json.dumps(t))


# --------------------------------------------------------------------------- #
# recorder through the USPP/PAW SCF loop                                      #
# --------------------------------------------------------------------------- #


def test_uspp_recorder_records():
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    torch.set_num_threads(4)
    paw = parse_upf_paw(PSEUDOS / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell, pos = si_fcc()
    # loose tolerances: this test asserts the recorder wiring through the
    # USPP/PAW driver, not converged physics
    s = setup_uspp(cell, pos, [0, 0], [paw], ecut=10 * RY,
                   kmesh=(1, 1, 1), ecutrho=40 * RY)
    res = scf_uspp(s, PBE(), etol=1e-6, rhotol=1e-5, diago_tol=1e-7,
                   verbose=False, max_iter=40)
    assert res["converged"]
    rec = res.recorder
    assert rec is not None
    assert len(rec.iters) == res["n_iter"]
    assert rec.summarize()["n_iter"] == res["n_iter"]
    # the recorder is NOT part of the legacy dict view (internal diagnostics)
    assert "recorder" not in res


# --------------------------------------------------------------------------- #
# scf.trace sidecar wiring + analysis.trace_frame round trip                  #
# --------------------------------------------------------------------------- #


def test_scf_trace_sidecar_and_frame(tmp_path):
    pytest.importorskip("pandas")
    from gradwave import analysis
    from gradwave.api import run
    from gradwave.inputs import load_input

    a = 5.43
    h = a / 2
    body = f"""
structure:
  cell: [[0.0, {h}, {h}], [{h}, 0.0, {h}], [{h}, {h}, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {12 * RY}
xc: pbe
kpoints:
  mesh: [1, 1, 1]
symmetry: false
nbands: 8
scf:
  trace: true
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    inp = load_input(p)
    assert inp.scf.trace is True
    summary = run(inp, verbose=False)

    # compact diagnostics ride in the JSON
    assert "scf_diagnostics" in summary
    assert summary["scf_diagnostics"]["n_iter"] == summary["scf"]["n_iter"]

    # the sidecar was written and names itself in outputs
    trace_path = tmp_path / "scf_trace.json"
    assert trace_path.exists()
    assert summary["outputs"].get("scf_trace") == "scf_trace.json"

    # trace_frame round-trips it
    df = analysis.trace_frame(trace_path)
    assert len(df) == summary["scf"]["n_iter"]
    assert "drho" in df.columns
    assert "shell_0" in df.columns and "shell_11" in df.columns
    assert df.attrs["version"] == TRACE_VERSION
    assert "scheme" in df.attrs["mixing"]


def test_scf_trace_off_by_default():
    from gradwave.inputs import SCFParams

    assert SCFParams().trace is False
