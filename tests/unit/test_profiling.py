"""Unit tests for the gradwave.profiling deep-profile harness + HTML report.

Laptop-safe by construction: no SCF and no process-level samplers run here. The
one end-to-end path drives :func:`deep_profile` on a trivial matmul workload
(``command=None`` so py-spy/memray are skipped), which exercises the timed loop,
torch.profiler, phase grouping, and report rendering in well under a second. The
heavy real-workload profile is validated separately on asus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gradwave.io import runinfo
from gradwave.profiling import (
    ProfileResult,
    Workload,
    build_compare_html,
    build_summary_html,
    compare,
    deep_profile,
    phase_breakdown,
    workload_from_bench_case,
    write_artifacts,
    write_summary,
)


def _fake_run():
    # A real op so torch.profiler has something to attribute; a trace dict so the
    # RSS-vs-iteration plot has data.
    x = torch.randn(48, 48)
    _ = x @ x
    return {"trace": {"iterations": [
        {"iter": 1, "rss_mb": 100.0},
        {"iter": 2, "rss_mb": 108.0},
        {"iter": 3, "rss_mb": 111.0},
    ]}}


def _fake_workload() -> Workload:
    return Workload(
        name="fake-matmul",
        spec={"system": "fake", "n_atoms": 1, "ecut_ry": 30, "kmesh": "1x1x1",
              "profiled": "one matmul"},
        run=_fake_run,
        command=None,
    )


# --- provenance extension -------------------------------------------------


def test_machine_snapshot_has_full_sha_and_dirty_flag():
    snap = runinfo.machine_snapshot()
    code = snap["code"]
    assert "git_full" in code and "git_dirty" in code
    # In a git checkout the full SHA is a 40-hex string; dirty is a bool.
    if code["git_full"] is not None:
        assert len(code["git_full"]) == 40
    assert code["git_dirty"] in (True, False, None)


def test_bench_git_sha_consolidated_onto_runinfo():
    from gradwave.bench import harness

    assert harness._git_sha() == runinfo._git_commit()


# --- pure helpers ---------------------------------------------------------


def test_phase_breakdown_buckets_ops():
    rows = [
        {"name": "aten::bmm", "self_cpu_time_us": 1_000_000.0},
        {"name": "aten::_fft_c2c", "self_cpu_time_us": 500_000.0},
        {"name": "aten::linalg_eigh", "self_cpu_time_us": 250_000.0},
        {"name": "aten::add", "self_cpu_time_us": 100_000.0},
        {"name": "aten::some_exotic_op", "self_cpu_time_us": 10_000.0},
    ]
    phases = phase_breakdown(rows)
    assert phases["h-apply"] == pytest.approx(1.0)
    assert phases["fft"] == pytest.approx(0.5)
    assert phases["diag"] == pytest.approx(0.25)
    assert phases["mixing"] == pytest.approx(0.1)
    assert phases["other"] == pytest.approx(0.01)


def test_workload_from_bench_case_builds_command_but_does_not_run():
    wl = workload_from_bench_case("Si")
    assert wl.name == "Si"
    assert wl.command is not None and "_runner" in " ".join(wl.command)
    assert wl.spec["system"] == "Si"


def test_workload_from_bench_case_rejects_unknown():
    with pytest.raises(ValueError, match="unknown bench case"):
        workload_from_bench_case("Unobtainium")


# --- end-to-end (trivial workload) ---------------------------------------


def test_deep_profile_end_to_end_no_write():
    res = deep_profile(_fake_workload(), n_timed=2, warmup=1, write=False,
                       run_pyspy=False, run_memray=False)
    assert isinstance(res, ProfileResult)
    assert res.headline["wall_median_s"] >= 0.0
    assert res.headline["torch_threads"] == torch.get_num_threads()
    assert res.op_rows, "torch.profiler should have captured at least one op"
    assert res.iterations and res.iterations[0]["rss_mb"] == 100.0
    # command=None → a note explaining py-spy/memray were skipped.
    assert any("no reproducible subprocess" in n for n in res.notes)


def test_summary_html_is_self_contained_and_has_provenance():
    res = deep_profile(_fake_workload(), n_timed=2, warmup=1, write=False,
                       run_pyspy=False, run_memray=False)
    doc = build_summary_html(res)

    # provenance header carries the full SHA + a dirty/clean marker
    code = res.provenance["code"]
    if code["git_full"]:
        assert code["git_full"] in doc
    assert ("DIRTY" in doc) or ("clean" in doc)

    # at least one inline SVG plot is embedded
    assert "<svg" in doc

    # self-contained: no references that would make the browser FETCH an
    # external resource. (matplotlib's inline SVG legitimately declares XML
    # namespace URIs like xmlns="http://www.w3.org/2000/svg" — those are
    # never fetched, so a blanket 'http://' ban is wrong.) The Perfetto pointer
    # is plain text 'ui.perfetto.dev' with no scheme.
    assert "<script src=" not in doc
    assert "<link " not in doc
    assert 'src="http' not in doc
    assert 'href="http' not in doc
    assert "@import" not in doc


def test_write_artifacts_roundtrips_op_table(tmp_path: Path):
    res = deep_profile(_fake_workload(), n_timed=2, warmup=1, write=False,
                       run_pyspy=False, run_memray=False)
    d = write_artifacts(res, tmp_path)
    # json round-trip
    ops = json.loads((d / "op_table.json").read_text())
    assert ops == res.op_rows
    # parquet round-trip (pandas/pyarrow are in the [profiling] extra)
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(d / "op_table.parquet")
    assert len(df) == len(res.op_rows)
    assert "self_cpu_time_us" in df.columns
    # profile.json sidecar is valid and omits the big blobs
    prof = json.loads((d / "profile.json").read_text())
    assert "op_table_str" not in prof and "flame_svg" not in prof
    assert prof["headline"]["wall_median_s"] == res.headline["wall_median_s"]


def test_write_summary_writes_offline_file(tmp_path: Path):
    res = deep_profile(_fake_workload(), n_timed=2, warmup=1, write=False,
                       run_pyspy=False, run_memray=False)
    write_artifacts(res, tmp_path)
    path = write_summary(res, tmp_path)
    assert path.name == "summary.html"
    assert "<svg" in path.read_text()


# --- cross-commit compare -------------------------------------------------


def _write_profile_dir(tmp: Path, sha: str, ops: list[dict]) -> Path:
    d = tmp / sha
    d.mkdir(parents=True, exist_ok=True)
    (d / "op_table.json").write_text(json.dumps(ops))
    (d / "profile.json").write_text(json.dumps(
        {"provenance": {"code": {"git": sha, "git_dirty": False}}}))
    return d


def test_compare_produces_signed_delta(tmp_path: Path):
    a = _write_profile_dir(tmp_path, "aaaaaaa", [
        {"name": "aten::bmm", "self_cpu_time_us": 1000.0},
        {"name": "aten::add", "self_cpu_time_us": 500.0},
    ])
    b = _write_profile_dir(tmp_path, "bbbbbbb", [
        {"name": "aten::bmm", "self_cpu_time_us": 1500.0},   # regression (+500)
        {"name": "aten::add", "self_cpu_time_us": 300.0},    # improvement (-200)
    ])
    rows = compare(a, b)
    by = {r["name"]: r for r in rows}
    assert by["aten::bmm"]["delta"] == pytest.approx(500.0)
    assert by["aten::bmm"]["pct"] == pytest.approx(50.0)
    assert by["aten::add"]["delta"] == pytest.approx(-200.0)
    # sorted by absolute delta: bmm (500) before add (200)
    assert rows[0]["name"] == "aten::bmm"


def test_compare_html_color_codes_regressions(tmp_path: Path):
    a = _write_profile_dir(tmp_path, "aaaaaaa",
                           [{"name": "aten::bmm", "self_cpu_time_us": 1000.0}])
    b = _write_profile_dir(tmp_path, "bbbbbbb",
                           [{"name": "aten::bmm", "self_cpu_time_us": 1500.0}])
    doc = build_compare_html(a, b)
    assert 'class="reg"' in doc
    assert "aaaaaaa" in doc and "bbbbbbb" in doc
    assert "http://" not in doc and "https://" not in doc
