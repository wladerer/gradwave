"""Provenance block (runinfo.py): every field JSON-serializable, the
process meter accounts wall/cpu/rss, and the report renderer accepts the
block. All collection is best-effort — the assertions are about shape and
serializability, never about platform-specific values being present."""

import json

import torch

from gradwave.runinfo import (
    ProcessMeter,
    cpu_info,
    load_info,
    machine_snapshot,
    memory_info,
    provenance_block,
    thermal_info,
)


def test_machine_snapshot_shape_and_json():
    snap = machine_snapshot()
    for key in ("timestamp", "host", "code", "cpu", "memory", "load",
                "thermal"):
        assert key in snap
    assert snap["cpu"]["logical_cores"] >= 1
    assert snap["cpu"]["torch_threads"] == torch.get_num_threads()
    assert snap["code"]["torch"] == torch.__version__
    json.dumps(snap)  # strict-JSON serializable, no tensors/paths inside


def test_process_meter_accounts_work():
    meter = ProcessMeter()
    x = torch.randn(200, 200, dtype=torch.float64)
    for _ in range(20):
        x = x @ x / x.norm()
    out = meter.stop()
    assert out["wall_s"] >= 0.0
    assert out["cpu_s"] > 0.0
    assert out["peak_rss_gb"] > 0.0
    json.dumps(out)


def test_provenance_block_renders_in_report():
    from gradwave.output import _provenance_lines

    prov = provenance_block(machine_snapshot(), ProcessMeter())
    assert "load_end" in prov and "process" in prov
    json.dumps(prov)
    text = "\n".join(_provenance_lines(prov))
    assert "machine" in text
    assert prov["host"]["hostname"] in text
    assert "wall" in text


def test_collectors_never_raise():
    # each collector must degrade to a (possibly empty/None-valued) mapping or
    # None on unsupported platforms — never throw — and stay JSON-serializable
    for fn in (cpu_info, memory_info, load_info, thermal_info):
        out = fn()
        assert out is None or isinstance(out, dict)
        json.dumps(out)
