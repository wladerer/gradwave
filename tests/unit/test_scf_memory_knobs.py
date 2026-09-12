"""The scf.memory footprint knobs: schema, validation, and the Input →
MemoryOptions → scf()/scf_uspp() → davidson argument threading that replaced
the retired api → environment bridge (the GRADWAVE_* env vars remain the
user-facing overrides, layered over the passed values per solve).

Mostly parse-only / monkeypatched; one tiny real SCF spies on the kwargs
davidson_batched actually receives. Fast tier.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ase import Atoms

from tests.helpers import PSEUDOS

# ---------------------------------------------------------------------------
# schema + parse
# ---------------------------------------------------------------------------

def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def _base(extra: str = "") -> str:
    return f"""
structure:
  cell: [[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]]
  positions: {{cart: [[0, 0, 0], [0.89175, 0.89175, 0.89175]]}}
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: 680.28
{extra}"""


def test_memory_defaults_are_a_noop():
    from gradwave.inputs.models import MemoryParams, SCFParams

    mem = SCFParams().memory
    assert isinstance(mem, MemoryParams)
    assert mem.k_chunk is None
    assert mem.max_dim_factor is None
    assert mem.subspace_budget_gb is None
    assert mem.subspace_storage == "complex128"


def test_memory_block_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base(
        "scf:\n"
        "  memory:\n"
        "    k_chunk: 4\n"
        "    max_dim_factor: 2\n"
        "    subspace_budget_gb: 1.5\n"
        "    subspace_storage: complex64\n")))
    mem = inp.scf.memory
    assert mem.k_chunk == 4
    assert mem.max_dim_factor == 2
    assert mem.subspace_budget_gb == pytest.approx(1.5)
    assert mem.subspace_storage == "complex64"


@pytest.mark.parametrize("extra, needle", [
    ("scf: {memory: {k_chunk: 0}}\n", "k_chunk must be >= 1"),
    ("scf: {memory: {max_dim_factor: 1}}\n", "max_dim_factor must be >= 2"),
    ("scf: {memory: {subspace_budget_gb: 0}}\n", "subspace_budget_gb must be > 0"),
    ("scf: {memory: {subspace_storage: float32}}\n", "subspace_storage"),
    ("scf: {memory: {k_chnk: 4}}\n", "did you mean"),  # typo → key suggestion
])
def test_memory_value_range_errors(tmp_path, extra, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(extra)))


# ---------------------------------------------------------------------------
# Input → MemoryOptions → scf()/scf_uspp() argument threading (the retired
# api → environment bridge's replacement: the knobs travel as ARGUMENTS, and
# the GRADWAVE_* env vars are layered over them per solve in the resolvers)
# ---------------------------------------------------------------------------

_ENV_KEYS = (
    "GRADWAVE_MAX_DIM_FACTOR",
    "GRADWAVE_SUBSPACE_BUDGET_GB",
    "GRADWAVE_SUBSPACE_STORAGE",
    "GRADWAVE_CPU_DENSE_BUDGET",
)


def _mk_input(**mem):
    from gradwave.inputs.models import Input, MemoryParams, SCFParams

    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]],
                  cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]], pbc=True)
    return Input(atoms=atoms, pseudo_dir=Path("."), pseudo_map={"Si": "Si.upf"},
                 ecut=200.0, scf=SCFParams(memory=MemoryParams(**mem)))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with the GRADWAVE_* subspace vars unset."""
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_memory_options_mapping():
    """api._memory_options maps the Input block 1:1 onto MemoryOptions; the
    default "complex128" storage maps to None (all-None group = the
    byte-for-byte historical solve), and k_levers=False strips the k knobs
    for the hybrid/spinor branches where they were always inert."""
    from gradwave.api.scf import _memory_options

    mo = _memory_options(_mk_input())
    assert (mo.k_chunk, mo.k_parallel, mo.max_dim_factor,
            mo.subspace_budget_gb, mo.subspace_storage,
            mo.dense_budget_gb) == (None,) * 6

    mo = _memory_options(_mk_input(k_chunk=4, k_parallel=2, max_dim_factor=2,
                                   subspace_budget_gb=1.5,
                                   subspace_storage="complex64",
                                   dense_budget_gb=0.5))
    assert mo.k_chunk == 4 and mo.k_parallel == 2
    assert mo.max_dim_factor == 2
    assert mo.subspace_budget_gb == pytest.approx(1.5)
    assert mo.subspace_storage == "complex64"
    assert mo.dense_budget_gb == pytest.approx(0.5)

    stripped = _memory_options(_mk_input(k_chunk=4, k_parallel=2),
                               k_levers=False)
    assert stripped.k_chunk is None and stripped.k_parallel is None


def test_run_scf_threads_memory_options_no_env(monkeypatch):
    """run_scf must hand scf() an SCFOptions whose memory group carries the
    Input's scf.memory values — and must NOT touch the process environment
    (the retired bridge's failure mode)."""
    import gradwave.api.scf as api_scf
    import gradwave.scf.loop as loop

    captured: dict = {}

    def fake_scf(system, xc, **kwargs):
        captured["opts"] = kwargs.get("opts")
        captured["env"] = {k: os.environ.get(k) for k in _ENV_KEYS}
        return "SENTINEL"

    monkeypatch.setattr(api_scf, "build_system", lambda inp: object())
    monkeypatch.setattr(api_scf, "_species_upfs", lambda inp: ([], [], None))
    monkeypatch.setattr(api_scf, "_is_uspp", lambda upfs: False)
    monkeypatch.setattr(loop, "scf", fake_scf)

    inp = _mk_input(k_chunk=2, max_dim_factor=2, subspace_budget_gb=1.5,
                    subspace_storage="complex64", dense_budget_gb=0.5)
    result = api_scf.run_scf(inp, verbose=False)

    assert result == "SENTINEL"
    opts = captured["opts"]
    assert opts is not None and opts.memory is not None
    mem = opts.memory
    assert mem.k_chunk == 2
    assert mem.max_dim_factor == 2
    assert mem.subspace_budget_gb == pytest.approx(1.5)
    assert mem.subspace_storage == "complex64"
    assert mem.dense_budget_gb == pytest.approx(0.5)
    # the whole point of the refactor: NO env mutation anywhere on the path
    assert captured["env"] == dict.fromkeys(_ENV_KEYS)
    for k in _ENV_KEYS:
        assert k not in os.environ


def test_run_scf_uspp_threads_dense_only(monkeypatch):
    """The USPP branch passes only dense_budget_gb (the subspace/k knobs are
    NC-only and were historically inert on this branch — kept a silent no-op
    at the api rather than turned into a new error)."""
    import gradwave.api.scf as api_scf
    import gradwave.scf.uspp as uspp_mod

    captured: dict = {}

    def fake_scf_uspp(system, xc, **kwargs):
        captured["opts"] = kwargs.get("opts")
        return "USPP-SENTINEL"

    monkeypatch.setattr(api_scf, "build_system", lambda inp: object())
    monkeypatch.setattr(api_scf, "_species_upfs", lambda inp: ([], [], None))
    monkeypatch.setattr(api_scf, "_is_uspp", lambda upfs: True)
    monkeypatch.setattr(uspp_mod, "scf_uspp", fake_scf_uspp)

    inp = _mk_input(k_chunk=2, max_dim_factor=2, dense_budget_gb=0.5)
    assert api_scf.run_scf(inp, verbose=False) == "USPP-SENTINEL"
    mem = captured["opts"].memory
    assert mem.dense_budget_gb == pytest.approx(0.5)
    assert mem.k_chunk is None and mem.max_dim_factor is None
    assert mem.subspace_budget_gb is None and mem.subspace_storage is None


def test_scf_threads_subspace_knobs_to_davidson(monkeypatch):
    """End-to-end spy: a real (tiny) NC SCF with a MemoryOptions must deliver
    force_dim_factor / subspace_budget_gb / subspace_storage as ARGUMENTS to
    davidson_batched — with no GRADWAVE_* env var set anywhere."""
    import importlib

    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system
    from gradwave.scf.options import MemoryOptions
    from tests.helpers import PSEUDOS as _P
    from tests.helpers import RY, SI_ONCV, si_fcc

    davmod = importlib.import_module("gradwave.solvers.davidson")
    real = davmod.davidson_batched
    seen: list[dict] = []

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        assert all(k not in os.environ for k in _ENV_KEYS)
        return real(*args, **kwargs)

    monkeypatch.setattr(davmod, "davidson_batched", spy)

    si = parse_upf(str(_P / SI_ONCV))
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si], ecut=6 * RY,
                          kmesh=(1, 1, 1), use_symmetry=False)
    mem = MemoryOptions(max_dim_factor=2, subspace_budget_gb=1.0,
                        subspace_storage="complex64")
    res = scf(system, LDA_PW92(), verbose=False, max_iter=8, etol=1e-6,
              rhotol=1e-5, memory=mem)
    assert len(seen) > 0
    for kw in seen:
        assert kw["force_dim_factor"] == 2
        assert kw["subspace_budget_gb"] == pytest.approx(1.0)
        assert kw["subspace_storage"] == "complex64"
    assert res is not None


def test_env_still_overrides_passed_options(monkeypatch):
    """The GRADWAVE_* env vars remain the documented user-facing override:
    each is layered OVER the passed option at its single per-solve read
    point in solvers.davidson."""
    from gradwave.solvers.davidson import (
        _resolve_max_dim_factor,
        _subspace_storage_mode,
        forced_dim_factor,
    )

    # passed force wins when env is unset; env wins over the passed force
    assert forced_dim_factor(None) is None
    assert forced_dim_factor(2) == 2
    monkeypatch.setenv("GRADWAVE_MAX_DIM_FACTOR", "6")
    assert forced_dim_factor(2) == 6
    monkeypatch.delenv("GRADWAVE_MAX_DIM_FACTOR")
    with pytest.raises(ValueError, match="max_dim_factor must be >= 2"):
        forced_dim_factor(1)

    # passed budget gates the factor exactly like the env budget did
    nk, nb, npw = 36, 60, 13000
    assert _resolve_max_dim_factor(nk, nb, npw, 16, 4, budget_gb=3.0) == 2
    assert _resolve_max_dim_factor(nk, nb, npw, 16, 4, budget_gb=16.0) == 4
    monkeypatch.setenv("GRADWAVE_SUBSPACE_BUDGET_GB", "16.0")
    assert _resolve_max_dim_factor(nk, nb, npw, 16, 4, budget_gb=3.0) == 4
    monkeypatch.delenv("GRADWAVE_SUBSPACE_BUDGET_GB")

    # storage: passed value applies; env layered over it
    assert _subspace_storage_mode(None) == "complex128"
    assert _subspace_storage_mode("complex64") == "complex64"
    monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", "complex128")
    assert _subspace_storage_mode("complex64") == "complex128"
    with pytest.raises(ValueError, match="SUBSPACE_STORAGE"):
        _subspace_storage_mode("float32")


def test_dense_budget_flows_as_override(monkeypatch):
    """dense_budget_gb (GB) reaches core.batch's process-local override in
    BYTES — for both drivers — instead of the retired env write."""
    from gradwave.core import batch as batch_mod
    from gradwave.scf.loop import _memory_solver_kwargs
    from gradwave.scf.options import MemoryOptions, SCFOptions
    from gradwave.scf.uspp_loop import _uspp_group_overrides

    solver_kw, dense = _memory_solver_kwargs(MemoryOptions(dense_budget_gb=0.5))
    assert solver_kw == {}
    assert dense == pytest.approx(0.5e9)
    assert _memory_solver_kwargs(None) == ({}, None)

    calls: list = []
    monkeypatch.setattr(batch_mod, "set_cpu_dense_budget_override",
                        lambda b: calls.append(b))
    _uspp_group_overrides(SCFOptions(memory=MemoryOptions(dense_budget_gb=0.5)))
    assert calls == [pytest.approx(0.5e9)]


def test_dense_budget_default_none_is_a_noop():
    from gradwave.inputs.models import SCFParams

    assert SCFParams().memory.dense_budget_gb is None


def test_dense_budget_nonpositive_rejected(tmp_path):
    from gradwave.inputs.models import InputError, MemoryParams

    with pytest.raises(InputError, match="dense_budget_gb must be > 0"):
        MemoryParams(dense_budget_gb=0.0)


def test_uspp_rejects_nc_only_memory_knobs():
    """scf_uspp fails loudly (not silently) on the NC-only memory knobs."""
    from gradwave.scf.options import MemoryOptions, SCFOptions
    from gradwave.scf.uspp_loop import _uspp_group_overrides

    with pytest.raises(NotImplementedError, match="k_chunk"):
        _uspp_group_overrides(SCFOptions(memory=MemoryOptions(k_chunk=2)))
    with pytest.raises(NotImplementedError, match="subspace_storage"):
        _uspp_group_overrides(
            SCFOptions(memory=MemoryOptions(subspace_storage="complex64")))


# ---------------------------------------------------------------------------
# CLI echo
# ---------------------------------------------------------------------------

def test_cli_summary_echoes_memory_knobs():
    from gradwave.cli import _summary_lines

    inp = _mk_input(k_chunk=4, subspace_storage="complex64")
    line = next((ln for ln in _summary_lines(inp) if "memory" in ln), None)
    assert line is not None
    assert "k_chunk 4" in line
    assert "subspace_storage complex64" in line


def test_cli_summary_omits_memory_when_default():
    from gradwave.cli import _summary_lines

    inp = _mk_input()
    assert all("memory" not in ln for ln in _summary_lines(inp))
