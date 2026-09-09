"""The scf.memory footprint knobs: schema, validation, the api → Davidson
environment bridge, and the k_chunk kwarg threading.

Parse-only / monkeypatched, no real SCF, so these run in the fast tier.
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
# the api → Davidson environment bridge
# ---------------------------------------------------------------------------

_ENV_KEYS = (
    "GRADWAVE_MAX_DIM_FACTOR",
    "GRADWAVE_SUBSPACE_BUDGET_GB",
    "GRADWAVE_SUBSPACE_STORAGE",
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


def test_env_bridge_sets_only_nondefaults_and_restores():
    from gradwave.api._common import _davidson_memory_env

    inp = _mk_input(max_dim_factor=2, subspace_storage="complex64")
    with _davidson_memory_env(inp):
        assert os.environ["GRADWAVE_MAX_DIM_FACTOR"] == "2"
        assert os.environ["GRADWAVE_SUBSPACE_STORAGE"] == "complex64"
        # a knob left unset is not exported
        assert "GRADWAVE_SUBSPACE_BUDGET_GB" not in os.environ
    # restored (removed) on exit
    for k in _ENV_KEYS:
        assert k not in os.environ


def test_env_bridge_budget_is_a_float_string():
    from gradwave.api._common import _davidson_memory_env

    with _davidson_memory_env(_mk_input(subspace_budget_gb=1.5)):
        assert float(os.environ["GRADWAVE_SUBSPACE_BUDGET_GB"]) == pytest.approx(1.5)


def test_env_bridge_default_input_is_a_noop():
    from gradwave.api._common import _davidson_memory_env

    with _davidson_memory_env(_mk_input()):
        for k in _ENV_KEYS:
            assert k not in os.environ


def test_env_bridge_does_not_clobber_a_preexisting_env(monkeypatch):
    from gradwave.api._common import _davidson_memory_env

    monkeypatch.setenv("GRADWAVE_MAX_DIM_FACTOR", "3")
    # the Input leaves max_dim_factor unset → the pre-existing env survives
    with _davidson_memory_env(_mk_input(subspace_storage="complex64")):
        assert os.environ["GRADWAVE_MAX_DIM_FACTOR"] == "3"
        assert os.environ["GRADWAVE_SUBSPACE_STORAGE"] == "complex64"
    # the pre-existing value is left exactly as it was, our knob is removed
    assert os.environ["GRADWAVE_MAX_DIM_FACTOR"] == "3"
    assert "GRADWAVE_SUBSPACE_STORAGE" not in os.environ


# ---------------------------------------------------------------------------
# run_scf threads k_chunk as a kwarg AND sets the env for the solve
# ---------------------------------------------------------------------------

def test_run_scf_threads_k_chunk_and_env(monkeypatch):
    """run_scf must (a) pass scf.memory.k_chunk to scf.loop.scf as a kwarg and
    (b) have the Davidson env vars live for the duration of that scf() call,
    then torn down afterwards."""
    import gradwave.api.scf as api_scf
    import gradwave.scf.loop as loop

    captured: dict = {}

    def fake_scf(system, xc, **kwargs):
        captured["k_chunk"] = kwargs.get("k_chunk")
        # the env bridge must be active WHILE the solver runs
        captured["env_factor"] = os.environ.get("GRADWAVE_MAX_DIM_FACTOR")
        captured["env_storage"] = os.environ.get("GRADWAVE_SUBSPACE_STORAGE")
        return "SENTINEL"

    monkeypatch.setattr(api_scf, "build_system", lambda inp: object())
    monkeypatch.setattr(api_scf, "_species_upfs", lambda inp: ([], [], None))
    monkeypatch.setattr(api_scf, "_is_uspp", lambda upfs: False)
    monkeypatch.setattr(loop, "scf", fake_scf)

    inp = _mk_input(k_chunk=2, max_dim_factor=2, subspace_storage="complex64")
    result = api_scf.run_scf(inp, verbose=False)

    assert result == "SENTINEL"
    assert captured["k_chunk"] == 2
    assert captured["env_factor"] == "2"
    assert captured["env_storage"] == "complex64"
    # env is restored (torn down) once run_scf returns
    for k in _ENV_KEYS:
        assert k not in os.environ


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
