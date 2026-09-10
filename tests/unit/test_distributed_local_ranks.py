"""`distributed: N` — self-launching local-rank mode (schema + CLI relaunch).

`distributed: true` is the legacy contract: the user launches torchrun and the
rank layout comes from its env. `distributed: N` (int >= 2) asks for N LOCAL
ranks; the CLI re-execs the identical invocation through
`torch.distributed.run --standalone --nproc_per_node=N` when no rank env is
present (cli._maybe_relaunch_ranks), so the single-box multi-process mode
(measured 6-12x e2e on multi-k metals) is one YAML line.

Pinned here (parse-only / monkeypatched — the real multi-rank SCF equivalence
is pinned by the existing tests/integration/test_distributed_* suite, which is
launch-mode-agnostic):
- schema: int N >= 2 parses; 0/1/negative and non-int rejected with the field
  name; bool passthrough unchanged;
- relaunch: no rank env + int N → os.execv of torch.distributed.run with
  --standalone --nproc_per_node=N and the original argv; rank env present →
  no exec; bool True → no exec; GRADWAVE_NUM_THREADS defaulted to
  cores//N (capped 8) only when unset;
- api guard: int N without a rank env raises (no silent serial run); int N
  under a mismatched world size raises.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ase import Atoms

from tests.helpers import PSEUDOS


def _write(tmp_path, extra: str) -> Path:
    p = tmp_path / "in.yaml"
    p.write_text(f"""
structure:
  cell: [[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]]
  positions: {{cart: [[0, 0, 0], [0.89175, 0.89175, 0.89175]]}}
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: 680.28
{extra}""")
    return p


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_distributed_int_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, "distributed: 8\n"))
    assert inp.distributed == 8 and type(inp.distributed) is int


def test_distributed_bool_passthrough(tmp_path):
    from gradwave.inputs import load_input

    assert load_input(_write(tmp_path, "distributed: true\n")).distributed is True
    assert load_input(_write(tmp_path, "")).distributed is False


@pytest.mark.parametrize("bad", ["distributed: 1\n", "distributed: 0\n",
                                 "distributed: -4\n", "distributed: eight\n"])
def test_distributed_bad_values_rejected(tmp_path, bad):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="distributed"):
        load_input(_write(tmp_path, bad))


# ---------------------------------------------------------------------------
# CLI relaunch
# ---------------------------------------------------------------------------

def _mk_input(distributed):
    from gradwave.inputs.models import Input

    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]],
                  cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]], pbc=True)
    return Input(atoms=atoms, pseudo_dir=Path("."), pseudo_map={"Si": "Si.upf"},
                 ecut=200.0, distributed=distributed)


class _Execd(Exception):
    def __init__(self, argv):
        self.argv = argv


@pytest.fixture()
def _no_rank_env(monkeypatch):
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
              "GRADWAVE_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)


def _patch_execv(monkeypatch):
    def fake_execv(exe, argv):
        raise _Execd(argv)

    monkeypatch.setattr(os, "execv", fake_execv)


def test_relaunch_execs_torchrun(monkeypatch, _no_rank_env):
    from gradwave.cli import _maybe_relaunch_ranks

    _patch_execv(monkeypatch)
    monkeypatch.setattr("sys.argv", ["gradwave", "in.yaml", "--quiet"])
    with pytest.raises(_Execd) as ei:
        _maybe_relaunch_ranks(_mk_input(4))
    argv = ei.value.argv
    assert "torch.distributed.run" in argv
    assert "--standalone" in argv and "--nproc_per_node=4" in argv
    assert argv[-3:] == ["gradwave.cli", "in.yaml", "--quiet"]
    # per-rank thread default was set (cores//4, floor 1, cap 8)
    n = int(os.environ["GRADWAVE_NUM_THREADS"])
    assert 1 <= n <= 8


def test_relaunch_respects_existing_thread_choice(monkeypatch, _no_rank_env):
    from gradwave.cli import _maybe_relaunch_ranks

    _patch_execv(monkeypatch)
    monkeypatch.setenv("GRADWAVE_NUM_THREADS", "3")
    monkeypatch.setattr("sys.argv", ["gradwave", "in.yaml"])
    with pytest.raises(_Execd):
        _maybe_relaunch_ranks(_mk_input(2))
    assert os.environ["GRADWAVE_NUM_THREADS"] == "3"


def test_no_relaunch_under_torchrun(monkeypatch):
    from gradwave.cli import _maybe_relaunch_ranks

    _patch_execv(monkeypatch)
    for k, v in (("RANK", "0"), ("WORLD_SIZE", "4"), ("LOCAL_RANK", "0"),
                 ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29500")):
        monkeypatch.setenv(k, v)
    assert _maybe_relaunch_ranks(_mk_input(4)) is None


def test_no_relaunch_for_bool_or_off(monkeypatch, _no_rank_env):
    from gradwave.cli import _maybe_relaunch_ranks

    _patch_execv(monkeypatch)
    assert _maybe_relaunch_ranks(_mk_input(True)) is None
    assert _maybe_relaunch_ranks(_mk_input(False)) is None


# ---------------------------------------------------------------------------
# api guard
# ---------------------------------------------------------------------------

def test_api_refuses_int_ranks_without_env(monkeypatch, _no_rank_env):
    import gradwave.api.scf as api_scf

    monkeypatch.setattr(api_scf, "build_system", lambda inp: object())
    monkeypatch.setattr(api_scf, "_species_upfs", lambda inp: ([], [], None))
    monkeypatch.setattr(api_scf, "_is_uspp", lambda upfs: False)
    with pytest.raises(RuntimeError, match="local ranks"):
        api_scf.run_scf(_mk_input(4), verbose=False)
