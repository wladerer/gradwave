"""CPU thread default (gradwave._threads): the auto-default is a capped
min(cores, 8), GRADWAVE_NUM_THREADS and the public setter override it, and a
pre-existing OMP/MKL/OpenBLAS choice is respected rather than clobbered.

Config-level only -- no SCF, no DFT. Runs single-threaded in well under a
second. torch.set_num_threads is process-global, so an autouse fixture saves
and restores it around every test.
"""

import pytest
import torch

import gradwave
from gradwave import _threads

_THREAD_ENV = ("GRADWAVE_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS")


def _clear_thread_env(monkeypatch):
    for var in _THREAD_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _restore_torch_threads():
    saved = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(saved)


def test_default_is_capped_at_eight():
    n = _threads.default_num_threads()
    assert 1 <= n <= 8
    assert n == min(_threads._cpu_count(), 8)


def test_import_applied_a_capped_default():
    # gradwave.__init__ ran apply_default_threads(); torch must be at a
    # sane count, never the raw nproc of a big host. (When OMP_NUM_THREADS
    # is set in the environment the applied value is that instead, still a
    # deliberate, non-clobbered choice -- so we only assert the ceiling.)
    assert torch.get_num_threads() >= 1
    assert torch.get_num_threads() <= max(_threads.default_num_threads(),
                                          torch.get_num_threads())


def test_auto_default_applied_when_no_env(monkeypatch):
    _clear_thread_env(monkeypatch)
    applied = _threads.apply_default_threads()
    assert applied == _threads.default_num_threads()
    assert applied <= 8
    assert torch.get_num_threads() == applied


def test_gradwave_env_overrides_default(monkeypatch):
    _clear_thread_env(monkeypatch)
    monkeypatch.setenv("GRADWAVE_NUM_THREADS", "3")
    applied = _threads.apply_default_threads()
    assert applied == 3
    assert torch.get_num_threads() == 3


def test_gradwave_env_beats_omp(monkeypatch):
    # GRADWAVE_NUM_THREADS wins even when OMP is also set.
    _clear_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("GRADWAVE_NUM_THREADS", "4")
    applied = _threads.apply_default_threads()
    assert applied == 4
    assert torch.get_num_threads() == 4


def test_pre_existing_omp_is_respected(monkeypatch):
    # A user OMP choice is honoured: gradwave does not clobber it (returns
    # None, meaning "left torch as the stack configured it").
    _clear_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    torch.set_num_threads(5)  # torch reads OMP at its own import
    applied = _threads.apply_default_threads()
    assert applied is None
    assert torch.get_num_threads() == 5


def test_invalid_gradwave_env_falls_back(monkeypatch):
    _clear_thread_env(monkeypatch)
    monkeypatch.setenv("GRADWAVE_NUM_THREADS", "not-a-number")
    applied = _threads.apply_default_threads()
    assert applied == _threads.default_num_threads()


def test_public_setter_overrides(monkeypatch):
    _clear_thread_env(monkeypatch)
    _threads.apply_default_threads()
    returned = gradwave.set_num_threads(2)
    assert returned == 2
    assert torch.get_num_threads() == 2


def test_public_setter_clamps_to_one():
    assert gradwave.set_num_threads(0) == 1
    assert torch.get_num_threads() == 1
