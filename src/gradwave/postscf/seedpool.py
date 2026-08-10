"""SeedPool — fan independent forward-SCF *spokes* out to worker processes.

The hub-and-spoke campaigns — supercell-phonon finite displacements
(``postscf.phonons_supercell``), elastic strains (``api.run_elastic``) and EOS
volumes (``api.run_eos``) — each run a set of INDEPENDENT forward SCFs that
today execute serially in one process, every spoke warm-started from a single
reference SCF. SeedPool computes that reference once (serially), writes its
checkpoint to a temp file, then dispatches the spoke SCFs across ``n_workers``
OS processes with :class:`concurrent.futures.ProcessPoolExecutor`. Each worker
rebuilds its own spoke system from cheap, picklable parameters, loads the shared
reference checkpoint as the warm-start seed, runs the SCF and returns only small
result arrays (forces / stress / energy). No live tensors or ``System`` objects
cross the process boundary — only the checkpoint PATH and the spoke's
scalar/ndarray parameters. This module is the *generic* executor; the
campaign-specific worker functions and specs live next to their drivers in
``gradwave.api`` (kept there so this module carries no SCF/api dependency and
stays import-cheap).

forward-only scope
------------------
The quantities that cross back from a worker are plain numbers. PyTorch autograd
graphs are process-local and do NOT serialize across a ``ProcessPoolExecutor``,
so a DIFFERENTIABLE campaign (inverse design, or a gradient threaded through a
phonon/elastic/EOS result) must stay on the SERIAL path (``n_workers=1``), where
the live result object is retained and its graph is intact. SeedPool is purely a
wall-time lever for the FORWARD campaigns; it never changes what a serial run
computes, only how many processes evaluate the spokes.

thread budget
-------------
N worker processes each fanning BLAS/FFT across every core would oversubscribe
the box and thrash (see ``gradwave._threads`` — gradwave's per-SCF sweet spot is
~8 threads, and more is slower on hybrid CPUs). SeedPool therefore SPLITS the
intra-op thread budget: each worker is pinned to ``floor(total_threads /
n_workers)`` threads (at least 1), set via ``GRADWAVE_NUM_THREADS`` +
``torch.set_num_threads`` in the pool initializer so both the fork path (torch
already imported) and the spawn path (fresh import reads the env var) are
covered.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def worker_thread_cap(total_threads: int, n_workers: int) -> int:
    """Intra-op thread count for one worker: ``floor(total / n_workers)``, ≥ 1.

    Splitting a fixed thread budget across the pool keeps N concurrent SCFs from
    each grabbing every core (see the module docstring / ``gradwave._threads``).
    Both arguments are clamped to ≥ 1 so a degenerate call still yields a usable
    (single-thread) cap rather than a zero or a ``ZeroDivisionError``.
    """
    n_workers = max(1, int(n_workers))
    return max(1, int(total_threads) // n_workers)


def resolve_workers(n_workers: int | None, n_spokes: int) -> int:
    """Effective worker count: ``None``/≤1 → 1 (serial), else clamped to the
    number of spokes (no point spawning more processes than there is work)."""
    if n_workers is None:
        return 1
    return max(1, min(int(n_workers), max(1, int(n_spokes))))


def _init_worker(n_threads: int) -> None:
    """Pool initializer: pin this worker process to ``n_threads`` intra-op
    threads. Sets ``GRADWAVE_NUM_THREADS`` (read by ``gradwave._threads`` on a
    fresh import — the spawn path) and calls ``torch.set_num_threads`` directly
    (the fork path, where gradwave/torch are already imported and the import-time
    default has already run)."""
    import os

    os.environ["GRADWAVE_NUM_THREADS"] = str(max(1, int(n_threads)))
    try:
        import torch

        torch.set_num_threads(max(1, int(n_threads)))
    except Exception:  # pragma: no cover - torch is always present in practice
        pass


def map_spokes(
    worker: Callable[[_T], _R],
    spokes: Iterable[_T],
    *,
    n_workers: int | None,
    total_threads: int | None = None,
    verbose: bool = False,
    mp_context: str | None = "spawn",
) -> list[_R]:
    """Evaluate ``worker(spoke)`` for every spoke, results in input order.

    ``n_workers`` (``None`` or ``1``) runs the spokes SERIALLY in-process —
    identical semantics to a plain ``[worker(s) for s in spokes]`` list
    comprehension, so a driver's default path is byte-for-byte unchanged. With
    ``n_workers > 1`` (and more than one spoke) the spokes fan out over a
    :class:`~concurrent.futures.ProcessPoolExecutor` of that width, each worker
    pinned to ``worker_thread_cap(total_threads, n_workers)`` threads.

    ``worker`` and every ``spoke`` MUST be picklable (module-level function,
    plain-data spec) — the executor pickles both to the worker process. Nothing
    else crosses the boundary; the workers are expected to reload any heavy state
    (a reference-SCF checkpoint) from a path they carry. ``total_threads``
    defaults to the parent's current ``torch.get_num_threads()``. ``mp_context``
    defaults to ``"spawn"``: forking a worker after torch/BLAS have started their
    thread pools deadlocks (a classic fork-after-threads hazard, and MEASURED here
    — the fork path hung on the real SCF workers), so each worker is spawned with a
    fresh interpreter. Spawn re-imports torch/gradwave once per worker (~a few
    seconds), amortized across the campaign's spokes. Pass ``"fork"`` only if you
    know the parent has no live thread pools.
    """
    spoke_list = list(spokes)
    w = resolve_workers(n_workers, len(spoke_list))
    if w <= 1 or len(spoke_list) <= 1:
        return [worker(s) for s in spoke_list]

    import concurrent.futures as cf

    if total_threads is None:
        import torch

        total_threads = torch.get_num_threads()
    cap = worker_thread_cap(total_threads, w)

    ctx = None
    if mp_context is not None:
        import multiprocessing as mp

        ctx = mp.get_context(mp_context)

    if verbose:
        print(
            f"  seedpool: {len(spoke_list)} spokes over {w} workers "
            f"({cap} threads each)",
            flush=True,
        )

    results: list[Any] = [None] * len(spoke_list)
    executor = cf.ProcessPoolExecutor(
        max_workers=w, initializer=_init_worker, initargs=(cap,), mp_context=ctx
    )
    try:
        fut_to_idx = {
            executor.submit(worker, s): i for i, s in enumerate(spoke_list)
        }
        done = 0
        for fut in cf.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            if verbose:
                print(
                    f"  seedpool: {done}/{len(spoke_list)} spokes done",
                    flush=True,
                )
    finally:
        executor.shutdown(wait=True)
    return results
