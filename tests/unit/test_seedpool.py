"""SeedPool dispatcher — thread-splitting, worker resolution and the
serial/parallel map, plus the phonon FD reassembly it feeds.

No SCF: the map is exercised with a trivial arithmetic worker (and the phonon
force-constant reassembly with an analytic force field), so the dispatcher's
scheduling, order-preservation and thread cap are validated without running any
DFT. The parallel-equals-serial CAMPAIGN check (real SCFs) lives in
tests/integration/test_seedpool_parallel_equals_serial.py.
"""

import numpy as np

from gradwave.postscf import seedpool
from gradwave.postscf.phonons_supercell import (
    build_supercell,
    displacement_list,
    force_constants_from_forces,
    force_constants_home,
)

# ---------------------------------------------------------------------------
# module-level workers (must be picklable for the ProcessPoolExecutor path)
# ---------------------------------------------------------------------------


def _square_worker(x: int) -> tuple[int, int]:
    """Trivial worker: returns (x**2, this-worker's torch thread count) so the
    test can check both the result mapping and that the pool pinned threads."""
    import torch

    return x * x, torch.get_num_threads()


def _fake_force(res):
    """Deterministic 'analytic force' as a smooth function of the (N_sc, 3)
    positions carried straight through by the fake make_scf below. Any smooth
    field works — the point is that serial and parallel assembly see the SAME
    per-displacement forces, so the resulting Φ must match exactly."""
    import torch

    p = np.asarray(res, dtype=float)
    f = np.stack([np.sin(p.sum(axis=1)), np.cos(p[:, 0]), p[:, 1] ** 2 - p[:, 2]],
                 axis=1)
    return torch.tensor(f)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_worker_thread_cap_splits_budget():
    assert seedpool.worker_thread_cap(8, 2) == 4
    assert seedpool.worker_thread_cap(22, 3) == 7   # floor(22/3)
    assert seedpool.worker_thread_cap(6, 6) == 1


def test_worker_thread_cap_never_below_one():
    assert seedpool.worker_thread_cap(2, 8) == 1    # more workers than threads
    assert seedpool.worker_thread_cap(0, 4) == 1
    assert seedpool.worker_thread_cap(4, 0) == 4    # n_workers clamped to 1


def test_resolve_workers():
    assert seedpool.resolve_workers(None, 12) == 1   # None -> serial
    assert seedpool.resolve_workers(1, 12) == 1
    assert seedpool.resolve_workers(3, 12) == 3
    assert seedpool.resolve_workers(20, 12) == 12    # clamped to #spokes
    assert seedpool.resolve_workers(4, 0) == 1       # no spokes -> at least 1


# ---------------------------------------------------------------------------
# map_spokes — serial path (no processes) and scheduling
# ---------------------------------------------------------------------------


def test_map_spokes_serial_is_in_process_and_ordered():
    calls = []

    def worker(x):
        calls.append(x)          # mutation visible only if in-process (serial)
        return x * x

    out = seedpool.map_spokes(worker, [3, 1, 2], n_workers=1)
    assert out == [9, 1, 4]
    assert calls == [3, 1, 2]    # serial: same process, same order


def test_map_spokes_none_workers_serial():
    out = seedpool.map_spokes(lambda x: x + 1, [10, 20], n_workers=None)
    assert out == [11, 21]


def test_map_spokes_single_spoke_stays_serial():
    # even n_workers>1 avoids spawning a pool for a lone spoke
    calls = []
    out = seedpool.map_spokes(lambda x: calls.append(x) or x, [42], n_workers=4)
    assert out == [42] and calls == [42]


# ---------------------------------------------------------------------------
# map_spokes — real process pool (trivial worker, no SCF)
# ---------------------------------------------------------------------------


def test_map_spokes_parallel_matches_serial_and_order():
    spokes = list(range(6))
    serial = seedpool.map_spokes(_square_worker, spokes, n_workers=1)
    parallel = seedpool.map_spokes(_square_worker, spokes, n_workers=3,
                                   total_threads=6)
    assert [v for v, _ in serial] == [x * x for x in spokes]
    # order is preserved regardless of completion order
    assert [v for v, _ in parallel] == [v for v, _ in serial]


def test_map_spokes_parallel_pins_worker_threads():
    # total 6 threads / 2 workers -> 3 threads pinned in each worker process
    out = seedpool.map_spokes(_square_worker, list(range(4)), n_workers=2,
                              total_threads=6)
    caps = {t for _, t in out}
    assert caps == {3}   # floor(6/2) = 3, applied by the pool initializer


# ---------------------------------------------------------------------------
# phonon FD reassembly: parallel force map == serial force_constants_home
# ---------------------------------------------------------------------------


def test_phonon_parallel_assembly_matches_force_constants_home():
    cell = np.eye(3) * 3.2
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    scmap = build_supercell(cell, pos, [0, 0], (2, 2, 1))
    h = 0.01

    def make_scf(positions, start_from=None):
        return positions   # the "result" is just the geometry; _fake_force reads it

    phi_serial = force_constants_home(make_scf, scmap, h=h, force_fn=_fake_force,
                                      verbose=False)
    force_map = {(a, i, sign): _fake_force(p).numpy()
                 for (a, i, sign, p) in displacement_list(scmap, h)}
    phi_parallel = force_constants_from_forces(force_map, scmap, h)
    assert phi_serial.shape == phi_parallel.shape
    assert np.allclose(phi_serial, phi_parallel, atol=1e-12)


def test_displacement_list_covers_home_atoms_both_signs():
    cell = np.eye(3) * 3.0
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    scmap = build_supercell(cell, pos, [0, 0], (2, 2, 2))
    tasks = displacement_list(scmap, 0.02)
    assert len(tasks) == 6 * scmap.n_prim            # 6*N_prim spokes
    tags = {(a, i, s) for (a, i, s, _p) in tasks}
    assert tags == {(a, i, s) for a in range(scmap.n_prim)
                    for i in range(3) for s in (+1, -1)}
    # each spoke moves exactly the one home atom by ±h along the one axis
    for a, i, sign, p in tasks:
        d = p - scmap.positions_super
        assert np.count_nonzero(np.abs(d) > 1e-12) == 1
        assert np.isclose(d[a, i], sign * 0.02)
