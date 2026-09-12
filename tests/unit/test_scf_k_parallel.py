"""k-PARALLEL eigensolve in the NC SCF (``scf.loop.scf``, the ``k_parallel`` knob).

The batched CPU Davidson gets no intra-op thread scaling at small/medium sizes
(batched LAPACK loops run serially — measured eager 8 threads == 1 thread), so
the cores sit idle while the per-k solves are embarrassingly parallel.
``k_parallel`` runs per-k Davidson tasks on a thread pool with torch intra-op
threading pinned to 1 inside — the CPU wall-clock lever (measured 1.9–5.4×)
plus per-k convergence retirement (each task stops when ITS k meets tol
instead of riding the uniform batch to the slowest k's round count).

Pinned here:
- resolution: unset / <= 1 / nk < 2 / CUDA device → off (None); the
  GRADWAVE_K_PARALLEL env default; clamping to nk; explicit arg beats env;
- correctness: a k-parallel SCF converges to the SAME energy, density and
  forces as the batched SCF — the per-band contract is identical (rn <= tol
  at each k's own final Rayleigh–Ritz), only the solve trajectory differs;
- the solver actually receives per-task k-blocks (k_chunk rows when composed
  with streaming, else 1);
- the global torch thread count is restored after the pooled solve;
- hybrid Fock is rejected up front;
- the Input knob (scf.memory.k_parallel) parses, validates, and is threaded
  by run_scf as the scf(k_parallel=) kwarg.

Same rig as test_scf_k_streaming: two-atom Si, one atom nudged so forces are
nonzero, symmetry off so nk is the full mesh.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import _resolve_k_parallel, scf, setup_system
from tests.helpers import PSEUDOS, RY, SI_ONCV, si_fcc

_KMESH = (2, 2, 2)  # 8 k-points with symmetry off
_NK = 8
_CPU = torch.device("cpu")


def _si_system(kmesh=_KMESH):
    si = parse_upf(str(PSEUDOS / SI_ONCV))
    cell, pos = si_fcc()
    pos = pos.copy()
    pos[1, 0] += 0.10  # break the equilibrium so forces are nonzero to compare
    return setup_system(cell, pos, [0, 0], [si], ecut=10 * RY,
                        kmesh=kmesh, use_symmetry=False)


def _run(system, k_parallel, k_chunk=None, max_iter=80):
    return scf(system, LDA_PW92(), smearing="gaussian", width=0.1,
               max_iter=max_iter, etol=1e-11, rhotol=1e-9, diago_tol=1e-10,
               verbose=False, k_parallel=k_parallel, k_chunk=k_chunk)


def test_resolve_k_parallel_semantics(monkeypatch):
    """None / <= 1 / nk < 2 / non-CPU → off (None); a valid count clamps to nk;
    GRADWAVE_K_PARALLEL is the default only when the arg is unset."""
    monkeypatch.delenv("GRADWAVE_K_PARALLEL", raising=False)
    assert _resolve_k_parallel(None, _NK, _CPU) is None   # default: off
    assert _resolve_k_parallel(1, _NK, _CPU) is None      # 1 worker == serial
    assert _resolve_k_parallel(0, _NK, _CPU) is None
    assert _resolve_k_parallel(-4, _NK, _CPU) is None
    assert _resolve_k_parallel(4, 1, _CPU) is None        # single k: nothing to pool
    assert _resolve_k_parallel(4, _NK, torch.device("cuda")) is None  # CPU-only
    assert _resolve_k_parallel(4, _NK, _CPU) == 4         # valid
    assert _resolve_k_parallel(100, _NK, _CPU) == _NK     # clamp to nk
    # env supplies the default only when the explicit arg is None
    monkeypatch.setenv("GRADWAVE_K_PARALLEL", "3")
    assert _resolve_k_parallel(None, _NK, _CPU) == 3
    assert _resolve_k_parallel(2, _NK, _CPU) == 2         # explicit arg wins
    monkeypatch.setenv("GRADWAVE_K_PARALLEL", "not-an-int")
    with pytest.raises(ValueError):
        _resolve_k_parallel(None, _NK, _CPU)


def test_solver_receives_per_task_k_blocks():
    """With k_parallel on, every Davidson call sees 1-k blocks (task size 1);
    composed with k_chunk=2 the task size is 2."""
    davmod = importlib.import_module("gradwave.solvers.davidson")
    seen: list[int] = []
    orig = davmod.davidson_batched

    def spy(h_apply, x0, t, mask, **kw):
        seen.append(x0.shape[0])
        return orig(h_apply, x0, t, mask, **kw)

    system = _si_system()
    with patch.object(davmod, "davidson_batched", side_effect=spy):
        _run(system, k_parallel=4, max_iter=3)
    assert seen and max(seen) == 1
    seen.clear()
    with patch.object(davmod, "davidson_batched", side_effect=spy):
        _run(system, k_parallel=4, k_chunk=2, max_iter=3)
    assert seen and max(seen) == 2


def test_torch_thread_count_restored():
    """The pooled solve pins torch to 1 thread internally and restores the
    caller's setting afterwards."""
    prev = torch.get_num_threads()
    system = _si_system()
    _run(system, k_parallel=2, max_iter=3)
    assert torch.get_num_threads() == prev


def test_k_parallel_rejects_fock():
    """Hybrid Fock couples orbitals across k — rejected up front, same as
    k_chunk streaming."""
    with pytest.raises(NotImplementedError, match="k_parallel"):
        scf(_si_system(), LDA_PW92(), smearing="gaussian", width=0.1,
            max_iter=1, verbose=False, k_parallel=2, fock=object())


@pytest.mark.standard
def test_k_parallel_matches_batched_energy_density_forces():
    """A k-parallel SCF converges to the SAME total energy, density, and
    forces as the all-k batched SCF. Tolerances are one notch looser than the
    k-streaming pin: streaming is a pure loop restructuring (identical
    trajectory), while the pool retires each k at ITS convergence — every
    band still meets rn <= diago_tol, but the batch over-polishes
    already-converged k, so agreement is at the tol contract, not round-off."""
    res_batched = _run(_si_system(), None)
    res_pool = _run(_si_system(), 4)
    assert res_batched.converged and res_pool.converged

    de = abs(float(res_batched.energies.free_energy)
             - float(res_pool.energies.free_energy))
    assert de < 1e-7, f"energy mismatch {de:.3e} eV"

    drho = float((res_batched.rho - res_pool.rho).abs().max())
    assert drho < 1e-6, f"density mismatch {drho:.3e}"

    f_b = forces(res_batched)
    f_p = forces(res_pool)
    df = float((f_b - f_p).abs().max())
    assert df < 1e-5, f"force mismatch {df:.3e} eV/Ang"
    assert float(f_b.abs().max()) > 1e-2  # the nudge produced real forces


# ---------------------------------------------------------------------------
# Input schema + api threading (parse-only / monkeypatched, fast tier)
# ---------------------------------------------------------------------------

def _write_input(tmp_path, extra: str):
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


def test_input_knob_parses_and_validates(tmp_path):
    from gradwave.inputs import InputError, load_input
    from gradwave.inputs.models import SCFParams

    assert SCFParams().memory.k_parallel is None  # default off
    inp = load_input(_write_input(
        tmp_path, "scf: {memory: {k_parallel: 4}}\n"))
    assert inp.scf.memory.k_parallel == 4
    with pytest.raises(InputError, match="k_parallel"):
        load_input(_write_input(tmp_path, "scf: {memory: {k_parallel: 1}}\n"))


def test_run_scf_threads_k_parallel(monkeypatch):
    """run_scf passes scf.memory.k_parallel to scf.loop.scf as a kwarg (same
    mock rig as test_scf_memory_knobs's k_chunk threading pin)."""
    from pathlib import Path

    from ase import Atoms

    import gradwave.api.scf as api_scf
    import gradwave.scf.loop as loop
    from gradwave.inputs.models import Input, MemoryParams, SCFParams

    captured: dict = {}

    def fake_scf(system, xc, **kwargs):
        opts = kwargs.get("opts")
        mem = opts.memory if opts is not None else None
        captured["k_parallel"] = (mem.k_parallel if mem is not None
                                  else kwargs.get("k_parallel"))
        return "SENTINEL"

    monkeypatch.setattr(api_scf, "build_system", lambda inp: object())
    monkeypatch.setattr(api_scf, "_species_upfs", lambda inp: ([], [], None))
    monkeypatch.setattr(api_scf, "_is_uspp", lambda upfs: False)
    monkeypatch.setattr(loop, "scf", fake_scf)

    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]],
                  cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]], pbc=True)
    inp = Input(atoms=atoms, pseudo_dir=Path("."), pseudo_map={"Si": "Si.upf"},
                ecut=200.0, scf=SCFParams(memory=MemoryParams(k_parallel=6)))
    assert api_scf.run_scf(inp, verbose=False) == "SENTINEL"
    assert captured["k_parallel"] == 6


def test_cli_summary_echoes_k_parallel():
    from pathlib import Path

    from ase import Atoms

    from gradwave.cli import _summary_lines
    from gradwave.inputs.models import Input, MemoryParams, SCFParams

    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]],
                  cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]], pbc=True)
    inp = Input(atoms=atoms, pseudo_dir=Path("."), pseudo_map={"Si": "Si.upf"},
                ecut=200.0, scf=SCFParams(memory=MemoryParams(k_parallel=8)))
    line = next((ln for ln in _summary_lines(inp) if "memory" in ln), None)
    assert line is not None and "k_parallel 8" in line
