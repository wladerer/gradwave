"""The SCF memory / performance KNOBS must not change the converged physics.

`test_scf_memory_knobs.py` pins the SCHEMA and the api→env bridge with
parse-only / monkeypatched calls (fast tier). This module closes the other half:
a real (tiny) SCF driven through the *public* surface — an `Input` carrying the
`scf.memory` block, run via `api.run_scf` — must reach the SAME converged answer
whether or not each knob is engaged, to the tolerance the knob's contract
implies. A knob that silently altered the result (or was silently ignored)
between the Input and the solver is a real bug, and each test here is written so
the assertion BITES: it asserts the converged energy agrees AND that the knob
demonstrably took effect during the actual solve (the resolver saw the env /
returned the streamed chunk / the Γ flag flipped). "Same answer" alone would pass
a silently-ignored knob; the pair does not.

Contracts pinned (see `scf.loop`, `solvers.davidson`, `api._common`):

  * k_chunk           — pure k-partitioning of the eigensolve  → BIT-EXACT (1e-10)
  * max_dim_factor    — a different Davidson subspace ceiling, same fixed point
                        → converged energy matches to ~1e-8
  * subspace_storage  — complex64 is a PRECISION tradeoff, NOT bit-exact → the
                        documented ~1e-6 eV floor (and a regression that loosened
                        that must fail, so the bound is tight)
  * GRADWAVE_GAMMA_REAL — the Γ real-wavefunction path is bit-exact (1e-10) to the
                        complex path on an eligible Γ-only cell, and `auto` falls
                        back (gamma_real False) — still converging — on multi-k.

Small Si-diamond cells, ONCV (valence-only), run on asus with
OMP_NUM_THREADS=8; each SCF is seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard

# Si diamond primitive cell (a = 5.43 Å); a/2 = 2.715.
_CELL = "[[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]]"
_POS = "{cart: [[0, 0, 0], [1.3575, 1.3575, 1.3575]]}"


def _write(
    tmp_path: Path,
    *,
    mem: str = "",
    mesh: tuple[int, int, int] = (1, 1, 1),
    symmetry: bool = True,
    etol: float = 1.0e-10,
    rhotol: float = 1.0e-9,
) -> Path:
    """A Si-diamond SCF input with a tight-convergence scf block and an optional
    scf.memory sub-block. Written through YAML so the test exercises the real
    load_input → Input → api.run_scf surface (the knob-plumbing bug surface)."""
    mem_block = ""
    if mem:
        mem_block = "  memory:\n" + "".join(f"    {ln}\n" for ln in mem.splitlines())
    body = f"""
structure:
  cell: {_CELL}
  positions: {_POS}
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {10 * RY}
xc: lda
symmetry: {"true" if symmetry else "false"}
kpoints:
  mesh: [{mesh[0]}, {mesh[1]}, {mesh[2]}]
scf:
  max_iter: 120
  etol: {etol}
  rhotol: {rhotol}
  diago:
    tol: 1.0e-10
{mem_block}"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def _run(path: Path):
    from gradwave.api import run_scf
    from gradwave.inputs import load_input

    return run_scf(load_input(path), verbose=False)


# a multi-k mesh with symmetry OFF, so nk is the full mesh (8) and k-streaming /
# the Davidson subspace genuinely engage.
_MULTIK = dict(mesh=(2, 2, 2), symmetry=False)


@pytest.fixture(scope="module")
def multik_reference(tmp_path_factory):
    """The DEFAULT (all knobs off) multi-k SCF — the baseline every knob is
    compared against. Computed once; the flipped runs each rebuild an identical
    system from the same YAML, differing only in the scf.memory block."""
    torch.set_num_threads(8)
    tmp = tmp_path_factory.mktemp("ref")
    res = _run(_write(tmp, **_MULTIK))
    assert res.converged
    return res


# ---------------------------------------------------------------------------
# 1. k_chunk — pure k-partitioning of the solve → BIT-EXACT.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k_chunk", [1, 3])
def test_k_chunk_is_bit_exact_through_api(tmp_path, monkeypatch, multik_reference, k_chunk):
    """scf.memory.k_chunk streams the eigensolve in k-blocks — the density is
    Σ_k either way, so the converged energy and density are bit-exact to the
    all-k solve. The spy on `_resolve_k_chunk` proves the knob actually reached
    the loop and engaged streaming during THIS solve (so a silently-dropped
    k_chunk, which would still match, cannot pass)."""
    import gradwave.scf.loop as loop

    real = loop._resolve_k_chunk
    seen: dict = {}

    def spy(kc, nk):
        out = real(kc, nk)
        seen["kc_arg"], seen["resolved"], seen["nk"] = kc, out, nk
        return out

    monkeypatch.setattr(loop, "_resolve_k_chunk", spy)
    res = _run(_write(tmp_path, mem=f"k_chunk: {k_chunk}", **_MULTIK))

    # the knob took effect: k_chunk reached the loop and resolved to streaming
    assert res.converged
    assert seen["kc_arg"] == k_chunk
    assert seen["nk"] == 8  # symmetry off → full 2×2×2 mesh
    assert seen["resolved"] == k_chunk  # a positive chunk < nk → streaming ON

    e_ref = float(multik_reference.energies.free_energy)
    e_chunk = float(res.energies.free_energy)
    assert abs(e_chunk - e_ref) <= 1e-10 * abs(e_ref), (
        f"k_chunk={k_chunk} not bit-exact: dE = {e_chunk - e_ref:.3e} eV")
    drho = float((res.rho - multik_reference.rho).abs().max())
    assert drho < 1e-8, f"k_chunk={k_chunk} density drift {drho:.3e}"


# ---------------------------------------------------------------------------
# 2. max_dim_factor — different subspace ceiling, same converged fixed point.
# ---------------------------------------------------------------------------


def test_max_dim_factor_matches_default_through_api(tmp_path, monkeypatch, multik_reference):
    """Halving the Davidson subspace ceiling (4 → 2) is a different restart path
    to the SAME fixed point, so the converged SCF energy is subspace-ceiling
    independent (~1e-8). The spy proves the api env bridge (GRADWAVE_MAX_DIM_FACTOR)
    was LIVE during the real solve and forced factor 2 — a silently-ignored knob
    would run factor 4 and match trivially, which this catches."""
    import importlib

    # the real submodule, not the `davidson` FUNCTION that solvers/__init__
    # re-exports under the same dotted name (attribute shadowing)
    davmod = importlib.import_module("gradwave.solvers.davidson")
    real = davmod._resolve_max_dim_factor
    seen: dict = {}

    def spy(nk, nb, m, elem_bytes, requested):
        import os

        out = real(nk, nb, m, elem_bytes, requested)
        seen["env"] = os.environ.get("GRADWAVE_MAX_DIM_FACTOR")
        seen["resolved"] = out
        return out

    monkeypatch.setattr(davmod, "_resolve_max_dim_factor", spy)
    res = _run(_write(tmp_path, mem="max_dim_factor: 2", **_MULTIK))

    assert res.converged
    # the env bridge was live during the solve and the factor was forced to 2
    assert seen["env"] == "2"
    assert seen["resolved"] == 2

    e_ref = float(multik_reference.energies.free_energy)
    e2 = float(res.energies.free_energy)
    assert abs(e2 - e_ref) < 1e-8, f"max_dim_factor=2 shifted the energy: dE = {e2 - e_ref:.3e} eV"


# ---------------------------------------------------------------------------
# 2b. dense_budget_gb — band-chunk the dense-grid FFT boxes, same fixed point.
# ---------------------------------------------------------------------------


def test_dense_budget_matches_default_through_api(tmp_path, monkeypatch, multik_reference):
    """An explicit scf.memory.dense_budget_gb band-chunks the H-apply / density
    dense-grid FFT boxes. Chunking is bit-identical for the apply (no cross-band
    reduction) and reorders the density band-sum at the ~ulp level, so the
    converged SCF reaches the SAME fixed point (~1e-8). The spy on
    core.batch._dense_band_chunk proves the budget was LIVE and actually chunked
    the bands during the real solve (returned a small chunk, not the unchunked
    sentinel) — a silently-dropped budget would run unchunked and match
    trivially, which this catches."""
    import gradwave.core.batch as batch

    real = batch._dense_band_chunk
    seen: dict = {"chunks": []}

    def spy(n_grid, nk, device, elem_bytes):
        out = real(n_grid, nk, device, elem_bytes)
        seen["chunks"].append(out)
        return out

    monkeypatch.setattr(batch, "_dense_band_chunk", spy)
    # a deliberately tiny budget forces chunk == 1 on this small cell (the value
    # is bridged GB→bytes → GRADWAVE_CPU_DENSE_BUDGET, which forces chunking
    # regardless of the auto estimator's size trigger).
    res = _run(_write(tmp_path, mem="dense_budget_gb: 1.0e-6", **_MULTIK))

    assert res.converged
    # the budget was live and genuinely chunked (a real chunk below the unchunked
    # sentinel appeared during the solve/density build)
    assert seen["chunks"], "the dense band-chunk helper was never called"
    assert min(seen["chunks"]) < 1_000_000, (
        f"dense_budget_gb did not chunk the bands: min chunk {min(seen['chunks'])}")

    e_ref = float(multik_reference.energies.free_energy)
    e_c = float(res.energies.free_energy)
    assert abs(e_c - e_ref) < 1e-8, (
        f"dense_budget_gb shifted the converged energy: dE = {e_c - e_ref:.3e} eV")
    drho = float((res.rho - multik_reference.rho).abs().max())
    assert drho < 1e-8, f"dense_budget_gb density drift {drho:.3e}"


# ---------------------------------------------------------------------------
# 3. subspace_storage complex64 — a PRECISION tradeoff (NOT bit-exact).
# ---------------------------------------------------------------------------


def test_subspace_storage_c64_precision_through_api(tmp_path, monkeypatch):
    """complex64 subspace storage keeps the apply + Rayleigh–Ritz in fp64 but
    stores V/HV in fp32 — a ~1e-6 eV PRECISION lever, NOT bit-exact. Because c64
    storage plateaus near the fp32 floor (~1e-6 eV eigenpairs) and cannot reach a
    tight density residual, both storages are run at a moderate rhotol both can
    reach and compared energy-to-energy: the converged energies must agree to the
    documented ~1e-6 eV floor and NO WORSE (a regression that loosened that floor
    must fail here, so the bound is kept tight). The spy confirms c64 storage was
    actually selected during the c64 solve — else a silent no-op would run fp64
    and match to ~0, slipping a broken knob through."""
    import importlib

    # the real submodule, not the re-exported `davidson` function (shadowing)
    davmod = importlib.import_module("gradwave.solvers.davidson")
    real = davmod._subspace_storage_c64
    seen: dict = {"modes": []}

    def spy(x0):
        out = real(x0)
        seen["modes"].append(bool(out))
        return out

    monkeypatch.setattr(davmod, "_subspace_storage_c64", spy)

    # a rhotol c64 storage can actually reach (its fp32 plateau is ~a few e-7)
    kw = dict(etol=1.0e-8, rhotol=1.0e-6, **_MULTIK)
    res128 = _run(_write(tmp_path, mem="subspace_storage: complex128", **kw))
    assert all(m is False for m in seen["modes"])  # fp64 storage on the c128 run
    seen["modes"].clear()
    res64 = _run(_write(tmp_path, mem="subspace_storage: complex64", **kw))
    assert res64.gamma_real is False and res128.gamma_real is False
    assert seen["modes"] and all(m is True for m in seen["modes"])  # c64 engaged

    assert res128.converged and res64.converged
    e128 = float(res128.energies.free_energy)
    e64 = float(res64.energies.free_energy)
    de = abs(e64 - e128)
    # NOT bit-exact, but must sit within the documented ~1e-6 eV c64 floor. The
    # bound is deliberately tight so a regression that degraded c64 precision
    # (e.g. dropping the fp64 RR eigensolve) fails here instead of passing loose.
    assert de < 1e-6, f"complex64 storage exceeded its ~1e-6 eV floor: dE = {de:.3e} eV"


# ---------------------------------------------------------------------------
# 4. GRADWAVE_GAMMA_REAL — the Γ real-WF path, bit-exact to complex + fallback.
# ---------------------------------------------------------------------------


def test_gamma_real_bit_exact_through_api(tmp_path, monkeypatch):
    """On an eligible Γ-only cell, forcing the real half-sphere path
    (GRADWAVE_GAMMA_REAL=1) reproduces the complex path (=0) byte-for-byte. The
    gamma_real flag on the result confirms the real path actually RAN under =1
    (so the test is not silently comparing complex-to-complex)."""
    import gradwave.scf.loop as loop

    torch.set_num_threads(8)
    path = _write(tmp_path, mesh=(1, 1, 1), symmetry=True)  # single Γ point

    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "0")
    res_c = _run(path)
    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "1")
    res_g = _run(path)

    assert res_c.converged and res_g.converged
    assert res_c.gamma_real is False  # complex path
    assert res_g.gamma_real is True   # real Γ path genuinely engaged

    e_c = float(res_c.energies.free_energy)
    e_g = float(res_g.energies.free_energy)
    assert abs(e_g - e_c) <= 1e-10 * abs(e_c), f"Γ-real not bit-exact: dE = {e_g - e_c:.3e} eV"
    drho = float((res_g.rho - res_c.rho).abs().max())
    assert drho < 1e-8, f"Γ-real density drift {drho:.3e}"


def test_gamma_real_auto_falls_back_on_multi_k(tmp_path, monkeypatch):
    """`auto` must NOT engage the real path on a multi-k cell (it is only
    provably safe at a single Γ point): the run falls back to the complex path
    (gamma_real False) and still converges — a silent mis-fire on multi-k would
    be a correctness bug."""
    import gradwave.scf.loop as loop

    torch.set_num_threads(8)
    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "auto")
    res = _run(_write(tmp_path, **_MULTIK))

    assert res.converged
    assert res.gamma_real is False
