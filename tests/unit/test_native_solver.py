"""The native (FFTW+CBLAS+LAPACKE) Davidson solver: registry, scope gates,
fallback semantics, and SCF-level correctness vs the eager solver.

The native kernel reimplements the identical algorithm as one C call with
OpenMP over k (solvers/native/davidson_native.c); the adapter
(solvers/native_davidson.py) gates on solve scope and falls back to the eager
``davidson_batched`` transparently, recording why in the diagnostics.

Correctness tests need the compiled library (scripts/build_native_solver.sh —
CI builds it in the standard-tier job; locally build once per machine). The
registry/scope/fallback tests run without it.
"""

from __future__ import annotations

import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.native_davidson import native_available
from gradwave.solvers.registry import available
from tests.helpers import PSEUDOS, RY, SI_ONCV, si_fcc

needs_native = pytest.mark.skipif(
    not native_available(),
    reason="native solver library not built (scripts/build_native_solver.sh)")


def _si_system():
    si = parse_upf(str(PSEUDOS / SI_ONCV))
    cell, pos = si_fcc()
    pos = pos.copy()
    pos[1, 0] += 0.10
    return setup_system(cell, pos, [0, 0], [si], ecut=10 * RY,
                        kmesh=(2, 2, 2), use_symmetry=False)


def _run(system, eigensolver, max_iter=80):
    return scf(system, LDA_PW92(), smearing="gaussian", width=0.1,
               max_iter=max_iter, etol=1e-11, rhotol=1e-9, diago_tol=1e-10,
               verbose=False, eigensolver=eigensolver)


def test_registered():
    assert "davidson-native" in available()


def test_missing_library_is_an_error_not_a_silent_eager_run(monkeypatch):
    """Explicitly requesting davidson-native without the .so must raise with
    the build command — never quietly run the eager solver."""
    import gradwave.solvers.native_davidson as nd

    monkeypatch.setenv("GRADWAVE_NATIVE_SO", "/nonexistent/libdavnative.so")
    monkeypatch.setattr(nd, "_lib", None)
    monkeypatch.setattr(nd, "_lib_path", None)
    with pytest.raises(RuntimeError, match="build_native_solver"):
        nd.native_davidson_adapter(
            lambda c: c, torch.zeros(1, 2, 4, dtype=torch.complex128),
            torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), tol=1e-9)


@needs_native
def test_composed_apply_falls_back_with_reason():
    """A non-BatchedHamiltonian apply (test operator / hybrid / meta-GGA
    closure) solves via the eager path and says so in diagnostics."""
    from gradwave.solvers.native_davidson import native_davidson_adapter

    nk, nb, m = 1, 2, 16
    gen = torch.Generator().manual_seed(7)
    a = torch.randn(m, m, generator=gen, dtype=torch.float64)
    hmat = (a + a.T).to(torch.complex128)

    def apply_H(c):
        return torch.einsum("ij,kbj->kbi", hmat, c)

    x0 = torch.randn(nk, nb, m, generator=gen,
                     dtype=torch.float64).to(torch.complex128)
    t = torch.arange(m, dtype=torch.float64)[None].repeat(nk, 1)
    mask = torch.ones(nk, m, dtype=torch.bool)
    r = native_davidson_adapter(apply_H, x0, t, mask, tol=1e-9)
    assert "composed apply" in r.diagnostics["fallback_reason"]
    ref = torch.linalg.eigvalsh(hmat.real)[:nb]
    assert torch.allclose(r.eigenvalues[0], ref, atol=1e-8)


@needs_native
def test_fp32_expansion_mode_falls_back(monkeypatch):
    """The fp32-expansion env mode belongs to the eager solver; the adapter
    must route eager (the run proceeds, no crash)."""
    monkeypatch.setenv("GRADWAVE_FP32_EXPANSION", "on")
    res = _run(_si_system(), "davidson-native", max_iter=2)
    assert res.n_iter <= 2  # ran (unconverged is fine)


@needs_native
@pytest.mark.standard
def test_native_scf_matches_eager_energy_density_forces():
    """Full SCF with eigensolver=davidson-native == eager davidson to the
    same contract as the k-parallel pin: per-band rn <= tol at each k's own
    final RR (retirement default on), so energy/density/forces agree at the
    tolerance level, not round-off."""
    res_e = _run(_si_system(), "davidson")
    res_n = _run(_si_system(), "davidson-native")
    assert res_e.converged and res_n.converged
    de = abs(float(res_e.energies.free_energy)
             - float(res_n.energies.free_energy))
    assert de < 1e-7, f"energy mismatch {de:.3e} eV"
    drho = float((res_e.rho - res_n.rho).abs().max())
    assert drho < 1e-6, f"density mismatch {drho:.3e}"
    f_e, f_n = forces(res_e), forces(res_n)
    df = float((f_e - f_n).abs().max())
    assert df < 1e-5, f"force mismatch {df:.3e} eV/Ang"
    assert float(f_e.abs().max()) > 1e-2


@needs_native
@pytest.mark.standard
def test_native_batch_mode_matches_eager_tightly(monkeypatch):
    """GRADWAVE_NATIVE_RETIRE=off reproduces the eager uniform-batch
    trajectory (same rounds structure), so the converged results agree an
    order tighter than the retirement contract."""
    monkeypatch.setenv("GRADWAVE_NATIVE_RETIRE", "off")
    res_e = _run(_si_system(), "davidson")
    res_n = _run(_si_system(), "davidson-native")
    assert res_e.converged and res_n.converged
    de = abs(float(res_e.energies.free_energy)
             - float(res_n.energies.free_energy))
    assert de < 1e-9, f"energy mismatch {de:.3e} eV"


@needs_native
def test_retire_env_validation(monkeypatch):
    from gradwave.solvers.native_davidson import _retire_on

    monkeypatch.setenv("GRADWAVE_NATIVE_RETIRE", "banana")
    with pytest.raises(ValueError, match="GRADWAVE_NATIVE_RETIRE"):
        _retire_on()
    monkeypatch.setenv("GRADWAVE_NATIVE_RETIRE", "off")
    assert _retire_on() is False
    monkeypatch.delenv("GRADWAVE_NATIVE_RETIRE")
    assert _retire_on() is True


def test_resolve_thread_split_gate(monkeypatch):
    """The few-k mode gate (solver-agnostic pure logic, no library needed).

    auto always resolves to kpar (fewk was measured a non-win — memory-
    bandwidth-bound — so it is never auto-selected and no solve regresses
    relative to the pre-split kernel). kpar always resolves to (nthreads, 1),
    byte-identical to that kernel. fewk is an explicit opt-in only."""
    from gradwave.solvers.native_davidson import _resolve_thread_split

    monkeypatch.delenv("GRADWAVE_NATIVE_KMODE", raising=False)
    monkeypatch.delenv("GRADWAVE_NATIVE_FEWK_OUTER", raising=False)
    # auto -> kpar regardless of nk vs nthreads (few-k or many-k alike).
    assert _resolve_thread_split(8, 16) == (16, 1)   # few k: still kpar
    assert _resolve_thread_split(16, 16) == (16, 1)
    assert _resolve_thread_split(36, 16) == (16, 1)  # al 4^3-mesh regression

    monkeypatch.setenv("GRADWAVE_NATIVE_KMODE", "kpar")
    assert _resolve_thread_split(8, 16) == (16, 1)   # forced kpar

    monkeypatch.setenv("GRADWAVE_NATIVE_KMODE", "fewk")
    assert _resolve_thread_split(8, 16) == (8, 2)
    monkeypatch.setenv("GRADWAVE_NATIVE_FEWK_OUTER", "4")
    assert _resolve_thread_split(8, 16) == (4, 4)    # A/B split override
    monkeypatch.setenv("GRADWAVE_NATIVE_FEWK_OUTER", "1")
    assert _resolve_thread_split(8, 16) == (1, 16)   # serial-outer, full BLAS
    monkeypatch.setenv("GRADWAVE_NATIVE_FEWK_OUTER", "99")
    assert _resolve_thread_split(8, 16) == (16, 1)   # capped at nthreads

    monkeypatch.setenv("GRADWAVE_NATIVE_KMODE", "banana")
    with pytest.raises(ValueError, match="GRADWAVE_NATIVE_KMODE"):
        _resolve_thread_split(8, 16)


@needs_native
@pytest.mark.standard
def test_fewk_mode_matches_eager(monkeypatch):
    """The few-k thread split changes only which cores do the arithmetic, not
    the algorithm: a fewk solve (outer=1, all BLAS threads) agrees with the
    eager reference to the same contract as the default kpar native path."""
    monkeypatch.setenv("GRADWAVE_NATIVE_KMODE", "fewk")
    monkeypatch.setenv("GRADWAVE_NATIVE_FEWK_OUTER", "1")  # serial-outer path
    res_e = _run(_si_system(), "davidson")
    res_n = _run(_si_system(), "davidson-native")
    assert res_e.converged and res_n.converged
    de = abs(float(res_e.energies.free_energy)
             - float(res_n.energies.free_energy))
    assert de < 1e-7, f"energy mismatch {de:.3e} eV"


@needs_native
def test_env_threads_respected():
    """The native solve uses torch.get_num_threads() — the same knob the rest
    of gradwave (and the per-rank pinning of distributed/k_parallel) uses; a
    1-thread setting must not crash or change results."""
    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        res = _run(_si_system(), "davidson-native", max_iter=3)
        assert res.n_iter <= 3
    finally:
        torch.set_num_threads(prev)
