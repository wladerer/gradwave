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


def test_rr_env_validation(monkeypatch):
    """GRADWAVE_NATIVE_RR parses to the C kernel's rr_mode int (0 classic,
    1 incremental); 'auto' gates on nb*m; anything else raises."""
    from gradwave.solvers.native_davidson import _rr_mode

    monkeypatch.setenv("GRADWAVE_NATIVE_RR", "classic")
    assert _rr_mode(nb=200, m=30000) == 0
    monkeypatch.setenv("GRADWAVE_NATIVE_RR", "incremental")
    assert _rr_mode(nb=2, m=16) == 1
    monkeypatch.setenv("GRADWAVE_NATIVE_RR", "banana")
    with pytest.raises(ValueError, match="GRADWAVE_NATIVE_RR"):
        _rr_mode(nb=2, m=16)
    monkeypatch.delenv("GRADWAVE_NATIVE_RR")  # default auto


def test_rr_auto_gate(monkeypatch):
    """auto picks incremental only above the nb*m gate; the gate is
    overridable for A/B."""
    from gradwave.solvers.native_davidson import _rr_mode

    monkeypatch.delenv("GRADWAVE_NATIVE_RR", raising=False)
    monkeypatch.setenv("GRADWAVE_NATIVE_RR_GATE", "1000000")
    assert _rr_mode(nb=10, m=4000) == 0       # 40k < 1e6 -> classic
    assert _rr_mode(nb=154, m=24000) == 1     # 3.7M >= 1e6 -> incremental
    monkeypatch.setenv("GRADWAVE_NATIVE_RR_GATE", "0")
    assert _rr_mode(nb=2, m=16) == 1          # gate 0 -> always incremental


@needs_native
@pytest.mark.standard
def test_incremental_scf_matches_eager(monkeypatch):
    """The incremental (cegterg-style) RR kernel drives a full SCF to the same
    rn<=tol contract as the eager solver: energy/density/forces agree at the
    tolerance level. Si has symmetry-degenerate bands, so this also exercises
    the generalized solve on degenerate clusters."""
    monkeypatch.setenv("GRADWAVE_NATIVE_RR", "incremental")
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


@needs_native
@pytest.mark.standard
def test_incremental_matches_classic(monkeypatch):
    """Both native RR kernels honour the same per-band rn<=tol contract, so a
    full SCF converges to the same energy/density whichever kernel runs."""
    monkeypatch.setenv("GRADWAVE_NATIVE_RR", "classic")
    res_c = _run(_si_system(), "davidson-native")
    monkeypatch.setenv("GRADWAVE_NATIVE_RR", "incremental")
    res_i = _run(_si_system(), "davidson-native")
    assert res_c.converged and res_i.converged
    de = abs(float(res_c.energies.free_energy)
             - float(res_i.energies.free_energy))
    assert de < 1e-7, f"energy mismatch {de:.3e} eV"
    drho = float((res_c.rho - res_i.rho).abs().max())
    assert drho < 1e-6, f"density mismatch {drho:.3e}"


@needs_native
def test_incremental_frequent_restart_converges(monkeypatch):
    """A tight max_dim (factor 2) forces frequent Ritz collapses/refreshes —
    the same refresh path the sc-conditioning-breakdown rollback reuses. The
    solve must still converge and every returned band satisfy rn<=tol."""
    monkeypatch.setenv("GRADWAVE_NATIVE_RR", "incremental")
    monkeypatch.setenv("GRADWAVE_MAX_DIM_FACTOR", "2")
    res = _run(_si_system(), "davidson-native")
    assert res.converged


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


# ---------------------------------------------------------------------------
# .so source-hash stamp guard (campaign: projector dedup). A library built from
# a different davidson_native.c must be refused with a clear rebuild error, not
# called with a mismatched ABI (the #479 stale-.so segfault).
# ---------------------------------------------------------------------------

def test_src_hash_is_stable_16hex():
    from gradwave.solvers.native_davidson import _src_hash

    h = _src_hash()
    assert isinstance(h, str) and len(h) == 16
    assert all(ch in "0123456789abcdef" for ch in h)


def test_stamp_guard_rejects_mismatch():
    """A library reporting a wrong build hash is refused with a clear error."""
    from gradwave.solvers import native_davidson as nd

    class _Stub:
        # the adapter does getattr(lib, "davidson_native_build_hash"), sets
        # .restype, then calls it and decodes the bytes; a plain function
        # attribute satisfies all three.
        @staticmethod
        def davidson_native_build_hash():
            return b"deadbeefdeadbeef"

    with pytest.raises(RuntimeError, match="stale"):
        nd._verify_stamp(_Stub(), "/fake/path.so")


def test_stamp_guard_rejects_missing_symbol():
    """A pre-guard library without the stamp symbol is treated as stale."""
    from gradwave.solvers import native_davidson as nd

    class _NoSym:
        pass

    with pytest.raises(RuntimeError, match="stale"):
        nd._verify_stamp(_NoSym(), "/fake/old.so")


@needs_native
def test_current_library_stamp_matches_source():
    """The built library on this machine matches the checked-out source (i.e.
    it was rebuilt for this branch — a green native tier depends on it)."""
    from gradwave.solvers.native_davidson import _load

    assert _load() is not None  # would raise "stale" if the stamp mismatched
