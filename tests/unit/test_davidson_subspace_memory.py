"""Subspace-footprint knobs for the large-nk / slab GPU-memory relief.

The batched Davidson holds V and HV, each (nk, max_dim_factor·nb, npw_max)
complex128; for a (6,6,1)=36-k slab with max_dim_factor·nb ≈ 240 and npw ≈ 13k
that pair is ~5.7 GB — the diagnosed RTX-3050 OOM. Three opt-in knobs cut that
footprint:

  * GRADWAVE_MAX_DIM_FACTOR / GRADWAVE_SUBSPACE_BUDGET_GB — halve the grown
    subspace; EXACT in the converged eigenpairs.
  * GRADWAVE_GPU_DENSE_BUDGET — shrink the dense-grid FFT-box band chunk;
    BIT-EXACT.
  * GRADWAVE_SUBSPACE_STORAGE=complex64 — store V/HV in complex64 with the apply
    and RR eigensolve kept in fp64; PRECISION lever (plateaus near the fp32
    floor, ~1e-6 eV).

Most tests are assertion/synthetic and need no SCF (byte-count arithmetic, the
resolution helpers, a small dense synthetic operator). The convergence-
preservation check on a real tiny SCF is `test_complex64_storage_scf_energy`.
"""

import numpy as np
import pytest
import torch

from gradwave.core.batch import _dense_band_chunk, _gpu_dense_budget_bytes
from gradwave.solvers.davidson import (
    _resolve_max_dim_factor,
    _subspace_storage_c64,
    davidson_batched,
)

TOL = 1e-9


def _make_operator(nk, npw, seed=0, spread=(1.0, 100.0)):
    """Random Hermitian H_k with a kinetic-like spectrum, exposed through a
    dtype-POLYMORPHIC apply (recomputes in the block dtype) — the same synthetic
    the fp32-expansion / CheFSI tests use, satisfied by the real
    BatchedHamiltonian.apply."""
    torch.manual_seed(seed)
    diag = torch.linspace(spread[0], spread[1], npw, dtype=torch.float64)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        h[k] = 0.5 * (a + a.conj().T) + torch.diag(diag.to(torch.complex128))
        h[k] = 0.5 * (h[k] + h[k].conj().T)
    mask = torch.ones(nk, npw, dtype=torch.bool)
    t = diag.expand(nk, npw).contiguous()

    def apply(c):
        return torch.einsum("kij,kbj->kbi", h.to(c.dtype), c)

    return h, apply, mask, t


# ---------------------------------------------------------------------------
# Lever 1: max_dim_factor override + memory auto-gate (EXACT in the result).
# ---------------------------------------------------------------------------


def test_max_dim_factor_env_override(monkeypatch):
    """GRADWAVE_MAX_DIM_FACTOR forces the factor; < 2 is rejected."""
    monkeypatch.setenv("GRADWAVE_MAX_DIM_FACTOR", "2")
    assert _resolve_max_dim_factor(36, 60, 13000, 16, requested=4) == 2
    monkeypatch.setenv("GRADWAVE_MAX_DIM_FACTOR", "6")
    assert _resolve_max_dim_factor(36, 60, 13000, 16, requested=4) == 6
    monkeypatch.setenv("GRADWAVE_MAX_DIM_FACTOR", "1")
    with pytest.raises(ValueError, match="GRADWAVE_MAX_DIM_FACTOR"):
        _resolve_max_dim_factor(36, 60, 13000, 16, requested=4)


def test_max_dim_factor_autogate_halves_over_budget(monkeypatch):
    """The auto-gate drops factor 4 -> 2 once the projected V+HV bytes exceed
    the budget, and leaves a comfortably-fitting solve untouched."""
    monkeypatch.delenv("GRADWAVE_MAX_DIM_FACTOR", raising=False)
    nk, nb, npw = 36, 60, 13000  # the OOMing slab shape
    # peak V+HV at factor 4 = 2 * 36 * 4*60 * 13000 * 16 B ≈ 3.35 GiB.
    peak4 = 2.0 * nk * 4 * nb * npw * 16
    assert peak4 / (1 << 30) == pytest.approx(3.35, abs=0.05)

    # A 3 GiB budget cannot hold factor 4 (3.35 GiB) but holds factor 2 (1.67 GiB).
    monkeypatch.setenv("GRADWAVE_SUBSPACE_BUDGET_GB", "3.0")
    assert _resolve_max_dim_factor(nk, nb, npw, 16, requested=4) == 2
    # A generous budget leaves the requested factor alone.
    monkeypatch.setenv("GRADWAVE_SUBSPACE_BUDGET_GB", "16.0")
    assert _resolve_max_dim_factor(nk, nb, npw, 16, requested=4) == 4
    # Never drops below the floor of 2, however tight the budget.
    monkeypatch.setenv("GRADWAVE_SUBSPACE_BUDGET_GB", "0.001")
    assert _resolve_max_dim_factor(nk, nb, npw, 16, requested=4) == 2


def test_max_dim_factor_2_matches_factor_4(monkeypatch):
    """Halving max_dim_factor converges to the SAME eigenpairs (a different
    restart path), so it is exact in the result — the byte win is free."""
    monkeypatch.delenv("GRADWAVE_SUBSPACE_STORAGE", raising=False)
    nk, npw, nb = 4, 90, 8
    _, apply, mask, t = _make_operator(nk, npw, seed=5)
    torch.manual_seed(3)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)

    r4 = davidson_batched(apply, x0.clone(), t, mask, tol=TOL, max_iter=200,
                          max_dim_factor=4)
    r2 = davidson_batched(apply, x0.clone(), t, mask, tol=TOL, max_iter=200,
                          max_dim_factor=2)
    assert float(r4.residual_norms.max()) < TOL
    assert float(r2.residual_norms.max()) < TOL
    assert float((r4.eigenvalues - r2.eigenvalues).abs().max()) < 1e-9


def test_subspace_bytes_halve_with_factor_2():
    """The deliverable, as arithmetic: V and HV bytes scale linearly in
    max_dim_factor, so factor 2 is exactly half the factor-4 footprint."""
    nk, nb, npw, elem = 36, 60, 13000, 16
    bytes4 = 2 * nk * (4 * nb) * npw * elem
    bytes2 = 2 * nk * (2 * nb) * npw * elem
    assert bytes2 == bytes4 // 2


# ---------------------------------------------------------------------------
# Lever 2: dense-grid GPU byte budget (BIT-EXACT — smaller batches only).
# ---------------------------------------------------------------------------


def test_gpu_dense_budget_env(monkeypatch):
    monkeypatch.setenv("GRADWAVE_GPU_DENSE_BUDGET", "1e8")
    assert _gpu_dense_budget_bytes() == pytest.approx(1e8)
    monkeypatch.delenv("GRADWAVE_GPU_DENSE_BUDGET", raising=False)
    assert _gpu_dense_budget_bytes() == pytest.approx(4e8)  # historical default
    monkeypatch.setenv("GRADWAVE_GPU_DENSE_BUDGET", "0")
    with pytest.raises(ValueError, match="GRADWAVE_GPU_DENSE_BUDGET"):
        _gpu_dense_budget_bytes()


def test_gpu_dense_budget_lowers_chunk(monkeypatch):
    """Lowering the budget lowers the band chunk (=> smaller FFT-box peak). The
    device.type=='cuda' branch runs on a CPU-only host: torch.device('cuda') is
    a descriptor, and the chunk is pure integer arithmetic (no CUDA touched)."""
    dev = torch.device("cuda")
    n_grid, nk, elem = 8_000, 8, 16  # sized so the default chunk is > 1
    monkeypatch.delenv("GRADWAVE_GPU_DENSE_BUDGET", raising=False)
    chunk_default = _dense_band_chunk(n_grid, nk, dev, elem)
    monkeypatch.setenv("GRADWAVE_GPU_DENSE_BUDGET", "1e8")  # quarter of 4e8
    chunk_small = _dense_band_chunk(n_grid, nk, dev, elem)
    assert chunk_small < chunk_default
    assert chunk_small == pytest.approx(chunk_default / 4, rel=0.02)
    # CPU path never chunks regardless of the budget.
    assert _dense_band_chunk(n_grid, nk, torch.device("cpu"), elem) >= 1_000_000


# ---------------------------------------------------------------------------
# Lever 3: complex64 subspace STORAGE, fp64 compute (PRECISION lever).
# ---------------------------------------------------------------------------


def test_subspace_storage_env(monkeypatch):
    """complex64 storage gates on the env AND a complex128 solve (a complex64
    x0 is already a low-precision draft); the env value is validated."""
    x128 = torch.zeros(1, 1, 1, dtype=torch.complex128)
    x64 = torch.zeros(1, 1, 1, dtype=torch.complex64)
    monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", "complex64")
    assert _subspace_storage_c64(x128) is True
    assert _subspace_storage_c64(x64) is False
    monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", "complex128")
    assert _subspace_storage_c64(x128) is False
    monkeypatch.delenv("GRADWAVE_SUBSPACE_STORAGE", raising=False)
    assert _subspace_storage_c64(x128) is False  # default off
    monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", "half")
    with pytest.raises(ValueError, match="GRADWAVE_SUBSPACE_STORAGE"):
        _subspace_storage_c64(x128)


def test_subspace_storage_bytes_halve():
    """The deliverable, as arithmetic: complex64 storage is 8 B/elem vs the
    complex128 16 B, so V/HV are exactly half the bytes."""
    shape = (36, 240, 13000)  # (nk, max_dim, npw_max)
    c128 = torch.empty(0, dtype=torch.complex128)
    c64 = torch.empty(0, dtype=torch.complex64)
    assert c64.element_size() == 8 and c128.element_size() == 16
    n = int(np.prod(shape))
    assert n * c64.element_size() == (n * c128.element_size()) // 2


def test_complex64_storage_matches_fp64_storage(monkeypatch):
    """complex64 storage returns complex128 eigenvectors and reproduces the
    fp64-storage eigenvalues to the fp32 floor (~1e-5 on this 1..30 spectrum),
    while the apply COMPUTE stays fp64 (distinct from fp32-expansion)."""
    monkeypatch.delenv("GRADWAVE_FP32_EXPANSION", raising=False)
    nk, npw, nb = 3, 70, 6
    _, apply, mask, t = _make_operator(nk, npw, seed=8, spread=(1.0, 30.0))
    torch.manual_seed(4)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)

    monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", "complex128")
    r128 = davidson_batched(apply, x0.clone(), t, mask, tol=1e-6, max_iter=200)
    monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", "complex64")
    r64 = davidson_batched(apply, x0.clone(), t, mask, tol=1e-6, max_iter=200)

    assert r64.eigenvectors.dtype == torch.complex128  # upcast on return
    assert float((r64.eigenvalues - r128.eigenvalues).abs().max()) < 1e-4


def test_complex64_storage_does_not_certify_to_fp64(monkeypatch):
    """Sanity on the documented limitation: at a tight fp64 tol, complex64
    storage plateaus at the fp32 floor rather than reaching it — its residual
    norms stay well above what fp64 storage attains (so the mode is honestly
    a ~1e-6 eV lever, not a silent fp64 substitute)."""
    monkeypatch.delenv("GRADWAVE_FP32_EXPANSION", raising=False)
    nk, npw, nb = 2, 60, 5
    _, apply, mask, t = _make_operator(nk, npw, seed=9, spread=(1.0, 30.0))
    torch.manual_seed(1)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)

    monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", "complex128")
    r128 = davidson_batched(apply, x0.clone(), t, mask, tol=TOL, max_iter=200)
    monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", "complex64")
    r64 = davidson_batched(apply, x0.clone(), t, mask, tol=TOL, max_iter=200)

    assert float(r128.residual_norms.max()) < TOL  # fp64 storage reaches tol
    assert float(r64.residual_norms.max()) > TOL   # c64 storage plateaus above
    # but the eigenvalues still agree to the fp32 floor
    assert float((r64.eigenvalues - r128.eigenvalues).abs().max()) < 1e-4


# ---------------------------------------------------------------------------
# Convergence preservation on a real (tiny) SCF — the one test that needs
# gradwave compute. Run on asus with OMP_NUM_THREADS=2 (tiny cells, seconds).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pseudo_name,smearing,width",
    [("Si_ONCV_PBE-1.2.upf", None, 0.0),        # insulator
     ("Al_ONCV_PBE-1.2.upf", "gaussian", 0.1)],  # metal
)
def test_complex64_storage_scf_energy(monkeypatch, pseudo_name, smearing, width):
    """A tiny insulator (Si) and metal (Al) SCF: complex64 subspace storage must
    reproduce the fp64-storage total energy to ~1e-6 eV. The ~fp32 eigenpair
    floor is looser than a tight fp64 SCF, so both runs use a matching moderate
    tolerance and the comparison is energy-to-energy."""
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system
    from tests.helpers import PSEUDOS, RY

    torch.set_num_threads(2)
    a = 5.43 if pseudo_name.startswith("Si") else 4.05
    if pseudo_name.startswith("Si"):
        cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
        pos = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        znum = [0, 0]
    else:
        cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
        pos = np.array([[0.0, 0.0, 0.0]])
        znum = [0]
    up = parse_upf(str(PSEUDOS / pseudo_name))

    def run(mode):
        monkeypatch.setenv("GRADWAVE_SUBSPACE_STORAGE", mode)
        system = setup_system(cell, pos, znum, [up], ecut=12 * RY, kmesh=(2, 2, 2))
        kw = {} if smearing is None else {"smearing": smearing, "width": width}
        return scf(system, LDA_PW92(), etol=1e-8, rhotol=1e-7, verbose=False, **kw)

    r128 = run("complex128")
    r64 = run("complex64")
    e128 = float(r128.energies.free_energy)
    e64 = float(r64.energies.free_energy)
    assert abs(e128 - e64) < 1e-6, f"{pseudo_name}: dE = {e128 - e64:.3e} eV"
    assert torch.allclose(r128.eigenvalues, r64.eigenvalues, atol=1e-5)
