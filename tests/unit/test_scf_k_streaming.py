"""k-point STREAMING in the NC SCF (``scf.loop.scf``, the ``k_chunk`` knob).

The batched Davidson holds V and HV, each ``nk·m·npw`` complex — so the resident
subspace scales with the TOTAL k-count and OOMs a small GPU on a many-k slab.
``k_chunk`` solves the bands in sequential chunks, so the resident subspace is
``nk_chunk·m·npw`` regardless of nk (the memory lever). It is the sequential,
single-device analogue of ``gradwave.distributed``'s k-sharding: eigenvalues are
collected across chunks for the shared Fermi level and the density accumulates
over chunks, so it is algebraically identical to the all-k solve.

Pinned here:
- resolution: ``k_chunk`` unset / non-positive / ``>= nk`` selects the all-k path
  (None), and ``GRADWAVE_K_CHUNK`` supplies the default when unset;
- resident-subspace reduction: the Davidson block the solver actually receives
  has at most ``k_chunk`` k-rows (so V/HV ∝ k_chunk, not nk);
- correctness: a k-streamed SCF gives the SAME converged energy, density, AND
  forces as the all-k batched SCF, to tight tol (pure loop restructuring).

Two-atom Si (valence-only ONCV, no NLCC core → forces need no XC), an 8-k
unshifted mesh with symmetry OFF so nk is the full mesh, and one atom nudged off
the ideal site so the forces are nonzero to compare.
"""

import importlib
from unittest.mock import patch

import pytest

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import _resolve_k_chunk, scf, setup_system
from tests.helpers import PSEUDOS, RY, SI_ONCV, si_fcc

_KMESH = (2, 2, 2)  # 8 k-points with symmetry off
_NK = 8


def _si_system(kmesh=_KMESH):
    si = parse_upf(str(PSEUDOS / SI_ONCV))
    cell, pos = si_fcc()
    pos = pos.copy()
    pos[1, 0] += 0.10  # break the equilibrium so forces are nonzero to compare
    return setup_system(cell, pos, [0, 0], [si], ecut=10 * RY,
                        kmesh=kmesh, use_symmetry=False)


def _run(system, k_chunk, max_iter=80):
    return scf(system, LDA_PW92(), smearing="gaussian", width=0.1,
               max_iter=max_iter, etol=1e-11, rhotol=1e-9, diago_tol=1e-10,
               verbose=False, k_chunk=k_chunk)


def test_resolve_k_chunk_semantics(monkeypatch):
    """None / non-positive / >= nk → all-k (None); a valid chunk passes through;
    GRADWAVE_K_CHUNK is the default when the arg is unset."""
    monkeypatch.delenv("GRADWAVE_K_CHUNK", raising=False)
    assert _resolve_k_chunk(None, _NK) is None       # default: all-k
    assert _resolve_k_chunk(_NK, _NK) is None         # == nk → all-k
    assert _resolve_k_chunk(100, _NK) is None         # > nk → all-k
    assert _resolve_k_chunk(0, _NK) is None           # non-positive → all-k
    assert _resolve_k_chunk(-2, _NK) is None
    assert _resolve_k_chunk(3, _NK) == 3              # valid chunk
    assert _resolve_k_chunk(1, _NK) == 1
    # env supplies the default only when the explicit arg is None
    monkeypatch.setenv("GRADWAVE_K_CHUNK", "2")
    assert _resolve_k_chunk(None, _NK) == 2
    assert _resolve_k_chunk(4, _NK) == 4              # explicit arg still wins
    monkeypatch.setenv("GRADWAVE_K_CHUNK", "not-an-int")
    with pytest.raises(ValueError):
        _resolve_k_chunk(None, _NK)


def test_resident_subspace_scales_with_k_chunk():
    """The Davidson block the solver receives has at most k_chunk k-rows, so the
    resident V/HV subspace (nk_block·m·npw) tracks k_chunk, not the total nk. The
    all-k path (k_chunk=None) hands the solver the full nk-row block."""
    # the real submodule, not the `davidson` FUNCTION that solvers/__init__
    # re-exports under the same dotted name (attribute shadowing)
    davmod = importlib.import_module("gradwave.solvers.davidson")
    real = davmod.davidson_batched
    seen = {"max_nk": 0}

    def spy(h_apply, x0, *a, **k):
        seen["max_nk"] = max(seen["max_nk"], int(x0.shape[0]))
        return real(h_apply, x0, *a, **k)

    with patch.object(davmod, "davidson_batched", spy):
        seen["max_nk"] = 0
        _run(_si_system(), None, max_iter=2)
        assert seen["max_nk"] == _NK  # all-k block

        seen["max_nk"] = 0
        _run(_si_system(), 3, max_iter=2)
        assert seen["max_nk"] == 3  # chunks 3,3,2 → largest block is 3 k

        seen["max_nk"] = 0
        _run(_si_system(), 1, max_iter=2)
        assert seen["max_nk"] == 1  # one k at a time


def test_fock_incompatible_with_k_chunk():
    """Hybrid Fock couples orbitals across the whole BZ — a per-chunk solve
    cannot see it, so the combination is rejected up front (like distributed)."""
    with pytest.raises(NotImplementedError, match="Fock"):
        scf(_si_system(), LDA_PW92(), smearing="gaussian", width=0.1,
            max_iter=1, verbose=False, k_chunk=2, fock=object())


@pytest.mark.standard
def test_streamed_matches_batched_energy_density_forces():
    """A k-streamed SCF converges to the SAME total energy, density, and forces
    as the all-k batched SCF (the density is Σ_k either way)."""
    res_batched = _run(_si_system(), None)
    res_stream = _run(_si_system(), 3)  # uneven chunks 3,3,2
    assert res_batched.converged and res_stream.converged

    de = abs(float(res_batched.energies.free_energy)
             - float(res_stream.energies.free_energy))
    assert de < 1e-9, f"energy mismatch {de:.3e} eV"

    drho = float((res_batched.rho - res_stream.rho).abs().max())
    assert drho < 1e-8, f"density mismatch {drho:.3e}"

    f_b = forces(res_batched)
    f_s = forces(res_stream)
    df = float((f_b - f_s).abs().max())
    assert df < 1e-7, f"force mismatch {df:.3e} eV/Ang"
    # forces are actually nonzero (the nudge worked), so the match is meaningful
    assert float(f_b.abs().max()) > 1e-2
