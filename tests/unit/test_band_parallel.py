"""Band-PARALLEL FFT H-apply in the NC SCF (``scf.loop.scf``, ``band_parallel``).

``k_parallel`` distributes whole k-solves across cores and strands when there
are fewer k-points than workers — the few-k large-cell regime (a small supercell
on a coarse mesh: a handful of IBZ k, many bands). ``band_parallel`` instead
thread-pools the FFT local term (``scatter → ifftn → ·v_eff → fftn → gather``,
one transform per band) over the BAND axis inside one k. Those per-band
transforms are independent and torch FFTs release the GIL, so they parallelize.

The load-bearing property is BIT-EXACTNESS: chunking the band axis changes only
which thread runs a chunk, not the arithmetic, and each chunk writes a disjoint
output slice with its own scatter buffer — so both the raw H-apply and a whole
SCF are byte-for-byte identical to the serial path (``torch.equal``, not a
tolerance).

Pinned here:
- resolution: unset / <= 1 / CUDA device → off (None); the GRADWAVE_BAND_PARALLEL
  env default; explicit arg beats env; a bad env raises;
- bit-exactness of ``BatchedHamiltonian.apply`` (on vs off) on a random block;
- bit-exactness of a whole SCF (on vs off);
- the compose rule with k_parallel (k_parallel wins when nk >= workers, else
  band_parallel), decided in ``scf``;
- torch's global thread count is restored after the pooled apply;
- the Input knob (scf.memory.band_parallel) parses, validates, is threaded by
  run_scf, and echoed by the CLI summary.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import gradwave.core.batch as batch
from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import _resolve_band_parallel, scf, setup_system
from tests.helpers import PSEUDOS, RY, SI_ONCV, si_fcc

_CPU = torch.device("cpu")
_CUDA = torch.device("cuda")


def _si_system(kmesh=(2, 2, 2), ecut=10, nbands=None, use_symmetry=False):
    si = parse_upf(str(PSEUDOS / SI_ONCV))
    cell, pos = si_fcc()
    pos = pos.copy()
    pos[1, 0] += 0.10  # break equilibrium so forces/density are nontrivial
    return setup_system(cell, pos, [0, 0], [si], ecut=ecut * RY,
                        kmesh=kmesh, nbands=nbands, use_symmetry=use_symmetry)


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #

def test_resolve_band_parallel_semantics(monkeypatch):
    """None / <= 1 / non-CPU → off (None); a valid count passes through;
    GRADWAVE_BAND_PARALLEL is the default only when the arg is unset."""
    monkeypatch.delenv("GRADWAVE_BAND_PARALLEL", raising=False)
    assert _resolve_band_parallel(None, _CPU) is None    # default: off
    assert _resolve_band_parallel(1, _CPU) is None        # 1 worker == serial
    assert _resolve_band_parallel(0, _CPU) is None
    assert _resolve_band_parallel(-4, _CPU) is None
    assert _resolve_band_parallel(4, _CUDA) is None        # CPU-only
    assert _resolve_band_parallel(4, _CPU) == 4            # valid, no clamp
    assert _resolve_band_parallel(64, _CPU) == 64
    monkeypatch.setenv("GRADWAVE_BAND_PARALLEL", "3")
    assert _resolve_band_parallel(None, _CPU) == 3         # env default
    assert _resolve_band_parallel(2, _CPU) == 2            # explicit arg wins
    monkeypatch.setenv("GRADWAVE_BAND_PARALLEL", "not-an-int")
    with pytest.raises(ValueError):
        _resolve_band_parallel(None, _CPU)


# --------------------------------------------------------------------------- #
# bit-exactness — the raw H-apply
# --------------------------------------------------------------------------- #

def _fft_H(system, band_parallel):
    """A BatchedHamiltonian on the FFT local path (Toeplitz forced off) with a
    random real potential — the identity under test is a property of the
    discretized operator, so a random v_eff is a stronger check than a physical
    one."""
    bk = system.batch
    shape = system.grid.shape
    projs = projectors_b(bk, system.positions)
    torch.manual_seed(0)
    v_eff = torch.randn(*shape, dtype=RDTYPE)
    h = BatchedHamiltonian(bk, shape, v_eff, projs, band_parallel=band_parallel)
    h._toep_eligible = False  # exercise the FFT local term, not the Toeplitz GEMM
    return h, bk


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_apply_bit_exact(workers):
    """H·c is byte-for-byte identical with band_parallel on vs off — same ops
    per band, only the thread that runs a band chunk changes."""
    system = _si_system()
    h_off, bk = _fft_H(system, None)
    h_on, _ = _fft_H(system, workers)
    assert not bk.mask.all(), "test needs padded slots to exercise masking"

    nk, npw = bk.mask.shape
    torch.manual_seed(1)
    c = torch.randn(nk, 16, npw, dtype=CDTYPE)  # nb=16 > workers so chunks split

    out_off = h_off.apply(c)
    out_on = h_on.apply(c)
    assert torch.equal(out_off, out_on), (
        f"band_parallel={workers} not bit-exact: "
        f"max|Δ|={float((out_off - out_on).abs().max()):.3e}")


def test_apply_single_band_no_split():
    """A one-band block cannot be split; the parallel gate is a no-op and the
    result is still exact."""
    system = _si_system()
    h_off, bk = _fft_H(system, None)
    h_on, _ = _fft_H(system, 8)
    nk, npw = bk.mask.shape
    torch.manual_seed(2)
    c = torch.randn(nk, 1, npw, dtype=CDTYPE)
    assert torch.equal(h_off.apply(c), h_on.apply(c))


def test_torch_thread_count_restored():
    """The pooled apply pins torch to 1 thread internally and restores the
    caller's setting afterwards."""
    prev = torch.get_num_threads()
    system = _si_system()
    h, bk = _fft_H(system, 4)
    nk, npw = bk.mask.shape
    torch.manual_seed(3)
    h.apply(torch.randn(nk, 8, npw, dtype=CDTYPE))
    assert torch.get_num_threads() == prev


# --------------------------------------------------------------------------- #
# compose rule with k_parallel (decided in scf())
# --------------------------------------------------------------------------- #

def _capture_band_parallel(monkeypatch):
    """Record the band_parallel worker count every BatchedHamiltonian sees."""
    seen: list[int | None] = []
    orig = batch.BatchedHamiltonian

    def spy(*args, **kw):
        seen.append(kw.get("band_parallel"))
        return orig(*args, **kw)

    monkeypatch.setattr("gradwave.core.batch.BatchedHamiltonian", spy)
    return seen


def test_compose_k_parallel_wins_when_enough_k(monkeypatch):
    """Both knobs set with nk (8) >= band_parallel workers (4): k_parallel fills
    its pool and wins, so the H-apply runs serial (band_parallel dropped)."""
    seen = _capture_band_parallel(monkeypatch)
    system = _si_system(kmesh=(2, 2, 2))  # nk = 8, symmetry off
    scf(system, LDA_PW92(), smearing="gaussian", width=0.1, max_iter=2,
        verbose=False, k_parallel=4, band_parallel=4)
    assert seen and all(bp is None for bp in seen)


def test_compose_band_parallel_wins_when_few_k(monkeypatch):
    """Both knobs set with nk (2) < band_parallel workers (4): k_parallel cannot
    fill its pool, so band_parallel takes over (H built with band_parallel=4)."""
    seen = _capture_band_parallel(monkeypatch)
    system = _si_system(kmesh=(2, 1, 1))  # nk = 2, symmetry off
    scf(system, LDA_PW92(), smearing="gaussian", width=0.1, max_iter=2,
        verbose=False, k_parallel=4, band_parallel=4)
    assert seen and all(bp == 4 for bp in seen)


# --------------------------------------------------------------------------- #
# bit-exactness — a whole SCF
# --------------------------------------------------------------------------- #

@pytest.mark.standard
def test_scf_bit_exact():
    """A band_parallel SCF is byte-for-byte identical to the serial SCF: the
    only thing that changes is FFT-local-term thread scheduling, which is exact,
    so every iteration — and thus the final energy, density and eigenvalues —
    matches under torch.equal, not merely to tolerance."""
    def run(bp):
        return scf(_si_system(), LDA_PW92(), smearing="gaussian", width=0.1,
                   max_iter=60, etol=1e-11, rhotol=1e-9, diago_tol=1e-10,
                   verbose=False, band_parallel=bp)

    res_off = run(None)
    res_on = run(4)
    assert res_off.converged and res_on.converged
    assert torch.equal(res_off.rho, res_on.rho), (
        f"density not bit-exact: max|Δ|={float((res_off.rho - res_on.rho).abs().max()):.3e}")
    assert float(res_off.energies.free_energy) == float(res_on.energies.free_energy)
    assert torch.equal(res_off.eigenvalues, res_on.eigenvalues)


# --------------------------------------------------------------------------- #
# Input schema + api threading + CLI echo (parse-only, fast tier)
# --------------------------------------------------------------------------- #

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

    assert SCFParams().memory.band_parallel is None  # default off
    inp = load_input(_write_input(tmp_path, "scf: {memory: {band_parallel: 8}}\n"))
    assert inp.scf.memory.band_parallel == 8
    with pytest.raises(InputError, match="band_parallel"):
        load_input(_write_input(tmp_path, "scf: {memory: {band_parallel: 1}}\n"))


def test_run_scf_threads_band_parallel(monkeypatch):
    """run_scf passes scf.memory.band_parallel to scf.loop.scf as a kwarg."""
    from ase import Atoms

    import gradwave.api.scf as api_scf
    import gradwave.scf.loop as loop
    from gradwave.inputs.models import Input, MemoryParams, SCFParams

    captured: dict = {}

    def fake_scf(system, xc, **kwargs):
        captured["band_parallel"] = kwargs.get("band_parallel")
        return "SENTINEL"

    monkeypatch.setattr(api_scf, "build_system", lambda inp: object())
    monkeypatch.setattr(api_scf, "_species_upfs", lambda inp: ([], [], None))
    monkeypatch.setattr(api_scf, "_is_uspp", lambda upfs: False)
    monkeypatch.setattr(loop, "scf", fake_scf)

    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]],
                  cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]], pbc=True)
    inp = Input(atoms=atoms, pseudo_dir=Path("."), pseudo_map={"Si": "Si.upf"},
                ecut=200.0, scf=SCFParams(memory=MemoryParams(band_parallel=8)))
    assert api_scf.run_scf(inp, verbose=False) == "SENTINEL"
    assert captured["band_parallel"] == 8


def test_cli_summary_echoes_band_parallel():
    from ase import Atoms

    from gradwave.cli import _summary_lines
    from gradwave.inputs.models import Input, MemoryParams, SCFParams

    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]],
                  cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]], pbc=True)
    inp = Input(atoms=atoms, pseudo_dir=Path("."), pseudo_map={"Si": "Si.upf"},
                ecut=200.0, scf=SCFParams(memory=MemoryParams(band_parallel=8)))
    line = next((ln for ln in _summary_lines(inp) if "memory" in ln), None)
    assert line is not None and "band_parallel 8" in line
