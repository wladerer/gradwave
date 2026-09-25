"""The batched H-apply as an evolve ``Problem`` -- the tolerance-oracle target.

The H-apply (``core.batch.BatchedHamiltonian.apply``: kinetic + local V.psi +
KB nonlocal, then mask) is the dominant FFT/GEMM cost of the SCF, and -- unlike
the Rayleigh-Ritz step -- its most valuable optimisations are deliberately NOT
bit-exact: the Toeplitz dense-GEMM local path and the complex64 (fp32) apply both
change the arithmetic. A hard ``torch.equal`` gate would reject exactly the
interesting candidates, so this Problem gates on a **relative-Frobenius
tolerance** instead. The tolerance is the operator's accuracy budget made an
explicit search parameter:

- exact grade (~1e-11): admits only arithmetic-reordering variants;
- expansion grade (~1e-5, the default here): admits fp32-apply and the Toeplitz
  path too -- legitimate for Davidson *expansion*-vector applies, which are
  re-certified against the full-precision operator before any band converges.

Every candidate's measured error is reported, so an approximate winner is never
mistaken for an exact one; the gate only decides admissibility.

Workload: a real (small) Si ``BatchedHamiltonian`` built from the bench suite's
Si case (so the pseudopotential and basis are the shipped ones), the FFT local
path forced for a deterministic baseline, and a random trial block ``c``. The
reference is the baseline fp64 FFT-path apply.
"""

from __future__ import annotations

import torch

from gradwave.core import batch as batch_mod
from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.dtypes import CDTYPE, RDTYPE

# Default gate: expansion grade. Arithmetic-reordering variants land ~1e-13, the
# Toeplitz local path ~1e-10, the fp32 apply ~1e-6 -- all admissible here, with
# the true error shown per candidate.
ORACLE_TOL = 1e-5


def _si_suite_system():
    """A real Si system from the bench suite (shipped pseudo + basis) -- the
    small primitive-cell workload (npw ~750, nk = IBZ of 4x4x4)."""
    from gradwave.bench.suite import build_suite

    for case in build_suite():
        if case.name == "Si":
            return case.build()
    raise RuntimeError("Si case not found in bench suite")


def _si_cubic_system(ecut_ry: float):
    """8-atom conventional-cubic diamond Si at Gamma -- a LARGE-cell workload
    (npw ~few-thousand, nk=1, 8x the projectors). This is the frontier regime
    where the local FFT dominates and the Toeplitz npw^2 local path is no longer
    a win, so it tests what the loop finds when the small-cell levers die."""
    import os
    from pathlib import Path

    import numpy as np

    from gradwave.constants import RY_EV
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system

    pdir = Path(os.environ.get(
        "GRADWAVE_PSEUDO_DIR",
        Path(__file__).resolve().parents[2] / "tests/fixtures/qe/pseudos"))
    upf = parse_upf(str(pdir / "Si_ONCV_PBE-1.2.upf"))
    a = 5.43
    cell = a * np.eye(3)
    frac = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                     [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])
    return setup_system(cell, frac * a, [0] * 8, [upf], ecut=ecut_ry * RY_EV,
                        kmesh=(1, 1, 1), nbands=40, use_symmetry=False)


class HApplyProblem:
    """Frozen real H-apply workload + relative-Frobenius tolerance oracle.

    ``cell="suite"`` is the small primitive Si (default); ``cell="cubic8"`` is
    the 8-atom cubic Gamma-point large-cell frontier workload."""

    name = "hamiltonian_apply"

    def __init__(self, *, nb: int = 24, seed: int = 0, tol: float = ORACLE_TOL,
                 cell: str = "suite", ecut_ry: float = 30.0) -> None:
        self.nb = nb
        self.tol = tol
        self.cell = cell
        self.ecut_ry = ecut_ry
        self._build(seed)

    def _build(self, seed: int) -> None:
        # Force the FFT local path so the baseline apply is deterministic (the
        # Toeplitz path is a candidate, gated on tolerance, not the reference).
        batch_mod._TOEPLITZ_MODE = "off"

        system = _si_cubic_system(self.ecut_ry) if self.cell == "cubic8" \
            else _si_suite_system()
        bk = system.batch
        shape = tuple(int(s) for s in system.grid.shape)
        self.nk = int(bk.nk)
        self.npw_max = int(bk.npw_max)
        self.n_grid = shape[0] * shape[1] * shape[2]

        torch.manual_seed(seed)
        v_eff_r = torch.rand(shape, dtype=RDTYPE)  # values irrelevant to FFT cost
        p = projectors_b(bk, system.positions)
        self._h = BatchedHamiltonian(bk, shape, v_eff_r, p)

        torch.manual_seed(seed + 1)
        self._c = torch.randn(self.nk, self.nb, self.npw_max, dtype=CDTYPE)
        self._c = self._c * bk.mask[:, None, :]

        ref = self._h.apply(self._c)
        self._ref = ref
        self._ref_norm = float(torch.linalg.norm(ref)) + 1e-30

    def workload(self) -> tuple[BatchedHamiltonian, torch.Tensor]:
        return (self._h, self._c)

    def oracle(self, out: torch.Tensor) -> tuple[bool, float]:
        if out.shape != self._ref.shape:
            return False, float("inf")
        rel = float(torch.linalg.norm(out.to(self._ref.dtype) - self._ref)) / self._ref_norm
        return (rel <= self.tol), rel
