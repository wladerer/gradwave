"""Mixed-precision Davidson: the fp32 draft phase must not perturb the
converged fp64 answer. On GPU the draft runs whenever the adaptive diago
tolerance is loose; here it is forced on (normally GPU-only) so CI exercises
the dtype-polymorphic H apply and the two-stage band solver on CPU.
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.davidson import davidson_batched_ms

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
A = 5.43
SI_CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0, 0], [A / 4] * 3])


def test_scf_mixed_precision_matches_fp64():
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")

    def run(mp):
        system = setup_system(SI_CELL, SI_POS, [0, 0], [upf], ecut=20 * RY,
                              kmesh=(2, 2, 2))
        return scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8,
                   verbose=False, mixed_precision=mp)

    r64 = run(False)
    rmp = run(True)
    assert r64.converged and rmp.converged
    # both re-polish to diago_tol in fp64 ⇒ the converged energy is unchanged
    assert abs(float(r64.energies.free_energy)
               - float(rmp.energies.free_energy)) < 1e-7
    assert torch.allclose(r64.eigenvalues, rmp.eigenvalues, atol=1e-6)


def test_davidson_ms_two_stage_synthetic():
    """The draft→polish wrapper reaches full-precision eigenvalues even when
    the first stage runs entirely in complex64 (H recomputes in the coeff
    dtype, so the draft is genuine fp32)."""
    torch.manual_seed(0)
    nk, n, nb = 2, 64, 6
    araw = torch.randn(nk, n, n, dtype=torch.complex128)
    amat = araw + araw.conj().transpose(-1, -2)
    amat = amat + torch.diag_embed(
        torch.arange(n, dtype=torch.float64) * 6.0).to(torch.complex128)  # diag-dominant
    mask = torch.ones(nk, n, dtype=torch.bool)
    t = torch.arange(1, n + 1, dtype=torch.float64).expand(nk, n).contiguous()

    def hop(c):  # recompute in c's precision → fp32 during the draft
        return torch.einsum("kij,kbj->kbi", amat.to(c.dtype), c) * mask[:, None, :]

    x0 = torch.zeros(nk, nb, n, dtype=torch.complex128)
    for b in range(nb):
        x0[:, b, b] = 1.0
    ref = torch.linalg.eigvalsh(amat)[:, :nb]
    out = davidson_batched_ms(hop, x0, t, mask, tol=1e-9, max_iter=200,
                              mixed_precision=True)
    assert torch.allclose(out.eigenvalues, ref, atol=1e-7)
