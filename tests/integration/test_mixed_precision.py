"""Mixed-precision Davidson: the fp32 draft phase must not perturb the
converged fp64 answer. On GPU the draft runs whenever the adaptive diago
tolerance is loose; here it is forced on (normally GPU-only) so CI exercises
the dtype-polymorphic H apply and the two-stage band solver on CPU.
"""

from pathlib import Path

import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.davidson import davidson_batched_ms
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
SI_CELL, SI_POS = si_fcc()


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


@pytest.mark.standard
def test_uspp_paw_scf_mixed_precision_matches_fp64():
    """USPP/PAW twin of the NC gate: the fp32 draft in the generalized
    batched Davidson (fp64 subspace reduction, fp64 S-normalization) must
    leave the converged answer unchanged."""
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    torch.set_num_threads(4)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")

    def run(mp):
        s = setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=15 * RY,
                       kmesh=(2, 2, 2), ecutrho=60 * RY)
        return scf_uspp(s, PBE(), etol=1e-10, rhotol=1e-9, verbose=False,
                        max_iter=60, mixed_precision=mp)

    r64 = run(False)
    rmp = run(True)
    assert r64["converged"] and rmp["converged"]
    assert abs(float(r64["energies"].free_energy)
               - float(rmp["energies"].free_energy)) < 1e-7
    for e64, emp in zip(r64["eigenvalues"], rmp["eigenvalues"], strict=True):
        assert torch.allclose(e64, emp, atol=1e-6)
