"""Local V.psi via the Toeplitz dense-GEMM path (one M[k] @ c per k) instead of
the FFT pair. Same local operator, different summation, so NOT bit-exact vs the
FFT baseline (~1e-10) -- an approximate-but-valid candidate the tolerance oracle
admits. This is exactly the trade the shipped ``_use_toeplitz`` calibration makes
per geometry; here the evolve loop *measures* whether it wins at this size,
turning that hand-written heuristic into a search outcome."""

import torch

from gradwave.core.batch import becp_b


def apply(H, c):
    t_r, v_eff, p, dij = H._tables(c.dtype)
    out = t_r[:, None, :] * c
    out = out + H._local_toep(c)  # dense-GEMM local term (builds M once, then GEMM)
    if p.shape[1]:
        b = becp_b(p, c)
        out = out + torch.einsum("kbp,pq,kqg->kbg", b, dij, p)
    if H.hub_q is not None and H.hub_dij is not None:
        hq, hq_conj, hd = H._hub_tables(c.dtype)
        bh = torch.einsum("kpg,kbg->kbp", hq_conj, c)
        out = out + torch.einsum("kbp,pq,kqg->kbg", bh, hd, hq)
    return out * H.bk.mask[:, None, :]
