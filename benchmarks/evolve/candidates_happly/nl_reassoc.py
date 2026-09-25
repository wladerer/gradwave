"""Reassociate the KB nonlocal contraction: fold dij into the projectors first
(``dijp = dij @ p``), then contract with becp, instead of the baseline's single
three-tensor einsum ``einsum('kbp,pq,kqg->kbg', b, dij, p)``. Same operator;
only the contraction order changes, so it stays exact to fp64 round-off. Tests
whether the reassociation the einsum optimiser does not pick is faster here."""

import torch

from gradwave.core.batch import becp_b


def apply(H, c):
    t_r, v_eff, p, dij = H._tables(c.dtype)
    out = t_r[:, None, :] * c
    H._local_fft_into(c, out, v_eff)  # unchanged local term (not what we vary)
    if p.shape[1]:
        b = becp_b(p, c)  # <beta|psi>
        dijp = torch.einsum("pq,kqg->kpg", dij, p)  # fold dij into the projectors
        out = out + torch.einsum("kbp,kpg->kbg", b, dijp)
    if H.hub_q is not None and H.hub_dij is not None:
        hq, hq_conj, hd = H._hub_tables(c.dtype)
        bh = torch.einsum("kpg,kbg->kbp", hq_conj, c)
        out = out + torch.einsum("kbp,pq,kqg->kbg", bh, hd, hq)
    return out * H.bk.mask[:, None, :]
