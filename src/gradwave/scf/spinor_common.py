"""2x2 spinor machinery shared by the noncollinear SCF drivers.

The norm-conserving spinor loop (scf/noncollinear.py, SpinorHamiltonian) and
the USPP/PAW spinor loop (scf/uspp_noncollinear.py, SpinorBatchedHS) share
their local-potential structure exactly: the 2x2 potential blocks
(v +/- B_z diagonal, B_x - iB_y off-diagonal) with the b_zero fast path, the
band-chunk memory heuristic, the fused-FFT local 2x2 mix, the Pauli density
accumulation, and the alternating up/down plane-wave seed. Those live here;
the nonlocal term (scalar D vs j-resolved SOC vs the screened 2x2 D channels)
and the S-operator remain per-variant.
"""

from __future__ import annotations

import torch

from gradwave.core.batch import box_to_sphere_b, g_to_r_b
from gradwave.dtypes import CDTYPE, RDTYPE

# Dense-grid memory budget [bytes] for the band-chunked spinor FFT mix and
# nonlocal einsums (~6 dense-grid temporaries per chunk: two psi components +
# products). The CPU bound matters at many k: an unchunked apply on a 144-k
# SOC metal materializes (nk, 2*nb, grid) FFT temporaries — 8+ GB — and
# OOM-kills small-RAM hosts (asus, 14 GB). On GPU the unchunked nonlocal
# einsums at 384 k and a 240-vector Davidson block spike >5 GB per temporary,
# which OOM-killed the A100 FePt run through allocator fragmentation.
BAND_CHUNK_BUDGET_CUDA = 2.5e8
BAND_CHUNK_BUDGET_CPU = 4.0e8


def spinor_band_chunk(shape, nk: int, device, elem_bytes: int = 16) -> int:
    """Bands per chunk keeping each chunk's dense-grid temporaries under the
    budget (~250 MB GPU / ~400 MB CPU). elem_bytes lets the fp32 draft (8 B)
    take twice the bands of fp64 (16 B)."""
    n = shape[0] * shape[1] * shape[2]
    budget = (BAND_CHUNK_BUDGET_CUDA if device.type == "cuda"
              else BAND_CHUNK_BUDGET_CPU)
    return max(1, int(budget / (elem_bytes * n * max(nk, 1))))


def spinor_potential_blocks(v_r, b_vec_r):
    """(b_zero, v_uu, v_dd, v_ud): the 2x2 potential blocks of
    V = v*1 + B.sigma, precomputed once per H — the diagonal spin channels
    v +/- B_z (real) and the off-diagonal B_x - iB_y (complex). Nonmagnetic
    fields (B == 0) flag b_zero so the apply skips the spin-flip term."""
    bx, by, bz = b_vec_r[0], b_vec_r[1], b_vec_r[2]
    b_zero = float(b_vec_r.abs().max()) == 0.0
    return b_zero, v_r + bz, v_r - bz, torch.complex(bx, -by)


def apply_local_spinor(out_u, out_d, cu, cd, fft_bk, fft_shape, chunk,
                       v_uu, v_dd, v_ud, b_zero):
    """Band-chunked local 2x2 mix, accumulated into out_u/out_d IN PLACE:
    both spinor components fused into ONE batched FFT pair per chunk (a per-k
    or per-component loop launches small FFTs and is kernel-launch-bound on
    multi-k GPU runs); band-chunking bounds the dense-grid memory."""
    nb = cu.shape[1]
    for lo in range(0, nb, chunk):
        hi = min(lo + chunk, nb)
        nbc = hi - lo
        cud = torch.cat([cu[:, lo:hi], cd[:, lo:hi]], dim=1)
        psi = g_to_r_b(cud, fft_bk, fft_shape)
        psi_u, psi_d = psi[:, :nbc], psi[:, nbc:]
        if b_zero:  # B = 0: diagonal spin blocks, no spin flip
            h_u = psi_u * v_uu
            h_d = psi_d * v_dd
        else:
            h_u = psi_u * v_uu + psi_d * v_ud
            h_d = psi_u * v_ud.conj() + psi_d * v_dd
        hud = box_to_sphere_b(torch.cat([h_u, h_d], dim=1), fft_bk)
        out_u[:, lo:hi] += hud[:, :nbc]
        out_d[:, lo:hi] += hud[:, nbc:]
    return out_u, out_d


def pauli_density_accumulate(coeffs, w_kb, bk, shape, m_pw, nbands, chunk,
                             device):
    """(rho_out, m_out) accumulated from spinor coefficients by Pauli
    decomposition — k-batched with both spinor components fused into ONE
    batched FFT per band chunk, exactly like the H-apply. NOT divided by the
    cell volume; callers divide (and symmetrize) afterwards."""
    rho_out = torch.zeros(shape, dtype=RDTYPE, device=device)
    m_out = torch.zeros(3, *shape, dtype=RDTYPE, device=device)
    for lo in range(0, nbands, chunk):
        hi = min(lo + chunk, nbands)
        nbb = hi - lo
        cud = torch.cat([coeffs[:, lo:hi, :m_pw], coeffs[:, lo:hi, m_pw:]],
                        dim=1)
        psi = g_to_r_b(cud, bk, shape)
        pu, pd = psi[:, :nbb], psi[:, nbb:]
        f = w_kb[:, lo:hi].to(pu.real.dtype)
        uu = torch.einsum("kb,kbxyz->xyz", f, pu.real**2 + pu.imag**2)
        dd = torch.einsum("kb,kbxyz->xyz", f, pd.real**2 + pd.imag**2)
        ud = torch.einsum("kb,kbxyz->xyz", f.to(CDTYPE), pu.conj() * pd)
        rho_out += uu + dd
        m_out[0] += 2.0 * ud.real
        m_out[1] += 2.0 * ud.imag
        m_out[2] += uu - dd
    return rho_out, m_out


def spinor_pw_seed(nk: int, nbands: int, m_pw: int, device) -> torch.Tensor:
    """Initial spinors: alternate up/down lowest plane waves on the doubled
    coefficient axis (nk, nbands, 2*m_pw)."""
    c0 = torch.zeros(nk, nbands, 2 * m_pw, dtype=CDTYPE, device=device)
    for b in range(nbands):
        c0[:, b, (b // 2) + (b % 2) * m_pw] = 1.0
    return c0
