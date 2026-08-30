"""Carry the retained subspace matrix through a restart (Piece 3, round-off exact).

At a Davidson restart the retained block s_ret = <x_orth|H|x_orth> is exactly
diag(eig) in the pre-QR Ritz basis (x are Ritz vectors), so under the
drift-killing QR factor R it is s_ret = R^-H diag(eig) R^-1 — two tiny batched
triangular solves with NO npw contraction, replacing the fat retained-block GEMM
matmul(x_orth.conj(), hx_orth.mT). On the fp64-crippled RTX 3050 the congruence
beat the GEMM by 2.4x-340x across the probe sweep (nk in {8,96,145}, nb in
{8,24,60}, npw in {500,2000,7000}), so it is wired.

This test pins the algebra: the congruence reproduces the fat GEMM to fp64
round-off on a real Ritz setup. The end-to-end round-off equivalence (the whole
restart path vs a full rebuild) is additionally covered by the incremental A/B
tests, which now exercise this path at every restart.
"""

import torch

from gradwave.solvers.davidson import davidson_batched


def _make_gapped_operator(nk, npw, seed=0):
    torch.manual_seed(seed)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    diag = torch.linspace(0.0, 50.0, npw, dtype=torch.float64)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        herm = 0.05 * 0.5 * (a + a.conj().T)
        h[k] = herm + torch.diag(diag.to(torch.complex128))
    mask = torch.ones(nk, npw, dtype=torch.bool)

    def apply(c):
        return torch.einsum("kij,kbj->kbi", h.to(c.dtype), c)

    return h, apply, mask


def test_restart_congruence_matches_fat_gemm():
    """s_ret = R^-H diag(eig) R^-1 reproduces matmul(x_orth.conj(), hx_orth.mT)
    to fp64 round-off, in the exact restart construction davidson_batched uses."""
    torch.manual_seed(0)
    nk, npw, m, nb = 3, 60, 20, 8
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    diag = torch.linspace(1.0, 60.0, npw, dtype=torch.float64)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        h[k] = 0.5 * (a + a.conj().T) + torch.diag(diag.to(torch.complex128))
        h[k] = 0.5 * (h[k] + h[k].conj().T)

    # an orthonormal subspace basis and its images
    torch.manual_seed(1)
    raw = torch.randn(nk, npw, m, dtype=torch.complex128)
    q0, _ = torch.linalg.qr(raw, mode="reduced")
    v = q0.transpose(-1, -2).contiguous()  # (nk, m, npw), rows orthonormal
    hv = torch.einsum("kij,kbj->kbi", h, v)

    # Rayleigh-Ritz -> Ritz vectors x, images hx, eigenvalues eig
    s = torch.matmul(v.conj(), hv.mT)
    s = 0.5 * (s + s.conj().transpose(-1, -2))
    w, u = torch.linalg.eigh(s)
    eig = w[:, :nb].real
    uu = u[:, :, :nb]
    x = torch.einsum("kja,kjg->kag", uu, v)
    hx = torch.einsum("kja,kjg->kag", uu, hv)

    # restart QR exactly as the solver does
    qq, rmat = torch.linalg.qr(x.transpose(-1, -2), mode="reduced")
    x_orth = qq.transpose(-1, -2)
    hx_orth = torch.linalg.solve_triangular(
        rmat.transpose(-1, -2), hx, upper=False)

    s_fat = torch.matmul(x_orth.conj(), hx_orth.mT)  # the block Piece 3 replaces

    rmat_c = rmat.to(torch.complex128)
    diag_e = torch.diag_embed(eig.to(torch.complex128))
    z = torch.linalg.solve_triangular(rmat_c, diag_e, upper=True, left=False)
    s_ret = torch.linalg.solve_triangular(rmat_c.mH, z, upper=False, left=True)

    rel = float((s_ret - s_fat).abs().max() / s_fat.abs().max())
    assert rel < 1e-10  # round-off exact
    # s_ret is Hermitian by construction (R^-H D R^-1, D real diagonal)
    assert float((s_ret - s_ret.mH).abs().max()) < 1e-11


def test_restart_congruence_converges(monkeypatch):
    """End-to-end: with Piece 3 active (default) the solver still converges to the
    operator's true lowest eigenvalues through many restarts (tight max_dim)."""
    nk, npw, nb = 2, 70, 8
    h, apply, mask = _make_gapped_operator(nk, npw, seed=3)
    ref = torch.linalg.eigvalsh(h)[:, :nb]
    t = torch.zeros(nk, npw, dtype=torch.float64)
    torch.manual_seed(7)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    monkeypatch.setenv("GRADWAVE_RR3M", "off")
    res = davidson_batched(apply, x0, t, mask, tol=1e-9, max_iter=400,
                           max_dim_factor=2)  # factor 2 -> restart most rounds
    assert float(res.residual_norms.max()) < 1e-9
    assert float((res.eigenvalues - ref).abs().max()) < 1e-7
