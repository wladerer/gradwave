"""Meta-GGA kinetic-energy density τ and its generalized-KS operator (Layer A/B).

A meta-GGA XC functional depends, on top of ρ and σ = |∇ρ|², on the
positive-definite kinetic-energy density

    τ(r) = ½ Σ_k w_k Σ_n f_nk |∇ψ_nk(r)|².

τ is NOT a functional of ρ on the grid — it is an independent orbital field —
so it does not ride the ρ autograd graph the way σ does (see core.density). Two
consequences, both handled here:

  * τ is built directly from the plane-wave coefficients, one extra i(k+G)
    factor on the density-build FFT (`tau_b`), mirroring `core.batch.density_b`.
  * its potential v_τ = ∂e_xc/∂τ does not act multiplicatively on ρ. It enters
    the Hamiltonian as the generalized-KS operator

        V_τ ψ(r) = −½ ∇·[ v_τ(r) ∇ψ(r) ],

    which in the plane-wave basis is (`metagga_tau_operator`)

        (V_τ ψ)(k+G) = −½ Σ_d i(k+G)_d · F[ v_τ(r) · F⁻¹[ i(k+G)_d c ] ],

    Hermitian by construction. This is what makes a meta-GGA a *generalized*
    Kohn–Sham scheme: the extra term touches the H-apply, not only v_eff.

Units mirror the rest of Layer A: (k+G) in Å⁻¹, so τ is in [e/Å⁵] (i.e. ρ·Å⁻²),
and the functional converts to atomic units internally exactly as it does for ρ
and σ. The operator is assembled in the same fftbox convention as
`core.batch.g_to_r_b`/`box_to_sphere_b`, whose ×N and ÷N cancel (norm-neutral),
so no explicit scaling appears below.
"""

from __future__ import annotations

import torch

from gradwave.core.batch import BatchedK, box_to_sphere_b, g_to_r_b

# GPU dense-grid temporary budget [bytes]; matches core.batch. The τ paths hold
# a handful of (nk, nb, n_grid) boxes at once, so bands are chunked to bound the
# peak the same way density_b/BatchedHamiltonian.apply do.
_GPU_DENSE_BUDGET_BYTES = 4e8


def _band_chunk(nk: int, n: int, elem_bytes: int, device) -> int:
    if device.type != "cuda":
        return 1_000_000
    return max(1, int(_GPU_DENSE_BUDGET_BYTES / (elem_bytes * n * max(nk, 1))))


def tau_b(
    coeffs: torch.Tensor,  # (nk, nb, npw_max)
    occ: torch.Tensor,  # (nk, nb)
    kweights: torch.Tensor,  # (nk,)
    bk: BatchedK,
    shape: tuple[int, int, int],
    volume: float,
) -> torch.Tensor:
    """τ(r) = ½ Σ_k w_k Σ_n f_nk |∇ψ_nk(r)|² on the dense grid [e/Å⁵].

    Differentiable in ``coeffs`` (built through the autograd-friendly
    `g_to_r_b`), so it serves both the energy assembly and — detached — the
    v_τ extraction. Band-chunked to bound dense-grid memory, mirroring
    `core.batch.density_b`.
    """
    nk, nb, _ = coeffs.shape
    n = shape[0] * shape[1] * shape[2]
    chunk = _band_chunk(nk, n, coeffs.element_size(), coeffs.device) if \
        coeffs.device.type == "cuda" else nb
    w = kweights[:, None] * occ
    kpg = bk.kpg  # (nk, npw_max, 3), Å⁻¹, zero in padding
    tau = None
    for lo in range(0, nb, chunk):
        hi = min(lo + chunk, nb)
        cc = coeffs[:, lo:hi]  # (nk, nbc, npw_max)
        grad2 = None  # Σ_d |∂_d ψ|²
        for d in range(3):
            # ∂_d ψ = Σ_G i(k+G)_d c e^{iGr} = g_to_r_b(i(k+G)_d c)
            gd = (1j * kpg[:, None, :, d]) * cc
            psid = g_to_r_b(gd, bk, shape)
            term = psid.real ** 2 + psid.imag ** 2
            grad2 = term if grad2 is None else grad2 + term
        contrib = 0.5 * torch.einsum(
            "kb,kbxyz->xyz", w[:, lo:hi].to(grad2.dtype), grad2
        )
        tau = contrib if tau is None else tau + contrib
    return tau / volume


def spinor_tau_matrix_b(
    coeffs: torch.Tensor,  # (nk, nb, 2·npw_max) doubled spinor coefficients
    occ: torch.Tensor,  # (nk, nb)
    kweights: torch.Tensor,  # (nk,)
    bk: BatchedK,
    shape: tuple[int, int, int],
    volume: float,
    m_pw: int,  # npw_max (the per-component plane-wave count)
) -> tuple[torch.Tensor, torch.Tensor]:
    """2×2 kinetic-energy-density-matrix Pauli components (τ_0, τ⃗) [e/Å⁵].

    The spinor generalization of `tau_b`: the KE-density matrix is

        τ_{αβ}(r) = ½ Σ_k w_k Σ_n f_nk Σ_d (∂_d ψ_{nα})* (∂_d ψ_{nβ}),

    Hermitian 2×2 at each grid point, decomposed exactly like the (ρ, m⃗) spin
    density in `scf.spinor_common.pauli_density_accumulate` but with the extra
    i(k+G) gradient factor of `tau_b`. Returns

        τ_0 = τ_↑↑ + τ_↓↓                         (total KE density, = plain τ)
        τ⃗  = (2 Re τ_↑↓, 2 Im τ_↑↓, τ_↑↑ − τ_↓↓)  (KE-density magnetization)

    The locally-collinear meta-GGA takes the local-frame per-spin
    τ_± = (τ_0 ± τ⃗·m̂)/2 (the diagonal of τ rotated into the frame that
    diagonalizes the spin density — see `core.xc.noncollinear.local_frame_tau`).
    With every moment along ẑ, τ_↑↓ → 0 and (τ_+, τ_-) → (τ_↑↑, τ_↓↓), the
    collinear per-spin τ. Differentiable in ``coeffs``; band-chunked like `tau_b`.
    """
    nk, nb, _ = coeffs.shape
    n = shape[0] * shape[1] * shape[2]
    chunk = _band_chunk(nk, n, coeffs.element_size(), coeffs.device) if \
        coeffs.device.type == "cuda" else nb
    w = kweights[:, None] * occ
    kpg = bk.kpg  # (nk, npw_max, 3), Å⁻¹, zero in padding
    tau_uu = tau_dd = tau_ud = None
    for lo in range(0, nb, chunk):
        hi = min(lo + chunk, nb)
        cu = coeffs[:, lo:hi, :m_pw]
        cd = coeffs[:, lo:hi, m_pw:]
        guu = gdd = gud = None  # Σ_d over gradient components
        for d in range(3):
            ikg = 1j * kpg[:, None, :, d]
            du = g_to_r_b(ikg * cu, bk, shape)  # ∂_d ψ↑
            dv = g_to_r_b(ikg * cd, bk, shape)  # ∂_d ψ↓
            uu = du.real ** 2 + du.imag ** 2
            dd = dv.real ** 2 + dv.imag ** 2
            ud = du.conj() * dv
            guu = uu if guu is None else guu + uu
            gdd = dd if gdd is None else gdd + dd
            gud = ud if gud is None else gud + ud
        wr = w[:, lo:hi].to(guu.dtype)
        wc = w[:, lo:hi].to(gud.dtype)
        cuu = 0.5 * torch.einsum("kb,kbxyz->xyz", wr, guu)
        cdd = 0.5 * torch.einsum("kb,kbxyz->xyz", wr, gdd)
        cud = 0.5 * torch.einsum("kb,kbxyz->xyz", wc, gud)
        tau_uu = cuu if tau_uu is None else tau_uu + cuu
        tau_dd = cdd if tau_dd is None else tau_dd + cdd
        tau_ud = cud if tau_ud is None else tau_ud + cud
    tau_uu, tau_dd, tau_ud = tau_uu / volume, tau_dd / volume, tau_ud / volume
    tau_scalar = tau_uu + tau_dd
    tau_vec = torch.stack([2.0 * tau_ud.real, 2.0 * tau_ud.imag, tau_uu - tau_dd])
    return tau_scalar, tau_vec


def spinor_metagga_tau_operator(
    c: torch.Tensor,  # (nk, nb, 2·npw_max)
    v0_r: torch.Tensor,  # (n1,n2,n3) real, scalar τ-potential v_τ0
    vvec_r: torch.Tensor,  # (3,n1,n2,n3) real, vector τ-potential v_τ⃗
    bk: BatchedK,
    shape: tuple[int, int, int],
    m_pw: int,
) -> torch.Tensor:
    """2×2 generalized-KS meta-GGA τ operator on doubled spinor coefficients:

        (V_τ ψ)_α = −½ Σ_d i(k+G)_d · F[ M_{αβ}(r) · F⁻¹[ i(k+G)_d c_β ] ],
        M(r) = v_τ0·𝟙 + v_τ⃗·σ⃗   (Hermitian 2×2),

    the τ analogue of the local potential V = v·𝟙 + B⃗·σ⃗ (same diagonal
    v_τ0 ± v_τz / spin-flip v_τx − i v_τy blocks) carried inside the −½∇·(M∇)
    kinetic form. Hermitian by construction; band-chunked to bound the
    dense-grid temporaries. When v_τ⃗ ∥ ẑ (collinear limit) M is diagonal and
    this reduces to `metagga_tau_operator` applied per spin component with
    v_τ↑ = v_τ0 + v_τz, v_τ↓ = v_τ0 − v_τz.
    """
    nk, nb, _ = c.shape
    n = shape[0] * shape[1] * shape[2]
    kpg = bk.kpg
    m_uu = (v0_r + vvec_r[2]).to(c.real.dtype)
    m_dd = (v0_r - vvec_r[2]).to(c.real.dtype)
    m_ud = torch.complex(vvec_r[0], -vvec_r[1]).to(c.dtype)  # ⟨↑|M|↓⟩
    chunk = _band_chunk(nk, n, c.element_size(), c.device) if \
        c.device.type == "cuda" else nb
    mask = bk.mask[:, None, :]
    out = torch.zeros_like(c)
    for lo in range(0, nb, chunk):
        hi = min(lo + chunk, nb)
        cu = c[:, lo:hi, :m_pw]
        cd = c[:, lo:hi, m_pw:]
        acc_u = acc_d = None
        for d in range(3):
            ikg = 1j * kpg[:, None, :, d]
            gu = g_to_r_b(ikg * cu, bk, shape)  # ∂_d ψ↑
            gv = g_to_r_b(ikg * cd, bk, shape)  # ∂_d ψ↓
            hu = m_uu * gu + m_ud * gv          # (M ∂_d ψ)↑
            hv = m_ud.conj() * gu + m_dd * gv   # (M ∂_d ψ)↓
            wu = ikg * box_to_sphere_b(hu, bk)
            wv = ikg * box_to_sphere_b(hv, bk)
            acc_u = wu if acc_u is None else acc_u + wu
            acc_d = wv if acc_d is None else acc_d + wv
        out[:, lo:hi, :m_pw] = -0.5 * acc_u * mask
        out[:, lo:hi, m_pw:] = -0.5 * acc_d * mask
    return out


def metagga_tau_operator(
    c: torch.Tensor,  # (nk, nb, npw_max)
    v_tau_r: torch.Tensor,  # (n1, n2, n3) real, the scaled v_τ = ∂E_xc/∂τ potential
    bk: BatchedK,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """V_τ c = −½ Σ_d i(k+G)_d · F[ v_τ · F⁻¹[ i(k+G)_d c ] ], masked.

    The generalized-KS meta-GGA term. Hermitian; band-chunked to bound the
    dense-grid temporaries. Reduces to c·(−½∇²) (i.e. v_τ times the kinetic
    operator) when v_τ is constant, the closed-form gate the tests pin.
    """
    nk, nb, m = c.shape
    n = shape[0] * shape[1] * shape[2]
    kpg = bk.kpg
    v = v_tau_r.to(c.real.dtype)
    chunk = _band_chunk(nk, n, c.element_size(), c.device) if \
        c.device.type == "cuda" else nb
    out = torch.zeros_like(c)
    for lo in range(0, nb, chunk):
        hi = min(lo + chunk, nb)
        cc = c[:, lo:hi]
        acc = None
        for d in range(3):
            ikg = 1j * kpg[:, None, :, d]  # (nk, 1, npw_max)
            grad_r = g_to_r_b(ikg * cc, bk, shape)  # ∂_d ψ (nk, nbc, box)
            w_g = box_to_sphere_b(v * grad_r, bk)  # F[v_τ ∂_d ψ] → sphere coeffs
            term = ikg * w_g  # i(k+G)_d · F[...]
            acc = term if acc is None else acc + term
        out[:, lo:hi] = acc
    return (-0.5 * out) * bk.mask[:, None, :]
