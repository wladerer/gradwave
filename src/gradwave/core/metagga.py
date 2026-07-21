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
    shape,
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


def metagga_tau_operator(
    c: torch.Tensor,  # (nk, nb, npw_max)
    v_tau_r: torch.Tensor,  # (n1, n2, n3) real, the scaled v_τ = ∂E_xc/∂τ potential
    bk: BatchedK,
    shape,
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
