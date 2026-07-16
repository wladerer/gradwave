"""Gamma-point real-wavefunction specialization (Layer A/B).

At k=0 the plane-wave sphere is closed under G -> -G, and a real Hamiltonian
(real V_eff, time-reversal symmetric) admits eigenstates with

    c(-G) = c(G)*        <=>        psi(r) real.

Only one member of each {G, -G} pair is then independent, plus the G=0
coefficient which is real. This module stores that half sphere, runs the
local-potential apply on a real-to-complex FFT (`irfftn`/`rfftn`, about 2x
on the hottest kernel with the real-space fields at half the memory), and
carries the subspace algebra as a real symmetric problem in the metric

    <psi_m|psi_n> = Re[ c_m(0)* c_n(0) ] + 2 Sum_{G>0} Re[ c_m(G)* c_n(G) ].

The half-sphere layout puts G=0 first, so the metric weight is 1 on the first
slot and 2 on the rest. G=0 stays real through the whole solve because H maps
a real field to a real field and every rotation here is real, so no G=0
constraint has to be re-imposed per step.

Everything is gated against the complex Gamma path to machine precision in
`tests/unit/test_gamma.py`. The apply matches `BatchedHamiltonian.apply`
restricted to a Hermitian-symmetric vector, and the eigenvalues match the
complex Davidson on the same frozen potential.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.dtypes import CDTYPE, RDTYPE


@dataclass
class GammaBasis:
    """Half-sphere index maps for one k=0 plane-wave sphere.

    Built once per geometry (setup layer), then frozen. `rep_full_idx` selects
    the independent half sphere out of the full sphere (G=0 first). `src_half`
    and `conj_full` reconstruct the full complex sphere from the half vector.
    `back_flat`/`back_conj` gather the half sphere back out of an `rfftn`
    half-box.
    """

    shape: tuple[int, int, int]
    nh3: int  # last-axis extent of the rfft half-box, n3 // 2 + 1
    t_half: torch.Tensor  # (nhalf,) kinetic HBAR2_2M|G|^2 on the half sphere
    metric_w: torch.Tensor  # (nhalf,) real metric weights, 1 at G=0 else 2
    full_flat_idx: torch.Tensor  # (npw,) sphere -> full-box flat index
    rep_full_idx: torch.Tensor  # (nhalf,) half -> full-sphere index
    src_half: torch.Tensor  # (npw,) full -> half source slot
    conj_full: torch.Tensor  # (npw,) bool, conjugate on reconstruction
    back_flat: torch.Tensor  # (nhalf,) half -> rfft-half-box flat index
    back_conj: torch.Tensor  # (nhalf,) bool, conjugate on back-gather

    @property
    def nhalf(self) -> int:
        return int(self.t_half.shape[0])

    @property
    def npw(self) -> int:
        return int(self.src_half.shape[0])


def build_gamma_basis(sphere, shape, device=None) -> GammaBasis:
    """Assemble the half-sphere maps for a GSphere at k=0.

    Raises if the sphere is not closed under G -> -G (i.e. k != Gamma), since
    the reality constraint only holds at a time-reversal-invariant k-point.
    """
    n1, n2, n3 = (int(s) for s in shape)
    nh3 = n3 // 2 + 1
    miller = sphere.miller.detach().cpu().numpy().astype(np.int64)
    npw = miller.shape[0]

    # box index of every sphere point and of its negation
    box = np.stack([miller[:, 0] % n1, miller[:, 1] % n2, miller[:, 2] % n3], axis=1)
    neg_box = np.stack([(-miller[:, 0]) % n1, (-miller[:, 1]) % n2, (-miller[:, 2]) % n3], axis=1)

    def flat3(b):
        return (b[:, 0] * n2 + b[:, 1]) * n3 + b[:, 2]

    lut = {int(f): g for g, f in enumerate(flat3(box))}
    partner = np.empty(npw, dtype=np.int64)
    for g, f in enumerate(flat3(neg_box)):
        j = lut.get(int(f))
        if j is None:
            raise ValueError("Gamma basis requires a sphere closed under G -> -G "
                             "(k must be the Gamma point)")
        partner[g] = j

    # representatives: G=0 first, then the lexicographically positive member of
    # each {G,-G} pair. With key(m) = (m3,m2,m1), exactly one of m,-m is positive.
    is_zero = np.all(miller == 0, axis=1)
    if int(is_zero.sum()) != 1:
        raise ValueError("Gamma sphere must contain exactly one G=0 vector")
    positive = (
        (miller[:, 2] > 0)
        | ((miller[:, 2] == 0) & (miller[:, 1] > 0))
        | ((miller[:, 2] == 0) & (miller[:, 1] == 0) & (miller[:, 0] > 0))
    )
    rep_mask = is_zero | positive
    rep_full = np.concatenate([np.where(is_zero)[0], np.where(positive)[0]])
    nhalf = rep_full.shape[0]
    if nhalf != (npw + 1) // 2:
        raise ValueError(f"half-sphere size {nhalf} != expected {(npw + 1) // 2}")

    # full -> half source slot and conjugate flag
    half_pos = np.full(npw, -1, dtype=np.int64)
    half_pos[rep_full] = np.arange(nhalf)
    src_half = np.empty(npw, dtype=np.int64)
    conj_full = np.zeros(npw, dtype=bool)
    for g in range(npw):
        if rep_mask[g]:
            src_half[g] = half_pos[g]
        else:
            src_half[g] = half_pos[partner[g]]
            conj_full[g] = True

    # back-gather out of the rfft half-box (last axis kept for i3 <= n3//2)
    rep_box = box[rep_full]
    keep = rep_box[:, 2] <= n3 // 2
    src_box = np.where(keep[:, None], rep_box, neg_box[rep_full])
    back_flat = (src_box[:, 0] * n2 + src_box[:, 1]) * nh3 + src_box[:, 2]
    back_conj = ~keep

    t_half = HBAR2_2M * sphere.kpg2.detach().cpu().numpy()[rep_full]
    metric_w = np.full(nhalf, 2.0)
    metric_w[0] = 1.0  # G=0 slot

    ai = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return GammaBasis(
        shape=(n1, n2, n3),
        nh3=nh3,
        t_half=ai(t_half, RDTYPE),
        metric_w=ai(metric_w, RDTYPE),
        full_flat_idx=sphere.flat_idx.to(device),
        rep_full_idx=ai(rep_full, torch.int64),
        src_half=ai(src_half, torch.int64),
        conj_full=ai(conj_full, torch.bool),
        back_flat=ai(back_flat, torch.int64),
        back_conj=ai(back_conj, torch.bool),
    )


def half_to_full(gb: GammaBasis, chalf: torch.Tensor) -> torch.Tensor:
    """(nb, nhalf) -> (nb, npw) full complex sphere, Hermitian by construction."""
    c = chalf[:, gb.src_half]
    return torch.where(gb.conj_full[None, :], c.conj(), c)


def full_to_half(gb: GammaBasis, cfull: torch.Tensor) -> torch.Tensor:
    """(nb, npw) -> (nb, nhalf). Lossless only for a Hermitian-symmetric cfull."""
    return cfull[:, gb.rep_full_idx]


class GammaHamiltonian:
    """H apply at Gamma on the half sphere, real-FFT local term.

    v_eff_r is real (n1,n2,n3), p complex (nproj, npw) full-sphere projectors,
    dij real (nproj, nproj). The normalization follows `BatchedHamiltonian`,
    where the irfftn/rfftn pair is norm-neutral and matches the fftbox
    ifftn(x N) / fftn(/N) contract on the retained coefficients.
    """

    def __init__(self, gb: GammaBasis, v_eff_r, p, dij):
        self.gb = gb
        self.shape = gb.shape
        self.n = gb.shape[0] * gb.shape[1] * gb.shape[2]
        self.v_eff_r = v_eff_r
        self.p = p  # (nproj, npw)
        self.dij = dij  # (nproj, nproj)

    def apply(self, chalf: torch.Tensor) -> torch.Tensor:
        gb = self.gb
        nb = chalf.shape[0]
        cfull = half_to_full(gb, chalf)  # (nb, npw)

        # kinetic (diagonal, real)
        out = gb.t_half * chalf

        # local potential via a real-space transform on the half-box
        box = torch.zeros(nb, self.n, dtype=chalf.dtype, device=chalf.device)
        box.index_add_(1, gb.full_flat_idx, cfull)
        box = box.reshape(nb, *self.shape)[..., : gb.nh3]
        psi = torch.fft.irfftn(box, s=self.shape, dim=(-3, -2, -1))  # real
        vg = torch.fft.rfftn(psi * self.v_eff_r, dim=(-3, -2, -1)).reshape(nb, -1)
        loc = vg[:, gb.back_flat]
        loc = torch.where(gb.back_conj[None, :], loc.conj(), loc)
        out = out + loc

        # nonlocal KB: Hermitian-symmetric result, project to the half sphere
        if self.p.shape[0]:
            b = cfull @ self.p.conj().T  # (nb, nproj)
            nl = (b @ self.dij.to(cfull.dtype)) @ self.p  # (nb, npw)
            out = out + nl[:, gb.rep_full_idx]
        return out


def metric_inner(gb: GammaBasis, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Real symmetric Gram <a_m|b_n> in the half-sphere metric. (nb, nb).

    Reproduces the full-sphere inner product Re[c_m(0)* c_n(0)] +
    2 Sum_{G>0} Re[c_m(G)* c_n(G)] from the half vectors, used to check
    orthonormality in tests."""
    return torch.einsum("mp,np->mn", a.conj(), gb.metric_w * b).real


def embed_real(gb: GammaBasis, chalf: torch.Tensor) -> torch.Tensor:
    """Complex half sphere (nb, nhalf) -> real feature vector (nb, 2*nhalf-1).

    The metric <a|b> = Re[Sum_p w_p a(p)* b(p)] becomes the ordinary real dot
    product phi(a) . phi(b): the G=0 slot contributes Re(c0), each G>0 slot
    contributes sqrt(2)*Re and sqrt(2)*Im. In this embedding the Gamma
    eigenproblem is a plain real symmetric one, so the standard Davidson (with
    its rank-revealing QR) solves it directly, and the whole subspace algebra
    runs in real arithmetic."""
    r2 = 2.0 ** 0.5
    nb = chalf.shape[0]
    phi = torch.empty(nb, 2 * gb.nhalf - 1, dtype=gb.t_half.dtype, device=chalf.device)
    phi[:, 0] = chalf[:, 0].real
    phi[:, 1::2] = r2 * chalf[:, 1:].real
    phi[:, 2::2] = r2 * chalf[:, 1:].imag
    return phi


def unembed_real(gb: GammaBasis, phi: torch.Tensor) -> torch.Tensor:
    """Inverse of `embed_real`: real feature vector -> complex half sphere."""
    r2i = 2.0 ** -0.5
    nb = phi.shape[0]
    chalf = torch.empty(nb, gb.nhalf, dtype=CDTYPE, device=phi.device)
    chalf[:, 0] = phi[:, 0].to(CDTYPE)
    chalf[:, 1:] = r2i * torch.complex(phi[:, 1::2], phi[:, 2::2])
    return chalf


def kinetic_phi(gb: GammaBasis) -> torch.Tensor:
    """Kinetic diagonal in the real feature space (2*nhalf-1,)."""
    t = torch.empty(2 * gb.nhalf - 1, dtype=gb.t_half.dtype, device=gb.t_half.device)
    t[0] = gb.t_half[0]
    t[1::2] = gb.t_half[1:]
    t[2::2] = gb.t_half[1:]
    return t


@dataclass
class GammaDavidsonResult:
    eigenvalues: torch.Tensor  # (nb,) ascending [eV]
    eigenvectors: torch.Tensor  # (nb, nhalf) metric-orthonormal complex half sphere
    n_iter: int
    residual_norms: torch.Tensor  # (nb,)


@torch.no_grad()
def davidson_gamma(
    ham: GammaHamiltonian,
    x0: torch.Tensor,  # (nb, nhalf) initial guess, G=0 slot real
    tol: float = 1e-9,
    max_iter: int = 40,
    max_dim_factor: int = 4,
) -> GammaDavidsonResult:
    """Real-symmetric block Davidson on the Gamma half sphere.

    Runs the standard `solvers.davidson.davidson` in the real feature embedding
    (`embed_real`), where the metric is the plain dot product. The H-apply
    wraps `GammaHamiltonian.apply` with the real FFT, so the FFT and the
    subspace algebra are both real. Eigenvectors are returned as complex half
    vectors."""
    from gradwave.solvers.davidson import davidson

    gb = ham.gb
    t_phi = kinetic_phi(gb)

    def happ(phi):
        return embed_real(gb, ham.apply(unembed_real(gb, phi)))

    res = davidson(happ, embed_real(gb, x0), t_phi, tol=tol,
                   max_iter=max_iter, max_dim_factor=max_dim_factor)
    return GammaDavidsonResult(
        eigenvalues=res.eigenvalues,
        eigenvectors=unembed_real(gb, res.eigenvectors),
        n_iter=res.n_iter,
        residual_norms=res.residual_norms,
    )
