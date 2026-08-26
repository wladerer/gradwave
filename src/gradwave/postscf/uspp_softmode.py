"""Soft-mode deflation wiring for the USPP/PAW (and +U) response.

The USPP/PAW twin of ``scf.soft_mode``. It assembles the composite screening
operator of the generalized SCF fixed point

    u = (v̄, 0, 0) + M u,     M = K χ̃ 𝒮,

as a single flat ``LinOp`` over the composite response vector

    x = (δρ_tot on the grid, δbecsum, δn_+U)   [per spin channel],

and hands it to the formalism-neutral deflation core in ``solvers.deflation``
(the same Arnoldi extraction and deflated solve the norm-conserving path uses).

Everything physical is already in ``uspp_implicit._ConvergedUSPP``: ``apply_chi0``
is the composite independent-particle response χ̃ (with the augmentation/becsum
channel, the S-orthonormal generalized Sternheimer, the smeared-metal Fermi-
surface terms, and the +U occupation-matrix response), and the kernel ``K`` is
block-diagonal ``diag(K_Hxc, H_1c, K_U)`` via ``k_hxc_grid`` (Hartree + f_xc),
``hvp_onecenter`` (the PAW one-center Hessian) and ``k_hub`` (the Dudarev +U
second derivative ``−(U−J)·herm(δn)``). So a +U ground state's screening operator
already carries its +U term — no separate "+U kernel" is needed here, only the
wiring that exposes ``M`` as a ``LinOp``.

Each apply uses a FRESH zero Sternheimer warm-start so ``M(x)`` is a deterministic
(to ``cg_tol``) function of ``x`` — required for the Arnoldi orthogonalisation and
the coarse Galerkin operator to be well-defined.

Instability knobs mirror the NC ``screening_apply``: ``coupling`` scales the whole
kernel; ``fxc_scale`` scales only the grid f_xc (the Stoner / soft-phonon knob),
leaving Hartree, the one-center Hessian and the +U kernel fixed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import CDTYPE
from gradwave.postscf._response import fxc_hvp, fxc_hvp_spin, hartree_kernel
from gradwave.postscf.uspp_implicit import _check_supported, _ConvergedUSPP
from gradwave.scf.results import USPPResult
from gradwave.solvers.deflation import (
    Field,
    LinOp,
    SoftSubspace,
    SolveResult,
    anderson_solve,
    deflated_solve,
    soft_subspace_from_operator,
)

__all__ = [
    "UsppScreening",
    "build_uspp_screening",
    "soft_subspace_uspp",
    "solve_adjoint_deflated_uspp",
]


def _k_hxc_grid_scaled(cs: _ConvergedUSPP, drho_sp: list[torch.Tensor],
                       fxc_scale: float) -> list[torch.Tensor]:
    """``cs.k_hxc_grid`` with only the grid f_xc term scaled by ``fxc_scale``.

    Mirrors ``_ConvergedUSPP.k_hxc_grid`` (total-density Hartree into every
    channel, spin HVP for nspin=2, NLCC core split) but multiplies the f_xc HVP by
    ``fxc_scale`` so the Hartree charge spectrum is untouched — the same
    Stoner/soft-phonon knob as the NC ``_k_hxc_fxc_scaled``. ``fxc_scale == 1.0``
    is identical to ``cs.k_hxc_grid``."""
    if fxc_scale == 1.0:
        return cs.k_hxc_grid(drho_sp)
    rho_tot_pert = drho_sp[0]
    for r in drho_sp[1:]:
        rho_tot_pert = rho_tot_pert + r
    kh = hartree_kernel(cs.grid, rho_tot_pert)
    if cs.nspin == 1:
        assert isinstance(cs.xc, XCFunctional)
        return [kh + fxc_scale * fxc_hvp(cs.xc, cs.rho_xc, cs.grid, drho_sp[0])]
    assert isinstance(cs.xc, SpinXC)
    core = cs.system.rho_core
    c2 = 0.0 if core is None else 0.5 * core
    fu, fd = fxc_hvp_spin(cs.xc, cs.rho_sp[0] + c2, cs.rho_sp[1] + c2, cs.grid,
                          drho_sp[0], drho_sp[1])
    return [kh + fxc_scale * fu, kh + fxc_scale * fd]


@dataclass
class UsppScreening:
    """The USPP/PAW composite screening operator ``M`` as a flat ``LinOp``.

    ``apply``: ``M`` on the flat composite vector (Arnoldi/deflation feed this).
    ``ref``: a zero flat vector of the right length (Arnoldi start template).
    ``pack``: build the RHS composite ``(v̄, 0, 0)`` from a per-spin grid field.
    ``split`` / ``join``: flat ↔ (grid, becsum, +U) composite, for interpreting a
    solution. ``cs``: the frozen converged-state operators."""
    apply: LinOp
    ref: Field
    pack: Callable[[torch.Tensor], Field]
    split: Callable[
        [Field],
        tuple[list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]] | None],
    ]
    join: Callable[..., Field]
    cs: _ConvergedUSPP


def build_uspp_screening(res: USPPResult, xc: XCFunctional | SpinXC, *,
                         coupling: float = 1.0, fxc_scale: float = 1.0,
                         cg_tol: float = 1e-6, cg_max_iter: int = 200
                         ) -> UsppScreening:
    """Assemble the composite screening operator ``M = K χ̃ 𝒮`` for a converged
    USPP/PAW (optionally +U) result as a flat ``LinOp``.

    ``coupling`` scales the whole kernel; ``fxc_scale`` scales only the grid f_xc
    (Stoner knob). ``cg_tol`` is the inner Sternheimer tolerance (loosened from the
    production adjoint's default — the eigenvalue/deflation estimates do not need a
    machine-tight χ̃ apply)."""
    _check_supported(res)
    cs = _ConvergedUSPP(res, xc)
    system, grid = cs.system, cs.grid
    n_pts = grid.n_points
    nsp = cs.nspin
    dev = system.positions.device

    nbec = [s1 - s0 for (s0, s1) in system.atom_slices]
    nsh = cs.nsh if cs.hub is not None else 0
    hub_dims = ([st["dim"] for st in cs.hub.sites] if cs.hub is not None else [])
    total = (nsp * n_pts + nsp * sum(n * n for n in nbec)
             + nsh * sum(2 * d * d for d in hub_dims))

    def split(u: Field):
        """Flat u → (per-spin grid fields, per-spin per-atom becsum mats,
        per-channel per-site hub mats or None). Same layout as the mixer /
        uspp_implicit's adjoint."""
        w_sp: list[torch.Tensor] = []
        off = 0
        for _ in range(nsp):
            w_sp.append(u[off:off + n_pts].reshape(grid.shape))
            off += n_pts
        mats_sp = []
        for _ in range(nsp):
            mats = []
            for n in nbec:
                mats.append(u[off:off + n * n].reshape(n, n))
                off += n * n
            mats_sp.append(mats)
        hub_sp = None
        if nsh:
            hub_sp = []
            for _ in range(nsh):
                ch = []
                for d in hub_dims:
                    re = u[off:off + d * d].reshape(d, d)
                    im = u[off + d * d:off + 2 * d * d].reshape(d, d)
                    ch.append(torch.complex(re, im))
                    off += 2 * d * d
                hub_sp.append(ch)
        return w_sp, mats_sp, hub_sp

    def join(w_sp, mats_sp, hub_sp=None) -> Field:
        parts = ([w.reshape(-1) for w in w_sp]
                 + [m.reshape(-1) for mats in mats_sp for m in mats])
        if nsh:
            assert hub_sp is not None
            for ch in hub_sp:
                for m in ch:
                    parts.append(m.real.reshape(-1))
                    parts.append(m.imag.reshape(-1))
        return torch.cat(parts)

    zero_bec = [[torch.zeros(n, n, dtype=torch.float64, device=dev)
                 for n in nbec] for _ in range(nsp)]
    zero_hub = ([[torch.zeros(d, d, dtype=CDTYPE, device=dev) for d in hub_dims]
                 for _ in range(nsh)] if nsh else None)

    def pack(vbar_grid: torch.Tensor) -> Field:
        """RHS composite (v̄ on every spin's grid block, zero becsum/hub)."""
        return join([vbar_grid] * nsp, zero_bec, zero_hub)

    def symmetrize(w_sp, d_bare_sp):
        """𝒮ᵀu = 𝒮u (self-adjoint projections), mirroring the SCF's per-iteration
        symmetrisation on the transposed side. Identity for use_symmetry=False."""
        from gradwave.core.fftbox import g_to_r_box, r_to_g

        if system.rho_symmetrizer is not None:
            w_sp = [g_to_r_box(
                system.rho_symmetrizer.apply(r_to_g(w.to(CDTYPE))), real=True)
                for w in w_sp]
        if system.becsum_sym is not None:
            d_bare_sp = [[m.real for m in system.becsum_sym.apply(
                [m.to(CDTYPE) for m in ch])] for ch in d_bare_sp]
        return w_sp, d_bare_sp

    def m_apply(u: Field) -> Field:
        w_sp, d_bare_sp, d_hub_sp = split(u)
        w_sp, d_bare_sp = symmetrize(w_sp, d_bare_sp)
        # fresh zero warm-start so M(u) is deterministic in u
        dpsi_warm = [[torch.zeros_like(c[:n_sv]) for c, n_sv in
                      zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
                     for isp in range(nsp)]
        drho, dbec, dnh = cs.apply_chi0(w_sp, d_bare_sp, dpsi_warm, cg_tol,
                                        cg_max_iter, d_hub_sp=d_hub_sp)
        hub_term = cs.k_hub(cast("list[list[torch.Tensor]]", dnh)) if nsh else None
        out = join(_k_hxc_grid_scaled(cs, drho, fxc_scale),
                   cs.hvp_onecenter(dbec), hub_term)
        return coupling * out if coupling != 1.0 else out

    ref = torch.zeros(total, dtype=torch.float64, device=dev)
    return UsppScreening(apply=m_apply, ref=ref, pack=pack, split=split,
                         join=join, cs=cs)


def soft_subspace_uspp(res: USPPResult, xc: XCFunctional | SpinXC, *,
                       krylov: int = 24, n_modes: int = 1, seed: int = 0,
                       real_tol: float = 1e-4, block: int = 1,
                       **screen_kw) -> SoftSubspace:
    """Soft subspace of the USPP/PAW (+U) composite screening operator.

    Thin convenience over :func:`solvers.deflation.soft_subspace_from_operator`
    bound to the composite operator. ``block`` selects single-vector (1) or
    block-Arnoldi (≥ the degeneracy). ``**screen_kw`` forwards to
    :func:`build_uspp_screening` (``coupling``, ``fxc_scale``, ``cg_tol``)."""
    sc = build_uspp_screening(res, xc, **screen_kw)
    return soft_subspace_from_operator(sc.apply, sc.ref, krylov=krylov,
                                       n_modes=n_modes, seed=seed,
                                       real_tol=real_tol, block=block)


def solve_adjoint_deflated_uspp(res: USPPResult, xc: XCFunctional | SpinXC,
                                vbar_grid: torch.Tensor, *,
                                subspace: list[Field] | None = None,
                                auto_threshold: float = 0.9, max_modes: int = 8,
                                krylov: int = 30, seed: int = 0, block: int = 1,
                                beta: float = 0.2, tol: float = 1e-8,
                                max_iter: int = 200, history: int = 8,
                                **screen_kw) -> SolveResult:
    """Deflated drop-in for the USPP/PAW composite adjoint solve.

    Solves ``(1 − M) u = (v̄, 0, 0)`` on the composite vector and returns a
    ``SolveResult`` whose ``u`` is the flat composite (use the returned screening's
    ``split`` to unpack). ``vbar_grid`` is the per-loss grid gradient ∂L/∂ρ (seeded
    on every spin's grid block). Away from criticality it is plain Anderson; near a
    soft cluster it deflates. ``block`` uses block-Arnoldi for a degenerate cluster.
    ``**screen_kw`` forwards to :func:`build_uspp_screening`."""
    sc = build_uspp_screening(res, xc, **screen_kw)
    vbar = sc.pack(vbar_grid)
    if subspace is None:
        sub = soft_subspace_from_operator(sc.apply, sc.ref, krylov=krylov,
                                          n_modes=max_modes, seed=seed, block=block)
        subspace = [v for v, val in zip(sub.vectors, sub.values, strict=True)
                    if val.real > auto_threshold]
    if not subspace:
        return anderson_solve(sc.apply, vbar, beta=beta, tol=tol,
                              max_iter=max_iter, history=history)
    return deflated_solve(sc.apply, vbar, subspace, method="post", beta=beta,
                          tol=tol, max_iter=max_iter, history=history)
