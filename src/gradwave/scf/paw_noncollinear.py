"""Non-collinear one-center (PAW augmentation) exchange-correlation.

The collinear ``OneCenter._exc_t`` evaluates a spin XC on the radial×angular
quadrature with the two channels being n_up and n_down. For a non-collinear moment
the on-site density at each quadrature point is a scalar n and a 3-vector m⃗; the
locally-collinear approximation diagonalizes n·1 + m⃗·σ into

    n± = (n ± |m⃗|) / 2,

feeds those to the *same* collinear XC, and (for the potential) rotates the result
back to v·1 + B⃗·σ with B⃗ ∥ m̂. This is the radial-mesh analog of
``core/xc/noncollinear.py``'s ``vxc_and_bxc`` on the smooth grid, and the on-site
half of a non-collinear PAW SCF (the augmentation one-center XC).

It reduces to the collinear ``_exc_t`` when m⃗ ∥ ẑ, and the energy depends only on
|m⃗| (rotation invariant, as it must be without spin-orbit) — the two validation
handles used in ``examples``/tests.

LDA only for now: a non-collinear GGA on-site XC needs the gradient of |m⃗| (the
smooth-grid ``NoncollinearXC`` carries the same restriction), so a GGA functional
raises here.

Scope: this module is the *on-site XC* half of a full non-collinear PAW SCF — the
one piece that is standalone and exactly testable (collinear limit + rotation +
autograd field, all to machine precision; see tests/integration/test_paw_
noncollinear.py). The remaining, tightly-coupled pieces of a spinor PAW SCF (to be
built on top of ``scf/uspp_loop.py`` + ``scf/noncollinear.py``) are:

  1. a doubled coefficient layout (nk, nb, 2·npw) replacing the per-spin coeff lists;
  2. a spinor generalized-eigen operator — merge ``SpinorHamiltonian.apply`` (the
     v·1+B⃗·σ local mix) with the batched PAW ``h``/dual-grid, plus an S⊗1₂ overlap
     apply, solved by ``davidson_gen_batched``;
  3. a 2×2-in-spin on-site becsum (accumulate uu/dd/ud/du blocks from the two spinor
     components) → decompose to (n_ij, m⃗_ij);
  4. this on-site XC returning a 2×2 ddd, added as spin-diagonal *and* off-diagonal
     D_ij blocks in the nonlocal apply;
  5. a 4-channel augmentation charge (n_aug→ρ, m⃗_aug→m⃗) from the 2×2 becsum;
  6. a 4-channel MixLayout (Kerker on n only) carrying the 2×2 becsum.

Validation ladder for the full loop: the collinear limit (all moments ∥ ẑ must
reproduce the collinear nspin=2 PAW energy) and rotation invariance (energy
independent of the global spin axis without SOC) — the same self-checks that
validated the norm-conserving spinor SCF, and the two handles already proven for
this on-site XC. SOC-in-the-augmentation (the all-electron core term that makes PAW
more accurate than a norm-conserving FR pseudo for MAE/DMI) is a further stage.
"""

from __future__ import annotations

import torch


def onsite_nc_exc(oc, nm_lms, what: str):
    """Non-collinear one-center E_xc [eV] (locally-collinear LDA).

    ``oc`` is a ``paw_onsite.OneCenter``; ``nm_lms`` = [n_lm, mx_lm, my_lm, mz_lm],
    each a real (mesh, l2) lm-expanded radial density (as returned by
    ``oc.rho_lm_t``); ``what`` is "ae" or "ps". Returns a torch scalar.
    """
    if getattr(oc.xc, "needs_gradient", False):
        raise NotImplementedError(
            "non-collinear one-center XC is LDA-only; got a GGA functional")
    tt = oc._torch_tables()
    core = tt["core_ae"] if what == "ae" else tt["core_ps"]
    ylm, rm2 = tt["ylm"], tt["rm2"]
    # radial values at the (mesh, nx) quadrature points; r² folded in via rm2.
    # the unpolarized core adds to n only (m⃗ is a pure valence-spin quantity).
    n_rad = (nm_lms[0] @ ylm.T) * rm2[:, None] + core[:, None]
    m_rad = torch.stack([(nm_lms[i] @ ylm.T) * rm2[:, None] for i in (1, 2, 3)])
    mmag = torch.linalg.norm(m_rad, dim=0)                 # |m⃗| per quad point
    up = (0.5 * (n_rad + mmag)).reshape(-1)
    dn = (0.5 * (n_rad - mmag)).reshape(-1)
    e = oc.xc.eval_energy_density(up, dn)
    return (e.reshape(oc.mesh, oc.nx) * tt["wq"][:, None] * tt["ww"][None, :]).sum()


def onsite_nc_energy_and_field(oc, nm_lms, what: str):
    """(E_xc [eV], [dE/dn_lm, dE/dmx_lm, dE/dmy_lm, dE/dmz_lm]) — the one-center XC
    energy and its exact autograd derivative w.r.t. the four density channels. The
    n-derivative is the scalar potential and the m⃗-derivative is the on-site B⃗-field
    (the spin part of the ddd), both consistent with ``onsite_nc_exc`` to machine
    precision by construction."""
    leaves = [x.detach().clone().requires_grad_(True) for x in nm_lms]
    with torch.enable_grad():
        e = onsite_nc_exc(oc, leaves, what)
        grads = torch.autograd.grad(e, leaves)
    return float(e.detach()), list(grads)


def e1c_nc_t(oc, comps):
    """Full non-collinear one-center energy [eV] (AE − PS), Hartree on the density
    channel + the non-collinear XC. ``comps`` = [n_ij, mx_ij, my_ij, mz_ij], the four
    Pauli channels of the 2×2-in-spin on-site becsum, each a real (nm, nm) tensor
    (n_ij = ρ↑↑+ρ↓↓, mz_ij = ρ↑↑−ρ↓↓, mx_ij = ρ↑↓+ρ↓↑, my_ij = i(ρ↑↓−ρ↓↑); all
    Hermitian → real-representable). The on-site Hartree is spin-independent, so it
    couples to n only."""
    e = torch.zeros((), dtype=torch.float64)
    for what, sgn in (("ae", 1.0), ("ps", -1.0)):
        lms = [oc.rho_lm_t(c, what) for c in comps]      # [n_lm, mx_lm, my_lm, mz_lm]
        _, e_h = oc.hartree_t(lms[0])                     # Hartree on n only
        e = e + sgn * (e_h + onsite_nc_exc(oc, lms, what))
    return e


def onsite_nc_energy_and_ddd(oc, comps):
    """(E_1c [eV], [ddd_n, ddd_mx, ddd_my, ddd_mz]) — the non-collinear one-center
    energy and the on-site potential as its exact autograd derivative w.r.t. the four
    becsum channels. The 2×2-in-spin on-site potential is D = ddd_n·1 + ddd_m⃗·σ⃗
    (i.e. D↑↑ = ddd_n+ddd_mz, D↓↓ = ddd_n−ddd_mz, D↑↓ from ddd_mx, ddd_my) — the
    v·1 + B⃗·σ that enters the spinor Hamiltonian's nonlocal D_ij. In the collinear
    limit (mx=my=0), ddd_n ± ddd_mz reproduce the collinear ``energy_and_ddd``'s
    [ddd_up, ddd_down]."""
    leaves = [c.detach().clone().requires_grad_(True) for c in comps]
    with torch.enable_grad():
        e = e1c_nc_t(oc, leaves)
        grads = torch.autograd.grad(e, leaves)
    return float(e.detach()), list(grads)
