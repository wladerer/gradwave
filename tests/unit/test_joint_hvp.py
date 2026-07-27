"""Exact-Hvp foundations for joint Newton-CG (ideas.md steps 2-4).

Three things are checked here, all on tiny NC-insulator Si cells:

1. **Double-differentiability audit (step 2).** The joint energy graph must be
   twice-differentiable so autograd can supply exact Hessian-vector products.
   The one custom ``autograd.Function`` on the strain graph — ``_SBT`` (spherical
   Bessel transform, q = |k+G|(ε)) — was ``once_differentiable`` and is now
   double-backward-able; ``gradgradcheck`` on the form factors proves it, and an
   Hvp-vs-central-finite-difference-of-gradients check in fp64 proves the whole
   ``joint_energy`` (both fixed- and variable-cell) is exact-Hvp clean.

2. **Band-chunk checkpointing (step 3).** Wrapping the density FFT sandwich in
   ``torch.utils.checkpoint`` must not change energies (bitwise at fixed chunking)
   or gradients, and must compose with double-backward (use_reentrant=False).

3. **Steihaug Newton-CG (step 4).** A smoke test that the trust-region optimizer
   runs end-to-end and pulls a displaced Si atom back to its ideal bond.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.opt.joint import (
    _coeffs_from_z,
    joint_energy,
    lowdin,
    teter_precond,
)
from gradwave.pseudo.radial_torch import RadialTables
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_upf

A0 = 5.43


def _tiny_si(ecut=8 * RY, kmesh=(1, 1, 1), disp=None):
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [A0 / 4] * 3])
    if disp is not None:
        pos = pos.copy()
        pos[1] += disp
    system = setup_system(cell=cell, positions=pos, species_of_atom=[0, 0],
                          upfs=[si_upf()], ecut=ecut, kmesh=kmesh,
                          use_symmetry=False)
    return cell, pos, system


def _seeded_leaves(system, xc, *, fix_cell, n_occ=4):
    """Real optimizer leaves (eps, frac, z…) at a loose-SCF orbital seed, fp64 —
    exactly the graph the Newton-CG optimizer differentiates twice."""
    res = scf(system, xc, verbose=False)
    coeffs0 = [lowdin(c[:n_occ].to(CDTYPE)) for c in res.coeffs]
    ekin = float(np.mean([
        (system.spheres[ik].kpg2 * (coeffs0[ik].real ** 2
         + coeffs0[ik].imag ** 2)).sum().item()
        for ik in range(len(system.spheres))]))
    precond = [teter_precond(sph.kpg2, ekin).to(RDTYPE) for sph in system.spheres]
    npws = [sph.npw for sph in system.spheres]
    cell = np.asarray(system.grid.cell)
    frac0 = np.asarray(system.positions.cpu()) @ np.linalg.inv(cell)
    eps = torch.zeros(3, 3, dtype=RDTYPE, requires_grad=not fix_cell)
    frac = torch.tensor(frac0, dtype=RDTYPE, requires_grad=True)
    zs = [torch.view_as_real((coeffs0[ik] / precond[ik][None, :].to(CDTYPE))
                             .contiguous()).clone().requires_grad_(True)
          for ik in range(len(system.spheres))]
    leaves = ([frac] if fix_cell else [eps, frac]) + zs
    occ = torch.full((len(system.spheres), n_occ), 2.0, dtype=RDTYPE)
    tabs = [RadialTables(u) for u in system.upfs]

    def energy(_leaves, *, checkpoint=False, band_chunk=None):
        if fix_cell:
            _eps = torch.zeros(3, 3, dtype=RDTYPE)
            _frac, _zs = _leaves[0], _leaves[1:]
        else:
            _eps, _frac, _zs = _leaves[0], _leaves[1], _leaves[2:]
        coeffs = _coeffs_from_z(_zs, precond, npws)
        return joint_energy(system, xc, tabs, _eps, _frac, coeffs, occ,
                            checkpoint=checkpoint, band_chunk=band_chunk)

    return leaves, energy


# ------------------------------------------------------- step 2: audit
def test_sbt_double_backward():
    """The strain-graph form factors (via _SBT, formerly once_differentiable)
    are now twice differentiable in |G| — gradcheck AND gradgradcheck in fp64."""
    tab = RadialTables(si_upf())
    q = torch.linspace(0.3, 6.0, 5, dtype=torch.float64, requires_grad=True)
    fns = [tab.vloc_of_g,
           *[lambda qq, i=i: tab.beta_of_g(i, qq)
             for i in range(len(tab.beta_l))]]
    if tab.core_g is not None:  # only when the pseudo carries an NLCC core
        fns.append(tab.core_of_g)
    for fn in fns:
        assert torch.autograd.gradcheck(fn, (q,), atol=1e-6, rtol=1e-4)
        assert torch.autograd.gradgradcheck(fn, (q,), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("fix_cell", [True, False])
def test_joint_energy_hvp_matches_fd(fix_cell):
    """Exact double-backward Hvp = central finite difference of gradients, fp64.

    fix_cell=False exercises the strain block — the one that rides _SBT — so this
    is the end-to-end proof the joint energy is exact-Hvp clean for variable cell.
    """
    _, _, system = _tiny_si()
    leaves, energy = _seeded_leaves(system, LDA_PW92(), fix_cell=fix_cell)
    torch.manual_seed(0)
    v = [torch.randn_like(t) for t in leaves]

    e = energy(leaves)
    g = torch.autograd.grad(e, leaves, create_graph=True)
    gv = sum((gi * vi).sum() for gi, vi in zip(g, v, strict=True))
    hv = torch.autograd.grad(gv, leaves)

    h = 1e-5

    def grad_at(shift):
        ls = [t.detach().clone().requires_grad_(True) for t in leaves]
        with torch.no_grad():
            for t, vi in zip(ls, v, strict=True):
                t.add_(shift * vi)
        ee = energy(ls)
        return torch.autograd.grad(ee, ls)

    gp, gm = grad_at(h), grad_at(-h)
    hv_fd = [(a - b) / (2 * h) for a, b in zip(gp, gm, strict=True)]

    num = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(hv, hv_fd, strict=True)))
    den = torch.sqrt(sum((b ** 2).sum() for b in hv_fd))
    assert float(num / (den + 1e-12)) < 2e-3


# --------------------------------------------- step 3: checkpointing
def test_checkpoint_preserves_energy_and_grad():
    """Band-chunk checkpointing changes neither the energy (bitwise at fixed
    chunking) nor the gradient, and composes with double-backward."""
    _, _, system = _tiny_si()
    leaves, energy = _seeded_leaves(system, LDA_PW92(), fix_cell=False)

    e_plain = float(energy(leaves, checkpoint=False, band_chunk=2).detach())
    e_ckpt = float(energy(leaves, checkpoint=True, band_chunk=2).detach())
    # identical chunk boundaries + identical ops → bitwise (allow 1e-14 slack)
    assert abs(e_plain - e_ckpt) <= 1e-14 * (1 + abs(e_plain))

    g_plain = torch.autograd.grad(energy(leaves, band_chunk=2), leaves,
                                  create_graph=True)
    g_ckpt = torch.autograd.grad(
        energy(leaves, checkpoint=True, band_chunk=2), leaves, create_graph=True)
    for a, b in zip(g_plain, g_ckpt, strict=True):
        assert torch.allclose(a, b, atol=1e-12)

    # double-backward through the checkpointed graph (use_reentrant=False)
    torch.manual_seed(1)
    v = [torch.randn_like(t) for t in leaves]
    hv_plain = torch.autograd.grad(sum((gi * vi).sum() for gi, vi
                                       in zip(g_plain, v, strict=True)), leaves,
                                   retain_graph=True)
    hv_ckpt = torch.autograd.grad(sum((gi * vi).sum() for gi, vi
                                      in zip(g_ckpt, v, strict=True)), leaves)
    for a, b in zip(hv_plain, hv_ckpt, strict=True):
        assert torch.allclose(a, b, atol=1e-10)


# ------------------------------------------- step 4: Newton-CG smoke
def test_newton_cg_fixed_cell_smoke():
    """Trust-region Newton-CG runs end-to-end and relaxes a displaced Si atom
    toward its ideal bond on a cheap Γ-only cell."""
    from gradwave.opt.newton import newton_cg_relax

    cell, pos, _ = _tiny_si(disp=[0.08, -0.05, 0.03])
    res = newton_cg_relax(cell, pos, [0, 0], [si_upf()], LDA_PW92(),
                          ecut=8 * RY, kmesh=(1, 1, 1), fix_cell=True,
                          fmax=0.02, ctol=2e-3, max_newton=25)
    bond = np.linalg.norm(res.positions[1] - res.positions[0])
    ideal = A0 * np.sqrt(3) / 4
    assert abs(bond - ideal) < abs(np.linalg.norm(pos[1] - pos[0]) - ideal)
    assert res.n_hvp > 0 and res.h_equiv > 0
    assert res.converged


@pytest.mark.standard
def test_newton_cg_fixed_cell_recovers_si_bond():
    """Positions-only Newton-CG converges to the ideal Si bond (2×2×2 mesh)."""
    from gradwave.opt.newton import newton_cg_relax

    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [A0 / 4] * 3])
    pos[1] += [0.10, -0.06, 0.04]
    res = newton_cg_relax(cell, pos, [0, 0], [si_upf()], LDA_PW92(),
                          ecut=12 * RY, kmesh=(2, 2, 2), fix_cell=True,
                          fmax=0.005, max_newton=40)
    assert res.converged
    bond = np.linalg.norm(res.positions[1] - res.positions[0])
    assert bond == pytest.approx(A0 * np.sqrt(3) / 4, abs=2e-3)
    assert res.fmax < 0.005
