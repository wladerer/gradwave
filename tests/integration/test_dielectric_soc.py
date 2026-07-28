"""ε∞ and Born effective charges for the fully-relativistic (spin-orbit)
spinor path (nonmagnetic manifold), postscf.dielectric._dielectric_born_soc.

Self-oracle: SOC "turned off" by construction. Take a plain scalar-relativistic
UPF (Si_ONCV_PBE_sr.upf, j=None on every beta) and build a SYNTHETIC
fully-relativistic twin where each l≥1 radial channel is duplicated into a
j=l+1/2 and a j=l-1/2 projector with IDENTICAL radial data and D-matrix value
(l=0 keeps its single physical j=1/2 channel unchanged). This is not an
approximation: the spin-angular completeness relation

    Σ_{j=l±1/2, mj} |Ω_{jlmj}⟩⟨Ω_{jlmj}| = Σ_m |Y_l^m⟩⟨Y_l^m| ⊗ 1_2 (spin)

means the synthetic j-resolved nonlocal operator is EXACTLY (not
approximately) the doubled scalar-relativistic nonlocal operator, for any
spinor wavefunction — i.e. this pseudopotential has, by construction, zero
spin-orbit splitting. Running it through the SOC (``is_fr``) SCF and dielectric
path must therefore reproduce the plain scalar-relativistic
(nspin=1) SCF/``dielectric_born`` result to solver precision; a bug in the
spinor ∂H/∂k, the SOC-projector Born-charge term, or the nonmagnetic K_Hxc
screening kernel would show up as an O(1) discrepancy, not a subtle one.
"""

import dataclasses

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import UPFData, parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import PSEUDOS, RY, si_fcc

pytestmark = pytest.mark.slow  # two full insulator SCFs + E-field DFPT


def _degenerate_fr_upf(upf: UPFData) -> UPFData:
    """A synthetic fully-relativistic twin of a scalar-relativistic UPF with
    exactly zero SOC splitting (see module docstring for why this is an EXACT
    reduction, not an approximation)."""
    betas = []
    new_idx_map = []  # per original beta index: the new beta index/indices
    for b in upf.betas:
        if b.l == 0:  # only one physical j (=1/2) exists for l=0
            betas.append(dataclasses.replace(b, j=0.5))
            new_idx_map.append([len(betas) - 1])
        else:
            betas.append(dataclasses.replace(b, j=b.l + 0.5))
            betas.append(dataclasses.replace(b, j=b.l - 0.5))
            new_idx_map.append([len(betas) - 2, len(betas) - 1])
    nnew = len(betas)
    dij_new = np.zeros((nnew, nnew))
    for i, idxs_i in enumerate(new_idx_map):
        for j, idxs_j in enumerate(new_idx_map):
            v = upf.dij[i, j]
            if len(idxs_i) != len(idxs_j):
                continue  # only possible if the two betas' l disagree in
                # a way the original dij had already zeroed (block-diagonal
                # in l), so nothing is lost by skipping the mismatched shape
            for a in range(len(idxs_i)):
                dij_new[idxs_i[a], idxs_j[a]] = v
    return dataclasses.replace(upf, betas=tuple(betas), dij=dij_new)


def test_dielectric_soc_zero_splitting_matches_scalar():
    """A j-degenerate (zero-SOC-splitting) synthetic FR Si reproduces the
    scalar-relativistic ε∞/Z* to solver precision."""
    torch.set_num_threads(8)
    cell, pos = si_fcc(5.43)
    si = parse_upf(PSEUDOS / "Si_ONCV_PBE_sr.upf")

    system = setup_system(cell, pos, [0, 0], [si], ecut=10 * RY,
                          kmesh=(2, 2, 2), use_symmetry=False)
    r1 = scf(system, LDA_PW92(), smearing="none", etol=1e-10, rhotol=1e-9,
             verbose=False)
    assert r1.converged
    kw = dict(cg_tol=1e-8, outer_tol=1e-6, max_outer=60)
    out1 = dielectric_born(r1, LDA_PW92(), **kw)

    si_fr = _degenerate_fr_upf(si)
    system_fr = setup_system(cell, pos, [0, 0], [si_fr], ecut=10 * RY,
                             kmesh=(2, 2, 2), use_symmetry=False,
                             time_reversal=False)
    assert system_fr.is_fr
    xc = NoncollinearXC(LSDA_PW92())
    r2 = scf_noncollinear(system_fr, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                          smearing="gaussian", width=0.01, nonmagnetic=True,
                          etol=1e-10, rhotol=1e-9, verbose=False, max_iter=200)
    assert r2.converged
    # the synthetic pseudo has ZERO SOC splitting: the FR total energy must
    # match the scalar-relativistic total to numerical precision, not just
    # "approximately" — a decisive check the degenerate construction itself
    # is correct before trusting the dielectric comparison below.
    assert abs(float(r2.energies.total) - float(r1.energies.total)) < 1e-8

    out2 = dielectric_born(r2, xc, **kw)

    eps_err = float((out2["eps"] - out1["eps"]).abs().max())
    born_err = float((out2["born"] - out1["born"]).abs().max())
    assert eps_err < 1e-5, f"eps mismatch {eps_err}\n{out1['eps']}\n{out2['eps']}"
    assert born_err < 1e-5, (
        f"born mismatch {born_err}\n{out1['born']}\n{out2['born']}")

    # still a physical isotropic diagonal tensor
    eps2 = out2["eps"]
    assert float((eps2 - torch.diag(torch.diagonal(eps2))).abs().max()) < 1e-3


def test_dielectric_soc_rejects_magnetic():
    """A nonzero moment must raise (the coupled (ρ, m⃗) K_Hxc HVP a magnetic
    SOC dielectric response needs is not implemented). Si's true ground state
    is nonmagnetic, so rather than fight the SCF to hold a spurious moment,
    patch a converged nonmagnetic result's ``m`` directly — the gate only
    needs to see a nonzero moment, not a self-consistent magnetic state."""
    torch.set_num_threads(4)
    cell, pos = si_fcc(5.43)
    si = parse_upf(PSEUDOS / "Si_ONCV_PBE_sr.upf")
    si_fr = _degenerate_fr_upf(si)
    system_fr = setup_system(cell, pos, [0, 0], [si_fr], ecut=8 * RY,
                             kmesh=(1, 1, 1), use_symmetry=False,
                             time_reversal=False)
    xc = NoncollinearXC(LSDA_PW92())
    res = scf_noncollinear(system_fr, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                           smearing="gaussian", width=0.05, nonmagnetic=True,
                           etol=1e-8, rhotol=1e-7, verbose=False, max_iter=100)
    assert res.converged
    res_mag = dataclasses.replace(res, m=res.m + 0.01)
    with pytest.raises(NotImplementedError, match="nonmagnetic"):
        dielectric_born(res_mag, xc)
