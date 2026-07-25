"""E-field DFPT dielectric / Born response WITH IBZ symmetry reduction.

The decisive oracle (metamorphic invariant): ε∞ and the Born effective charges
computed on the symmetry-reduced (IBZ) mesh must equal those from the full
(time-reversal-only) mesh. This is the whole correctness statement for the
symmetrized field response — the density response Δρ_α is a polar vector field,
so the three field directions are folded together through the point group
(VectorFieldSymmetrizer) and the ε/Born tensors are star-summed after
convergence. A naive scalar fold (treating Δρ_α as totally symmetric) breaks it.

Two levels are checked, one non-circular and one full-tensor:

  * ε∞ is isotropic for cubic Si, so ``eps_iso`` and the mean Born charge are
    direct scalars — the IBZ values match the full mesh WITHOUT invoking the
    symmetrizer under test.
  * The full ε and Born tensors match the SYMMETRIC PROJECTION of the full-mesh
    result to ~1e-5. The full-mesh path does not symmetrize, so its raw Born
    carries a ~2e-3 numerical site-asymmetry (the two equivalent Si atoms come
    out unequal on the discrete mesh); the IBZ path's job is precisely to
    produce the projected (physical) tensor directly, which it does — the two
    Si Born tensors come out identical and ε exactly isotropic-diagonal.

Silicon (diamond Fd-3̄m, non-symmorphic glides) is the fixture: the (2×2×2)
mesh folds to 3 IBZ k-points from 8, so the run genuinely exercises the fold.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.symmetry import (
    find_spacegroup,
    symmetrize_atom_tensor,
    symmetrize_tensor,
)
from tests.helpers import RY, pseudo

FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.standard  # two insulator SCFs + two E-field DFPT solves
def test_dielectric_born_ibz_equals_full_mesh():
    torch.set_num_threads(8)
    a = 5.43
    si = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    cell = a / 2 * FCC
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    sg = find_spacegroup(cell, pos @ np.linalg.inv(cell), [0, 0])

    def run(use_sym):
        system = setup_system(cell, pos, [0, 0], [si], ecut=12 * RY,
                              kmesh=(2, 2, 2), use_symmetry=use_sym)
        res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
                  verbose=False)
        assert res.converged
        return system, dielectric_born(res, PBE(), cg_tol=1e-8, outer_tol=1e-6,
                                       max_outer=100)

    sys_full, full = run(False)
    sys_ibz, ibz = run(True)

    # the IBZ run must actually fold the mesh (else the test is vacuous)
    assert sys_ibz.sym is not None
    assert len(sys_ibz.kweights) < len(sys_full.kweights)

    # --- non-circular scalar oracle: isotropic ε∞ and mean Born charge ---
    assert abs(full["eps_iso"] - ibz["eps_iso"]) < 1e-4
    z_full = float(torch.diagonal(full["born"], dim1=1, dim2=2).mean())
    z_ibz = float(torch.diagonal(ibz["born"], dim1=1, dim2=2).mean())
    assert abs(z_full - z_ibz) < 1e-4

    # --- full-tensor oracle: IBZ == symmetric projection of the full mesh ---
    assert torch.allclose(symmetrize_tensor(full["eps"], sg, cell),
                          ibz["eps"], atol=5e-5)
    assert torch.allclose(symmetrize_atom_tensor(full["born"], sg, cell),
                          ibz["born"], atol=5e-5)

    # --- symmetry constraints the IBZ path imposes exactly (cubic Si) ---
    eps = ibz["eps"]
    assert float((eps - torch.diag(torch.diagonal(eps))).abs().max()) < 1e-8
    assert float(torch.diagonal(eps).std()) < 1e-8  # isotropic
    born = ibz["born"]
    assert torch.allclose(born[0], born[1], atol=1e-8)  # equivalent Si sites
    assert float((born[0] - torch.diag(torch.diagonal(born[0]))).abs().max()) < 1e-8
