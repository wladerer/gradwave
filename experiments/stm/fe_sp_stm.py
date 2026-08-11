"""Spin-polarized STM of a bcc Fe(100) surface — spin-resolved Tersoff-Hamann.

True graphene/Fe is a Moire supercell (hundreds of atoms); this instead showcases
the STM module on a magnetic transition metal: a spin-polarized Fe(100) slab, then
separate spin-up / spin-down LDOS maps at E_F -> spin-resolved STM (SP-STM), which
resolves the surface magnetic structure.
"""

from __future__ import annotations

import numpy as np

from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.stm import stm_constant_height
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
UPF = "tests/fixtures/qe/pseudos/Fe_ONCV_PBE-1.2.upf"


def main():
    a = 2.866
    nlayer, vac = 3, 7.0
    c = (nlayer - 1) * a / 2 + 2 * vac
    cell = np.diag([a, a, c])
    z0 = vac
    pos = []
    for L in range(nlayer):
        off = 0.0 if L % 2 == 0 else 0.5
        pos.append([off * a, off * a, z0 + L * a / 2])
    pos = np.array(pos)
    upf = parse_upf(UPF)
    print(f"# Fe(100) SP-STM: {nlayer} layers, a={a} A, ecut=38 Ry, k=4x4x1, nspin=2", flush=True)
    system = setup_system(
        cell, pos, [0] * nlayer, [upf], ecut=38 * RY, kmesh=(4, 4, 1), use_symmetry=True
    )
    print(
        f"# npw~{system.spheres[0].npw}, nk_ibz={len(system.spheres)}, grid={system.grid.shape}",
        flush=True,
    )
    res = scf(
        system,
        SpinPBE(),
        smearing="gaussian",
        width=0.2,
        nspin=2,
        start_mag=[0.35] * nlayer,
        mixing_alpha=0.3,
        max_iter=200,
        etol=1e-6,
        rhotol=1e-5,
        verbose=True,
    )
    print(
        f"# converged={res.converged} in {res.n_iter}; E_F={res.fermi:.3f} eV; "
        f"M_tot={res.mag_total:.3f} uB",
        flush=True,
    )
    up, z = stm_constant_height(res, height=2.5, energy=res.fermi, sigma=0.3, spin=0)
    dn, _ = stm_constant_height(res, height=2.5, energy=res.fermi, sigma=0.3, spin=1)
    np.save("/tmp/fe_stm_up.npy", up.detach().cpu().numpy())
    np.save("/tmp/fe_stm_dn.npy", dn.detach().cpu().numpy())
    np.save("/tmp/fe_cell.npy", cell)
    np.save("/tmp/fe_pos.npy", pos)
    print(
        f"# spin-up contrast {float(up.max() / max(up.min(), 1e-30)):.1f}, "
        f"spin-dn contrast {float(dn.max() / max(dn.min(), 1e-30)):.1f}, z_tip={z:.2f}",
        flush=True,
    )
    print("EXIT_OK", flush=True)


if __name__ == "__main__":
    main()
