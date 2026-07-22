"""COHP fat-band structure for Bi2Se3, with and without spin-orbit coupling.

Bi2Se3 is the textbook Z2 topological insulator: WITHOUT SOC it is an ordinary
narrow-gap semiconductor, and turning SOC on INVERTS the gap-edge states at Gamma
(their inversion parities swap — Zhang et al., Nat. Phys. 2009). This example
draws the LOBSTER-style COHP fat-band figure of examples/cohp_fatbands.py for
both cases along Gamma-Z-F-Gamma-L, so the Bi-Se bonding character of the
inverted states is visible alongside the parity/irrep labels at Gamma.

  no SOC : scalar-relativistic PseudoDojo pseudos, collinear PBE, operator-route
           COHP; Gamma states labelled by Mulliken irrep + g/u parity (band_irreps).
  SOC    : fully-relativistic PseudoDojo pseudos, non-magnetic spinor PBE,
           eigenvalue-route spinor COHP (cohp_soc); Gamma states labelled by the
           inversion parity of the spinor wavefunction (spinor_parity).

The SOC fat-band weights come from a per-path-k spinor solve that mirrors
scf.noncollinear.band_structure_nc but KEEPS the eigenvectors, then feeds them
through the spinor COHP helpers (the same quantity cohp_soc exposes on the SCF
mesh, evaluated along the path). Absolute ICOHP magnitude is qualitative (see the
gradwave.postscf.cohp docstring); sign and bonding/antibonding shape are correct.

Run:  uv run python examples/bi2se3_cohp_fatbands.py [--outdir out] [--npoints 60]
      uv run python examples/bi2se3_cohp_fatbands.py --only soc   # one branch
Heavy (5-atom cell, 3d/5d semicore, spinor path solve) — minutes to tens of
minutes on CPU; offload to a many-core box if available.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from ase import Atoms

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.bands import bands_along_ase_path
from gradwave.postscf.cohp import (
    _htilde_eig,
    _pair_block_weights,
    cohp,
    cohp_soc,
)
from gradwave.postscf.pdos import (
    _ao_spinor_projectors_k,
    _atomic_columns_so,
    _lowdin_project,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear

# reuse the plotting + collinear fat-band machinery from the sibling example
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from examples.cohp_fatbands import (  # noqa: E402
    PSEUDOS,
    cohp_fatbands,
    irreps_at_specials,
    plot_material,
)

RY = 13.605693122994

# Rhombohedral R-3m primitive cell (Se1 at the inversion centre), same geometry
# as examples/bi2se3_inversion.py and bi2se3_bands_compare.py.
A_HEX, C_HEX = 4.138, 28.64
MU, NU = 0.399, 0.206
CELL = np.array([
    [A_HEX / 2, A_HEX / (2 * np.sqrt(3)), C_HEX / 3],
    [-A_HEX / 2, A_HEX / (2 * np.sqrt(3)), C_HEX / 3],
    [0.0, -A_HEX / np.sqrt(3), C_HEX / 3],
])
FRAC = np.array([[0.0, 0, 0], [NU, NU, NU], [-NU, -NU, -NU],
                 [MU, MU, MU], [-MU, -MU, -MU]])
SPECIES = [0, 0, 0, 1, 1]        # 0 = Se (atoms 0,1,2), 1 = Bi (atoms 3,4)
BI_IDX, SE_IDX = [3, 4], [0, 1, 2]
ECUT = 45 * RY
KMESH = (2, 2, 2)
PATH = "GZFGL"
WINDOW = (-6.0, 5.0)             # eV around the VBM; deep 3d/5d semicore excluded
NBANDS = 44                      # 78 valence electrons -> 39 occupied collinear bands


def _vbm(eigenvalues, fermi) -> float:
    """Valence-band maximum = highest SCF-mesh eigenvalue below the Fermi level."""
    e = np.asarray(eigenvalues)
    below = e[e < fermi]
    return float(below.max()) if below.size else float(fermi)


def _nearest_bond(system):
    """Shortest Bi-Se atom pair (0-based) by a true lattice search.

    cohp's _min_image_dist uses the naive per-component round() convention, which
    is unreliable for this acute rhombohedral cell (the nearest Bi-Se image needs
    a lattice-vector combination independent rounding misses). Search a 5x5x5 shell
    of translations so the reported bond length is physical (~2.8 A).
    """
    cell = np.asarray(system.grid.cell, float)
    pos = system.positions.detach().cpu().numpy()
    ns = range(-2, 3)
    shifts = np.array([[a, b, c] for a in ns for b in ns for c in ns]) @ cell
    best, pair = np.inf, None
    for i in BI_IDX:
        for j in SE_IDX:
            dmin = float(np.linalg.norm(pos[i] - pos[j] + shifts, axis=1).min())
            if dmin < best:
                best, pair = dmin, (min(i, j), max(i, j))
    return pair, best


# --------------------------------------------------------------------------- #
#  no-SOC branch (collinear, operator-route COHP)                             #
# --------------------------------------------------------------------------- #
def run_nosoc(outdir: pathlib.Path, npoints: int) -> dict:
    print("[Bi2Se3 no-SOC] SCF (scalar-rel. PseudoDojo, collinear PBE) ...",
          flush=True)
    upfs = [parse_upf(str(PSEUDOS / "PD_Se_PBE_std.upf")),
            parse_upf(str(PSEUDOS / "PD_Bi_PBE_std.upf"))]
    system = setup_system(CELL, FRAC @ CELL, SPECIES, upfs, ecut=ECUT,
                          kmesh=KMESH, nbands=NBANDS, use_symmetry=True)
    res = scf(system, PBE(), smearing="gaussian", width=0.05,
              etol=1e-7, rhotol=1e-6, verbose=False)
    print(f"[Bi2Se3 no-SOC] converged={res.converged} in {res.n_iter} iters, "
          f"E_F={res.fermi:.3f} eV", flush=True)
    bond, dist = _nearest_bond(system)
    ref = _vbm(res.eigenvalues.cpu().numpy(), res.fermi)
    print(f"[Bi2Se3 no-SOC] Bi-Se bond {bond} d={dist:.3f} A, VBM={ref:.3f} eV",
          flush=True)

    atoms = Atoms("Se3Bi2", scaled_positions=FRAC % 1.0, cell=CELL, pbc=True)
    bs = bands_along_ase_path(res, atoms, path=PATH, npoints=npoints, nbands=NBANDS)
    bp = atoms.cell.bandpath(path=PATH, npoints=npoints)
    print(f"[Bi2Se3 no-SOC] COHP fat-bands along {PATH} ({len(bp.kpts)} k) ...",
          flush=True)
    eig, wgt = cohp_fatbands(res, bp.kpts, bond, NBANDS)

    tick_names = list(dict.fromkeys(lab for _, lab in bs.labels))
    irr = irreps_at_specials(res, bp.special_points, tick_names, NBANDS)
    for nm, cl in irr.items():
        print(f"    {nm}: {', '.join(c['label'] for c in cl[:8])} ...", flush=True)

    ch = cohp(res, pairs=[bond], rcut=dist + 0.3, width=0.1, npoints=800,
              window=(ref - 6.5, ref + 5.5))
    blab = f"{bond[0] + 1}-{bond[1] + 1}"
    data = _pack("bi2se3_nosoc", "Bi$_2$Se$_3$ (no SOC)", ref, res.fermi, bs, eig,
                 wgt, bond, irr, ch, blab)
    _dump(data, outdir)
    return data


# --------------------------------------------------------------------------- #
#  SOC branch (spinor path solve + eigenvalue-route spinor COHP)              #
# --------------------------------------------------------------------------- #
def _rebuild_nc_potential(res, xc):
    """Converged spinor V(r) and B_xc(r) from (rho, m), mirroring
    band_structure_nc; B_xc is zeroed for a non-magnetic run."""
    from gradwave.core.energies.hartree import hartree_potential_g
    from gradwave.core.energies.local_pp import local_potential_g
    from gradwave.core.fftbox import r_to_g
    from gradwave.core.xc.noncollinear import vxc_and_bxc
    from gradwave.dtypes import CDTYPE

    system = res.system
    grid = system.grid
    rho_g_box = r_to_g(res.rho.to(CDTYPE))
    v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                           dim=(-3, -2, -1)) * grid.n_points).real
    v_xc, b_xc, _ = vxc_and_bxc(xc, res.rho, res.m, grid, rho_core=system.rho_core)
    if float(res.m.abs().max()) < 1e-12:
        b_xc = torch.zeros_like(b_xc)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real
    return v_h + v_xc + vloc_r, b_xc


@torch.no_grad()
def spinor_cohp_fatbands(res, xc, kpts_frac, bond, nbands):
    """Per-(path-k, band) spinor COHP weight on `bond` and the eigenvalues.

    Mirrors scf.noncollinear.band_structure_nc (frozen-potential spinor Davidson)
    but keeps the eigenvectors, then applies the eigenvalue-route spinor COHP of
    cohp_soc per k. Returns (eigenvalues (nk, nb) [eV], weights (nk, nb) [eV]).
    """
    from gradwave.core.batch import build_batched, projectors_b
    from gradwave.core.hamiltonian import build_projector_data
    from gradwave.core.spinor_proj import build_so_projectors
    from gradwave.dtypes import CDTYPE, RDTYPE
    from gradwave.grids import build_gsphere
    from gradwave.pseudo.kb import beta_form_factors
    from gradwave.scf.noncollinear import SpinorHamiltonian
    from gradwave.solvers.davidson import davidson_batched_ms

    system = res.system
    grid = system.grid
    device = res.rho.device
    v_r, b_xc = _rebuild_nc_potential(res, xc)
    cols = _atomic_columns_so(system)
    atom_of = np.array([c.atom for c in cols])
    i, j = bond

    kpts = np.asarray(kpts_frac, dtype=float)
    eig = np.empty((len(kpts), nbands))
    wgt = np.empty((len(kpts), nbands))

    for ik, k in enumerate(kpts):
        sph = build_gsphere(grid, system.ecut, k, device=device)
        npw = sph.npw
        so_tabs = [torch.zeros(1, u.n_proj, npw, dtype=RDTYPE, device=device)
                   for u in system.upfs]
        q_amp = np.sqrt(sph.kpg2.cpu().numpy())
        for sp_i, u in enumerate(system.upfs):
            so_tabs[sp_i][0, :, :npw] = torch.as_tensor(
                beta_form_factors(u, q_amp), dtype=RDTYPE, device=device)
        pd = build_projector_data(
            sph, system.species_of_atom, [t[0, :0] for t in so_tabs],
            [[] for _ in system.upfs],
            [torch.as_tensor(u.dij, dtype=RDTYPE, device=device)
             for u in system.upfs], grid.volume)
        bk = build_batched([sph], [pd], device=device)
        q_so, dij_so = build_so_projectors(bk, system, so_tables=so_tabs)
        h = SpinorHamiltonian(bk, grid.shape, v_r, b_xc,
                              projectors_b(bk, system.positions),
                              q=q_so, dij_so=dij_so)
        m_pw = bk.npw_max
        c0 = torch.zeros(1, nbands, 2 * m_pw, dtype=CDTYPE, device=device)
        for b_i in range(nbands):
            c0[0, b_i, (b_i // 2) + (b_i % 2) * m_pw] = 1.0
        t2 = torch.cat([bk.t, bk.t], dim=-1)
        mask2 = torch.cat([bk.mask, bk.mask], dim=-1)
        out = davidson_batched_ms(h.apply, c0, t2, mask2, tol=1e-8, max_iter=100)
        c = out.eigenvectors[0]                          # (nb, 2*m_pw)
        e = out.eigenvalues[0]                           # (nb,)

        cu, cd = c[:, :npw], c[:, m_pw:m_pw + npw]
        qu, qd = _ao_spinor_projectors_k(system, sph, cols, device)
        becp = (torch.einsum("bg,pg->bp", cu, qu.conj())
                + torch.einsum("bg,pg->bp", cd, qd.conj()))
        overlap = (torch.einsum("pg,qg->pq", qu.conj(), qu)
                   + torch.einsum("pg,qg->pq", qd.conj(), qd))
        proj = _lowdin_project(becp, overlap)            # (nb, nproj)
        htilde = _htilde_eig(proj, e)
        w = _pair_block_weights(proj, htilde, atom_of, i, j, 2.0)  # g_spin=1

        eig[ik] = e.cpu().numpy()
        wgt[ik] = w
    return eig, wgt


def _spinor_parity_gamma(system, coeffs_gamma, eig_gamma, ne, ref):
    """Inversion parity (g/u) of the spinor gap-edge states at Gamma.

    <psi|P|psi> for each band (P = inversion about the origin, permuting G -> -G
    on both spin components). Returns plot-ready clusters near the gap; Kramers
    pairs share a parity, so plot_material's proximity dedup collapses them.
    """
    ig = [i for i, sp in enumerate(system.spheres)
          if np.abs(sp.k_frac).max() < 1e-9][0]
    sph = system.spheres[ig]
    miller = sph.miller.cpu().numpy()
    index = {tuple(m): i for i, m in enumerate(miller)}
    perm = np.array([index[tuple(-m)] for m in miller])
    npw, m_pw = sph.npw, system.batch.npw_max
    clusters = []
    lo, hi = max(0, ne - 8), min(len(eig_gamma), ne + 8)
    for band in range(lo, hi):
        c = coeffs_gamma[band].cpu().numpy()
        p = sum(np.vdot(c[off:off + npw], c[off:off + npw][perm]).real
                for off in (0, m_pw))
        clusters.append({"e": float(eig_gamma[band]), "dim": 1,
                         "label": "g" if p >= 0 else "u",
                         "warning": f"parity {p:+.2f}"})
    return {"G": clusters}


def run_soc(outdir: pathlib.Path, npoints: int) -> dict:
    print("[Bi2Se3 SOC] spinor SCF (fully-rel. PseudoDojo, non-magnetic PBE) ...",
          flush=True)
    upfs = [parse_upf(str(PSEUDOS / "PD_Se_FR.upf")),
            parse_upf(str(PSEUDOS / "PD_Bi_FR.upf"))]
    system = setup_system(CELL, FRAC @ CELL, SPECIES, upfs, ecut=ECUT,
                          kmesh=KMESH, nbands=NBANDS, time_reversal=False)
    xc = NoncollinearXC(SpinPBE())
    res = scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 0]] * 5,
                           smearing="gaussian", width=0.05, etol=1e-7,
                           rhotol=1e-6, verbose=False, nonmagnetic=True)
    ne = int(round(system.n_electrons))     # spinor bands hold one electron
    print(f"[Bi2Se3 SOC] converged={res.converged} in {res.n_iter} iters, "
          f"E_F={res.fermi:.3f} eV, ne={ne}", flush=True)
    bond, dist = _nearest_bond(system)
    ref = _vbm(res.eigenvalues.cpu().numpy(), res.fermi)
    print(f"[Bi2Se3 SOC] Bi-Se bond {bond} d={dist:.3f} A, VBM={ref:.3f} eV",
          flush=True)

    atoms = Atoms("Se3Bi2", scaled_positions=FRAC % 1.0, cell=CELL, pbc=True)
    bp = atoms.cell.bandpath(path=PATH, npoints=npoints)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    labels = list(zip(xticks.tolist(), list(xlabels), strict=True))
    nb_path = 2 * system.nbands
    print(f"[Bi2Se3 SOC] spinor COHP fat-bands along {PATH} "
          f"({len(bp.kpts)} k, {nb_path} bands) ...", flush=True)
    eig, wgt = spinor_cohp_fatbands(res, xc, bp.kpts, bond, nb_path)

    ig = [i for i, sp in enumerate(system.spheres)
          if np.abs(sp.k_frac).max() < 1e-9][0]
    irr = _spinor_parity_gamma(system, res.coeffs[ig],
                               res.eigenvalues[ig].cpu().numpy(), ne, ref)
    print("    G parity (g/u) near gap: "
          + ", ".join(f"{c['label']}@{c['e'] - ref:+.2f}" for c in irr["G"]),
          flush=True)

    ch = cohp_soc(res, pairs=[bond], rcut=dist + 0.3, width=0.1, npoints=800,
                  window=(ref - 6.5, ref + 5.5))
    blab = f"{bond[0] + 1}-{bond[1] + 1}"
    data = _pack("bi2se3_soc", "Bi$_2$Se$_3$ (SOC)", ref, res.fermi, None, eig,
                 wgt, bond, irr, ch, blab, x=x, labels=labels)
    _dump(data, outdir)
    return data


# --------------------------------------------------------------------------- #
def _pack(slug, title, ref, fermi, bs, eig, wgt, bond, irr, ch, blab,
          x=None, labels=None):
    if bs is not None:
        x, labels = np.asarray(bs.x).tolist(), \
            [[float(xt), lab] for xt, lab in bs.labels]
    else:
        x, labels = np.asarray(x).tolist(), \
            [[float(xt), lab] for xt, lab in labels]
    return {
        "material": slug, "title": title,
        "reference_eV": float(ref), "fermi_eV": float(fermi),
        "x": x, "labels": labels,
        "eigenvalues_eV": eig.tolist(), "cohp_weight": wgt.tolist(),
        "bond": list(bond), "irreps": irr,
        "curve_energy_eV": ch.energy_eV.tolist(),
        "curve_cohp": ch.pair_cohp[blab].tolist(),
        "pair_icohp_eV": float(ch.pair_icohp[blab]),
        "spilling": float(ch.spilling), "charge_spilling": float(ch.charge_spilling),
    }


def _dump(data, outdir):
    jpath = outdir / f"{data['material']}_cohp_fatbands.json"
    jpath.write_text(json.dumps(data))
    print(f"[{data['material']}] wrote {jpath}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--npoints", type=int, default=60)
    ap.add_argument("--only", choices=["nosoc", "soc"], default=None)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = {"nosoc": run_nosoc, "soc": run_soc}
    for key, fn in runs.items():
        if args.only in (None, key):
            data = fn(outdir, args.npoints)
            plot_material(data, outdir, window=WINDOW)


if __name__ == "__main__":
    main()
