"""Discretization-error estimator calibration probe: raw sweep.

Feasibility probe for docs/ideas.md "Error estimation: the rest of the budget".
Measures how truthful gradwave's single-shot plane-wave (Ecut) error estimate
(postscf.discretization_error) is across a small diverse set of in-repo systems,
whether its bias is systematic enough to calibrate, and how large the
(ecut, k-mesh) cross term is that the additive error budget ignores.

Design (the controlled comparison):
  * One CONVERGED reference cutoff ecut_ref per system.
  * Three loose cutoffs at 0.50, 0.65, 0.80 * ecut_ref.
  * The FFT (density) grid is FIXED to the reference cutoff's natural grid for
    every run of a system, so the only thing changing between loose and
    reference is the plane-wave orbital sphere -- exactly what the estimator
    addresses. (If the grid stepped with ecut, the truth would carry a density-
    grid jump the estimator never claims to see. delta_gauge fixes the grid the
    same way, benchmarks/delta_gauge/ecut_scan.py.)
  * ecut_large is set to ecut_ref, so the estimator's annulus [ecut_loose,
    ecut_ref] matches the truth window [E(ecut_loose) -> E(ecut_ref)] exactly.
  * TRUTH = E_total(ecut_ref) - E_total(ecut_loose), same k-mesh, same grid.
    Both KS total energies (res.energies.total), same smearing/width.

Outputs results.json (raw numbers). analyze.py does the fits.

Run locally (small) or offload to asus per CLAUDE.md:
  GW_THREADS=8 uv run python experiments/error_calibration/sweep.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("GW_THREADS", "2")))

from gradwave.constants import RY_EV  # noqa: E402
from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.postscf.discretization_error import estimate_density_error  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402

PSE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "qe" / "pseudos"
OUT = Path(__file__).resolve().parent / "results.json"

# fcc / diamond share the fcc primitive lattice; rocksalt too (2-atom basis).
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _fcc(a):
    return (a / 2.0) * FCC


def build_geometry(kind, a):
    """(cell, cart positions, species-index list) for a cubic crystal.

    kind: 'fcc' (1 atom), 'diamond' (2 same), 'rocksalt' (2 different).
    a is the conventional cubic lattice constant [Ang].
    """
    cell = _fcc(a)
    if kind == "fcc":
        frac = np.array([[0.0, 0.0, 0.0]])
        sp = [0]
    elif kind == "diamond":
        frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        sp = [0, 0]
    elif kind == "rocksalt":
        frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        sp = [0, 1]
    else:
        raise ValueError(kind)
    return cell, frac @ cell, sp


# ecut_ref chosen genuinely converged (well above the ONCV PBE hint); loose
# cutoffs are fractions of it. widths/kmesh: insulators no smearing (well-
# defined gap descriptor), metals gaussian on a denser mesh.
GAUSS_W = 0.01 * RY_EV  # eV

SYSTEMS = [
    dict(name="Si_diamond", kind="diamond", a=5.43, pseudos=["Si_ONCV_PBE-1.2.upf"],
         ecut_ref_ry=55.0, kmesh=4, smear="none", width=0.0, nbands=None, metal=False),
    dict(name="C_diamond", kind="diamond", a=3.567, pseudos=["C_ONCV_PBE-1.2.upf"],
         ecut_ref_ry=90.0, kmesh=4, smear="none", width=0.0, nbands=None, metal=False),
    dict(name="Al_fcc", kind="fcc", a=4.05, pseudos=["Al_ONCV_PBE-1.2.upf"],
         ecut_ref_ry=45.0, kmesh=8, smear="gaussian", width=GAUSS_W, nbands=12, metal=True),
    dict(name="MgO_rocksalt", kind="rocksalt", a=4.21,
         pseudos=["Mg_ONCV_PBE-1.2.upf", "O_ONCV_PBE-1.2.upf"],
         ecut_ref_ry=90.0, kmesh=4, smear="none", width=0.0, nbands=None, metal=False),
    dict(name="Cu_fcc", kind="fcc", a=3.61, pseudos=["Cu_ONCV_PBE-1.2.upf"],
         ecut_ref_ry=90.0, kmesh=8, smear="gaussian", width=GAUSS_W, nbands=24, metal=True),
    dict(name="NaCl_rocksalt", kind="rocksalt", a=5.64,
         pseudos=["Na_ONCV_PBE_sr.upf", "Cl_ONCV_PBE_sr.upf"],
         ecut_ref_ry=75.0, kmesh=4, smear="none", width=0.0, nbands=None, metal=False),
]

FACTORS = [0.50, 0.65, 0.80]


def _gap(res, nspin=1):
    ev = np.concatenate([np.asarray(x).reshape(-1) for x in res.eigenvalues])
    oc = np.concatenate([np.asarray(x).reshape(-1) for x in res.occupations])
    full = 2.0 / nspin
    tol = 1e-6
    frac = (oc > tol) & (np.abs(oc - full) > tol)
    if frac.any() or not (oc > tol).any() or not (oc <= tol).any():
        return None
    homo = ev[oc > tol].max()
    lumo = ev[oc <= tol].min()
    return float(lumo - homo) if lumo > homo else 0.0


def run_scf(cell, pos, sp, upfs, ecut, kmesh, smear, width, nbands, fft_shape):
    system = setup_system(cell, pos, sp, upfs, ecut=ecut, kmesh=(kmesh,) * 3,
                          nbands=nbands, use_symmetry=True, fft_shape=fft_shape)
    res = scf(system, PBE(), smearing=smear, width=width, etol=1e-9,
              rhotol=1e-8, max_iter=200, verbose=False)
    return system, res


def truthfulness_and_grid():
    results = {"systems": {}, "meta": {"factors": FACTORS}}
    for s in SYSTEMS:
        upfs = [parse_upf(str(PSE / p)) for p in s["pseudos"]]
        cell, pos, sp = build_geometry(s["kind"], s["a"])
        ecut_ref = s["ecut_ref_ry"] * RY_EV
        nat = len(sp)

        t0 = time.time()
        # reference: natural grid at ecut_ref -> fixes the FFT grid for the system
        sys_ref, res_ref = run_scf(cell, pos, sp, upfs, ecut_ref, s["kmesh"],
                                   s["smear"], s["width"], s["nbands"], None)
        fixed_fft = tuple(int(x) for x in sys_ref.grid.shape)
        e_ref = float(res_ref.energies.total)
        gap_ref = _gap(res_ref)
        print(f"[{s['name']}] ref ecut={s['ecut_ref_ry']:.0f}Ry grid={fixed_fft} "
              f"E={e_ref:.6f} conv={res_ref.converged} it={res_ref.n_iter} "
              f"gap={gap_ref} ({time.time()-t0:.0f}s)", flush=True)

        points = []
        for f in FACTORS:
            ecut_loose = f * ecut_ref
            t1 = time.time()
            _sysL, resL = run_scf(cell, pos, sp, upfs, ecut_loose, s["kmesh"],
                                  s["smear"], s["width"], s["nbands"], fixed_fft)
            e_loose = float(resL.energies.total)
            gap_loose = _gap(resL)
            # estimator: annulus [ecut_loose, ecut_ref] == truth window
            err = estimate_density_error(resL, ecut_large=ecut_ref)
            est = float(err.denergy)                 # <= 0
            true = e_ref - e_loose                   # <= 0 (loose sits above ref)
            ratio = est / true if true != 0 else float("nan")
            points.append(dict(
                factor=f, ecut_loose_ry=f * s["ecut_ref_ry"],
                e_loose=e_loose, e_ref=e_ref,
                est_denergy_eV=est, true_denergy_eV=true, ratio=ratio,
                est_meV_atom=1000.0 * est / nat, true_meV_atom=1000.0 * true / nat,
                gap_loose_eV=gap_loose, converged=bool(resL.converged),
                n_iter=int(resL.n_iter)))
            print(f"    f={f:.2f} ecut={f*s['ecut_ref_ry']:.1f}Ry "
                  f"est={est*1e3:.3f} true={true*1e3:.3f} meV ratio={ratio:.3f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

        results["systems"][s["name"]] = dict(
            kind=s["kind"], a=s["a"], natoms=nat, metal=s["metal"],
            ecut_ref_ry=s["ecut_ref_ry"], kmesh=s["kmesh"],
            fixed_fft=list(fixed_fft), e_ref=e_ref, gap_ref_eV=gap_ref,
            points=points)
    return results


def cross_term():
    """(ecut, k-mesh) coupling on fcc Al. 3 cutoffs x 3 meshes, fixed grid.

    Reference corner = (ecut_ref, k=8). Marginals:
      dE_ecut(i) = E(ecut_i, 8) - E(ecut_ref, 8)
      dE_k(j)    = E(ecut_ref, k_j) - E(ecut_ref, 8)
      joint(i,j) = E(ecut_i, k_j) - E(ecut_ref, 8)
      cross(i,j) = joint - dE_ecut - dE_k
    """
    s = next(x for x in SYSTEMS if x["name"] == "Al_fcc")
    upfs = [parse_upf(str(PSE / p)) for p in s["pseudos"]]
    cell, pos, sp = build_geometry(s["kind"], s["a"])
    ecut_ref = s["ecut_ref_ry"] * RY_EV
    factors = [0.50, 0.65, 1.00]          # tightest (1.00) is the reference cut
    meshes = [4, 6, 8]                     # tightest (8) is the reference mesh

    # fix the FFT grid to the reference-cutoff natural grid
    sys_ref = setup_system(cell, pos, sp, upfs, ecut=ecut_ref, kmesh=(8, 8, 8),
                           nbands=s["nbands"], use_symmetry=True)
    fixed_fft = tuple(int(x) for x in sys_ref.grid.shape)

    E = {}   # (fi, mj) -> total energy [eV]
    for f in factors:
        for m in meshes:
            t0 = time.time()
            _sy, res = run_scf(cell, pos, sp, upfs, f * ecut_ref, m,
                               s["smear"], s["width"], s["nbands"], fixed_fft)
            E[(f, m)] = float(res.energies.total)
            print(f"[Al cross] f={f:.2f} k={m} E={E[(f,m)]:.6f} "
                  f"conv={res.converged} ({time.time()-t0:.0f}s)", flush=True)

    e0 = E[(1.00, 8)]
    grid = []
    for f in factors:
        for m in meshes:
            joint = E[(f, m)] - e0
            marg_e = E[(f, 8)] - e0
            marg_k = E[(1.00, m)] - e0
            cross = joint - marg_e - marg_k
            grid.append(dict(factor=f, kmesh=m, e=E[(f, m)],
                             joint_eV=joint, marg_ecut_eV=marg_e,
                             marg_k_eV=marg_k, cross_eV=cross))
    return dict(fixed_fft=list(fixed_fft), factors=factors, meshes=meshes,
                e_ref_corner=e0, grid=grid)


def main():
    t0 = time.time()
    results = truthfulness_and_grid()
    results["cross_term_Al"] = cross_term()
    results["meta"]["wall_seconds"] = time.time() - t0
    results["meta"]["threads"] = torch.get_num_threads()
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
