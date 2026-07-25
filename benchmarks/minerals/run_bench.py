"""Run one mineral through BOTH gradwave and QE at identical settings and emit
a JSON result row. Heavy: intended to run on asus via gwq, never on the laptop.

  uv run python benchmarks/minerals/run_bench.py nio \
      --pseudo-dir ~/mineral_pseudos --outdir ~/mineral_bench/out --ranks 8

For the AFM minerals gradwave uses the collinear-magnetic (Shubnikov) k-fold
(setup_system(magmoms=..., collinear_magnetic=True), unshifted mesh) and the
johnson mixer; QE uses per-sublattice starting_magnetization so it detects the
same magnetic symmetry. We report the irreducible k-count from both codes, plus
gradwave's time-reversal-only count, so the magnetic reduction factor is visible
and the two codes' k-reduction can be compared apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # structures.py / qe.py

RY = 13.605693122994


def compute_nbands(upfs, species) -> tuple[int, float]:
    nelec = float(sum(upfs[s].z_valence for s in species))
    occ = nelec / 2.0
    nb = int(math.ceil(occ)) + max(8, int(round(0.20 * occ)))
    return nb, nelec


def gradwave_kcount(m, ecut_ev, upfs, magnetic: bool) -> int:
    """Irreducible k-count only (ecut-independent) -- build cheaply at low ecut."""
    from gradwave.scf.loop import setup_system
    if magnetic:
        sysm = setup_system(m.cell, m.positions, m.species, upfs,
                            ecut=10 * RY, kmesh=m.kmesh, use_symmetry=True,
                            magmoms=m.magmoms, collinear_magnetic=True)
    else:
        sysm = setup_system(m.cell, m.positions, m.species, upfs,
                            ecut=10 * RY, kmesh=m.kmesh, use_symmetry=False)
    return len(sysm.spheres)


def run_gradwave(m, upfs, nbands, fft_shape, threads) -> dict:
    import torch

    from gradwave.core.xc.pbe import PBE
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.scf.loop import scf, setup_system
    torch.set_num_threads(threads)

    ecut_ev = m.ecut_ry * RY
    width_ev = m.degauss_ry * RY

    # irreducible k counts (cheap builds): magnetic (Shubnikov) vs TR-only
    nk_tr = gradwave_kcount(m, ecut_ev, upfs, magnetic=False)
    res = {"nk_tr_only": nk_tr}

    t0 = time.perf_counter()
    if m.nspin == 2:
        system = setup_system(m.cell, m.positions, m.species, upfs, ecut=ecut_ev,
                              kmesh=m.kmesh, nbands=nbands, use_symmetry=True,
                              magmoms=m.magmoms, collinear_magnetic=True,
                              fft_shape=fft_shape)
    else:
        system = setup_system(m.cell, m.positions, m.species, upfs, ecut=ecut_ev,
                              kmesh=m.kmesh, nbands=nbands, use_symmetry=True,
                              fft_shape=fft_shape)
    t_setup = time.perf_counter() - t0
    res["nk_irr"] = len(system.spheres)

    xc = SpinPBE() if m.nspin == 2 else PBE()
    kwargs = dict(smearing="gaussian", width=width_ev, etol=1e-8, rhotol=1e-7,
                  max_iter=200, mixing_scheme="johnson", verbose=False)
    if m.nspin == 2:
        kwargs.update(nspin=2, start_mag=m.start_mag)
    t0 = time.perf_counter()
    r = scf(system, xc, **kwargs)
    t_scf = time.perf_counter() - t0

    e = r.energies
    one_elec = float(e.kinetic) + float(e.local) + float(e.nonlocal_)
    res.update(
        converged=bool(r.converged), n_iter=int(r.n_iter),
        etot_eV=float(r.energies.free_energy), fermi_eV=float(r.fermi),
        tot_mag=float(r.mag_total), abs_mag=float(r.mag_abs),
        setup_s=t_setup, scf_s=t_scf, wall_s=t_setup + t_scf, threads=threads,
        fft_shape=list(system.grid.shape),
        e_one_electron_eV=one_elec, e_hartree_eV=float(e.hartree),
        e_xc_eV=float(e.xc), e_ewald_eV=float(e.ewald),
        e_smearing_eV=float(e.smearing),
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mineral")
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-qe", action="store_true")
    a = ap.parse_args()

    import qe as QE
    import structures as S

    from gradwave.pseudo.upf import parse_upf

    pdir = os.path.expanduser(a.pseudo_dir)
    outdir = Path(os.path.expanduser(a.outdir)) / a.mineral
    outdir.mkdir(parents=True, exist_ok=True)

    m = S.build(a.mineral)
    upfs = [parse_upf(Path(pdir) / p) for p in m.pseudos]
    nbands, nelec = compute_nbands(upfs, m.species)

    row: dict = {
        "mineral": m.name, "key": a.mineral, "nat": m.nat, "afm": m.afm,
        "kmesh": list(m.kmesh), "ecut_ry": m.ecut_ry,
        "ecutrho_ry": m.rho_factor * m.ecut_ry, "degauss_ry": m.degauss_ry,
        "nspin": m.nspin, "pseudo_kind": m.pseudo_kind, "pseudos": m.pseudos,
        "nbands": nbands, "nelec": nelec, "note": m.note,
    }

    # ---- QE first (pins the shared FFT grid) ----
    qe = None
    if not a.skip_qe:
        try:
            qe = QE.run(m, nbands, pdir, outdir / "qe", nranks=a.ranks)
        except Exception as e:  # noqa: BLE001
            qe = {"error": repr(e)}
    row["qe"] = qe
    fft_shape = qe.get("fft_dims") if qe and qe.get("job_done") else None

    # ---- gradwave (pinned to QE's FFT grid if available) ----
    try:
        row["gradwave"] = run_gradwave(m, upfs, nbands, fft_shape, a.threads)
    except Exception as e:  # noqa: BLE001
        import traceback
        row["gradwave"] = {"error": repr(e), "trace": traceback.format_exc()[-2000:]}

    # ---- comparison ----
    gw, qeo = row.get("gradwave", {}), qe or {}
    if gw.get("etot_eV") is not None and qeo.get("etot_eV") is not None:
        dE = gw["etot_eV"] - qeo["etot_eV"]
        row["dE_total_eV"] = dE
        row["dE_total_meV_per_atom"] = dE / m.nat * 1000
        # split off the position-independent Ewald constant: the ELECTRONIC
        # (KS) energy agreement is the real validation, unaffected by the
        # ion-ion Ewald term. dE_electronic == what dE_total would be if the
        # two codes' Ewald constants agreed.
        if gw.get("e_ewald_eV") is not None and qeo.get("e_ewald_eV") is not None:
            row["dE_ewald_eV"] = gw["e_ewald_eV"] - qeo["e_ewald_eV"]
            dE_el = dE - row["dE_ewald_eV"]
            row["dE_electronic_eV"] = dE_el
            row["dE_electronic_meV_per_atom"] = dE_el / m.nat * 1000
    if gw.get("wall_s") and qeo.get("wall_s"):
        row["speed_ratio_qe_over_gw"] = gw["wall_s"] / qeo["wall_s"]

    (outdir / "result.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
