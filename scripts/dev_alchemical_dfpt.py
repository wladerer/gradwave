"""Dev validation: relaxed alchemical eigenvalue/gap gradient (DFPT) vs FD.

Not a committed test — a scratch driver to confirm alchemical_gap_gradient's
composition DFPT matches a central finite difference of re-converged SCF
eigenvalues, on the CsPbI3 -> CsPbCl3 X-site substitution. Run on asus:

    uv run python scripts/dev_alchemical_dfpt.py

The analytic dε_i/dλ (relaxed) must match FD(re-converged ε_i) for a
non-degenerate band, and d(E_gap)/dλ must match FD(gap). use_symmetry=False and
smearing='none' throughout (the response regime).
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import alchemical_gap_gradient, setup_alchemical_substitution
from gradwave.scf.loop import scf

torch.set_num_threads(8)

RY = 13.605693122994
ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "qe" / "pseudos"
DG = ROOT / "benchmarks" / "delta_gauge" / "pseudos"

cs = parse_upf(FIX / "Cs_ONCV_PBE_sr.upf")
pb = parse_upf(DG / "Pb.upf")
iodine = parse_upf(FIX / "I_ONCV_PBE_sr.upf")
cl = parse_upf(FIX / "Cl_ONCV_PBE_sr.upf")

A = 6.29
cell = A * np.eye(3)
frac = np.array([
    [0.0, 0.0, 0.0],   # Cs
    [0.5, 0.5, 0.5],   # Pb
    [0.5, 0.5, 0.0],   # I
    [0.5, 0.0, 0.5],   # I
    [0.0, 0.5, 0.5],   # I
])
pos = frac @ cell
species = [0, 1, 2, 2, 2]
pseudos = [cs, pb, iodine]
X_SITES = {2: cl, 3: cl, 4: cl}

# fast dev settings: Gamma-only, modest cutoff. use_symmetry=False required.
ECUT = 30 * RY
KMESH = (1, 1, 1)
LAM0 = 0.5
SCF_KW = dict(smearing="none", etol=1e-10, rhotol=1e-9, max_iter=300, verbose=False)


def build(lam):
    return setup_alchemical_substitution(
        cell, pos, pseudos, species, X_SITES, lam, ecut=ECUT, kmesh=KMESH,
        use_symmetry=False)


def run(lam):
    return scf(build(lam), PBE(), **SCF_KW)


def main():
    res = run(LAM0)
    assert res.converged, "base SCF did not converge"
    eig = res.eigenvalues  # (nk, nb)
    occ = res.occupations
    n_occ = int((occ[0] > 1e-6).sum())
    print(f"nk={eig.shape[0]} nb={eig.shape[1]} n_occ={n_occ} fermi={res.fermi:.4f}")

    grad = alchemical_gap_gradient(res, PBE())
    print("\nanalytic (DFPT):")
    print(f"  VBM {grad.vbm_index}  dε/dλ relaxed={grad.dvbm:+.4f}  frozen={grad.dvbm_frozen:+.4f}")
    print(f"  CBM {grad.cbm_index}  dε/dλ relaxed={grad.dcbm:+.4f}  frozen={grad.dcbm_frozen:+.4f}")
    print(f"  d(E_gap)/dλ  relaxed={grad.dgap:+.4f}  frozen={grad.dgap_frozen:+.4f} eV")

    # central FD of re-converged eigenvalues for the SAME (k, band) edges, plus a
    # deep non-degenerate band as a clean DFPT check.
    h = 0.01
    rp, rm = run(LAM0 + h), run(LAM0 - h)
    assert rp.converged and rm.converged

    def fd_eps(k, b):
        return (float(rp.eigenvalues[k, b]) - float(rm.eigenvalues[k, b])) / (2 * h)

    vk, vb = grad.vbm_index
    ck, cb = grad.cbm_index
    print(f"\nfinite difference (h={h}, re-converged):")
    print(f"  VBM dε/dλ = {fd_eps(vk, vb):+.4f}   (analytic relaxed {grad.dvbm:+.4f})")
    print(f"  CBM dε/dλ = {fd_eps(ck, cb):+.4f}   (analytic relaxed {grad.dcbm:+.4f})")

    # deep valence band 0 at k=0 (usually non-degenerate, well separated)
    from gradwave.scf.alchemical import (
        _band_deps_dlam,
        alchemical_density_response,
    )
    from gradwave.scf.implicit import apply_k_hxc
    drho, dvloc_r, ddij = alchemical_density_response(res, PBE())
    dv_ks = dvloc_r + apply_k_hxc(res, PBE(), drho)
    for b in (0, 1, max(0, n_occ - 2)):
        ana = _band_deps_dlam(res, 0, b, dv_ks, ddij)
        print(f"  band0,{b}: analytic {ana:+.4f}  FD {fd_eps(0, b):+.4f}")

    # gap FD
    def gap_of(r):
        e, f = r.eigenvalues, r.occupations
        m = f > 1e-6
        return float(e[~m].min() - e[m].max())
    fd_gap = (gap_of(rp) - gap_of(rm)) / (2 * h)
    print(f"\n  d(E_gap)/dλ  FD = {fd_gap:+.4f}   analytic relaxed = {grad.dgap:+.4f} eV")
    print(f"  gap(λ0)={gap_of(res):.4f}  gap(λ0+h)={gap_of(rp):.4f}  gap(λ0-h)={gap_of(rm):.4f}")


if __name__ == "__main__":
    main()
