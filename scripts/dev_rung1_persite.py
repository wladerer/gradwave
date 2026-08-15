"""Dev validation: per-site vector-λ alchemical gap gradient (rung 1).

Two experiments, run on asus:

  A. Charge-conserving CO-substitution MgO -> NaCl (rocksalt): Mg(10)+O(6)=16 ->
     Na(9)+Cl(7)=16, so each component is aliovalent (cation -1e, anion +1e) but
     the TOTAL valence is conserved -> N=16 fixed, insulating throughout. The
     coupled d(E_gap)/dλ is finite-difference-verifiable; the per-site
     decomposition sums to it.

  B. Per-site vector λ on CsPbI3's three X->Cl sites (each isovalent, 7->7): each
     site's d(E_gap)/dλ_k is individually FD-verifiable, and Σ_k = the coupled
     scalar gradient = coupled FD.

    uv run python scripts/dev_rung1_persite.py
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import (
    alchemical_gap_gradient,
    alchemical_gap_gradient_per_site,
    setup_alchemical_substitution,
)
from gradwave.scf.loop import scf

torch.set_num_threads(8)
RY = 13.605693122994
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qe" / "pseudos"
DG = Path(__file__).resolve().parents[1] / "benchmarks" / "delta_gauge" / "pseudos"


def gap(res):
    e, f = res.eigenvalues, res.occupations
    m = f > 1e-6
    return float(e[~m].min() - e[m].max())


def rocksalt(a):
    cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    return cell, pos


def test_a_cosubstitution():
    print("=== A. MgO -> NaCl charge-conserving co-substitution ===")
    mg = parse_upf(FIX / "Mg_ONCV_PBE-1.2.upf")
    o = parse_upf(FIX / "O_ONCV_PBE-1.2.upf")
    na_ = parse_upf(FIX / "Na_ONCV_PBE_sr.upf")
    cl = parse_upf(FIX / "Cl_ONCV_PBE_sr.upf")
    cell, pos = rocksalt(4.9)
    subs = {0: na_, 1: cl}          # Mg->Na (site 0), O->Cl (site 1)
    ecut, km = 45 * RY, (2, 2, 2)
    kw = dict(smearing="none", etol=1e-10, rhotol=1e-9, max_iter=400, verbose=False)

    def build(lam):
        return setup_alchemical_substitution(cell, pos, [mg, o], [0, 1], subs, lam,
                                             ecut=ecut, kmesh=km, use_symmetry=False)

    def run(lam):
        return scf(build(lam), PBE(), **kw)

    lam = 0.5
    res = run(lam)
    print(f"  N_elec={res.system.n_electrons:.3f}  converged={res.converged}  gap={gap(res):.4f} eV")
    g = alchemical_gap_gradient(res, PBE())
    ps = alchemical_gap_gradient_per_site(res, PBE())
    h = 0.02
    fd = (gap(run(lam + h)) - gap(run(lam - h))) / (2 * h)
    per_site_sum = sum(v.dgap for v in ps.values())
    print(f"  coupled  d(E_gap)/dλ  analytic={g.dgap:+.4f}  FD={fd:+.4f}  "
          f"(agree {abs(g.dgap - fd) * 1e3:.2f} meV)")
    print(f"  per-site: " + "  ".join(f"site{k}={v.dgap:+.4f}" for k, v in ps.items()))
    print(f"  Σ per-site={per_site_sum:+.4f}  vs coupled={g.dgap:+.4f}  "
          f"(consistency {abs(per_site_sum - g.dgap) * 1e3:.3f} meV)")
    print(f"  frozen coupled={g.dgap_frozen:+.4f} eV")


def test_b_persite_isovalent():
    print("\n=== B. CsPbI3 three X->Cl sites, per-site isovalent vector λ ===")
    cs = parse_upf(FIX / "Cs_ONCV_PBE_sr.upf")
    pb = parse_upf(DG / "Pb.upf")
    iod = parse_upf(FIX / "I_ONCV_PBE_sr.upf")
    cl = parse_upf(FIX / "Cl_ONCV_PBE_sr.upf")
    A = 6.29
    cell = A * np.eye(3)
    frac = np.array([[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    pos = frac @ cell
    species = [0, 1, 2, 2, 2]
    subs = {2: cl, 3: cl, 4: cl}
    ecut, km = 30 * RY, (1, 1, 1)
    kw = dict(smearing="none", etol=1e-9, rhotol=1e-9, max_iter=300, verbose=False)
    sub_order = [2, 3, 4]           # substitutions insertion order == vector order

    def build(lam):
        return setup_alchemical_substitution(cell, pos, [cs, pb, iod], species, subs,
                                             lam, ecut=ecut, kmesh=km,
                                             use_symmetry=False)

    def run(lam):
        return scf(build(lam), PBE(), **kw)

    lam0 = 0.4
    res = run(lam0)
    print(f"  converged={res.converged}  gap={gap(res):.4f} eV")
    ps = alchemical_gap_gradient_per_site(res, PBE())
    g = alchemical_gap_gradient(res, PBE())

    # single-site FD: perturb ONLY site k (isovalent, stays insulating)
    h = 0.03
    for j, k in enumerate(sub_order):
        vp = [lam0, lam0, lam0]; vp[j] += h
        vm = [lam0, lam0, lam0]; vm[j] -= h
        fd = (gap(run(vp)) - gap(run(vm))) / (2 * h)
        ana = ps[k].dgap
        print(f"  site {k}: analytic dGap/dλ_k={ana:+.4f}  FD={fd:+.4f}  "
              f"(agree {abs(ana - fd) * 1e3:.2f} meV)")
    per_site_sum = sum(v.dgap for v in ps.values())
    fd_coupled = (gap(run(lam0 + h)) - gap(run(lam0 - h))) / (2 * h)
    print(f"  Σ per-site={per_site_sum:+.4f}  coupled analytic={g.dgap:+.4f}  "
          f"coupled FD={fd_coupled:+.4f}")


if __name__ == "__main__":
    test_a_cosubstitution()
    test_b_persite_isovalent()
