# ruff: noqa: E402, E501  # scratch probe (hardcoded conventions / sys.path)
"""Stage 0.5: decompose ELK's O1s core-level reference into own / external / total.

Mirror of the gradwave decomposition in ``xps_madelung_decomposition.py``, applied to
Elk's converged STATE.OUT. For each O muffin tin we read Elk's l=0 muffin-tin density
(rhomt) and Coulomb potential (vclmt) and split the O1s reference the same way gradwave
does:

  own_es(r)  = radial_poisson_to_R(rho_l0) - Z e^2/r         (own electrostatic, boundary=own monopole)
  full_es(r) = vclmt_l0(r)                                    (Elk's full Coulomb; has the nuclear -Z/r)
  V_ext(r)   = full_es(r) - own_es(r)                         (SMOOTH: the -Z/r cancels; the Madelung
                                                               field the rest of the crystal imposes)

Re-solving the deep O1s eigenvalue directly is grid-pinned (a ~1 eV shift on a -517 eV level
needs accuracy the coarse Elk core cusp mesh can't give), so the site-shift of each component
is taken to first order as <psi_1s| dV |psi_1s> with a fixed reference 1s density weight w(r)
(the eigenVECTOR shape is well-resolved even when the eigenVALUE pins). This is exactly the
physical decomposition and is what gradwave's re-solve approximates when its grid is fine.

  Delta_own  = <w| (own_es+vxc)_long - (own_es+vxc)_short >   (analogue of gw's -3.8 eV)
  Delta_ext  = <w| V_ext_long - V_ext_short >                 (what the core actually feels)
  Delta_extR = V_ext_long(R) - V_ext_short(R)                 (gw-style rigid BOUNDARY constant = dC0_ext)
  total      = Delta_own + Delta_ext  (cross-checked vs EVALCORE.OUT, Elk's own scalar-rel solve)

Elk stores real-harmonic coeffs f_lm(r); physical l=0 part = f_00(r)*Y00, Y00=1/sqrt(4pi). Elk is
atomic units (Bohr/Hartree); we convert to gradwave eV/Angstrom to reuse the radial machinery.

Usage (asus):
    uv run python experiments/autoapw/xps_elk_decomp.py <STATE.OUT> <EVALCORE.OUT> [Z_O=8] [ias_short,ias_long=1,2]
"""
import math
import sys

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, E2, HARTREE_EV
from gradwave.flapw.coulomb import radial_poisson_to_R
from gradwave.flapw.functionals import vxc_lda
from gradwave.flapw.radial import radial_eigs_tridiag

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from elk_onsite import load  # noqa: E402

Y00 = 1.0 / math.sqrt(4.0 * math.pi)


def read_evalcore_1s(path):
    o_vals, cur_o = [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("Species"):
            cur_o = "(O)" in s
        elif cur_o and s.startswith("n =  1, l =  0"):
            o_vals.append(float(s.split(":")[-1]))
    return o_vals


def atom_fields(d, ias, Z):
    """Return per-atom fields on the gradwave eV/Angstrom radial grid."""
    sp = d["ias_sp"][ias]
    nr = d["nrmt"][sp]
    rr = np.asarray(d["rsp"][sp][:nr], dtype=float)          # Bohr
    rho00 = np.asarray(d["rho"][ias][:nr, 0], dtype=float)
    vcl00 = np.asarray(d["vcl"][ias][:nr, 0], dtype=float)
    dx = float(np.log(rr[1]) - np.log(rr[0]))
    q_sph = float(np.sqrt(4 * math.pi) * np.trapezoid(rho00 * rr**2, rr))
    r_A = rr * BOHR_ANG
    rho_A = (rho00 * Y00) / BOHR_ANG**3                      # e / Angstrom^3
    drw_A = r_A * dx
    own_es = radial_poisson_to_R(rho_A, r_A, r_A[-1], drw_A) - Z * E2 / r_A   # eV
    vxc = vxc_lda(torch.tensor(rho_A)).numpy()               # eV
    full_es = vcl00 * Y00 * HARTREE_EV                       # eV (Elk, has -Z/r)
    v_ext = full_es - own_es                                 # eV (smooth Madelung field)
    return dict(rr=rr, r_A=r_A, dx=dx, q_sph=q_sph, own_es=own_es, vxc=vxc,
                full_es=full_es, v_ext=v_ext, rho_A=rho_A)


def core_weight(f, Z):
    """|u_1s|^2 weight (sums to 1 with the log-mesh dr=r*dx) from the own potential solve.
    The eigenVECTOR shape is robust even where the eigenVALUE pins."""
    v_own = f["own_es"] + f["vxc"]
    _, u = radial_eigs_tridiag(0, torch.tensor(f["r_A"]), f["dx"], torch.tensor(v_own), 1)
    u = np.asarray(u[:, 0])
    w = u**2 * (f["r_A"] * f["dx"])
    return w / w.sum()


def wavg(w, x):
    return float(np.sum(w * x))


def sample_vext(f, radii_bohr):
    return [float(np.interp(rb, f["rr"], f["v_ext"])) for rb in radii_bohr]


def main():
    state, evalcore = sys.argv[1], sys.argv[2]
    Z = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    ias_s, ias_l = ([int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 2])
    d = load(state)
    ev = read_evalcore_1s(evalcore)
    fs, fl = atom_fields(d, ias_s, Z), atom_fields(d, ias_l, Z)
    w = core_weight(fs, Z)                                   # fixed reference weight (short's 1s)

    R = fs["rr"][-1]
    probe = [0.02, 0.05, 0.10, 0.20, R * 0.5, R]
    print(f"# STATE={state}   R_MT={R:.3f} Bohr")
    print(f"# q_l0(short)={fs['q_sph']:.4f}  q_l0(long)={fl['q_sph']:.4f}  dq(l-s)={fl['q_sph']-fs['q_sph']:+.4f} e")
    print(f"# V_ext(r) [eV] at r(Bohr)={['%.3f'%x for x in probe]}")
    print(f"#   short: {['%+.3f'%v for v in sample_vext(fs, probe)]}")
    print(f"#   long : {['%+.3f'%v for v in sample_vext(fl, probe)]}")
    print(f"#   d(l-s):{['%+.3f'%(a-b) for a,b in zip(sample_vext(fl,probe), sample_vext(fs,probe))]}")

    d_own = wavg(w, (fl["own_es"] + fl["vxc"]) - (fs["own_es"] + fs["vxc"]))
    d_ext_core = wavg(w, fl["v_ext"] - fs["v_ext"])
    d_ext_R = float(np.interp(R, fl["rr"], fl["v_ext"]) - np.interp(R, fs["rr"], fs["v_ext"]))
    print("  === ELK O1s shift (long - short), eV ===")
    print(f"  own            = {d_own:+.4f}   <w| d(own_es+vxc) >")
    print(f"  external(core) = {d_ext_core:+.4f}   <w| dV_ext >  (near-nucleus weighted)")
    print(f"  external(R)    = {d_ext_R:+.4f}   dV_ext at R_MT  (gw-style rigid boundary const)")
    print(f"  total(own+extc)= {d_own + d_ext_core:+.4f}")
    if len(ev) >= 2:
        print(f"  EVALCORE total = {(ev[1]-ev[0])*HARTREE_EV:+.4f}   (Elk scalar-rel ground truth)")


if __name__ == "__main__":
    main()
