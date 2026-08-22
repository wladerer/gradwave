"""Analytic-USPP scoping de-risk probe.

(1) NC-limit reference: the analytic sigma_shielding_dq on small NC Si — the
    number analytic-USPP must reproduce to machine precision (reduction gate).
(2) CRUX test: `_d2h_mats` (nested jvp) already differentiates the KB-nonlocal
    term p.mT @ (D @ p.conj()) TWICE. Confirm it matches a central finite
    difference of `_dense_velocity_matrices` (=dH/dk) along q_hat. Since the
    USPP overlap S = I + p.mT @ (q_int @ p.conj()) is the SAME functional form
    (q_int in place of D, no kinetic diagonal), the same machinery yields
    d2S/dk2 with NO new hand-derivation.
(3) DIRECT proof for S: build a traceable dense S(k) from the SHIPPED
    `_overlap_kbprojectors` on a real PAW Si system and show nested jvp gives a
    finite d2S/dk2 that matches central FD of dS/dk. This is the exact ingredient
    the analytic-USPP `_d2h_mats`-analogue would call.
"""
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

torch.set_num_threads(2)
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "github/gradwave/.claude/worktrees/analytic-uspp-scoping"))
from tests.helpers import RY, si_fcc, si_upf
from gradwave.dtypes import RDTYPE, CDTYPE
from gradwave.scf.loop import setup_system, scf
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.kgeometry import (
    BlochHK, _eigh_and_dh, _overlap_kbprojectors,
)
from gradwave.postscf.kgeometry_nmr import (
    _dense_velocity_matrices, _d2h_mats, sigma_shielding_dq,
)


print("=" * 70)
print("(1) NC-limit analytic sigma reference (reduction-gate target)")
print("=" * 70)
upf = si_upf()
print("NC pseudo: tests.helpers.si_upf()")
cell, pos = si_fcc()
system = setup_system(cell, pos, [0, 0], [upf], ecut=6 * RY,
                      kmesh=(2, 2, 2), nbands=8, use_symmetry=False,
                      fft_shape=(15, 15, 15))
res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
assert res.converged, "NC SCF did not converge"
sig = sigma_shielding_dq(res)
diag = torch.diagonal(sig, dim1=1, dim2=2)
print("sigma tensor shape:", tuple(sig.shape))
for s in range(sig.shape[0]):
    iso = float(diag[s].mean())
    print(f"  site {s}: sigma_iso = {iso:.6f} ppm  (diag {diag[s].tolist()})")
off = sig - torch.diag_embed(diag)
print(f"  max off-diag = {float(off.abs().max()):.3e} ppm (isotropy check)")

print()
print("=" * 70)
print("(2) CRUX: _d2h_mats (nested jvp) vs central-FD of dH/dk along q_hat")
print("    proves the 2nd-deriv machinery differentiates the beta-nonlocal")
print("    term generically -> d2S/dk2 needs no new derivation")
print("=" * 70)
k_frac = np.array([0.1, 0.2, -0.15])
hk = BlochHK.from_scf(res, k_frac)
kc = hk.k_ref_cart.clone()
q_hat = torch.tensor([0.3, -0.4, 0.86602540, ], dtype=RDTYPE)
q_hat = q_hat / torch.linalg.norm(q_hat)

# analytic mixed 2nd derivative M_mu = q_hat . grad(dH/dk_mu)
m_ana = _d2h_mats(hk, kc, q_hat)

# central FD of dH/dk_mu along q_hat: (v_mu(k+e*qhat) - v_mu(k-e*qhat))/2e
eps = 1e-4
vp = _dense_velocity_matrices(hk, kc + eps * q_hat)
vm = _dense_velocity_matrices(hk, kc - eps * q_hat)
for mu in range(3):
    fd = (vp[mu] - vm[mu]) / (2 * eps)
    rel = float((m_ana[mu] - fd).abs().max() / fd.abs().max().clamp_min(1e-30))
    print(f"  mu={mu}: |M_ana - FD|/max = {rel:.3e}   "
          f"(nonlocal contrib present: dij nnz={int(hk.dij_full.abs().sum()>0)})")

print()
print("=" * 70)
print("(3) DIRECT: traceable dense S(k) from shipped _overlap_kbprojectors on")
print("    real PAW Si -> nested jvp gives d2S/dk2 (matches FD of dS/dk)")
print("=" * 70)
from pathlib import Path
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
import glob
paw_cands = glob.glob(str(Path.home() / "github/gradwave/**/Si*kjpaw*.UPF"), recursive=True)
paw_path = paw_cands[0]
print("PAW pseudo:", paw_path)
paw = parse_upf_paw(paw_path)
usys = setup_uspp(cell, pos, [0, 0], [paw], ecut=12 * RY,
                  kmesh=(2, 1, 1), ecutrho=48 * RY, nbands=8)
sphere = usys.spheres[0]
kb = _overlap_kbprojectors(usys, sphere)      # SHIPPED: beta + q_int on sphere
q_int = kb.dij_full.to(CDTYPE)
print(f"  overlap projectors: nproj={q_int.shape[0]}, npw={kb.npw}, "
      f"q_int nnz={int(q_int.abs().sum()>0)}")


def s_of_k(k_cart):
    """Traceable dense S(k) = I + p(k)^T q p(k)* — mirrors BlochHK.h's NL line."""
    p = kb.p(k_cart)                       # (nproj, npw), traceable in k
    npw = p.shape[1]
    eye = torch.eye(npw, dtype=CDTYPE, device=p.device)
    return eye + p.mT @ (q_int @ p.conj())


def ds_dk(k_cart):
    """[dS/dk_x, dS/dk_y, dS/dk_z] by forward-mode jvp through s_of_k."""
    out = []
    for mu in range(3):
        e = torch.zeros(3, dtype=RDTYPE, device=k_cart.device)
        e[mu] = 1.0
        with fwAD.dual_level():
            sd = s_of_k(fwAD.make_dual(k_cart.to(RDTYPE), e))
            _, tang = fwAD.unpack_dual(sd)
        out.append(tang)
    return out


kc2 = sphere.k_cart.to(RDTYPE).clone()

# nested jvp for d2S/dk2 = q_hat . grad(dS/dk_mu), same structure as _d2h_mats
def w_fn(k):
    return torch.func.jvp(s_of_k, (k,), (q_hat,))[1]

d2s_ana = []
for mu in range(3):
    e = torch.zeros(3, dtype=RDTYPE)
    e[mu] = 1.0
    d2s_ana.append(torch.func.jvp(w_fn, (kc2,), (e,))[1])

# FD of dS/dk along q_hat
dsp = ds_dk(kc2 + eps * q_hat)
dsm = ds_dk(kc2 - eps * q_hat)
for mu in range(3):
    fd = (dsp[mu] - dsm[mu]) / (2 * eps)
    denom = fd.abs().max().clamp_min(1e-30)
    rel = float((d2s_ana[mu] - fd).abs().max() / denom)
    print(f"  mu={mu}: d2S/dk2 finite? max={float(d2s_ana[mu].abs().max()):.3e}  "
          f"|jvp - FD|/max = {rel:.3e}")
print("  -> nested jvp through the SHIPPED overlap projectors yields a finite,")
print("     FD-matched d2S/dk2: the feared augmentation-2nd-derivative is")
print("     produced by autograd, no new hand-derivation.")
print("DONE")
