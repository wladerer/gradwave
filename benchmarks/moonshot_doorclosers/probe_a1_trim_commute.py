"""Probe A1 — antiunitary (time-reversal) commutation at a general TRIM k.

Door-closer for the "TRIM realification" performance idea (GO-1). At a
time-reversal-invariant momentum (2k ≡ G0 a reciprocal-lattice vector), H_k
should commute with the antiunitary J = P ∘ K, where K is complex conjugation
on the plane-wave coefficients and P is the G-shift permutation induced by
G -> -G - G0 (this is c(-G-2k)=c(G)* generalizing the Gamma constraint
c(-G)=c(G)*).

If J H_k = H_k J to machine precision using the SHIPPED H-apply
(core.batch.BatchedHamiltonian), the complex Hermitian eigenproblem at that k
is the complexification of a real symmetric one and realification is available.
If it does NOT commute to ~1e-12..1e-13, realification is NOT available at
general TRIM in this basis — a decisive KILL.

This is a correctness probe (NOT timing-sensitive). It uses a random REAL
v_eff_r (the commutation only needs V_eff real + TR-symmetric projectors, not a
self-consistent density) so no SCF is run.

Usage: uv run python benchmarks/moonshot_doorclosers/probe_a1_trim_commute.py
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.dtypes import CDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

torch.manual_seed(0)
RY = 13.605693122994


def build_J_permutation(miller: np.ndarray, k_frac: np.ndarray):
    """partner[g] = sphere index of Miller(-m_g - G0), G0 = 2k (integer at TRIM).

    Returns (partner, ok, g0) where ok is False if the sphere is not closed
    under the map (then J is not representable and realification is unavailable).
    """
    g0 = np.rint(2.0 * k_frac).astype(np.int64)
    if not np.allclose(2.0 * k_frac, g0, atol=1e-9):
        return None, False, g0  # not a TRIM: 2k not a reciprocal-lattice vector
    npw = miller.shape[0]
    # map every Miller triple -> sphere index via a dict (sphere injective)
    key = {tuple(int(x) for x in miller[i]): i for i in range(npw)}
    partner = np.full(npw, -1, dtype=np.int64)
    for i in range(npw):
        tgt = (-miller[i] - g0)
        j = key.get((int(tgt[0]), int(tgt[1]), int(tgt[2])), -1)
        partner[i] = j
    ok = bool(np.all(partner >= 0))
    return partner, ok, g0


def check_k(system, veff_r, ik: int, label: str):
    sph = system.spheres[ik]
    miller = sph.miller.cpu().numpy()
    k_frac = np.asarray(sph.k_frac, dtype=np.float64)

    partner, ok, g0 = build_J_permutation(miller, k_frac)
    if partner is None:
        print(f"[{label}] k={k_frac} is NOT a TRIM (2k={2*k_frac}); skip")
        return
    if not ok:
        print(f"[{label}] k={k_frac} G0={g0}: sphere NOT closed under "
              f"G->-G-G0 -> J not representable (KILL for this k)")
        return
    # verify the permutation is an involution composed with conj (J^2 = +1):
    # partner[partner[i]] must be i (K^2=1, P^2=1 since -(-m-G0)-G0=m).
    inv_ok = bool(np.all(partner[partner] == np.arange(len(partner))))

    # single-k Hamiltonian with the shipped apply
    bk1 = system.batch.reindex(torch.tensor([ik]))
    projs = projectors_b(system.batch, system.positions)[ik : ik + 1]
    H = BatchedHamiltonian(bk1, tuple(system.grid.shape), veff_r, projs)

    npw = sph.npw
    partner_t = torch.as_tensor(partner, dtype=torch.long)

    def Jop(c):  # c: (1, nb, npw_max); act on the true sphere slots only
        cc = c.clone()
        real = cc[:, :, :npw]
        real = real.conj()[:, :, partner_t]
        cc[:, :, :npw] = real
        return cc

    nb = 6
    c = torch.zeros(1, nb, bk1.npw_max, dtype=CDTYPE)
    c[:, :, :npw] = torch.randn(1, nb, npw, dtype=CDTYPE)
    # mask padded slots
    c = c * bk1.mask[:, None, :]

    Hc = H.apply(c)
    JHc = Jop(Hc)
    HJc = H.apply(Jop(c))

    num = torch.linalg.norm((JHc - HJc)[:, :, :npw]).item()
    den = torch.linalg.norm(Hc[:, :, :npw]).item()
    rel = num / den if den else float("nan")
    print(f"[{label}] k={k_frac} G0={g0} npw={npw} nproj={projs.shape[1]} "
          f"J^2=+1:{inv_ok}  ||JH-HJ||/||H|| = {rel:.3e}")


def main():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    si = parse_upf(root / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    # 2x2x2 unshifted MP -> k in {0, 1/2} per axis: all 8 points are TRIMs
    system = setup_system(cell, pos, [0, 0], [si], ecut=30 * RY,
                          kmesh=(2, 2, 2), use_symmetry=False)

    n = system.grid.shape
    veff_r = torch.randn(*n, dtype=torch.float64)  # arbitrary REAL local potential

    kfracs = [np.asarray(s.k_frac, dtype=float) for s in system.spheres]
    print(f"nk={len(kfracs)} grid={tuple(n)}")
    for ik, kf in enumerate(kfracs):
        # label by which axes are at the zone boundary
        nz = int(np.sum(np.abs(kf - 0.5) < 1e-9) + np.sum(np.abs(kf + 0.5) < 1e-9))
        label = "GAMMA" if np.allclose(kf, 0.0) else f"TRIM-{nz}bnd"
        check_k(system, veff_r, ik, label)


if __name__ == "__main__":
    main()
