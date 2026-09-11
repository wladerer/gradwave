"""Validation anchor: the magnon spin-wave stiffness D of bcc Fe, from
gradwave-extracted Heisenberg exchange fed through linear spin-wave theory
(postscf/magnons.py).

Pipeline
--------
1. Build ferromagnetic bcc Fe as a supercell of the 1-atom primitive cell so
   that the central atom's first neighbour shells are *distinct* atoms (the
   2-atom cell of examples/fe_exchange.py folds every shell into one
   inter-sublattice sum — see the spin_exchange module docstring). A
   ``--nrep n`` supercell of the primitive rhombohedral cell holds shells out to
   the box's minimum-image radius.
2. Extract the exchange to every other atom in one shot with
   ``exchange_from_atom`` (tilt the central atom, read the induced transverse
   torque on all others), and bin the isotropic part by neighbour distance into
   per-shell J_n (curvature convention: eV per unit-moment pair).
3. Build a :class:`HeisenbergModel` on the bcc primitive cell with those shells
   and read the ferromagnetic stiffness D from the small-q fit, cumulatively
   (J1, J1+J2, J1+J2+J3) to show shell convergence.

The magnon stiffness is reported in the standard adiabatic / frozen-magnon
convention ħω(q) = (4/M) [J(0) − J(q)] used to compare DFT exchange to the
measured stiffness (M the moment in μ_B); with S = M/2 this is exactly
2·(the LSWT ω of the S-length model), which the script also prints so the
convention is explicit. Experiment: D ≈ 280–310 meV·Å² for bcc Fe. PBE/LSDA
exchange overestimates somewhat, so D coming out at the right *order* and
*converging* with shells is the gate — not exact agreement.

Heavy (constrained non-collinear SCFs on a supercell). Run on a many-core box:
    PYTHONPATH=src python examples/fe_magnon_stiffness.py --nrep 2 --kmesh 3
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.magnons import ExchangeBond, HeisenbergModel, magnon_dispersion
from gradwave.postscf.spin_exchange import decompose, exchange_from_atom
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"
A_BCC = 2.87  # Å, bcc Fe lattice constant
M_MOMENT = 2.222  # μ_B ferromagnetic moment (matches examples/fe_exchange.py)


def _bcc_primitive(a: float) -> np.ndarray:
    """Primitive rhombohedral cell of bcc (rows = lattice vectors)."""
    return 0.5 * a * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)


def _supercell(prim: np.ndarray, nrep: int):
    """(cell, cart positions) of an nrep³ supercell of a 1-atom primitive cell."""
    cell = nrep * prim
    idx = range(nrep)
    frac = np.array([[i, j, k] for i in idx for j in idx for k in idx], dtype=float)
    pos = (frac / nrep) @ cell
    return cell, pos


def _shell_distances(cell: np.ndarray, pos: np.ndarray, center: int):
    """Minimum-image distance from ``center`` to every other atom (Å)."""
    inv = np.linalg.inv(cell)
    d = pos - pos[center]
    f = d @ inv
    f -= np.round(f)  # minimum image
    cart = f @ cell
    return np.linalg.norm(cart, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=2, help="primitive-cell repeats")
    ap.add_argument("--kmesh", type=int, default=3)
    ap.add_argument("--ecut_ry", type=float, default=50.0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = "cpu"  # fp64 SCF: CPU beats the fp64-crippled 3050

    prim = _bcc_primitive(A_BCC)
    cell, pos = _supercell(prim, args.nrep)
    natom = len(pos)
    center = int(np.argmin(np.linalg.norm(pos - pos.mean(0), axis=1)))
    dists = _shell_distances(cell, pos, center)

    # theoretical bcc shell radii (Å): 1nn √3/2 a, 2nn a, 3nn √2 a
    shell_r = {1: np.sqrt(3) / 2 * A_BCC, 2: A_BCC, 3: np.sqrt(2) * A_BCC}
    print(f"supercell {args.nrep}^3 = {natom} atoms; center atom {center}")
    for n, r in shell_r.items():
        cnt = int(np.sum(np.abs(dists - r) < 0.05))
        print(f"  shell {n}: r={r:.3f} Å, resolved neighbours in cell = {cnt}")

    fe = parse_upf(f"{PSE}/Fe_ONCV_PBE-1.2.upf")
    species = [0] * natom
    system = setup_system(cell, pos, species, [fe] * natom, ecut=args.ecut_ry * RY,
                          kmesh=(args.kmesh,) * 3, nbands=int(natom * 12),
                          time_reversal=False)
    xc = NoncollinearXC(LSDA_PW92())
    m0 = torch.full((natom,), M_MOMENT, dtype=torch.float64)

    print("running constrained-moment exchange extraction (3 SCFs)...", flush=True)
    tensors, _ = exchange_from_atom(
        system, xc, j=center, m0=m0, ref_dir=(0, 0, 1), delta=0.08, lam=8.0,
        smearing="gaussian", width=0.1, etol=1e-7, rhotol=1e-6, max_iter=200,
        mixing_alpha=0.4, verbose=False)

    # bin the isotropic exchange by shell distance (curvature, eV per pair)
    per_shell: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for i, J in tensors.items():
        j_iso = decompose(J)[0]
        for n, r in shell_r.items():
            if abs(dists[i] - r) < 0.05:
                per_shell[n].append(j_iso)
    j_shell = {n: (float(np.mean(v)) if v else 0.0) for n, v in per_shell.items()}
    z_bcc = {1: 8, 2: 6, 3: 12}
    print("per-shell exchange (curvature, eV per unit-moment pair):")
    for n in (1, 2, 3):
        got = len(per_shell[n])
        print(f"  J{n} = {j_shell[n] * 1000:+.2f} meV   "
              f"(z={z_bcc[n]}, resolved {got} bonds)")

    # LSWT stiffness on the bcc primitive cell, cumulative over shells.
    # Model convention H = -1/2 Σ J_model S_i·S_j; the extracted curvature is
    # J_curv = J_model·S² (unit-moment pairs), so J_model = J_curv / S².
    s = M_MOMENT / 2.0
    # bcc-primitive neighbour lattice vectors per shell, generated by distance
    # (integer combos of the primitive rows, selected by Cartesian length —
    # correct by construction rather than hand-derived).
    shell_rs: dict[int, list[tuple[int, int, int]]] = {1: [], 2: [], 3: []}
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                if (n1, n2, n3) == (0, 0, 0):
                    continue
                rvec = np.array([n1, n2, n3], dtype=float) @ prim
                dist = float(np.linalg.norm(rvec))
                for n, r in shell_r.items():
                    if abs(dist - r) < 0.05:
                        shell_rs[n].append((n1, n2, n3))
    for n in (1, 2, 3):
        assert len(shell_rs[n]) == z_bcc[n], \
            f"shell {n}: generated {len(shell_rs[n])} vecs, expected {z_bcc[n]}"
    print("\ncumulative magnon stiffness D (frozen-magnon 4/M convention):")
    for upto in (1, 2, 3):
        bonds = []
        for n in range(1, upto + 1):
            jm = j_shell[n] / s**2  # model coupling (eV)
            for r in shell_rs[n]:
                bonds.append(ExchangeBond(0, 0, r, jm))
        model = HeisenbergModel(cell=prim, spins=[s], bonds=bonds)
        # small-q stiffness along x, in the frozen-magnon convention (= 2× the
        # S-length LSWT ω, since M = 2S): fit ω[meV] vs q²[Å⁻²].
        recip = 2 * np.pi * np.linalg.inv(prim).T
        qmax = 0.02 * float(np.linalg.norm(recip[0]))
        qs = np.linspace(0, qmax, 13)[1:]
        inv_recip = np.linalg.inv(recip)
        w = []
        for qm in qs:
            qf = (qm * np.array([1.0, 0, 0])) @ inv_recip
            w.append(magnon_dispersion(model, qf[None])[0, 0] * 1e3)  # meV (S-model)
        w = np.asarray(w) * 2.0  # -> frozen-magnon (4/M) convention
        d_stiff = float((qs**2 @ w) / (qs**2 @ qs**2))
        tag = "+".join(f"J{k}" for k in range(1, upto + 1))
        print(f"  {tag:<10s} D = {d_stiff:7.1f} meV·Å²")
    print("  (experiment: D ≈ 280–310 meV·Å²)")


if __name__ == "__main__":
    main()
