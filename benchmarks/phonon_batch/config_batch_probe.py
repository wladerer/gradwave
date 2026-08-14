"""Proof-of-mechanism: does folding phonon-displacement CONFIGS into the batch
dimension speed up the eigensolve — the SCF's dominant cost?

A phonon force-constant campaign runs N independent displacement SCFs. They share
the IDENTICAL supercell, ecut, k-mesh and therefore G-sphere / kinetic operator /
position-free projector tables; only the atomic positions differ (a structure-factor
phase on the KB projectors, and a per-config v_eff). `core/batch.py` already runs one
SCF's k-points as a single batched `(nk, nb, npw_max)` tensor op. This probe asks the
gating question for CONFIG-batching: fold the N configs into that same leading batch
axis — `(N·nk, nb, npw_max)` — and does one big batched Davidson beat N separate ones?

It does NOT run a full config-batched SCF (that needs per-config v_eff/density/Fermi/
mixing — see docs/design/phonon-config-batching.md). It isolates the ONE lever that
gates the whole build: the batched-eigensolve wall time, N-separate vs 1-folded, with a
correctness check that the folded per-config eigenvalues match the separate ones.

  separate : for each config c → BatchedHamiltonian(bk, v_eff, p_c); davidson over (nk)
  folded   : one BatchedHamiltonian(bk_folded, v_eff, p_folded); davidson over (N·nk)

Shared v_eff across configs is used (the eigensolve FLOP count is identical whether
v_eff is broadcast or per-row, so the timing is faithful; a real SCF would carry a
per-config v_eff, a cheap apply generalization). CPU saturates BLAS already at one
SCF's nk, so the CPU win may be small — the thesis is a GPU-saturation lever, so run
--device cuda too.

Run:
    uv run python benchmarks/phonon_batch/config_batch_probe.py --nrep 2 --n-configs 24
    uv run python benchmarks/phonon_batch/config_batch_probe.py --device cuda --dtype c64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
_PSEUDO_DIR = _ROOT / "tests/fixtures/qe/pseudos"


def _build_reference(nrep: int, ecut_ry: float, kmesh: int, device: str):
    """A Si supercell reference SCF → (system, v_eff, warm x0). Insulator,
    smearing=none: the eigensolve is clean and the warm start mirrors a spoke."""
    from ase import Atoms

    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    a = 5.43
    prim = (a / 2) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    at = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
               cell=prim, pbc=True).repeat((nrep, nrep, nrep))
    upf = parse_upf(str(_PSEUDO_DIR / "Si_ONCV_PBE-1.2.upf"))
    species_of_atom = [0] * len(at)  # integer indices into the upf list
    system = setup_system(np.asarray(at.cell.array), at.get_positions(),
                          species_of_atom, [upf], ecut=ecut_ry * RY,
                          kmesh=(kmesh, kmesh, kmesh), use_symmetry=False)
    if device != "cpu":
        system = system.to(device)
    res = scf(system, PBE(), smearing="none", width=0.0, etol=1e-9, rhotol=1e-8,
              verbose=False)
    return system, res


def _pad_x0(coeffs_per_k, bk, nb, cdtype):
    """Ragged per-k converged coeffs [(nb, npw_k)] → padded (nk, nb, npw_max) warm
    start, masked. Sphere order matches the batch's first-npw slots."""
    nk, m = bk.nk, bk.npw_max
    x0 = torch.zeros(nk, nb, m, dtype=cdtype, device=bk.mask.device)
    for ik in range(nk):
        ck = coeffs_per_k[ik]
        npw_k = ck.shape[1]
        x0[ik, :, :npw_k] = ck[:nb].to(cdtype)
    return x0 * bk.mask[:, None, :]


def _tile_bk(bk, n: int):
    """Fold n identical-geometry configs into the leading batch axis: repeat the
    per-k fields n× → (n·nk, ...). Per-projector fields (atom_index, dij) are shared."""
    import dataclasses

    rep = {f: getattr(bk, f).repeat(n, *[1] * (getattr(bk, f).dim() - 1))
           for f in ("npw", "mask", "flat_idx", "kpg", "t", "proj_phase_free")}
    return dataclasses.replace(bk, **rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=2, help="supercell replication (n^3 cells)")
    ap.add_argument("--ecut", type=float, default=20.0, help="Ry")
    ap.add_argument("--kmesh", type=int, default=2)
    ap.add_argument("--n-configs", type=int, default=24)
    ap.add_argument("--disp", type=float, default=0.02, help="displacement [Å]")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="c128", choices=["c128", "c64"])
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3, help="timed repetitions (min reported)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cdtype = torch.complex128 if a.dtype == "c128" else torch.complex64

    from gradwave.core.batch import BatchedHamiltonian, projectors_b
    from gradwave.solvers.davidson import davidson_batched

    dev = torch.device(a.device)
    system, res = _build_reference(a.nrep, a.ecut, a.kmesh, a.device)
    bk = system.batch
    shape = system.grid.shape
    v_eff = res.v_eff.to(dev)
    nb = res.eigenvalues.shape[1]
    x0 = _pad_x0(res.coeffs, bk, nb, cdtype)
    pos0 = system.positions  # (na, 3) [Å]
    na = pos0.shape[0]

    # N displacement configs: move a rotating atom/axis by ±disp (the real FD set)
    rng_pairs = [(i % na, (i // na) % 3, 1 if (i // (na * 3)) % 2 == 0 else -1)
                 for i in range(a.n_configs)]
    configs = []
    for (atom, axis, sgn) in rng_pairs:
        p = pos0.clone()
        p[atom, axis] += sgn * a.disp
        configs.append(p)

    def _solve(bk_, v, p, x):
        h = BatchedHamiltonian(bk_, shape, v, p)
        return davidson_batched(h.apply, x, bk_.t, bk_.mask, tol=a.tol, max_iter=60)

    def _sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # correctness: folded per-config eigenvalues must equal the separate solve's
    p_list = [projectors_b(bk, p).to(cdtype) for p in configs]
    ev_sep0 = _solve(bk, v_eff, p_list[0], x0).eigenvalues
    bk_f = _tile_bk(bk, a.n_configs)
    p_fold = torch.cat(p_list, dim=0)
    x0_fold = x0.repeat(a.n_configs, 1, 1)
    ev_fold = _solve(bk_f, v_eff, p_fold, x0_fold).eigenvalues
    ev_fold0 = ev_fold[: bk.nk]
    max_dev = float((ev_sep0 - ev_fold0).abs().max())

    # timings (min of reps)
    def _time_separate():
        _sync()
        t0 = time.time()
        for p in p_list:
            _solve(bk, v_eff, p, x0)
        _sync()
        return time.time() - t0

    def _time_folded():
        _sync()
        t0 = time.time()
        _solve(bk_f, v_eff, p_fold, x0_fold)
        _sync()
        return time.time() - t0

    _time_separate()  # warm caches
    _time_folded()
    t_sep = min(_time_separate() for _ in range(a.reps))
    t_fold = min(_time_folded() for _ in range(a.reps))

    print(f"# phonon config-batch probe  Si {a.nrep}^3 supercell  na={na}  "
          f"ecut={a.ecut}Ry  kmesh={a.kmesh}^3", flush=True)
    print(f"#   nk={bk.nk}  npw_max={bk.npw_max}  nb={nb}  n_configs={a.n_configs}  "
          f"folded_batch={bk_f.nk}  device={a.device}  dtype={a.dtype}  threads={a.threads}",
          flush=True)
    print(f"#   correctness: |Δeig| config0 sep-vs-folded = {max_dev:.2e} eV "
          f"({'OK' if max_dev < 1e-6 else 'MISMATCH'})", flush=True)
    print(f"  separate ({a.n_configs} solves) : {t_sep:8.3f} s", flush=True)
    print(f"  folded   (1 solve, {bk_f.nk:>4} batch): {t_fold:8.3f} s", flush=True)
    print(f"  SPEEDUP (separate/folded)      : {t_sep / max(t_fold, 1e-9):6.2f}x", flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
