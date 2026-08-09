#!/usr/bin/env python
"""Spin-Batched Davidson A/B benchmark (collinear nspin=2).

For nspin=2 collinear metals the SCF runs TWO serial per-spin Davidson solves
per iteration (scf/loop.py: ``for sp in range(nspin): _solve_bands(...)``). Each
is a ``davidson_batched`` over ``(nk, nb, npw)``. Since only ``v_eff`` is
spin-dependent (kinetic + KB projectors are shared), the two channels can fold
into ONE ``davidson_batched`` over a stacked ``(2·nk, nb, npw)`` block
(``core.batch.SpinBatchedHamiltonian``). Same FLOPs; the win is HALVED per-op
Python/ATen dispatch + kernel-launch overhead.

This script measures whether that win is real:

  STAGE 1  Capture the real per-spin state (v_eff↑/↓, seed coeffs, bk, projectors,
           tol) from one iteration of a live FM SCF, then A/B:
             (a) two serial davidson_batched calls (shipped path)
             (b) one spin-batched davidson_batched over 2·nk
           Assert eigenvalues match per spin block, report Davidson wall (median
           of several reps) and the speedup.

  STAGE 2  Run the full SCF both ways (spin_batch=False vs True) and verify the
           SAME fixed point: |ΔE| and matching n_iter, with full-SCF wall.

Systems (parametrized): FM bcc-Fe nspin=2 (primary) and a smaller/cheaper
magnetic cell. Pass --quick to shrink ecut/kmesh for a fast local signal; the
default parameters are the heavy end-to-end config meant for a workstation
(asus). Runnable via ``uv run python benchmarks/spin_batch/spin_batched_davidson_bench.py``.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gradwave.core.batch import BatchedHamiltonian, SpinBatchedHamiltonian
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.registry import get as get_solver

RY = 13.605693122994

# Fe ONCV pseudo shipped as a QE-comparison fixture (a 5D PSWFC, verified
# magnetic — see memory: scalar-rel-nc-pt / Fe fixtures). Resolve relative to
# the repo root so the script runs from anywhere.
_REPO = Path(__file__).resolve().parents[2]
_FE_UPF = _REPO / "tests" / "fixtures" / "qe" / "pseudos" / "Fe_ONCV_PBE-1.2.upf"


@dataclass
class SysConfig:
    name: str
    cell: np.ndarray
    pos: np.ndarray
    species_index: list[int]
    upf_path: Path
    ecut: float
    kmesh: tuple[int, int, int]
    nbands: int
    start_mag: list[float]
    width: float = 0.1

    def build(self):
        upf = parse_upf(self.upf_path)
        return setup_system(
            self.cell, self.pos, self.species_index, [upf],
            ecut=self.ecut, kmesh=self.kmesh, nbands=self.nbands,
        )


def fe_bcc_primitive(quick: bool) -> SysConfig:
    """FM bcc-Fe, 1-atom primitive cell. Primary heavy system on the default
    (asus) settings; --quick shrinks it to a fast local signal."""
    a = 2.87
    cell = a * 0.5 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    pos = np.array([[0.0, 0.0, 0.0]])
    return SysConfig(
        name="FM bcc-Fe (primitive, 1 atom)",
        cell=cell, pos=pos, species_index=[0], upf_path=_FE_UPF,
        ecut=(30 if quick else 45) * RY,
        kmesh=(4, 4, 4) if quick else (8, 8, 8),
        nbands=12, start_mag=[0.4],
    )


def fe_bcc_conventional(quick: bool) -> SysConfig:
    """FM bcc-Fe, 2-atom conventional cell — the smaller/cheaper magnetic cell."""
    a = 2.87
    cell = a * np.eye(3)
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ cell
    return SysConfig(
        name="FM bcc-Fe (conventional, 2 atoms)",
        cell=cell, pos=pos, species_index=[0, 0], upf_path=_FE_UPF,
        ecut=(25 if quick else 40) * RY,
        kmesh=(2, 2, 2) if quick else (4, 4, 4),
        nbands=20, start_mag=[0.4, 0.4],
    )


# ---------------------------------------------------------------------------
# Capture: stash the real per-spin solve inputs from a live SCF iteration.
# ---------------------------------------------------------------------------


@dataclass
class Captured:
    veff_s: list[torch.Tensor]
    coeffs_b_s: list[torch.Tensor]
    bk: object
    projs_b: torch.Tensor
    grid_shape: tuple[int, int, int]
    tol: float
    cdtype: torch.dtype
    t_solve: torch.Tensor


def capture_state(cfg: SysConfig, target_iter: int = 3) -> Captured:
    """Run a live FM SCF and stash both spins' (v_eff, seed coeffs, bk, ...) from
    ``target_iter`` by monkeypatching ``_solve_bands``, then abort the SCF."""
    from gradwave.scf import loop as loop_mod

    system = cfg.build()
    grid_shape = system.grid.shape
    orig = loop_mod._solve_bands
    box: dict[str, object] = {"calls": 0, "veff": [None, None], "coeffs": [None, None]}

    class _Done(Exception):
        pass

    def spy(veff_sp, coeffs_sp, bk, gshape, projs_b, *a, **kw):
        it0 = box["calls"] // 2  # 0-based iteration
        sp = box["calls"] % 2
        # a = (hub, hub_q, n_hub_sp, hub_alpha, fock_sp, mgga_sp, eigensolver,
        #      tol_eff, use_low, cdtype, t_solve, device, u_scale)
        if it0 == target_iter:
            box["veff"][sp] = veff_sp.clone()
            box["coeffs"][sp] = coeffs_sp.clone()
            box["bk"] = bk
            box["projs_b"] = projs_b
            box["tol"] = a[7]
            box["cdtype"] = a[9]
            box["t_solve"] = a[10]
        box["calls"] += 1
        out = orig(veff_sp, coeffs_sp, bk, gshape, projs_b, *a, **kw)
        if it0 == target_iter and sp == 1:
            raise _Done
        return out

    loop_mod._solve_bands = spy
    try:
        scf(system, SpinPBE(), smearing="gaussian", width=cfg.width, nspin=2,
            start_mag=cfg.start_mag, max_iter=target_iter + 2,
            etol=1e-9, rhotol=1e-8, verbose=False)
    except _Done:
        pass
    finally:
        loop_mod._solve_bands = orig

    return Captured(
        veff_s=[box["veff"][0], box["veff"][1]],
        coeffs_b_s=[box["coeffs"][0], box["coeffs"][1]],
        bk=box["bk"], projs_b=box["projs_b"], grid_shape=grid_shape,
        tol=float(box["tol"]), cdtype=box["cdtype"], t_solve=box["t_solve"],
    )


# ---------------------------------------------------------------------------
# Stage 1: A/B the Davidson solve on the captured state.
# ---------------------------------------------------------------------------


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_time(fn, reps: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def stage1(cap: Captured, reps: int) -> dict[str, float]:
    davidson = get_solver("davidson")
    bk = cap.bk
    nk = bk.nk
    c_up = cap.coeffs_b_s[0].to(cap.cdtype)
    c_dn = cap.coeffs_b_s[1].to(cap.cdtype)
    nb = c_up.shape[1]

    def serial():
        Hu = BatchedHamiltonian(bk, cap.grid_shape, cap.veff_s[0], cap.projs_b)
        Hd = BatchedHamiltonian(bk, cap.grid_shape, cap.veff_s[1], cap.projs_b)
        ru = davidson(Hu.apply, c_up, cap.t_solve, bk.mask, tol=cap.tol, nbands=nb)
        rd = davidson(Hd.apply, c_dn, cap.t_solve, bk.mask, tol=cap.tol, nbands=nb)
        return ru, rd

    t2 = torch.cat([cap.t_solve, cap.t_solve], dim=0)
    mask2 = torch.cat([bk.mask, bk.mask], dim=0)
    c0 = torch.cat([c_up, c_dn], dim=0)

    def spin_batched():
        Hs = SpinBatchedHamiltonian(bk, cap.grid_shape, cap.veff_s[0], cap.veff_s[1], cap.projs_b)
        return davidson(Hs.apply, c0, t2, mask2, tol=cap.tol, nbands=nb)

    # correctness: eigenvalues per spin block must match the serial solve
    (ru, rd) = serial()
    rs = spin_batched()
    eu_ref, ed_ref = ru.eigenvalues, rd.eigenvalues
    eu, ed = rs.eigenvalues[:nk], rs.eigenvalues[nk:]
    d_up = float((eu - eu_ref).abs().max())
    d_dn = float((ed - ed_ref).abs().max())

    t_serial = _median_time(serial, reps)
    t_spin = _median_time(spin_batched, reps)
    return {
        "nk": nk, "nb": nb, "npw_max": bk.npw_max,
        "t_serial": t_serial, "t_spin": t_spin,
        "speedup": t_serial / t_spin if t_spin > 0 else float("nan"),
        "eig_dmax_up": d_up, "eig_dmax_dn": d_dn,
        "n_iter_serial": max(ru.n_iter, rd.n_iter), "n_iter_spin": rs.n_iter,
    }


# ---------------------------------------------------------------------------
# Stage 2: full-SCF fixed-point verification.
# ---------------------------------------------------------------------------


def stage2(cfg: SysConfig) -> dict[str, float]:
    def run(spin_batch: bool):
        system = cfg.build()
        t0 = time.perf_counter()
        r = scf(system, SpinPBE(), smearing="gaussian", width=cfg.width, nspin=2,
                start_mag=cfg.start_mag, etol=1e-9, rhotol=1e-8, verbose=False,
                spin_batch=spin_batch)
        _sync()
        return r, time.perf_counter() - t0

    r_ser, w_ser = run(False)
    r_spin, w_spin = run(True)
    e_ser = float(r_ser.energies.free_energy)
    e_spin = float(r_spin.energies.free_energy)
    return {
        "e_serial": e_ser, "e_spin": e_spin, "dE": abs(e_ser - e_spin),
        "n_iter_serial": r_ser.n_iter, "n_iter_spin": r_spin.n_iter,
        "w_serial": w_ser, "w_spin": w_spin,
        "mag": r_ser.mag_total, "conv_serial": r_ser.converged,
        "conv_spin": r_spin.converged,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="shrink ecut/kmesh for a fast local signal")
    ap.add_argument("--reps", type=int, default=5, help="Stage-1 timing reps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-stage2", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# spin-batched Davidson bench | device={dev} | threads={args.threads} "
          f"| quick={args.quick} | reps={args.reps}", flush=True)

    systems = [fe_bcc_conventional(args.quick), fe_bcc_primitive(args.quick)]

    print("\n=== STAGE 1: Davidson A/B (captured single-iteration state) ===", flush=True)
    print(f"{'system':<34} {'nk':>3} {'nb':>3} {'npw':>5} "
          f"{'serial[s]':>10} {'spin[s]':>10} {'speedup':>8} {'eigΔmax':>10}", flush=True)
    for cfg in systems:
        cap = capture_state(cfg)
        s1 = stage1(cap, args.reps)
        eig_dmax = max(s1["eig_dmax_up"], s1["eig_dmax_dn"])
        ok = "OK" if eig_dmax < 1e-6 else "!!"
        print(f"{cfg.name:<34} {s1['nk']:>3d} {s1['nb']:>3d} {s1['npw_max']:>5d} "
              f"{s1['t_serial']:>10.4f} {s1['t_spin']:>10.4f} {s1['speedup']:>7.3f}x "
              f"{eig_dmax:>9.1e} {ok}", flush=True)

    if not args.skip_stage2:
        print("\n=== STAGE 2: full-SCF fixed point (serial vs spin_batch) ===", flush=True)
        print(f"{'system':<34} {'iters s/sb':>11} {'E_serial[eV]':>16} "
              f"{'dE[eV]':>10} {'wall s/sb[s]':>16} {'mag':>6}", flush=True)
        for cfg in systems:
            s2 = stage2(cfg)
            iters = f"{s2['n_iter_serial']}/{s2['n_iter_spin']}"
            wall = f"{s2['w_serial']:.1f}/{s2['w_spin']:.1f}"
            # same fixed point: energy matches to ~machine precision; the SCF
            # iteration count may differ by ±1 (uniform vs per-spin Davidson
            # expansion count — see the module docstring / integration test).
            same = "OK" if (s2["dE"] < 1e-7 and abs(
                s2["n_iter_serial"] - s2["n_iter_spin"]) <= 1) else "!!"
            print(f"{cfg.name:<34} {iters:>11} {s2['e_serial']:>16.6f} "
                  f"{s2['dE']:>10.1e} {wall:>16} {s2['mag']:>6.2f} {same}", flush=True)

    print("\nEXIT=0", flush=True)


if __name__ == "__main__":
    main()
