"""Learned multi-pole Kerker preconditioner: iterations-to-convergence.

Run one or all cases (``uv run python benchmarks/bench_learned_precond.py
[synthetic|al|cu|cu3al|fe|pt|both]``, default runs synthetic+al+cu+fe+pt+cu3al):

- synthetic — a diagonal response with two length scales, where a single Kerker
  pole is provably the wrong shape. Isolates the mechanism from any DFT cost.
- al / cu — the full probe→fit→deploy loop on a real solver. A short plain-mixing
  PROBE captures the residual history through `mixer_hook`, `response_from_
  residuals` estimates d(G), `fit_multipole` fits the poles (DIIS-aware), and the
  filter deploys as `scf(..., precond_op=...)`. Al ties (single-scale charge); Cu
  wins (3s3p semicore → multi-scale charge).
- cu3al — L1₂ Cu₃Al intermetallic (two chemical species → two screening scales).
  Extends the Cu win to a two-component cell and probes whether a wider pole-seed
  range (4 poles over [0.05, 4.0] Å⁻¹) beats the default 3-pole [0.3, 3.0] fit.
- fe — bcc Fe, collinear ferromagnet (nspin=2). Charge-channel filter on the total
  block. The FM convergence bottleneck is the MAGNETIZATION channel, not charge, so
  this is expected to tie-or-lose; it maps the limit, and motivates a separate mag-
  channel operator (see docs/ideas.md).
- pt — fcc Pt, nonmagnetic + spin-orbit (`scf_noncollinear`). Exercises the
  precond_op wiring on the noncollinear driver; Pt's charge response is single-
  scale so it ties, at a fixed point identical to Kerker's.

Iteration count, energy-gated, is the trustworthy metric for a solver-logic
question (docs/manual/performance.md, bench_precond.py). The filter's demonstrated
headroom is systems whose CHARGE response carries more than one G-space scale; it
does not by construction address a magnetization-channel bottleneck.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradwave.dtypes import RDTYPE  # noqa: E402
from gradwave.scf.learned_precond import (  # noqa: E402
    BlockPrecond,
    MultipoleKerkerPrecond,
    _inv_softplus,
    fit_multipole,
    response_from_residuals,
    spectral_radius,
)

RY = 13.605693122994
FIX = ROOT / "tests/fixtures/qe/pseudos"
torch.set_num_threads(8)


def iters_to_tol(rho: float, tol: float = 1e-8) -> float:
    """Iterations for the residual to fall by `tol` at asymptotic rate rho."""
    import math
    return math.inf if rho >= 1.0 else math.log(tol) / math.log(rho)


def run_synthetic():
    print("\n=== synthetic two-scale response ===")
    alpha = 0.7
    # finite-cell |G|² range (~16 A box → smallest |G| ~0.39 A⁻¹); two response
    # length scales 1/q1, 1/q2 that no single Kerker pole can screen together.
    g2 = torch.linspace(0.15, 40.0, 400, dtype=RDTYPE)
    q1, q2 = 0.3, 2.5
    d = 0.5 * g2 / (g2 + q1**2) + 0.5 * g2 / (g2 + q2**2)

    q_grid = torch.linspace(0.2, 4.0, 120)
    rho1, q0_best = min(
        (float(spectral_radius(g2 / (g2 + q0**2), d, alpha)), float(q0))
        for q0 in q_grid.tolist()
    )
    P, info = fit_multipole(g2, d, n_poles=3, alpha=alpha, n_unroll=40, steps=600)
    rho3 = info["rho_final"]

    print(f"  best single-pole Kerker  q0={q0_best:.2f} A⁻¹   "
          f"rho={rho1:.4f}   ~{iters_to_tol(rho1):.0f} iters")
    print(f"  learned 3-pole           rho={rho3:.4f}   "
          f"~{iters_to_tol(rho3):.0f} iters")
    print(f"  {P.summary()}")
    print(f"  speedup (iteration ratio): {iters_to_tol(rho1)/iters_to_tol(rho3):.2f}x")


METALS = {
    # label: (upf, lattice a [Å], ecut [Ry], nbands, kmesh) — fcc metals
    "al": ("Al_ONCV_PBE-1.2.upf", 4.05, 30, 12, (6, 6, 6)),
    "cu": ("Cu_ONCV_PBE-1.2.upf", 3.615, 45, 20, (6, 6, 6)),  # 3s3p semicore d-band
}


def _fcc_system(upf_name, a, ecut_ry, nbands, kmesh):
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system
    pp = parse_upf(FIX / upf_name)
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return setup_system(cell, np.array([[0.0, 0, 0]]), [0], [pp],
                        ecut=ecut_ry * RY, kmesh=kmesh, nbands=nbands,
                        use_symmetry=True)


def run_metal(label, q0=1.1):
    upf, a, ecut_ry, nbands, kmesh = METALS[label]
    kx = "x".join(map(str, kmesh))
    print(f"\n=== fcc {label} ({upf}, gaussian 0.1 eV, {ecut_ry} Ry, {kx}) ===")
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.loop import scf
    xc = PBE()
    common = dict(smearing="gaussian", width=0.1, nspin=1, verbose=False,
                  rhotol=1e-6, etol=1e-8)

    def sys_():
        return _fcc_system(upf, a, ecut_ry, nbands, kmesh)

    grid = sys_().grid
    g2_dens = grid.g2.reshape(-1)[grid.dens_mask.reshape(-1)].to(RDTYPE)

    # reference: bare Kerker
    t = time.perf_counter()
    ref = scf(sys_(), xc, mixing_alpha=0.7, **common)
    f_ref = float(ref.energies.free_energy)
    dt = time.perf_counter() - t
    print(f"  kerker            {ref.n_iter:3d} iters   F={f_ref:+.6f} eV   {dt:.1f}s")

    # probe: Kerker-on plain damping (history=1, no DIIS) — stable on d-band
    # metals where plain damping alone sloshes. Divide the Kerker factor back out
    # of the residual ratios to recover the bare response d(G).
    res_hist: list[torch.Tensor] = []

    def hook(it, rho_in, rho_out):
        res_hist.append((rho_out - rho_in).detach().clone())

    alpha_probe = 0.7
    scf(sys_(), xc, mixing_alpha=alpha_probe, mixing_history=1, kerker=True,
        max_iter=14, mixer_hook=hook, **common)
    kfac = g2_dens / (g2_dens + q0**2)
    g2_shell, d_shell, count = response_from_residuals(
        res_hist, g2_dens, alpha_probe, n_bins=48, skip=2, precond_fac=kfac)
    print(f"  probe {len(res_hist)} residuals → d(G) over {len(d_shell)} shells, "
          f"d in [{float(d_shell.min()):.2f}, {float(d_shell.max()):.2f}]")

    # fit through the ACTUAL DIIS mixer the deploy run uses (history 8), so the
    # filter complements DIIS rather than duplicating its low-G work.
    P, info = fit_multipole(g2_shell, d_shell, n_poles=3, alpha=0.7,
                            mixer="diis", history=8, q0=q0, n_unroll=30,
                            steps=700, weight=count)
    P = P.rebind(g2_dens).detach_()
    print(f"  fitted {P.summary()}  (plain rho {info['rho_init']:.3f}→{info['rho_final']:.3f})")

    t = time.perf_counter()
    learned = scf(sys_(), xc, mixing_alpha=0.7, precond_op=P, **common)
    f_l = float(learned.energies.free_energy)
    dt = time.perf_counter() - t
    verdict = ("win" if learned.n_iter < ref.n_iter else
               "tie" if learned.n_iter == ref.n_iter else "loss")
    print(f"  learned filter    {learned.n_iter:3d} iters   F={f_l:+.6f} eV   "
          f"{dt:.1f}s   [{verdict}]")
    print(f"  dF vs kerker = {f_l-f_ref:+.2e} eV  (same fixed point; only the path differs)")


CU3AL_CELL = 3.70 * np.eye(3)
CU3AL_FRAC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def run_cu3al():
    """L1₂ Cu₃Al intermetallic — a genuinely two-scale CHARGE response.

    Two chemical species screen at different lengths (Cu's 3s3p semicore d-band
    is a short scale, Al's spread valence a longer one), so a single Kerker pole
    cannot match both — the multi-scale regime the docs name as the next frontier
    after the fcc Cu win (docs/ideas.md, "more multi-scale systems ... Cu₃Al").
    Same cell/pseudos as the QE-validated tests/integration/test_metal_forces_vs_qe
    :test_cu3al_vs_qe, so the fixed point is a known-correct reference.

    Compares three preconditioners at a fixed point identical to a few 1e-12 eV:
      - bare Kerker (single pole q0=1.1);
      - the DEFAULT DIIS-aware 3-pole fit over the [0.3, 3.0] Å⁻¹ seed range;
      - a WIDER 4-pole fit over [0.05, 4.0] Å⁻¹. Cu's winning fit already placed a
        pole at q≈0.07 (below the default 0.3 seed), so a two-species cell whose
        response spans an even wider band is the case where the seed range matters:
        the wider fit can seed a long-wavelength pole directly instead of relying
        on the optimizer to walk one down four-fold in q.
    """
    print("\n=== L1₂ Cu₃Al (Al+3Cu, gaussian 0.1 eV, 40 Ry, 2x2x2) ===")
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf
    al = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")
    cu = parse_upf(FIX / "Cu_ONCV_PBE-1.2.upf")
    xc = PBE()
    q0 = 1.1
    common = dict(smearing="gaussian", width=0.1, nspin=1, verbose=False,
                  rhotol=1e-6, etol=1e-8)

    def sys_():
        from gradwave.scf.loop import setup_system
        return setup_system(CU3AL_CELL, CU3AL_FRAC @ CU3AL_CELL, [0, 1, 1, 1],
                            [al, cu], ecut=40 * RY, kmesh=(2, 2, 2), nbands=45,
                            use_symmetry=True)

    grid = sys_().grid
    g2_dens = grid.g2.reshape(-1)[grid.dens_mask.reshape(-1)].to(RDTYPE)

    # reference: bare Kerker
    t = time.perf_counter()
    ref = scf(sys_(), xc, mixing_alpha=0.7, **common)
    f_ref = float(ref.energies.free_energy)
    print(f"  kerker                {ref.n_iter:3d} iters   F={f_ref:+.6f} eV   "
          f"{time.perf_counter()-t:.1f}s")

    # probe: Kerker-on plain damping (history=1) — stable on the d-band metal.
    res_hist: list[torch.Tensor] = []

    def hook(it, rho_in, rho_out):
        res_hist.append((rho_out - rho_in).detach().clone())

    alpha_probe = 0.7
    scf(sys_(), xc, mixing_alpha=alpha_probe, mixing_history=1, kerker=True,
        max_iter=14, mixer_hook=hook, **common)
    kfac = g2_dens / (g2_dens + q0**2)
    g2_shell, d_shell, count = response_from_residuals(
        res_hist, g2_dens, alpha_probe, n_bins=48, skip=2, precond_fac=kfac)
    print(f"  probe {len(res_hist)} residuals → d(G) over {len(d_shell)} shells, "
          f"d in [{float(d_shell.min()):.2f}, {float(d_shell.max()):.2f}]")

    # two DIIS-aware fits: the default seed range, and a wider/more-poled seed.
    variants = [
        ("learned 3-pole [0.3,3.0]", dict(n_poles=3, q_min=0.3, q_max=3.0)),
        ("learned 4-pole [0.05,4.0]", dict(n_poles=4, q_min=0.05, q_max=4.0)),
    ]
    for label, fit_kw in variants:
        P, info = fit_multipole(g2_shell, d_shell, alpha=0.7, mixer="diis",
                                history=8, q0=q0, n_unroll=30, steps=700,
                                weight=count, **fit_kw)
        P = P.rebind(g2_dens).detach_()
        t = time.perf_counter()
        learned = scf(sys_(), xc, mixing_alpha=0.7, precond_op=P, **common)
        f_l = float(learned.energies.free_energy)
        v = _verdict(learned.n_iter, ref.n_iter)
        print(f"  {label:26s} {learned.n_iter:3d} iters   F={f_l:+.6f} eV   "
              f"{time.perf_counter()-t:.1f}s   [{v}]")
        print(f"    {P.summary()}  (plain rho {info['rho_init']:.3f}"
              f"→{info['rho_final']:.3f})  dF={f_l-f_ref:+.2e} eV")


def _fit_charge_filter(res_total, g2_dens, q0=1.1, alpha_probe=0.7):
    """Probe residuals (density-total block) → estimated d(G) → DIIS-aware fit →
    deployable filter on the density sphere. Shared by every metal runner."""
    kfac = g2_dens / (g2_dens + q0**2)
    g2_shell, d_shell, count = response_from_residuals(
        res_total, g2_dens, alpha_probe, n_bins=48, skip=2, precond_fac=kfac)
    print(f"  probe {len(res_total)} residuals → d(G) over {len(d_shell)} shells, "
          f"d in [{float(d_shell.min()):.2f}, {float(d_shell.max()):.2f}]")
    P, _ = fit_multipole(g2_shell, d_shell, n_poles=3, alpha=0.7, mixer="diis",
                         history=8, q0=q0, n_unroll=30, steps=700, weight=count)
    P = P.rebind(g2_dens).detach_()
    print(f"  fitted {P.summary()}")
    return P


def _verdict(learned_it, ref_it):
    return ("win" if learned_it < ref_it else
            "tie" if learned_it == ref_it else "loss")


def run_fe():
    """bcc Fe, collinear ferromagnet (nspin=2). The hard convergence here is the
    MAGNETIZATION channel (Stoner mode); this tests only the charge-channel filter
    on the total block, where Fe's 3s3p semicore gives a multi-scale response."""
    print("\n=== bcc Fe (nspin=2 FM, johnson, 45 Ry, 6x6x6) ===")
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system
    a = 2.87
    fe = parse_upf(FIX / "Fe_ONCV_PBE-1.2.upf")
    cell = a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])

    def sys_():
        return setup_system(cell, np.zeros((1, 3)), [0], [fe], ecut=45 * RY,
                            kmesh=(6, 6, 6), nbands=12, use_symmetry=True)

    common = dict(smearing="gaussian", width=0.1, nspin=2, start_mag=[0.4],
                  verbose=False, rhotol=1e-5, etol=1e-8)
    grid = sys_().grid
    ng = int(grid.dens_mask.sum())
    g2_dens = grid.g2.reshape(-1)[grid.dens_mask.reshape(-1)].to(RDTYPE)

    t = time.perf_counter()
    ref = scf(sys_(), SpinPBE(), mixing_alpha=0.7, mixing_scheme="johnson", **common)
    dt = time.perf_counter() - t
    print(f"  kerker            {ref.n_iter:3d} iters   m={ref.mag_total:+.3f} muB   {dt:.1f}s")

    res_hist: list[torch.Tensor] = []

    def hook(it, vin, vout):
        res_hist.append((vout[:ng] - vin[:ng]).detach().clone())  # charge block

    scf(sys_(), SpinPBE(), mixing_alpha=0.5, mixing_history=1, kerker=True,
        max_iter=12, mixer_hook=hook, **common)
    P = _fit_charge_filter([r for r in res_hist if len(r) == ng], g2_dens)

    t = time.perf_counter()
    learned = scf(sys_(), SpinPBE(), mixing_alpha=0.7, mixing_scheme="johnson",
                  precond_op=P, **common)
    dm = learned.mag_total - ref.mag_total
    v = _verdict(learned.n_iter, ref.n_iter)
    print(f"  learned filter    {learned.n_iter:3d} iters   m={learned.mag_total:+.3f} muB   "
          f"{time.perf_counter()-t:.1f}s   [{v}]   dm={dm:+.1e}")


def run_pt_soc():
    """fcc Pt, nonmagnetic + spin-orbit (scf_noncollinear, m⃗≡0). No spin problem,
    so the charge channel is the whole convergence story — a 5d + 5s5p-semicore
    multi-scale response, the SOC analog of the Cu win, and it exercises the new
    precond_op wiring on the noncollinear driver."""
    print("\n=== fcc Pt (nonmagnetic + SOC, 45 Ry, 4x4x4) ===")
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system
    from gradwave.scf.noncollinear import scf_noncollinear
    a = 3.92
    pt = parse_upf(FIX / "Pt_ONCV_PBE_FR-1.0.upf")
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])

    def sys_():
        return setup_system(cell, np.array([[0.0, 0, 0]]), [0], [pt],
                            ecut=45 * RY, kmesh=(4, 4, 4), nbands=16,
                            use_symmetry=True)

    xc = NoncollinearXC(SpinPBE())
    common = dict(smearing="gaussian", width=0.1, nonmagnetic=True,
                  mag_vec_init=[[0, 0, 0]], verbose=False, rhotol=1e-6, etol=1e-8)
    grid = sys_().grid
    ng = int(grid.dens_mask.sum())
    g2_dens = grid.g2.reshape(-1)[grid.dens_mask.reshape(-1)].to(RDTYPE)

    t = time.perf_counter()
    ref = scf_noncollinear(sys_(), xc, mixing_alpha=0.7, **common)
    f_ref = float(ref.energies.free_energy)
    dt = time.perf_counter() - t
    print(f"  kerker            {ref.n_iter:3d} iters   F={f_ref:+.4f} eV   {dt:.1f}s")

    res_hist: list[torch.Tensor] = []

    def hook(it, vin, vout):
        res_hist.append((vout[:ng] - vin[:ng]).detach().clone())

    scf_noncollinear(sys_(), xc, mixing_alpha=0.5, mixing_history=1,
                     adaptive=False, max_iter=12, mixer_hook=hook, **common)
    P = _fit_charge_filter([r for r in res_hist if len(r) == ng], g2_dens)

    t = time.perf_counter()
    learned = scf_noncollinear(sys_(), xc, mixing_alpha=0.7, precond_op=P, **common)
    f_l = float(learned.energies.free_energy)
    v = _verdict(learned.n_iter, ref.n_iter)
    print(f"  learned filter    {learned.n_iter:3d} iters   F={f_l:+.4f} eV   "
          f"{time.perf_counter()-t:.1f}s   [{v}]")
    print(f"  dF vs kerker = {f_l-f_ref:+.2e} eV")


def _mag_filter(g2, w0, q0=1.1):
    """Hand-set magnetization-channel filter f_mag(G²) = w0 + (1−w0)·G²/(G²+q0²):
    Kerker's shape rescaled to [w0, 1] so the uniform moment mode (G=0) is damped
    to w0 (not frozen, as bare Kerker would) while finite-G spin modes mix at ~full
    rate. w0 → 1 recovers plain mag mixing; w0 → 0 recovers bare Kerker (freezes
    the moment)."""
    import math
    c_raw = torch.tensor(math.log(w0 / (1.0 - w0)), dtype=RDTYPE)
    w_raw = torch.tensor([_inv_softplus(1.0 - w0)], dtype=RDTYPE)
    logq2 = torch.tensor([2.0 * math.log(q0)], dtype=RDTYPE)
    return MultipoleKerkerPrecond(g2, w_raw, logq2, c_raw)


def run_ni(q0=1.1):
    """fcc Ni near the Stoner instability — the adversarial FM convergence case
    (wisdom.md). Tests a MAGNETIZATION-channel preconditioner: bare Kerker on the
    charge block, f_mag on the spin block, via BlockPrecond, under johnson mixing.
    Sweeps the uniform-mode damping w0; reports iterations AND the converged moment
    (a collapsed moment is a failure, not a win)."""
    print("\n=== fcc Ni (nspin=2, near Stoner, johnson, 45 Ry, 4x4x4) ===")
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system
    a = 3.52
    ni = parse_upf(FIX / "PD_Ni_PBE.upf")
    cell = 0.5 * a * np.array([[0, 1, 1.0], [1, 0, 1], [1, 1, 0]])

    def sys_():
        return setup_system(cell, np.zeros((1, 3)), [0], [ni], ecut=45 * RY,
                            kmesh=(4, 4, 4), nbands=14, use_symmetry=True)

    common = dict(smearing="gaussian", width=0.1, nspin=2, start_mag=[0.5],
                  mixing_scheme="johnson", verbose=False, rhotol=1e-5, etol=1e-8)
    grid = sys_().grid
    g2_dens = grid.g2.reshape(-1)[grid.dens_mask.reshape(-1)].to(RDTYPE)
    kerker_tot = MultipoleKerkerPrecond.kerker(g2_dens, q0)

    t = time.perf_counter()
    ref = scf(sys_(), SpinPBE(), mixing_alpha=0.7, **common)
    print(f"  johnson (Kerker charge, plain mag)   {ref.n_iter:3d} iters   "
          f"m={ref.mag_total:+.3f} muB   {time.perf_counter()-t:.1f}s")

    for w0 in (0.4, 0.6, 0.8):
        block = BlockPrecond([(len(g2_dens), kerker_tot),
                              (len(g2_dens), _mag_filter(g2_dens, w0, q0))])
        t = time.perf_counter()
        r = scf(sys_(), SpinPBE(), mixing_alpha=0.7, precond_op=block, **common)
        collapsed = abs(r.mag_total) < 0.3 and abs(ref.mag_total) > 0.3
        v = _verdict(r.n_iter, ref.n_iter) if not collapsed else "COLLAPSE"
        print(f"  + mag filter w0={w0:.1f}                {r.n_iter:3d} iters   "
              f"m={r.mag_total:+.3f} muB   {time.perf_counter()-t:.1f}s   [{v}]")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("synthetic", "both"):
        run_synthetic()
    if which in ("al", "both"):
        run_metal("al")
    if which in ("cu", "both"):
        run_metal("cu")
    if which in ("fe", "magnetism", "both"):
        run_fe()
    if which in ("pt", "soc", "both"):
        run_pt_soc()
    if which in ("cu3al", "intermetallic", "both"):
        run_cu3al()
    if which in ("ni", "stoner"):
        run_ni()
