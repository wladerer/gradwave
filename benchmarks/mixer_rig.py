"""Offline mixer laboratory on the linearized SCF map (task #65).

Testing mixers on the real FM Ni SCF costs 15-50 min per data point. This
rig replaces that loop: it applies the TRUE one-iteration SCF map F by
finite differences around a converged fixed point x*, extracts the
dominant Krylov subspace of the Jacobian J = dF/dx by Arnoldi (the modes
that decide stability: Stoner-expansive spin, stiff on-site, Kerker-class
charge), and then tests any mixer against the reduced linear model in
milliseconds. The Ritz spectrum of J is reported — the measured gain
spectrum the mixer must tame, not folklore.

Stages:
  build <rig.pt>   converge the Ni reference case, run m Arnoldi steps of
                   FD J-applies (one warm-started tight SCF iteration per
                   apply), save {x*, V, H, spectrum, packing info}
  test  <rig.pt>   run the mixer zoo on the linear model x -> x* + J(x-x*)
                   and print iterations-to-tolerance per configuration

The FD map evaluation calls _scf_iteration directly (tight
diago_tol) and reads the RAW pre-mixing output (rho_out_spin +
rho_ij_atoms), so every physical coupling (spin, becsum<->ddd, smearing,
Fermi shift, augmentation) is in J exactly.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradwave.core.fftbox import r_to_g  # noqa: E402
from gradwave.core.xc.spin import SpinPBE  # noqa: E402
from gradwave.dtypes import CDTYPE  # noqa: E402
from gradwave.pseudo.upf_paw import parse_upf_paw  # noqa: E402
from gradwave.scf.uspp import scf_uspp, setup_uspp  # noqa: E402
from gradwave.scf.uspp_loop import _build_iter_ops, _scf_iteration  # noqa: E402

RY = 13.605693122994
FIX = ROOT / "tests/fixtures/qe"


def ni_system():
    import json

    ref = json.loads((FIX / "ni_paw_spin_ci" / "reference.json").read_text())
    paw = parse_upf_paw(FIX / "pseudos" / "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF")
    cell = np.array([[0.0, 1.76, 1.76], [1.76, 0.0, 1.76], [1.76, 1.76, 0.0]])
    return setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=50 * RY,
                      kmesh=(4, 4, 4), ecutrho=400 * RY, nbands=18,
                      fft_shape=ref["fft_dims"])


SCF_KW = dict(nspin=2, start_mag=[0.5], smearing="gaussian", width=0.1)


class Packing:
    """Composite vector <-> (rho_spin r-space, becsum) in the SAME layout
    as scf_uspp's mixing vector: [rho_tot(G), m(G), bec_up, bec_dn]."""

    def __init__(self, system):
        grid = system.grid
        self.shape = tuple(grid.shape)
        self.n_pts = grid.n_points
        self.mask = grid.dens_mask.reshape(-1)
        self.ng = int(self.mask.sum())
        self.slices = system.atom_slices
        self.nbec = sum((s1 - s0) ** 2 for (s0, s1) in self.slices)

    def pack(self, rho_spin, rho_ij):
        vecs = [r_to_g(c.to(CDTYPE)).reshape(-1)[self.mask] for c in rho_spin]
        vecs = [vecs[0] + vecs[1], vecs[0] - vecs[1]]
        bec = [torch.cat([m.reshape(-1) for m in rho_ij[isp]])
               for isp in (0, 1)]
        return torch.cat(vecs + bec)

    def unpack(self, v):
        ng = self.ng
        tot, mag = v[:ng], v[ng:2 * ng]
        chans = [(tot + mag) / 2.0, (tot - mag) / 2.0]
        rho_spin = []
        for c in chans:
            box = torch.zeros(self.n_pts, dtype=CDTYPE)
            box[self.mask] = c
            rho_spin.append(torch.fft.ifftn(
                box.reshape(self.shape) * self.n_pts, dim=(-3, -2, -1)).real)
        rho_ij, off = [[], []], 2 * ng
        for isp in (0, 1):
            for (s0, s1) in self.slices:
                n = s1 - s0
                rho_ij[isp].append(
                    v[off:off + n * n].reshape(n, n).clone())
                off += n * n
        return rho_spin, rho_ij


_RAW_CACHE = {}


def raw_map(system, xc, pk, v):
    """One tight SCF iteration: packed density in -> RAW packed density out.
    Direct _scf_iteration evaluation (stage 3): operators built once per
    system, orbital warm starts carried across J-applies — roughly half
    the old scf_uspp(max_iter=1) round-trip cost."""
    key = id(system)
    if key not in _RAW_CACHE:
        ops = _build_iter_ops(system, xc, nspin=2,
                              smearing=SCF_KW["smearing"],
                              width=SCF_KW["width"], batched=True)
        _RAW_CACHE[key] = (ops, [[None] * ops.nk for _ in range(2)],
                           [None, None])
    ops, coeffs, coeffs_b = _RAW_CACHE[key]
    rho_spin, rho_ij = pk.unpack(v)
    step = _scf_iteration(ops, rho_spin, rho_ij, coeffs, coeffs_b, None,
                          1e-11, 0)
    return pk.pack(step["rho_out_s"], step["rho_ij_s"])


def build(out_path):
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 36
    system = ni_system()
    xc = SpinPBE()
    pk = Packing(system)
    print(f"converging the reference state (ng={pk.ng}, nbec={pk.nbec}) ...")
    t0 = time.time()
    res = scf_uspp(system, xc, mixing_alpha=0.3, etol=1e-8,
                   criterion="energy", verbose=False, max_iter=150, **SCF_KW)
    print(f"  converged={res['converged']} it={res['n_iter']} "
          f"m={res['mag_total']:+.4f}  ({time.time() - t0:.0f}s)")
    xstar = pk.pack(res["rho_spin"], res["rho_ij_atoms"])
    fx0 = raw_map(system, xc, pk, xstar)
    fp_err = float(torch.linalg.norm(fx0 - xstar))
    print(f"  fixed-point self-consistency |F(x*)-x*| = {fp_err:.3e}")

    # Arnoldi on J = dF/dx via central-free FD: J d = (F(x*+h d) - F(x*))/h
    scale = float(torch.linalg.norm(xstar))
    torch.manual_seed(11)
    q = torch.randn_like(xstar)
    q = q / torch.linalg.norm(q)
    vbasis = [q]
    hmat = torch.zeros(m + 1, m, dtype=CDTYPE)
    for j in range(m):
        t0 = time.time()
        h = 1e-4 * scale
        w = (raw_map(system, xc, pk, xstar + h * vbasis[j]) - fx0) / h
        for i, vi in enumerate(vbasis):
            hmat[i, j] = (vi.conj() @ w)
            w = w - hmat[i, j] * vi
        hmat[j + 1, j] = torch.linalg.norm(w).to(CDTYPE)
        print(f"  arnoldi {j + 1:2d}/{m}: |w_perp| = "
              f"{float(hmat[j + 1, j].real):.3e}  ({time.time() - t0:.0f}s)")
        if float(hmat[j + 1, j].real) < 1e-12:
            break
        vbasis.append(w / hmat[j + 1, j])
    hm = hmat[:len(vbasis) - 1, :len(vbasis) - 1]
    ritz = torch.linalg.eigvals(hm)
    order = torch.argsort(-ritz.abs())
    print("Ritz values of J (gain spectrum, |.| desc):")
    for lam in ritz[order][:12]:
        print(f"  {lam.real:+.4f} {lam.imag:+.4f}i   |.| = {abs(lam):.4f}")

    # freeze the Stoner preconditioner at x* so `test` can run
    # preconditioned variants without any SCF
    from gradwave.core.occupations import SCHEMES
    from gradwave.scf.spin_precond import build_stoner_precond

    sp = build_stoner_precond(
        system, res["coeffs"], res["eigenvalues"], res["fermi"],
        SCHEMES["gaussian"], SCF_KW["width"],
        res["rho_spin"][0] + res["rho_spin"][1],
        res["rho_spin"][0] - res["rho_spin"][1], xc)
    stoner = (dict(u=sp._u, w=sp._w, c=sp.cvals, vol=sp.volume)
              if sp is not None else None)
    torch.save(dict(xstar=xstar, v=torch.stack(vbasis[:-1]), h=hm,
                    ritz=ritz, ng=pk.ng, nbec=pk.nbec, fp_err=fp_err,
                    stoner=stoner),
               out_path)
    print(f"saved rig -> {out_path} (stoner rank "
          f"{0 if stoner is None else stoner['c'].shape[0]})")


def test(rig_path):
    from gradwave.scf.mixing import BroydenMixer, PulayMixer

    rig = torch.load(rig_path, weights_only=False)
    xstar, v, hm = rig["xstar"], rig["v"], rig["h"]
    ng, nbec = rig["ng"], rig["nbec"]
    n = xstar.shape[0]

    def j_apply(d):
        c = v.conj() @ d  # (m,) Krylov projection
        return (hm @ c) @ v

    # worst-mode start: perturb along the Ritz vector of largest |gain|
    evals, evecs = torch.linalg.eig(hm)
    worst = evecs[:, torch.argmax(evals.abs())]
    x0 = xstar + 1e-2 * float(torch.linalg.norm(xstar)) * (worst @ v)

    g2 = torch.zeros(n, dtype=torch.float64)  # metric-neutral rig space
    step_scale = torch.cat([
        torch.ones(2 * ng, dtype=torch.float64),
        torch.full((2 * nbec,), 0.4, dtype=torch.float64)])

    def run(mixer, label, max_it=300, tol=1e-10):
        x = x0.clone()
        r0 = None
        for it in range(1, max_it + 1):
            r = j_apply(x - xstar) - (x - xstar)
            rn = float(torch.linalg.norm(r))
            r0 = r0 or rn
            if not np.isfinite(rn) or rn > 1e6 * r0:
                print(f"  {label:34s} DIVERGED at it {it}")
                return
            if rn < tol * r0:
                print(f"  {label:34s} converged in {it:3d}")
                return
            x = mixer.step(x, x + r)
        print(f"  {label:34s} NOT converged in {max_it} (res {rn / r0:.1e})")

    stoner_fn = None
    if rig.get("stoner") is not None:
        from gradwave.scf.spin_precond import StonerSpinPrecond

        st = rig["stoner"]
        sp = StonerSpinPrecond(st["u"], st["w"], st["c"], st["vol"])

        def stoner_fn(rvec, _sp=sp, _ng=ng):
            out = rvec.clone()
            out[_ng:2 * _ng] = _sp.apply(rvec[_ng:2 * _ng])
            return out

    from gradwave.scf.mixing import JohnsonMixer

    zoo = []
    for alpha in (0.7, 0.3):
        zoo.append((PulayMixer(g2, alpha=alpha, history=8, check_g0=False,
                               step_scale=step_scale), f"pulay a={alpha}"))
        zoo.append((BroydenMixer(g2, alpha=alpha, history=8, check_g0=False,
                                 step_scale=step_scale), f"broyden a={alpha}"))
        zoo.append((JohnsonMixer(g2, alpha=alpha, history=8, check_g0=False,
                                 step_scale=step_scale), f"johnson a={alpha}"))
    if stoner_fn is not None:
        for cls, name in ((PulayMixer, "pulay"), (BroydenMixer, "broyden"),
                          (JohnsonMixer, "johnson")):
            mx = cls(g2, alpha=0.7, history=8, check_g0=False,
                     step_scale=step_scale)
            mx.extra_precond = stoner_fn
            zoo.append((mx, f"{name} a=0.7 + stoner"))
    for mx, label in zoo:
        run(mx, label)


if __name__ == "__main__":
    mode = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/ni_rig.pt"
    if mode == "build":
        build(path)
    elif mode == "test":
        test(path)
    else:
        sys.exit("usage: mixer_rig.py build|test [rig.pt] [m]")
