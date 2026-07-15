"""End-to-end learned-XC training through the PAW SCF (the adjoint demo).

Four systems span the adjoint's coverage: a PAW insulator (Si), a smeared
metal with the Fermi-surface response (Al), a correlated insulator with
the +U occupation response (Si, U = 4 on 3p), and a spin-polarized
molecule (triplet O2). Target densities are generated at the PBE values
of the learnable exchange parameters (kappa, mu); training starts from a
perturbed point and must recover PBE through the full self-consistent
response — every gradient is one adjoint solve, no finite differences.

Run: uv run python examples/train_xc_paw.py [epochs]
Writes results to examples/train_xc_paw.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)

from gradwave.core.xc.learnable import (  # noqa: E402
    PBE_KAPPA,
    PBE_MU,
    LearnableSpinX,
    LearnableX,
    _inv_softplus,
)
from gradwave.postscf.uspp_implicit import (  # noqa: E402
    uspp_density_loss_param_grads,
)
from gradwave.pseudo.upf_paw import parse_upf_paw  # noqa: E402
from gradwave.scf.uspp import scf_uspp, setup_uspp  # noqa: E402
from gradwave.scf.uspp_hubbard import HubbardManifold  # noqa: E402

RY = 13.605693122994
FIX = Path(__file__).parents[1] / "tests/fixtures/qe"
OUT = Path(__file__).parent / "train_xc_paw.json"

SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0, 0], [1.3575] * 3])
AL_CELL = 4.04 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
O2_CELL = 6.0 * np.eye(3)
O2_POS = np.array([[3.0, 3.0, 2.40], [3.06, 3.0, 3.75]])

paw_si = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
paw_al = parse_upf_paw(FIX / "pseudos" / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
paw_o = parse_upf_paw(FIX / "pseudos" / "O.pbe-n-kjpaw_psl.1.0.0.UPF")

SYSTEMS = {
    "si": dict(
        build=lambda: setup_uspp(SI_CELL, SI_POS, [0, 0], [paw_si],
                                 ecut=15 * RY, kmesh=(2, 2, 2),
                                 ecutrho=60 * RY),
        spin=False,
        scf=dict(etol=1e-11, rhotol=1e-9, verbose=False, max_iter=80),
        adj=dict(floor_tol=1e-4)),
    "al": dict(
        build=lambda: setup_uspp(AL_CELL, np.zeros((1, 3)), [0], [paw_al],
                                 ecut=20 * RY, kmesh=(2, 2, 2),
                                 ecutrho=100 * RY, nbands=8),
        spin=False,
        scf=dict(smearing="gaussian", width=0.5, etol=1e-11, rhotol=1e-9,
                 verbose=False, max_iter=120),
        adj=dict(floor_tol=1e-4)),
    "si_u4": dict(
        build=lambda: setup_uspp(SI_CELL, SI_POS, [0, 0], [paw_si],
                                 ecut=15 * RY, kmesh=(2, 2, 2),
                                 ecutrho=60 * RY),
        spin=False,
        scf=dict(etol=1e-11, rhotol=1e-9, verbose=False, max_iter=80,
                 hubbard=[HubbardManifold(species=0, l=1, u=4.0)]),
        adj=dict(floor_tol=1e-4)),
    "o2": dict(
        build=lambda: setup_uspp(O2_CELL, O2_POS, [0, 0], [paw_o],
                                 ecut=35 * RY, kmesh=(1, 1, 1),
                                 ecutrho=280 * RY, nbands=10),
        spin=True,
        scf=dict(nspin=2, start_mag=[0.5], smearing="gaussian",
                 width=0.01 * RY, etol=3e-7, criterion="energy",
                 rhotol=1e-9, verbose=False, max_iter=90),
        # vacuum spin-f_xc broadens the kernel spectrum (O2 test
        # lessons); kerker_q0 damps the low-G vacuum noise that set the
        # old stagnation floor (measured: floored 1.4e-4 -> converges
        # 1.4e-5). floor_tol stays as the safety net for parameter
        # points where the floor wanders back up.
        adj=dict(history=40, beta=0.3, max_outer=120, outer_tol=2e-5,
                 cg_tol=1e-10, floor_tol=3e-4, kerker_q0=1.5)),
}


def make_xc(spin, kappa_raw, mu_raw):
    xc = LearnableSpinX() if spin else LearnableX()
    with torch.no_grad():
        xc.raw_kappa.copy_(kappa_raw)
        xc.raw_mu.copy_(mu_raw)
    return xc


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    systems = {name: dict(cfg, sys=cfg["build"]()) for name, cfg
               in SYSTEMS.items()}

    # target densities at the PBE parameter values
    k_pbe = _inv_softplus(PBE_KAPPA)
    m_pbe = _inv_softplus(PBE_MU)
    print("generating target densities at PBE (kappa, mu) ...")
    for name, s in systems.items():
        t0 = time.time()
        xc = make_xc(s["spin"], k_pbe, m_pbe)
        r = scf_uspp(s["sys"], xc, **s["scf"])
        assert r["converged"], name
        s["rho_ref"] = r["rho"].detach().clone()
        s["norm"] = float((s["rho_ref"] ** 2).sum())
        s["prev"] = None  # warm-start chain across epochs
        print(f"  {name}: {r['n_iter']} it ({time.time() - t0:.0f}s)")

    # perturbed start
    kap = _inv_softplus(1.10).clone().requires_grad_(False)
    mu = _inv_softplus(0.30).clone().requires_grad_(False)
    opt_state = {"m": torch.zeros(2), "v": torch.zeros(2), "t": 0}
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    prev_loss = None
    hist = []
    print(f"start: kappa={float(torch.nn.functional.softplus(kap)):.4f} "
          f"mu={float(torch.nn.functional.softplus(mu)):.4f} "
          f"(PBE {PBE_KAPPA:.4f}/{PBE_MU:.4f})")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        total, grad = 0.0, torch.zeros(2)
        for name, s in systems.items():
            xc = make_xc(s["spin"], kap, mu)
            t_s = time.time()
            r = scf_uspp(s["sys"], xc, start_from=s["prev"], **s["scf"])
            print(f"    {name}: scf {r['n_iter']} it "
                  f"({time.time() - t_s:.0f}s)")
            assert r["converged"], (name, ep)
            s["prev"] = r
            rho_ref, norm = s["rho_ref"], s["norm"]

            def loss_fn(rho, rho_ref=rho_ref, norm=norm):
                d = rho - rho_ref
                return (d * d).sum() / norm

            loss, g = uspp_density_loss_param_grads(r, xc, loss_fn,
                                                    **s["adj"])
            total += float(loss)
            grad += torch.tensor([float(g["raw_kappa"]),
                                  float(g["raw_mu"])])
        # Adam with backtracking: halve the rate when the loss rises
        # (fixed lr 0.05 overshoots past the optimum — the same artifact
        # the NC two-parameter fit documents)
        if prev_loss is not None and total > prev_loss:
            lr *= 0.5
        prev_loss = total
        opt_state["t"] += 1
        opt_state["m"] = b1 * opt_state["m"] + (1 - b1) * grad
        opt_state["v"] = b2 * opt_state["v"] + (1 - b2) * grad * grad
        mh = opt_state["m"] / (1 - b1 ** opt_state["t"])
        vh = opt_state["v"] / (1 - b2 ** opt_state["t"])
        step = lr * mh / (vh.sqrt() + eps)
        kap = kap - step[0]
        mu = mu - step[1]
        k_now = float(torch.nn.functional.softplus(kap))
        m_now = float(torch.nn.functional.softplus(mu))
        hist.append(dict(epoch=ep, loss=total, kappa=k_now, mu=m_now,
                         lr=lr, seconds=round(time.time() - t0)))
        print(f"epoch {ep:3d}: loss {total:.6e}  kappa {k_now:.4f}  "
              f"mu {m_now:.4f}  ({time.time() - t0:.0f}s)")
        OUT.write_text(json.dumps(dict(
            target=dict(kappa=PBE_KAPPA, mu=PBE_MU),
            start=dict(kappa=1.10, mu=0.30), history=hist), indent=1))
    print(f"final: kappa {k_now:.4f} (PBE {PBE_KAPPA:.4f}, "
          f"err {abs(k_now - PBE_KAPPA):.1e}), mu {m_now:.4f} "
          f"(PBE {PBE_MU:.4f}, err {abs(m_now - PBE_MU):.1e})")


if __name__ == "__main__":
    main()
