"""Self-consistent Hubbard U for Cr2O3 by autograd.

Cr2O3 (eskolaite) is an antiferromagnetic insulator, exactly the nspin=2
insulating regime that `linear_response_u_autodiff` targets. This runs the
Sternheimer/autograd linear-response U on the Cr 3d manifold in ONE ground-state
SCF with no finite differences and no probe re-runs, and the conventional
alpha-perturbation Cococcioni U on the same cell for comparison. QE-HP and
Cococcioni linear response put U(Cr 3d) in the 3 to 4 eV range.

  uv run python benchmarks/hubbard_cr2o3/run.py \
      --pseudo-dir ~/mineral_pseudos --outdir ~/cr2o3_u --ecut-ry 50 --kmesh 2 2 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "minerals"))  # structures.py
import structures as S  # noqa: E402

RY = 13.605693122994


def _to_py(x):
    """JSON default: torch tensors and numpy scalars become plain floats."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ecut-ry", type=float, default=50.0)
    ap.add_argument("--kmesh", type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-conventional", action="store_true")
    a = ap.parse_args()

    import torch

    from gradwave.core.xc.spin import SpinPBE
    from gradwave.postscf.hubbard_u import linear_response_u, linear_response_u_autodiff
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system
    torch.set_num_threads(8)

    m = S.eskolaite(ecut_ry=a.ecut_ry, kmesh=tuple(a.kmesh))
    pdir = Path(a.pseudo_dir).expanduser()
    upfs = [parse_upf(pdir / p) for p in m.pseudos]  # [Cr, O]

    nelec = float(sum(upfs[s].z_valence for s in m.species))
    nocc = int(np.ceil(nelec / 2))
    nbands = nocc + max(12, int(0.25 * nocc))

    system = setup_system(m.cell, m.positions, m.species, upfs,
                          ecut=a.ecut_ry * RY, kmesh=tuple(a.kmesh), nbands=nbands)
    if a.device != "cpu":
        system = system.to(a.device)

    width_ev = m.degauss_ry * RY
    scf_kw = dict(nspin=2, start_mag=m.start_mag, etol=1e-7, rhotol=1e-6,
                  max_iter=200, mixing_scheme="johnson", verbose=False)

    out = {"material": "Cr2O3", "afm": "Cr +-+- along c", "ecut_ry": a.ecut_ry,
           "kmesh": a.kmesh, "nbands": nbands, "nelec": nelec,
           "device": a.device, "start_mag": list(m.start_mag),
           "manifold": "Cr 3d (l=2, species=0)",
           "literature_U_eV": "3 to 4 (Cococcioni/HP linear response)"}

    # --- autograd U: one SCF, Sternheimer response, no finite differences ---
    try:
        t0 = time.perf_counter()
        ad = linear_response_u_autodiff(system, SpinPBE(), l=2, species=0, site=0,
                                        smearing="gaussian", width=width_ev,
                                        scf_kwargs=scf_kw)
        ad["wall_s"] = time.perf_counter() - t0
        out["autodiff"] = ad
        print("AUTODIFF  U_eV=%.4f  chi0=%.4f chi=%.4f  wall=%.1fs" %
              (float(ad["U_eV"]), float(ad["chi0"]), float(ad["chi"]), ad["wall_s"]))
    except Exception as e:  # noqa: BLE001
        out["autodiff"] = {"error": repr(e), "trace": traceback.format_exc()[-2500:]}
        print("AUTODIFF FAILED:", repr(e))

    # --- conventional alpha-perturbation U on the same cell ---
    if not a.skip_conventional:
        try:
            t0 = time.perf_counter()
            cv = linear_response_u(system, SpinPBE(), l=2, species=0, site=0,
                                   alpha=0.1, smearing="gaussian", width=width_ev,
                                   scf_kwargs=scf_kw)
            cv["wall_s"] = time.perf_counter() - t0
            out["conventional"] = cv
            print("CONVENTIONAL  U_eV=%.4f  chi0=%.4f chi=%.4f  wall=%.1fs" %
                  (float(cv["U_eV"]), float(cv["chi0"]), float(cv["chi"]), cv["wall_s"]))
        except Exception as e:  # noqa: BLE001
            out["conventional"] = {"error": repr(e), "trace": traceback.format_exc()[-2500:]}
            print("CONVENTIONAL FAILED:", repr(e))

    if "U_eV" in out.get("autodiff", {}) and "U_eV" in out.get("conventional", {}):
        out["dU_autodiff_minus_conventional_eV"] = (
            float(out["autodiff"]["U_eV"]) - float(out["conventional"]["U_eV"]))

    outdir = Path(a.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text(json.dumps(out, indent=2, default=_to_py))
    print(json.dumps(out, indent=2, default=_to_py))


if __name__ == "__main__":
    main()
