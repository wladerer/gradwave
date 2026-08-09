"""Constant-potential (grand-canonical) electrochemistry demo on Pt(111).

The Level-2 ESM feature, end-to-end: hold the electrode at a fixed Fermi level µ
(the electrode potential), let the electron count N *float*, and let the ESM
capacitor plates (boundary="open_z_metal") carry the counter-charge. This is the
thing constant-CHARGE DFT cannot do — the electrode is a reservoir at a chosen
potential, not a cell with a fixed number of electrons.

Reports, for a Pt(111) slab:
  - PZC work function Φ (neutral run) and U_PZC vs SHE (Φ − 4.44 V, Trasatti),
  - N(µ): the charge the slab draws to sit at each potential (ΔN vs the neutral N),
  - ∂N/∂µ: the differential (interfacial) capacitance — must be > 0 (DOS > 0),
  - U vs SHE and grand potential Ω = F − µN at each µ.

Path note: the constant-µ SCF and work function are wired ONLY for the
norm-conserving driver `scf.loop.scf` (not USPP/PAW — scf_uspp has no target_mu,
and USPPResult has no v_eff), and must be driven through `scf()` directly (the ASE
calculator drops boundary/target_mu). So this uses the ONCV Pt pseudo at the NC
cutoff. Template: tests/integration/test_esm_constant_mu.py.

    uv run python constant_potential.py            # full µ sweep on the Pt slab
    uv run python constant_potential.py --quick    # neutral + ±0.2 V only (fast check)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.work_function import electrode_potential, work_function
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

HERE = Path(__file__).parent
RESULTS = HERE / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ECUT_NC = 80.0 * RY          # ONCV 5d metal needs a high wavefunction cutoff
KPTS = (4, 4, 1)
WIDTH = 0.3                  # eV, fermi-dirac. A high-DOS metal needs a BROAD
# smearing for the grand-canonical SCF: Pt's large d-band DOS at E_F makes dN/dµ
# huge, so a sharp (0.1 eV) smearing lets the floating N oscillate and the fixed-µ
# SCF never converges (even at µ=µ0). Na tolerated 0.1 only because it is
# free-electron (low DOS); 0.3 eV ≈ the validated Na-test width.
U_SHE_ABS = 4.44            # V, absolute SHE potential (Trasatti)


def run(quick: bool = False) -> dict:
    RESULTS.mkdir(exist_ok=True)
    out_f = RESULTS / "constant_potential_Pt.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}

    def save():
        out_f.write_text(json.dumps(res, indent=2))

    # Pt(111) slab: reuse the PBE-relaxed clean-slab geometry (work function is
    # insensitive to the sub-0.1 Å pseudo-dependent relaxation difference), with the
    # ONCV Pt pseudo at the NC cutoff. Build a FRESH System per SCF call (matches the
    # validated test pattern — no reused-state assumptions on the metered box).
    pt = parse_upf(str(HERE / "pseudos_nc" / "Pt_ONCV.upf"))
    xyz = RESULTS / "Pt_slab_relaxed.xyz"
    if not xyz.exists():
        raise SystemExit(f"missing {xyz} — run run_pair.py Pt H (or Pt CO) first")
    atoms = read(xyz)
    cell = np.array(atoms.get_cell())
    pos = atoms.get_positions()
    species = [0] * len(atoms)                    # pure Pt
    xc = PBE()
    # local_tf: the vacuum-aware preconditioner (the slab is mostly vacuum + plates),
    # which damps the spatial charge-sloshing between slab and capacitor plates.
    # mixing_alpha=0.4: extra damping for the oscillatory grand-canonical N feedback
    # (the instability is oscillatory, so a more conservative step stabilizes it).
    common = dict(smearing="fermi-dirac", width=WIDTH, boundary="open_z_metal",
                  precond="local_tf", mixing_alpha=0.4, max_iter=250,
                  etol=1e-6, rhotol=1e-5, verbose=False)

    def _scf(**kw):
        sysx = setup_system(cell, pos, species, [pt], ecut=ECUT_NC,
                            kmesh=KPTS, use_symmetry=True)
        if DEVICE == "cuda":
            sysx = sysx.to(DEVICE)
        return scf(sysx, xc, **common, **kw)

    # 0. neutral fixed-N run → the PZC reference AND the warm-start seed for every
    #    fixed-µ run. A charged high-DOS metal cold-starts terribly; seeding the
    #    fixed-µ SCF from the converged neutral density means only N has to adjust,
    #    not the whole density. Always recomputed (needed live as the seed).
    t = time.time()
    r0 = _scf()
    if not r0.converged:
        raise SystemExit("neutral SCF did not converge")
    phi_pzc = float(work_function(r0))
    mu0, n0 = float(r0.fermi), float(r0.n_electrons)
    res["pzc"] = dict(mu0=mu0, n0=n0, phi_pzc=phi_pzc,
                      u_pzc_vs_she=phi_pzc - U_SHE_ABS, n_iter=int(r0.n_iter))
    print(f"[PZC] µ0={mu0:+.3f} eV  N0={n0:.3f}  Φ_PZC={phi_pzc:.3f} eV  "
          f"U_PZC={phi_pzc - U_SHE_ABS:+.3f} V vs SHE  ({time.time() - t:.0f}s)",
          flush=True)
    save()

    # 1. fixed-µ (grand-canonical) sweep — N floats, plates carry the counter-charge
    dmus = [-0.2, 0.0, 0.2] if quick else [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
    res.setdefault("sweep", {})
    for dmu in dmus:
        key = f"{dmu:+.2f}"
        if key in res["sweep"]:
            continue
        t = time.time()
        r = _scf(target_mu=mu0 + dmu, start_from=r0)  # seed from neutral density
        if not r.converged:
            print(f"[µ{key}] did NOT converge — skipping", flush=True)
            continue
        ep = electrode_potential(r, phi_pzc=phi_pzc)
        omega = float(r.energies.free_energy) - r.fermi * r.n_electrons
        row = dict(dmu=dmu, mu=mu0 + dmu, N=float(r.n_electrons),
                   dN=float(r.n_electrons) - n0, phi=float(ep.work_function),
                   u_vs_she=float(ep.potential_vs_she),
                   u_vs_pzc=float(ep.potential_vs_pzc), omega=omega,
                   n_iter=int(r.n_iter))
        res["sweep"][key] = row
        print(f"[µ{key}] N={row['N']:.3f} (ΔN={row['dN']:+.3f}e)  Φ={row['phi']:.3f} eV"
              f"  U={row['u_vs_she']:+.3f} V vs SHE  ΔU_PZC={row['u_vs_pzc']:+.3f} V"
              f"  ({time.time() - t:.0f}s)", flush=True)
        save()

    # 2. differential capacitance ∂N/∂µ (central difference around the PZC)
    sw = res["sweep"]
    if "+0.20" in sw and "-0.20" in sw:
        dN_dmu = (sw["+0.20"]["N"] - sw["-0.20"]["N"]) / 0.4
        # areal capacitance: e·(∂N/∂µ) per surface area, both faces → /2. Units:
        # e/(V·Å²)·... → µF/cm². 1 e/(V·Å²) = 1.602e-19 C /(V·1e-16 cm²) = 1602 µF/cm².
        area_A2 = float(np.linalg.norm(np.cross(
            np.array(read(RESULTS / "Pt_slab_relaxed.xyz").get_cell())[0],
            np.array(read(RESULTS / "Pt_slab_relaxed.xyz").get_cell())[1])))
        cap_uF_cm2 = dN_dmu / (2.0 * area_A2) * 1602.0
        res["capacitance"] = dict(dN_dmu=dN_dmu, area_A2=area_A2,
                                  cap_uF_cm2=cap_uF_cm2)
        print(f"\n[∂N/∂µ] = {dN_dmu:+.4f} e/eV  →  C ≈ {cap_uF_cm2:.1f} µF/cm² "
              f"(differential, both faces)", flush=True)
        save()

    print("\n=== constant-potential summary (Pt(111), NC/ONCV, PBE) ===", flush=True)
    print(f"  PZC: Φ={phi_pzc:.2f} eV,  U_PZC={phi_pzc - U_SHE_ABS:+.2f} V vs SHE",
          flush=True)
    for k in sorted(res["sweep"], key=lambda s: float(s)):
        r = res["sweep"][k]
        print(f"  µ{k}: U={r['u_vs_she']:+.2f} V vs SHE,  ΔN={r['dN']:+.3f} e",
              flush=True)
    save()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="neutral + ±0.2 V only")
    a = ap.parse_args()
    run(a.quick)
