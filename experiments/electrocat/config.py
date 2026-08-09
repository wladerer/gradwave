"""Shared settings + calculator factory for the electrocatalysis pipeline.

PAW (psl.1.0.0 PBE) is the production set — lower ecut than NC for the 5d metals,
and adsorption energies are what we want. The r2SCAN/ISDF stretch uses the NC set
(meta-GGA is NC-only in gradwave); see r2scan_isdf.py.
"""

from __future__ import annotations

from pathlib import Path

from gradwave.constants import RY_EV as RY

HERE = Path(__file__).parent

# --- PAW production set (psl.1.0.0, PBE) ---
_PAW = {
    "Pt": "Pt.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Au": "Au.pbe-n-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
}
PSEUDOS = {el: str(HERE / "pseudos" / f) for el, f in _PAW.items()}

# cutoffs from the pseudo headers (max wfc ~47 Ry, max rho ~401 Ry) + margin
ECUT = 50.0 * RY        # eV
ECUTRHO = 400.0 * RY    # eV
KPTS_SLAB = (4, 4, 1)   # 2×2 surface cell (bump to 6×6×1 for production accuracy)
KPTS_GAS = (1, 1, 1)
SMEARING = "cold"       # Marzari-Vanderbilt (metals)
WIDTH = 0.15            # eV
XC = "pbe"
FMAX = 0.03             # eV/Å relaxation threshold
MAX_STEPS = 80
DEVICE = "cuda"         # "cpu" for a local debug run

# a fast debug profile (tiny, runs on a laptop CPU in ~minutes) — see run_pair.py --debug
DEBUG = dict(ecut=25.0 * RY, ecutrho=150.0 * RY, kpts_slab=(2, 2, 1),
             device="cpu", fmax=0.1, max_steps=8)


def make_calc(*, is_gas: bool = False, device: str | None = None,
              kpts=None, xc: str | None = None, ecut: float | None = None,
              ecutrho: float | None = None, **kw):
    """A GradWave ASE calculator, tuned per the docs (wisdom/performance/
    convergence.md) for a vacuum-slab PAW-metal relaxation on GPU:

    - precond="local_tf": constant Kerker over-damps the vacuum region of a slab
      (or molecule box); local Thomas-Fermi tracks local density, capped at the
      Kerker value in the bulk (cuts iterations, energy bit-identical).
    - mixing_scheme="johnson": the PAW default resolver already picks it; pinned.
    - metals (slabs): loosen rhotol to 1e-5 — a smeared metal floors the density
      residual at occupation noise while the free energy is long settled; gate on
      the energy tail (etol) instead of fighting the 1e-7 floor. Verify fractional
      occupations actually appear at WIDTH on the chosen k-mesh (else it is
      silently a fixed-occupation insulator).
    - eigensolver stays davidson (chebyshev is NC-only, errors on PAW, slower on
      H100); mixed_precision stays OFF (regresses metals; no win on H100 fp64);
      nbands stays auto (20% metal headroom — do NOT trim). use_symmetry on.

    Gas molecules are closed-shell insulators (smearing='none', Γ-only, tight tol).
    """
    from gradwave.calculator import GradWave

    metal = not is_gas
    return GradWave(
        ecut=ecut or ECUT,
        ecutrho=ecutrho or ECUTRHO,
        pseudopotentials=PSEUDOS,
        xc=xc or XC,
        kpts=kpts or (KPTS_GAS if is_gas else KPTS_SLAB),
        smearing="none" if is_gas else SMEARING,
        width=WIDTH,
        device=device or DEVICE,
        use_symmetry=True,
        precond="local_tf",
        mixing_scheme="johnson",
        etol=1e-6 if metal else 1e-8,      # adsorption energies need ~10 meV
        rhotol=1e-5 if metal else 1e-7,    # metals floor the density residual
        **kw,
    )
