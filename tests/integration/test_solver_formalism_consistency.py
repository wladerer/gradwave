"""SCF-level consistency guards over two large, previously untested surfaces.

Test 1 — CheFSI reaches the SAME fixed point as Davidson through the REAL
``scf.loop.scf`` loop. The ``eigensolver="auto"/"chebyshev"`` path (size-gated
CheFSI, #432) was covered only by (a) pure gate LOGIC with a monkeypatched
threshold and (b) CheFSI==Davidson on a SYNTHETIC dense operator. Neither runs
the production loop, so a broken loop→chebyshev wiring or a smearing/nspin
interaction was invisible. Here we run a small smeared metal (fcc-Al) and an
insulator (diamond Si) through the loop twice — once pinned to Davidson, once to
CheFSI — and assert the converged free energy AND the converged density agree to
the diagonalizer floor (they solve the same fixed H each SCF step). We also spy
the solver registry to prove CheFSI actually ran and did NOT silently fall back
to Davidson.

Test 2 — cross-formalism NC vs PAW energy consistency. Absolute energies carry
different references (NC ~ -107, PAW ~ -636 eV/atom for Si), so they must NOT be
compared directly. Instead we compare a DIFFERENCE that is reference-free: the
equilibrium lattice constant a0 (and bulk modulus B0) from a small volume sweep
with each formalism. NC (SG15 ONCV) and PAW (psl kjpaw) both approximate the same
all-electron PBE solid, so a0 must agree to the pseudopotential-difference level.
A formalism-specific energy-reference or augmentation/Pulay bug would show up as
a gross a0/B0 disagreement.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

import gradwave.solvers.registry as _registry
from gradwave.api import run_eos
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.inputs import EOSParams, Input, KPointsParams, SmearingParams
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])

# fcc-Al: a smeared metal (Gaussian smearing), one atom. The metallic occupation
# response is the interaction most likely to differ between solvers.
AL_CELL = 4.05 / 2 * FCC
AL_POS = np.zeros((1, 3))
SI_CELL, SI_POS = si_fcc()

# case -> (pseudo, xc, smearing, ecut[Ry], kmesh, nbands)
_T1_CASES = {
    "al_metal": ("Al_ONCV_PBE-1.2.upf", PBE, "gaussian", 20, (4, 4, 4), 10),
    "si_insulator": ("Si_ONCV_PBE-1.2.upf", LDA_PW92, "none", 15, (2, 2, 2), None),
}
_T1_GEOM = {"al_metal": (AL_CELL, AL_POS), "si_insulator": (SI_CELL, SI_POS)}


class _SolverSpy:
    """Wrap a registry entry to count how many times the loop dispatched to it.

    Restores the original entry on exit so the global registry is left clean."""

    def __init__(self, name):
        self.name = name
        self.count = 0
        self._orig = _registry._REGISTRY[name]

    def __enter__(self):
        spy = self

        def wrapped(*a, **k):
            spy.count += 1
            return spy._orig(*a, **k)

        _registry._REGISTRY[self.name] = wrapped
        return self

    def __exit__(self, *exc):
        _registry._REGISTRY[self.name] = self._orig
        return False


@pytest.mark.standard
@pytest.mark.parametrize("case", list(_T1_CASES))
def test_chebyshev_matches_davidson_through_scf_loop(case):
    """CheFSI and Davidson must reach the same SCF fixed point through the real
    loop: equal converged free energy (<1e-6 eV/atom) and equal converged
    density (<1e-8 e/Å³ pointwise). Also proves CheFSI ran and did not silently
    fall back to Davidson."""
    torch.set_num_threads(8)
    pseudo, xc, smear, ecut_ry, kmesh, nbands = _T1_CASES[case]
    cell, pos = _T1_GEOM[case]
    upf = parse_upf(FIX / "pseudos" / pseudo)
    nat = pos.shape[0]

    def solve(solver):
        system = setup_system(cell, pos, [0] * nat, [upf], ecut=ecut_ry * RY,
                              kmesh=kmesh, nbands=nbands)
        # Spy BOTH solver entries so we can prove the requested one ran and the
        # other was never dispatched (the "silent fallback to davidson" guard).
        with _SolverSpy("chebyshev") as che, _SolverSpy("davidson") as dav:
            res = scf(system, xc(), smearing=smear, width=0.1, etol=1e-10,
                      rhotol=1e-9, verbose=False, eigensolver=solver)
        return res, che.count, dav.count

    r_dav, che_in_dav, dav_in_dav = solve("davidson")
    r_che, che_in_che, dav_in_che = solve("chebyshev")

    assert r_dav.converged, f"{case}: davidson SCF did not converge"
    assert r_che.converged, f"{case}: chebyshev SCF did not converge"

    # CheFSI actually ran and Davidson was NOT used as a fallback in that run.
    assert che_in_che > 0, f"{case}: chebyshev never dispatched (silent fallback?)"
    assert dav_in_che == 0, (
        f"{case}: davidson was dispatched {dav_in_che}x during a chebyshev run "
        "(silent fallback to davidson)")
    # sanity: the davidson run used davidson and not chebyshev
    assert dav_in_dav > 0 and che_in_dav == 0

    # same fixed point: free energy to the diagonalizer floor
    e_dav = float(r_dav.energies.free_energy)
    e_che = float(r_che.energies.free_energy)
    assert abs(e_dav - e_che) / nat < 1e-6, (
        f"{case}: davidson {e_dav:.10f} vs chebyshev {e_che:.10f} eV")

    # same fixed point: the converged TOTAL densities agree pointwise. Both were
    # driven to rhotol=1e-9 on the same H, so they coincide to the diagonalizer
    # floor (measured ~9e-12 e/Å³ on fcc-Al); 1e-8 is a robust guard.
    maxabs = float((r_dav.rho - r_che.rho).abs().max())
    assert maxabs < 1e-8, f"{case}: density max|Δρ| = {maxabs:.3e} e/Å³"


# ---------------------------------------------------------------------------
# Test 2 — NC vs PAW equilibrium lattice constant / bulk modulus of Si
# ---------------------------------------------------------------------------

# Measured on asus (40 Ry, 6³, 5-point sweep about a=5.47 Å):
#   NC  (Si_ONCV_PBE-1.2.upf):        a0 = 5.4797 Å, B0 = 88.16 GPa, B0' = 3.82
#   PAW (Si.pbe-n-kjpaw_psl.1.0.0):   a0 = 5.4708 Å, B0 = 89.39 GPa, B0' = 4.02
# ⇒ |Δa0| = 8.9e-3 Å, ΔB0 = 1.4%. This is the genuine SG15-vs-psl pseudopotential
# difference (both approximate all-electron PBE Si), NOT under-convergence — it
# does not shrink with cutoff. The gates below sit at 1.5e-2 Å / 5% (same order,
# with robustness margin), so they still flag a gross formalism bug (a broken
# energy reference or augmentation/Pulay term shows up as ≫1e-2 Å / tens of %),
# not this expected sub-percent pseudo spread.
_SI_A = 5.47
_SI_SCALES = (0.96, 0.98, 1.00, 1.02, 1.04)


def _si_a0_b0(pseudo):
    """Equilibrium fcc conventional-cell a0 [Å] and B0 [GPa] of diamond Si from a
    5-point BM3 EOS with the given pseudo (formalism auto-dispatched by UPF kind)."""
    cell = _SI_A / 2 * FCC
    pos = np.array([[0.0, 0, 0], [_SI_A / 4] * 3])
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Si": pseudo}, ecut=40 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(6, 6, 6)),
        smearing=SmearingParams(type="none"),
        eos=EOSParams(scales=_SI_SCALES))
    r = run_eos(inp, verbose=False)
    assert r["all_converged"], f"{pseudo}: not all EOS points converged"
    # V0 is per atom; the diamond conventional cell holds 8 atoms.
    a0 = (r["v0_ang3_per_atom"] * 8) ** (1 / 3)
    return a0, r["b0_GPa"], r


@pytest.mark.slow
def test_nc_vs_paw_equilibrium_lattice_and_bulk_modulus():
    """NC (ONCV) and PAW (kjpaw) Si must agree on a reference-free observable —
    the equilibrium lattice constant and bulk modulus — even though their
    absolute energies use different references. Disagreement ≫1e-2 Å / tens of %
    would indicate a formalism-specific energy-reference or augmentation bug."""
    torch.set_num_threads(8)
    a0_nc, b0_nc, _ = _si_a0_b0("Si_ONCV_PBE-1.2.upf")
    a0_paw, b0_paw, _ = _si_a0_b0("Si.pbe-n-kjpaw_psl.1.0.0.UPF")

    da0 = abs(a0_nc - a0_paw)
    assert da0 < 1.5e-2, (
        f"NC a0={a0_nc:.5f} Å vs PAW a0={a0_paw:.5f} Å → |Δa0|={da0:.4e} Å "
        "(cross-formalism a0 disagreement — possible formalism bug)")

    rel_b0 = abs(b0_nc - b0_paw) / (0.5 * (b0_nc + b0_paw))
    assert rel_b0 < 0.05, (
        f"NC B0={b0_nc:.2f} GPa vs PAW B0={b0_paw:.2f} GPa → {rel_b0 * 100:.2f}% "
        "(cross-formalism bulk-modulus disagreement)")
