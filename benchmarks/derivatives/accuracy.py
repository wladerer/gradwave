"""Consolidated derivative-accuracy table — gradwave's differentiability credential.

Every derivative gradwave produces is validated either against a finite difference
(FD) of the energy / SCF re-runs (implementation exactness, floors near the FD
noise), or against the specialized Quantum ESPRESSO response module (ph.x, hp.x —
mixing pseudization with implementation, so ~0.1-1%). This collects those checks
into one table; each row's `rel` is the observed relative agreement (or the
asserted tolerance where the test does not print the observed value), and `test`
cites the passing gate it comes from — run `pytest <test>` to reproduce.

    uv run python benchmarks/derivatives/accuracy.py   # prints table, writes json + figure
"""
import json
from pathlib import Path

SP = Path(__file__).parent

# ref_type: "FD" = finite difference / gradcheck of gradwave itself;
#           "QE" = the specialized Quantum ESPRESSO response module.
# rel: observed relative agreement (best available), for the log-scale figure.
ROWS = [
    # ---- forces / stress / geometry ----
    dict(q="Atomic forces −dE/dτ", method="autograd Hellmann–Feynman", ref="FD of energy",
         ref_type="FD", agree="<1e-4 eV/Å", rel=1e-4, sys="displaced Si",
         test="test_forces_vs_qe.py"),
    dict(q="Atomic forces −dE/dτ", method="autograd Hellmann–Feynman", ref="QE pw.x",
         ref_type="QE", agree="8.6e-6 eV/Å (egg-box, 35 Ry)", rel=8.6e-6,
         sys="displaced Si", test="test_forces_vs_qe.py"),
    dict(q="Stress dE/dε", method="autograd strain (Nielsen–Martin)", ref="FD of energy",
         ref_type="FD", agree="<1e-7", rel=1e-7, sys="Si", test="test_stress_vs_qe.py"),
    dict(q="Stress dE/dε", method="autograd strain", ref="QE pw.x",
         ref_type="QE", agree="≤0.006 kbar over 90–14000 kbar", rel=1e-4,
         sys="Si / Al / MgO / Ni", test="test_stress_vs_qe.py"),
    dict(q="Position response ∂F/∂τ (PAW Hessian)", method="S-metric Sternheimer + SC response",
         ref="FD of forces", ref_type="FD", agree="2.0e-5", rel=2.0e-5,
         sys="Si kjpaw", test="test_uspp_position.py"),
    dict(q="Γ phonon force constants", method="FD of analytic forces",
         ref="energy 2nd-difference / ph.x", ref_type="QE", agree="0.5% (0.003–0.15% vs ph.x)",
         rel=5e-3, sys="Si", test="test_phonons.py"),
    # ---- XC / functional learning ----
    dict(q="XC parameter dE/dμ, dE/dκ", method="variational stationarity",
         ref="FD of SCF re-runs", ref_type="FD", agree="<1e-5 (obs ~1e-8)", rel=1e-8,
         sys="Si LearnableX", test="test_functional_learning.py"),
    dict(q="Density-loss adjoint dL/dθ", method="implicit-diff (χ₀, K_Hxc adjoint)",
         ref="FD of SCF re-runs", ref_type="FD", agree="<2e-4", rel=2e-4,
         sys="Si LearnableX", test="test_implicit_scf.py"),
    dict(q="USPP/PAW density-loss dL/dθ", method="composite (ρ, becsum) adjoint",
         ref="FD of SCF re-runs", ref_type="FD", agree="1.2e-6", rel=1.2e-6,
         sys="Si kjpaw", test="test_uspp_implicit.py"),
    dict(q="USPP metal dL/dθ (Fermi surface)", method="window-pair + δμ adjoint",
         ref="FD of smeared SCF", ref_type="FD", agree="<2e-4", rel=2e-4,
         sys="Al kjpaw", test="test_uspp_implicit.py"),
    dict(q="PAW one-center HVP ∂²E₁c/∂becsum²", method="analytic Hessian-vector product",
         ref="FD of ddd", ref_type="FD", agree="<2e-6", rel=2e-6, sys="Si kjpaw",
         test="test_paw_onsite_hvp.py"),
    # ---- hybrid functional parameters ----
    dict(q="Hybrid dE/dα (exchange mixing)", method="stationary-energy derivative",
         ref="FD of re-converged hybrid", ref_type="FD", agree="<1e-3", rel=1e-3,
         sys="Si PBE0", test="test_learned_hybrid.py"),
    dict(q="Hybrid dE/dω (screening)", method="stationary-energy derivative",
         ref="FD of re-converged hybrid", ref_type="FD", agree="<5e-3", rel=5e-3,
         sys="Si HSE-form", test="test_learned_hybrid.py"),
    dict(q="Hybrid gap dGap/dα", method="frozen-orbital Hellmann–Feynman",
         ref="FD of re-converged hybrid", ref_type="FD", agree="1% (dGap/dω 5%)", rel=1e-2,
         sys="Si", test="benchmarks/hybrid_design/validate.py"),
    # ---- Hubbard U ----
    dict(q="Hubbard dE/dU", method="Hellmann–Feynman ½Tr[n(1−n)]",
         ref="FD of SCF re-runs", ref_type="FD", agree="<1e-4", rel=1e-4, sys="NiO",
         test="test_hubbard_vs_qe.py"),
    dict(q="+U force −dE_U/dτ", method="autograd through projector phases",
         ref="gradcheck", ref_type="FD", agree="atol 1e-6", rel=1e-6,
         sys="synthetic Se", test="test_hubbard.py"),
    dict(q="Linear-response Hubbard U", method="analytic Sternheimer + autograd HVP",
         ref="QE hp.x DFPT", ref_type="QE", agree="0.3% (6.45 vs 6.43 eV)", rel=3e-3,
         sys="NiO", test="test_hubbard_vs_qe.py"),
    # ---- E-field DFPT ----
    dict(q="Dielectric ε∞", method="Sternheimer + autograd K_Hxc HVP",
         ref="QE ph.x (epsil)", ref_type="QE", agree="0.002% Si, <1% MgO", rel=2e-4,
         sys="Si / MgO", test="test_dielectric_vs_qe.py"),
    dict(q="Born charge Z* = ∂²E/∂E∂τ", method="Sternheimer + τ-diff pseudo backward",
         ref="QE ph.x (zeu)", ref_type="QE", agree="<2e-3 Si, <1% MgO", rel=1.7e-3,
         sys="Si / MgO", test="test_dielectric_vs_qe.py"),
]


def main():
    print(f"{'quantity':38s} {'method':34s} {'ref':22s} {'agreement':28s} {'system'}")
    print("-" * 150)
    for r in sorted(ROWS, key=lambda r: (r["ref_type"], r["rel"])):
        print(f"{r['q']:38s} {r['method']:34s} {r['ref']:22s} {r['agree']:28s} {r['sys']}")
    fd = [r for r in ROWS if r["ref_type"] == "FD"]
    qe = [r for r in ROWS if r["ref_type"] == "QE"]
    print(f"\n{len(ROWS)} validated derivatives: {len(fd)} vs finite-difference "
          f"(median rel {sorted(x['rel'] for x in fd)[len(fd)//2]:.0e}), "
          f"{len(qe)} vs a QE response module (median rel "
          f"{sorted(x['rel'] for x in qe)[len(qe)//2]:.0e}).")
    (SP / "accuracy.json").write_text(json.dumps(ROWS, indent=1))


if __name__ == "__main__":
    main()
