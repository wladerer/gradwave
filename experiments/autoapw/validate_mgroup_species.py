"""Validate the injected atomic species (B, N, Na, Li, Al) vs NIST-LSD eigenvalues before use.

gradwave ``atomic_scf`` is scalar-nonrelativistic LDA; the valence levels should match NIST-LSD to
~1-2 %, the deep 1s to within the expected relativistic shift. Run on asus. NO src change.
"""
import _mgroup  # noqa: F401  (injects the species at import)

from gradwave.flapw.atom import NIST_LDA_EV, atomic_scf
from gradwave.flapw.radial import log_mesh

r, dx = log_mesh(1e-5, 28.0, 2500)
print("species validation (gradwave atomic_scf LDA vs NIST-LSD ref, eV)\n")
for sym in ("Li", "B", "N", "Na", "Al"):
    eigs, _ = atomic_scf(sym, r, dx)
    ref = NIST_LDA_EV[sym]
    print(f"  {sym}:")
    for lvl, e in eigs.items():
        rv = ref.get(lvl)
        if rv is None:
            print(f"    {lvl:>4} {e:+10.3f}")
            continue
        rel = abs(e - rv) / abs(rv) * 100
        flag = "  <-- valence" if not lvl.startswith("1") else ""
        print(f"    {lvl:>4} {e:+9.3f} ref {rv:+9.3f} |Δ|={abs(e-rv):6.3f} ({rel:4.1f}%){flag}")
    print(flush=True)
