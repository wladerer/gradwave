# Cu anomaly — the PseudoDojo standard UPF is defective, not gradwave

Cu is the one element whose Δ vs WIEN2k is large (7.89 meV/atom, vs a median of
0.8 for the set). Two cross-checks isolate the cause to the pseudopotential
*file*, not gradwave.

## 1. gradwave reproduces Quantum ESPRESSO to 0.08 meV with the same pseudo

Matched settings (PseudoDojo `Cu.upf`, 104 Ry, 35³ grid pinned, 16³ k, gaussian
0.01 Ry), fcc Cu at v = 0.94 / 1.00 / 1.06:

| quantity | QE (pw.x) | gradwave | AE (WIEN2k) |
|---|---|---|---|
| B0 (3-point central) | 166.9 GPa | 167.3 GPa | 141.3 GPa |
| ΔE(0.94) vs eq. | +38.8 meV | +38.9 meV | — |
| ΔE(1.06) vs eq. | +6.1 meV | +6.1 meV | — |

gradwave matches QE on the EOS curvature to **0.08 meV** (constant −2.92 meV
absolute offset, a pseudo reference convention that cancels in the EOS). Both
codes land ~16–26 GPa *above* the all-electron B0. The implementation is exact;
the pseudopotential is what disagrees with all-electron.

## 2. SG15 Cu, same harness, is correct

Swapping only the pseudopotential to SG15 `Cu_ONCV_PBE-1.2` (same code, cutoff,
mesh, smearing) gives V0 = 11.994 (AE 11.951), B0 = 139.1 (AE 141.3), B1 = 5.10,
**Δ = 1.33 meV/atom** — normal. So the fault is specific to the PseudoDojo
`Cu.upf` from the `nc-sr-04_pbe_standard_upf` tarball.

## 3. It is not a version bug — v0.4 and v0.5 are byte-identical

The `Cu.upf` in the v0.4 and v0.5 standard-UPF tarballs have the same md5
(`5fe87f3b4a9befbfed2d60ad3362dad9`), so the discrepancy is persistent across
PseudoDojo releases, not a stale-conversion artifact fixed downstream.

## Conclusion

PseudoDojo's published Δ-factor for Cu is 0.53 meV/atom, computed on the psp8
`Cu-sp` pseudo. The UPF in the standard-UPF tarball (identical in v0.4 and v0.5)
reproduces neither that value nor the all-electron EOS, and it fails identically
in QE and gradwave, so the psp8→UPF conversion (or the packaged variant) is
inconsistent with the psp8. Reported upstream candidate: the UPF for Cu.
This is the two-axis validation working as intended (see
`docs/manual/wisdom.md`, "Use two comparison axes"): Cu is bad on the
all-electron axis (pseudization) and exact on the QE axis (implementation).
For a corrected Cu, use the SG15 pseudo or PseudoDojo's psp8 (via a trusted
UPF conversion).
