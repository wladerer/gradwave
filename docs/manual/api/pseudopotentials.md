# Pseudopotentials

gradwave reads UPF v2 files. `parse_upf` handles norm-conserving datasets;
`parse_upf_paw` handles ultrasoft and PAW. The high-level API and the ASE
calculator detect the family from the file, so you rarely call these directly,
but the parsed data objects are what the rest of the code consumes.

Only norm-conserving UPF v2 (ONCV, SG15/PseudoDojo) and ultrasoft/PAW are
supported. Mixing the two families in one calculation is rejected.

## Norm-conserving

::: gradwave.pseudo.upf.parse_upf

::: gradwave.pseudo.upf.UPFData

::: gradwave.pseudo.upf.BetaProjector

::: gradwave.pseudo.upf.AtomicOrbital

## Ultrasoft / PAW

::: gradwave.pseudo.upf_paw.parse_upf_paw

::: gradwave.pseudo.upf_paw.PAWData

::: gradwave.pseudo.upf_paw.PartialWave
