# ESM open_z residual vacuum field — localization (diagnostic only, no fix)

Component decomposition of the in-plane-averaged vacuum potential on a neutral
Al slab under `boundary="open_z"`, pinned `dz=0.15 Å` (identical grid across all
boxes), Al ONCV 3e, ecut=20 Ry, LDA. Scripts:
`scripts/diag_esm_residual_field.py` (symmetric, Lz scan) and
`scripts/diag_esm_offcenter.py` (box-position scan). Slopes in meV/Å; sign test =
same-sign on both vacuum faces → uniform field (monopole/G∥=0), opposite-sign →
symmetric/dipole.

## Verdict

The residual field is the **G∥=0 channel of `esm_delta_potential`** (the
open-minus-periodic *correction* the SCF adds), NOT the ESM-core open Hartree
`hartree_potential_esm`, NOT the gaussian-ion neutralization. The correction
develops a **spurious uniform vacuum field whose magnitude scales linearly with
the slab's offset from the box center** — a translation-covariance failure. In
exact ESM the vacuum field must not depend on where a fixed neutral slab sits in
the box; here it does.

## Evidence

### 1. `rho_tot = rho_elec − gaussian_ion_density` is CLEAN
Net charge ~1e-12 e, dipole ∫z·ρ ~1e-3 e·Å (≈0), symmetric. Vacuum slope
0.06–0.4 meV/Å. Neutralization / gaussian-ion width is NOT the culprit.

### 2. `hartree_potential_esm(rho_tot)` (ESM core) is field-free & position-stable
Deep-vacuum values <0.1 eV, flat. Face slopes stay OPPOSITE-sign (symmetric) and
do **not** grow into a uniform field as the slab is translated:
shift 0/+2/+4 Å → left −25.3/−25.9/−33.3, right +22.3/+26.7/+25.6 meV/Å.
The ESM-core G∥=0 open-Coulomb kernel is behaving correctly.

### 3. `esm_delta_potential` G∥=0 → a UNIFORM field ∝ slab offset (THE BUG)
The open-minus-periodic correction that actually enters `v_eff` shows an
identical-on-both-faces (SAME-sign) tilt that scales linearly with offset:

| slab offset from box center | dV_esm L-face | dV_esm R-face |
|---|---|---|
| 0 Å   | −0.88 | −0.88 |
| +2 Å  | −22.9 | −22.9 |
| +4 Å  | −50.5 | −50.5 |

A fixed neutral slab merely shifted in the box must give an unchanged vacuum
field; instead a uniform field appears and grows. It propagates straight into
`v_eff`, whose right-face slope hits **+402 meV/Å at +4 Å offset** (matching the
prior probe's ~343 meV/Å report), turning SAME-sign as `dV_esm` dominates.

### Lz-scaling (symmetric slab, centered)
`dV_esm` uniform tilt decreases with Lz (0.88 → 0.68 meV/Å over Lz 24 → 30 Å) —
the residual is offset-driven, not Lz-driven, and the symmetric case is small.
The large drift is excited by asymmetry/off-centering, which real slabs (NaH
dipole, asymmetric Al column) always have.

## Root cause pointer (for the fix, not done here)

`esm_delta_potential` G∥=0 channel (esm.py ~L306–317):

```python
u = zc[:, None] - zc[None, :]
u_wrap = u - lz * torch.round(u / lz)          # minimum-image
k0 = (-0.5 * u.abs()) - (-0.5 * u_wrap.abs() + u_wrap * u_wrap / (2.0 * lz))
dv00 = 4.0 * math.pi * E2 * dz * (k0 @ s)      # s = rho_tot(G∥=0), NOT neutralized here
```

The open part −½|u| is translation-invariant; the subtracted *periodic* reference
`−½|u_wrap| + u_wrap²/(2L)` uses `round(u/L)` minimum-imaging, whose seam flips as
the charge straddles the box boundary → the difference kernel stops being a pure
function of (z−z'), so its convolution with `s` acquires a slab-position-dependent
linear (uniform-field) term. Contrast `hartree_potential_esm`, which neutralizes
`s` (`s -= s.mean()`) and uses only the open |z−z'| kernel — and stays clean. The
fix target is making the periodic G∥=0 reference in `esm_delta_potential` match
the spectral periodic Hartree (`hartree_potential_r`) consistently and
translation-covariantly (and/or applying the same background neutralization),
so the open-minus-periodic difference has no residual uniform field.
