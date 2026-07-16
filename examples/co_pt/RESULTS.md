# CO on Pt(111), learned-U study — results and status

## Status

PARKED after the bulk EOS, in favor of a performance-acceleration sprint (the
per-SCF cost made the slab campaign impractical, see `docs/manual/performance.md`
"a hard PAW metal vs QE"). Resume the slab and binding-energy work
(`slab_co.py`) once the accelerations land.

## Bulk fcc Pt EOS (done)

PBE, PAW (`Pt.pbe-n-kjpaw_psl.1.0.0.UPF`), 40/400 Ry, 12x12x12, gaussian 0.2 eV,
Birch-Murnaghan fit over a in [3.82, 4.04] Å (`bulk_pt_eos.py`, on asus CPU).

| quantity | gradwave | reference |
|---|---|---|
| a0 | 3.9776 Å | expt 3.92, PBE literature ~3.97 |
| B0 | 247.9 GPa | expt 278, PBE literature ~250 |

The a0 and B0 match PBE expectations (PBE overestimates the lattice constant and
softens the bulk modulus for late transition metals). This a0 seeds the slab.

Full curve in `bulk_pt_eos.json`. Ran in 981 s on 16 CPU threads (~140 s/point);
the same run on the RTX 3050 was ~975 s per point, ~7.8x slower, hence the CPU
switch and the acceleration sprint.

## Parked deliverables

- Pt(111) slab from a0, rigid-substrate CO binding energy at ontop and fcc
  (`slab_co.py`, ready, symmetry-tuned to 7 irreducible k on the rhombic cell).
- CO 2pi* PDOS as the site diagnostic.
- Learned / linear-response Hubbard U on the CO 2pi* (needs a USPP `hub_alpha`
  linear-response path, currently NC-only in `postscf/hubbard_u.py`).
- C-O stretch frequency per site (Gamma-Hessian, works on the metal).
