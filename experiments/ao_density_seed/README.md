# Atomic-orbital seeding for the starting density

Research-notes register. This revisits the seeding no-go recorded in
`docs/ideas.md` ("Atomic-orbital seeding for the initial wavefunctions"), which
covered the initial **wavefunctions** (the Davidson `c0`). The object here is
different, namely the starting **density** rho0 that the first iteration's
potential is built from.

## What the current density seed actually is

The starting density is already a superposition of atomic densities (SAD) built
from the UPF `PP_RHOATOM` table, not a cruder guess. Both formalisms share one
builder.

- `scf/guess.py::sad_density` assembles rho0 from `pseudo/atomic.py::rhoatom_of_q`
  (the l=0 transform of `PP_RHOATOM`), placed by structure factor and rescaled so
  `Omega * rho(G=0) = N_e` exactly.
- NC path wires it at `scf/loop.py::_seed_density` (line ~545), called from
  `scf/loop.py::scf` at line ~1193.
- USPP/PAW path wires it at `scf/uspp_loop.py::_seed_scf_density` (line ~1131),
  called at line ~1374.
- The UPF parsers read `PP_RHOATOM` (`pseudo/upf.py:291`, `pseudo/upf_paw.py:138`)
  and the `PP_PSWFC` orbitals with their reference occupations
  (`pseudo/upf.py::AtomicOrbital`, shared parser `_parse_pswfc_chi`).

For `nspin=2` the total SAD is split into up/down by a uniform per-atom factor
`(1 +/- m)/2` (`_seed_density` lines ~587-606, `_seed_scf_density` lines
~1161-1178). The seeded magnetization density is therefore
`m(r) = m_atom * rho_atom(r)`, proportional to the full atomic valence density
including the diffuse s/p tail. The per-atom seeded moment is `m_atom * Z_val`.

Consequences for the three candidate improvements:

- (a) PP_RHOATOM superposition where the seed is cruder. Moot. The seed is
  already SAD from `PP_RHOATOM`.
- (c) normalization/Z-consistency audit. `sad_density` rescales `rho(G=0)` to
  `N_e` and re-normalizes after the positive floor. Verified below.
- (b) spin-resolved seeding. The only live lever. The uniform split puts
  magnetization proportional to the total density rather than into the d shell
  where the physical spin density of a 3d metal sits.

## The change under test (experiment only, no default touched)

`experiments/ao_density_seed/seed.py::d_localized_spin_densities` keeps the total
density as the default SAD and shapes the magnetization by the `PP_PSWFC` d-orbital
density `|R_3d(r)|^2`, integrated per atom to the SAME `m_atom * Z_val` the default
seeds. Only the magnetization SHAPE differs. It is installed by monkeypatching
(`patch.py`), never by changing a default. `nspin=1` is untouched, so the change
affects only spin-polarized runs.

## Measurements

(seed-construction cost, nonmagnetic anchor, magnetic branch selection tables
filled from the runs below.)

## Verdict

(go/no-go with evidence.)
