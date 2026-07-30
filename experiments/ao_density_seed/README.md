# Atomic-orbital seeding for the starting density

Research notes, 2026-07-29/30. This revisits the seeding no-go recorded in
`docs/ideas.md` ("Atomic-orbital seeding for the initial wavefunctions"), which
covered the initial wavefunctions (the Davidson `c0`). The object here is
different, namely the starting density rho0 that the first iteration's
potential is built from.

## What the current density seed actually is

The starting density is already a superposition of atomic densities (SAD) built
from the UPF `PP_RHOATOM` table, not a cruder guess, and both formalisms share
one builder.

- `scf/guess.py::sad_density` assembles rho0 from
  `pseudo/atomic.py::rhoatom_of_q` (the l=0 spherical Bessel transform of
  `PP_RHOATOM`), placed by structure factor, masked to the density sphere, and
  rescaled so `Omega * rho(G=0) = N_e` exactly. After the positive floor it
  re-normalizes, so the electron count is exact by construction.
- The NC path wires it at `scf/loop.py::_seed_density` (line 545), called from
  `scf/loop.py::scf` at line 1193.
- The USPP/PAW path wires it at `scf/uspp_loop.py::_seed_scf_density`
  (line 1131), called at line 1374.
- The parsers read `PP_RHOATOM` at `pseudo/upf.py:291` and
  `pseudo/upf_paw.py:138`, and the `PP_PSWFC` orbitals with reference
  occupations via the shared `_parse_pswfc_chi` (`pseudo/upf.py:224`).

For nspin=2 the total SAD is split into up and down by a uniform per-atom
factor `(1 +/- m)/2` (`_seed_density` lines 587-606, `_seed_scf_density` lines
1161-1178). The seeded magnetization density is therefore
`m(r) = m_atom * rho_atom(r)`, proportional to the full atomic valence density
including the diffuse s and p tail. The per-atom seeded moment is
`m_atom * Z_val`.

That reading disposes of two of the three candidate improvements immediately.
Candidate (a), seed from PP_RHOATOM where the current seed is cruder, is moot
because the seed is already SAD. Candidate (c), a normalization audit, checks
out clean. A direct probe (`probe_pswfc.py` plus a smoke check of the builder)
confirms N = 18.0000 exactly on fcc Ni and a seeded moment of exactly
`m * Z_val`, and the flat-seed anchor below shows the SAD seed behaving as a
good seed should. Candidate (b), spin-resolved shaping of the magnetization, is
the only live lever, and it is what this experiment measures.

## The change under test

`seed.py::d_localized_spin_densities` keeps the total density as the default
SAD and shapes the magnetization by the `PP_PSWFC` d-orbital density
`|R_3d(r)|^2`, integrated per atom to the same `m_atom * Z_val` the default
seeds. Only the magnetization shape differs. It is installed by monkeypatching
(`patch.py`), and no default is changed anywhere. nspin=1 is untouched by
construction.

Pseudo availability constrains the matrix. SG15 ONCV datasets (Si, Al, Mg, O,
Fe_ONCV) carry no `PP_PSWFC`, so the NC magnetic system uses the PseudoDojo
`PD_Ni_PBE.upf` (3D occ 8) and the PAW systems use the psl kjpaw Fe and Ni
(3D occ 6 and 8).

## Setup

All comparisons run at identical settings, seeds differing only in rho0.
Magnetic runs use SpinPBE, gaussian smearing width 0.1 eV, etol 1e-6, rhotol
1e-5, mixing_alpha 0.3, max_iter 120. Systems are fcc Ni NC (PD_Ni, ecut 45 Ry,
6x6x6), fcc Ni PAW (psl kjpaw, ecut 50/400 Ry, 4x4x4), and bcc Fe PAW (psl
kjpaw, ecut 50/400 Ry, 6x6x6). start_mag sweeps 0.02, 0.05, 0.1, 0.3, where
0.02 seeds 0.36 muB on Ni and 0.32 muB on Fe (deliberately marginal against the
Stoner boundary). Full sweep on asus at 7 threads (`results_asus.jsonl`), with
a 4-row local cross-check (`results_local_crosscheck.jsonl`) agreeing on
iteration counts and on F to 1e-11 eV.

## Results

### Nonmagnetic anchor (nspin=1, SAD vs deliberately flat seed)

| system | seed | iters | iter-1 residual | F (eV) |
|---|---|---|---|---|
| Si fcc | SAD (current) | 7 | 1.87 | -214.30224330245 |
| Si fcc | flat | 8 | 7.39 | -214.30224330245 |
| MgO | SAD (current) | 8 | 2.64 | -1914.25467779896 |
| MgO | flat | 11 | 48.9 | -1914.25467779897 |

The current SAD seed buys 1 to 3 iterations over a flat start and cuts the
iteration-1 residual by 4x to 18x, and both seeds reach identical energies.
The existing seed is healthy, and there is no headroom to claim on the charge
channel.

### Magnetic branch selection (nspin=2, uniform split vs d-localized)

n_iter marked * means not converged at max_iter 120. Branch is the converged
magnetic state, with FM at 0.592 muB (Ni) or 2.222 muB (Fe) and NM at ~0.

| system | start_mag | seed | iters | branch | iter-1 res | F (eV) |
|---|---|---|---|---|---|---|
| Ni NC | 0.02 | default | 23 | NM | 6.76 | -4527.56080937101 |
| Ni NC | 0.02 | d-loc | 23 | NM | 6.75 | -4527.56080937102 |
| Ni NC | 0.05 | default | 13 | FM | 6.85 | -4527.57751056814 |
| Ni NC | 0.05 | d-loc | 12 | FM | 6.77 | -4527.57751056815 |
| Ni NC | 0.10 | default | 14 | FM | 7.36 | -4527.57751056813 |
| Ni NC | 0.10 | d-loc | 14 | FM | 6.98 | -4527.57751056815 |
| Ni NC | 0.30 | default | 15 | FM | 13.0 | -4527.57751056816 |
| Ni NC | 0.30 | d-loc | 16 | FM | 11.4 | -4527.57751056814 |
| Ni PAW | 0.02 | default | 120* | FM vicinity | 1.15 | -5838.39832287809 |
| Ni PAW | 0.02 | d-loc | 76 | FM | 1.09 | -5838.39832294044 |
| Ni PAW | 0.05 | default | 120* | FM vicinity | 0.89 | -5838.39832477632 |
| Ni PAW | 0.05 | d-loc | 28 | FM | 1.50 | -5838.39832294045 |
| Ni PAW | 0.10 | default | 120* | NM (wrong, +77 meV) | 2.50 | -5838.32137126611 |
| Ni PAW | 0.10 | d-loc | 74 | FM | 1.55 | -5838.39832294044 |
| Ni PAW | 0.30 | default | 120* | FM vicinity | 10.1 | -5838.39836151983 |
| Ni PAW | 0.30 | d-loc | 120* | FM vicinity | 4.26 | -5838.39832274682 |
| Fe PAW | 0.02 | default | 26 | FM | 1.34 | -4479.68781870330 |
| Fe PAW | 0.02 | d-loc | 32 | NM (wrong, +660 meV) | 1.23 | -4479.02748963244 |
| Fe PAW | 0.05 | default | 20 | FM | 2.26 | -4479.68781870330 |
| Fe PAW | 0.05 | d-loc | 26 | FM | 1.93 | -4479.68781870331 |
| Fe PAW | 0.10 | default | 18 | FM | 2.02 | -4479.68781870330 |
| Fe PAW | 0.10 | d-loc | 20 | FM | 3.11 | -4479.68781870331 |
| Fe PAW | 0.30 | default | 21 | FM | 5.03 | -4479.68781870331 |
| Fe PAW | 0.30 | d-loc | 17 | FM | 3.93 | -4479.68781870331 |

The fixed-point oracle holds. Wherever both seeds converge to the same branch,
F agrees to 2e-11 eV or better, so the seed changes only the trajectory.

### Seed-construction cost

Measured on the same grids as the runs (2 threads, mean of 20 builds). NC Ni
21^3 grid, default split 115 ms vs d-localized 125 ms. PAW Ni 30^3 grid,
default 372 ms vs d-localized 328 ms. The d-localized build is cost-neutral
because it builds one SAD plus one cheap magnetization term where the default
builds two SADs. Against per-iteration costs of 10 to 30 s in these systems,
seed construction is negligible either way, so the comparison is decided
entirely by iterations and branch outcomes.

## Reading the table

Three distinct behaviors, one per system.

- Ni NC is neutral. The d-localized seed lowers the iteration-1 residual
  slightly on every point but saves at most one iteration, and it does not
  rescue the 0.02 collapse to NM. Both seeds fall into the same basin.
- Ni PAW is a systematic win, on three independent marginal points. The
  default seed failed rhotol 1e-5 on all four start_mag values, stagnating
  near the FM fixed point at 0.02, 0.05, and 0.30 and landing on the wrong NM
  branch at 0.10. The d-localized seed converged at 0.02, 0.05, and 0.10 (76,
  28, 74 iterations), always to the FM branch. The in-tree Ni PAW test
  (`test_uspp_vs_qe.py::test_ni_paw_spin_vs_qe`) already gates on loose
  tolerances with a comment that the QE-matching criterion is the energy and
  moment rather than the formal rhotol, which is exactly the stagnation the
  default seed shows here. The uniform split injects magnetization error into
  a slow spin-channel mode that the mixer takes over 120 iterations to drain,
  and the d-localized shape largely avoids exciting it.
- Fe PAW is a regression at marginal seeds. At 0.02 the d-localized seed
  collapses bcc Fe to the nonmagnetic branch, 660 meV above the FM ground
  state that the default reaches from the same 0.32 muB seeded moment. At
  0.05 and 0.10 it costs 6 and 2 extra iterations, and only at 0.30 is it
  faster (17 vs 21).

I don't have a mechanism I fully trust for the Fe flip. A plausible reading is
that the diffuse default magnetization produces a larger local spin splitting
in the low-density interstitial (the spin XC kernel grows as the density
falls), which nucleates the FM branch harder from a tiny seed, whereas the
d-localized shape concentrates the same moment where the density is highest.
Whatever the mechanism, one demonstrated wrong-branch collapse on the easiest
strong ferromagnet in the set is disqualifying for a default.

## Verdict

No-go for changing the default seed. The prior no-go stands, extended to the
density channel. The evidence, per the original bar, is that iterations saved
do not systematically exceed zero (Ni NC within one iteration either way, Fe
PAW -6 to +4 against the default at physical start_mag values), the seed build
cost is neutral so it never decides anything, and the branch-selection
improvement is real but not portable. Three marginal Ni PAW points show a
large systematic win and one marginal Fe PAW point shows a catastrophic loss,
and a default seed cannot trade Fe robustness for Ni convergence.

Two follow-ups are worth recording rather than acting on now.

- The d-localized seed is a legitimate rescue knob for marginal-Stoner PAW
  systems that stagnate under the default seed, the way Ni PAW does at rhotol
  1e-5. If that stagnation starts biting users, an opt-in
  `start_mag_shape="d"` keyword is the shape of the fix, defaulted to the
  current behavior.
- The Ni PAW default-seed stagnation itself (unconverged at every start_mag,
  including the comfortable 0.30) is a spin-channel mixing problem, not a seed
  problem, and it is already half-documented in the in-tree test comment. The
  seed experiment localizes it further, since a different magnetization shape
  with the identical moment converges fine. That points at the magnetization
  channel of the mixer, not at the charge channel.

## Files

- `seed.py` builds the d-localized spin pair (total SAD unchanged).
- `patch.py` installs and restores the monkeypatch on both SCF paths.
- `run.py` runs one system in `magnetic` (default vs d-loc over a start_mag
  sweep) or `flat` (SAD vs flat anchor) mode, printing JSON rows.
- `probe_pswfc.py` lists PSWFC availability and occupations per pseudo.
- `results_asus.jsonl` holds the full 28-row sweep (asus, 7 threads).
- `results_local_crosscheck.jsonl` holds the 4-row thinkpad cross-check.
