# Non-collinear / SOC SCF convergence: where the spinor path stands

Research note, branch `research/noncollinear-convergence`. Follows the collinear
spin-channel study (`research/uspp-spin-channel`), which localized the Ni PAW
stagnation to the mixer's magnetization channel and fixed it by defaulting
nspin=2 USPP/PAW mixing to johnson (PR #205). This note asks the same questions
of the spinor path. Which mixer does it use, which channel carries the residual
floor, and does the moment direction converge or wander.

RESULTS PENDING - baseline matrix running on asus.

## The mixing story on the non-collinear path, precisely

The #205 johnson default does NOT reach the spinor path. The three drivers
resolve their mixing scheme independently, and `scf_noncollinear` never calls
either resolver.

- `scf/noncollinear.py:523`. The driver takes `mag_mixer: str = "pulay"` and
  `_build_nc_mixer` (`scf/noncollinear.py:263-298`) instantiates that class by
  name from `_MAG_MIXERS` (`scf/noncollinear.py:259`, pulay/johnson/broyden).
  The magnetic default is PulayMixer. A nonmagnetic (m pinned to zero) run
  hardcodes PulayMixer with plain Kerker (`scf/noncollinear.py:276-279`).
- No auto-resolution. The collinear NC path resolves scheme=None to johnson for
  nspin=2 (`scf/loop.py`, `_resolve_mixing_scheme`), and the USPP path now
  defaults nspin=2 to johnson (#205, `scf/uspp_loop.py`). The spinor driver has
  no scheme argument in its public surface beyond `mag_mixer`, and `api.py`'s
  `_run_scf_noncollinear` (`api.py:340-349`) does not pass it, so YAML inputs
  cannot select it and every `task: scf, noncollinear: true` run mixes with
  pulay.
- Kerker masks the m channels. The magnetic mixer packs [rho, m_x, m_y, m_z]
  with Kerker on the rho block only (`kerker_mask`, `scf/noncollinear.py:282`),
  matching the collinear layout finding.
- The m channels get a decoupled, LARGER step. `mag_mixing_alpha` defaults to
  `max(mixing_alpha, 0.6)` (`scf/noncollinear.py:589-590`), applied as a
  per-component `step_scale` on the three m blocks, a moment-collapse guard
  documented at `scf/noncollinear.py:579-588` (bcc O2 collapses at alpha 0.4).
- The Stoner spin preconditioner is wired but opt-in. `spin_precond: bool =
  False` (`scf/noncollinear.py:522`) builds `build_stoner_precond_nc`
  (`scf/spin_precond.py:186`) each iteration and applies it to all three
  Cartesian m blocks (`scf/noncollinear.py:788-802`). Off by default, not
  reachable from the input layer.
- `precond_op` is a callable-only override on the charge block
  (`scf/noncollinear.py:535-536`), no string selector, unreachable from YAML.
- An adaptive backoff (`_nc_adaptive_backoff`) halves the global step and drops
  the DIIS history on a stalled residual (`scf/noncollinear.py:803-807`),
  active by default.

So the honest summary is that the spinor path is the last driver still mixing
its magnetization with plain pulay by default, and none of the magnetic knobs
(`mag_mixer`, `spin_precond`, `mag_mixing_alpha`, `mag_diago_schedule`) are
exposed through `inputs.py`/`api.py`.

## Prior art: fix/ni-soc-convergence

The branch (one WIP commit, 371ecc2, 2026-07-24, "gentler magnetization mixing
+ Stoner spin-precond boost [UNVALIDATED]") is MERGED in substance. Its two
source changes, the `mag_mixer` selector in `_build_nc_mixer` and the
non-collinear Stoner builder `build_stoner_precond_nc`, are on main verbatim at
the lines cited above. The branch itself is stale (its diff against current
main is dominated by months of main-side drift) and can be deleted.

What it was fighting is recorded in `tests/integration/test_ni_soc_convergence.py`
(PR #79): fcc Ni + SOC at 40 Ry, 4x4x4, gaussian 0.1 eV. The stock driver pins
the residual in a 3e-2 to 1e-1 band with the moment thrashing between -0.4 and
+5 uB and exhausts the budget. The merged recipe (`spin_precond=True`,
`mag_diago_schedule="quadratic"`, `mag_mixing_alpha=0.3`) locks the moment to
~0.6 uB and drops the residual 30-60x. The test's honesty caveat matters for
everything below: it gates at rhotol 5e-3, deliberately loose, because "a
separate, much smaller Fermi-surface residual floor (~1e-3 at this coarse mesh)
means the strict 1e-6 gate is not reached".

## Probes

`probe.py` implements a `mixer_hook` (an existing driver argument,
`scf/noncollinear.py:537`, called with the raw packed (vin, vout) before every
mix, `scf/noncollinear.py:808-809`), so no src changes and no monkeypatching.
Per iteration it records the charge and magnetization residual norms (the m
part as a vector-field norm and per Cartesian component), the integrated moment
vector from the G=0 coefficients, the angle the total moment turned since the
previous iteration, and a 12-shell |G| decomposition of both channels' residual
power (the flight recorder's binning, which does not support this path, PR
#203).

## Baseline matrix

Settings shared by every run: PBE (NoncollinearXC over SpinPBE), gaussian 0.1
eV, etol 1e-6, rhotol 1e-5, diago_tol 1e-9, max_iter 80, mixing_alpha 0.5,
history 8, no symmetry, full mesh. Systems (committed fixture pseudos):

- `pt_soc_nm`: fcc Pt, `Pt_ONCV_PBE_FR-1.0`, 40 Ry, 4x4x4, nonmagnetic=True.
  The no-magnetization SOC anchor.
- `ni_soc_*`: fcc Ni, `Ni_ONCV_PBE_FR-1.0`, 40 Ry, 4x4x4, the #79 system at
  tight tolerances. Arms: stock defaults, the #79 fix recipe, johnson
  (`mag_mixer="johnson"`), and a 200-iteration johnson oracle attempt. Seeds
  0.3/0.6/1.0 along z plus a [111] tilt at 0.6.
- `ni_socfree_*` / `ni_collinear_*`: the same cell with the scalar
  `PD_Ni_PBE` pseudo, spinor versus nspin=2 on the same physics, the honest
  cost of going non-collinear.
- `fe_*`: bcc Fe, `Fe_ONCV_PBE_FR-1.0` / `Fe_ONCV_PBE-1.2`, 40 Ry, 4x4x4, the
  robust-ferromagnet control.
- `fe2_canted90_*`: 2-atom bcc Fe cube, 35 Ry, 3x3x3, moments seeded 90 deg
  apart (z and x), with and without SOC. The genuinely non-collinear case,
  unconstrained, so exchange should align the moments during the SCF.

RESULTS TABLE PENDING

## Channel characterization

PENDING

## Prototype

PENDING

## Flags

- `docs/manual/noncollinear-soc.md` says nothing about spinor-SCF convergence
  tolerances; the only convergence advice in the magnetism pages is the
  multi-stability warning. Nothing measured so far contradicts the manual, but
  the manual also does not warn that the spinor driver cannot reach the
  rhotol=1e-7 driver default on a metallic magnet (the driver signature
  advertises 1e-7, `scf/noncollinear.py:518`).
