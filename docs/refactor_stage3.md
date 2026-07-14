# Stage 3 preparation — the uspp.py split

Execution prep for docs/refactor_plan.md stage 3, gathered while the
file is fresh (2026-07-13, HEAD cb007be, uspp.py at 1031 lines). Line
numbers WILL drift; every boundary below is anchored to a string.

## Consumer map (measured, not assumed)

External (16 call sites: tests, benchmarks, calculator, examples,
newton): only `scf_uspp` and `setup_uspp`.

Internal, beyond those two:
- `_HkS` — postscf/uspp_implicit.py (lazy import inside
  `_ConvergedUSPP.__init__`), postscf/uspp_bands.py
- `davidson_gen` — postscf/uspp_bands.py
- Everything else in the file is private to it.

tests/conftest.py rebinds the ATTRIBUTE `gradwave.scf.uspp.setup_uspp`
at conftest-import time for GRADWAVE_TEST_DEVICE. Consequences: the
facade module must remain the name everyone imports from, and
`uspp_loop.py` must never call `setup_uspp` internally (it doesn't
today — `scf_uspp` takes a built system).

## Target layout

- `scf/uspp_setup.py` (~330 lines): `_MINUS_I_POW_L`, `AugSpecies`,
  `USPPSystem` (+`.to`), `_mexp_index_map`, `_aug_tables`,
  `setup_uspp`, `_make_becsum_sym`.
- `scf/uspp_loop.py` (~700 lines): `_HkS`, `davidson_gen`, `scf_uspp`
  (driver) with the loop body extracted as `_scf_iteration`.
- `scf/uspp.py` (facade, ~10 lines): re-export `scf_uspp`,
  `setup_uspp`, `USPPSystem`, `AugSpecies`, `_HkS`, `davidson_gen`.
  Nothing else imports the split modules directly until the facade has
  survived a full slow suite.

## Anatomy of scf_uspp today (string anchors)

Pre-loop (driver setup, stays in the driver):
1. opts unpack + per-scheme defaults — anchor `mixing_w0, bec_step_scale = 0.01, None`.
   TRAP: `bec_step_scale is None` resolution must run BEFORE MixLayout
   construction (johnson → 1.0); it sits directly above the
   `layout = MixLayout(` line for that reason.
2. start_from warm start (grid/nspin checks, volume rescale of ρ,
   becsum carry) — anchor `start_from requires the same FFT grid`.
3. SAD / atomic-becsum guess — anchor `rho_ij_s = [[] for _ in range(nspin)]`.
4. MixLayout + mixer construction (kerker default, metric_w with the
   G=0 exclusion, per-scheme history) — anchor `# per-scheme defaults, measured on FM Ni`.
5. Hubbard manifold setup (hub_q, per-site slots).
6. `_becsum_for_onec` closure over `rho_ij_mix` — the MIXER-side becsum
   (feeds the one-center ddd), distinct from `rho_ij_s` (fresh output).

Loop body (`for it in range(1, max_iter + 1):`), in stage order —
this becomes `_scf_iteration`:
- potentials: v_h from ρ_tot, per-spin v_xc (`vxc_spin_potential` for
  nspin=2 with the half/half NLCC core split), vloc — anchor
  `rho_tot = rho_s[0] if nspin == 1`.
- screened D per spin (`∫ṽ_σ Q` + `dij_full`) + one-center
  `energy_and_ddd(_becsum_for_onec(a))` — anchor `# screened D per spin/atom`.
- diago tolerance schedule (it==1 SAD-vs-warm-start branch) — anchor
  `# SAD starts don't deserve a tight first solve`.
- diagonalization: batched (`uspp_batch`, default) or per-k
  `davidson_gen` reference path, warm-started from `coeffs`.
- occupations: shared-Fermi across spins (kweights concatenated per
  spin), then +U occupation matrices (V_U lags one iteration — the +U
  occupation state crosses iterations).
- densities: per-spin smooth ρ + becsum (TR-Hermitian part) +
  augmentation; becsum symmetrization before aug when use_symmetry.
- energy assembly: `EnergyBreakdown` incl. onecenter and hubbard —
  anchor `energies = EnergyBreakdown(`.

Post-body (driver keeps all of it):
- solver-blowup rescue (task #55: dE>5 eV from res<1e-2 → discard warm
  starts, salted reseed, skip the mixer this iteration) — anchor
  `# solver-blowup rescue`. Rescue MUTATES coeffs/seed_salt and does
  `continue` — in the split it becomes "call _scf_iteration again with
  fresh seeds", which is exactly why the iteration must take seeds as
  an argument instead of closing over them.
- pack via `layout.to_mix`, residual, history append, convergence
  (drho criterion / energy tail-of-3 + rho_safety).
- trust region (windowed recent-best baseline + 5-iteration reset
  cooldown) + `mixer.step` + unpack + becsum hermitization — anchor
  `# trust region: a residual jump`.
- result dict — MUST preserve every key: notably `rho_out_spin` (raw
  pre-mix map output; the rig's contract), `mixer_mult`
  (getattr-guarded), `smearing`/`width` (the adjoint reads them),
  `becps`, `hub_occ` when +U.

## The _scf_iteration contract

```
@dataclass
class IterOps:      # frozen once per scf_uspp call
    system, xc, nspin, layout, smearing, width, batched,
    hubbard data (manifolds, hub_q, alpha), onec (per species),
    vloc_r, phase_pos, symmetrizers, nbands, g_spin

@dataclass
class IterState:    # crosses iterations, owned by the driver
    rho_s            # per-spin densities (mixer output)
    rho_ij_mix       # mixer-side becsum (feeds ddd)
    coeffs           # per-spin per-k warm starts (or None)
    hub_v            # +U potential state (lags one step)
    seed_salt        # rescue reseeding

_scf_iteration(state, ops, diago_tol) -> IterResult:
    rho_out_s, rho_ij_out, becps, eigs, occ, mu, energies,
    coeffs (updated warm starts), hub_occ/e_u
```

Purity rule: `_scf_iteration` reads state, never mixes, never judges
convergence, never rescues. One evaluation of the SCF map, nothing
else. That is exactly what newton.py's raw-map evaluation and the
rig's J-applies need.

## Consumer switch (step d)

- `scf/newton.py`: replace the
  `scf_uspp(max_iter=1, start_from=..., etol=1e-300, rhotol=1e-300)`
  round trip with ops built once + `_scf_iteration` per evaluation.
  Kills the sentinel-tolerance hack and the per-call projector/guess
  rebuild. Same for `benchmarks/mixer_rig.py` (roughly halves its
  J-apply cost).
- `start_from` in `scf_uspp` stays — scans use it; only the
  max_iter=1 sentinel pattern dies.

## Commit sequence, each with its gate

1. Move setup block → `uspp_setup.py`, facade re-exports. Pure move.
   Gate: fast (81 s) + golden 4.
2. Move `_HkS`/`davidson_gen`/driver → `uspp_loop.py`; `uspp.py`
   becomes the facade. Pure move. Gate: fast + golden + mixer
   trajectory fixture.
3. Extract `_scf_iteration` inside `uspp_loop.py` (no consumer
   changes). Behavior-critical step. Gate: fast + golden + slow
   anchors (test_uspp_vs_qe si_paw, test_paw_spin_vs_qe ni, 
   test_uspp_batched_equality both spins) + test_uspp_warmstart +
   test_uspp_criteria.
4. Switch newton + rig to `_scf_iteration`. Gate: test_newton_polish
   (both legs) + a rig `build` smoke on Si (spectrum unchanged to
   float noise) + fast.

## Trap list (things that WILL bite if forgotten)

- conftest attribute-patches the facade; never bypass it internally.
- `rho_ij_mix` vs `rho_ij_s`: the ddd must see the MIXED becsum, the
  result/mixing must see the FRESH one. The USPP mixing-stability
  lessons (memory: gain~9 oscillation) live on this distinction.
- becsum symmetrization ORDER: becsum-sym BEFORE augmentation, ρ-sym
  after — group-equivariance of the aug map is what makes the adjoint
  transpose collapse; do not reorder while moving.
- Davidson warm-start rotation state lives in `coeffs` per (spin, k);
  the rescue path must be able to null them selectively.
- The `it == 1` diago_tol branch depends on whether start_from was
  given — that flag must reach `_scf_iteration` (via ops or the
  driver passing tol explicitly; prefer the driver computing tol_eff
  and passing it, keeping the schedule in one place).
- Energy criterion needs `e_free_prev` and the 3-entry tail — driver
  state, not iteration state.
- `float()` on e_free warns under parameterized functionals (the
  no_grad hygiene TODO from task #58) — do not "fix" it mid-refactor.
- Keep `# noqa` and comment blocks with their code; the loop comments
  encode task #55 and trust-region archaeology that must not be lost.

## Explicitly out of scope for stage 3

- No signature changes to scf_uspp/setup_uspp.
- No @torch.no_grad on scf_uspp (needs enable_grad islands first).
- No NC/USPP sharing (stage 4).
- No behavior change of any kind: golden energies to 1e-9, trajectory
  fixture to 1e-13.
