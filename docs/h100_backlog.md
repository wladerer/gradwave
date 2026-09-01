# H100 / datacenter backlog

A prioritized queue of gradwave work that has been explicitly **parked because it
needs datacenter hardware** — items that OOM on the 6 GB RTX 3050, need native
(uncrippled) fp64, need GPU saturation to show a win, or only cross over at large
N. The RTX 3050 has fp64 at ~1/64 of fp32, so any fp64-heavy or memory-heavy idea
is untestable on the consumer cards (thinkpad has no usable GPU; asus is the 3050).
H100 access is a **rental**, done once a feature is fully built out (per
`memory/batching-ux-and-h100-workflow.md`).

This manifest is the input to the resumable runner **`scripts/h100_backlog.sh`**.
The runner and this file are kept in lockstep: `scripts/h100_backlog.sh --list`
prints the same queue with its RUN-READY / BUILD-FIRST flags.

Related prior art already on `main`: the perf-A/B pack under **`benchmarks/h100/`**
(`provision.sh`, `run_all.sh`, `orchestrator.py`, `collect.py`, from the 2026-07-31
4×H100 rental session, PR #212) and the fuller test plan on branch
`feat/h100-test-plan` (`benchmarks/h100/H100_TEST_PLAN.md`, tests 0–6). That pack
covers the older **perf** lanes (spin-batch A/B, CUDA-graph, delta-gauge,
memory-ceiling). This backlog is the broader **feature/validation** queue that
accumulated afterward across the project memory.

## How to point the runner at an H100

Mirrors the asus offload pattern in the project `CLAUDE.md`. The runner drives the
box over ssh; results and done-markers stay on the orchestrating machine.

```bash
# 1. provision the box (reuse the shipped one-shot; clones + uv sync + records env)
#    curl -fsSL <raw-url>/benchmarks/h100/provision.sh | bash    # optional
# 2. point the backlog runner at it (ssh host + remote gradwave path)
H100_HOST=h100 GW_PATH=~/gradwave scripts/h100_backlog.sh --list        # show queue
H100_HOST=h100 GW_PATH=~/gradwave scripts/h100_backlog.sh               # run all RUN-READY
H100_HOST=h100 scripts/h100_backlog.sh --only rr_stack_recheck,flapw_efg_kconv
H100_HOST=h100 scripts/h100_backlog.sh --setup-only                     # provision check only
```

The **setup phase** ssh's in, `git pull && uv sync`, then verifies
`torch.cuda.is_available()` **and** that fp64 is native-fast (a quick fp64 GEMM
timing — H100 should land in the tens of TFLOP/s; a crippled card lands near ~0.1).
It prints the GPU name and free memory before any item runs.

Each item runs as its own ssh job writing to `results/<item>/run.log` with an
`EXIT=<rc>` marker, and drops `results/<item>/DONE` on success so a re-run skips it
(`--force` to re-run). One item failing does not abort the rest.

---

## The queue

Tiers: **P0** = quick, decisive probes that gate the bigger builds; **P1** = larger
builds / long validations. Flags: **RUN-READY** = an entrypoint exists (possibly on
a named branch); **BUILD-FIRST** = a driver/probe must be written before the H100
run is meaningful (the runner emits `SKIPPED (driver not built)` for these).

| # | item id | tier | flag | what it is |
|---|---------|------|------|------------|
| 1 | `surface_vacuum_ladder` | P0 | BUILD-FIRST | CO/Pt(111) vacuum ladder via `api.run` + ESM `open_z` |
| 2 | `config_batch_probe` | P0 | RUN-READY (branch) | fold N phonon configs into the PW batch axis — H100 saturation A/B |
| 3 | `stochastic_dft_variance` | P0 | BUILD-FIRST | stochastic-DFT trace-estimator variance vs N_χ at a few sizes |
| 4 | `tt_rank_slab` | P0 | BUILD-FIRST | tensor-train rank of a converged slab density along the normal vs bulk |
| 5 | `rr_stack_recheck` | P0 | RUN-READY (branch) | large-nb exact-fp64 Rayleigh-Ritz stack re-measured on native fp64 |
| 6 | `flapw_efg_kconv` | P1 | RUN-READY | FLAPW real-material EFG validation at converged k vs Elk |
| 7 | `dmi_full_vector` | P1 | BUILD-FIRST | FeGe full DMI vector at tight etol (~1e-9) |
| 8 | `config_batch_driver` | P1 | BUILD-FIRST | the batched multi-config SCF driver + batching-UX validation |
| 9 | `defect_gipaw_supercell` | P1 | BUILD-FIRST | Na-ion defect PW/PAW supercell + GIPAW reconstruction |

---

### 1. `surface_vacuum_ladder` — P0, BUILD-FIRST

- **What:** CO/Pt(111) 2×2×3 slab, vacuum ladder {5,6,8 Å/face} + a 15 Å reference,
  driven through `api.run` with ESM `boundary="open_z"`, atop **and** fcc-hollow.
  Report E_ads, the **atop−fcc site-preference difference** (the sharp gate),
  adsorbate forces, and `work_function(both_faces=True)`.
- **Why datacenter:** the 3050 OOMs on Pt slabs (heavy pseudo + large FFT box); the
  H100 fits them and runs the ladder fast. The probe also confirms the ESM-on-ASE
  correctness fix (`memory/surface-slab-efficiency-stack.md`: the flagship
  `examples/co_pt/slab_co.py` currently runs plain 3D-periodic through the ASE path
  = a live correctness bug for the asymmetric adsorbate slab).
- **Entrypoint:** `experiments/surface_efficiency/vacuum_ladder.py` — **TODO: build.**
  The ESM engine (`core/energies/esm.py`) and `inp.scf.boundary="open_z"` are shipped
  and FD-validated; the probe is a self-contained driver that builds the CO/Pt
  geometry, sweeps vacuum, and calls `api.run`. Scaffolding may exist on branch
  `exp/surface-efficiency-foundation` / asus `/tmp/surface_eff`; the runner is
  self-contained and does not depend on it.
- **Output / runtime:** a small JSON/table of E_ads + site-difference + forces +
  Φ vs vacuum; a free npw-ratio pre-check (no SCF). ~couple GPU-hours.
- **Decision:** GO on the Tier-1 slab stack (ESM default, slab k-mesh nz→1,
  clean-slab warm-start) if E_ads and the site-difference plateau by ~6 Å/face
  within 5 meV and forces within 5 meV/Å.

### 2. `config_batch_probe` — P0, RUN-READY (branch)

- **What:** fold N phonon-displacement configs into the leading PW batch axis
  (`(N·nk, nb, npw_max)` in `core/batch.py`) and A/B one folded `davidson_batched`
  vs N serial solves, with an eigenvalue-equality check.
- **Why datacenter:** measured neutral-to-negative on CPU (0.46×) and the 3050
  (1.07×) because one spoke already fills those. The whole premise — *one spoke
  under-fills a large GPU, folding fills it* — is an H100-class claim
  (`memory/config-batching-probe-parked.md`, PR #288).
- **Entrypoint:** `benchmarks/phonon_batch/config_batch_probe.py` on branch
  `worktree-phonon-config-batching` (PR #288, **closed**). The runner checks out the
  branch and runs the probe; if the file is absent it emits `SKIPPED (entrypoint
  missing)` and points here. Restore/rebase the branch before the H100 session.
- **Output / runtime:** folded-vs-serial wall ratio at a few supercell sizes + the
  bit-exactness delta. Minutes to ~1 h.
- **Decision:** GO on `config_batch_driver` (#8) only if the fold is > 1.3× at a
  realistic supercell where a single spoke leaves the H100 idle.

### 3. `stochastic_dft_variance` — P0, BUILD-FIRST

- **What:** the cheap feasibility probe for stochastic DFT (Baer–Neuhauser–Rabani):
  density as a Chebyshev Fermi-operator trace over N_χ random vectors. Measure the
  **estimator variance vs N_χ** at a few system sizes (electrons), and check the
  variance falls as expected (≈ 1/N_χ, self-averaging with system size).
- **Why datacenter:** the order-of-magnitude ceiling is asymptotic in N — it only
  pays at hundreds+ electrons, on the FLAPW/defect-supercell path
  (`memory/esoteric-paradigm-scf-verdict.md`). Small cells are latency-bound and see
  nothing. Native fp64 + memory for the large trace vectors is H100-only.
- **Entrypoint:** `experiments/stochastic_dft/variance_probe.py` — **TODO: build.**
  A ~50–100 line probe: build H-apply for a fixed converged density, apply a
  Chebyshev expansion of the Fermi function to random ±1 vectors, estimate n(r) and
  its variance vs N_χ. No SCF, no new solver — a feasibility measurement only.
- **Output / runtime:** variance-vs-N_χ curves at 2–3 sizes + the self-averaging
  check. ~1 GPU-hour.
- **Decision:** GO on a stochastic-density SCF prototype only if variance at
  achievable N_χ is small enough to be cheaper than the deterministic eigensolve at
  the crossover size.

### 4. `tt_rank_slab` — P0, BUILD-FIRST

- **What:** measure the tensor-train (QTT) rank of a **converged slab density**
  along the surface normal, compared to a bulk density. Decides the QTT/wavelet
  surface-compression bet: the vacuum region plausibly makes the normal direction
  low-rank.
- **Why datacenter:** needs a converged Pt/Au **slab** density to analyze, which is
  exactly the cell that OOMs on the 3050 (same blocker as #1).
  `memory/surface-slab-efficiency-stack.md` lists this as the deferred probe that
  gates the normal-compression research bet.
- **Entrypoint:** `experiments/surface_efficiency/tt_rank.py` — **TODO: build.**
  Takes a converged density (checkpoint from #1 or a fresh slab SCF), reshapes the
  normal axis into a QTT, runs TT-SVD at a tolerance, reports rank vs a bulk
  reference. Analysis-only.
- **Output / runtime:** rank(tol) curves, normal vs bulk. ~1 GPU-hour (dominated by
  the one slab SCF, shareable with #1).
- **Decision:** GO on QTT/wavelet normal compression only if the slab normal rank is
  materially below bulk at a useful tolerance.

### 5. `rr_stack_recheck` — P0, RUN-READY (branch)

- **What:** re-measure the exact-fp64 Rayleigh-Ritz stack (3M/Karatsuba complex
  GEMM + incremental build + restart-carry + Hermitian half-build) on a large-nb
  magnetic SCF (α-Fe₂O₃ hematite) on **native** fp64.
- **Why datacenter:** the whole stack was measured on the 3050, where fp64 is
  crippled and the 3M path OOM'd the 6 GB card — PR #411 was **closed** as
  not-worth-it *on that card*. On the H100 fp64 is native-fast and the memory is
  ample, a distinct regime; a cheap re-measure settles whether any piece is worth
  reviving on datacenter HW (`memory/consumer-gpu-3050-frontier-verdict.md`).
- **Entrypoint:** `benchmarks/bench_scf.py` (+ `benchmarks/minerals/run_bench.py`
  hematite) on branch `feat/rr-exact-fp64-stack`, toggling `GRADWAVE_RR3M=on/off`
  and the piece flags across A/B runs. The runner checks out the branch; the SCF
  entrypoints are on `main`, the stack is the branch delta.
- **Output / runtime:** whole-SCF and eigensolver wall for {native ZGEMM, pieces
  2–4, +3M} on hematite, all bit-exact (ΔE ≤ 1e-9). ~1–2 GPU-hours.
- **Decision:** revive any piece only if it clears a meaningful whole-SCF margin on
  native fp64 that it could not on the 3050 (where pieces 2–4 gave 1.17×, 3M 0.94×).

### 6. `flapw_efg_kconv` — P1, RUN-READY

- **What:** FLAPW real-material EFG validation at a converged, code-matched k-mesh
  (≥ k6) across the rutile / anatase / corundum / MgF₂ / h-BN / Li₃N set, vs Elk
  11.0.2. The k-standardization campaign (`memory/autoapw-flapw-shipped-prod-g.md`,
  PR #362) showed the earlier ratios were k222 artifacts and set matched-k6 as the
  standard.
- **Why datacenter:** FLAPW is dense O(N³) with no iterative escape and a bad
  all-electron/fp64 prefactor; converged-k full-potential runs are ~35 min each on
  asus CPU and multiply across materials × k-meshes. Native fp64 + throughput make
  the full matched-k sweep tractable. (FLAPW stays the **oracle**, not a production
  engine — `memory/flapw-oracle-vs-production-defect-path.md`.)
- **Entrypoint:** `experiments/autoapw/kconv_efg.py` (env-driven: `MAT`, `KMESH`,
  `HELO`, `ECUT`, `LMAX`, `FPLMAX`). RUN-READY on `main`. The gradwave side runs
  standalone; the Elk cross-check needs Elk on the box
  (`nix shell nixpkgs#elk` or a provisioned build).
- **Output / runtime:** per-site V_zz / η / C_Q + on-site/boundary decomposition +
  sphere-charge partition check, per material at matched k6/k8. Hours across the set.
- **Decision:** confirms which sites are genuinely accurate at converged k (corundum
  O/Al, MgF₂ F were ~0.95–0.98) and which under-capture (rutile Ti/O); feeds the
  EFG-accuracy levers.

### 7. `dmi_full_vector` — P1, BUILD-FIRST

- **What:** the full 3-reference FeGe DMI **vector** |D| at tight convergence
  (etol ~1e-9), from the validated `spin_exchange` extractor via
  `characterize_magnetism(..., exchange=True)`.
- **Why datacenter:** the DMI channel is correct but only above the
  convergence-noise floor at etol ~1e-9; the full vector is ~9–12 h of tight SOC
  SCFs on the 8-atom B20 cell — flagged H100/batching in
  `memory/dmi-extractor-validated-fege.md`.
- **Entrypoint:** the probe scripts (`fege_fullvec`, `dmi_tight`) were **uncommitted
  job-tmp** on asus — **TODO: commit a driver** (e.g.
  `experiments/dmi/fege_full_vector.py`) that runs the 3 constrained-SCF references
  at tight etol and assembles D. The extractor itself is shipped.
- **Output / runtime:** the Fe–Fe D vector + control (bcc-Fe noise floor) + the
  |D|/J ratio → pitch. ~9–12 GPU-hours (or less batched).
- **Decision:** unblocks the differentiable d(pitch)/d(knob) target once the vector
  is trustworthy at converged basis (k/ecut still to converge).

### 8. `config_batch_driver` — P1, BUILD-FIRST (gated on #2)

- **What:** the batched multi-config SCF driver — per-config v_eff / density / Fermi
  / mixing / convergence-masking on a config batch axis in `core/batch.py` — plus
  its batching-UX surface (opt-in flag default-off, consistent naming in both the
  Python API and the YAML schema, per `memory/batching-ux-and-h100-workflow.md`).
  Its one durable edge over SeedPool is that the folded solve is **autograd-compatible**.
- **Why datacenter:** only worth building **after** `config_batch_probe` (#2)
  confirms the under-fill→fill win on the H100; do not build blind
  (`memory/config-batching-probe-parked.md`). This is the flagship
  batch-into-one-tensor from `docs/ideas.md`.
- **Entrypoint:** none — **TODO: build after #2 is positive.** The runner emits
  `SKIPPED (driver not built)`.
- **Output / runtime:** a campaign (EOS volumes / U-scan / phonon displacements)
  run folded vs SeedPool-serial, same ΔE, wall compared. Days of build + a
  validation run.
- **Decision:** ship if it beats SeedPool on the H100 node **and** the UX passes the
  API+YAML consistency bar; otherwise SeedPool stays the campaign lever.

### 9. `defect_gipaw_supercell` — P1, BUILD-FIRST

- **What:** the production defect path — relax + ground-state density on a Na-ion
  defect **supercell** with PW/PAW (the side that scales and is GPU-capable), then
  reconstruct the local observable (EFG / NMR shielding) at the defect site via
  GIPAW / PAW reconstruction, without an all-electron SCF of the whole cell.
- **Why datacenter:** defect supercells (2×2×2+ to dilute the defect) are large-N;
  the PW/PAW relax is where H100 throughput pays off, and GIPAW reconstruction rides
  on top (`memory/flapw-oracle-vs-production-defect-path.md`). This is the realistic
  battery-materials destination.
- **Entrypoint:** none end-to-end — **TODO: build.** The pieces are shipped (PW/PAW
  SCF+relax via `api.run`; GIPAW bare shielding PR #330; PAW-EFG). The gap is a
  driver that ties supercell relax → GIPAW reconstruction at the defect site plus a
  reference-energy/observable provenance. LiFePO₄ paramagnetic (Fermi-contact) is a
  separate, later module — do not conflate.
- **Output / runtime:** defect-site EFG / ²³Na shielding on a converged supercell.
  A multi-day build; the H100 run is the relax + reconstruction.
- **Decision:** the strategically important production build once the closed-shell
  oracle work lands.

---

## Not (re-)parked here — already settled or on other tracks

- **Perf A/B lanes** (spin-batch, CUDA-graph, delta-gauge, memory ceiling): covered
  by `benchmarks/h100/` (PR #212) + `feat/h100-test-plan`; this backlog does not
  duplicate them. Run those via `benchmarks/h100/run_all.sh` on a multi-GPU box.
- **Approximate/INT8 FFT, fp64-QR attacks:** measured-closed on the 3050 and
  hardware-independent-dead; not H100-revivable (`consumer-gpu-3050-frontier-verdict.md`).
- **Reduction correctness/counts** (DisplacementStar #274, StrainStar #275),
  **SeedPool correctness** (#276), **tol-ladder**, **import diet**: hardware-independent,
  settled on asus.
</content>
</invoke>
