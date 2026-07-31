# H100 benchmark pack

A self-contained benchmark suite for a rented 4x H100 instance. The whole
session is three actions. Paste a provisioning URL when you rent the box, wait
for the lanes to finish, and scp one results tarball back. Everything runs
unattended, and a lane that crashes leaves the other three running.

gradwave is strict fp64. The RTX 3050 numbers this pack compares against are
fp64-throttled to about 1/64 of fp32, so the H100 comparison measures what a
real datacenter fp64 unit buys the code.

## What each lane answers

Four lanes run at once, one pinned to each GPU by `CUDA_VISIBLE_DEVICES`.

- **Lane 0** times the reference forward SCFs, `bench_scf` (Si LDA, 30 Ry,
  4x4x4) and two magnetic minerals (Cr2O3 eskolaite and Fe2O3 hematite), on the
  GPU and on the box CPU. It answers how the H100 forward SCF compares to the
  recorded asus CPU and RTX 3050 numbers.
- **Lane 1** runs the delta-gauge equation-of-state chain for Al, Cu, Si, and
  Ge on the GPU. These four have recorded RTX 3050 and CPU timings, so the lane
  answers the per-volume speedup and whether the fitted delta reproduces on
  H100.
- **Lane 2** is the A/B matrix of the 3050-era optimizations on one midweight
  system, Cr2O3 eskolaite. It answers whether three tricks still pay on H100,
  the CPU-offloaded batched QR from PR #174, the choice of davidson against
  chebyshev, and the Rayleigh-Ritz GEMM residency question. Every arm records
  wall time, iteration count, and the converged energy, and an energy oracle
  checks the arms agree.
- **Lane 3** climbs a Si supercell ladder to find the memory ceiling, then runs
  the multi-GPU probe. It answers where peak GPU memory and per-iteration wall
  land as the cell grows, and where the run goes out of memory.

## Renting the box on vast.ai

1. Pick a **verified datacenter** host, not a community host. The lanes run for
   an hour or more and a community box can vanish mid-run.
2. Set the instance to **non-interruptible** (on-demand, not spot). A spot
   instance can be reclaimed and take a partial results set with it.
3. Give it at least **50 GB of disk**. The clone, the uv venv with torch and
   CUDA, and the results tree fit inside that with room to spare.
4. Use the vast.ai **PyTorch template** (Ubuntu container, root in container, uv
   preinstalled, SSH and tmux). The pack does not use the template's
   `/venv/main`. It builds the project venv with `uv sync` and addresses it only
   through `uv run`.

**SXM against PCIe.** Prefer an SXM host if the multi-GPU probe interests you.
SXM cards run the full power envelope and are linked by NVLink, whereas PCIe
cards are often power-capped by the host and talk over the slower bus. The
single-GPU lanes are indifferent to this. The lane-3 probe is not, because it
moves tensors between ranks.

## Running it

Two ways to start, both idempotent.

**Provisioning URL.** Paste the raw URL of `provision.sh` into the vast.ai
`PROVISIONING_SCRIPT` field when you create the instance. vast.ai fetches and
runs it once on first boot.

```
https://raw.githubusercontent.com/wladerer/gradwave/main/benchmarks/h100/provision.sh
```

**By hand after SSH.** If you would rather watch it start, SSH in and pipe the
same script to bash.

```bash
curl -fsSL https://raw.githubusercontent.com/wladerer/gradwave/main/benchmarks/h100/provision.sh | bash
```

`provision.sh` clones gradwave over public https, runs `uv sync`, records the
torch and CUDA versions and an `nvidia-smi` snapshot into the results directory,
then launches `run_all.sh` detached under `nohup` with a timestamped master log.
Watch progress by tailing that log. The path is printed at the end of
provisioning.

```bash
tail -f ~/gradwave/benchmarks/results/$(hostname)/run_all-*.log
```

Each lane also writes its own `lane<n>.log` ending in a `LANE<n>_EXIT=<rc>`
marker, and the master log ends in `RUN_ALL_EXIT` once all four have finished.
Grep for the marker rather than watching the tail.

When `RUN_ALL_EXIT` appears, bundle and pull the results.

```bash
cd ~/gradwave && uv run python benchmarks/h100/collect.py
```

`collect.py` tars the results tree, prints the exact scp line to run on your
workstation, and writes `comparison.md`, a table pre-filled with the recorded
RTX 3050 and asus CPU reference numbers next to the H100 columns the JSON
records fill.

## Expected wall time and cost

Times are estimates for a 4x H100 SXM box. The lanes run in parallel, so the
session length is the slowest lane plus the few minutes `uv sync` takes.

| lane | what runs | rough wall |
|---|---|---|
| 0 | bench_scf GPU and CPU, two minerals GPU and CPU | 30 to 60 min, the CPU mineral SCFs dominate |
| 1 | delta-gauge Al, Cu, Si, Ge on the GPU | 20 to 40 min |
| 2 | six SCF arms on eskolaite plus a GEMM microbench | 15 to 30 min |
| 3 | Si supercell ladder plus the multi-GPU probe | 30 to 60 min, the probe is hard-capped at 15 min |

A full session lands around one to two hours of wall time. A 4x H100 instance
rents for roughly 8 to 12 dollars an hour depending on the host and the moment,
so a session costs on the order of 10 to 30 dollars. Check the live price when
you rent, because it moves.

If you want a faster and cheaper first pass, cap the CPU minerals and the ladder
depth through the environment. `H100_CPU_THREADS`, `H100_LANE3_MAXITER`, and the
per-run `H100_T_*` timeouts are all read at launch.

## How the QR-offload toggle works

Lane 2's first pair toggles the CPU-offloaded batched QR that PR #174 added to
the Davidson solver. That offload is a CUDA-only, hardcoded device conditional
in `solvers/davidson._qr_offload`, gated by the module global `_QR_CPU_MAX_COLS`
(16). There is no environment knob for it in the source, and this pack does not
add one to the source. The lane script monkeypatches the module global for the
duration of each arm and restores it afterward. Setting it to 16 leaves the
offload on, its shipped default. Setting it to 0 turns it off, because the
`cols <= 0` test is never true, so the QR stays on the GPU. Both arms run on
CUDA, since the offload is a no-op on CPU. Nothing in `src/` changes.

## Lane 3 is diagnostic, and the multi-GPU probe is untested

The multi-GPU single-SCF path has never run on real multi-GPU hardware. The
k-point-sharded distributed SCF uses the Gloo backend, which is a CPU collective
and may stage CUDA tensors through the host on every reduction. Whether that is
correct and whether it is faster than one GPU are both open. The probe launches
a 4-rank `torchrun` job with `distributed: true` and `device: cuda`, each rank
pinned to its own GPU by `mgpu_wrap.sh`, under a hard 15-minute timeout. It
captures whichever outcome happens, the verbatim error if it fails or the
speedup against a single GPU if it runs. Either outcome is the result. Read the
probe's record as a diagnostic, not as a validated speedup.

`mgpu_wrap.sh` is syntax-validated only. Everything else in the pack was
validated on CPU before shipping, the orchestrator dry-run, the delta-gauge and
mineral invocations against their real CLIs, and a minimal Si SCF through the
JSON-writing path. The multi-GPU wrapper could not be, because it needs the
hardware this pack exists to test.
