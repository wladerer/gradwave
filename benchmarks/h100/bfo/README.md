# BiFeO3 PAW+U on an H100 — one-hour run

A self-contained, paste-and-go benchmark for a single rented H100: does the
USPP/PAW(+U) soft-mode deflation (this branch, `postscf.uspp_softmode`) run on a
real fp64 datacenter GPU, and does the composite response *scale* there?

gradwave is strict fp64. The RTX 3050 throttles fp64 to ~1/64 of fp32, so the
forward SCF of a small cell is CPU-won there; the H100 has real fp64, so this
run measures what that buys the batched response (the conduction-projected
generalized Sternheimer — the fp64 GEMM/FFT the H100 is for).

## Run it

Rent a **verified-datacenter, non-interruptible** H100 with **>=50 GB disk**.
Then either set `provision_bfo.sh` as the vast.ai PROVISIONING_SCRIPT, or SSH in
and run:

```bash
curl -fsSL https://raw.githubusercontent.com/wladerer/gradwave/worktree-uspp-softmode-deflate/benchmarks/h100/bfo/provision_bfo.sh | bash
```

It clones the branch, `uv sync`s (pulls torch — the slow step), downloads the Bi
PAW pseudo (Fe/O kjpaw ship in the repo), records the CUDA env, and launches
`bfo_h100_driver.py` **detached** (survives an SSH drop). Watch the printed log;
when `BFO_H100_DONE` appears, scp back `benchmarks/results/<host>/bfo_h100.json`.

## What the driver does (phased, defensive)

1. **smoke** — a tiny BiFeO3+U SCF on the GPU. The USPP/PAW+U path had never run
   on CUDA; if it errors here the run STOPS and says so (a real finding, not a
   wasted hour).
2. **de-risk** — times one composite response apply (χ̃) and one SCF iteration
   CPU vs GPU. The **χ̃-apply GPU speedup is the number** that says whether the
   response scales; everything else is context.
3. **bench** — converges BiFeO3+U on the GPU at `BFO_KMESH` (default 3³), then
   the fxc=1.5 deflation comparison (baseline Anderson vs deflated), with peak
   VRAM.

Each phase writes `bfo_h100.json` incrementally, so a later crash keeps the
earlier numbers.

## Knobs (env vars)

| var | default | note |
|---|---|---|
| `GRADWAVE_BRANCH` | `worktree-uspp-softmode-deflate` | set to `main` once #263 merges |
| `BFO_KMESH` | `3` | main-run k-mesh/axis; bump to 4 to load the GPU harder |
| `BFO_ECUT` / `BFO_ECUTRHO` | `45` / `360` | Ry |
| `BFO_U` / `BFO_J` | `4.0` / `1.0` | Dudarev +U/J on Fe 3d (eV) |
| `BFO_THREADS` | `min(16, ncpu)` | CPU threads for the CPU-side timings |

## Honest expectations

- The **5-atom cubic cell is small** — the *forward SCF* is launch-overhead-bound
  and may not beat CPU. The win, if any, is in the **batched χ̃ apply** (step 2)
  and grows with `BFO_KMESH` / `ECUT`. gradwave is pure-eager (no custom
  kernels), so expect "meaningfully faster on batched fp64", not peak H100.
- An hour is provisioning (~10-15 min) + smoke + de-risk + **one** converge +
  deflation. It is not a survey; scope to the one number that matters (step 2).
- Cubic FM is a proxy for the real 10-atom R3c G-AFM ground state; a flagship
  physics run wants that cell (heavier) and the SOC path.
