# Job queue (pueue)

A shared, load-limited way to run tests and benchmarks across the thinkpad +
asus fleet. The problem it solves: several agents (and you) launch heavy runs at
once, and nothing stops three `make test-fast` invocations from spawning 12
xdist workers on an 8-core laptop. `pueue` fixes that with a per-host daemon
that enforces a fixed slot budget per group — submit as much as you want, the
daemon serialises it so the boxes never thrash.

`scripts/gwq` is a thin wrapper over `pueue` (plain Python, no project env
needed). Agents submit through it instead of launching raw background jobs.

## Install (home-manager, per box)

pueue is a **user-level** daemon — jobs run as you, need your `uv`, `~/.venvs`,
the repo at `~/github/gradwave`, and your ssh keys for cross-host submits. Do
**not** install it system-wide. There is no nixpkgs module for pueue, so the
unit is hand-written. Add to your home-manager config (in willnix), then rebuild
on **both** the thinkpad and asus:

```nix
{ pkgs, ... }:
{
  home.packages = [ pkgs.pueue ];

  systemd.user.services.pueued = {
    Unit.Description = "pueue daemon";
    Service = {
      ExecStart = "${pkgs.pueue}/bin/pueued -v";
      Restart = "on-failure";
    };
    Install.WantedBy = [ "default.target" ];
  };
}
```

After `sudo nixos-rebuild switch --flake ~/github/willnix#<host>`:

```bash
systemctl --user enable --now pueued   # if not already started by the rebuild
./scripts/gwq init                     # create groups with this host's slot budget
```

Run `gwq init` **once per box** — it creates the groups below (idempotent; safe
to rerun after changing the budgets in `scripts/gwq`).

## Groups and slot budgets

A job is **one slot** regardless of how many cores/threads it spawns internally
(pueue has no core-weighting). The budgets encode the real limits — the fast
gate already forks `-n4` xdist workers, so one at a time on the laptop:

| group     | thinkpad | asus | used for |
|-----------|:--------:|:----:|----------|
| `default` |    2     |  4   | lint, light one-offs |
| `test`    |    1     |  2   | pytest tiers |
| `bench`   |    1     |  1   | captured benchmarks |
| `sweep`   |    1     |  1   | exclusive sweeps (pauses `test`/`default`) |
| `gpu`     |    —     |  1   | GPU-tagged work (asus only) |

## Usage

```bash
gwq test-fast                 # queue the fast gate on this host
gwq test-standard
gwq lint
gwq bench bench_scf cpu 8 nosym          # captured — writes benchmarks/results/<host>/
gwq --host asus bench bench_scf cpu 8 nosym
gwq run --group bench -- uv run python benchmarks/mixer_rig.py
gwq sweep -- uv run python benchmarks/delta_gauge/run.py   # exclusive; then `gwq resume`
gwq status                    # live queue across both hosts
gwq log <id>                  # tail a job's output (forwards to `pueue log`)
gwq pueue -- kill <id>        # forward raw args to pueue on --host
```

Defaults worth knowing:

- **Benches submitted on the thinkpad go to asus** (keep perf off the laptop).
  Force local with `--host thinkpad`.
- **Queued jobs run against `~/github/gradwave`**, not the worktree the
  submitting agent sits in — so results are reproducible and independent of who
  submitted them. Commit/pull the canonical checkout before queueing if you need
  a specific revision.
- `gwq bench` routes through `benchmarks/_capture.py`, which times the run and
  writes a JSON record (host, git sha, wall time, the reported metric line, exit
  status) to `benchmarks/results/<host>/` — the history a dashboard charts.

## Design rule (keep pueue forward-compatible with Dask)

Keep pueue **coarse**: one job == one whole test run or one whole sweep. Never
enqueue 60 per-element jobs — fan-out lives *inside* a job (xdist today, a Dask
cluster tomorrow). A future large sweep is submitted as a single `gwq sweep`
whose command spins up a Dask scheduler + workers internally; it holds the box
in the `sweep` group without co-scheduling the test gate. Nothing about pueue
has to be undone to adopt Dask — it nests under one slot.

## Dashboard

`gwq status` is the live view. Historical benchmark trends (wall time and the
reported metric vs git sha) accumulate in `benchmarks/results/<host>/` and get a
static HTML trend view in a follow-up once enough records exist.
