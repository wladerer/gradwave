#!/usr/bin/env bash
# h100_backlog.sh -- resumable runner for the datacenter-gated gradwave backlog.
#
# Drives an H100 (or any real-fp64 GPU) box over ssh, mirroring the asus offload
# pattern in the project CLAUDE.md: ssh in, `git pull && uv sync`, then run each
# parked backlog item as its own job with a per-item log + done-marker so a
# re-run skips completed work. One item failing never aborts the rest.
#
# The queue and its RUN-READY / BUILD-FIRST flags are documented in
# docs/h100_backlog.md; `--list` prints the same set.
#
# Usage:
#   H100_HOST=h100 GW_PATH=~/gradwave scripts/h100_backlog.sh [OPTIONS]
#
# Options:
#   --list                 print the backlog queue and exit (no ssh)
#   --only id[,id,...]      run only these item ids
#   --force                re-run items even if results/<id>/DONE exists
#   --setup-only           run the provision/fp64 check, then stop
#   --skip-setup           skip the provision/fp64 check (assume box is ready)
#   --results-dir DIR      where logs + done-markers land (default below)
#   --item-timeout SECS    per-item wall ceiling (default 21600 = 6 h)
#   -h, --help             this help
#
# Environment:
#   H100_HOST     ssh host/alias for the datacenter box       (default: h100)
#   GW_PATH       gradwave checkout path ON the remote box    (default: ~/gradwave)
#   GRADWAVE_BRANCH  branch to sync the remote to for the whole session (default: main)
#   H100_RESULTS_DIR local results dir (overridden by --results-dir)
#
# NOTE: run entirely from the orchestrating machine. Never run gradwave compute on
# the laptop. This script does no local compute -- it only ssh-drives the remote.

set -uo pipefail   # NOT -e: a failing item must not abort the runner.

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

H100_HOST="${H100_HOST:-h100}"
GW_PATH="${GW_PATH:-~/gradwave}"
REMOTE_BRANCH="${GRADWAVE_BRANCH:-main}"
RESULTS_DIR="${H100_RESULTS_DIR:-$REPO_DIR/benchmarks/results/h100-backlog}"
ITEM_TIMEOUT="${H100_ITEM_TIMEOUT:-21600}"

ONLY=""
FORCE=0
SETUP_ONLY=0
SKIP_SETUP=0
LIST_ONLY=0

# --------------------------------------------------------------------------- #
# Backlog registry
#
# One record per line, TAB-separated:
#   id  <TAB>  tier  <TAB>  status  <TAB>  branch  <TAB>  entrypoint  <TAB>  remote_cmd  <TAB>  desc
#
# status : run   = an entrypoint exists (possibly on `branch`); the runner runs it.
#          build = BUILD-FIRST; no driver yet -> the runner prints SKIPPED and moves on.
# branch : remote branch to checkout for this item, or "-" for the session default.
# entrypoint : a file that must exist on the remote for a `run` item; if absent the
#          runner downgrades it to a runtime SKIPPED (entrypoint missing).
# remote_cmd : executed on the remote inside GW_PATH via `uv run ...`. Use "-" for build items.
#
# Keep in lockstep with docs/h100_backlog.md.
# --------------------------------------------------------------------------- #
backlog_records() {
    # Fields are TAB-separated; printf keeps the tabs literal.
    printf '%s\n' \
"surface_vacuum_ladder	P0	build	-	experiments/surface_efficiency/vacuum_ladder.py	-	CO/Pt(111) vacuum ladder via api.run + ESM open_z (build driver; ESM engine shipped)" \
"config_batch_probe	P0	run	worktree-phonon-config-batching	benchmarks/phonon_batch/config_batch_probe.py	uv run python benchmarks/phonon_batch/config_batch_probe.py	fold N phonon configs into the PW batch axis: H100 saturation A/B (PR #288 branch)" \
"stochastic_dft_variance	P0	build	-	experiments/stochastic_dft/variance_probe.py	-	stochastic-DFT trace-estimator variance vs N_chi at a few sizes (build ~50-line probe)" \
"tt_rank_slab	P0	build	-	experiments/surface_efficiency/tt_rank.py	-	tensor-train rank of a converged slab density along the normal vs bulk" \
"rr_stack_recheck	P0	run	feat/rr-exact-fp64-stack	benchmarks/bench_scf.py	GRADWAVE_RR3M=auto uv run python benchmarks/minerals/run_bench.py hematite --device cuda --outdir results/h100-backlog/rr_stack_recheck --skip-qe	large-nb exact-fp64 Rayleigh-Ritz stack re-measured on native fp64 (PR #411 was 3050-only)" \
"flapw_efg_kconv	P1	run	-	experiments/autoapw/kconv_efg.py	MAT=corundum KMESH=6 HELO=1 uv run python experiments/autoapw/kconv_efg.py	FLAPW real-material EFG at converged k vs Elk (env MAT/KMESH; Elk optional)" \
"dmi_full_vector	P1	build	-	experiments/dmi/fege_full_vector.py	-	FeGe full DMI vector at tight etol ~1e-9 (extractor shipped; commit a driver)" \
"config_batch_driver	P1	build	-	src/gradwave/core/batch_configs.py	-	batched multi-config SCF driver + batching-UX (gated on config_batch_probe being positive)" \
"defect_gipaw_supercell	P1	build	-	experiments/defects/na_defect_gipaw.py	-	Na-ion defect PW/PAW supercell + GIPAW reconstruction (pieces shipped; tie together)"
}

# --------------------------------------------------------------------------- #
# Arg parsing
# --------------------------------------------------------------------------- #
usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)          LIST_ONLY=1; shift ;;
        --only)          ONLY="${2:-}"; shift 2 ;;
        --only=*)        ONLY="${1#*=}"; shift ;;
        --force)         FORCE=1; shift ;;
        --setup-only)    SETUP_ONLY=1; shift ;;
        --skip-setup)    SKIP_SETUP=1; shift ;;
        --results-dir)   RESULTS_DIR="${2:?}"; shift 2 ;;
        --results-dir=*) RESULTS_DIR="${1#*=}"; shift ;;
        --item-timeout)  ITEM_TIMEOUT="${2:?}"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# comma-list membership test
_selected() {
    local id="$1"
    [[ -z "$ONLY" ]] && return 0
    local x
    IFS=',' read -ra _sel <<< "$ONLY"
    for x in "${_sel[@]}"; do [[ "$x" == "$id" ]] && return 0; done
    return 1
}

# --------------------------------------------------------------------------- #
# --list
# --------------------------------------------------------------------------- #
if [[ "$LIST_ONLY" -eq 1 ]]; then
    printf '%-24s %-4s %-11s %s\n' "ITEM" "TIER" "FLAG" "WHAT"
    printf '%-24s %-4s %-11s %s\n' "----" "----" "----" "----"
    while IFS=$'\t' read -r id tier status branch entry cmd desc; do
        flag="RUN-READY"; [[ "$status" == "build" ]] && flag="BUILD-FIRST"
        [[ "$branch" != "-" ]] && flag="$flag*"
        printf '%-24s %-4s %-11s %s\n' "$id" "$tier" "$flag" "$desc"
    done < <(backlog_records)
    echo
    echo "* = entrypoint lives on a named branch (see docs/h100_backlog.md)"
    exit 0
fi

# --------------------------------------------------------------------------- #
# Remote helpers
# --------------------------------------------------------------------------- #
# Run a command on the remote inside GW_PATH. Non-interactive, batch-mode ssh.
rsh() { ssh -o BatchMode=yes "$H100_HOST" "cd $GW_PATH && $*"; }

mkdir -p "$RESULTS_DIR"
MASTER_LOG="$RESULTS_DIR/backlog_$(date +%Y%m%d_%H%M%S).log"
echo "[backlog] host=$H100_HOST remote=$GW_PATH branch=$REMOTE_BRANCH results=$RESULTS_DIR" | tee "$MASTER_LOG"

# --------------------------------------------------------------------------- #
# Setup phase: provision + fp64 sanity
# --------------------------------------------------------------------------- #
setup_phase() {
    echo "[setup] ssh $H100_HOST : git pull + uv sync (branch $REMOTE_BRANCH)" | tee -a "$MASTER_LOG"
    if ! rsh "git fetch --all --quiet && git checkout $REMOTE_BRANCH && git pull --ff-only --quiet && uv sync --quiet"; then
        echo "[setup] FATAL: remote provision (git/uv) failed on $H100_HOST" | tee -a "$MASTER_LOG"
        return 1
    fi

    echo "[setup] verifying CUDA + native fp64 ..." | tee -a "$MASTER_LOG"
    # Quick fp64 GEMM timing: H100 lands in the tens of TFLOP/s; a crippled card
    # (RTX 3050, fp64 ~1/64) lands near ~0.1, which we warn on loudly.
    local probe
    probe='
import torch, time
if not torch.cuda.is_available():
    print("CUDA_AVAILABLE=0"); raise SystemExit(0)
d = torch.device("cuda")
name = torch.cuda.get_device_name(d)
free, total = torch.cuda.mem_get_info(d)
print(f"CUDA_AVAILABLE=1")
print(f"GPU={name}")
print(f"MEM_FREE_GB={free/1e9:.1f}")
print(f"MEM_TOTAL_GB={total/1e9:.1f}")
n = 4096
a = torch.randn(n, n, dtype=torch.float64, device=d)
b = torch.randn(n, n, dtype=torch.float64, device=d)
for _ in range(3):
    torch.matmul(a, b)
torch.cuda.synchronize()
t = time.perf_counter()
reps = 10
for _ in range(reps):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / reps
tflops = 2 * n**3 / dt / 1e12
print(f"FP64_GEMM_TFLOPS={tflops:.2f}")
print("FP64_NATIVE=" + ("1" if tflops > 5.0 else "0"))
'
    local out
    if ! out="$(rsh "uv run python -c '$probe'" 2>&1)"; then
        echo "[setup] FATAL: fp64/CUDA probe failed to run:" | tee -a "$MASTER_LOG"
        echo "$out" | tee -a "$MASTER_LOG"
        return 1
    fi
    echo "$out" | tee -a "$MASTER_LOG"

    if ! grep -q "CUDA_AVAILABLE=1" <<< "$out"; then
        echo "[setup] FATAL: torch.cuda.is_available() is False on $H100_HOST" | tee -a "$MASTER_LOG"
        return 1
    fi
    if grep -q "FP64_NATIVE=0" <<< "$out"; then
        echo "[setup] WARNING: fp64 GEMM is SLOW (<5 TFLOP/s) -- this is NOT a native-fp64 datacenter card." | tee -a "$MASTER_LOG"
        echo "[setup] WARNING: fp64-gated items (rr_stack_recheck, flapw, dmi) will not reflect H100 behavior." | tee -a "$MASTER_LOG"
    fi
    return 0
}

if [[ "$SKIP_SETUP" -eq 0 ]]; then
    if ! setup_phase; then
        echo "[backlog] setup failed -- aborting before any item." | tee -a "$MASTER_LOG"
        exit 1
    fi
else
    echo "[setup] skipped (--skip-setup)" | tee -a "$MASTER_LOG"
fi

if [[ "$SETUP_ONLY" -eq 1 ]]; then
    echo "[backlog] --setup-only: stopping after provision check." | tee -a "$MASTER_LOG"
    exit 0
fi

# --------------------------------------------------------------------------- #
# Run one item
# --------------------------------------------------------------------------- #
declare -A RESULT   # id -> DONE|SKIP-done|SKIP-build|SKIP-missing|FAIL|OK

run_item() {
    local id="$1" tier="$2" status="$3" branch="$4" entry="$5" cmd="$6" desc="$7"
    local dir="$RESULTS_DIR/$id"
    local log="$dir/run.log"
    mkdir -p "$dir"

    # BUILD-FIRST: no driver exists yet.
    if [[ "$status" == "build" ]]; then
        echo "[$id] SKIPPED (driver not built): $desc" | tee -a "$MASTER_LOG"
        RESULT["$id"]="SKIP-build"
        return 0
    fi

    # Idempotency: skip completed items unless --force.
    if [[ -f "$dir/DONE" && "$FORCE" -eq 0 ]]; then
        echo "[$id] SKIPPED (already DONE; --force to re-run)" | tee -a "$MASTER_LOG"
        RESULT["$id"]="SKIP-done"
        return 0
    fi

    # Optional per-item branch checkout on the remote.
    if [[ "$branch" != "-" ]]; then
        echo "[$id] remote checkout branch $branch" | tee -a "$MASTER_LOG"
        if ! rsh "git fetch --all --quiet && git checkout $branch --quiet && git pull --ff-only --quiet 2>/dev/null; uv sync --quiet"; then
            echo "[$id] SKIPPED (branch $branch unavailable on remote -- restore/rebase it first)" | tee -a "$MASTER_LOG"
            RESULT["$id"]="SKIP-missing"
            return 0
        fi
    fi

    # RUN-READY at declaration but entrypoint file absent on the remote -> runtime SKIP.
    if [[ "$entry" != "-" ]] && ! rsh "test -f $entry"; then
        echo "[$id] SKIPPED (entrypoint missing on remote: $entry -- see docs/h100_backlog.md)" | tee -a "$MASTER_LOG"
        RESULT["$id"]="SKIP-missing"
        # restore the session branch if we moved off it
        [[ "$branch" != "-" ]] && rsh "git checkout $REMOTE_BRANCH --quiet" >/dev/null 2>&1
        return 0
    fi

    echo "[$id] RUN ($tier): $cmd" | tee -a "$MASTER_LOG"
    local start; start=$(date +%s)
    # Detached-safe, bounded, greppable: the whole remote command is time-limited
    # and stamps an EXIT marker into the item log regardless of outcome.
    {
        echo "=== $id @ $(date -u +%FT%TZ) branch=${branch/-/$REMOTE_BRANCH} ==="
        echo "=== cmd: $cmd ==="
        rsh "timeout $ITEM_TIMEOUT bash -lc '$cmd'"
        echo "EXIT=$?"
    } > "$log" 2>&1
    local rc; rc=$(grep -oE 'EXIT=[0-9]+' "$log" | tail -1 | cut -d= -f2)
    local elapsed=$(( $(date +%s) - start ))

    # restore the session branch if this item used its own
    [[ "$branch" != "-" ]] && rsh "git checkout $REMOTE_BRANCH --quiet" >/dev/null 2>&1

    if [[ "$rc" == "0" ]]; then
        touch "$dir/DONE"
        echo "[$id] OK (${elapsed}s) -> $log" | tee -a "$MASTER_LOG"
        RESULT["$id"]="OK"
    else
        echo "[$id] FAIL (rc=$rc, ${elapsed}s) -> $log  [continuing]" | tee -a "$MASTER_LOG"
        RESULT["$id"]="FAIL"
    fi
    return 0
}

# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
while IFS=$'\t' read -r id tier status branch entry cmd desc; do
    _selected "$id" || continue
    run_item "$id" "$tier" "$status" "$branch" "$entry" "$cmd" "$desc"
done < <(backlog_records)

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo | tee -a "$MASTER_LOG"
echo "[backlog] ===== SUMMARY =====" | tee -a "$MASTER_LOG"
n_ok=0; n_fail=0; n_skip=0
while IFS=$'\t' read -r id tier status branch entry cmd desc; do
    _selected "$id" || continue
    r="${RESULT[$id]:-(not run)}"
    case "$r" in
        OK)          n_ok=$((n_ok+1)) ;;
        FAIL)        n_fail=$((n_fail+1)) ;;
        SKIP-*)      n_skip=$((n_skip+1)) ;;
    esac
    printf '  %-24s %-6s %s\n' "$id" "$tier" "$r" | tee -a "$MASTER_LOG"
done < <(backlog_records)
echo "[backlog] ok=$n_ok fail=$n_fail skipped=$n_skip  logs in $RESULTS_DIR" | tee -a "$MASTER_LOG"
echo "BACKLOG_EXIT=$(( n_fail > 0 ? 1 : 0 ))" | tee -a "$MASTER_LOG"
exit $(( n_fail > 0 ? 1 : 0 ))
