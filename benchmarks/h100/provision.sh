#!/usr/bin/env bash
# provision.sh -- one-shot, idempotent setup for the 4x H100 benchmark pack.
#
# Meant to be the vast.ai PROVISIONING_SCRIPT (a URL'd script run once on first
# boot) OR run by hand after SSH:
#
#   curl -fsSL <raw-url>/benchmarks/h100/provision.sh | bash
#
# It clones gradwave (public https), syncs the project venv with uv, records the
# torch/CUDA and nvidia-smi environment into the results dir, then launches
# run_all.sh detached under nohup with a master log. Re-running is safe: the
# clone becomes a pull, uv sync is idempotent, and each launch stamps a fresh
# timestamped master log.
#
# It does NOT assume the PyTorch template's /venv/main -- the environment is the
# project venv that `uv sync` builds, addressed only through `uv run`.
set -euo pipefail

REPO_URL="${GRADWAVE_REPO_URL:-https://github.com/wladerer/gradwave.git}"
REPO_DIR="${GRADWAVE_DIR:-$HOME/gradwave}"
BRANCH="${GRADWAVE_BRANCH:-main}"

echo "[provision] repo=$REPO_URL branch=$BRANCH dir=$REPO_DIR"

# 1. clone or update ------------------------------------------------------- #
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "[provision] existing checkout -> fetch + reset to origin/$BRANCH"
    git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
    echo "[provision] cloning"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
HOST="$(hostname)"
RESDIR="$REPO_DIR/benchmarks/results/$HOST"
mkdir -p "$RESDIR"

# 2. sync the project venv (this, not /venv/main, is the environment) ------ #
if ! command -v uv >/dev/null 2>&1; then
    echo "[provision] uv not found -- installing to ~/.local/bin" >&2
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[provision] uv sync ..."
uv sync

# 3. record the environment ------------------------------------------------ #
echo "[provision] recording environment snapshot -> $RESDIR"
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda', torch.cuda.device_count())" \
    | tee "$RESDIR/torch_env.txt" || echo "torch import FAILED" | tee "$RESDIR/torch_env.txt"
{
    echo "host: $HOST"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_sha: $(git rev-parse --short HEAD)"
    echo "uv: $(uv --version 2>/dev/null || echo missing)"
} > "$RESDIR/provision_meta.txt"
nvidia-smi > "$RESDIR/nvidia_smi.txt" 2>&1 || echo "nvidia-smi unavailable" > "$RESDIR/nvidia_smi.txt"

# 4. launch the lanes detached with a master log --------------------------- #
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="$RESDIR/run_all-$TS.log"
echo "[provision] launching run_all.sh (nohup) -> $MASTER_LOG"
cd "$REPO_DIR/benchmarks/h100"
nohup bash run_all.sh > "$MASTER_LOG" 2>&1 &
echo "[provision] run_all.sh PID $! ; master log $MASTER_LOG"
echo "[provision] tail with:  tail -f $MASTER_LOG"
echo "[provision] done. When RUN_ALL_EXIT appears in the master log, run:"
echo "             uv run python $REPO_DIR/benchmarks/h100/collect.py"
