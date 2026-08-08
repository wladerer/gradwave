#!/usr/bin/env bash
# provision_bfo.sh -- one-shot setup for the BiFeO3 PAW+U H100 run.
#
# Paste-and-go on a freshly rented CUDA box (vast.ai verified datacenter host,
# non-interruptible, >=50 GB disk). Either set this as the vast.ai
# PROVISIONING_SCRIPT, or after SSH run:
#
#   curl -fsSL <raw-url>/benchmarks/h100/bfo/provision_bfo.sh | bash
#
# Or fully by hand:
#   git clone --depth 1 -b worktree-uspp-softmode-deflate \
#       https://github.com/wladerer/gradwave.git ~/gradwave
#   bash ~/gradwave/benchmarks/h100/bfo/provision_bfo.sh
#
# Clones the branch (PAW+U soft-mode deflation lives on it, not yet on main),
# uv-syncs the project venv, downloads the Bi PAW pseudo (Fe/O kjpaw ship in the
# repo), records the CUDA env, and launches bfo_h100_driver.py DETACHED under
# setsid+nohup so it survives an SSH drop. Idempotent.
set -euo pipefail

REPO_URL="${GRADWAVE_REPO_URL:-https://github.com/wladerer/gradwave.git}"
REPO_DIR="${GRADWAVE_DIR:-$HOME/gradwave}"
BRANCH="${GRADWAVE_BRANCH:-worktree-uspp-softmode-deflate}"
BI_URL="https://pseudopotentials.quantum-espresso.org/upf_files/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF"

echo "[bfo] repo=$REPO_URL branch=$BRANCH dir=$REPO_DIR"

# 1. clone / update -------------------------------------------------------- #
if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# 2. project venv ---------------------------------------------------------- #
if ! command -v uv >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[bfo] uv sync (this pulls torch — the slow step) ..."
uv sync

# 3. Bi PAW pseudo (Fe/O kjpaw already in fixtures) ------------------------- #
PSDIR="$REPO_DIR/tests/fixtures/qe/pseudos"
if [[ ! -s "$PSDIR/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF" ]]; then
    echo "[bfo] downloading Bi PAW pseudo ..."
    curl -fsSL -o "$PSDIR/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF" "$BI_URL"
fi
for f in Bi.pbe-dn-kjpaw_psl.1.0.0.UPF Fe.pbe-spn-kjpaw_psl.1.0.0.UPF O.pbe-n-kjpaw_psl.1.0.0.UPF; do
    [[ -s "$PSDIR/$f" ]] || { echo "[bfo] MISSING pseudo $f — aborting" >&2; exit 1; }
done

# 4. env snapshot ---------------------------------------------------------- #
RESDIR="$REPO_DIR/benchmarks/results/$(hostname)"
mkdir -p "$RESDIR"
uv run python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-CUDA')" \
    | tee "$RESDIR/torch_env.txt" || echo "torch import FAILED" | tee "$RESDIR/torch_env.txt"
nvidia-smi > "$RESDIR/nvidia_smi.txt" 2>&1 || echo "no nvidia-smi" > "$RESDIR/nvidia_smi.txt"
git rev-parse --short HEAD > "$RESDIR/git_sha.txt"

# 5. launch the driver DETACHED (survives SSH drop) ------------------------ #
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RESDIR/bfo_h100-$TS.log"
echo "[bfo] launching driver (setsid+nohup) -> $LOG"
cd "$REPO_DIR"
setsid nohup stdbuf -oL uv run python -u benchmarks/h100/bfo/bfo_h100_driver.py \
    > "$LOG" 2>&1 < /dev/null &
echo "[bfo] PID $! ; log $LOG"
echo
echo "[bfo] watch:     tail -f $LOG"
echo "[bfo] result:    $RESDIR/bfo_h100.json   (written incrementally)"
echo "[bfo] done when 'BFO_H100_DONE' appears; then scp the json + log back."
