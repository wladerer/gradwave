#!/usr/bin/env bash
# mgpu_wrap.sh -- per-rank launcher for the lane-3 multi-GPU probe. torchrun
# starts one copy of this per rank with LOCAL_RANK set; the wrapper pins that
# rank to its own GPU and execs the gradwave CLI on the shared input.
#
# Invoked by the orchestrator as:
#   python -m torch.distributed.run --nproc-per-node N --no-python \
#       benchmarks/h100/mgpu_wrap.sh benchmarks/h100/inputs/si_dist.yaml
#
# UNTESTED end to end -- the k-shard path uses Gloo (CPU collectives) which may
# stage CUDA tensors through the host; nobody has run it on real multi-GPU.
# Syntax-validated only (bash -n). See README lane 3.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${LOCAL_RANK:-0}"
# GW_PY is the project venv interpreter (sys.executable), passed by the
# orchestrator so we do not pay a nested `uv run` per rank.
PY="${GW_PY:-python}"
exec "$PY" -m gradwave.cli run "$@"
