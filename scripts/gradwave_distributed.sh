#!/usr/bin/env bash
# gradwave_distributed.sh — launch a k-point-sharded gradwave run across N ranks
# with torchrun (Gloo backend). See docs/manual/distributed.md for the full
# picture (what gets distributed, what doesn't, correctness scope).
#
# The input YAML must set `distributed: true` (see inputs.Input.distributed) --
# this script only handles process launch, not the opt-in itself.
#
# Single box, N ranks (the case this repo's tests actually exercise, N=2):
#
#   scripts/gradwave_distributed.sh input.yaml --nproc-per-node 2
#
# Two machines over Tailscale, one rank per machine -- run the SAME command on
# each box, changing only --node-rank. --master-addr must be reachable from
# every node (use the Tailscale IP), and GLOO_SOCKET_IFNAME pins the collective
# to the Tailscale interface rather than whatever LAN/loopback one torchrun's
# rendezvous would otherwise pick:
#
#   # box A (rank 0), Tailscale IP 100.x.y.z:
#   GLOO_SOCKET_IFNAME=tailscale0 scripts/gradwave_distributed.sh input.yaml \
#       --nnodes 2 --node-rank 0 --master-addr 100.x.y.z
#
#   # box B (rank 1):
#   GLOO_SOCKET_IFNAME=tailscale0 scripts/gradwave_distributed.sh input.yaml \
#       --nnodes 2 --node-rank 1 --master-addr 100.x.y.z
#
# Every rank must run against the SAME gradwave revision and the SAME input
# file (in particular the same k-mesh) -- shard_system splits k-points
# deterministically from (nk, rank, world_size), so a mismatch silently
# produces the wrong system on one side rather than an obvious error.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <input.yaml> [--nnodes N] [--nproc-per-node N] [--node-rank N] [--master-addr HOST] [-- <extra gradwave args>]" >&2
    exit 1
fi

input=$1
shift

nnodes=1
nproc_per_node=2
node_rank=0
master_addr=127.0.0.1
extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) nnodes=$2; shift 2 ;;
        --nproc-per-node) nproc_per_node=$2; shift 2 ;;
        --node-rank) node_rank=$2; shift 2 ;;
        --master-addr) master_addr=$2; shift 2 ;;
        --) shift; extra_args=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

exec uv run --project "$repo_root" torchrun \
    --nnodes="$nnodes" --nproc-per-node="$nproc_per_node" \
    --node-rank="$node_rank" \
    --master-addr="$master_addr" --master-port=29500 \
    -m gradwave.cli run "$input" "${extra_args[@]}"
