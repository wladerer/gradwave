#!/usr/bin/env bash
# Build the native Davidson solver library (solvers/native/davidson_native.c)
# into ~/.cache/gradwave/libdavnative.so (override the destination with
# GRADWAVE_NATIVE_SO). One-time per machine; the solver is then selected with
# `scf.eigensolver: davidson-native` (see solvers/native_davidson.py).
#
# NixOS: nix-shell supplies gcc + fftw + openblas headers/libs, and the
# cc/ld wrappers bake store-path rpaths into the .so so ctypes.CDLL resolves
# them with no environment fixup. Other distros: any gcc with libfftw3-dev
# and libopenblas-dev (or another CBLAS/LAPACKE) works — see the CI job.
# -march=native is stripped under nix (NIX_ENFORCE_NO_NATIVE); x86-64-v3
# (AVX2+FMA) is the portable modern baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
src=src/gradwave/solvers/native/davidson_native.c
out="${GRADWAVE_NATIVE_SO:-$HOME/.cache/gradwave/libdavnative.so}"
mkdir -p "$(dirname "$out")"
# Bake a short hash of the source into the .so; the adapter refuses a library
# whose stamp does not match the checked-out source (stale-.so segfault guard,
# see solvers/native_davidson.py). Must match the adapter's hash: sha256 of the
# raw file bytes, first 16 hex chars.
srchash=$(sha256sum "$src" | cut -c1-16)
flags="-O3 -march=x86-64-v3 -funroll-loops -fopenmp -shared -fPIC -DGW_NATIVE_SRC_HASH=\"$srchash\""
if command -v nix-shell >/dev/null 2>&1; then
  nix-shell -p gcc fftw openblas --run \
    "gcc $flags $src -o '$out' -lfftw3 -lopenblas -lm"
else
  gcc $flags "$src" -o "$out" -lfftw3 -lopenblas -llapacke -lm
fi
echo "built $out (src hash $srchash)"
