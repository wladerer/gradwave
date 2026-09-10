#!/usr/bin/env bash
# Build the native solver library into ~/.cache/gradwave/libdavnative.so
# (override the destination with GRADWAVE_NATIVE_SO). One-time per machine.
# One .so, TWO symbols compiled from two sources:
#   * davidson_native.c — the full native block Davidson (solvers/native_davidson.py),
#     selected with `scf.eigensolver: davidson-native`.
#   * blas_gemm.c — the batched zgemm the eager Davidson routes its Rayleigh-Ritz
#     GEMMs through (solvers/blas_routing.py), gated by GRADWAVE_BLAS_GEMM.
#
# NixOS: nix-shell supplies gcc + fftw + openblas headers/libs, and the
# cc/ld wrappers bake store-path rpaths into the .so so ctypes.CDLL resolves
# them with no environment fixup. Other distros: any gcc with libfftw3-dev
# and libopenblas-dev (or another CBLAS/LAPACKE) works — see the CI job.
# -march=native is stripped under nix (NIX_ENFORCE_NO_NATIVE); x86-64-v3
# (AVX2+FMA) is the portable modern baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
src="src/gradwave/solvers/native/davidson_native.c src/gradwave/solvers/native/blas_gemm.c"
out="${GRADWAVE_NATIVE_SO:-$HOME/.cache/gradwave/libdavnative.so}"
mkdir -p "$(dirname "$out")"
flags="-O3 -march=x86-64-v3 -funroll-loops -fopenmp -shared -fPIC"
if command -v nix-shell >/dev/null 2>&1; then
  nix-shell -p gcc fftw openblas --run \
    "gcc $flags $src -o '$out' -lfftw3 -lopenblas -lm"
else
  gcc $flags $src -o "$out" -lfftw3 -lopenblas -llapacke -lm
fi
echo "built $out"
