#!/usr/bin/env bash
# Compile the native Davidson kernel. NixOS: nix-shell supplies gcc + headers
# + libs and the cc/ld wrappers bake store-path rpaths into the .so, so the
# uv-managed Python's ctypes.CDLL resolves fftw3/openblas with no env fixup.
set -euo pipefail
cd "$(dirname "$0")"
# -march=native is stripped by NIX_ENFORCE_NO_NATIVE; x86-64-v3 (AVX2+FMA)
# passes and covers both the thinkpad and asus.
nix-shell -p gcc fftw openblas --run \
  "gcc -O3 -march=x86-64-v3 -funroll-loops -fopenmp -shared -fPIC \
   davidson_native.c -o libdavnative.so -lfftw3 -lopenblas -lm"
echo "built libdavnative.so"
