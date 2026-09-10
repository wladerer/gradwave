/* Strided batched complex128 GEMM via CBLAS (OpenBLAS) zgemm.
 *
 * torch's CPU complex128 matmul is MEASURED 2.6-5x slower than a plain
 * OpenBLAS zgemm on the batched Rayleigh-Ritz shapes the eager Davidson runs
 * (asus: 210 vs 718 us at batch 8 x 754x754; 1237 vs 6354 us at 27x1260x1260),
 * so routing those few RR contractions straight through cblas_zgemm — same
 * math, one BLAS call per batch element, zero framework dispatch — is a small
 * but free win that composes with the k-parallel path. FFTs are already at
 * torch/FFTW parity and are NOT touched.
 *
 * This lives in the SAME shared library as davidson_native.c (built by
 * scripts/build_native_solver.sh) — one .so, two exported symbols. The Python
 * side (solvers/blas_routing.py) loads it with ctypes and falls back to
 * torch.matmul silently when it is absent (identical-math micro-routing).
 *
 * Memory layout: row-major, complex128 interleaved (matches a contiguous
 * torch complex128 CPU tensor's storage), one matrix per batch element at a
 * fixed byte stride.
 *
 * trans codes (transa / transb): 0 = NoTrans, 1 = Trans, 2 = ConjTrans.
 * For CblasRowMajor the leading dimension is the physical column count of the
 * stored operand regardless of the trans flag, so the caller passes M, N, K
 * (the op()-applied dimensions) and the physical column counts lda/ldb; ldc is
 * always N. The caller (Python) computes all of these and asserts the K match.
 */

#include <complex.h>
#include <stdint.h>

#include <cblas.h>

typedef double complex c128;

extern void openblas_set_num_threads(int);

static CBLAS_TRANSPOSE trans_of(int code) {
    if (code == 1) return CblasTrans;
    if (code == 2) return CblasConjTrans;
    return CblasNoTrans;
}

/* C[b] = op(A[b]) @ op(B[b]) for b in [0, batch), each op(A) is (M, K),
 * op(B) is (K, N), C is (M, N). lda/ldb are the physical column counts of the
 * stored A/B matrices; strides are element counts between batch elements. */
void blas_zbmm(int64_t batch, int64_t M, int64_t N, int64_t K,
               int transa, int transb, int64_t lda, int64_t ldb,
               int64_t stride_a, int64_t stride_b, int64_t stride_c,
               const void *A, const void *B, void *C) {
    const c128 one = 1.0, zero = 0.0;
    const c128 *a = (const c128 *)A;
    const c128 *b = (const c128 *)B;
    c128 *c = (c128 *)C;
    const CBLAS_TRANSPOSE ta = trans_of(transa);
    const CBLAS_TRANSPOSE tb = trans_of(transb);
    /* The RR shapes are large enough per call that OpenBLAS threading helps;
     * unlike davidson_native.c (which pins serial inside an OpenMP-over-k
     * region), here there is no outer parallel region, so leave OpenBLAS to
     * use its own thread pool — do not force it to 1. */
    for (int64_t i = 0; i < batch; i++) {
        cblas_zgemm(CblasRowMajor, ta, tb, (int)M, (int)N, (int)K, &one,
                    a + i * stride_a, (int)lda, b + i * stride_b, (int)ldb,
                    &zero, c + i * stride_c, (int)N);
    }
}
