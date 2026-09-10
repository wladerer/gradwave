/* Native (zero-dispatch) batched block-Davidson round loop — research probe.
 *
 * Reimplements the synchronous fp64 path of gradwave's davidson_batched
 * (solvers/davidson.py) plus the FFT-path BatchedHamiltonian.apply
 * (core/batch.py) as one C call: FFTW3 for the dense-box transforms, CBLAS
 * zgemm for every contraction, LAPACKE zheevd/zgeqrf/zungqr for the subspace
 * eigensolve and orthonormalization. No Python, no dispatcher, no allocator
 * churn between ops — this measures how much of the eager-mode "glue" wall
 * (~48% at small cells) converts to real speedup when the inner loop leaves
 * the framework.
 *
 * Scope (asserted, not silently approximated): fp64 only, no fp32-expansion,
 * no complex64 storage, no DFT+U, no sync_free, no restart (caller sizes
 * max_dim so the measured rounds fit; hitting it aborts), no rank-repair
 * jitter (degenerate rows abort). Convergence logic mirrors davidson_batched:
 * per-round RR -> residual norms -> early exit on rn.max() < tol -> n_add =
 * max_k #(rn_k > tol) -> top-n_add residuals -> Teter precond -> unit-norm ->
 * two-pass projection against V -> QR -> append + H-apply.
 *
 * Memory layout: row-major everywhere, complex128 interleaved (C99 double
 * complex == numpy complex128 == torch complex128 on CPU).
 */

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>

typedef double complex c128;

/* ------------------------------------------------------------------ */
/* problem context                                                     */

typedef struct {
    int64_t nk, nb, m;      /* k-points, bands, padded plane waves    */
    int64_t n1, n2, n3, n;  /* dense box dims, n = n1*n2*n3           */
    int64_t nproj;
    const double *t;        /* (nk, m) kinetic diag (also precond)    */
    const uint8_t *mask;    /* (nk, m)                                */
    const int64_t *idx_sc;  /* (nk, m) scatter, trash slot == n       */
    const int64_t *idx_ga;  /* (nk, m) gather                         */
    const double *v_eff;    /* (n,) real                              */
    const c128 *p;          /* (nk, nproj, m)                         */
    const c128 *pc;         /* conj(p), precomputed once              */
    const c128 *dij;        /* (nproj, nproj)                         */
    fftw_plan plan_f, plan_b;   /* single-box out-of-place c2c        */
    int nthreads;
} ctx_t;

/* ------------------------------------------------------------------ */
/* H apply: out(overwritten) = H c for a (nk, nbc, m) block            */

static void h_apply(const ctx_t *cx, const c128 *c, c128 *out, int64_t nbc,
                    c128 *becp_ws, c128 *tmp_ws) {
    const int64_t nk = cx->nk, m = cx->m, n = cx->n, np = cx->nproj;
    /* kinetic diagonal */
    for (int64_t k = 0; k < nk; k++)
        for (int64_t b = 0; b < nbc; b++) {
            const double *tk = cx->t + k * m;
            const c128 *ck = c + (k * nbc + b) * m;
            c128 *ok = out + (k * nbc + b) * m;
            for (int64_t g = 0; g < m; g++) ok[g] = tk[g] * ck[g];
        }
    /* local term: scatter -> ifft -> * v_eff/n -> fft -> gather, per (k,b) */
    const double invn = 1.0 / (double)n;
#pragma omp parallel num_threads(cx->nthreads)
    {
        c128 *box = fftw_alloc_complex(n + 1);
        c128 *work = fftw_alloc_complex(n);
#pragma omp for collapse(2) schedule(static)
        for (int64_t k = 0; k < nk; k++)
            for (int64_t b = 0; b < nbc; b++) {
                const c128 *ck = c + (k * nbc + b) * m;
                const int64_t *sc = cx->idx_sc + k * m;
                const int64_t *ga = cx->idx_ga + k * m;
                memset(box, 0, sizeof(c128) * (size_t)(n + 1));
                for (int64_t g = 0; g < m; g++) box[sc[g]] = ck[g];
                fftw_execute_dft(cx->plan_b, box, work);           /* ifft, unnormalized */
                for (int64_t i = 0; i < n; i++) work[i] *= cx->v_eff[i] * invn;
                fftw_execute_dft(cx->plan_f, work, box);           /* fft */
                c128 *ok = out + (k * nbc + b) * m;
                for (int64_t g = 0; g < m; g++) ok[g] += box[ga[g]];
            }
        fftw_free(box);
        fftw_free(work);
    }
    /* nonlocal: becp = C P^H ; out += (becp dij) P */
    if (np > 0) {
        const c128 one = 1.0, zero = 0.0;
        for (int64_t k = 0; k < nk; k++) {
            const c128 *Ck = c + k * nbc * m;
            const c128 *Pk = cx->p + k * np * m;
            c128 *Bk = becp_ws;               /* (nbc, np) */
            c128 *Tk = tmp_ws;                /* (nbc, np) */
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                        (int)nbc, (int)np, (int)m, &one, Ck, (int)m,
                        Pk, (int)m, &zero, Bk, (int)np);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)nbc, (int)np, (int)np, &one, Bk, (int)np,
                        cx->dij, (int)np, &zero, Tk, (int)np);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)nbc, (int)m, (int)np, &one, Tk, (int)np,
                        Pk, (int)m, &one, out + k * nbc * m, (int)m);
        }
    }
    /* mask */
    for (int64_t k = 0; k < nk; k++) {
        const uint8_t *mk = cx->mask + k * m;
        for (int64_t b = 0; b < nbc; b++) {
            c128 *ok = out + (k * nbc + b) * m;
            for (int64_t g = 0; g < m; g++)
                if (!mk[g]) ok[g] = 0.0;
        }
    }
}

/* ------------------------------------------------------------------ */
/* per-k QR row-orthonormalization of a (rows, m) block, in place.
 * Mirrors _orthonormalize_b's qr(x^T) -> q^T: LAPACK zgeqrf/zungqr on the
 * (m, rows) transpose. Returns 0, or -1 on a (near-)zero row (jitter path
 * in the reference — out of probe scope).                              */

static int qr_rows(c128 *x, int64_t rows, int64_t m, c128 *ws_t, c128 *r_out) {
    /* transpose into ws_t (m, rows) row-major; optionally return R (rows,rows) */
    for (int64_t j = 0; j < rows; j++)
        for (int64_t g = 0; g < m; g++) ws_t[g * rows + j] = x[j * m + g];
    c128 *tau = malloc(sizeof(c128) * (size_t)rows);
    int info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, (int)m, (int)rows, ws_t,
                              (int)rows, tau);
    if (info == 0 && r_out != NULL)
        for (int64_t i = 0; i < rows; i++)
            for (int64_t j = 0; j < rows; j++)
                r_out[i * rows + j] = j >= i ? ws_t[i * rows + j] : 0.0;
    if (info == 0)
        info = LAPACKE_zungqr(LAPACK_ROW_MAJOR, (int)m, (int)rows, (int)rows,
                              ws_t, (int)rows, tau);
    free(tau);
    if (info != 0) return -1;
    for (int64_t j = 0; j < rows; j++)
        for (int64_t g = 0; g < m; g++) x[j * m + g] = ws_t[g * rows + j];
    return 0;
}

/* project rows of d (nd, m) against rows of v (dim, m), two passes:
 * d -= (d v^H) v                                                      */
static void project_out(c128 *d, const c128 *v, int64_t nd, int64_t dim,
                        int64_t m, c128 *gram_ws) {
    const c128 one = 1.0, zero = 0.0, neg = -1.0;
    for (int pass = 0; pass < 2; pass++) {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, (int)nd,
                    (int)dim, (int)m, &one, d, (int)m, v, (int)m, &zero,
                    gram_ws, (int)dim);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)nd,
                    (int)m, (int)dim, &neg, gram_ws, (int)dim, v, (int)m,
                    &one, d, (int)m);
    }
}

/* ------------------------------------------------------------------ */
/* the solve. Returns rounds executed (>0) or a negative error code.
 *   -1 degenerate row in orthonormalization   -2 LAPACK failure
 *   -3 restart would be required (max_dim hit)                        */

int davidson_native(
    /* sizes */
    int64_t nk, int64_t nb, int64_t m, int64_t n1, int64_t n2, int64_t n3,
    int64_t nproj, int64_t max_dim, int64_t max_iter, double tol,
    int nthreads,
    /* inputs (row-major, as documented above) */
    const c128 *x0, const double *t, const uint8_t *mask,
    const int64_t *idx_sc, const int64_t *idx_ga, const double *v_eff,
    const c128 *p, const c128 *dij,
    /* outputs */
    double *eig_out,   /* (nk, nb) */
    c128 *x_out,       /* (nk, nb, m) */
    double *rn_out,    /* (nk, nb) */
    int64_t *napply_out /* band*k H-apply tally, matches reference counter */)
{
    ctx_t cx = {.nk = nk, .nb = nb, .m = m, .n1 = n1, .n2 = n2, .n3 = n3,
                .n = n1 * n2 * n3, .nproj = nproj, .t = t, .mask = mask,
                .idx_sc = idx_sc, .idx_ga = idx_ga, .v_eff = v_eff, .p = p,
                .dij = dij, .nthreads = nthreads};
    const int64_t n = cx.n;

    /* plans: one-box c2c, out of place; FFTW_MEASURE amortized across calls
     * by the caller keeping the process alive (plans cached via wisdom would
     * be overkill for a probe — ESTIMATE is within a few % for these sizes) */
    c128 *pb1 = fftw_alloc_complex(n + 1), *pb2 = fftw_alloc_complex(n);
    int dims[3] = {(int)n1, (int)n2, (int)n3};
    cx.plan_f = fftw_plan_dft(3, dims, pb2, pb1, FFTW_FORWARD, FFTW_ESTIMATE);
    cx.plan_b = fftw_plan_dft(3, dims, pb1, pb2, FFTW_BACKWARD, FFTW_ESTIMATE);

    /* precompute conj(p) not needed: zgemm ConjTrans covers both uses */
    cx.pc = NULL;

    /* workspaces */
    c128 *V = malloc(sizeof(c128) * (size_t)(nk * max_dim * m));
    c128 *HV = malloc(sizeof(c128) * (size_t)(nk * max_dim * m));
    c128 *X = malloc(sizeof(c128) * (size_t)(nk * nb * m));
    c128 *HX = malloc(sizeof(c128) * (size_t)(nk * nb * m));
    c128 *D = malloc(sizeof(c128) * (size_t)(nk * nb * m));
    c128 *S = malloc(sizeof(c128) * (size_t)(nk * max_dim * max_dim));
    double *W = malloc(sizeof(double) * (size_t)max_dim);
    c128 *becp_ws = malloc(sizeof(c128) * (size_t)(nb * (nproj > 0 ? nproj : 1)));
    c128 *tmp_ws = malloc(sizeof(c128) * (size_t)(nb * (nproj > 0 ? nproj : 1)));
    c128 *gram_ws = malloc(sizeof(c128) * (size_t)(nb * max_dim));
    c128 *qr_ws = malloc(sizeof(c128) * (size_t)(m * max_dim));
    c128 *hd = malloc(sizeof(c128) * (size_t)(nk * nb * m));
    double *rn = malloc(sizeof(double) * (size_t)(nk * nb));
    double *tband = malloc(sizeof(double) * (size_t)nb);
    int64_t *sel = malloc(sizeof(int64_t) * (size_t)nb);
    int64_t napply = 0;
    int ret = -2;

    /* ---- init: V = orthonormalize(x0 * mask); HV = H V ---- */
    int64_t dim = nb;
    for (int64_t k = 0; k < nk; k++) {
        c128 *Vk = V + k * max_dim * m;  /* rows contiguous per k while dim==nb */
        const c128 *xk = x0 + k * nb * m;
        const uint8_t *mk = mask + k * m;
        for (int64_t b = 0; b < nb; b++)
            for (int64_t g = 0; g < m; g++)
                Vk[b * m + g] = mk[g] ? xk[b * m + g] : 0.0;
        /* degenerate-row check mirrors _orthonormalize_b's 1e-8 gate */
        for (int64_t b = 0; b < nb; b++) {
            double s = 0;
            for (int64_t g = 0; g < m; g++) {
                double re = creal(Vk[b * m + g]), im = cimag(Vk[b * m + g]);
                s += re * re + im * im;
            }
            if (sqrt(s) < 1e-8) { ret = -1; goto done; }
        }
        if (qr_rows(Vk, nb, m, qr_ws) != 0) { ret = -2; goto done; }
    }
    /* NOTE: V rows for k are stored at stride max_dim*m per k; H-apply and
     * GEMMs below always address the (dim) leading rows via Vk pointers, so
     * per-k blocks stay independent. h_apply expects (nk, nbc, m) contiguous;
     * we apply per k to keep the strided layout simple. */
    for (int64_t k = 0; k < nk; k++) {
        ctx_t ck1 = cx; ck1.nk = 1;
        ck1.t = t + k * m; ck1.mask = mask + k * m;
        ck1.idx_sc = idx_sc + k * m; ck1.idx_ga = idx_ga + k * m;
        ck1.p = p + k * nproj * m;
        h_apply(&ck1, V + k * max_dim * m, HV + k * max_dim * m, nb,
                becp_ws, tmp_ws);
    }
    napply += nk * nb;

    int64_t it;
    for (it = 1; it <= max_iter; it++) {
        /* ---- Rayleigh-Ritz ---- */
        const c128 one = 1.0, zero = 0.0;
        for (int64_t k = 0; k < nk; k++) {
            const c128 *Vk = V + k * max_dim * m;
            const c128 *HVk = HV + k * max_dim * m;
            c128 *Sk = S + k * max_dim * max_dim;
            /* D = V HV^H ; s = conj(D) ; symmetrize */
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, (int)dim,
                        (int)dim, (int)m, &one, Vk, (int)m, HVk, (int)m,
                        &zero, Sk, (int)dim);
            for (int64_t i = 0; i < dim; i++)
                for (int64_t j = i; j < dim; j++) {
                    c128 sij = conj(Sk[i * dim + j]);
                    c128 sji = conj(Sk[j * dim + i]);
                    c128 hij = 0.5 * (sij + conj(sji));
                    Sk[i * dim + j] = hij;
                    Sk[j * dim + i] = conj(hij);
                }
            int info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', (int)dim,
                                      Sk, (int)dim, W);
            if (info != 0) { ret = -2; goto done; }
            for (int64_t b = 0; b < nb; b++) eig_out[k * nb + b] = W[b];
            /* X = U[:, :nb]^T V ; HX likewise (U unconjugated, ref convention) */
            c128 *Xk = X + k * nb * m, *HXk = HX + k * nb * m;
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)nb,
                        (int)m, (int)dim, &one, Sk, (int)dim, Vk, (int)m,
                        &zero, qr_ws, (int)m); /* qr_ws misused as (nb,m) tmp: fits (m*max_dim) */
            memcpy(Xk, qr_ws, sizeof(c128) * (size_t)(nb * m));
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)nb,
                        (int)m, (int)dim, &one, Sk, (int)dim, HVk, (int)m,
                        &zero, qr_ws, (int)m);
            memcpy(HXk, qr_ws, sizeof(c128) * (size_t)(nb * m));
            /* wait — U columns 0..nb-1 with CblasTrans uses FULL row length as
             * lda=dim and M=nb: rows of U^T are columns of U — correct: op(A)
             * is (nb, dim) taken from the first nb columns of U. */
            for (int64_t b = 0; b < nb; b++) {
                double s = 0, e = W[b];
                c128 *xb = Xk + b * m, *hb = HXk + b * m;
                for (int64_t g = 0; g < m; g++) {
                    c128 r = hb[g] - e * xb[g];
                    double re = creal(r), im = cimag(r);
                    s += re * re + im * im;
                }
                rn[k * nb + b] = sqrt(s);
            }
        }
        /* ---- convergence / n_add (uniform across k) ---- */
        double rnmax = 0;
        int64_t n_add = 0;
        for (int64_t k = 0; k < nk; k++) {
            int64_t cnt = 0;
            for (int64_t b = 0; b < nb; b++) {
                if (rn[k * nb + b] > rnmax) rnmax = rn[k * nb + b];
                if (rn[k * nb + b] > tol) cnt++;
            }
            if (cnt > n_add) n_add = cnt;
        }
        if (rnmax < tol) break;
        int restart = (dim + n_add > max_dim);

        /* ---- expansion: top-n_add residuals, Teter, ortho, append ---- */
        for (int64_t k = 0; k < nk; k++) {
            const double *tk = t + k * m;
            c128 *Xk = X + k * nb * m, *HXk = HX + k * nb * m;
            /* selection: argsort rn descending, take n_add */
            for (int64_t b = 0; b < nb; b++) sel[b] = b;
            for (int64_t i = 0; i < n_add; i++) {  /* partial selection sort */
                int64_t best = i;
                for (int64_t j = i + 1; j < nb; j++)
                    if (rn[k * nb + sel[j]] > rn[k * nb + sel[best]]) best = j;
                int64_t tmp = sel[i]; sel[i] = sel[best]; sel[best] = tmp;
            }
            /* t_band for selected, then d = teter(r_sel) row-normalized */
            c128 *Dk = D + k * nb * m;
            for (int64_t i = 0; i < n_add; i++) {
                int64_t b = sel[i];
                double tb = 0;
                c128 *xb = Xk + b * m;
                for (int64_t g = 0; g < m; g++) {
                    double re = creal(xb[g]), im = cimag(xb[g]);
                    tb += tk[g] * (re * re + im * im);
                }
                tband[i] = tb < 1e-12 ? 1e-12 : tb;
                double e = eig_out[k * nb + b];
                c128 *hb = HXk + b * m, *db = Dk + i * m;
                double nrm2 = 0;
                for (int64_t g = 0; g < m; g++) {
                    double xx = tk[g] / tband[i];
                    double x2 = xx * xx;
                    double num = 27.0 + 18.0 * xx + 12.0 * x2 + 8.0 * x2 * xx;
                    double coeff = num / (num + 16.0 * x2 * x2);
                    c128 r = hb[g] - e * xb[g];
                    db[g] = r * coeff;
                    double re = creal(db[g]), im = cimag(db[g]);
                    nrm2 += re * re + im * im;
                }
                double nrm = sqrt(nrm2);
                if (nrm > 1e-300)
                    for (int64_t g = 0; g < m; g++) db[g] /= nrm;
            }
            /* restart: collapse the basis to the (re-orthonormalized) Ritz
             * block, transform its images by the same triangular factor
             * (x_old = R^T x_new  =>  hx_new = (R^T)^{-1} hx_old), and ortho
             * d against the collapsed block instead of the full V. */
            c128 *Vk = V + k * max_dim * m;
            c128 *HVk = HV + k * max_dim * m;
            int64_t against_dim = dim;
            if (restart) {
                memcpy(Vk, Xk, sizeof(c128) * (size_t)(nb * m));
                if (qr_rows(Vk, nb, m, qr_ws, gram_ws /* R (nb,nb) */) != 0) {
                    ret = -2; goto done;
                }
                memcpy(HVk, HXk, sizeof(c128) * (size_t)(nb * m));
                const c128 done_ = 1.0;
                cblas_ztrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
                            CblasNonUnit, (int)nb, (int)m, &done_, gram_ws,
                            (int)nb, HVk, (int)m);
                against_dim = nb;
            }
            /* mask, project (2 passes), degenerate check, QR */
            const uint8_t *mk = mask + k * m;
            for (int64_t i = 0; i < n_add; i++)
                for (int64_t g = 0; g < m; g++)
                    if (!mk[g]) Dk[i * m + g] = 0.0;
            project_out(Dk, Vk, n_add, against_dim, m, gram_ws);
            for (int64_t i = 0; i < n_add; i++) {
                double s = 0;
                for (int64_t g = 0; g < m; g++) {
                    double re = creal(Dk[i * m + g]), im = cimag(Dk[i * m + g]);
                    s += re * re + im * im;
                }
                if (sqrt(s) < 1e-8) { ret = -1; goto done; }
            }
            if (qr_rows(Dk, n_add, m, qr_ws, NULL) != 0) { ret = -2; goto done; }
            /* append to V; H-apply into HV */
            memcpy(Vk + against_dim * m, Dk,
                   sizeof(c128) * (size_t)(n_add * m));
            ctx_t ck1 = cx; ck1.nk = 1;
            ck1.t = tk; ck1.mask = mk;
            ck1.idx_sc = idx_sc + k * m; ck1.idx_ga = idx_ga + k * m;
            ck1.p = p + k * nproj * m;
            h_apply(&ck1, Vk + against_dim * m, HVk + against_dim * m,
                    n_add, becp_ws, tmp_ws);
        }
        napply += nk * n_add;
        dim = (restart ? nb : dim) + n_add;
    }

    memcpy(x_out, X, sizeof(c128) * (size_t)(nk * nb * m));
    memcpy(rn_out, rn, sizeof(double) * (size_t)(nk * nb));
    *napply_out = napply;
    ret = (int)(it > max_iter ? max_iter : it);

done:
    fftw_destroy_plan(cx.plan_f);
    fftw_destroy_plan(cx.plan_b);
    fftw_free(pb1); fftw_free(pb2);
    free(V); free(HV); free(X); free(HX); free(D); free(S); free(W);
    free(becp_ws); free(tmp_ws); free(gram_ws); free(qr_ws); free(hd);
    free(rn); free(tband); free(sel);
    return ret;
}
