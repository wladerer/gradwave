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
 * Parallel structure: FFTs parallelize over (k, band); all per-k linear
 * algebra (RR build, eigh, Ritz combine, projections, QR) parallelizes over
 * k with OpenBLAS pinned to 1 thread inside the parallel regions. FFTW plans
 * are cached per box shape across calls (a real SCF plans once per grid).
 *
 * Scope (asserted, not silently approximated): fp64 only, no fp32-expansion,
 * no complex64 storage, no DFT+U, no sync_free, no rank-repair jitter
 * (degenerate rows abort). Restart mirrors the reference: Ritz collapse with
 * QR drift repair + triangular image transform. Convergence logic mirrors
 * davidson_batched exactly (verified: identical n_iter, identical band*k
 * H-apply tally, eigenvalues to ~1e-13).
 *
 * Memory layout: row-major everywhere, complex128 interleaved.
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

extern void openblas_set_num_threads(int);

/* ------------------------------------------------------------------ */
/* cached single-box plans (per box shape, kept across calls)          */

static fftw_plan s_plan_f = NULL, s_plan_b = NULL;
static int s_dims[3] = {0, 0, 0};

/* FFTW_MEASURE with wisdom persisted to a per-user cache: the first plan of a
 * given box shape pays the measurement (~0.1-2 s), every later process reads
 * the wisdom and plans in microseconds. Wisdom is keyed by shape internally
 * by FFTW, so one file accumulates every box ever planned. */
static void ensure_plans(int n1, int n2, int n3) {
    if (s_plan_f && s_dims[0] == n1 && s_dims[1] == n2 && s_dims[2] == n3)
        return;
    if (s_plan_f) { fftw_destroy_plan(s_plan_f); fftw_destroy_plan(s_plan_b); }
    static int wisdom_loaded = 0;
    char wpath[512];
    const char *home = getenv("HOME");
    snprintf(wpath, sizeof wpath, "%s/.cache/gradwave_fftw_wisdom",
             home ? home : "/tmp");
    if (!wisdom_loaded) {
        fftw_import_wisdom_from_filename(wpath);  /* missing file: harmless */
        wisdom_loaded = 1;
    }
    int64_t n = (int64_t)n1 * n2 * n3;
    c128 *a = fftw_alloc_complex(n), *b = fftw_alloc_complex(n);
    int dims[3] = {n1, n2, n3};
    s_plan_f = fftw_plan_dft(3, dims, a, b, FFTW_FORWARD, FFTW_MEASURE);
    s_plan_b = fftw_plan_dft(3, dims, a, b, FFTW_BACKWARD, FFTW_MEASURE);
    fftw_export_wisdom_to_filename(wpath);
    s_dims[0] = n1; s_dims[1] = n2; s_dims[2] = n3;
    fftw_free(a); fftw_free(b);
}

/* ------------------------------------------------------------------ */
/* problem context                                                     */

typedef struct {
    int64_t nk, nb, m, n;
    int64_t nproj;
    int64_t nhub;           /* DFT+U atomic-orbital projector count (0 = off) */
    const double *t;
    const uint8_t *mask;
    const int64_t *idx_sc;
    const int64_t *idx_ga;
    const double *v_eff;
    const c128 *p;
    const c128 *dij;
    const c128 *hub_q;      /* (nk, nhub, m) */
    const c128 *hub_dij;    /* (nhub, nhub), already the apply-convention D^T */
    int nthreads;
} ctx_t;

/* H-apply for the rows of ONE k: out = H_k c, (nbc, m). box/work are caller
 * thread-local FFT buffers (box has n+1 slots: trash slot == n). */
static void h_apply_k(const ctx_t *cx, int64_t k, const c128 *c, c128 *out,
                      int64_t nbc, c128 *box, c128 *work, c128 *becp_ws,
                      c128 *tmp_ws) {
    const int64_t m = cx->m, n = cx->n, np = cx->nproj;
    const double *tk = cx->t + k * m;
    const uint8_t *mk = cx->mask + k * m;
    const int64_t *sc = cx->idx_sc + k * m;
    const int64_t *ga = cx->idx_ga + k * m;
    const double invn = 1.0 / (double)n;
    for (int64_t b = 0; b < nbc; b++) {
        const c128 *cb = c + b * m;
        c128 *ob = out + b * m;
        for (int64_t g = 0; g < m; g++) ob[g] = tk[g] * cb[g];
        memset(box, 0, sizeof(c128) * (size_t)(n + 1));
        for (int64_t g = 0; g < m; g++) box[sc[g]] = cb[g];
        fftw_execute_dft(s_plan_b, box, work);
        for (int64_t i = 0; i < n; i++) work[i] *= cx->v_eff[i] * invn;
        fftw_execute_dft(s_plan_f, work, box);
        for (int64_t g = 0; g < m; g++) ob[g] += box[ga[g]];
    }
    if (np > 0) {
        const c128 one = 1.0, zero = 0.0;
        const c128 *Pk = cx->p + k * np * m;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, (int)nbc,
                    (int)np, (int)m, &one, c, (int)m, Pk, (int)m, &zero,
                    becp_ws, (int)np);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)nbc,
                    (int)np, (int)np, &one, becp_ws, (int)np, cx->dij,
                    (int)np, &zero, tmp_ws, (int)np);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)nbc,
                    (int)m, (int)np, &one, tmp_ws, (int)np, Pk, (int)m, &one,
                    out, (int)m);
    }
    if (cx->nhub > 0) {
        /* DFT+U second nonlocal term: identical becp pattern against the
         * atomic-orbital projectors; hub_dij arrives in the apply convention
         * (D^T = conj(D), see scf/loop.py's hubbard block). */
        const int64_t nh = cx->nhub;
        const c128 one = 1.0, zero = 0.0;
        const c128 *Qk = cx->hub_q + k * nh * m;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, (int)nbc,
                    (int)nh, (int)m, &one, c, (int)m, Qk, (int)m, &zero,
                    becp_ws, (int)nh);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)nbc,
                    (int)nh, (int)nh, &one, becp_ws, (int)nh, cx->hub_dij,
                    (int)nh, &zero, tmp_ws, (int)nh);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)nbc,
                    (int)m, (int)nh, &one, tmp_ws, (int)nh, Qk, (int)m, &one,
                    out, (int)m);
    }
    for (int64_t b = 0; b < nbc; b++) {
        c128 *ob = out + b * m;
        for (int64_t g = 0; g < m; g++)
            if (!mk[g]) ob[g] = 0.0;
    }
}

/* per-k QR row-orthonormalization (rows, m) in place via qr(x^T) -> q^T.
 * Optionally returns R (rows, rows). 0 ok, -1 LAPACK failure. */
static int qr_rows(c128 *x, int64_t rows, int64_t m, c128 *ws_t, c128 *r_out) {
    for (int64_t j = 0; j < rows; j++)
        for (int64_t g = 0; g < m; g++) ws_t[g * rows + j] = x[j * m + g];
    c128 tau[512];
    if (rows > 512) return -1;
    int info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, (int)m, (int)rows, ws_t,
                              (int)rows, tau);
    if (info == 0 && r_out != NULL)
        for (int64_t i = 0; i < rows; i++)
            for (int64_t j = 0; j < rows; j++)
                r_out[i * rows + j] = j >= i ? ws_t[i * rows + j] : 0.0;
    if (info == 0)
        info = LAPACKE_zungqr(LAPACK_ROW_MAJOR, (int)m, (int)rows, (int)rows,
                              ws_t, (int)rows, tau);
    if (info != 0) return -1;
    for (int64_t j = 0; j < rows; j++)
        for (int64_t g = 0; g < m; g++) x[j * m + g] = ws_t[g * rows + j];
    return 0;
}

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

/* per-thread scratch bundle */
typedef struct {
    c128 *box, *work, *becp, *tmp, *gram, *qrt, *rmat;
    double *W, *tband;
    int64_t *sel;
} scratch_t;

static scratch_t scratch_alloc(const ctx_t *cx, int64_t max_dim) {
    scratch_t s;
    int64_t npz = cx->nproj > cx->nhub ? cx->nproj : cx->nhub;
    if (npz < 1) npz = 1;
    s.box = fftw_alloc_complex(cx->n + 1);
    s.work = fftw_alloc_complex(cx->n);
    s.becp = malloc(sizeof(c128) * (size_t)(cx->nb * npz));
    s.tmp = malloc(sizeof(c128) * (size_t)(cx->nb * npz));
    s.gram = malloc(sizeof(c128) * (size_t)(cx->nb * max_dim));
    s.qrt = malloc(sizeof(c128) * (size_t)(cx->m * max_dim));
    s.rmat = malloc(sizeof(c128) * (size_t)(cx->nb * cx->nb));
    s.W = malloc(sizeof(double) * (size_t)max_dim);
    s.tband = malloc(sizeof(double) * (size_t)cx->nb);
    s.sel = malloc(sizeof(int64_t) * (size_t)cx->nb);
    return s;
}

static void scratch_free(scratch_t *s) {
    fftw_free(s->box); fftw_free(s->work);
    free(s->becp); free(s->tmp); free(s->gram); free(s->qrt); free(s->rmat);
    free(s->W); free(s->tband); free(s->sel);
}

/* ------------------------------------------------------------------ */
/* the solve. Returns n_iter (>0) or:
 *   -1 degenerate row  -2 LAPACK failure  -4 nb > 512                 */

int davidson_native(
    int64_t nk, int64_t nb, int64_t m, int64_t n1, int64_t n2, int64_t n3,
    int64_t nproj, int64_t nhub, int64_t max_dim, int64_t max_iter, double tol,
    int nthreads, int per_k_retire,
    const c128 *x0, const double *t, const uint8_t *mask,
    const int64_t *idx_sc, const int64_t *idx_ga, const double *v_eff,
    const c128 *p, const c128 *dij, const c128 *hub_q, const c128 *hub_dij,
    double *eig_out, c128 *x_out, double *rn_out, int64_t *napply_out)
{
    if (nb > 512) return -4;
    ensure_plans((int)n1, (int)n2, (int)n3);
    ctx_t cx = {.nk = nk, .nb = nb, .m = m, .n = (int64_t)n1 * n2 * n3,
                .nproj = nproj, .nhub = nhub, .t = t, .mask = mask,
                .idx_sc = idx_sc, .idx_ga = idx_ga, .v_eff = v_eff, .p = p,
                .dij = dij, .hub_q = hub_q, .hub_dij = hub_dij,
                .nthreads = nthreads};

    openblas_set_num_threads(1);  /* all BLAS inside omp-over-k regions */

    c128 *V = malloc(sizeof(c128) * (size_t)(nk * max_dim * m));
    c128 *HV = malloc(sizeof(c128) * (size_t)(nk * max_dim * m));
    c128 *X = malloc(sizeof(c128) * (size_t)(nk * nb * m));
    c128 *HX = malloc(sizeof(c128) * (size_t)(nk * nb * m));
    c128 *D = malloc(sizeof(c128) * (size_t)(nk * nb * m));
    c128 *S = malloc(sizeof(c128) * (size_t)(nk * max_dim * max_dim));
    double *rn = malloc(sizeof(double) * (size_t)(nk * nb));
    int64_t napply = 0;
    volatile int err = 0;
    /* per-k subspace dims + activity. Batch mode (per_k_retire=0) keeps them
     * lockstep — bit-for-bit the reference davidson_batched. Retire mode lets
     * each k converge and drop out independently (the "per-k retirement"
     * contract: every returned band satisfies rn <= tol at ITS OWN final RR;
     * trajectories differ from the uniform batch by construction). */
    int64_t *dim_k = malloc(sizeof(int64_t) * (size_t)nk);
    int64_t *n_add_k = malloc(sizeof(int64_t) * (size_t)nk);
    uint8_t *active = malloc(sizeof(uint8_t) * (size_t)nk);
    for (int64_t k = 0; k < nk; k++) { dim_k[k] = nb; active[k] = 1; }

    /* ---- init: V = qr(x0 * mask); HV = H V ---- */
#pragma omp parallel num_threads(nthreads)
    {
        scratch_t ws = scratch_alloc(&cx, max_dim);
#pragma omp for schedule(dynamic)
        for (int64_t k = 0; k < nk; k++) {
            if (err) continue;
            c128 *Vk = V + k * max_dim * m;
            const c128 *xk = x0 + k * nb * m;
            const uint8_t *mk = mask + k * m;
            for (int64_t b = 0; b < nb; b++)
                for (int64_t g = 0; g < m; g++)
                    Vk[b * m + g] = mk[g] ? xk[b * m + g] : 0.0;
            for (int64_t b = 0; b < nb; b++) {
                double s2 = 0;
                for (int64_t g = 0; g < m; g++) {
                    double re = creal(Vk[b * m + g]), im = cimag(Vk[b * m + g]);
                    s2 += re * re + im * im;
                }
                if (sqrt(s2) < 1e-8) { err = -1; break; }
            }
            if (err) continue;
            if (qr_rows(Vk, nb, m, ws.qrt, NULL) != 0) { err = -2; continue; }
            h_apply_k(&cx, k, Vk, HV + k * max_dim * m, nb, ws.box, ws.work,
                      ws.becp, ws.tmp);
        }
        scratch_free(&ws);
    }
    if (err) goto done;
    napply += nk * nb;

    int64_t it;
    for (it = 1; it <= max_iter; it++) {
        /* ---- Rayleigh-Ritz (parallel over k) ---- */
#pragma omp parallel num_threads(nthreads)
        {
            scratch_t ws = scratch_alloc(&cx, max_dim);
            const c128 one = 1.0, zero = 0.0;
#pragma omp for schedule(dynamic)
            for (int64_t k = 0; k < nk; k++) {
                if (err || !active[k]) continue;
                const int64_t dim = dim_k[k];
                const c128 *Vk = V + k * max_dim * m;
                const c128 *HVk = HV + k * max_dim * m;
                c128 *Sk = S + k * max_dim * max_dim;
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                            (int)dim, (int)dim, (int)m, &one, Vk, (int)m,
                            HVk, (int)m, &zero, Sk, (int)dim);
                /* s = conj(V HV^H); Hermitian part */
                for (int64_t i = 0; i < dim; i++)
                    for (int64_t j = i; j < dim; j++) {
                        c128 sij = conj(Sk[i * dim + j]);
                        c128 sji = conj(Sk[j * dim + i]);
                        c128 hij = 0.5 * (sij + conj(sji));
                        Sk[i * dim + j] = hij;
                        Sk[j * dim + i] = conj(hij);
                    }
                if (LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', (int)dim, Sk,
                                   (int)dim, ws.W) != 0) { err = -2; continue; }
                for (int64_t b = 0; b < nb; b++) eig_out[k * nb + b] = ws.W[b];
                c128 *Xk = X + k * nb * m, *HXk = HX + k * nb * m;
                cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)nb,
                            (int)m, (int)dim, &one, Sk, (int)dim, Vk, (int)m,
                            &zero, Xk, (int)m);
                cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)nb,
                            (int)m, (int)dim, &one, Sk, (int)dim, HVk, (int)m,
                            &zero, HXk, (int)m);
                for (int64_t b = 0; b < nb; b++) {
                    double s2 = 0, e = ws.W[b];
                    c128 *xb = Xk + b * m, *hb = HXk + b * m;
                    for (int64_t g = 0; g < m; g++) {
                        c128 r = hb[g] - e * xb[g];
                        double re = creal(r), im = cimag(r);
                        s2 += re * re + im * im;
                    }
                    rn[k * nb + b] = sqrt(s2);
                }
            }
            scratch_free(&ws);
        }
        if (err) goto done;

        /* ---- convergence / n_add (global-uniform or per-k) ---- */
        double rnmax = 0;
        int64_t n_add_max = 0, n_active = 0;
        for (int64_t k = 0; k < nk; k++) {
            if (!active[k]) { n_add_k[k] = 0; continue; }
            int64_t cnt = 0;
            double rnk = 0;
            for (int64_t b = 0; b < nb; b++) {
                if (rn[k * nb + b] > rnk) rnk = rn[k * nb + b];
                if (rn[k * nb + b] > tol) cnt++;
            }
            if (rnk > rnmax) rnmax = rnk;
            if (per_k_retire && rnk < tol) { active[k] = 0; n_add_k[k] = 0; continue; }
            n_add_k[k] = cnt;
            if (cnt > n_add_max) n_add_max = cnt;
            n_active++;
        }
        if (per_k_retire) {
            if (n_active == 0) break;
        } else {
            if (rnmax < tol) break;
            for (int64_t k = 0; k < nk; k++) n_add_k[k] = n_add_max;
        }

        /* ---- expansion (parallel over k) ---- */
#pragma omp parallel num_threads(nthreads)
        {
            scratch_t ws = scratch_alloc(&cx, max_dim);
#pragma omp for schedule(dynamic)
            for (int64_t k = 0; k < nk; k++) {
                if (err || !active[k] || n_add_k[k] == 0) continue;
                const int64_t n_add = n_add_k[k];
                const int64_t dim = dim_k[k];
                const int restart = (dim + n_add > max_dim);
                const double *tk = t + k * m;
                const uint8_t *mk = mask + k * m;
                c128 *Xk = X + k * nb * m, *HXk = HX + k * nb * m;
                c128 *Vk = V + k * max_dim * m, *HVk = HV + k * max_dim * m;
                c128 *Dk = D + k * nb * m;
                /* selection: top-n_add residual rows (descending) */
                for (int64_t b = 0; b < nb; b++) ws.sel[b] = b;
                for (int64_t i = 0; i < n_add; i++) {
                    int64_t best = i;
                    for (int64_t j = i + 1; j < nb; j++)
                        if (rn[k * nb + ws.sel[j]] > rn[k * nb + ws.sel[best]])
                            best = j;
                    int64_t tv = ws.sel[i]; ws.sel[i] = ws.sel[best];
                    ws.sel[best] = tv;
                }
                /* Teter-preconditioned, unit-normalized directions */
                for (int64_t i = 0; i < n_add; i++) {
                    int64_t b = ws.sel[i];
                    double tb = 0;
                    c128 *xb = Xk + b * m;
                    for (int64_t g = 0; g < m; g++) {
                        double re = creal(xb[g]), im = cimag(xb[g]);
                        tb += tk[g] * (re * re + im * im);
                    }
                    if (tb < 1e-12) tb = 1e-12;
                    double e = eig_out[k * nb + b];
                    c128 *hb = HXk + b * m, *db = Dk + i * m;
                    double nrm2 = 0;
                    for (int64_t g = 0; g < m; g++) {
                        double xx = tk[g] / tb;
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
                int64_t against_dim = dim;
                if (restart) {
                    /* collapse to re-orthonormalized Ritz block; repair images
                     * by the triangular factor: hx_new = (R^T)^{-1} hx */
                    memcpy(Vk, Xk, sizeof(c128) * (size_t)(nb * m));
                    if (qr_rows(Vk, nb, m, ws.qrt, ws.rmat) != 0) {
                        err = -2; continue;
                    }
                    memcpy(HVk, HXk, sizeof(c128) * (size_t)(nb * m));
                    const c128 cone = 1.0;
                    cblas_ztrsm(CblasRowMajor, CblasLeft, CblasUpper,
                                CblasTrans, CblasNonUnit, (int)nb, (int)m,
                                &cone, ws.rmat, (int)nb, HVk, (int)m);
                    against_dim = nb;
                }
                for (int64_t i = 0; i < n_add; i++)
                    for (int64_t g = 0; g < m; g++)
                        if (!mk[g]) Dk[i * m + g] = 0.0;
                project_out(Dk, Vk, n_add, against_dim, m, ws.gram);
                for (int64_t i = 0; i < n_add; i++) {
                    double s2 = 0;
                    for (int64_t g = 0; g < m; g++) {
                        double re = creal(Dk[i * m + g]);
                        double im = cimag(Dk[i * m + g]);
                        s2 += re * re + im * im;
                    }
                    if (sqrt(s2) < 1e-8) { err = -1; break; }
                }
                if (err) continue;
                if (qr_rows(Dk, n_add, m, ws.qrt, NULL) != 0) {
                    err = -2; continue;
                }
                memcpy(Vk + against_dim * m, Dk,
                       sizeof(c128) * (size_t)(n_add * m));
                h_apply_k(&cx, k, Vk + against_dim * m, HVk + against_dim * m,
                          n_add, ws.box, ws.work, ws.becp, ws.tmp);
                dim_k[k] = against_dim + n_add;
            }
            scratch_free(&ws);
        }
        if (err) goto done;
        for (int64_t k = 0; k < nk; k++) napply += n_add_k[k];
    }

    memcpy(x_out, X, sizeof(c128) * (size_t)(nk * nb * m));
    memcpy(rn_out, rn, sizeof(double) * (size_t)(nk * nb));
    *napply_out = napply;
    err = (int)(it > max_iter ? max_iter : it);

done:
    free(V); free(HV); free(X); free(HX); free(D); free(S); free(rn);
    free(dim_k); free(n_add_k); free(active);
    return err;
}
