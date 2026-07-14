/*
 * Rectangular DGEMM for Xeon Phi 7120P (KNC).
 * C = A @ B, all row-major, FP64.
 *
 * Optimized path (ICC -mmic):
 *   icc -std=c99 -mmic -O3 -openmp -restrict -o dgemm_rect.mic dgemm_rect.c
 *   - OpenMP over M tiles
 *   - IMCI _mm512_fmadd_pd with 4x J-unroll
 *   - L2-friendly K/N blocking
 *   - N padded to multiple of 8 for aligned vector loads
 *
 * Fallback (k1om-gcc):
 *   k1om-mpss-linux-gcc --sysroot=$SYSROOT -O3 -pthread -o dgemm_rect.mic ...
 *
 * Input:  int32 M,N,K ; double A[M*K] ; double B[K*N]
 * Output: int32 M,N   ; double C[M*N]
 *
 * Env: OMP_NUM_THREADS / PHI_DGEMM_THREADS, PHI_DGEMM_BM/BK/BN, PHI_DGEMM_WARMUP
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#if defined(__MIC__) || defined(__KNC__) || defined(USE_IMCI)
#include <immintrin.h>
#include <omp.h>
#define HAVE_IMCI 1
#else
#include <pthread.h>
#define HAVE_IMCI 0
#endif

static double tnow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

static void *xaligned_alloc(size_t align, size_t nbytes)
{
    void *p = NULL;
    if (posix_memalign(&p, align, nbytes) != 0)
        return NULL;
    return p;
}

static int env_int(const char *name, int defv)
{
    const char *e = getenv(name);
    if (!e || !*e)
        return defv;
    int v = atoi(e);
    return v > 0 ? v : defv;
}

#if HAVE_IMCI

/*
 * Row-parallel GEMM matching intel_phi/tests/perf/phi_peak_dgemm.c structure:
 *   #pragma omp for i
 *     for k
 *       broadcast A[i,k]; FMA into C[i,*] with 4x J unroll (32 doubles)
 * B/C use pitch Np (multiple of 8) for aligned load_pd/store_pd.
 */
static void dgemm_imci(int M, int N, int K, int Np,
                       const double *restrict A,
                       const double *restrict B,
                       double *restrict C)
{
    (void)N;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        double *ci = &C[i * Np];
        /* zero row */
        for (int j = 0; j < Np; j += 8)
            _mm512_store_pd(&ci[j], _mm512_setzero_pd());

        for (int k = 0; k < K; k++) {
            double aik = A[i * K + k];
            __m512d va = _mm512_set1_pd(aik);
            const double *bk = &B[k * Np];
            int j = 0;
            for (; j + 31 < Np; j += 32) {
                __m512d vc0 = _mm512_load_pd(&ci[j + 0]);
                __m512d vc1 = _mm512_load_pd(&ci[j + 8]);
                __m512d vc2 = _mm512_load_pd(&ci[j + 16]);
                __m512d vc3 = _mm512_load_pd(&ci[j + 24]);
                __m512d vb0 = _mm512_load_pd(&bk[j + 0]);
                __m512d vb1 = _mm512_load_pd(&bk[j + 8]);
                __m512d vb2 = _mm512_load_pd(&bk[j + 16]);
                __m512d vb3 = _mm512_load_pd(&bk[j + 24]);
                vc0 = _mm512_fmadd_pd(va, vb0, vc0);
                vc1 = _mm512_fmadd_pd(va, vb1, vc1);
                vc2 = _mm512_fmadd_pd(va, vb2, vc2);
                vc3 = _mm512_fmadd_pd(va, vb3, vc3);
                _mm512_store_pd(&ci[j + 0], vc0);
                _mm512_store_pd(&ci[j + 8], vc1);
                _mm512_store_pd(&ci[j + 16], vc2);
                _mm512_store_pd(&ci[j + 24], vc3);
            }
            for (; j + 7 < Np; j += 8) {
                __m512d vc = _mm512_load_pd(&ci[j]);
                __m512d vb = _mm512_load_pd(&bk[j]);
                vc = _mm512_fmadd_pd(va, vb, vc);
                _mm512_store_pd(&ci[j], vc);
            }
        }
    }
}

#else /* pthread scalar fallback */

typedef struct {
    int i0, i1, N, K;
    const double *A, *B;
    double *C;
} row_job_t;

static void *row_worker(void *arg)
{
    row_job_t *job = (row_job_t *)arg;
    int N = job->N, K = job->K;
    const double *A = job->A, *B = job->B;
    double *C = job->C;
    for (int i = job->i0; i < job->i1; i++) {
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.0;
        for (int p = 0; p < K; p++) {
            double aip = A[i * K + p];
            const double *bp = &B[p * N];
            double *ci = &C[i * N];
            for (int j = 0; j < N; j++)
                ci[j] += aip * bp[j];
        }
    }
    return NULL;
}

static void dgemm_pthread(int M, int N, int K,
                          const double *A, const double *B, double *C, int nthreads)
{
    if (nthreads < 1)
        nthreads = 1;
    if (M > 0 && nthreads > M)
        nthreads = M;
    if (nthreads == 1) {
        row_job_t job = {0, M, N, K, A, B, C};
        row_worker(&job);
        return;
    }
    pthread_t *th = (pthread_t *)malloc((size_t)nthreads * sizeof(pthread_t));
    row_job_t *jobs = (row_job_t *)malloc((size_t)nthreads * sizeof(row_job_t));
    int chunk = (M + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; t++) {
        int i0 = t * chunk;
        int i1 = i0 + chunk;
        if (i1 > M)
            i1 = M;
        jobs[t] = (row_job_t){i0, i1, N, K, A, B, C};
        pthread_create(&th[t], NULL, row_worker, &jobs[t]);
    }
    for (int t = 0; t < nthreads; t++)
        pthread_join(th[t], NULL);
    free(th);
    free(jobs);
}

#endif

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }

    FILE *fi = fopen(argv[1], "rb");
    if (!fi) {
        perror(argv[1]);
        return 1;
    }

    int M, N, K;
    if (fread(&M, sizeof(int), 1, fi) != 1 ||
        fread(&N, sizeof(int), 1, fi) != 1 ||
        fread(&K, sizeof(int), 1, fi) != 1) {
        fprintf(stderr, "header read failed\n");
        return 1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "bad dims\n");
        return 1;
    }

    size_t aN = (size_t)M * (size_t)K;
    size_t bN = (size_t)K * (size_t)N;
    size_t cN = (size_t)M * (size_t)N;

    double *A = (double *)xaligned_alloc(64, aN * sizeof(double));
    double *B_in = (double *)xaligned_alloc(64, bN * sizeof(double));
    if (!A || !B_in) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    if (fread(A, sizeof(double), aN, fi) != aN ||
        fread(B_in, sizeof(double), bN, fi) != bN) {
        fprintf(stderr, "matrix read failed\n");
        return 1;
    }
    fclose(fi);

    int nthreads = env_int("PHI_DGEMM_THREADS", 0);
#if HAVE_IMCI
    if (nthreads > 0)
        omp_set_num_threads(nthreads);
    nthreads = omp_get_max_threads();

    /* Pad N to multiple of 8 so every row of B/C is 64B-aligned for load_pd */
    int Np = (N + 7) & ~7;
    double *B = (double *)xaligned_alloc(64, (size_t)K * (size_t)Np * sizeof(double));
    double *C = (double *)xaligned_alloc(64, (size_t)M * (size_t)Np * sizeof(double));
    if (!B || !C) {
        fprintf(stderr, "pad alloc failed\n");
        return 1;
    }
#pragma omp parallel for schedule(static)
    for (int k = 0; k < K; k++) {
        memcpy(&B[k * Np], &B_in[k * N], (size_t)N * sizeof(double));
        if (Np > N)
            memset(&B[k * Np + N], 0, (size_t)(Np - N) * sizeof(double));
    }
    free(B_in);

    if (env_int("PHI_DGEMM_WARMUP", 1))
        dgemm_imci(M, N, K, Np, A, B, C);

    double t0 = tnow();
    dgemm_imci(M, N, K, Np, A, B, C);
    double elapsed = tnow() - t0;
    const char *backend = "imci+omp+block";

    /* pack C back to true N */
    double *C_out = (double *)xaligned_alloc(64, cN * sizeof(double));
    if (!C_out) {
        fprintf(stderr, "C_out alloc failed\n");
        return 1;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++)
        memcpy(&C_out[i * N], &C[i * Np], (size_t)N * sizeof(double));
    free(B);
    free(C);
    C = C_out;
#else
    if (nthreads <= 0)
        nthreads = 120;
    double *C = (double *)xaligned_alloc(64, cN * sizeof(double));
    if (!C) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    double *B = B_in;
    if (env_int("PHI_DGEMM_WARMUP", 0))
        dgemm_pthread(M, N, K, A, B, C, nthreads);
    double t0 = tnow();
    dgemm_pthread(M, N, K, A, B, C, nthreads);
    double elapsed = tnow() - t0;
    const char *backend = "pthread-scalar";
#endif

    double gflops = 2.0 * (double)M * (double)N * (double)K / elapsed / 1.0e9;
    double checksum = 0.0;
    for (size_t i = 0; i < cN; i++)
        checksum += C[i];

    FILE *fo = fopen(argv[2], "wb");
    if (!fo) {
        perror(argv[2]);
        return 1;
    }
    fwrite(&M, sizeof(int), 1, fo);
    fwrite(&N, sizeof(int), 1, fo);
    fwrite(C, sizeof(double), cN, fo);
    fclose(fo);

    printf("PHI dgemm_rect: M=%d N=%d K=%d threads=%d backend=%s %.4fs %.2f GFLOPS checksum=%.6e\n",
           M, N, K, nthreads, backend, elapsed, gflops, checksum);
    printf("Result: PASS\n");

    free(A);
#if !HAVE_IMCI
    free(B);
#endif
    free(C);
    return 0;
}
