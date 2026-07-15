/*
 * Host DGEMM: OpenMP + AVX-512 with K-panel blocking for better L2 reuse of B.
 *
 *   gcc -O3 -fopenmp -mavx512f -mfma -fPIC -shared -o libhost_dgemm_avx512.so dgemm_avx512.c
 *
 * Env: HOST_DGEMM_BK (default 256) — K-panel size
 */

#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static int env_int(const char *name, int defv)
{
    const char *e = getenv(name);
    if (!e || !*e)
        return defv;
    int v = atoi(e);
    return v > 0 ? v : defv;
}

static void dgemm_row_kpanel(int i, int N, int k0, int k1, int K,
                             const double *A, const double *B, double *C)
{
    double *ci = &C[i * N];
    for (int k = k0; k < k1; k++) {
        double aik = A[i * K + k];
        __m512d va = _mm512_set1_pd(aik);
        const double *bk = &B[k * N];
        int j = 0;
        for (; j + 31 < N; j += 32) {
            __m512d vc0 = _mm512_loadu_pd(&ci[j + 0]);
            __m512d vc1 = _mm512_loadu_pd(&ci[j + 8]);
            __m512d vc2 = _mm512_loadu_pd(&ci[j + 16]);
            __m512d vc3 = _mm512_loadu_pd(&ci[j + 24]);
            __m512d vb0 = _mm512_loadu_pd(&bk[j + 0]);
            __m512d vb1 = _mm512_loadu_pd(&bk[j + 8]);
            __m512d vb2 = _mm512_loadu_pd(&bk[j + 16]);
            __m512d vb3 = _mm512_loadu_pd(&bk[j + 24]);
            vc0 = _mm512_fmadd_pd(va, vb0, vc0);
            vc1 = _mm512_fmadd_pd(va, vb1, vc1);
            vc2 = _mm512_fmadd_pd(va, vb2, vc2);
            vc3 = _mm512_fmadd_pd(va, vb3, vc3);
            _mm512_storeu_pd(&ci[j + 0], vc0);
            _mm512_storeu_pd(&ci[j + 8], vc1);
            _mm512_storeu_pd(&ci[j + 16], vc2);
            _mm512_storeu_pd(&ci[j + 24], vc3);
        }
        for (; j + 7 < N; j += 8) {
            __m512d vc = _mm512_loadu_pd(&ci[j]);
            __m512d vb = _mm512_loadu_pd(&bk[j]);
            vc = _mm512_fmadd_pd(va, vb, vc);
            _mm512_storeu_pd(&ci[j], vc);
        }
        for (; j < N; j++)
            ci[j] += aik * bk[j];
    }
}

EXPORT void host_dgemm_avx512(int M, int N, int K,
                              const double *A, const double *B, double *C)
{
    int BK = env_int("HOST_DGEMM_BK", 256);

    /* zero C */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++)
        memset(&C[i * N], 0, (size_t)N * sizeof(double));

    /* K panels: all threads stream the same B[k0:k1,*] together → better L2 */
    for (int k0 = 0; k0 < K; k0 += BK) {
        int k1 = k0 + BK;
        if (k1 > K)
            k1 = K;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            dgemm_row_kpanel(i, N, k0, k1, K, A, B, C);
    }
}

EXPORT int host_dgemm_avx512_available(void)
{
    return 1;
}
