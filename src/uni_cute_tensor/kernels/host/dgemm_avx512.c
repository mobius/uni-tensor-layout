/*
 * Host DGEMM microkernel: OpenMP + AVX-512 FMA (Cascade Lake / Gold 6252).
 * Builds as shared library for ctypes:
 *   gcc -O3 -fopenmp -mavx512f -mfma -fPIC -shared -o libhost_dgemm_avx512.so dgemm_avx512.c
 *
 * void host_dgemm_avx512(int M, int N, int K,
 *                        const double *A, const double *B, double *C);
 *   C = A @ B, row-major FP64. A is MxK, B is KxN, C is MxN.
 */

#include <immintrin.h>
#include <omp.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void host_dgemm_avx512(int M, int N, int K,
                              const double *A, const double *B, double *C)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        double *ci = &C[i * N];
        memset(ci, 0, (size_t)N * sizeof(double));

        for (int k = 0; k < K; k++) {
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
}

EXPORT int host_dgemm_avx512_available(void)
{
    return 1;
}
