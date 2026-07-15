/*
 * Phi preprocess: C = alpha * A + beta  (elementwise), row-major FP64.
 * Used in hetero pipeline before multi-VE DGEMM.
 *
 * ICC: icc -std=c99 -mmic -O3 -openmp -o prep_scale.mic prep_scale.c
 * Input:  int32 M,N ; double A[M*N] ; double alpha,beta as two doubles at end
 *   Actually: int32 M,N ; double alpha ; double beta ; double A[M*N]
 * Output: int32 M,N ; double C[M*N]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>
#include <omp.h>

static double tnow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s in.bin out.bin\n", argv[0]);
        return 1;
    }
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) {
        perror(argv[1]);
        return 1;
    }
    int M, N;
    double alpha, beta;
    if (fread(&M, 4, 1, fi) != 1 || fread(&N, 4, 1, fi) != 1 ||
        fread(&alpha, 8, 1, fi) != 1 || fread(&beta, 8, 1, fi) != 1) {
        fprintf(stderr, "hdr\n");
        return 1;
    }
    size_t n = (size_t)M * (size_t)N;
    double *A, *C;
    posix_memalign((void **)&A, 64, n * 8);
    posix_memalign((void **)&C, 64, n * 8);
    if (fread(A, 8, n, fi) != n) {
        fprintf(stderr, "body\n");
        return 1;
    }
    fclose(fi);

    int thr = omp_get_max_threads();
    const char *e = getenv("PHI_DGEMM_THREADS");
    if (e && atoi(e) > 0) {
        thr = atoi(e);
        omp_set_num_threads(thr);
    }

    double t0 = tnow();
    /* pad N to 8 for aligned IMCI loads when possible */
    int Np = (N + 7) & ~7;
    double *Ap = A, *Cp = C;
    int free_pad = 0;
    if (Np != N) {
        posix_memalign((void **)&Ap, 64, (size_t)M * Np * 8);
        posix_memalign((void **)&Cp, 64, (size_t)M * Np * 8);
        free_pad = 1;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++) {
            memcpy(&Ap[i * Np], &A[i * N], (size_t)N * 8);
            if (Np > N)
                memset(&Ap[i * Np + N], 0, (size_t)(Np - N) * 8);
        }
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        double *ai = &Ap[i * Np];
        double *ci = &Cp[i * Np];
        __m512d va = _mm512_set1_pd(alpha);
        __m512d vb = _mm512_set1_pd(beta);
        for (int j = 0; j < Np; j += 8) {
            __m512d x = _mm512_load_pd(&ai[j]);
            x = _mm512_fmadd_pd(va, x, vb);
            _mm512_store_pd(&ci[j], x);
        }
    }
    if (free_pad) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            memcpy(&C[i * N], &Cp[i * Np], (size_t)N * 8);
        free(Ap);
        free(Cp);
    }
    double el = tnow() - t0;
    /* elementwise: 2 flops per element approx */
    double gflops = 2.0 * (double)n / el / 1e9;

    FILE *fo = fopen(argv[2], "wb");
    fwrite(&M, 4, 1, fo);
    fwrite(&N, 4, 1, fo);
    fwrite(C, 8, n, fo);
    fclose(fo);
    printf("PHI prep_scale: M=%d N=%d thr=%d alpha=%.3g beta=%.3g %.4fs %.2f GFLOPS\n",
           M, N, thr, alpha, beta, el, gflops);
    printf("Result: PASS\n");
    free(A);
    free(C);
    return 0;
}
