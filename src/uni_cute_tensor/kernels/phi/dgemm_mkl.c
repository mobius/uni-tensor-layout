/*
 * Rectangular DGEMM on Xeon Phi via Intel MKL (native KNC).
 *
 * Compile (in centos7-phi-dev):
 *   source /opt/intel/bin/compilervars.sh intel64
 *   MKL=$MKLROOT  # or compilers_and_libraries_.../linux/mkl
 *   icc -std=c99 -mmic -O3 -openmp -I$MKL/include \
 *       -o dgemm_mkl.mic dgemm_mkl.c \
 *       -L$MKL/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
 *       -liomp5 -lpthread -lm
 *
 * Input/output binary format same as dgemm_rect.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mkl.h>
#include <omp.h>

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
    double *B = (double *)xaligned_alloc(64, bN * sizeof(double));
    double *C = (double *)xaligned_alloc(64, cN * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    if (fread(A, sizeof(double), aN, fi) != aN ||
        fread(B, sizeof(double), bN, fi) != bN) {
        fprintf(stderr, "matrix read failed\n");
        return 1;
    }
    fclose(fi);

    int nthreads = omp_get_max_threads();
    const char *envt = getenv("PHI_DGEMM_THREADS");
    if (envt && atoi(envt) > 0) {
        nthreads = atoi(envt);
        mkl_set_num_threads(nthreads);
        omp_set_num_threads(nthreads);
    } else {
        mkl_set_num_threads(nthreads);
    }

    /* warmup */
    if (getenv("PHI_DGEMM_WARMUP") == NULL || atoi(getenv("PHI_DGEMM_WARMUP")) != 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    }

    double t0 = tnow();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    double elapsed = tnow() - t0;
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

    printf("PHI dgemm_mkl: M=%d N=%d K=%d threads=%d backend=mkl %.4fs %.2f GFLOPS checksum=%.6e\n",
           M, N, K, nthreads, elapsed, gflops, checksum);
    printf("Result: PASS\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
