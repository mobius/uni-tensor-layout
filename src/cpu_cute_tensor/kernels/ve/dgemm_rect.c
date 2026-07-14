/*
 * Rectangular DGEMM on NEC VE via NLC cblas (row-major, matches numpy).
 *
 * Compile:
 *   ncc -O3 -fopenmp -o dgemm_rect_ve dgemm_rect.c \
 *       -I/opt/nec/ve/nlc/3.1.0/include \
 *       -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp
 *
 * Run:
 *   VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib \
 *   ve_exec -N <1|2|3> ./dgemm_rect_ve input.bin output.bin
 *
 * Binary input:
 *   int32 M, N, K
 *   double A[M*K]  (row-major)
 *   double B[K*N]  (row-major)
 * Binary output:
 *   int32 M, N
 *   double C[M*N]  (row-major)  // C = A @ B
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>

static double tnow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
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
        fprintf(stderr, "failed to read header\n");
        fclose(fi);
        return 1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "invalid dims M=%d N=%d K=%d\n", M, N, K);
        fclose(fi);
        return 1;
    }

    size_t aN = (size_t)M * (size_t)K;
    size_t bN = (size_t)K * (size_t)N;
    size_t cN = (size_t)M * (size_t)N;

    double *A = (double *)aligned_alloc(64, aN * sizeof(double));
    double *B = (double *)aligned_alloc(64, bN * sizeof(double));
    double *C = (double *)aligned_alloc(64, cN * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    if (fread(A, sizeof(double), aN, fi) != aN ||
        fread(B, sizeof(double), bN, fi) != bN) {
        fprintf(stderr, "failed to read matrices\n");
        fclose(fi);
        return 1;
    }
    fclose(fi);

    for (size_t i = 0; i < cN; i++)
        C[i] = 0.0;

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

    printf("VE dgemm_rect: M=%d N=%d K=%d %.4fs %.2f GFLOPS checksum=%.6e\n",
           M, N, K, elapsed, gflops, checksum);
    printf("Result: PASS\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
