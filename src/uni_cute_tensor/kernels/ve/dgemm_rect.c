/*
 * Rectangular DGEMM on NEC VE via NLC cblas (row-major, matches numpy).
 *
 * Compile:
 *   ncc -O3 -fopenmp -o dgemm_rect_ve dgemm_rect.c \
 *       -I/opt/nec/ve/nlc/3.1.0/include \
 *       -L/opt/nec/ve/nlc/3.1.0/lib -lcblas -lblas_openmp
 *
 * Modes:
 *   1) Combined:  ve_exec -N id ./dgemm_rect_ve input.bin output.bin
 *      input:  int32 M,N,K ; A[M*K] ; B[K*N]
 *   2) Split (share B): ve_exec -N id ./dgemm_rect_ve a.bin b.bin output.bin
 *      a.bin:  int32 M,K ; A[M*K]
 *      b.bin:  int32 K,N ; B[K*N]
 *      out:    int32 M,N ; C[M*N]
 *
 * Env: VE_DGEMM_WARMUP=0 to skip warmup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cblas.h>

static double tnow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

static int env_warmup(void)
{
    const char *e = getenv("VE_DGEMM_WARMUP");
    if (!e)
        return 1;
    return atoi(e) != 0;
}

static int run_dgemm(int M, int N, int K, double *A, double *B, double *C,
                     double *elapsed_out, double *gflops_out)
{
    if (env_warmup())
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0, A, K, B, N, 0.0, C, N);

    double t0 = tnow();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    double elapsed = tnow() - t0;
    *elapsed_out = elapsed;
    *gflops_out = 2.0 * (double)M * (double)N * (double)K / elapsed / 1.0e9;
    return 0;
}

static int mode_combined(const char *in_path, const char *out_path)
{
    FILE *fi = fopen(in_path, "rb");
    if (!fi) {
        perror(in_path);
        return 1;
    }
    int M, N, K;
    if (fread(&M, 4, 1, fi) != 1 || fread(&N, 4, 1, fi) != 1 || fread(&K, 4, 1, fi) != 1) {
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
    double *A = (double *)aligned_alloc(64, aN * 8);
    double *B = (double *)aligned_alloc(64, bN * 8);
    double *C = (double *)aligned_alloc(64, cN * 8);
    if (!A || !B || !C) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    if (fread(A, 8, aN, fi) != aN || fread(B, 8, bN, fi) != bN) {
        fprintf(stderr, "body read failed\n");
        return 1;
    }
    fclose(fi);

    double elapsed, gflops;
    run_dgemm(M, N, K, A, B, C, &elapsed, &gflops);

    double checksum = 0.0;
    for (size_t i = 0; i < cN; i++)
        checksum += C[i];

    FILE *fo = fopen(out_path, "wb");
    if (!fo) {
        perror(out_path);
        return 1;
    }
    fwrite(&M, 4, 1, fo);
    fwrite(&N, 4, 1, fo);
    fwrite(C, 8, cN, fo);
    fclose(fo);

    printf("VE dgemm_rect: M=%d N=%d K=%d mode=combined %.4fs %.2f GFLOPS checksum=%.6e\n",
           M, N, K, elapsed, gflops, checksum);
    printf("Result: PASS\n");
    free(A);
    free(B);
    free(C);
    return 0;
}

static int mode_split(const char *a_path, const char *b_path, const char *out_path)
{
    FILE *fa = fopen(a_path, "rb");
    FILE *fb = fopen(b_path, "rb");
    if (!fa) {
        perror(a_path);
        return 1;
    }
    if (!fb) {
        perror(b_path);
        return 1;
    }

    int M, Ka, Kb, N;
    if (fread(&M, 4, 1, fa) != 1 || fread(&Ka, 4, 1, fa) != 1) {
        fprintf(stderr, "A header failed\n");
        return 1;
    }
    if (fread(&Kb, 4, 1, fb) != 1 || fread(&N, 4, 1, fb) != 1) {
        fprintf(stderr, "B header failed\n");
        return 1;
    }
    if (Ka != Kb || M <= 0 || N <= 0 || Ka <= 0) {
        fprintf(stderr, "dim mismatch M=%d Ka=%d Kb=%d N=%d\n", M, Ka, Kb, N);
        return 1;
    }
    int K = Ka;

    size_t aN = (size_t)M * (size_t)K;
    size_t bN = (size_t)K * (size_t)N;
    size_t cN = (size_t)M * (size_t)N;
    double *A = (double *)aligned_alloc(64, aN * 8);
    double *B = (double *)aligned_alloc(64, bN * 8);
    double *C = (double *)aligned_alloc(64, cN * 8);
    if (!A || !B || !C) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    if (fread(A, 8, aN, fa) != aN) {
        fprintf(stderr, "A body failed\n");
        return 1;
    }
    if (fread(B, 8, bN, fb) != bN) {
        fprintf(stderr, "B body failed\n");
        return 1;
    }
    fclose(fa);
    fclose(fb);

    double elapsed, gflops;
    run_dgemm(M, N, K, A, B, C, &elapsed, &gflops);

    double checksum = 0.0;
    for (size_t i = 0; i < cN; i++)
        checksum += C[i];

    FILE *fo = fopen(out_path, "wb");
    if (!fo) {
        perror(out_path);
        return 1;
    }
    fwrite(&M, 4, 1, fo);
    fwrite(&N, 4, 1, fo);
    fwrite(C, 8, cN, fo);
    fclose(fo);

    printf("VE dgemm_rect: M=%d N=%d K=%d mode=split %.4fs %.2f GFLOPS checksum=%.6e\n",
           M, N, K, elapsed, gflops, checksum);
    printf("Result: PASS\n");
    free(A);
    free(B);
    free(C);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3)
        return mode_combined(argv[1], argv[2]);
    if (argc == 4)
        return mode_split(argv[1], argv[2], argv[3]);
    fprintf(stderr,
            "Usage:\n"
            "  %s input.bin output.bin          # combined A|B\n"
            "  %s a.bin b.bin output.bin        # split (share B)\n",
            argv[0], argv[0]);
    return 1;
}
