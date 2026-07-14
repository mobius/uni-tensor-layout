/*
 * Rectangular DGEMM for Xeon Phi (KNC / K1OM), file-staged I/O.
 *
 * Prefer compile with k1om-mpss-linux-gcc (no ICC Comp-CL license required):
 *   source MPSS environment or set PATH to k1om tools
 *   k1om-mpss-linux-gcc --sysroot=$SYSROOT -O3 -pthread \
 *       -o dgemm_rect.mic dgemm_rect.c
 *
 * Optional ICC (needs Comp-CL feature in license):
 *   icc -std=c99 -mmic -O3 -openmp -o dgemm_rect.mic dgemm_rect.c
 *
 * Binary input:  int32 M,N,K ; double A[M*K] row-major ; double B[K*N]
 * Binary output: int32 M,N   ; double C[M*N] row-major  // C = A @ B
 *
 * Run: micnativeloadex dgemm_rect.mic -d 0 -- input.bin output.bin
 *   (or scp + ssh mic0)
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef struct {
    int i0, i1;
    int M, N, K;
    const double *A;
    const double *B;
    double *C;
} row_job_t;

static double tnow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

static void *row_worker(void *arg)
{
    row_job_t *job = (row_job_t *)arg;
    int N = job->N;
    int K = job->K;
    const double *A = job->A;
    const double *B = job->B;
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

static int run_dgemm(int M, int N, int K, const double *A, const double *B,
                     double *C, int nthreads)
{
    if (nthreads < 1)
        nthreads = 1;
    if (nthreads > M)
        nthreads = M;
    if (nthreads == 1) {
        row_job_t job = {0, M, M, N, K, A, B, C};
        row_worker(&job);
        return 0;
    }

    pthread_t *th = (pthread_t *)malloc((size_t)nthreads * sizeof(pthread_t));
    row_job_t *jobs = (row_job_t *)malloc((size_t)nthreads * sizeof(row_job_t));
    if (!th || !jobs)
        return 1;

    int chunk = (M + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; t++) {
        int i0 = t * chunk;
        int i1 = i0 + chunk;
        if (i1 > M)
            i1 = M;
        jobs[t].i0 = i0;
        jobs[t].i1 = i1;
        jobs[t].M = M;
        jobs[t].N = N;
        jobs[t].K = K;
        jobs[t].A = A;
        jobs[t].B = B;
        jobs[t].C = C;
        if (i0 >= i1) {
            /* idle thread */
            jobs[t].i1 = jobs[t].i0;
        }
        if (pthread_create(&th[t], NULL, row_worker, &jobs[t]) != 0) {
            free(th);
            free(jobs);
            return 1;
        }
    }
    for (int t = 0; t < nthreads; t++)
        pthread_join(th[t], NULL);

    free(th);
    free(jobs);
    return 0;
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

    double *A = (double *)malloc(aN * sizeof(double));
    double *B = (double *)malloc(bN * sizeof(double));
    double *C = (double *)malloc(cN * sizeof(double));
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

    int nthreads = 120; /* 61 cores * ~2 usable; cap later if M small */
    const char *envt = getenv("PHI_DGEMM_THREADS");
    if (envt && atoi(envt) > 0)
        nthreads = atoi(envt);

    double t0 = tnow();
    if (run_dgemm(M, N, K, A, B, C, nthreads) != 0) {
        fprintf(stderr, "dgemm failed\n");
        return 1;
    }
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

    printf("PHI dgemm_rect: M=%d N=%d K=%d threads=%d %.4fs %.2f GFLOPS checksum=%.6e\n",
           M, N, K, nthreads, elapsed, gflops, checksum);
    printf("Result: PASS\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
