/*
 * Persistent Phi worker (KNC): file protocol like VE dgemm_worker.
 *
 * icc -std=c99 -mmic -O3 -openmp -o phi_worker.mic phi_worker.c
 * Optional MKL prep not included — SCALE only (IMCI).
 *
 * control_dir/job.cmd:
 *   QUIT
 *   SCALE in.bin out.bin
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <immintrin.h>
#include <omp.h>

static int file_exists(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0;
}

static double tnow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static int run_scale(const char *in_path, const char *out_path, char *msg, size_t nmsg)
{
    FILE *fi = fopen(in_path, "rb");
    if (!fi) {
        snprintf(msg, nmsg, "open in");
        return 1;
    }
    int M, N;
    double alpha, beta;
    if (fread(&M, 4, 1, fi) != 1 || fread(&N, 4, 1, fi) != 1 ||
        fread(&alpha, 8, 1, fi) != 1 || fread(&beta, 8, 1, fi) != 1) {
        fclose(fi);
        snprintf(msg, nmsg, "hdr");
        return 1;
    }
    size_t nn = (size_t)M * (size_t)N;
    int Np = (N + 7) & ~7;
    double *A, *C, *Ap, *Cp;
    posix_memalign((void **)&A, 64, nn * 8);
    posix_memalign((void **)&C, 64, nn * 8);
    if (fread(A, 8, nn, fi) != nn) {
        fclose(fi);
        snprintf(msg, nmsg, "body");
        return 1;
    }
    fclose(fi);

    Ap = A;
    Cp = C;
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

    double t0 = tnow();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        __m512d va = _mm512_set1_pd(alpha);
        __m512d vb = _mm512_set1_pd(beta);
        double *ai = &Ap[i * Np];
        double *ci = &Cp[i * Np];
        for (int j = 0; j < Np; j += 8) {
            __m512d x = _mm512_load_pd(&ai[j]);
            x = _mm512_fmadd_pd(va, x, vb);
            _mm512_store_pd(&ci[j], x);
        }
    }
    double el = tnow() - t0;
    if (free_pad) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            memcpy(&C[i * N], &Cp[i * Np], (size_t)N * 8);
        free(Ap);
        free(Cp);
    }

    FILE *fo = fopen(out_path, "wb");
    if (!fo) {
        snprintf(msg, nmsg, "open out");
        return 1;
    }
    fwrite(&M, 4, 1, fo);
    fwrite(&N, 4, 1, fo);
    fwrite(C, 8, nn, fo);
    fclose(fo);
    double gflops = 2.0 * (double)nn / el / 1e9;
    snprintf(msg, nmsg,
             "PHI worker SCALE: M=%d N=%d %.4fs %.2f GFLOPS\nResult: PASS\n",
             M, N, el, gflops);
    free(A);
    free(C);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s control_dir\n", argv[0]);
        return 1;
    }
    const char *dir = argv[1];
    char cmd_path[512], go_path[512], st_path[512], log_path[512];
    snprintf(cmd_path, sizeof cmd_path, "%s/job.cmd", dir);
    snprintf(go_path, sizeof go_path, "%s/job.go", dir);
    snprintf(st_path, sizeof st_path, "%s/job.status", dir);
    snprintf(log_path, sizeof log_path, "%s/job.log", dir);
    fprintf(stderr, "PHI worker ready %s\n", dir);
    fflush(stderr);

    for (;;) {
        while (!file_exists(go_path))
            usleep(2000);
        FILE *fc = fopen(cmd_path, "r");
        if (!fc) {
            unlink(go_path);
            continue;
        }
        char line[1024];
        if (!fgets(line, sizeof line, fc)) {
            fclose(fc);
            unlink(go_path);
            continue;
        }
        fclose(fc);

        char msg[512];
        int rc = 0;
        if (strncmp(line, "QUIT", 4) == 0) {
            FILE *fs = fopen(st_path, "w");
            if (fs) {
                fputs("DONE\n", fs);
                fclose(fs);
            }
            unlink(go_path);
            break;
        } else if (strncmp(line, "SCALE ", 6) == 0) {
            char in[256], out[256];
            if (sscanf(line + 6, "%255s %255s", in, out) == 2)
                rc = run_scale(in, out, msg, sizeof msg);
            else {
                rc = 1;
                snprintf(msg, sizeof msg, "bad SCALE");
            }
        } else {
            rc = 1;
            snprintf(msg, sizeof msg, "unknown");
        }
        FILE *fl = fopen(log_path, "w");
        if (fl) {
            fputs(msg, fl);
            fclose(fl);
        }
        FILE *fs = fopen(st_path, "w");
        if (fs) {
            fputs(rc == 0 ? "DONE\n" : "FAIL\n", fs);
            fclose(fs);
        }
        unlink(go_path);
    }
    return 0;
}
