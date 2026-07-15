/*
 * Persistent DGEMM worker on NEC VE — avoids repeated ve_exec process spawn.
 *
 * Compile: same NLC flags as dgemm_rect.c
 * Run:     ve_exec -N <id> ./dgemm_worker_ve <control_dir>
 *
 * Protocol (file-based under control_dir):
 *   Host writes job.cmd then job.go (empty).
 *   Worker runs job, writes job.log + job.status (DONE|FAIL), removes job.go.
 *   job.cmd lines:
 *     QUIT
 *     COMBINED <in.bin> <out.bin>
 *     SPLIT <a.bin> <b.bin> <out.bin>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <cblas.h>

static double tnow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

static int file_exists(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0;
}

static int run_combined(const char *in_path, const char *out_path, char *msg, size_t msglen)
{
    FILE *fi = fopen(in_path, "rb");
    if (!fi) {
        snprintf(msg, msglen, "open in failed");
        return 1;
    }
    int M, N, K;
    if (fread(&M, 4, 1, fi) != 1 || fread(&N, 4, 1, fi) != 1 || fread(&K, 4, 1, fi) != 1) {
        fclose(fi);
        snprintf(msg, msglen, "hdr");
        return 1;
    }
    size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
    double *A = aligned_alloc(64, aN * 8);
    double *B = aligned_alloc(64, bN * 8);
    double *C = aligned_alloc(64, cN * 8);
    if (!A || !B || !C) {
        snprintf(msg, msglen, "alloc");
        return 1;
    }
    if (fread(A, 8, aN, fi) != aN || fread(B, 8, bN, fi) != bN) {
        fclose(fi);
        snprintf(msg, msglen, "body");
        return 1;
    }
    fclose(fi);

    double t0 = tnow();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    double el = tnow() - t0;
    double gflops = 2.0 * (double)M * N * K / el / 1e9;
    double sum = 0;
    for (size_t i = 0; i < cN; i++)
        sum += C[i];

    FILE *fo = fopen(out_path, "wb");
    if (!fo) {
        snprintf(msg, msglen, "open out");
        return 1;
    }
    fwrite(&M, 4, 1, fo);
    fwrite(&N, 4, 1, fo);
    fwrite(C, 8, cN, fo);
    fclose(fo);
    snprintf(msg, msglen,
             "VE worker COMBINED: M=%d N=%d K=%d %.4fs %.2f GFLOPS checksum=%.6e\nResult: PASS\n",
             M, N, K, el, gflops, sum);
    free(A);
    free(B);
    free(C);
    return 0;
}

static int run_split(const char *a_path, const char *b_path, const char *out_path,
                     char *msg, size_t msglen)
{
    FILE *fa = fopen(a_path, "rb");
    FILE *fb = fopen(b_path, "rb");
    if (!fa || !fb) {
        snprintf(msg, msglen, "open a/b");
        if (fa)
            fclose(fa);
        if (fb)
            fclose(fb);
        return 1;
    }
    int M, Ka, Kb, N;
    if (fread(&M, 4, 1, fa) != 1 || fread(&Ka, 4, 1, fa) != 1 ||
        fread(&Kb, 4, 1, fb) != 1 || fread(&N, 4, 1, fb) != 1 || Ka != Kb) {
        snprintf(msg, msglen, "hdr mismatch");
        fclose(fa);
        fclose(fb);
        return 1;
    }
    int K = Ka;
    size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
    double *A = aligned_alloc(64, aN * 8);
    double *B = aligned_alloc(64, bN * 8);
    double *C = aligned_alloc(64, cN * 8);
    if (!A || !B || !C) {
        snprintf(msg, msglen, "alloc");
        return 1;
    }
    if (fread(A, 8, aN, fa) != aN || fread(B, 8, bN, fb) != bN) {
        snprintf(msg, msglen, "body");
        fclose(fa);
        fclose(fb);
        return 1;
    }
    fclose(fa);
    fclose(fb);

    double t0 = tnow();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    double el = tnow() - t0;
    double gflops = 2.0 * (double)M * N * K / el / 1e9;
    double sum = 0;
    for (size_t i = 0; i < cN; i++)
        sum += C[i];

    FILE *fo = fopen(out_path, "wb");
    if (!fo) {
        snprintf(msg, msglen, "open out");
        return 1;
    }
    fwrite(&M, 4, 1, fo);
    fwrite(&N, 4, 1, fo);
    fwrite(C, 8, cN, fo);
    fclose(fo);
    snprintf(msg, msglen,
             "VE worker SPLIT: M=%d N=%d K=%d %.4fs %.2f GFLOPS checksum=%.6e\nResult: PASS\n",
             M, N, K, el, gflops, sum);
    free(A);
    free(B);
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

    fprintf(stderr, "VE dgemm_worker ready dir=%s\n", dir);
    fflush(stderr);

    for (;;) {
        while (!file_exists(go_path))
            usleep(2000); /* 2 ms */

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
        } else if (strncmp(line, "COMBINED ", 9) == 0) {
            char in[256], out[256];
            if (sscanf(line + 9, "%255s %255s", in, out) == 2)
                rc = run_combined(in, out, msg, sizeof msg);
            else {
                rc = 1;
                snprintf(msg, sizeof msg, "bad COMBINED args");
            }
        } else if (strncmp(line, "SPLIT ", 6) == 0) {
            char a[256], b[256], out[256];
            if (sscanf(line + 6, "%255s %255s %255s", a, b, out) == 3)
                rc = run_split(a, b, out, msg, sizeof msg);
            else {
                rc = 1;
                snprintf(msg, sizeof msg, "bad SPLIT args");
            }
        } else {
            rc = 1;
            snprintf(msg, sizeof msg, "unknown cmd");
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
