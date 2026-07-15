/*
 * Host helper: AVEO DGEMM (alloc/write/call/read on one VE node).
 *
 * gcc -O2 -fPIC -shared -o libhost_veo_dgemm.so veo_dgemm_host.c \
 *   -I/opt/nec/ve/veos/include -L/opt/nec/ve/veos/lib64 -lveo \
 *   -Wl,-rpath,/opt/nec/ve/veos/lib64
 *
 * int aveo_dgemm(int node, int M, int N, int K,
 *                const double *A, const double *B, double *C,
 *                const char *ve_lib_path, double *elapsed_out);
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ve_offload.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static double tnow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

EXPORT int aveo_dgemm(int node, int M, int N, int K,
                      const double *A, const double *B, double *C,
                      const char *ve_lib_path, double *elapsed_out)
{
    if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C || !ve_lib_path)
        return -1;

    struct veo_proc_handle *proc = veo_proc_create(node);
    if (!proc)
        return -2;

    uint64_t libh = veo_load_library(proc, ve_lib_path);
    if (!libh) {
        veo_proc_destroy(proc);
        return -3;
    }
    uint64_t sym = veo_get_sym(proc, libh, "ve_dgemm_rm");
    if (!sym) {
        veo_unload_library(proc, libh);
        veo_proc_destroy(proc);
        return -4;
    }

    size_t a_bytes = (size_t)M * (size_t)K * sizeof(double);
    size_t b_bytes = (size_t)K * (size_t)N * sizeof(double);
    size_t c_bytes = (size_t)M * (size_t)N * sizeof(double);
    uint64_t va = 0, vb = 0, vc = 0;
    if (veo_alloc_mem(proc, &va, a_bytes) ||
        veo_alloc_mem(proc, &vb, b_bytes) ||
        veo_alloc_mem(proc, &vc, c_bytes)) {
        veo_unload_library(proc, libh);
        veo_proc_destroy(proc);
        return -5;
    }
    if (veo_write_mem(proc, va, A, a_bytes) ||
        veo_write_mem(proc, vb, B, b_bytes)) {
        veo_free_mem(proc, va);
        veo_free_mem(proc, vb);
        veo_free_mem(proc, vc);
        veo_unload_library(proc, libh);
        veo_proc_destroy(proc);
        return -6;
    }

    struct veo_args *arg = veo_args_alloc();
    veo_args_set_u64(arg, 0, va);
    veo_args_set_u64(arg, 1, vb);
    veo_args_set_u64(arg, 2, vc);
    veo_args_set_i64(arg, 3, M);
    veo_args_set_i64(arg, 4, N);
    veo_args_set_i64(arg, 5, K);

    double t0 = tnow();
    uint64_t retval = 0;
    int rc = veo_call_sync(proc, sym, arg, &retval);
    double el = tnow() - t0;
    if (elapsed_out)
        *elapsed_out = el;

    if (rc != 0 || veo_read_mem(proc, C, vc, c_bytes) != 0) {
        veo_args_free(arg);
        veo_free_mem(proc, va);
        veo_free_mem(proc, vb);
        veo_free_mem(proc, vc);
        veo_unload_library(proc, libh);
        veo_proc_destroy(proc);
        return -7;
    }

    veo_args_free(arg);
    veo_free_mem(proc, va);
    veo_free_mem(proc, vb);
    veo_free_mem(proc, vc);
    veo_unload_library(proc, libh);
    veo_proc_destroy(proc);
    return 0;
}

/* Persistent-session API: create once, many gemms */
struct aveo_session {
    struct veo_proc_handle *proc;
    struct veo_thr_ctxt *ctx; /* async context; may be NULL */
    uint64_t libh;
    uint64_t sym;
    int node;
};

EXPORT struct aveo_session *aveo_session_open(int node, const char *ve_lib_path)
{
    struct aveo_session *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->node = node;
    s->proc = veo_proc_create(node);
    if (!s->proc) {
        free(s);
        return NULL;
    }
    s->libh = veo_load_library(s->proc, ve_lib_path);
    if (!s->libh) {
        veo_proc_destroy(s->proc);
        free(s);
        return NULL;
    }
    s->sym = veo_get_sym(s->proc, s->libh, "ve_dgemm_rm");
    if (!s->sym) {
        veo_unload_library(s->proc, s->libh);
        veo_proc_destroy(s->proc);
        free(s);
        return NULL;
    }
    s->ctx = veo_context_open(s->proc); /* optional async path */
    return s;
}

EXPORT int aveo_session_dgemm(struct aveo_session *s, int M, int N, int K,
                              const double *A, const double *B, double *C,
                              double *elapsed_out)
{
    if (!s || !s->proc)
        return -1;
    size_t a_bytes = (size_t)M * K * sizeof(double);
    size_t b_bytes = (size_t)K * N * sizeof(double);
    size_t c_bytes = (size_t)M * N * sizeof(double);
    uint64_t va = 0, vb = 0, vc = 0;
    if (veo_alloc_mem(s->proc, &va, a_bytes) ||
        veo_alloc_mem(s->proc, &vb, b_bytes) ||
        veo_alloc_mem(s->proc, &vc, c_bytes))
        return -2;
    veo_write_mem(s->proc, va, A, a_bytes);
    veo_write_mem(s->proc, vb, B, b_bytes);
    struct veo_args *arg = veo_args_alloc();
    veo_args_set_u64(arg, 0, va);
    veo_args_set_u64(arg, 1, vb);
    veo_args_set_u64(arg, 2, vc);
    veo_args_set_i64(arg, 3, M);
    veo_args_set_i64(arg, 4, N);
    veo_args_set_i64(arg, 5, K);
    double t0 = tnow();
    uint64_t retval = 0;
    int rc = veo_call_sync(s->proc, s->sym, arg, &retval);
    double el = tnow() - t0;
    if (elapsed_out)
        *elapsed_out = el;
    if (rc == 0)
        veo_read_mem(s->proc, C, vc, c_bytes);
    veo_args_free(arg);
    veo_free_mem(s->proc, va);
    veo_free_mem(s->proc, vb);
    veo_free_mem(s->proc, vc);
    return rc == 0 ? 0 : -3;
}

/*
 * Async-phase DGEMM: async H2D (A||B) → wait → async call → wait → async D2H.
 * Fills optional phase times: h2d_out, kernel_out, d2h_out (seconds).
 */
EXPORT int aveo_session_dgemm_async(struct aveo_session *s, int M, int N, int K,
                                    const double *A, const double *B, double *C,
                                    double *h2d_out, double *kernel_out,
                                    double *d2h_out, double *elapsed_out)
{
    if (!s || !s->proc || !s->ctx)
        return -1;
    size_t a_bytes = (size_t)M * K * sizeof(double);
    size_t b_bytes = (size_t)K * N * sizeof(double);
    size_t c_bytes = (size_t)M * N * sizeof(double);
    uint64_t va = 0, vb = 0, vc = 0;
    if (veo_alloc_mem(s->proc, &va, a_bytes) ||
        veo_alloc_mem(s->proc, &vb, b_bytes) ||
        veo_alloc_mem(s->proc, &vc, c_bytes))
        return -2;

    double t0 = tnow();
    uint64_t ra = veo_async_write_mem(s->ctx, va, A, a_bytes);
    uint64_t rb = veo_async_write_mem(s->ctx, vb, B, b_bytes);
    uint64_t res = 0;
    if (veo_call_wait_result(s->ctx, ra, &res) != VEO_COMMAND_OK ||
        veo_call_wait_result(s->ctx, rb, &res) != VEO_COMMAND_OK) {
        veo_free_mem(s->proc, va);
        veo_free_mem(s->proc, vb);
        veo_free_mem(s->proc, vc);
        return -3;
    }
    double t1 = tnow();

    struct veo_args *arg = veo_args_alloc();
    veo_args_set_u64(arg, 0, va);
    veo_args_set_u64(arg, 1, vb);
    veo_args_set_u64(arg, 2, vc);
    veo_args_set_i64(arg, 3, M);
    veo_args_set_i64(arg, 4, N);
    veo_args_set_i64(arg, 5, K);
    uint64_t rcid = veo_call_async(s->ctx, s->sym, arg);
    if (veo_call_wait_result(s->ctx, rcid, &res) != VEO_COMMAND_OK) {
        veo_args_free(arg);
        veo_free_mem(s->proc, va);
        veo_free_mem(s->proc, vb);
        veo_free_mem(s->proc, vc);
        return -4;
    }
    double t2 = tnow();
    veo_args_free(arg);

    uint64_t rr = veo_async_read_mem(s->ctx, C, vc, c_bytes);
    if (veo_call_wait_result(s->ctx, rr, &res) != VEO_COMMAND_OK) {
        veo_free_mem(s->proc, va);
        veo_free_mem(s->proc, vb);
        veo_free_mem(s->proc, vc);
        return -5;
    }
    double t3 = tnow();

    if (h2d_out)
        *h2d_out = t1 - t0;
    if (kernel_out)
        *kernel_out = t2 - t1;
    if (d2h_out)
        *d2h_out = t3 - t2;
    if (elapsed_out)
        *elapsed_out = t3 - t0;

    veo_free_mem(s->proc, va);
    veo_free_mem(s->proc, vb);
    veo_free_mem(s->proc, vc);
    return 0;
}

EXPORT void aveo_session_close(struct aveo_session *s)
{
    if (!s)
        return;
    if (s->proc) {
        if (s->ctx)
            veo_context_close(s->ctx);
        if (s->libh)
            veo_unload_library(s->proc, s->libh);
        veo_proc_destroy(s->proc);
    }
    free(s);
}
