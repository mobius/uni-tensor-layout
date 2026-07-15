/*
 * VE-side shared library for AVEO offload DGEMM (NLC cblas).
 *
 * ncc -O3 -fpic -shared -o libve_dgemm.so dgemm_lib.c \
 *   -I$NLC/include -L$NLC/lib -lcblas -lblas_openmp -Wl,-rpath,$NLC/lib
 */

#include <cblas.h>
#include <stdint.h>

/* Pointers are VE virtual addresses passed from host as uint64. */
int64_t ve_dgemm_rm(uint64_t a_ptr, uint64_t b_ptr, uint64_t c_ptr,
                    int64_t M, int64_t N, int64_t K)
{
    const double *A = (const double *)(uintptr_t)a_ptr;
    const double *B = (const double *)(uintptr_t)b_ptr;
    double *C = (double *)(uintptr_t)c_ptr;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0, A, (int)K, B, (int)N, 0.0, C, (int)N);
    return 0;
}
