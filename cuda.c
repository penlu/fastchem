/* CUDA library interface */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>

#include "cuda.h"

// error handlers

#define CUDA_CALL(x) check_cuda_error(x, __FILE__, __LINE__);
#define CUBLAS_CALL(x) check_cublas_error(x, __FILE__, __LINE__);
#define CUSPARSE_CALL(x) check_cusparse_error(x, __FILE__, __LINE__);
#define CURAND_CALL(x) check_curand_error(x, __FILE__, __LINE__);

void check_cuda_error(cudaError_t error, char *file, int line) {
    if (error) {
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(0);
    }
}

void check_cublas_error(cublasStatus_t status, char *file, int line) {
    char *s;
#ifdef DEBUG
    cuda_device_synchronize();
#endif
    // taken from helper_cuda.h
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return;

        case CUBLAS_STATUS_NOT_INITIALIZED:
            s = "CUBLAS_STATUS_NOT_INITIALIZED";
            break;

        case CUBLAS_STATUS_ALLOC_FAILED:
            s = "CUBLAS_STATUS_ALLOC_FAILED";
            break;

        case CUBLAS_STATUS_INVALID_VALUE:
            s = "CUBLAS_STATUS_INVALID_VALUE";
            break;

        case CUBLAS_STATUS_ARCH_MISMATCH:
            s = "CUBLAS_STATUS_ARCH_MISMATCH";
            break;

        case CUBLAS_STATUS_MAPPING_ERROR:
            s = "CUBLAS_STATUS_MAPPING_ERROR";
            break;

        case CUBLAS_STATUS_EXECUTION_FAILED:
            s = "CUBLAS_STATUS_EXECUTION_FAILED";
            break;

        case CUBLAS_STATUS_INTERNAL_ERROR:
            s = "CUBLAS_STATUS_INTERNAL_ERROR";
            break;

        case CUBLAS_STATUS_NOT_SUPPORTED:
            s = "CUBLAS_STATUS_NOT_SUPPORTED";
            break;

        case CUBLAS_STATUS_LICENSE_ERROR:
            s = "CUBLAS_STATUS_LICENSE_ERROR";
            break;

        default:
            s = "<unknown>";
            break;
    }

    printf("CUBLAS error: %s at %s:%d\n", s, file, line);
    exit(0);
}

void check_cusparse_error(cusparseStatus_t status, char *file, int line) {
    char *s;
#ifdef DEBUG
    cuda_device_synchronize();
#endif
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS:
            return;

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            s = "CUSPARSE_STATUS_NOT_INITIALIZED";
            break;

        case CUSPARSE_STATUS_ALLOC_FAILED:
            s = "CUSPARSE_STATUS_ALLOC_FAILED";
            break;

        case CUSPARSE_STATUS_INVALID_VALUE:
            s = "CUSPARSE_STATUS_INVALID_VALUE";
            break;

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            s = "CUSPARSE_STATUS_ARCH_MISMATCH";
            break;

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            s = "CUSPARSE_STATUS_EXECUTION_FAILED";
            break;

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            s = "CUSPARSE_STATUS_INTERNAL_ERROR";
            break;

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            s = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            break;

        default:
            s = "<unknown>";
            break;
    }

    printf("CUSPARSE error: %s at %s:%d\n", s, file, line);
    exit(0);
}

void check_curand_error(curandStatus_t status, char *file, int line) {
    char *s;
#ifdef DEBUG
    cuda_device_synchronize();
#endif
    switch (status) {
        case CURAND_STATUS_SUCCESS:
            return;

        case CURAND_STATUS_VERSION_MISMATCH:
            s = "CURAND_STATUS_VERSION_MISMATCH";
            break;
        
        case CURAND_STATUS_NOT_INITIALIZED:
            s = "CURAND_STATUS_NOT_INITIALIZED";
            break;

        case CURAND_STATUS_ALLOCATION_FAILED:
            s = "CURAND_STATUS_ALLOCATION_FAILED";
            break;
        
        case CURAND_STATUS_TYPE_ERROR:
            s = "CURAND_STATUS_TYPE_ERROR";
            break;

        case CURAND_STATUS_OUT_OF_RANGE:
            s = "CURAND_STATUS_OUT_OF_RANGE";
            break;

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            s = "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
            break;
        
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            s = "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
            break;

        case CURAND_STATUS_LAUNCH_FAILURE:
            s = "CURAND_STATUS_LAUNCH_FAILURE";
            break;

        case CURAND_STATUS_PREEXISTING_FAILURE:
            s = "CURAND_STATUS_PREEXISTING_FAILURE";
            break;

        case CURAND_STATUS_INITIALIZATION_FAILED:
            s = "CURAND_STATUS_INITIALIZATION_FAILED";
            break;

        case CURAND_STATUS_ARCH_MISMATCH:
            s = "CURAND_STATUS_ARCH_MISMATCH";
            break;

        case CURAND_STATUS_INTERNAL_ERROR:
            s = "CURAND_STATUS_INTERNAL_ERROR";
            break;

        default:
            s = "<unknown>";
            break;
    }

    printf("CURAND error: %s at %s:%d\n", s, file, line);
    exit(0);
}

// CUDA library calls

void cuda_malloc(void **p, size_t size) {
    CUDA_CALL(cudaMalloc(p, size));
}

void cuda_free(void *p) {
    CUDA_CALL(cudaFree(p));
}

void cuda_memcpy_htod(void *dp, void *hp, size_t size) {
    CUDA_CALL(cudaMemcpy(dp, hp, size, cudaMemcpyHostToDevice));
}

void cuda_memcpy_dtoh(void *hp, void *dp, size_t size) {
    CUDA_CALL(cudaMemcpy(hp, dp, size, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_2d(void *dst, size_t dpitch, void *src, size_t spitch,
        size_t width, size_t height) {
    CUDA_CALL(cudaMemcpy2D(dst, dpitch, src, spitch,
        width, height, cudaMemcpyDeviceToDevice));
}

void cuda_device_synchronize() {
    CUDA_CALL(cudaDeviceSynchronize());
}

// CUBLAS library calls

cublasHandle_t cublas_handle() {
    static int init = 0;
    static cublasHandle_t handle;
    if (!init) {
        CUBLAS_CALL(cublasCreate(&handle));
        init = 1;
    }
    return handle;
}

// C = alpha * A * B + beta * C
// matrices are in COLUMN-MAJOR FORMAT
// dimensions are listed as rows x cols
// A is m x k (or k x m if transa)
// B is k x n (or n x k if transb)
// C is m x n
// lda, ldb, ldc should be the row count of the actual allocation
void cublas_sgemm(int transa, int transb, int m, int n, int k,
        float alpha, float *A, int lda, float *B, int ldb,
        float beta, float *C, int ldc) {
    CUBLAS_CALL(cublasSgemm(cublas_handle(),
        transa ? CUBLAS_OP_T : CUBLAS_OP_N,
        transb ? CUBLAS_OP_T : CUBLAS_OP_N,
        m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

// y = alpha * A * x + beta * y
// matrices are in COLUMN-MAJOR FORMAT
// dimensions are listed as rows x cols
// A is m x n (or n x m if transa)
// x is m (or n if transa)
// y is n
// lda should be the row count of the actual allocation
void cublas_sgemv(int transa, int m, int n,
        float alpha, float *A, int lda, float *x, float beta, float *y) {
    CUBLAS_CALL(cublasSgemv(cublas_handle(),
        transa ? CUBLAS_OP_T : CUBLAS_OP_N,
        m, n, &alpha, A, lda, x, 1, &beta, y, 1));
}

// CUSPARSE library calls

cusparseHandle_t cusparse_handle() {
    static int init = 0;
    static cusparseHandle_t handle;
    if (!init) {
        CUSPARSE_CALL(cusparseCreate(&handle));
        init = 1;
    }
    return handle;
}

void cusparse_sgemmi(int m, int n, int k, int nnz,
        float alpha, float *A, int lda,
        float *cscValB, int *cscColPtrB, int *cscRowIndB,
        float beta, float *C, int ldc) {
    CUSPARSE_CALL(cusparseSgemmi(cusparse_handle(),
        m, n, k, nnz, &alpha, A, lda, cscValB, cscColPtrB, cscRowIndB,
        &beta, C, ldc));
}

void cusparse_scsrmm2(int transa, int transb, int m, int n, int k,
        int nnz, float alpha, float *csrValA, int *csrRowPtrA, int *csrColIndA,
        float *B, int ldb, float beta, float *C, int ldc) {
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);

    CUSPARSE_CALL(cusparseScsrmm2(cusparse_handle(),
        transa ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
        transb ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
        B, ldb, &beta, C, ldc));
}

// CURAND library calls

curandGenerator_t curand_generator() {
    static int init = 0;
    static curandGenerator_t gen;
    if (!init) {
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        init = 1;
    }
    return gen;
}

void curand_seed_generator(unsigned long long seed) {
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(curand_generator(), seed));
}

void curand_generate_uniform(float *output, size_t n) {
    CURAND_CALL(curandGenerateUniform(curand_generator(), output, n));
}
