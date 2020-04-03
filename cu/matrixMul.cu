/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

#define MAX_BLOCK_SIZE 32

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * dim_j and dim_k are column counts of matrices A and B
 */
__global__ void MatrixMulCUDA(float *C, float *A, float *B, const int BLOCK_SIZE, int dim_j, int dim_k) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // process BLOCK_SIZE-sized groups of rows in A
    int aBegin = dim_j * BLOCK_SIZE * by;
    int aEnd   = aBegin + dim_j;
    int aStep  = BLOCK_SIZE;

    // process BLOCK_SIZE-sized groups of columns in B
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * dim_k;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a < aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + dim_j * ty + tx];
        Bs[ty][tx] = B[b + dim_k * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = dim_k * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + dim_k * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int block_size, const int dim_i, const int dim_j, const int dim_k) {

    dim3 dimsA(dim_i, dim_j, 1);
    dim3 dimsB(dim_j, dim_k, 1);
    dim3 dimsC(dim_i, dim_k, 1);

    // Allocate host memory for matrices
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));

    unsigned int size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A);
    cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B);
    cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C);

    // Initialize host memory
    float valA = 2.0f;
    float valB = 0.2f;
    ConstantInit(h_A, size_A, valA);
    ConstantInit(h_B, size_B, valB);

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dim_i / threads.x, dim_k / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    MatrixMulCUDA<<<grid, threads>>>(d_C, d_A, d_B, block_size, dim_i, dim_k);

    cudaDeviceSynchronize();

    printf("done\n");

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaEventCreate(&start);

    cudaEvent_t stop;
    cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);

    // Record the start event
    cudaEventRecord(start, NULL);

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        MatrixMulCUDA<<<grid, threads>>>(d_C, d_A, d_B, block_size, dim_j, dim_k);
    }

    // Record the stop event
    cudaEventRecord(stop, NULL);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dim_i) *
                               static_cast<double>(dim_j) *
                               static_cast<double>(dim_k);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
                       (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time/Iter= %.3f msec, Size= %.0f Ops," \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero

    for (int i = 0; i < static_cast<int>(dim_i * dim_k); i++) {
        double abs_err = fabs(h_C[i] - (dim_j * valA * valB));
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dim_j;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * valB, eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}

int main(int argc, char **argv) {
    printf("WELCOME TO THE MATRIX MULTIPLICATION TEST\n");

    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    int matrix_result;
    for (int i = 32; i <= MAX_BLOCK_SIZE; i++) {
        int DIM_I = (4096 / i) * i;
        int DIM_J = (4096 / i) * i;
        int DIM_K = (4096 / i) * i;
        int block_size = i;
        printf("DIM_I = %d\n", DIM_I);
        printf("DIM_J = %d\n", DIM_J);
        printf("DIM_K = %d\n", DIM_K);
        printf("BLOCK_SIZE = %d\n", block_size);
        matrix_result = MatrixMultiply(block_size, DIM_I, DIM_J, DIM_K);
    }

    exit(matrix_result);
}

