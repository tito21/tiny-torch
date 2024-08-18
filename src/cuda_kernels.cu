
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Sum reduction from https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/03_sum_reduction/reduce_idle/sumReduction.cu
#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction(double* v, double* v_r) {
    // Allocate shared memory
    __shared__ float partial_sum[SHMEM_SIZE];

    // Calculate thread ID
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Store first partial result instead of just the elements
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

__global__ void set_array_kernel(double* arr, double value, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        arr[i] = value;
    }
}

__global__ void sum_arrays_kernel(double* a, double* b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sum_accu_arrays_kernel(double* a, double* b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] += a[i] + b[i];
    }
}

__global__ void sum_accu_scalar_kernel(double* a, double b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] += a[i] + b;
    }
}

__global__ void sub_arrays_kernel(double* a, double* b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = a[i] - b[i];
    }
}

__global__ void sub_accu_arrays_kernel(double* a, double* b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] += a[i] - b[i];
    }
}

__global__ void mul_arrays_kernel(double* a, double* b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = a[i] * b[i];
    }
}

__global__ void mul_accu_arrays_kernel(double* a, double* b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] += a[i] * b[i];
    }
}

__global__ void mul_accu_scalar_kernel(double* a, double b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] += a[i] * b;
    }
}

__global__ void div_arrays_kernel(double* a, double* b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = a[i] / b[i];
    }
}

__global__ void div_accu_arrays_kernel(double* a, double* b, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] += a[i] / b[i];
    }
}

__global__ void matmul_kernel(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k)
{
    // a: m x n
    // b: l x k
    // n = l
    // (m x n) @ (l x k) = m x k
    // c_{xy} = sum_{i} a_{xi} * b_{iy}
    for (size_t x = 0; x < m; x++) {
        for (size_t y = 0; y < k; y++) {
            c[x * k + y] = 0;
            for (size_t i = 0; i < n; i++) {
                c[x * k + y] += a[x * n + i] * b[i * k + y];
            }
        }
    }
}

__global__ void matmul_transpose_a_kernel(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k)
{
    // m = l
    // (m x n).T @ (l x k) = n x k
    // c_{xy} = sum_{i} a_{ix} * b_{iy}
    for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < k; y++) {
            c[x * k + y] = 0;
            for (size_t i = 0; i < m; i++) {
                c[x * k + y] += a[i * n + x] * b[i * k + y];
            }
        }
    }
}

__global__ void matmul_transpose_b_kernel(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k)
{
    // n = k
    // (m x n) @ (l x k).T = m x l
    // c_{xy} = sum_{i} a_{xi} * b_{yi}
    for (size_t x = 0; x < m; x++) {
        for (size_t y = 0; y < l; y++) {
            c[x * l + y] = 0;
            for (size_t i = 0; i < n; i++) {
                c[x * l + y] += a[x * n + i] * b[y * k + i];
            }
        }
    }
}

__global__ void matmul_transpose_ab_kernel(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k)
{
    // m = k
    // (m x n).T @ (l x k).T = n x l
    // c_{xy} = sum_{i} a_{ix} * b_{yi}
    for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < l; y++) {
            c[x * l + y] = 0;
            for (size_t i = 0; i < m; i++) {
                c[x * l + y] += a[i * n + x] * b[y * k + i];
            }
        }
    }
}

__global__ void sigmoid_array_kernel(double* out, double* x, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        out[i] = 1.0 / (1.0 + exp(-x[i]));
    }
}

__global__ void sigmoid_backward_kernel(double* grad, double* out, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] += grad[i] * out[i] * (1.0 - out[i]);
    }
}

__global__ void tanh_array_kernel(double* a, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = tanh(a[i]);
    }
}

__global__ void tanh_backward_kernel(double* grad, double* out, double* c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = grad[i] * (1 - out[i] * out[i]);
    }
}