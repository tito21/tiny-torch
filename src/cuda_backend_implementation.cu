#include <cuda_runtime.h>

#include "cuda_kernels.cu"
#include "memutils.h"

#define BLOCK_SIZE 32

double* cuda_new_array(size_t size)
{
    return (double* ) memutils_cuda_malloc(size * sizeof(double));
}

void cuda_free_array(double* arr) {
    memutils_cuda_free(arr);
}

void cuda_set_array(double* arr, double value, size_t size)
{
    set_array_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(arr, value, size);
    cudaCheck(cudaGetLastError());
}

void cuda_copy_array(double* dest, double* src, size_t size)
{
    cudaCheck(cudaMemcpy(dest, src, size * sizeof(double), cudaMemcpyDeviceToDevice));
}

void cuda_sum_arrays(double* a, double* b, double* c, size_t size)
{
    sum_arrays_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_sum_accu_arrays(double* a, double* b, double* c, size_t size)
{
    sum_accu_arrays_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_sum_accu_scalar(double* a, double b, double* c, size_t size)
{
    sum_accu_scalar_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_sub_arrays(double* a, double* b, double* c, size_t size)
{
    sub_arrays_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_sub_accu_arrays(double* a, double* b, double* c, size_t size)
{
    sub_accu_arrays_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_mul_arrays(double* a, double* b, double* c, size_t size)
{
    mul_arrays_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_mul_accu_arrays(double* a, double* b, double* c, size_t size)
{
    mul_accu_arrays_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_mul_accu_scalar(double* a, double b, double* c, size_t size)
{
    mul_accu_scalar_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_div_arrays(double* a, double* b, double* c, size_t size)
{
    div_arrays_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

void cuda_div_accu_arrays(double* a, double* b, double* c, size_t size)
{
    div_accu_arrays_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, size);
    cudaCheck(cudaGetLastError());
}

int cuda_matmul(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k, int transpose_a, int transpose_b)
{
    // a: m x n
    // b: l x k
    if (!transpose_a && !transpose_b) {
        // n = l
        // (m x n) @ (l x k) = m x k
        // c_{xy} = sum_{i} a_{xi} * b_{iy}
        if (n != l) {
            return -1;
        }
        dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((k + threads_per_block.x - 1) / threads_per_block.x, (m + threads_per_block.y - 1) / threads_per_block.y);
        matmul_kernel<<<blocks, threads_per_block>>>(a, b, c, m, n, l, k);
        cudaCheck(cudaGetLastError());

    } else if (transpose_a && !transpose_b) {
        // m = l
        // (m x n).T @ (l x k) = n x k
        // c_{xy} = sum_{i} a_{ix} * b_{iy}
        if (m != l) {
            return -1;
        }

        dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((k + threads_per_block.x - 1) / threads_per_block.x, (n + threads_per_block.y - 1) / threads_per_block.y);
        matmul_transpose_a_kernel<<<blocks, threads_per_block>>>(a, b, c, m, n, l, k);
        cudaCheck(cudaGetLastError());

    } else if (!transpose_a && transpose_b) {
        // n = k
        // (m x n) @ (l x k).T = m x l
        // c_{xy} = sum_{i} a_{xi} * b_{yi}
        if (n != k) {
            return -1;
        }
        dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((l + threads_per_block.x - 1) / threads_per_block.x, (m + threads_per_block.y - 1) / threads_per_block.y);
        matmul_transpose_b_kernel<<<blocks, threads_per_block>>>(a, b, c, m, n, l, k);
        cudaCheck(cudaGetLastError());

    } else {
        // m = k
        // (m x n).T @ (l x k).T = n x l
        // c_{xy} = sum_{i} a_{ix} * b_{yi}
        if (m != k) {
            return -1;
        }
        dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((l + threads_per_block.x - 1) / threads_per_block.x, (n + threads_per_block.y - 1) / threads_per_block.y);
        matmul_transpose_ab_kernel<<<blocks, threads_per_block>>>(a, b, c, m, n, l, k);
        cudaCheck(cudaGetLastError());

    }
    return 0;
}

void cuda_sigmoid_array(double* a, double* c, size_t size)
{
    sigmoid_array_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(c, a, size);
}

void cuda_sigmoid_backward(double* grad, double* out, double* c, size_t size)
{
    sigmoid_backward_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(grad, out, c, size);
}

void cuda_tanh_array(double* a, double* c, size_t size)
{
    tanh_array_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, c, size);
}

void cuda_tanh_backward(double* grad, double* out, double* c, size_t size)
{
    tanh_backward_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(grad, out, c, size);
}

void cuda_relu_array(double* a, double* c, size_t size)
{
    relu_array_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, c, size);
}

void cuda_relu_backward(double* grad, double* out, double* c, size_t size)
{
    relu_backward_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(grad, out, c, size);
}