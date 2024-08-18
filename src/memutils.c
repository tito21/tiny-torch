
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "memutils.h"

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%d %s\n", file, line, error,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void *memutils_cuda_malloc(size_t size) {
  void *ptr;
  cudaCheck(cudaMalloc(&ptr, size));
  return ptr;
}

void memutils_cuda_free(void *ptr) {
  cudaCheck(cudaFree(ptr));
}

void memutils_to_device(void *dst, const void *src, size_t size) {
  cudaCheck(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void memutils_to_host(void *dst, const void *src, size_t size) {
  cudaCheck(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}