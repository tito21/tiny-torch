#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line);
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))


void *memutils_cuda_malloc(size_t size);
void memutils_cuda_free(void *ptr);
void memutils_to_device(void *dst, const void *src, size_t size);
void memutils_to_host(void *dst, const void *src, size_t size);