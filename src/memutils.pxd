cdef extern from "memutils.c":

    void *memutils_cuda_malloc(size_t size)
    void memutils_cuda_free(void *ptr)
    void memutils_to_device(void *dst, const void *src, size_t size)
    void memutils_to_host(void *dst, const void *src, size_t size)
