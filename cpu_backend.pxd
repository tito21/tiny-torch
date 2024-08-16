cdef extern from "cpu_backend.c":

    double* new_array(size_t size)
    void free_array(double* arr)
    void set_array(double* arr, double value, size_t size)
    void copy_array(double* dest, double* src, size_t size)
    void sum_arrays(double* a, double* b, double* c, size_t size)
