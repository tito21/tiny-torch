cdef extern from "cpu_backend.c":

    double* new_array(size_t size)
    void free_array(double* arr)
    void set_array(double* arr, double value, size_t size)
    void copy_array(double* dest, double* src, size_t size)

    void sum_arrays(double* a, double* b, double* c, size_t size)
    void sum_accu_arrays(double* a, double* b, double* c, size_t size)
    void sum_accu_scalar(double* a, double b, double* c, size_t size)

    void sub_arrays(double* a, double* b, double* c, size_t size)
    void sub_accu_arrays(double* a, double* b, double* c, size_t size)

    void mul_arrays(double* a, double* b, double* c, size_t size)
    void mul_accu_arrays(double* a, double* b, double* c, size_t size)
    void mul_accu_scalar(double* a, double b, double* c, size_t size)

    void div_arrays(double* a, double* b, double* c, size_t size)
    void div_accu_arrays(double* a, double* b, double* c, size_t size)

    int matmul(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k, int transpose_a, int transpose_b)

    void tanh_array(double* a, double* c, size_t size)
    void tanh_backward(double* grad, double* out, double* c, size_t size)

    void sigmoid_array(double* a, double* c, size_t size)
    void sigmoid_backward(double* grad, double* out, double* c, size_t size)