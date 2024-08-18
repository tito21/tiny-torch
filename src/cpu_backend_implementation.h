#include <stdlib.h>
#include <string.h>
#include <math.h>

double* cpu_new_array(size_t size);
void cpu_free_array(double* arr);
void cpu_set_array(double* arr, double value, size_t size);
void cpu_copy_array(double* dest, double* src, size_t size);

void cpu_sum_arrays(double* a, double* b, double* c, size_t size);
void cpu_sum_accu_arrays(double* a, double* b, double* c, size_t size);
void cpu_sum_accu_scalar(double* a, double b, double* c, size_t size);

void cpu_sub_arrays(double* a, double* b, double* c, size_t size);
void cpu_sub_accu_arrays(double* a, double* b, double* c, size_t size);

void cpu_mul_arrays(double* a, double* b, double* c, size_t size);
void cpu_mul_accu_arrays(double* a, double* b, double* c, size_t size);
void cpu_mul_accu_scalar(double* a, double b, double* c, size_t size);

void cpu_div_arrays(double* a, double* b, double* c, size_t size);
void cpu_div_accu_arrays(double* a, double* b, double* c, size_t size);

int cpu_matmul(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k, int transpose_a, int transpose_b);

void cpu_sigmoid_array(double* a, double* c, size_t size);
void cpu_sigmoid_backward(double* grad, double* out, double* c, size_t size);

void cpu_tanh_array(double* a, double* c, size_t size);
void cpu_tanh_backward(double* grad, double* out, double* c, size_t size);

void cpu_relu_array(double* a, double* c, size_t size);
void cpu_relu_backward(double* grad, double* out, double* c, size_t size);