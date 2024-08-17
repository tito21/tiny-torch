#include <stdlib.h>
#include <string.h>
#include <math.h>

double* new_array(size_t size)
{
    return (double* ) malloc(size * sizeof(double));
}

void free_array(double* arr) {
    free(arr);
}

void set_array(double* arr, double value, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        arr[i] = value;
    }
}

void copy_array(double* dest, double* src, size_t size)
{
    memcpy(dest, src, size * sizeof(double));
}

void sum_arrays(double* a, double* b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

void sum_accu_arrays(double* a, double* b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] += a[i] + b[i];
    }
}

void sub_arrays(double* a, double* b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] - b[i];
    }
}

void sub_accu_arrays(double* a, double* b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] += a[i] - b[i];
    }
}

void mul_arrays(double* a, double* b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

void mul_accu_arrays(double* a, double* b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] += a[i] * b[i];
    }
}

void div_arrays(double* a, double* b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] / b[i];
    }
}

void div_accu_arrays(double* a, double* b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] += a[i] / b[i];
    }
}

void tanh_array(double* a, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] = tanh(a[i]);
    }
}

void tanh_backward(double* grad, double* out, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] = grad[i] * (1 - out[i] * out[i]);
    }
}