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

void sum_accu_scalar(double* a, double b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] += a[i] + b;
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

void mul_accu_scalar(double* a, double b, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] += a[i] * b;
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

int matmul(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k, int transpose_a, int transpose_b)
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
        for (size_t x = 0; x < m; x++) {
            for (size_t y = 0; y < k; y++) {
                c[x * k + y] = 0;
                for (size_t i = 0; i < n; i++) {
                    c[x * k + y] += a[x * n + i] * b[i * k + y];
                }
            }
        }
    } else if (transpose_a && !transpose_b) {
        // m = l
        // (m x n).T @ (l x k) = n x k
        // c_{xy} = sum_{i} a_{ix} * b_{iy}
        if (m != l) {
            return -1;
        }
        for (size_t x = 0; x < n; x++) {
            for (size_t y = 0; y < k; y++) {
                c[x * k + y] = 0;
                for (size_t i = 0; i < m; i++) {
                    c[x * k + y] += a[i * n + x] * b[i * k + y];
                }
            }
        }
    } else if (!transpose_a && transpose_b) {
        // n = k
        // (m x n) @ (l x k).T = m x l
        // c_{xy} = sum_{i} a_{xi} * b_{yi}
        if (n != k) {
            return -1;
        }
        for (size_t x = 0; x < m; x++) {
            for (size_t y = 0; y < l; y++) {
                c[x * l + y] = 0;
                for (size_t i = 0; i < n; i++) {
                    c[x * l + y] += a[x * n + i] * b[y * k + i];
                }
            }
        }
    } else {
        // m = k
        // (m x n).T @ (l x k).T = n x l
        // c_{xy} = sum_{i} a_{ix} * b_{yi}
        if (m != k) {
            return -1;
        }
        for (size_t x = 0; x < n; x++) {
            for (size_t y = 0; y < l; y++) {
                c[x * l + y] = 0;
                for (size_t i = 0; i < m; i++) {
                    c[x * l + y] += a[i * n + x] * b[y * k + i];
                }
            }
        }
    }
    return 0;
}

void sigmoid_array(double* a, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] = 1 / (1 + exp(-a[i]));
    }
}

void sigmoid_backward(double* grad, double* out, double* c, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        c[i] += grad[i] * out[i] * (1 - out[i]);
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