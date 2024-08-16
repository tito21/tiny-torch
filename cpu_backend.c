#include <stdlib.h>
#include <string.h>


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