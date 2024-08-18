
double* cuda_new_array(size_t size);
void cuda_free_array(double* arr);
void cuda_set_array(double* arr, double value, size_t size);
void cuda_copy_array(double* dest, double* src, size_t size);

void cuda_sum_arrays(double* a, double* b, double* c, size_t size);
void cuda_sum_accu_arrays(double* a, double* b, double* c, size_t size);
void cuda_sum_accu_scalar(double* a, double b, double* c, size_t size);

void cuda_sub_arrays(double* a, double* b, double* c, size_t size);
void cuda_sub_accu_arrays(double* a, double* b, double* c, size_t size);

void cuda_mul_arrays(double* a, double* b, double* c, size_t size);
void cuda_mul_accu_arrays(double* a, double* b, double* c, size_t size);
void cuda_mul_accu_scalar(double* a, double b, double* c, size_t size);

void cuda_div_arrays(double* a, double* b, double* c, size_t size);
void cuda_div_accu_arrays(double* a, double* b, double* c, size_t size);

int cuda_matmul(double* a, double* b, double* c, size_t m, size_t n, size_t l, size_t k, int transpose_a, int transpose_b);

void cuda_tanh_array(double* a, double* c, size_t size);
void cuda_tanh_backward(double* grad, double* out, double* c, size_t size);

void cuda_sigmoid_array(double* a, double* c, size_t size);
void cuda_sigmoid_backward(double* grad, double* out, double* c, size_t size);