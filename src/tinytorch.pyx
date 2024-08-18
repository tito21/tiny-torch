from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "numpy/arrayobject.h":
    ctypedef void PyArrayObject
    PyArrayObject* PyArray_SimpleNewFromData(int nd, np.npy_intp* dims, int typenum, void* data)
    object PyArray_Return(PyArrayObject* arr)

cimport memutils
from cpu_backend_c cimport *
from cuda_backend_cu cimport *

cdef np.ndarray to_numpy(double* data, tuple shape):
    # cdef Py_ssize_t ndim = len(shape)
    # cdef np.npy_intp* dims = <np.npy_intp*>malloc(ndim * sizeof(np.npy_intp))
    # if not dims:
    #     raise MemoryError("Unable to allocate memory for dimensions")
    # cdef int i
    # cdef object element
    # for i in range(ndim):
    #     dims[i] = <np.npy_intp>shape[i]
    # cdef int typenum = np.NPY_FLOAT64
    # cdef PyArrayObject* np_array = PyArray_SimpleNewFromData(ndim, dims, typenum, <void*>data)
    # result = PyArray_Return(np_array)
    # free(dims)
    return np.array([data[i] for i in range(np.prod(shape))], dtype=np.float64).reshape(shape)

cdef class Tensor:

    cdef double* data
    cdef double* grad
    cdef int size

    cdef readonly str device
    cdef public str label
    cdef readonly str op
    cdef readonly tuple shape

    cdef public tuple _prev
    cdef public _backward_func

    def __cinit__(self, tuple shape, device="cpu", label="", tuple prev = (), op=""):

        self.device = device
        self.size = np.prod(shape)
        self.shape = shape

        if device not in ["cpu", "cuda"]:
            raise ValueError("device can be 'cpu' or 'cuda' found ", device)

        if device == "cpu":
            self.data = cpu_new_array(self.size)
            self.grad = cpu_new_array(self.size)
        else:
            self.data = cuda_new_array(self.size)
            self.grad = cuda_new_array(self.size)
        self.zero_grad()

        self.label = label
        self._prev = prev
        self.op = op
        self._backward_func = lambda: None

    def __repr__(self):
        return f"Label: {self.label} shape {self.shape} op {self.op}"

    def __dealloc__(self):
        if self.device == "cpu":
            cpu_free_array(self.data)
            cpu_free_array(self.grad)
        else:
            cuda_free_array(self.data)
            cuda_free_array(self.grad)

    def backward(self):
        """The calculation gradient will always start with np.ones_like()"""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build_topo(c)
                topo.append(v)
        build_topo(self)

        if self.device == "cpu":
            cpu_set_array(self.grad, 1.0, self.size)
        else:
            cuda_set_array(self.grad, 1.0, self.size)

        for n in reversed(topo):
            n._backward_func()

    cpdef zero_grad(self):
        if self.device == "cpu":
            cpu_set_array(self.grad, 0.0,  self.size)
        else:
            cuda_set_array(self.grad, 0.0,  self.size)

    def cpu(self):
        if self.device == "cpu":
            return self
        else:
            out = Tensor(self.shape, device="cpu", label=self.label, prev=self._prev, op=self.op)
            out._backward_func = self._backward_func
            memutils.memutils_to_host(out.data, self.data, sizeof(double) * self.size)
            memutils.memutils_to_host(out.grad, self.grad, sizeof(double) * self.size)
            return out

    def cuda(self):
        if self.device == "cuda":
            return self
        else:
            out = Tensor(self.shape, device="cuda", label=self.label, prev=self._prev, op=self.op)
            out._backward_func = self._backward_func
            memutils.memutils_to_device(out.data, self.data, sizeof(double) * self.size)
            memutils.memutils_to_device(out.grad, self.grad, sizeof(double) * self.size)
            return out

    def numpy(self):
        if self.device == "cuda":
            raise RuntimeError("Copy to cpu first")
        return to_numpy(self.data, self.shape)

    def grad_numpy(self):
        if self.device == "cuda":
            raise RuntimeError("Copy to cpu first")
        return to_numpy(self.grad, self.shape)

    def __add__(self, other: Tensor) -> Tensor:
        if self.device != other.device:
            raise RuntimeError(f"self and other are on diferent devices found {self.device} and {other.device}")

        out = Tensor(self.shape, device=self.device, prev=(self, other), op="+")
        if self.device == "cpu":
            cpu_sum_arrays(self.data, other.data, out.data, self.size)
        else:
            cuda_sum_arrays(self.data, other.data, out.data, self.size)

        def _backward():
            # self.grad += out.grad
            if self.device == "cpu":
                cpu_sum_arrays(self.grad, out.grad, self.grad, self.size)
            else:
                cuda_sum_arrays(self.grad, out.grad, self.grad, self.size)
            # other.grad += out.grad
            if self.device == "cpu":
                cpu_sum_arrays(other.grad, out.grad, other.grad, self.size)
            else:
                cuda_sum_arrays(other.grad, out.grad, other.grad, self.size)

        out._backward_func = _backward

        return out

    def __radd__(self, other: Tensor) -> Tensor:
        return self + other

    def __mul__(self, other: Tensor) -> Tensor:
        if self.device != other.device:
            raise RuntimeError(f"self and other are on diferent devices found {self.device} and {other.device}")

        out = Tensor(self.shape, device=self.device, prev=(self, other), op="*")
        if self.device == "cpu":
            cpu_mul_arrays(self.data, other.data, out.data, self.size)
        else:
            cuda_mul_arrays(self.data, other.data, out.data, self.size)

        def _backward():
            # self.grad += out.grad * other.data
            if self.device == "cpu":
                cpu_mul_accu_arrays(out.grad, other.data, self.grad, self.size)
            else:
                cuda_mul_accu_arrays(out.grad, other.data, self.grad, self.size)

            # other.grad += out.grad * self.data
            if self.device == "cpu":
                cpu_mul_accu_arrays(out.grad, self.data, other.grad, self.size)
            else:
                cuda_mul_accu_arrays(out.grad, self.data, other.grad, self.size)

        out._backward_func = _backward

        return out

    def __rmul__(self, other: Tensor) -> Tensor:
        return self * other

    def __sub__(self, other: Tensor) -> Tensor:
        if self.device != other.device:
            raise RuntimeError(f"self and other are on diferent devices found {self.device} and {other.device}")

        out = Tensor(self.shape, device=self.device, prev=(self, other), op="-")
        if self.device == "cpu":
            cpu_sub_arrays(self.data, other.data, out.data, self.size)
        else:
            cuda_sub_arrays(self.data, other.data, out.data, self.size)

        def _backward():
            # self.grad += out.grad
            if self.device == "cpu":
                cpu_sum_arrays(self.grad, out.grad, self.grad, self.size)
            else:
                cuda_sum_arrays(self.grad, out.grad, self.grad, self.size)
            # other.grad -= out.grad
            if self.device == "cpu":
                cpu_sub_accu_arrays(other.grad, out.grad, other.grad, self.size)
            else:
                cuda_sub_accu_arrays(other.grad, out.grad, other.grad, self.size)

        out._backward_func = _backward

        return out

    def __rsub__(self, other: Tensor) -> Tensor:
        return other - self

    def __neg__(self) -> Tensor:
        out = Tensor(self.shape, device=self.device, prev=(self,), op="-1")
        if self.device == "cpu":
            cpu_set_array(out.data, -1.0, self.size)
            cpu_mul_arrays(self.data, out.data, out.data, self.size)
        else:
            cuda_set_array(out.data, -1.0, self.size)
            cuda_mul_arrays(self.data, out.data, out.data, self.size)

        def _backward():
            # self.grad += out.grad * -1.0
            if self.device == "cpu":
                cpu_mul_accu_scalar(out.grad, -1.0, self.grad, self.size)
            else:
                cuda_mul_accu_scalar(out.grad, -1.0, self.grad, self.size)

        out._backward_func = _backward

        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        if self.device != other.device:
            raise RuntimeError(f"self and other are on diferent devices found {self.device} and {other.device}")

        assert self.shape[1] == other.shape[0], "Incompatible shapes for matmul"

        out_shape = (self.shape[0], other.shape[1])
        out = Tensor(out_shape, device=self.device, prev=(self, other), op="@")
        if self.device == "cpu":
            cpu_matmul(self.data, other.data, out.data, self.shape[0], self.shape[1], other.shape[0], other.shape[1], False, False)
        else:
            cuda_matmul(self.data, other.data, out.data, self.shape[0], self.shape[1], other.shape[0], other.shape[1], False, False)

        def _backward():
            # self.grad += out.grad @ other.data.T
            # (m x n) @ (l x k).T = m x l
            if self.device == "cpu":
                result = cpu_new_array(self.shape[0] * other.shape[0])
                cpu_matmul(out.grad, other.data, result, out_shape[0], out_shape[1], other.shape[0], other.shape[1], transpose_a=False, transpose_b=True)
                cpu_sum_arrays(self.grad, result, self.grad, self.size)
                cpu_free_array(result)
            else:
                result = cuda_new_array(self.shape[0] * other.shape[0])
                cuda_matmul(out.grad, other.data, result, out_shape[0], out_shape[1], other.shape[0], other.shape[1], transpose_a=False, transpose_b=True)
                cuda_sum_arrays(self.grad, result, self.grad, self.size)
                cuda_free_array(result)
            # other.grad += self.data.T @ out.grad
            # (m x n).T @ (l x k) = n x k
            if self.device == "cpu":
                result = cpu_new_array(self.shape[1] * other.shape[1])
                cpu_matmul(self.data, out.grad, result, self.shape[0], self.shape[1], out_shape[0], out_shape[1], transpose_a=True, transpose_b=False)
                cpu_sum_arrays(other.grad, result, other.grad, other.size)
                cpu_free_array(result)
            else:
                result = cuda_new_array(other.shape[0] * other.shape[1])
                cuda_matmul(self.data, out.grad, result, self.shape[0], self.shape[1], out_shape[0], out_shape[1], transpose_a=True, transpose_b=False)
                cuda_sum_arrays(other.grad, result, other.grad, other.size)
                cuda_free_array(result)

        out._backward_func = _backward

        return out

    def tanh(self) -> Tensor:

        out = Tensor(self.shape, device=self.device, prev=(self,), op="tanh")
        if self.device == "cpu":
            cpu_tanh_array(self.data, out.data, self.size)
        else:
            cuda_tanh_array(self.data, out.data, self.size)

        def _backward():
            # self.grad += out.grad * (1 - out.data * out.data)
            if self.device == "cpu":
                cpu_tanh_backward(out.grad, out.data, self.grad, self.size)
            else:
                cuda_tanh_backward(out.grad, out.data, self.grad, self.size)

        out._backward_func = _backward

        return out

    def sigmoid(self):

        out = Tensor(self.shape, device=self.device, prev=(self,), op="sigmoid")
        if self.device == "cpu":
            cpu_sigmoid_array(self.data, out.data, self.size)
        else:
            cuda_sigmoid_array(self.data, out.data, self.size)

        def _backward():
            # self.grad += out.grad * out.data * (1 - out.data)
            if self.device == "cpu":
                cpu_sigmoid_backward(out.grad, out.data, self.grad, self.size)
            else:
                cuda_sigmoid_backward(out.grad, out.data, self.grad, self.size)

        out._backward_func = _backward

        return out

def fill(arr: Tensor, double value):
    """In place filling of an array"""
    if arr.device == "cpu":
        cpu_set_array(arr.data, value, arr.size)
    else:
        cuda_set_array(arr.data, value, arr.size)

def from_numpy(np.ndarray arr, str label="") -> Tensor:
    if arr.dtype != np.float64:
        raise ValueError("Only float64 is supported")

    cdef double* data = <double*>arr.data
    shape = tuple(arr.shape[i] for i in range(arr.ndim))
    cdef Tensor out = Tensor(shape, label=label)
    cpu_copy_array(out.data, data, out.size)
    return out