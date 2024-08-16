from typing import Tuple, Callable

import numpy as np
cimport numpy as np
np.import_array()

cimport cpu_backend

def print_tensor(Tensor arr):
    for i in range(arr.size):
        print(arr.data[i], end=" ")
    print()

def print_grad(Tensor arr):
    for i in range(arr.size):
        print(arr.grad[i], end=" ")
    print()

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

    def __cinit__(self, shape: Tuple[int], device="cpu", label="", prev: Tuple[Tensor] = (), op=""):

        self.device = device
        self.size = np.prod(shape)
        self.shape = shape

        if device == "cpu":
            self.data = cpu_backend.new_array(self.size)
            self.grad = cpu_backend.new_array(self.size)
            self.zero_grad()
        elif device == "cuda":
            raise NotImplementedError("cuda is not implemented yet")
        else:
            raise ValueError("device can be 'cpu' or 'cuda' found ", device)

        self.label = label
        self._prev = prev
        self.op = op
        self._backward_func = lambda: None

    def __repr__(self):
        return f"Label: {self.label} shape {self.shape} op {self.op}"

    def __dealloc__(self):
        if self.device == "cpu":
            cpu_backend.free_array(self.data)
            cpu_backend.free_array(self.grad)
        elif self.device == "cuda":
            raise NotImplementedError("cuda is not implemented yet")

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

        cpu_backend.set_array(self.grad, 1.0, self.size)

        for n in reversed(topo):
            # print("tree grad", n.label, n.grad)
            n._backward_func()

    cpdef zero_grad(self):
        if self.device == "cpu":
            cpu_backend.set_array(self.data, 0.0,  self.size)
        elif self.device == "cuda":
            raise NotImplementedError("cuda is not implemented yet")

    def cpu(self):
        if self.device == "cpu":
            return self
        else:
            raise NotImplementedError("cuda is not implemented yet")

    def cuda(self):
        raise NotImplementedError("cuda is not implemented yet")


    def __add__(self, other: Tensor) -> Tensor:
        if self.device != other.device:
            raise RuntimeError(f"self and other are on diferent devices found {self.device} and {other.device}")

        cdef double* result = cpu_backend.new_array(self.size)
        cpu_backend.sum_arrays(self.data, other.data, result, self.size)
        # This results in an extra memory allocation
        out = Tensor(self.shape, device=self.device, prev=(self, other), op="+")
        cpu_backend.free_array(out.data)
        out.data = result
        print(out.label, out._prev)

        def _backward():
            cpu_backend.sum_arrays(self.grad, out.grad, self.grad, self.size)
            cpu_backend.sum_arrays(other.grad, out.grad, other.grad, self.size)

        out._backward_func = _backward

        return out

def fill(arr: Tensor, double value):
    """In place filling of an array"""
    cpu_backend.set_array(arr.data, value, arr.size)

