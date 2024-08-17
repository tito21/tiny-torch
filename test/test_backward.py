import unittest

import numpy as np
import torch
import torch.nn.functional as F

from tinytorch import from_numpy


SHAPE = (4, 8)

def test_add():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = a + b
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = a_torch + b_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.grad_numpy(), b_torch.grad.numpy()))

def test_sub():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = a - b
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = a_torch - b_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.grad_numpy(), b_torch.grad.numpy()))

def test_mul():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = a * b
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = a_torch * b_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.grad_numpy(), b_torch.grad.numpy()))

def test_neg():
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    out = -a
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True

    out_torch = -a_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.grad_numpy(), a_torch.grad.numpy()))

def test_matmul():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = a @ b
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = a_torch @ b_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.grad_numpy(), b_torch.grad.numpy()))

def test_tanh():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = (a @ b).tanh()
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = (a_torch @ b_torch).tanh()
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.grad_numpy(), b_torch.grad.numpy()))

def test_sigmoid():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = (a @ b).sigmoid()
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = (a_torch @ b_torch).sigmoid()
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.grad_numpy(), b_torch.grad.numpy()))
