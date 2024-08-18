import pytest

import numpy as np
import torch
import torch.nn.functional as F

from tinytorch import from_numpy


SHAPE = (4, 8)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_add(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a = a.cuda()
        b = b.cuda()

    out = a + b
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = a_torch + b_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.cpu().grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.cpu().grad_numpy(), b_torch.grad.numpy()))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sub(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a = a.cuda()
        b = b.cuda()

    out = a - b
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = a_torch - b_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.cpu().grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.cpu().grad_numpy(), b_torch.grad.numpy()))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mul(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a = a.cuda()
        b = b.cuda()

    out = a * b
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = a_torch * b_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.cpu().grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.cpu().grad_numpy(), b_torch.grad.numpy()))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neg(device):
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)

    if device == "cuda":
        a = a.cuda()

    out = -a
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True

    out_torch = -a_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.cpu().grad_numpy(), a_torch.grad.numpy()))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matmul(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a = a.cuda()
        b = b.cuda()

    out = a @ b
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = a_torch @ b_torch
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.cpu().grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.cpu().grad_numpy(), b_torch.grad.numpy()))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tanh(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a = a.cuda()
        b = b.cuda()

    out = (a @ b).tanh()
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = (a_torch @ b_torch).tanh()
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.cpu().grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.cpu().grad_numpy(), b_torch.grad.numpy()))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sigmoid(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a = a.cuda()
        b = b.cuda()

    out = (a @ b).sigmoid()
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    a_torch.requires_grad = True
    b_torch = torch.from_numpy(b_numpy)
    b_torch.requires_grad = True

    out_torch = (a_torch @ b_torch).sigmoid()
    out_torch.backward(gradient=torch.ones_like(out_torch))

    assert (np.allclose(a.cpu().grad_numpy(), a_torch.grad.numpy()) and np.allclose(b.cpu().grad_numpy(), b_torch.grad.numpy()))
