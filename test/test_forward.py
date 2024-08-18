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
        a.cuda()
        b.cuda()

    out = a + b

    a = torch.from_numpy(a_numpy)
    b = torch.from_numpy(b_numpy)

    out_torch = a + b
    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sub(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a.cuda()
        b.cuda()

    out = a - b

    a = torch.from_numpy(a_numpy)
    b = torch.from_numpy(b_numpy)

    out_torch = a - b
    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mul(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a.cuda()
        b.cuda()

    out = a * b

    a = torch.from_numpy(a_numpy)
    b = torch.from_numpy(b_numpy)

    out_torch = a * b
    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neg(device):
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)

    if device == "cuda":
        a.cuda()

    out = -a

    a = torch.from_numpy(a_numpy)

    out_torch = -a
    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matmul(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a.cuda()
        b.cuda()

    out = a @ b

    a = torch.from_numpy(a_numpy)
    b = torch.from_numpy(b_numpy)

    out_torch = a @ b
    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tanh(device):
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)

    if device == "cuda":
        a.cuda()

    out = a.tanh()

    a = torch.from_numpy(a_numpy)
    out_torch = a.tanh()

    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sigmoid(device):
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)

    if device == "cuda":
        a.cuda()

    out = a.sigmoid()

    a = torch.from_numpy(a_numpy)
    out_torch = a.sigmoid()

    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_relu(device):
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)

    if device == "cuda":
        a = a.cuda()

    out = a.relu()

    a = torch.from_numpy(a_numpy)
    out_torch = a.relu()

    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sum_reduce(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a.cuda()
        b.cuda()

    out = (a @ b).sum()
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    b_torch = torch.from_numpy(b_numpy)

    out_torch = (a_torch @ b_torch).sum()

    assert np.allclose(out.cpu().numpy(), out_torch.numpy())

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_power(device):
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))
    p = np.random.rand()

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)

    if device == "cuda":
        a.cuda()
        b.cuda()

    out = (a @ b) ** p
    out.backward()

    a_torch = torch.from_numpy(a_numpy)
    b_torch = torch.from_numpy(b_numpy)

    out_torch = (a_torch @ b_torch) ** p

    assert np.allclose(out.cpu().numpy(), out_torch.numpy())
