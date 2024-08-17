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

    a = torch.from_numpy(a_numpy)
    b = torch.from_numpy(b_numpy)

    out_torch = a + b
    assert np.allclose(out.numpy(), out_torch.numpy())

def test_sub():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = a - b

    a = torch.from_numpy(a_numpy)
    b = torch.from_numpy(b_numpy)

    out_torch = a - b
    assert np.allclose(out.numpy(), out_torch.numpy())

def test_mul():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = a * b

    a = torch.from_numpy(a_numpy)
    b = torch.from_numpy(b_numpy)

    out_torch = a * b
    assert np.allclose(out.numpy(), out_torch.numpy())

def test_neg():
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    out = -a

    a = torch.from_numpy(a_numpy)

    out_torch = -a
    assert np.allclose(out.numpy(), out_torch.numpy())

def test_matmul():
    a_numpy = np.random.rand(*SHAPE)
    b_numpy = np.random.rand(*reversed(SHAPE))

    a = from_numpy(a_numpy)
    b = from_numpy(b_numpy)
    out = a @ b

    a = torch.from_numpy(a_numpy)
    b = torch.from_numpy(b_numpy)

    out_torch = a @ b
    assert np.allclose(out.numpy(), out_torch.numpy())

def test_tanh():
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    out = a.tanh()

    a = torch.from_numpy(a_numpy)
    out_torch = a.tanh()

    assert np.allclose(out.numpy(), out_torch.numpy())

def test_sigmoid():
    a_numpy = np.random.rand(*SHAPE)

    a = from_numpy(a_numpy)
    out = a.sigmoid()

    a = torch.from_numpy(a_numpy)
    out_torch = a.sigmoid()

    assert np.allclose(out.numpy(), out_torch.numpy())
