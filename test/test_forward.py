import unittest

import numpy as np
import torch
import torch.nn.functional as F

from tinytorch import from_numpy

SHAPE = (4, 8)

class TestOp(unittest.TestCase):

    def test_add(self):
        a_numpy = np.random.rand(*SHAPE)
        b_numpy = np.random.rand(*SHAPE)

        a = from_numpy(a_numpy)
        b = from_numpy(b_numpy)

        out = a + b

        a = torch.from_numpy(a_numpy)
        b = torch.from_numpy(b_numpy)

        out_torch = a + b
        self.assertTrue(np.allclose(out.numpy(), out_torch.numpy()))

    def test_sub(self):
        a_numpy = np.random.rand(*SHAPE)
        b_numpy = np.random.rand(*SHAPE)

        a = from_numpy(a_numpy)
        b = from_numpy(b_numpy)
        out = a - b

        a = torch.from_numpy(a_numpy)
        b = torch.from_numpy(b_numpy)

        out_torch = a - b
        self.assertTrue(np.allclose(out.numpy(), out_torch.numpy()))

    def test_mul(self):
        a_numpy = np.random.rand(*SHAPE)
        b_numpy = np.random.rand(*SHAPE)

        a = from_numpy(a_numpy)
        b = from_numpy(b_numpy)
        out = a * b

        a = torch.from_numpy(a_numpy)
        b = torch.from_numpy(b_numpy)

        out_torch = a * b
        self.assertTrue(np.allclose(out.numpy(), out_torch.numpy()))

    def test_tanh(self):
        a_numpy = np.random.rand(*SHAPE)


        a = from_numpy(a_numpy)
        out = a.tanh()

        a = torch.from_numpy(a_numpy)
        out_torch = a.tanh()

        self.assertTrue(np.allclose(out.numpy(), out_torch.numpy()))

if __name__ == '__main__':
    unittest.main()