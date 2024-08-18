import numpy as np
import torch

# import backends
# import cpu_backend
# import engine

# import tinytorch
# import cuda_backend
from tinytorch import Tensor, fill, from_numpy

a_data = np.random.rand(3, 3)
a = from_numpy(a_data, label="a")
# a = Tensor((3, 3), label="a", device="cuda")
# fill(a, 3.0)
# print(a.numpy())
a = a.cuda()
print(a.cpu().numpy())
print("f")
a_torch = torch.from_numpy(a.cpu().numpy())
a_torch.requires_grad = True
# print("pytorch", a_torch)
# b_data = np.random.rand(3, 2)
# b_data = np.random.rand(2, 3)
# b = from_numpy(b_data, label="b")
b = Tensor((3, 3), label="b", device="cuda")
# fill(b, 3.0)
print(b.cpu().numpy())
b_torch = torch.from_numpy(b.cpu().cpu().cuda().cpu().numpy())
b_torch.requires_grad = True

# print(a.numpy() @ b.numpy())

c = a * b
c.label = "c"
# c = c.cpu()
print(c.cpu().numpy())
c.backward()

print(a.cpu().grad_numpy())
print(b.cpu().grad_numpy())

# self.grad += out.grad @ other.data.T
# other.grad += self.data.T @ out.grad

# a_grad = c.grad_numpy() @ b.numpy().T
# b_grad = a.numpy().T @ c.grad_numpy()

a_grad = c.cpu().grad_numpy() * b.cpu().numpy()
b_grad = a.cpu().numpy() * c.cpu().grad_numpy()


print(a_grad)
print(b_grad)

c_torch = a_torch * b_torch
c_torch.backward(gradient=torch.ones_like(c_torch))

print(np.allclose(a.cpu().grad_numpy(), a_grad))
print(np.allclose(b.cpu().grad_numpy(), b_grad))


print(np.allclose(a_torch.grad.numpy(), a_grad))
print(np.allclose(b_torch.grad.numpy(), b_grad))