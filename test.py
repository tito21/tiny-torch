import numpy as np

from tinytorch import Tensor, fill, from_numpy

a_data = np.random.rand(*(1, 10))

a = from_numpy(a_data, label="a")
b = Tensor((1, 10), label="b")
fill(b, 3.0)

print(a.numpy())
print(b.numpy())

print(a.grad_numpy())
print(b.grad_numpy())

c = a + b
c.label = "c"
d = c.tanh()
print(d.numpy())

print(a.grad_numpy())
print(b.grad_numpy())
c.backward()

print(a.grad_numpy())
print(b.grad_numpy())