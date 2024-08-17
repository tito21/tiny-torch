from tinytorch import Tensor, fill

a = Tensor((1, 10), label="a")
fill(a, 2.0)
b = Tensor((1, 10), label="b")
fill(b, 3.0)

print(a.numpy())
print(b.numpy())

print(a.grad_numpy())
print(b.grad_numpy())

c = a * b
c.label = "c"
print(c.numpy())

print(a.grad_numpy())
print(b.grad_numpy())
c.backward()

print(a.grad_numpy())
print(b.grad_numpy())