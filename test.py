from tinytorch import Tensor, fill, print_tensor, print_grad

a = Tensor((10, 1), label="a")
print_tensor(a)
fill(a, 2.0)
print_tensor(a)
b = Tensor((10, 1), label="b")
fill(b, 2.0)
print_tensor(b)

c = a + b
c.label = "c"
print(c)
print_tensor(c)

c.backward()

print_grad(a)
print_grad(b)