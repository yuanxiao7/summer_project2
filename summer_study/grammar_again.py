import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = w + x
b = w + 1
y = a * b

y.backward()
print(w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
print(w.grad, x.grad, a.grad, b.grad, y.grad)
