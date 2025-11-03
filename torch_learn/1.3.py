import torch

shape = (3,)

x = torch.tensor([1.,2,3], requires_grad = True) # must be [1., 2, 3] rather than [1,2,3]
y = torch.ones(shape, requires_grad = True)
y_detached = y.detach() # share the memory with y
loss = ((x *2 + y)**2).sum()
loss.backward()


# 错误！
# y.numpy()  
# 正确！
y_np = y.detach().numpy() # share memory with y
print(type(y_np))
print(type(y))
print(type(y_detached))

print(x.grad)
print(y.grad)
print(y_detached[0])

y_detached[0] = 2  # this is fine
# y[0] = 2  # this will throw error
print(y_detached[0])

new_y = y.detach().clone()
