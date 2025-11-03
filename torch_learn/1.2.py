import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  #必须要等于True才可计算图
y = x ** 2
loss = y.sum()  # loss = 2+4+6=12

loss.backward()  # 反向传播
print(x.grad)    # 输出梯度