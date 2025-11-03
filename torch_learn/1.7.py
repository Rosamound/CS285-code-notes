# GPU support
import torch

print(torch.cuda.is_available())  # check there is GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y = torch.ones((2,3), device=device)
print(y)

# Note! you can not add something that is on GPU with that on CPU
# x = torch.ones(2,3)
# z = x+y
#------> RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

x = torch.ones(2,3).to(device)
z = x+y
print(z)

# Also can not convert someting on GPU to numpy array
# z.numpy()
#------>TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first

z_cpu = z.to('cpu')
print(z_cpu.numpy())

