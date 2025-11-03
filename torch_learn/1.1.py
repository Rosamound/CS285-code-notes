import torch

x = torch.Tensor([
    [4,3.5],
    [4, 4]
])

print(torch.softmax(x, dim = 0))