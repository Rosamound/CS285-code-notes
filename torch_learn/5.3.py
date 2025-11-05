# 模型返回分布
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return distributions.Normal(x, scale=1)

distribution_network = Net(1, 1)
x = torch.randn(100, 1)
distribution = distribution_network(x)
print(distribution)