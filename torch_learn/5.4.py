# return a gaussian distribution
# 1. train on parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import optim

class GaussianPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super(GaussianPolicy, self).__init__()
        self.mean_fc1 = nn.Linear(input_size, 32)
        self.mean_fc2 = nn.Linear(32,32)
        self.mean_fc3 = nn.Linear(32, output_size)
        # define a learnable parameter, 
        # which is the log of the standard error of the gaussian distribution
        self.log_std = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        mean = F.relu(self.mean_fc1(x))
        mean = F.relu(self.mean_fc2(mean))
        mean = self.mean_fc3(mean)
        return mean, self.log_std

states = torch.randn(100,2) -0.5
true_means = states**3 + 4.5*states
true_cov = torch.diag(torch.Tensor([0.1, 0.05]))
expert_actions = torch.distributions.MultivariateNormal(true_means, true_cov).sample()

plt.figure(figsize=(8, 5))
plt.scatter(expert_actions[:,0],expert_actions[:,1])
plt.title("Expert Actions")
plt.savefig('demo6.png', dpi=300, bbox_inches='tight')
print("losses over iterations saved to demo6.png")

from torch.utils.data import DataLoader, TensorDataset

policy = GaussianPolicy(2,2)
optimizer = optim.Adam(policy.parameters(), lr = 0.01)

dataset = TensorDataset(states, expert_actions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

losses=[]
for epoch in range(200):
    epoch_loss = 0
    for curr_states, curr_actions in loader:
        mean, log_std = policy(curr_states)
        dist = torch.distributions.Normal(mean, torch.exp(log_std))
        loss = -dist.log_prob(curr_actions).sum()

        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.detach().cpu().numpy().squeeze()
        optimizer.step()

    losses.append(epoch_loss/len(loader))

plt.figure(figsize=(8, 5))
plt.plot(losses)  
# 保存图片为demo1.png
# plt.legend()
plt.title("losses")
plt.savefig('demo7.png', dpi=300, bbox_inches='tight')
print("losses over iterations saved to demo7.png")


policy.eval()
with torch.no_grad():
    mean, log_std = policy(states)
    dist = torch.distributions.Normal(mean, torch.exp(log_std))
    pred_means = dist.mean.cpu().numpy()
    pred_actions = dist.sample().cpu().numpy()

plt.figure()
plt.title("Sampled actions")
plt.scatter(pred_actions[:,0], pred_actions[:,1], color='r', label='learned policy')
plt.scatter(expert_actions[:,0], expert_actions[:,1], color='b', label='expert')
plt.legend()
plt.savefig('demo8.png', dpi=300, bbox_inches='tight')

plt.figure()
plt.title("Action means")
plt.scatter(pred_means[:,0], pred_means[:,1], color='r', label='learned policy')
plt.scatter(true_means[:,0], true_means[:,1], color='b', label='expert')
plt.legend()
plt.savefig('demo9.png', dpi=300, bbox_inches='tight')
