import torch
from torch.distributions import Categorical

# 方式1：通过概率probs初始化
probs = torch.tensor([0.2, 0.3, 0.5])  # 3个类别，概率和为1
cat_dist = Categorical(probs=probs)

# 方式2：通过对数概率logits初始化（更常用，避免概率计算的数值问题）
logits = torch.tensor([1.0, 2.0, 3.0])
cat_dist = Categorical(logits=logits)

sample = cat_dist.sample()  # 采样一个类别
samples = cat_dist.sample((5,))  # 采样5个类别，形状为(5,)

# 假设 cat_dist 是一个描述多个类别的概率分布（例如有 3 个类别，概率分别为 p 0 ,p 1​ ,p 2），
# log_prob(2) 就是对第 2 个类别的概率 p 2  取对数，即 log(p 2 )
log_p = cat_dist.log_prob(torch.tensor(2))  # 计算类别2的对数概率

# 熵越大，分布的不确定性越高（例如 “均匀分布” 的熵最高，因为每个类别被选中的概率相同，最难预测）。
# 熵越小，分布越 “确定”（例如某个类别概率接近 1，其他接近 0 时，熵接近 0，结果几乎可预测）。
entropy = cat_dist.entropy()
print("log_p:", log_p)
print("entropy:", entropy)

