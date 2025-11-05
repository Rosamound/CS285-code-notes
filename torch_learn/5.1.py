import torch
from torch.distributions import MultivariateNormal

# 定义均值向量（二维）
mean = torch.zeros(2)  # 形状为 (2,)

# 对角线元素：变量的方差协方差矩阵对角线上的元素（第 i 行第 i 列）
# 表示第 i 个变量的方差（variance），方差越大，说明该变量的取值越分散（离散程度越高）。

# 例如，协方差矩阵 [[2, 0], [0, 3]] 中，第一个变量的方差为 2，
# 第二个变量的方差为 3，意味着第二个变量的取值更分散。

# 非对角线元素：变量间的相关性协方差矩阵非对角线上的元素（第 i 行第 j 列，i≠j）表示第 i 个变量和第 j 个变量的协方差（covariance），用于衡量两个变量的线性相关程度：
# 协方差 > 0：两个变量正相关（一个增大时，另一个倾向于增大）；
# 协方差 < 0：两个变量负相关（一个增大时，另一个倾向于减小）；
# 协方差 = 0：两个变量线性无关（但可能有非线性关系）。
# 例如，协方差矩阵 [[1, 0.8], [0.8, 2]] 中，0.8 表示两个变量呈较强的正相关。

# 定义协方差矩阵集合（3个二维协方差矩阵，形状为 (3, 2, 2)）
covariance = torch.tensor([
    [[1.0, 0.8], [0.8, 2.0]],    # 第一个协方差矩阵
    [[1.0, -0.2], [-0.2, 1.0]],  # 第二个协方差矩阵
    [[4.0, 0.6], [0.6, 0.5]]     # 第三个协方差矩阵
])

# 创建多元正态分布对象
gaussian = MultivariateNormal(loc=mean, covariance_matrix=covariance)

# 从分布中抽取5个样本（每个批次对应一个协方差矩阵）
sample = gaussian.sample((5,))  # 采样形状为 (5, 3, 2)，表示5个样本×3个批次×2个维度

# 打印分布属性和样本信息
print("分布的批次形状（batch_shape）:", gaussian.batch_shape)  # 输出：torch.Size([3])
print("分布的事件形状（event_shape）:", gaussian.event_shape)  # 输出：torch.Size([2])
print("样本的第一个维度大小:", sample.shape[0])  # 输出：5（即采样数量）
print("样本的完整形状:", sample.shape)  # 输出：torch.Size([5, 3, 2])
print(sample[0])