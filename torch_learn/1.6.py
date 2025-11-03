# 如何实现一个手动梯度更新的简单网络拟合sin函数 using  optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))  #better than relu because sin(x) is periodic
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = Net(input_size = 1, output_size=1)
    print(net)

    # x = torch.linspace(-5, 5, 100).view(100,1)
    x = torch.linspace(-5, 5, 100).reshape(-1,1)
    y_target = torch.sin(x)

    # use optimizer
    from torch import optim
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(1000):
        y = net(x)
        loss = loss_fn(y, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(figsize=(8, 5))
    plt.plot(x.detach().numpy(), y.detach().numpy(), label='fit_result', color='blue')  # 真实值
    plt.plot(x.detach().numpy(),y_target.numpy(), label='original curve', color="red")
    plt.xlim(-5, 5)  # x轴范围
    plt.ylim(-1.2, 1.2)  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # 保存图片为demo1.png
    plt.savefig('demo3.png', dpi=300, bbox_inches='tight')
    print("图片已保存为 demo3.png")
