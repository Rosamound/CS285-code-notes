# GPU to train and evaluate
# save and load

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from torch import optim
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    x = torch.linspace(-5, 5, 100).reshape(-1,1)
    y_target = torch.sin(x).to(device) # 这里一般会分batch计算 不会一次给GPU
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()
    losses = []

    for _ in range(300):
        x = x.to(device)

        y = net(x)
        loss = loss_fn(y, y_target)
        losses.append(loss.detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(300), losses, color='blue')  # 真实值
    plt.xlim(0,300)  # x轴范围
    # 保存图片为demo1.png
    plt.savefig('demo4.png', dpi=300, bbox_inches='tight')
    print("losses over iterations saved to demo4.png")

    # net.eval() 常与 with torch.no_grad(): 一起使用：
    net.eval() # 确保模型处于评估模式（关闭随机层）。
    with torch.no_grad(): # 临时禁用梯度计算，减少内存占用并加速推理（评估时不需要反向传播，无需计算梯度）。
        y = net(x)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label = "eval",color='blue')  
    plt.plot(x.cpu().numpy(), y_target.cpu().numpy(),label="real", color='red')
    plt.xlim(-5,5)  
    # 保存图片为demo1.png
    plt.legend()
    plt.title("evaluation curve VS real curve")
    plt.savefig('demo5.png', dpi=300, bbox_inches='tight')
    print("losses over iterations saved to demo5.png")

    PATH = "checkpoint.pt"
    torch.save(net.state_dict(), PATH)

    new_model = Net(input_size=1,output_size=1)
    new_model.to(device)
    new_model.load_state_dict(torch.load(PATH))
    new_model.eval()
    # print(net.state_dict().items())
    for (name1, val1), (name2, val2) in zip(net.state_dict().items(), new_model.state_dict().items()):
        assert name1 == name2 and torch.equal(val1, val2), f"{name1} and {name2} states differ!"