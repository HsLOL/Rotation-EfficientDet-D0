# coding=utf-8
import torch
from torch import nn

# hyper parameters
in_dim = 1
n_hidden_1 = 1
n_hidden_2 = 1
out_dim = 1


class Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, out_dim)

        )
        # 迭代循环初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, -100)
            # 也可以判断是否为conv2d，使用相应的初始化方式
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.item(), 1)
                nn.init.constant_(m.bias.item(), 0)

    def forward(self, x):
        x = self.layer(x)

        return x


model = Net(in_dim, n_hidden_1, n_hidden_2, out_dim)


# 打印参数信息
def print_weight(m):
    if isinstance(m, nn.Linear):
        print("weight", m.weight.item())
        print("bias:", m.bias.item())
        print("next...")


model.apply(print_weight)
