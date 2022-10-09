from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Inception(nn.Module):
    def __init__(self, in_channel, out_c1, out_c2, out_c3, out_c4):
        super(Inception, self).__init__()
        # 线路1：1*1卷积
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channel, out_c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # 线路2：1*1卷积降维，3*3卷积提取信息并升维
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channel, out_c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_c2[0], out_channels=out_c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 线路3：1*1卷积降维，5*5卷积提取信息并升维
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channel, out_c3[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c3[0], out_c3[1], kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # 线路3：3*3最大池化层，1*1卷积升维
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, out_c4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out1 = self.p1(x)
        out2 = self.p2(x)
        out3 = self.p3(x)
        out4 = self.p4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


