'''
Descripttion: 
version: 
Author: jhq
Date: 2022-09-25 14:16:51
LastEditors: jhq
LastEditTime: 2022-09-26 23:21:22
'''
from re import L
from turtle import forward
from xml.etree.ElementInclude import include
import torch
import torch.nn as nn

def vgg_block(nums, in_channel, out_channel, kernel, padding):
    # 每个网络块的第一层输入channel和输出channel不同，所以需要单独写
    net = [nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, padding=padding), 
            nn.ReLU(inplace=True)]
    for i in range(nums-1):
        net.append(nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel, padding=padding))
        net.append(nn.ReLU(inplace=True))

    net.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*net)

def vgg_stack(nums_layer, channels):
    net = []
    for num, c in zip(nums_layer, channels):
        c_in = c[0]
        c_out = c[1]
        net.append(vgg_block(num, c_in, c_out, 3, 1))
    return nn.Sequential(*net)


class VGG(nn.Module):
    def __init__(self, type=16, num_classes=1000):
        super(VGG, self).__init__()
        if type == 16:
            self.nums_layer = [2, 2, 3, 3, 3]
            self.channels = [[3, 64], [64, 128], [128, 256], [256, 512], [512, 512]]            
        elif type == 19:
            self.nums_layer = [2, 2, 4, 4, 4]
            self.channels = [[3, 64], [64, 128], [128, 256], [256, 512], [512, 512]]
        self.feature = vgg_stack(self.nums_layer, self.channels)
        self.fc = nn.Sequential(nn.Linear(512*7*7, 4096),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(4096, 4096),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(4096, num_classes),
                                nn.Softmax(dim=1))
    
    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def conv3x3(in_channels, out_channels, stride=1, padding=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, 
                    stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, padding=1, downsample=None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels=in_channel, out_channels=out_channel,
                        stride=stride,padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels=out_channel, out_channels=out_channel,
                        stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, mid_channel, stride=1, padding=1, downsample=None) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels=in_channel, out_channels=mid_channel)    
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels=mid_channel, out_channels=mid_channel, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = conv1x1(in_channels=mid_channel, out_channels=mid_channel*self.expansion)
        self.bn3 = nn.BatchNorm2d(mid_channel*self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += identity
        x = self.relu(x)
        return x
        

class ResNet(nn.Module):
    def __init__(self, block, block_list, num_classes, include_top=True) -> None:
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, 
                                stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block=block, channel=64, block_list=block_list[0])
        self.layer2 = self.make_layer(block=block, channel=128, block_list=block_list[1], stride=2)
        self.layer3 = self.make_layer(block=block, channel=258, block_list=block_list[2], stride=2)
        self.layer4 = self.make_layer(block=block, channel=512, block_list=block_list[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def make_layer(self, block, channel, block_list, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel*block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, channel, stride=stride, downsample=downsample))
        self.in_channels = channel * block.expansion
        for _ in range(1, block_list):
            layers.append(block(self.in_channels, channel))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        
        return x


def get_resnet(layer_num, num_classes=1000, include_top=True):
    if layer_num == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)
    elif layer_num == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    elif layer_num == 50:
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    elif layer_num == 101:
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
    elif layer_num == 152:
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


class BottleneckX(nn.Module):
    expansion = 2
    def __init__(self, in_channel, mid_channel, stride=1, padding=1, downsample=None, group=1):
        super(BottleneckX, self).__init__()
        self.conv1 = conv1x1(in_channels=in_channel, out_channels=mid_channel)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels=mid_channel, out_channels=mid_channel, stride=stride, padding=padding, groups=group)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = conv1x1(in_channels=mid_channel, out_channels=mid_channel*self.expansion)
        self.bn3 = nn.BatchNorm2d(mid_channel*self.expansion)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x



# model = VGG(num_classes=10)
model = get_resnet(50, 10)
print(model)
