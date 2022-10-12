from re import T
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# 定义一个卷积加一个batchnorm,一个relu作为基本的层结构
class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_c1, out_c2, out_c3, out_c4):
        super(InceptionBlock, self).__init__()
        # 线路1：1*1卷积
        self.p1 = BasicConv2d(in_channel, out_c1, 1)
        # 线路2：1*1卷积降维，3*3卷积提取信息并升维
        self.p2 = nn.Sequential(
            BasicConv2d(in_channel, out_c2[0], 1),
            BasicConv2d(out_c2[0], out_c2[1], kernel_size=3, padding=1)
        )
        # 线路3：1*1卷积降维，5*5卷积提取信息并升维
        self.p3 = nn.Sequential(
            BasicConv2d(in_channel, out_c3[0], 1),
            BasicConv2d(out_c3[0], out_c3[1], kernel_size=5, padding=2)
        )
        # 线路3：3*3最大池化层，1*1卷积升维
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channel, out_c4, kernel_size=1)
        )
    
    def forward(self, x):
        out1 = self.p1(x)
        out2 = self.p2(x)
        out3 = self.p3(x)
        out4 = self.p4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


class AuxLogits(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(AuxLogits, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)  # 论文中k=5, s=3
        self.conv = BasicConv2d(in_channel, 128, kernel_size=1)   # [batch, 128, 4, 4]
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc1(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


class InceptionV1(nn.Module):
    def __init__(self, num_classes, aux_logits=False, verbose=False, init_weights=True):
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.verbose = verbose
        self.init_weights = init_weights

        # block1
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # block2
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # block3
        self.inception3a = InceptionBlock(192, 64, [96, 128], [16, 32], 32)
        self.inception3b = InceptionBlock(256, 128, [128, 192], [32, 96], 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # block4
        self.inception4a = InceptionBlock(480, 192, [96, 208], [16, 48], 64)
        self.inception4b = InceptionBlock(512, 160, [112, 224], [24, 64], 64)
        self.inception4c = InceptionBlock(512, 128, [128, 256], [24, 64], 64)
        self.inception4d = InceptionBlock(512, 112, [144, 288], [32, 64], 64)
        self.inception4e = InceptionBlock(528, 256, [160, 320], [32, 128], 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # block5
        self.inception5a = InceptionBlock(832, 256, [160, 320], [32, 128], 128)
        self.inception5b = InceptionBlock(832, 384, [192, 384], [48, 128], 128)
        
        if self.aux_logits:
            self.aux1 = AuxLogits(512, num_classes)
            self.aux2 = AuxLogits(528, num_classes)
        
        self.avgpool = GlobalAvgPool()
        self.dropout = nn.Dropout2d(p=0.4)
        self.classifier = nn.Linear(1024, num_classes)
        if self.init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
            if self.verbose:
                print('aux 1 output: {}'.format(aux1.shape))
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
            if self.verbose:
                print('aux 2 output: {}'.format(aux2.shape))
        x = self.inception4e(x)
        x = self.maxpool4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.inception5a(x)
        x = self.inception5b(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        out = self.classifier(x)
        if self.training and self.aux_logits:
            return out, aux1, aux2
        return x
    
    def _initialize_weights(self):
        for m in self.modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


