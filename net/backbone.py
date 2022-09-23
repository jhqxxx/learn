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

model = VGG(num_classes=10)
print(model)
