'''
Author: jhq
Date: 2022-11-27 20:48:17
LastEditTime: 2022-12-04 23:01:15
Description: repvgg最重要模块，可重参数化
'''
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

def confused_conv3x3_bn():
    torch.random.manual_seed(0)
    input = torch.randn(1, 2, 3, 3)

    module = nn.Sequential(OrderedDict(
        conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False),
        bn = nn.BatchNorm2d(num_features=2)
    ))

    module.eval()

    with torch.no_grad():
        output1 = module(input)
        print(f'output1:{output1}')
    
    # confused conv + bn
    kernel = module.conv.weight
    running_mean = module.bn.running_mean
    running_var = module.bn.running_var
    gamma = module.bn.weight
    beta = module.bn.bias 
    eps = module.bn.eps 
    std = (running_var+eps).sqrt()
    t = (gamma/std).reshape(-1, 1, 1, 1) # [ch] -> [ch, 1, 1, 1]
    print(f't_shape:{t.shape}, kernel_shape:{kernel.shape}')
    print(kernel)
    ker = t * kernel
    print(t)
    print(ker)
    bias = beta - running_mean*gamma/std
    confused_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
    confused_conv.load_state_dict(OrderedDict(weight=ker, bias=bias))
    
    with torch.no_grad():
        output2 = confused_conv(input)
        print(f'output2:{output2}')
    
    print(torch.allclose(output1, output2))

def confused_conv1x1_bn():
    torch.random.manual_seed(0)
    input = torch.randn(1, 2, 3, 3)
    module = nn.Sequential(OrderedDict(
        conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, bias=False),
        bn = nn.BatchNorm2d(num_features=2)
    ))

    module.eval()
    with torch.no_grad():
        output1 = module(input)
        print(f'output1:{output1}\n,shape:{output1.shape}')
    kernel = module.conv.weight
    running_mean = module.bn.running_mean
    running_var = module.bn.running_var
    gamma = module.bn.weight
    beta = module.bn.bias 
    eps = module.bn.eps 
    std = (running_var + eps).sqrt()
    t =  (gamma / std).reshape(-1, 1, 1, 1)
    kernel = kernel * t
    bias = beta - running_mean * gamma / std
    fused_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, bias=True)
    fused_conv.load_state_dict(OrderedDict(weight=kernel, bias=bias))
    with torch.no_grad():
        output2 = fused_conv(input)
        print(f'output2:{output2}\n,shape:{output2.shape}')

def convert_1x1_3x3():
    torch.random.manual_seed(0)
    input = torch.randn(1, 2, 3, 3)
    conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, bias=True)
    with torch.no_grad():
        output1 = conv1(input)
        print(f'output1:{output1}\n,shape:{output1.shape}')
    
    kernel = conv1.weight
    bias = conv1.bias
    # print(f'kernel:{kernel},shape:{kernel.shape}\n, bias:{bias}, shape:{bias.shape}')
    kernel3 = torch.zeros(2, 2, 3, 3)
    kernel3[0, 0, 1, 1] = kernel[0, 0, 0, 0]
    kernel3[0, 1, 1, 1] = kernel[0, 1, 0, 0]
    kernel3[1, 0, 1, 1] = kernel[1, 0, 0, 0]
    kernel3[1, 1, 1, 1] = kernel[1, 1, 0, 0]

    conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
    conv2.load_state_dict(OrderedDict(weight=kernel3, bias=bias))
    with torch.no_grad():
        output2 = conv2(input)
        print(f'output2:{output2}\n,shape:{output2.shape}')
    
    print(torch.allclose(output1, output2))
    
def convert_bn_conv3():
    torch.random.manual_seed(0)
    input = torch.randn(1, 2, 3, 3)
    print(f'input:{input}\n')
    bn = nn.BatchNorm2d(num_features=2)
    bn.eval()
    with torch.no_grad():
        output1 = bn(input)
        print(f'output1:{output1}\n,shape:{output1.shape}')
    
    kernel = torch.zeros(2, 2, 3, 3)
    kernel[0, 0, 1, 1] = 1
    kernel[1, 1, 1, 1] = 1
    # print(kernel)
    bn_conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
    bn_conv3.load_state_dict(OrderedDict(weight=kernel))
    input_conv = bn_conv3(input)
    print(f'input_conv:{input_conv}\n')
    print(torch.allclose(input, input_conv))

    kernel = bn_conv3.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias 
    eps = bn.eps 
    std = (running_var+eps).sqrt()
    t = (gamma/std).reshape(-1, 1, 1, 1) # [ch] -> [ch, 1, 1, 1]
    print(f't_shape:{t.shape}, kernel_shape:{kernel.shape}')
    print(kernel)
    ker = t * kernel
    print(t)
    print(ker)
    bias = beta - running_mean*gamma/std
    confused_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
    confused_conv.load_state_dict(OrderedDict(weight=ker, bias=bias))

    with torch.no_grad():
        output2 = confused_conv(input)
        print(f'output2:{output2}\n,shape:{output2.shape}')
    
    print(torch.allclose(output1, output2))
    # running_mean = bn.running_mean
    # running_var = bn.running_var
    # gamma = bn.weight
    # beta = bn.bias
    # eps = bn.eps
    # std = (running_var + eps).sqrt()
    
    # bias = beta - running_mean * gamma / std
    
    # t = (gamma / std).reshape(-1, 1, 1, 1)    
    # print(t)
    # kernel = kernel * t
    # print(kernel)
    # bn_conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
    # bn_conv3.load_state_dict(OrderedDict(weight=kernel, bias=bias))

    # with torch.no_grad():
    #     output2 = bn_conv3(input)
    #     print(f'output2:{output2}\n,shape:{output2.shape}')
    
    # print(torch.allclose(output1, output2))

if __name__ == '__main__':
    # confused_conv3x3_bn()
    # confused_conv1x1_bn()
    # convert_1x1_3x3()
    convert_bn_conv3()