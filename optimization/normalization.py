'''
Author: jhq
Date: 2022-11-26 14:38:27
LastEditTime: 2022-11-26 23:25:39
Description: about normalizaiton
'''
import torch
import torch.nn as nn


batch_size = 2
in_channel = 4
width = 4
height = 4
num_groups = 2
input = torch.randn(batch_size, in_channel, height, width)

def about_batch_normalization():
    #  batch_normalization
    # per channel across mini-batch    
    
    # print(f'input:{input}\ninput_shape:{input.shape}')
    torch_batch_norm = nn.BatchNorm2d(in_channel, affine=False)
    output = torch_batch_norm(input)
    print(f'output:{output}\noutput_shape:{output.shape}')

    # batch_norm:
    bn_mean = input.mean(dim=(0, 2, 3)).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(batch_size, 1, height, width)
    bn_std = input.std(dim=(0, 2, 3), unbiased=False).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(batch_size, 1, height, width)
    verify_bn_y = (input - bn_mean) / (bn_std+1e-5)
    print(f'verify_bn_y:{verify_bn_y}\nerify_bn_y:{verify_bn_y.shape}')

def about_layer_normalization():
    # layer_normalization
    # nlp中用的多 
    # per sample, per layer   
    pass

def about_instance_normalization():
    # 实例归一化
    # 分格迁移中用的多
    # per sample, per channel
    pass

def about_group_normalization():
    # per sample, per group
    torch_group_norm = nn.GroupNorm(num_groups, in_channel, affine=False)
    torch_gn_out = torch_group_norm(input)
    print(f'torch_gn_out:{torch_gn_out}\ntorch_gn_out:{torch_gn_out.shape}')

    group_inputs = torch.split(input, split_size_or_sections=in_channel // num_groups, dim=1)
    results = []
    for g_index in group_inputs:
        gn_mean = g_index.mean(dim=(1, 2, 3), keepdim=True)
        gn_std = g_index.std(dim=(1, 2, 3), unbiased=False, keepdim=True)
        gn_result = (g_index-gn_mean) / (gn_std+1e-5)
        results.append(gn_result)
    verify_gn_out = torch.cat(results, dim=1)
    print(f'verify_gn_out:{verify_gn_out}\nverify_gn_out:{verify_gn_out.shape}')


def about_weight_normalization():
    # 权重归一化
    input = torch.randn(batch_size, in_channel)
    linear = nn.Linear(in_channel, 3, bias=False)
    wn_linear = nn.utils.weight_norm(linear)
    wn_linear_output = wn_linear(input)
    print(f'wn_linear_output:{wn_linear_output}\nwn_linear_output:{wn_linear_output.shape}')

    weight_direction = linear.weight/linear.weight.norm(dim=1, keepdim=True)
    weight_magnitude = wn_linear.weight_g
    print(weight_direction.shape)
    print(weight_magnitude.shape)
    verify_output = input @ (weight_direction.transpose(-1, -2)) * (weight_magnitude.transpose(-1, -2))
    print(f'verify_output:{verify_output}\nverify_output:{verify_output.shape}')


# about_batch_normalization()
# about_group_normalization()
about_weight_normalization()