'''
Author: jhq
Date: 2022-11-24 11:42:51
LastEditTime: 2022-11-24 13:16:04
Description: 
'''
import torch
from einops import rearrange, reduce, repeat

x = torch.randn(2,3,4,4)  #  4D tensor: bs*ic*h*w
# 1.转置
out1 = x.transpose(1, 2)
out2 = rearrange(x, 'b i h w -> b h i w')

# 2.变形
out1 = x.reshape(6, 4, 4)
out2 = rearrange(x, 'b i h w -> (b i) h w')
out3 = rearrange(out2, '(b i) h w -> b i h w', b=2)
flag = torch.allclose(out1, out2)
flag = torch.allclose(x, out3)

# 3. image2patch
out1 = rearrange(x, 'b i (h1 p1) (w1 p2)-> b i (h1 w1) (p1 p2)', p1=2, p2=2) # p1, p2 patch_height,patch_width
out2 = rearrange(out1, 'b i n a -> b n (a i)')   # out2 bs*num_patch*patch_depth

# 4. 求平均池化
out1 = reduce(x, 'b i h w -> b i h', 'mean') # mean, min, max, sum, prob
out2 = reduce(x, 'b i h w -> b i h 1', 'sum') # keep dimension
out3 = reduce(x, 'b i h w -> b i', 'max')

# 5. 堆叠张量
tensor_list = [x, x, x]
out1 = rearrange(tensor_list, 'n b i h w -> n b i h w')
print(out1.shape)

# 6.扩维
out1 = rearrange(x, 'b i h w -> b i h w 1')  # torch.unsqueeze
print(out1.shape)

# 7. 复制
out2 = repeat(out1, 'b i h w 1 -> b i h w 2')  # torch.tile
print(out2.shape)

out3 = repeat(x, 'b i h w -> b i (2 h) (2 w)')
print(out3.shape)