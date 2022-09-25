'''
Descripttion: 
version: 
Author: jhq
Date: 2022-09-22 22:02:07
LastEditors: jhq
LastEditTime: 2022-09-25 15:56:49
'''
import torch

'''index_select'''
# # torch.randn(size): 生成满足标准正态分布的随机数，维度为size，size可以是一维，二维...n维
# tensor_input = torch.randn(2, 4, 3)
# print(tensor_input)
# print(tensor_input.shape)

# indices = torch.tensor([1, 0, 0, 1])
# tensor_select = torch.index_select(tensor_input, 0, indices)
# print(tensor_select)
# print(tensor_select.shape)

# indices = torch.tensor([1, 2, 0, 3])
# tensor_select = torch.index_select(tensor_input, 1, indices)
# print(tensor_select)
# print(tensor_select.shape)

# indices = torch.tensor([1, 2, 0])
# tensor_select = torch.index_select(tensor_input, 2, indices)
# print(tensor_select)
# print(tensor_select.shape)

'''gather'''
# tensor_input = torch.randn(2, 4, 3)
# index = torch.tensor([[[0, 1, 2], [0, 2, 1], [1, 2, 0]],[[0, 0, 0], [2, 2, 2], [1, 1, 1]]])
# tensor_gather = torch.gather(tensor_input, 2, index)
# print(tensor_gather)

'''squeeze'''
# squeeze_input = torch.randn(2, 1, 3)
# print(squeeze_input)
# print(squeeze_input.shape)

# dim_none = torch.squeeze(squeeze_input)
# print(dim_none)
# print(dim_none.shape)
# dim_0 = torch.squeeze(squeeze_input, dim=0)
# print(dim_0)
# print(dim_0.shape)
# dim_1 = torch.squeeze(squeeze_input, dim=1)
# print(dim_1)
# print(dim_1.shape)

'''unsqueeze'''
# unsqueeze_input = torch.randn(2, 3)
# print(unsqueeze_input)
# print(unsqueeze_input.shape)
# unsqueeze_0 = torch.unsqueeze(unsqueeze_input, dim=0)
# print(unsqueeze_0)
# print(unsqueeze_0.shape)
# unsqueeze_1 = torch.unsqueeze(unsqueeze_input, dim=1)
# print(unsqueeze_1)
# print(unsqueeze_1.shape)
# unsqueeze_2 = torch.unsqueeze(unsqueeze_input, dim=2)
# print(unsqueeze_2)
# print(unsqueeze_2.shape)

'''permute'''
# permute_input = torch.randn(2, 3, 4)
# print(permute_input)
# print(permute_input.shape)
# after_permute = torch.permute(permute_input, dims=(1, 2, 0))
# print(after_permute)
# print(after_permute.shape)

'''transpose'''
# transpose_input = torch.randn(2, 3)
# print(transpose_input)
# print(transpose_input.shape)
# after_transpose = torch.transpose(transpose_input, dim0=0, dim1=1)
# print(after_transpose)
# print(after_transpose.shape)

'''concat'''
# tensor1 = torch.randn(2, 3)
# tensor2 = torch.randn(2, 3)
# tensor3 = torch.randn(2, 3)
# print(tensor1, tensor2, tensor3)
# cat_tensor = torch.concat([tensor1, tensor2, tensor3], dim=1)
# print(cat_tensor)

'''split'''
# split_input = torch.randn(2, 5)
# print(split_input)
# print(split_input.shape)
# split_tensor_int = torch.split(split_input, 3, dim=1)
# print(split_tensor_int)
# split_tensor_list = torch.split(split_input, [1, 2, 2], dim=1)
# print(split_tensor_list)

'''chunk'''
chunk_input = torch.randn(2, 11)
print(chunk_input)
chunk_out = torch.chunk(chunk_input, 6, dim=1)
print(chunk_out)
