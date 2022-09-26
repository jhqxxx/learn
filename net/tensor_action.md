<!--
 * @Descripttion: 
 * @version: 
 * @Author: jhq
 * @Date: 2022-09-21 23:27:24
 * @LastEditors: jhq
 * @LastEditTime: 2022-09-25 15:55:15
-->
### 张量操作算子
* 索引切片算子
    - index_select:
        * torch.index_select(input, dim, index)
        * 返回一个新的张量，返回的张量与原始张量具有相同的维数, 第dim维的大小与index的长度相同，其他维度的大小与原始张量中的大小相同

    - gather:收集、聚集
        * torch.gather(input, dim, index)
        * 沿指定dim，利用index来索引input特定位置的数值
        * input和index必须具有相同的维度数

* 维度变换算子
    - squeeze：挤、压榨
        * torch.squeeze(input, dim=None, *)
        * if dim is None,返回移除input中维度为1的维度，如：A*1*B*C*1*D->A*B*C*D
        * if dim is given,维度操作只作用于给定的维度，如：A*1*B，如果dim=0,返回维度还是A*1*B,如果dim=1, 返回A*B

    - unsqueeze：解压
        * torch.unsqueeze(input, dim)
        * 指定位置新增维度1

    - permute:交易、交换
        * torch.permute(input, dims)
        * 返回维度变换后的tensor
    
    - transpose: 转置
        * torch.transpose(input, dim0, dim1)

* 合并分割
    - concat:连接
        * torch.concat(tensors, dim=0): torch.cat()的别名
        * torch.cat(tensors, dim=0, *, out=None)
        * 将给定tensor序列沿指定维度连接，给定tensor维度需相同或为空
    
    - split: 分裂、分割
        * torch.split(tensor, split_size_or_sections, dim=0)
        * 将tensor沿指定维度拆分
        * 如果split_size_or_sections是整数，则tensor沿dim平均分成大小为split_size_or_sections的块(如果可能的话)
        * 如果其是list,则分成len(list)个块，每个块尺寸对应list[i]
    
    - chunk: 块
        * torch.chunk(input, chunks, dim=0)
        * 尝试将张量拆分成指定数量的块，如果给定维度张量大小不能被整除，则最后一个大小会不同
