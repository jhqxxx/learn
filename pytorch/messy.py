'''
Author: jhq
Date: 2025-04-10 09:38:11
LastEditTime: 2025-04-10 09:38:16
Description: 
'''
import torch

CUDA_DEVICE = "cuda:0"
if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片