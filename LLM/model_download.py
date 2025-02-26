'''
Author: jhq
Date: 2025-02-24 17:33:48
LastEditTime: 2025-02-25 17:40:34
Description: 
'''
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                              cache_dir='C:/jhq/huggingface_model', revision='master')
