'''
Author: jhq
Date: 2025-02-24 17:33:48
LastEditTime: 2025-02-27 16:01:46
Description: 
'''
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('sentence-transformers/all-mpnet-base-v2',
                              cache_dir='C:/jhq/huggingface_model', revision='master')
