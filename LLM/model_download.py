'''
Author: jhq
Date: 2025-02-24 17:33:48
LastEditTime: 2025-03-04 20:55:01
Description: 
'''
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('iic/cv_resnet18_card_correction',
                              cache_dir='C:/jhq/huggingface_model', revision='master')
