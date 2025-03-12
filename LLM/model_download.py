'''
Author: jhq
Date: 2025-02-24 17:33:48
LastEditTime: 2025-03-11 13:29:39
Description: 
'''
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('black-forest-labs/FLUX.1-dev',
                              cache_dir='C:/jhq/huggingface_model', revision='master')
# snapshot_download('iic/CosyVoice-300M', local_dir='C:/jhq/huggingface_model/iic/CosyVoice-300M')
# snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='C:/jhq/huggingface_model/iic/CosyVoice-300M-25Hz')
# snapshot_download('iic/CosyVoice-300M-SFT', local_dir='C:/jhq/huggingface_modeliic/CosyVoice-300M-SFT')
# snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='C:/jhq/huggingface_modeliic/CosyVoice-300M-Instruct')
# snapshot_download('iic/CosyVoice-ttsfrd', local_dir='C:/jhq/huggingface_modeliic/CosyVoice-ttsfrd')
