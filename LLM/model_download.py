'''
Author: jhq
Date: 2025-02-24 17:33:48
LastEditTime: 2025-04-07 16:28:51
Description: 
'''
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
from modelscope.msdatasets import MsDataset

import kagglehub

# Download latest version
# path = kagglehub.dataset_download("lunarwhite/anime-face-dataset-ntumlds")

# print("Path to dataset files:", path)

model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct',
                              cache_dir='C:/jhq/huggingface_model', revision='master')
# ds = MsDataset.load('AI-ModelScope/webnovel_cn', split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD) 
# dataset = MsDataset.load("AI-ModelScope/TinyStories", cache_dir=r"C:/jhq/huggingface_dataset", trust_remote_code=True)
# ds =  MsDataset.load('fimine/anime_dataset', subset_name='default', split='train')
# snapshot_download('iic/CosyVoice-300M', local_dir='C:/jhq/huggingface_model/iic/CosyVoice-300M')
# snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='C:/jhq/huggingface_model/iic/CosyVoice-300M-25Hz')
# snapshot_download('iic/CosyVoice-300M-SFT', local_dir='C:/jhq/huggingface_modeliic/CosyVoice-300M-SFT')
# snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='C:/jhq/huggingface_modeliic/CosyVoice-300M-Instruct')
# snapshot_download('iic/CosyVoice-ttsfrd', local_dir='C:/jhq/huggingface_modeliic/CosyVoice-ttsfrd')
