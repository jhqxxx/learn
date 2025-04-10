'''
Author: jhq
Date: 2025-04-10 09:16:24
LastEditTime: 2025-04-10 09:16:56
Description: 预训练模型
1. 数据准备
    1.1 文本数据先使用tokenizer进行分词，得到单词索引，存为bin文件
2. 模型训练
'''
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import torch
import torch.nn as nn
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.utils.versions import require_version
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from conf

