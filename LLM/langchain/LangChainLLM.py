'''
Author: jhq
Date: 2025-02-26 13:49:50
LastEditTime: 2025-03-04 22:57:20
Description: 
参考
https://github.com/datawhalechina/self-llm/blob/master/models/DeepSeek-R1-Distill-Qwen/02-DeepSeek-R1-Distill-Qwen-7B%20Langchain%20%E6%8E%A5%E5%85%A5.md
'''
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import torch
import re
from langchain_core.prompt_values import PromptValue
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import LLMResult
from langchain_community.llms.anyscale import create_llm_result

class DeepSeek_R1_Distill_Qwen_LLM(LLM):
    # 基于本地 DeepSeek_R1_Distill_Qwen 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
      
    def __init__(self, mode_name_or_path :str):

        super().__init__()
        print("正在从本地加载模型...")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, quantization_config=nf4_config)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, quantization_config=nf4_config, device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        print("完成本地模型的加载")
      
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):

        messages = [{"role": "user", "content": prompt }]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(model_inputs.input_ids, generation_config=self.model.generation_config, attention_mask=model_inputs['attention_mask'], max_new_tokens=8192) # 思考需要输出更多的Token数，设为8K
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
      
        return response
    
    def split_text(self, text):
        pattern = re.compile(r'(.*?)</think>(.*)', re.DOTALL) # 定义正则表达式模式
        match = pattern.search(text) # 匹配 <think>思考过程</think>回答
    
        if match: # 如果匹配到思考过程
            think_content = match.group(1).strip() # 获取思考过程
            answer_content = match.group(2).strip() # 获取回答
        else:
            think_content = "" # 如果没有匹配到思考过程，则设置为空字符串
            answer_content = text.strip() # 直接返回回答
    
        return think_content, answer_content

    # def generate_prompt(self,
    #     prompts: list[PromptValue],
    #     stop: Optional[list[str]] = None,
    #     callbacks: Callbacks = None,
    #     **kwargs: Any,
    # ):
    #     input_ids = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    #     model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
    #     generated_ids = self.model.generate(model_inputs.input_ids, generation_config=self.model.generation_config, attention_mask=model_inputs['attention_mask'], max_new_tokens=8192) # 思考需要输出更多的Token数，设为8K
    #     generated_ids = [
    #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]
    #     response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #     think_content, answer_content = self.split_text(response)
         
    #     return create_llm_result()
        
        
    @property
    def _llm_type(self) -> str:
        return "DeepSeek_R1_Distill_Qwen_LLM"