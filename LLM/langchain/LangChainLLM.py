'''
Author: jhq
Date: 2025-02-26 13:49:50
LastEditTime: 2025-03-05 22:38:20
Description: 
参考
https://github.com/datawhalechina/self-llm/blob/master/models/DeepSeek-R1-Distill-Qwen/02-DeepSeek-R1-Distill-Qwen-7B%20Langchain%20%E6%8E%A5%E5%85%A5.md
'''
from langchain.llms.base import LLM
from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import torch
import re
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class DeepSeek_R1_Distill_Qwen_LLM(LLM):
    # 基于本地 DeepSeek_R1_Distill_Qwen 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_name_or_path: str):

        super().__init__()
        print("正在从本地加载模型...")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, quantization_config=nf4_config)
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path, quantization_config=nf4_config, device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(
            mode_name_or_path)
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(
            [input_ids], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            # 思考需要输出更多的Token数，设为8K
            model_inputs.input_ids, attention_mask=model_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=8192)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        think_content, answer_content = self.split_text(response)
        return answer_content

    def split_text(self, text):
        pattern = re.compile(r'(.*?)</think>(.*)', re.DOTALL)  # 定义正则表达式模式
        match = pattern.search(text)  # 匹配 <think>思考过程</think>回答

        if match:  # 如果匹配到思考过程
            think_content = match.group(1).strip()  # 获取思考过程
            answer_content = match.group(2).strip()  # 获取回答
        else:
            think_content = ""  # 如果没有匹配到思考过程，则设置为空字符串
            answer_content = text.strip()  # 直接返回回答

        return think_content, answer_content

    @property
    def _llm_type(self) -> str:
        return "DeepSeek_R1_Distill_Qwen_LLM"


class Meta_Llama_3_ChatModel(BaseChatModel):  # Meta-Llama-3

    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    custom_get_token_ids: AutoTokenizer = None

    def __init__(self, mode_name_or_path: str, custom_get_token_ids_path: str):

        super().__init__()
        print("正在从本地加载模型...")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, quantization_config=nf4_config)
        self.custom_get_token_ids = AutoTokenizer.from_pretrained(
            custom_get_token_ids_path, quantization_config=nf4_config)
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path, quantization_config=nf4_config, device_map="auto")
        print("完成本地模型的加载")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # Replace this with actual logic to generate a response from a list
        # of messages.
        last_message = messages[-1].content
        input_messages = [
            {"role": "user", "content": last_message, "temperature": 1}]
        input_ids = self.tokenizer.apply_chat_template(
            input_messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(
            [input_ids], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids, attention_mask=model_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        tokens = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        ct_input_tokens = sum(len(message.content) for message in messages)
        ct_output_tokens = len(tokens)
        message = AIMessage(
            content=tokens,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "Meta_Llama_3_ChatModel"
