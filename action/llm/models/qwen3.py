'''
Author: jhq
Date: 2025-04-29 14:15:50
LastEditTime: 2025-05-30 20:41:04
Description: 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import uuid
import time
import json
from threading import Thread
from sse_starlette.sse import EventSourceResponse
import asyncio

class Qwen3Chatbot():
    def __init__(self, model_name="/mnt/c/jhq/huggingface_model/Qwen/Qwen3-8B"):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,     # 使用 4-bit 量化
            bnb_4bit_quant_type="nf4",     # 使用 NF4量化类型
            bnb_4bit_use_double_quant=True,  # 使用嵌套量化：将会在第一轮量化之后启用第二轮量化
            bnb_4bit_compute_dtype=torch.bfloat16 # 计算数据类型
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            quantization_config=nf4_config
        )
        self.history = []
        self.history_num = 3
        self.model_name = model_name.split('/')[-1]
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
    
    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][inputs.input_ids.shape[1]:].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        while len(self.history) > self.history_num: 
            self.history.pop(0)
        
        return response
    def get_chat_completions(self, output_text, start_time):        
        chat_completions = {}
        chat_completions["id"] = str(uuid.uuid4())
        chat_completions["object"] = "chat.completion"
        chat_completions["created"] = int(start_time)
        chat_completions["model"] = self.model_name
        chat_completions["choices"] = []
        for idx, txt in enumerate(output_text):
            chat_completions["choices"].append(
                Choice(
                    finish_reason="stop",
                    index=idx,
                    message=ChatCompletionMessage(
                        role="assistant", 
                        content=txt) 
                        ))  
        return chat_completions
    def generate_use_mes(self, messages):
        start_time = time.time()
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][inputs.input_ids.shape[1]:].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        chat_completions = self.get_chat_completions(response, start_time)        
        return chat_completions    
    def stream_completions(self, text):
        chat_completions = {}
        chat_completions["id"] = str(uuid.uuid4())
        chat_completions["object"] = "chat.completion.chunk"
        chat_completions["created"] = int(time.time())
        chat_completions["model"] = self.model_name
        chat_completions["choices"] = []
        chat_completions["choices"].append(
            {"index": 0,
             "delta": {"content": text}})  
        return json.dumps(chat_completions)
    
    def generate_stream(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generation_args = {
            "max_new_tokens": 32768,
            "streamer": self.streamer,
            "do_sample": False
        }
        generation_args.update(inputs)
        thread = Thread(target=self.model.generate, kwargs=generation_args)
        thread.start()
        async def stream_output(streamer):
            for text in streamer:
                if text:
                    res = self.stream_completions(text)
                    print(time.time())
                    print(res)
                    yield {
                        "event": "message",
                        "data": res
                        }
                    await asyncio.sleep(0.05)
            yield {
                "event": "message",
                "data": "[DONE]"
                }
        
        return EventSourceResponse(stream_output(self.streamer))

if __name__ == "__main__":
    chatbot = Qwen3Chatbot()  
    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print("----------------------")

    # Third input with /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")     

def test_qwen3():
    model_name = "/mnt/c/jhq/huggingface_model/Qwen/Qwen3-8B"

    # 以4bit加载模型，以 nf4 为量化类型、使用嵌套量化并使用 bfloat16 作为计算数据类型的模型
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,     # 使用 4-bit 量化
        bnb_4bit_quant_type="nf4",     # 使用 NF4量化类型
        bnb_4bit_use_double_quant=True,  # 使用嵌套量化：将会在第一轮量化之后启用第二轮量化
        bnb_4bit_compute_dtype=torch.bfloat16 # 计算数据类型
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        quantization_config=nf4_config
    )

    prompts = ["草莓种植有哪些要点", "八月份猕猴桃园怎么管理", "避免萝卜糠心有什么办法", "冬春育雏鸡如何防缺氧", "瓜实蝇咋消灭", "花卉如何安全度夏"]
    for p in prompts:
        messages = [
            {"role": "system", "content": "你是一个厉害的农业助手"},
            {"role": "user", "content": p},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            print(p)
            print("thinking content:", thinking_content)
            print("content:", content)
            print("-" * 50)