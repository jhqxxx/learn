'''
Author: jhq
Date: 2025-04-29 12:27:30
LastEditTime: 2025-04-29 12:29:06
Description: 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import torch
import re
import json
import ast

model_name = "/mnt/c/jhq/huggingface_model/ZhipuAI/GLM-4-9B-0414"

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

def is_function_call(single_message):
    """判断当前系统消息是否为function call"""
    pattern = re.compile(r'([^\n`]*?)\n({.*?})(?=\w*\n|$)', re.DOTALL)
    matches = pattern.findall(single_message)
    if not matches:
        return False

    func_name, args_str = matches[0]
    func_name = func_name.strip()
    try:
        parsed_args = json.loads(args_str)
    except json.JSONDecodeError:
        try:
            parsed_args = ast.literal_eval(args_str)
        except:
            return False
    
    return {"name": func_name, "arguments": parsed_args}

def realtime_aqi(city):
    """天气查询工具"""
    if '北京' in city.lower():
        return json.dumps({'city': '北京', 'aqi': '10', 'unit': 'celsius'}, ensure_ascii=False)
    elif '上海' in city.lower():
        return json.dumps({'city': '上海', 'aqi': '72', 'unit': 'fahrenheit'}, ensure_ascii=False)
    else:
        return json.dumps({'city': city, 'aqi': 'unknown'}, ensure_ascii=False)

def build_system_prompt(tools):
    """基于工具列表构建系统提示"""
    if tools is None:
        tools = []
    value = "# 可用工具"
    contents = []
    for tool in tools:
        content = f"\n\n## {tool['function']['name']}\n\n{json.dumps(tool['function'], ensure_ascii=False, indent=4)}"
        content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
        contents.append(content)
    value += "".join(contents)
    return value

tools = [
  {
    "type": "function", 
    "function": {
      "name": "realtime_aqi",
      "description": "天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息",
      "parameters": {
          "type": "object",
          "properties": {
              "city": {
                  "description": "城市名"
              }
          },
          "required": [
              "city"
          ]
      }
	}
  }
]

system_prompt = build_system_prompt(tools)

message = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "北京和上海今天的天气情况"}
]
print(f"User Message: {message[-1]['content']}")

while True:
    inputs = tokenizer.apply_chat_template(
        message,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 1024,
        "do_sample": True,
    }
    out = model.generate(**generate_kwargs)
    generate_resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:-1], skip_special_tokens=False)
    stop_sequence = tokenizer.decode(out[0][-1:], skip_speical_tokens=False)
    if stop_sequence == "<|user|>":
        print(f"Assistant Response: {generate_resp.strip()}")
        break
    print(f"Generate Response: {generate_resp.strip()}")
    function_calls = []
    for m in generate_resp.split("<|assistant|>"):
        fc_decode = is_function_call(m.strip())
        if fc_decode:
            message.append({"role": "assistant", "metadata": fc_decode['name'], "content": json.dumps(fc_decode['arguments'], ensure_ascii=False)})
            print(f"Function Call: {fc_decode}")
            function_calls.append(fc_decode)
        else:
            message.append({"role": "assistant", "content": m})
            print(f"Assistant Response: {m.strip()}")
    
    for fc in function_calls:
        function_response = realtime_aqi(
            city=fc["arguments"]["city"],
        )
        print(f"Function Response: {function_response}")
        message.append({"role": "observation", "content": function_response})