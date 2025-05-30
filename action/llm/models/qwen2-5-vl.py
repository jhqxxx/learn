from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import time
import torch

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,     # 使用 4-bit 量化
    bnb_4bit_quant_type="nf4",     # 使用 NF4量化类型
    bnb_4bit_use_double_quant=True,  # 使用嵌套量化：将会在第一轮量化之后启用第二轮量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算数据类型
)

model_dir = r"C:\jhq\huggingface_model\Qwen\Qwen2___5-VL-3B-Instruct"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype='auto', device_map='auto')
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, quantization_config=nf4_config,
#     low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(model_dir, quantization_config=nf4_config)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:////D:\messy\qwen-vl-test.png",
            },
            {"type": "text", "text": "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本"},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)
print(image_inputs)
print(video_inputs)

# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

# inputs = inputs.to("cuda")

# generated_ids = model.generate(**inputs, max_new_tokens=512)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]

# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)
# print("-" * 50)
# start_time = time.time()
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "file:///D:\messy\qwen-vl-test2.png",
#             },
#             {"type": "text", "text": "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本"},
#         ],
#     }
# ]

# text = processor.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# image_inputs, video_inputs = process_vision_info(messages)
# print(image_inputs)
# print(video_inputs)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

# inputs = inputs.to("cuda")
# generated_ids = model.generate(**inputs, max_new_tokens=512)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]

# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)
# print(time.time() - start_time)