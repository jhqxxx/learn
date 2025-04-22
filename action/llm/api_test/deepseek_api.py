'''
Author: jhq
Date: 2025-04-16 19:25:40
LastEditTime: 2025-04-16 19:56:17
Description: 
'''
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
# 
# client = OpenAI(api_key="xxx", base_url="https://api.deepseek.com")
client = OpenAI(api_key="xxx", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
response = client.chat.completions.create(
    model="deepseek-v3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "麻辣豆腐包怎么做"},
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content)