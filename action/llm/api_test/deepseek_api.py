'''
Author: jhq
Date: 2025-04-16 19:25:40
LastEditTime: 2025-05-01 16:55:30
Description: 
'''
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
# 
# client = OpenAI(api_key="xxx", base_url="https://api.deepseek.com")
client = OpenAI(api_key="0c8b6b7cd5684a9c89ed565ff9d1e770.4sQqvGmNkjFzYdo2", base_url="https://open.bigmodel.cn/api/paas/v4")
response = client.chat.completions.create(
    model="glm-4-flash-250414",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "麻辣豆腐包怎么做"},
    ],
    stream=False
)

print(response)