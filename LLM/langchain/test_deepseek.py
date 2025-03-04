'''
Author: jhq
Date: 2025-03-04 18:27:40
LastEditTime: 2025-03-04 18:29:56
Description: 
'''
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="C:/jhq/huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",
    # other params...
)


messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
llm.invoke(messages)