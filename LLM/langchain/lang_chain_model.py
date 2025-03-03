'''
Author: jhq
Date: 2025-02-27 11:58:39
LastEditTime: 2025-02-27 17:10:14
Description:
ollama加载模型时生成token需要配置环境变量
内存和显存的占用不知道在哪里配置，CPU 和GPU的使用也不知道怎么配
使用起来不如自己封装langchain的模型
demo测试老是报in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectError: [WinError 10049] 在其上下文中，该请求的地址无效。
应该是ollma的网络访问有问题，不想在这个问题上花时间了。

'''
# from langchain_ollama import OllamaLLM

# # 报错待解决
# llm = OllamaLLM(model="deepseek-r1")

# llm.invoke("langsmith 怎么帮助测试？")


from LangChainLLM import DeepSeek_R1_Distill_Qwen_LLM
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 文本分割函数
def split_text(text):
    pattern = re.compile(r'(.*?)</think>(.*)', re.DOTALL) # 定义正则表达式模式
    match = pattern.search(text) # 匹配 <think>思考过程</think>回答
  
    if match: # 如果匹配到思考过程
        think_content = match.group(1).strip() # 获取思考过程
        answer_content = match.group(2).strip() # 获取回答
    else:
        think_content = "" # 如果没有匹配到思考过程，则设置为空字符串
        answer_content = text.strip() # 直接返回回答
  
    return think_content, answer_content
  
llm = DeepSeek_R1_Distill_Qwen_LLM(mode_name_or_path = "C:/jhq/huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
prompt = ChatPromptTemplate.from_messages([
    ("system", "您是世界一流的技术文档作家"),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
chain.invoke({"input": "langsmith怎样帮助测试"})
# response = llm("我如何为学习大模型制定目标？")
# think, answer = split_text(response) # 调用split_text函数，分割思考过程和回答
# print(f"{"-"*20}思考{"-"*20}")
# print(think) # 输出思考
# print(f"{"-"*20}回答{"-"*20}")
# print(answer) # 输出回答