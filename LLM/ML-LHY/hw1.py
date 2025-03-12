'''
Author: jhq
Date: 2025-03-11 20:52:26
LastEditTime: 2025-03-12 16:19:27
Description:
'''
import torch
from llama_cpp import Llama
from typing import List
from googlesearch import search as _search
from bs4 import BeautifulSoup
from charset_normalizer import detect
import asyncio
from requests_html import AsyncHTMLSession
import urllib3
urllib3.disable_warnings()

if not torch.cuda.is_available():
    raise Exception(
        'You are not using the GPU runtime. Change it first or you will suffer from the super slow inference speed!')
else:
    print('You are good to go!')

# async:声明异步函数


async def worker(s: AsyncHTMLSession, url: str):
    try:
        # await:声明程序挂起，等待异步操作完成
        header_response = await asyncio.wait_for(s.head(url, verify=False), timeout=10)
        if "text/html" not in header_response.headers.get("Content-Type", ""):
            return None
        r = await asyncio.wait_for(s.get(url, verify=False), timeout=10)
        return r.text
    except:
        return None


async def get_html(urls):
    session = AsyncHTMLSession()
    tasks = (worker(session, url) for url in urls)
    return await asyncio.gather(*tasks)


async def search(keyword: str, n_results: int = 3) -> List[str]:
    keyword = keyword[:100]
    results = list(_search(keyword, n_results * 2, lang="zh-CN", unique=True))
    results = await get_html(results)
    results = [x for x in results if x is not None]
    results = [BeautifulSoup(x, "html.parser") for x in results]
    results = ["".join(x.get_text().split()) for x in results if detect(
        x.encode()).get("encoding") == "utf-8"]
    return results[:n_results]

llama3 = Llama(model_path=r"C:\jhq\huggingface_model\bartowski\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-abliterated-Q8_0.gguf",
               verbose=False,
               n_gpu_layers=-1,
               n_ctx=8192,)

def generate_response(model: Llama, messages: str) -> str:
    output = model.create_chat_completion(
        messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=512,
        temperature=0.1,
        repeat_penalty=2.0,
    )["choices"][0]["message"]["content"]
    return output

# test_question='請問誰是 Taylor Swift？'

# messages = [
#     {"role": "system", "content": "你是 LLaMA-3.1-8B，是用來回答問題的 AI。使用中文時只會使用繁體中文來回問題。"},    # System prompt
#     {"role": "user", "content": test_question}, # User prompt
# ]

# print(generate_response(llama3, messages))


class LLMAgent():
    def __init__(self, role_description: str, task_description: str, llm: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"):
        self.role_description = role_description
        self.task_description = task_description
        self.llm = llm

    def inference(self, message: str) -> str:
        if self.llm == "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF":
            messages = [
                # System prompt
                {"role": "system", "content": self.role_description},
                # User prompt
                {"role": "user", "content": f"{self.task_description}\n{message}"},
            ]
            return generate_response(llama3, messages)
        else:
            return ""


question_extraction_agent = LLMAgent(
    role_description="你是一个专门的问题提取器，你会从描述性文字中提取出用户的真实问题。回答只输出问题，不要输出其他内容",
    task_description="请从以下文本中提取出问题",
)

keyword_extraction_agent = LLMAgent(
    role_description="你是一个专门的问题关键词提取器，你会从给定问题提取出问题的关键词，以适应搜索，请只输出关键词，用\"，\"分隔开，不要输出其他内容。",
    task_description="请从以下文本中提取出问题的关键词：",
)

qa_agent = LLMAgent(
    role_description="你是一个用来回答问题的AI。使用简体中文来回答问题",
    task_description="请回答以下问题：",
)

async def pipeline(question: str) -> str:
    print(f"question: {question}, \n")
    question_msg = question_extraction_agent.inference(question)
    print(f"question_msg: {question_msg}\n")
    results = []
    # if len(question_msg) > 10:
    #     keyword_msg = keyword_extraction_agent.inference(question)
    #     print(f"keyword_msg: {keyword_msg}, \n")
    #     results = await search(keyword_msg)
    messages = question_msg
    if len(results) > 0:
        messages = messages + ",以下是搜索引擎根据问题关键词返回的信息\n" + "\n".join(results)
        print(f"使用搜索引擎：messages:{messages}")
    if len(messages) > 8192:
        messages = messages[:8192]
    return qa_agent.inference(messages)


async def main():
    with open("./public.txt", "r", encoding="utf8") as input_f:
        questions = input_f.readlines()
        questions = [x.strip().split(",")[0] for x in questions]
        for id, question in enumerate(questions, 1):
            answer = await pipeline(question)
            answer = answer.replace("\n", " ")
            print(f"{id}. {question} \n {answer}")

asyncio.run(main())
# with open("./private.txt", "r") as input_f:
#     questions = input_f.readlines()
#     questions = [x.strip().split(",")[0] for x in questions]
#     for id, question in enumerate(questions, 1):
#         answer = await pipeline(question)
#         answer = answer.replace("\n", " ")
#         print(f"{id}. {question} \n {answer}")
