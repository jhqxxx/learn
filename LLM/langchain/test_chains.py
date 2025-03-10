'''
Author: jhq
Date: 2025-03-06 12:05:40
LastEditTime: 2025-03-06 13:16:49
Description: 
'''
from LangChainLLM import DeepSeek_R1_Distill_Qwen_LLM, Meta_Llama_3_ChatModel
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.sequential import SimpleSequentialChain, SequentialChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

llm = Meta_Llama_3_ChatModel(mode_name_or_path="C:/jhq/huggingface_model/LLM-Research/Meta-Llama-3___1-8B-Instruct",
                             custom_get_token_ids_path="C:/jhq/huggingface_model/Ransake/gpt2-tokenizer-fast")

# 简单链
# prompt_template = "给我讲一个{adjective}笑话"

# prompt = PromptTemplate(
#     input_variables=["adjective"], template=prompt_template
# )
# chain = prompt | llm | StrOutputParser()

# print(chain.invoke("有趣"))

# 简单顺序链
# first_prompt = ChatPromptTemplate.from_template(
#     "描述制造{product}的一个公司的最好的名称是什么"
# )
# chain_one = LLMChain(llm=llm, prompt=first_prompt)

# # 提示模板 2 ：接受公司名称，然后输出该公司的长为20个单词的描述
# second_prompt = ChatPromptTemplate.from_template(
#     "写一个20字的描述对于下面这个\
#     公司：{company_name}的"
# )
# chain_two = LLMChain(llm=llm, prompt=second_prompt)

# overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
#                                              verbose=True)

# product = "大号床单套装"
# overall_simple_chain.run(product)

# # 多个链
# first_prompt = ChatPromptTemplate.from_template(
#     "把下面的评论review翻译成英文:"
#     "\n\n{Review}"
# )
# # chain 1: 输入：Review    输出：英文的 Review
# chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

# #子链2
# # prompt模板 2: 用一句话总结下面的 review
# second_prompt = ChatPromptTemplate.from_template(
#     "请你用一句话来总结下面的评论review:"
#     "\n\n{English_Review}"
# )
# # chain 2: 输入：英文的Review   输出：总结
# chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")


# #子链3
# # prompt模板 3: 下面review使用的什么语言
# third_prompt = ChatPromptTemplate.from_template(
#     "下面的评论review使用的什么语言:\n\n{Review}"
# )
# # chain 3: 输入：Review  输出：语言
# chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")


# #子链4
# # prompt模板 4: 使用特定的语言对下面的总结写一个后续回复
# fourth_prompt = ChatPromptTemplate.from_template(
#     "使用特定的语言对下面的总结写一个后续回复:"
#     "\n\n总结: {summary}\n\n语言: {language}"
# )
# # chain 4: 输入： 总结, 语言    输出： 后续回复
# chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

# overall_chain = SequentialChain(
#     chains=[chain_one, chain_two, chain_three, chain_four],
#     input_variables=["Review"],
#     output_variables=["English_Review", "summary","followup_message"],
#     verbose=True
# )

# print(overall_chain("DeepSeek 是一款功能强大的工具，尤其在信息检索和数据分析领域表现出色。它的智能搜索算法能够快速定位高价值信息，帮助用户在海量数据中精准找到所需内容。无论是学术研究、市场分析还是日常查询，DeepSeek 都能提供高效、准确的解决方案。此外，其用户界面简洁友好，操作流畅，即使是新手也能轻松上手。总体而言，DeepSeek 是一款值得信赖的工具，能够显著提升工作效率，是信息时代的得力助手。"))


# 路由链有问题，教程很多api和新的版本对不上
