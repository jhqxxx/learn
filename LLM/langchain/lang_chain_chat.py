'''
Author: jhq
Date: 2025-02-27 11:58:39
LastEditTime: 2025-03-04 16:55:57
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
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

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
# print(llm)


# 首先，构造一个提示模版字符串：`template_string`
template_string = """把由三个反引号分隔的文本\
翻译成一种{style}风格。\
文本: ```{text}```
"""

# 然后，我们调用`ChatPromptTemplatee.from_template()`函数将
# 上面的提示模版字符`template_string`转换为提示模版`prompt_template`
prompt_template = ChatPromptTemplate.from_template(template_string)
print("\n", prompt_template.messages[0].prompt)

customer_style = """正式普通话 \
用一个平静、尊敬的语气
"""

customer_email = """
嗯呐，我现在可是火冒三丈，我那个搅拌机盖子竟然飞了出去，把我厨房的墙壁都溅上了果汁！
更糟糕的是，保修条款可不包括清理我厨房的费用。
伙计，赶紧给我过来！
"""

# 使用提示模版
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

print("客户消息类型:",type(customer_messages),"\n")

# 打印第一个客户消息类型
print("第一个客户客户消息类型类型:", type(customer_messages[0]),"\n")

# 打印第一个元素
print("第一个客户客户消息类型类型: ", customer_messages[0],"\n")
test_string = f"""把由三个反引号分隔的文本\
翻译成一种{customer_style}风格。\
文本: ```{customer_email}```
"""

# customer_response = llm(customer_messages[0].content)
# # customer_response = llm(test_string)
# print(customer_response)

service_reply = """嘿，顾客， \
保修不包括厨房的清洁费用， \
因为您在启动搅拌机之前 \
忘记盖上盖子而误用搅拌机, \
这是您的错。 \
倒霉！ 再见！
"""

service_style_pirate = """普通话\
一个有礼貌的语气 \
使用海盗风格\
"""
service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

print("\n", service_messages[0].content)

# service_response = llm(service_messages[0].content)
# print(service_response)

customer_review = """\
这款吹叶机非常神奇。 它有四个设置：\
吹蜡烛、微风、风城、龙卷风。 \
两天后就到了，正好赶上我妻子的\
周年纪念礼物。 \
我想我的妻子会喜欢它到说不出话来。 \
到目前为止，我是唯一一个使用它的人，而且我一直\
每隔一天早上用它来清理草坪上的叶子。 \
它比其他吹叶机稍微贵一点，\
但我认为它的额外功能是值得的。
"""

review_template = """\
对于以下文本，请从中提取以下信息：

礼物：该商品是作为礼物送给别人的吗？ \
如果是，则回答 是的；如果否或未知，则回答 不是。

交货天数：产品需要多少天\
到达？ 如果没有找到该信息，则输出-1。

价钱：提取有关价值或价格的任何句子，\
并将它们输出为逗号分隔的 Python 列表。

使用以下键将输出格式化为 JSON：
礼物
交货天数
价钱

文本: {text}
"""

prompt_template = ChatPromptTemplate.from_template(review_template)
print("提示模版：", prompt_template)


messages = prompt_template.format_messages(text=customer_review)

# response = llm(messages[0].content)

# print("结果类型:", type(response))
# print("结果:", response)

review_template_2 = """\
对于以下文本，请从中提取以下信息：：

礼物：该商品是作为礼物送给别人的吗？
如果是，则回答 是的；如果否或未知，则回答 不是。

交货天数：产品到达需要多少天？ 如果没有找到该信息，则输出-1。

价钱：提取有关价值或价格的任何句子，并将它们输出为逗号分隔的 Python 列表。

文本: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

gift_schema = ResponseSchema(name="礼物",
                             description="这件物品是作为礼物送给别人的吗？\
                            如果是，则回答 是的，\
                            如果否或未知，则回答 不是。")

delivery_days_schema = ResponseSchema(name="交货天数",
                                      description="产品需要多少天才能到达？\
                                      如果没有找到该信息，则输出-1。")

price_value_schema = ResponseSchema(name="价钱",
                                    description="提取有关价值或价格的任何句子，\
                                    并将它们输出为逗号分隔的 Python 列表")


response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print("输出格式规定：",format_instructions)

messages = prompt.format_messages(text=customer_review, format_instructions=format_instructions)
print("第一条客户消息:",messages[0].content)
response = llm(messages[0].content)
think, answer = split_text(response) # 调用split_text函数，分割思考过程和回答
print(f"{"-"*20}思考{"-"*20}")
print(think) # 输出思考
print(f"{"-"*20}回答{"-"*20}")
print(answer) # 输出回答

output_dict = output_parser.parse(answer)

print("解析后的结果类型:", type(output_dict))
print("解析后的结果:", output_dict)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "您是世界一流的技术文档作家"),
#     ("user", "{input}")
# ])
# output_parser = StrOutputParser()

# chain = prompt | llm | output_parser
# chain.invoke({"input": "langsmith怎样帮助测试"})

# response = llm("我如何为学习大模型制定目标？")
# think, answer = split_text(response) # 调用split_text函数，分割思考过程和回答
# print(f"{"-"*20}思考{"-"*20}")
# print(think) # 输出思考
# print(f"{"-"*20}回答{"-"*20}")
# print(answer) # 输出回答