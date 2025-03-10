'''
Author: jhq
Date: 2025-03-04 17:01:52
LastEditTime: 2025-03-05 22:44:45
Description: 
'''
from LangChainLLM import DeepSeek_R1_Distill_Qwen_LLM, Meta_Llama_3_ChatModel
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import (ConversationBufferMemory, 
            ConversationBufferWindowMemory, 
            ConversationTokenBufferMemory, 
            ConversationSummaryBufferMemory)


llm = Meta_Llama_3_ChatModel(mode_name_or_path = "C:/jhq/huggingface_model/LLM-Research/Meta-Llama-3___1-8B-Instruct",
                             custom_get_token_ids_path = "C:/jhq/huggingface_model/Ransake/gpt2-tokenizer-fast")

# 对话缓存储存
memory = ConversationBufferMemory()
memory.save_context({"input": "你好，我叫皮皮鲁"}, {"output": "你好啊，我叫鲁西西"})
# memory.load_memory_variables({})
# conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
# print(conversation.predict(input="你好,我叫皮皮鲁"))
# print(conversation.predict(input="1+1等于多少"))
# print(conversation.predict(input="我叫什么名字"))
# # memory.save_context({"input": "很高兴和你成为朋友！"}, {"output": "是的，让我们一起去冒险吧！"})
# # memory.load_memory_variables({})
# # print(memory.buffer)
print(memory.load_memory_variables({}))

# 窗口缓存储存，只保存k个对话记录
window_memory = ConversationBufferWindowMemory(k=2)
window_memory.save_context({"input": "你好，我叫皮皮鲁"}, {"output": "你好啊，我叫鲁西西"})
window_memory.save_context({"input": "很高兴和你成为朋友！"}, {"output": "是的，让我们一起去冒险吧！"})
window_memory.save_context({"input": "我们去北京吧"}, {"output": "好啊好啊，一起去北京玩玩"})
print(window_memory.load_memory_variables({}))

# 对话字符缓存存储
token_memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
token_memory.save_context({"input": "朝辞白帝彩云间，"}, {"output": "千里江陵一日还。"})
token_memory.save_context({"input": "两岸猿声啼不住，"}, {"output": "轻舟已过万重山。"})
# conversation = ConversationChain(llm=llm, memory=token_memory, verbose=True)
# print(conversation.predict(input="朝辞白帝彩云间"))
print(token_memory.load_memory_variables({}))

# 创建一个长字符串
schedule = "在八点你和你的产品团队有一个会议。 \
你需要做一个PPT。 \
上午9点到12点你需要忙于LangChain。\
Langchain是一个有用的工具，因此你的项目进展的非常快。\
中午，在意大利餐厅与一位开车来的顾客共进午餐 \
走了一个多小时的路程与你见面，只为了解最新的 AI。 \
确保你带了笔记本电脑可以展示最新的 LLM 样例."

summary_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
summary_memory.save_context({"input": "你好，我叫皮皮鲁"}, {"output": "你好啊，我叫鲁西西"})
summary_memory.save_context({"input": "很高兴和你成为朋友！"}, {"output": "是的，让我们一起去冒险吧！"})
summary_memory.save_context({"input": "今天的日程安排是什么？"}, {"output": f"{schedule}"})

print(summary_memory.load_memory_variables({})['history'])

conversation = ConversationChain(llm=llm, memory=summary_memory, verbose=True)
print(conversation.predict(input="展示什么样的样例最好呢？"))