'''
Author: jhq
Date: 2025-03-04 17:01:52
LastEditTime: 2025-03-04 17:25:51
Description: 
'''
from LangChainLLM import DeepSeek_R1_Distill_Qwen_LLM
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models.base import ChatOpenAI


llm = DeepSeek_R1_Distill_Qwen_LLM(mode_name_or_path = "C:/jhq/huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
conversation.predict(input="你好,我叫皮皮鲁")

