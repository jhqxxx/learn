from huggingface_hub import login, InferenceClient 
from transformers import CodeAgent, HfApiEngine


# 有bug，不知道怎么加载本地模型
# client = InferenceClient(model="C:\jhq\huggingface_model\LLM-Research\Meta-Llama-3___1-8B-Instruct")

# def llm_engine(messages, stop_sequences=["Task"]) -> str:
#     response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1024)
#     answer = response.choices[0].message.content
#     return answer

# # llm_engine = HfApiEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
# agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)
# agent.run("Could you translate this sentence from French, say it out loud and return the audio.",
#     sentence="Où est la boulangerie la plus proche?",
# )