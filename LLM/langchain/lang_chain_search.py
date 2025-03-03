'''
Author: jhq
Date: 2025-02-26 13:55:44
LastEditTime: 2025-02-27 16:40:19
Description: 
'''
from LangChainLLM import DeepSeek_R1_Distill_Qwen_LLM
import re
from langchain_community.document_loaders import PyPDFLoader
import getpass
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

os.environ["LANGSMITH_TRACING"] = "true"
# getpass提供了平台无关的在命令行下输入密码的方法
os.environ["LANGSMITH_API_KEY"] = getpass.getpass() 


# 增加检索链
file_path = "D:\\file\\Rust-程序设计语言-简体中文版.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
print(len(docs))

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

embeddings = HuggingFaceEmbeddings(model_name="C:/jhq/huggingface_model/sentence-transformers/all-mpnet-base-v2")

# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "rust的vector怎么使用"
)
print(results[0])

results = vector_store.similarity_search_with_score("rust的枚举怎么使用")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

embedding = embeddings.embed_query("生命周期如何理解")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])
