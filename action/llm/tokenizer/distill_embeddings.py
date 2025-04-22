'''
Author: jhq
Date: 2025-04-12 14:02:26
LastEditTime: 2025-04-21 11:25:20
Description: 
'''
from model2vec.distill import distill, distill_from_model
from transformers import AutoModel, AutoTokenizer
from model2vec import StaticModel
import time

# model = AutoModel.from_pretrained(r"C:\jhq\huggingface_model\jinaai\jina-embeddings-v3", trust_remote_code=True)
# tokennizer = AutoTokenizer.from_pretrained(r"C:\jhq\huggingface_model\jinaai\jina-embeddings-v3", trust_remote_code=True)

texts = [
    "Follow the white rabbit.",  # English
    "Sigue al conejo blanco.",  # Spanish
    "Suis le lapin blanc.",  # French
    "跟着白兔走。",  # Chinese
    "اتبع الأرنب الأبيض.",  # Arabic
    "Folge dem weißen Kaninchen.",  # German
]

# When calling the `encode` function, you can choose a `task` based on the use case:
# 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
# Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
# embeddings = model.encode(texts, task="text-matching")

# Compute similarities
# print(embeddings[0] @ embeddings[1].T)

# m2v_model = distill_from_model(model, tokennizer, pca_dims=128)
# m2v_model.save_pretrained(r"C:\jhq\huggingface_model\jinaai\m2v_jina-embeddings-v3-pca128")


model = StaticModel.from_pretrained(r"C:\jhq\huggingface_model\jinaai\m2v_jina-embeddings-v3-pca128", token=None)
start_time = time.time()
embeddings = model.encode(texts)
print(time.time() - start_time)
print(embeddings.shape)
print(embeddings[0] @ embeddings.T)