# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("/data1/sx/jl/pro/model/modelscope/hub/models/Qwen/Qwen3-Embedding-4B")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-4B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]


# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
# query_embeddings = model.encode(queries, prompt_name="query")
# document_embeddings = model.encode(documents)

# # Compute the (cosine) similarity between the query and document embeddings
# similarity = model.similarity(query_embeddings, document_embeddings)
# print(similarity)


text1 = '我喜欢吃苹果'
text2 = "苹果是我最喜欢吃的水果"
text3 = "我喜欢用苹果手机"

vector1 = model.encode(text1)
vector2 = model.encode(text2)
vector3 = model.encode(text3)

print(vector1)
print(vector2)
print(vector3)

# Compute the (cosine) similarity between the vectors
similarity1 = model.similarity(vector1, vector2)
similarity2 = model.similarity(vector1, vector3)
similarity3 = model.similarity(vector3, vector2)

print(similarity1)
print(similarity2)
print(similarity3)

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("你好，好久不见！"))