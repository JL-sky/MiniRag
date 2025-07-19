import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from embedding import BaseEmbeddings 
from embedding import SentenceTransformerEmbedding 

class VectorStore:
    def __init__(self, document: List[str] = None) -> None:
        """
        初始化向量存储类，存储文档和对应的向量表示。
        :param document: 文档列表，默认为空。
        """
        if document is None:
            document = []
        self.document = document  # 存储文档内容
        self.vectors = []  # 存储文档的向量表示

    # 使用传入的 `EmbeddingModel` 对所有文档进行向量化，并将这些向量存储在 `self.vectors` 中
    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        """
        使用传入的 Embedding 模型将文档向量化。
        :param EmbeddingModel: 传入的用于生成向量的模型（需继承 BaseEmbeddings 类）。
        :return: 返回文档对应的向量列表。
        """
        # 遍历所有文档，获取每个文档的向量表示
        self.vectors = [EmbeddingModel.get_embedding(doc) for doc in self.document]
        return self.vectors
    
    # 将文档片段及其向量表示保存到本地文件系统，便于持久化存储。
    def persist(self, path: str = 'storage'):
        """
        将文档和对应的向量表示持久化到本地目录中，以便后续加载使用。
        :param path: 存储路径，默认为 'storage'。
        """
        if not os.path.exists(path):
            os.makedirs(path)  # 如果路径不存在，创建路径
        # 保存向量为 numpy 文件
        np.save(os.path.join(path, 'vectors.npy'), self.vectors)
        # 将文档内容存储到文本文件中
        with open(os.path.join(path, 'documents.txt'), 'w') as f:
            for doc in self.document:
                f.write(f"{doc}\n")
    
    # 从本地文件系统加载已保存的文档片段和向量，供后续检索使用
    def load_vector(self, path: str = 'storage'):
        """
        从本地加载之前保存的文档和向量数据。
        :param path: 存储路径，默认为 'storage'。
        """
        # 加载保存的向量数据
        self.vectors = np.load(os.path.join(path, 'vectors.npy')).tolist()
        # 加载文档内容
        with open(os.path.join(path, 'documents.txt'), 'r') as f:
            self.document = [line.strip() for line in f.readlines()]

    # 接收用户输入的查询，通过向量化后在数据库中检索最相关的文档片段，并返回最匹配的文档
    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # 将查询文本向量化
        query_vector = EmbeddingModel.get_embedding(query)
        # 计算相似度并提取标量值
        similarities = [
            float(EmbeddingModel.cosine_similarity(query_vector, vector).item()) 
            for vector in self.vectors
        ]
        print("查询向量与每个文档向量的相似度：", similarities)
        
        # 获取相似度最高的 k 个文档索引
        top_k_indices = np.argsort(similarities)[-k:][::-1].tolist()
        print("相似度最高的 k 个文档索引：", top_k_indices)
        
        # 返回对应的文档内容
        return [self.document[idx] for idx in top_k_indices]
    
if __name__ == '__main__':
    # 初始化文档列表
  documents = [
      "机器学习是人工智能的一个分支。",
      "深度学习是一种特殊的机器学习方法。",
      "监督学习是一种训练模型的方式。",
      "强化学习通过奖励和惩罚进行学习。",
      "无监督学习不依赖标签数据。",
  ]
  # 创建向量数据库
  vector_store = VectorStore(document=documents)

  # 使用 SentenceTransformer Embedding 模型对文档进行向量化
  path="/data1/sx/jl/pro/model/modelscope/hub/models/Qwen/Qwen3-Embedding-4B"
  embedding_model = SentenceTransformerEmbedding(path=path, is_api=False)
  # 获取文档向量并存储
  vector_store.get_vector(embedding_model)
  # 打印文档向量
  print(vector_store.vectors)
  # 持久化存储到本地
  vector_store.persist('storage')
  # 模拟用户查询
  query = "什么是深度学习？"
  result = vector_store.query(query, embedding_model) 
  print("检索结果：", result)