import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

class BaseEmbeddings:
    """
    向量化的基类，用于将文本转换为向量表示。不同的子类可以实现不同的向量获取方法。
    """
    def __init__(self,is_api: bool) -> None:
        """
        初始化基类。
        参数：
        path (str) - 如果是本地模型，path 表示模型路径；如果是API模式，path可以为空
        is_api (bool) - 表示是否使用API调用，如果为True表示通过API获取Embedding
        """
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        抽象方法，用于获取文本的向量表示，具体实现需要在子类中定义。
        
        参数：
        text (str) - 需要转换为向量的文本
        model (str) - 所使用的模型名称
        
        返回：
        list[float] - 文本的向量表示
        """
        raise NotImplementedError
    
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度，用于衡量它们的相似程度。
        
        参数：
        vector1 (list[float]) - 第一个向量
        vector2 (list[float]) - 第二个向量
        
        返回：
        float - 余弦相似度值，范围从 -1 到 1，越接近 1 表示向量越相似
        """
        dot_product = np.dot(vector1, vector2)  # 向量点积
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)  # 向量的模
        if not magnitude:
            return 0
        return dot_product / magnitude  # 计算余弦相似度

class SentenceTransformerEmbedding(BaseEmbeddings):
  def __init__(self, is_api: bool = True) -> None:
          """
          初始化类，设置 OpenAI API 客户端，如果使用的是 API 调用。
          参数：
          path (str) - 本地模型的路径，使用API时可以为空
          is_api (bool) - 是否通过 API 获取 Embedding，默认为 True
          """
          load_dotenv()
          super().__init__(is_api)
          if self.is_api:
                from openai import OpenAI
                self.client = OpenAI()
                self.client.api_key = os.getenv("EMBEDDING_API_KEY")  
                self.client.base_url = os.getenv("EMBEDDING_URL") 
                self.model = os.getenv("EMBEDDING_MODEL")  
          if self.is_api==False:
                self.path = os.getenv("LOCAL_MODEL_PATH")  
                self.model=SentenceTransformer(self.path)
  
  def get_embedding(self, text: str) -> List[float]:
      if self.is_api==False:
          return self.model.encode(text)
      else:
          text = text.replace("\n", " ")
         # 调用 OpenAI API 获取文本的向量表示
          return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

  def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
    return self.model.similarity(vector1, vector2)
      
if __name__ == '__main__':
    text1 = '我喜欢吃苹果'
    text2 = "苹果是我最喜欢吃的水果"
    text3 = "我喜欢用苹果手机"
    embeding=SentenceTransformerEmbedding(is_api=False)
    vector1=embeding.get_embedding(text1)
    vector2=embeding.get_embedding(text2)
    vector3=embeding.get_embedding(text3)

    sim=embeding.cosine_similarity(vector1,vector2)
    print(sim)