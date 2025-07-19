from chunk_file import ReadFiles
from vector import VectorStore
from embedding import SentenceTransformerEmbedding
from chat_model  import ChatModel
from IPython.display import display, Code, Markdown
from bs4 import BeautifulSoup

def run_mini_rag(question: str, knowledge_base_path: str, k: int = 1) -> str:
  """
  运行一个简化版的RAG项目。
  
  :param question: 用户提出的问题
  :param knowledge_base_path: 知识库的路径，包含文档的文件夹路径
  :param api_key: OpenAI API密钥，用于调用GPT-4o模型
  :param k: 返回与问题最相关的k个文档片段，默认为1
  :return: 返回GPT-4o模型生成的回答
  """
  # 加载并切分文档
  docs = ReadFiles(knowledge_base_path).get_content(max_token_len=600, cover_content=150)
  vector = VectorStore(docs)
  # 使用 SentenceTransformer Embedding 模型对文档进行向量化
  embedding=SentenceTransformerEmbedding(is_api=False)
  # 获取文档向量并存储
  vector.get_vector(embedding)
  # 打印文档向量
  print(vector.vectors)
  # 持久化存储到本地
  vector.persist('storage')
  # 在数据库中检索最相关的文档片段
  content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
  # 使用 Qwen3 模型生成答案
  chat = ChatModel() 
  answer=chat.chat(question, [], content)
  soup =BeautifulSoup(answer, 'html.parser')
  # print("模型回答:", answer)
  think_content = soup.find('think').text.strip() if soup.find('think') else None
  print ("🤖 \n思考结果:\n", think_content)
  print("🤖 \n正在生成回答...\n")
  non_think_content = []
  for element in soup.children:
      if element.name != 'think' and str(element).strip():
          non_think_content.append(str(element).strip())
  print('\n'.join(non_think_content))
  return non_think_content

if __name__ == '__main__':
  # 用户提出问题
  question = 'OpenAI的分词技术经历了哪些迭代'
  knowledge_base_path = './data'
  run_mini_rag(question, knowledge_base_path)