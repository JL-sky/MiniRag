import os
from contextlib import AsyncExitStack
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

PROMPT_TEMPLATE = dict(
    PROMPT_TEMPLATE="""
    下面有一个或许与这个问题相关的参考段落，若你觉得参考段落能和问题相关，则先总结参考段落的内容。
    若你觉得参考段落和问题无关，则使用你自己的原始知识来回答用户的问题，并且总是使用中文来进行回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    有用的回答:"""
)

class ChatModel:
    def __init__(self):
        load_dotenv()
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")  # 读取 OpenAI API Key
        self.base_url = os.getenv("BASE_URL")  # 读取 BASE URL
        self.model = os.getenv("MODEL")  # 读取 model
        print("🤖 正在初始化聊天模型...",self.model)

        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 OPENAI_API_KEY")

        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
    def chat(self, prompt: str, history: List = [], content: str = '') -> str:
        """
        :param prompt: 用户的提问
        :param history: 之前的对话历史（可选）
        :param content: 可参考的上下文信息（可选）
        :return: 生成的回答
        """
        print("🤖 问题:", prompt)
        print("🤖 上下文:", content)

        # 构建包含问题和上下文的完整提示
        full_prompt = PROMPT_TEMPLATE['PROMPT_TEMPLATE'].format(question=prompt, context=content)
        messages=[
            {"role": "user", "content": full_prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        # 返回模型生成的第一个回答
        return response.choices[0].message.content
    
if __name__ == '__main__':
    chat_model = ChatModel()
    prompt = "你好，我想问一下，你们家的冰箱里面有什么东西可以用来做冰淇淋？"
    print(chat_model.chat(prompt))
    chat_model.chat(prompt)
