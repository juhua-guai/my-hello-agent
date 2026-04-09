from dotenv import load_dotenv

from chapter7.my_llm import MyLLM

load_dotenv()

llm = MyLLM(provider="modelscope")

# 准备消息
messages = [{"role": "user", "content": "你好，请介绍一下自己"}]

# 发起调用，think从父类中继承
response_stream = llm.think(messages, temperature=0.7)

# 打印响应
for chunk in response_stream:
    pass


