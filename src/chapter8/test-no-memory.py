import os

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

from chapter7.my_simple_agent import MySimpleAgent

load_dotenv()

llm = HelloAgentsLLM(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

agent = MySimpleAgent(name="学习助手", llm=llm)

response = agent.run(input_text="我叫张三，正在学习Python，目前掌握了基础语法")
print(response)
print("=" * 60)
# 第二次对话（新的会话），因为有 self._history 属性，所以现在有了基础记忆
response2 = agent.run("你还记得我名字吗？")
print(response2)
