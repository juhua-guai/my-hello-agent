import os

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool

from chapter7.my_simple_agent import MySimpleAgent

load_dotenv()

llm = HelloAgentsLLM(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

agent = MySimpleAgent(name="学习助手", llm=llm)


def test_empty_memory():
    response = agent.run(input_text="我叫张三，正在学习Python，目前掌握了基础语法")
    print(response)
    print("=" * 60)
    # 第二次对话（新的会话），因为有 self._history 属性，所以现在有了基础记忆
    response2 = agent.run("你还记得我名字吗？")
    print(response2)


def test_memory():
    tool_registry = ToolRegistry()

    # 添加记忆工具
    memory_tool = MemoryTool(user_id="user001")
    tool_registry.register_tool(memory_tool)

    # 为Agent配置工具
    agent.tool_registry = tool_registry

    # 体验记忆功能
    print("=== 添加多个记忆 ===")
    result1 = memory_tool.execute("add", content="用户张三是一名Python开发者，专注于机器学习和数据分析", memory_type="semantic",
                        importance=0.8)
    print(f"记忆1: {result1}")

    result2 = memory_tool.execute("add", content="李四是前端工程师，擅长React和Vue.js开发", memory_type="semantic",
                                  importance=0.8)
    print(f"记忆2: {result2}")


if __name__ == '__main__':
    test_memory()
