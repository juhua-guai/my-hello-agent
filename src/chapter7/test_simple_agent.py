import os

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry, CalculatorTool

from chapter7.my_simple_agent import MySimpleAgent

load_dotenv()

llm = HelloAgentsLLM(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"), provider="modelscope")

basic_agent = MySimpleAgent(name="小豆", llm=llm, system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。")


# 测试1:基础对话Agent（无工具）
def test1():
    print("=== 测试1:基础对话 ===")
    response1 = basic_agent.run("你好，请你介绍一下自己")
    print(f"基础对话响应: {response1}\n")


tool_registry = ToolRegistry()
calculator = CalculatorTool()
tool_registry.register_tool(calculator)
enhanced_agent = MySimpleAgent(
    name="增强助手",
    llm=llm,
    system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
    tool_registry=tool_registry,
    enable_tool_calling=True
)


# 测试2:带工具的Agent
def test2():
    print("=== 测试2:工具增强对话 ===")
    response2 = enhanced_agent.run("帮我计算数学公式： {4 + 6 / 2} 的最终结果 ", max_tool_iterations=3)
    print(f"工具增强响应: {response2}\n")


# 测试3:流式响应
def test3():
    print("=== 测试3:流式响应 ===")
    print("流式响应: ", end="")

    response3 = basic_agent.stream_run("作为一名java开发工程师，学习python代码感觉有难度，难以理解语法，有什么好的建议。")
    for chunk in response3:
        pass


# 测试4:动态添加工具
print("\n=== 测试4:动态工具管理 ===")
print(f"添加工具前: {basic_agent.has_tools()}")
basic_agent.add_tool(calculator)
print(f"添加工具后: {basic_agent.has_tools()}")
print(f"可用工具: {basic_agent.list_tools()}")


def main():
    test3()
    pass


if __name__ == '__main__':
    main()
