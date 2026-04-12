from dotenv import load_dotenv

from chapter7.my_tools import MyToolRegistry

load_dotenv()


def test_with_simple_agent():
    """测试与SimpleAgent的集成"""
    from hello_agents import HelloAgentsLLM
    from my_calculator_tool import my_calculate

    llm = HelloAgentsLLM()
    tool_registry = MyToolRegistry()
    tool_registry.register_function(
        name="my_calculator",
        description="简单的数学计算工具，支持基本运算(+,-,*,/)",
        func=my_calculate,
    )

    print("🤖 与SimpleAgent集成测试:")
    user_question = "请帮我计算 16 + 2 * 3"
    print(f"用户问题: {user_question}")
    # 使用工具计算
    calc_result = tool_registry.execute_tool("my_calculator", user_question)
    print(f"计算结果: {calc_result}")


if __name__ == '__main__':
    test_with_simple_agent()
