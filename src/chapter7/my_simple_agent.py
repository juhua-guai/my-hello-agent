from typing import Optional

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry

from chapter7.config import Config
from chapter7.message import Message


class MySimpleAgent(SimpleAgent):
    """
    重写的简单对话Agent
    展示如何基于框架积累构建自定义Agent
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            tool_registry: Optional["ToolRegistry"] = None,
            enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and self.tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """重写的运行方法 - 实现简单对话逻辑，支持可选工具调用"""
        print(f"🤖 {self.name} 正在处理: {input_text}")
        # 消息列表
        messages = []
        # 添加系统消息（可能包含工具信息）
        system_prompt = self._get_enhanced_system_prompt()
        messages.append(Message(system_prompt, "system").to_dict())
        # 历史消息
        for history in self._history:
            messages.append(history.to_dict())

        # 添加用户消息
        messages.append(Message(content=input_text, role="user").to_dict())
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(role="user", content=input_text))
            self.add_message(Message(role="assistant", content=response))
            print(f"✅ {self.name} 响应完成")
            return response
        # 需要调用工具
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个通用的AI助手。"
        if not self.enable_tool_calling or self.tool_registry is None:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        tools_section = "\n\n## 可用工具"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + f"{tools_section}"

    def _run_with_tools(self, messages, input_text, max_tool_iterations, **kwargs) -> str:

        pass
