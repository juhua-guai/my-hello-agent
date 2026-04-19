import datetime
from dataclasses import dataclass
from typing import Optional, Any

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.context import ContextBuilder
from hello_agents.tools import MemoryTool, RAGTool


@dataclass
class ContextPacket:
    """候选信息包：

    Attributes:
        content: 信息内容
        timestamp: 时间戳
        token_count: Token 数量
        relevance_score: 相关性分数(0.0-1.0)
        metadata: 可选的元数据
    """
    content: str
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.5
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        """初始化后的处理"""
        if self.metadata is None:
            self.metadata = {}
        # 确保相关性分数在有效范围内
        self.relevance_score = max(0.0, min(self.relevance_score, 1.0))


@dataclass
class ContextConfig:
    """
    上下文构建配置

    Attributes:
        max_tokens: 最大token数量
        reserve_ratio: 为系统指令预留的比例（0.0-1.0）
        min_relevance: 最低相关性阈值
        enable_compression: 是否启用压缩
        recency_weight: 新近性权重（0.0-1.0）
        relevance_weight: 相关性权重(0.0-1.0)
    """
    max_tokens: int
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7

    def __post_init__(self):
        """验证配置参数"""
        assert 0.0 <= self.reserve_ratio <= 1.0, "reserve_ratio 必须在 [0, 1] 范围内"
        assert 0.0 <= self.min_relevance <= 1.0, "min_relevance 必须在 [0, 1] 范围内"
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6, \
            "recency_weight + relevance_weight 必须等于 1.0"


config = ContextConfig(max_tokens=10000)
print(config)


class ContextAwareAgent(SimpleAgent):
    """具有上下文感知能力的Agent"""

    def __init__(self, name: str, llm: HelloAgentsLLM, **kwargs):
        super().__init__(name, llm, **kwargs)

        # 初始化上下文构建器
        self.memory_tool = MemoryTool(user_id=kwargs.get("user_id", "default"))
        self.rag_tool = RAGTool(knowledge_base_path=kwargs.get("knowledge_base_path", "./kb"))
        self.context_builder = ContextBuilder(memory_tool=self.memory_tool, rag_tool=self.rag_tool,
                                              config=ContextConfig(max_tokens=4000), )
        self.conversation_history = []

    def run(self, user_input: str) -> str:
        """运行 Agent，自动构建优化的上下文"""
        # 1. 使用 ContextBuilder 构建优化的上下文
        optimized_context = self.context_builder.build(user_query=user_input,
                                                       conversation_history=self.conversation_history,
                                                       system_instructions=self.system_prompt)
        # 2. 使用优化后的上下文调用 LLM
        messages = [
            {"role": "system", "content": optimized_context},
            {"role": "user", "content": user_input}
        ]
        response = self.llm.invoke(messages)

        # 3. 更新对话历史
        from hello_agents.core.message import Message
        from datetime import datetime

        self.conversation_history.append(
            Message(content=user_input, role="user", timestamp=datetime.now(), )
        )
        self.conversation_history.append(
            Message(content=response, role="assistant", timestamp=datetime.now())
        )

        # 4. 将重要交互记录到记忆系统
        self.memory_tool.run({
            "action": "add",
            "content": f"Q: {user_input}\nA: {response[:200]}...",  # 摘要
            "memory_type": "episodic",
            "importance": 0.6
        })
        return response


llm = HelloAgentsLLM()
agent = ContextAwareAgent(
    name="数据分析顾问",
    llm=HelloAgentsLLM(),
    system_prompt="你是一位资深的Python数据工程顾问。",
    user_id="user123",
    knowledge_base_path="./data_science_kb"
)

response = agent.run("如何优化Pandas的内存占用?")
print(response)
