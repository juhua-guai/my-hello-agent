from abc import ABC, abstractmethod
from typing import Optional

from hello_agents import HelloAgentsLLM

from chapter7.config import Config
from chapter7.message import Message


class BaseAgent(ABC):
    """Agent基类"""

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None

    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> None:
        """运行"""
        pass

    def add_message(self, message: Message) -> None:
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> list[Message]:
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name}, provider={self.llm.provider})"
