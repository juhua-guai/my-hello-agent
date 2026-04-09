import os
from typing import Optional

from hello_agents import HelloAgentsLLM
from hello_agents.core.llm import SUPPORTED_PROVIDERS
from openai import OpenAI


class MyLLM(HelloAgentsLLM):
    """
    一个自定义的LLM客户端，通过继承增加了对ModelScope的支持
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 provider: Optional[SUPPORTED_PROVIDERS] = None, temperature: float = 0.7,
                 max_tokens: Optional[int] = None, timeout: Optional[int] = None, **kwargs):

        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"

            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("LLM_API_KEY")
            self.base_url = base_url or os.getenv("LLM_BASE_URL")

            if not self.api_key:
                raise ValueError("api key is empty!!")

            self.model = model or os.getenv("LLM_MODEL_ID")
            self.temperature = temperature
            self.timeout = timeout or 60
            self.max_tokens = max_tokens or 64000
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout)
