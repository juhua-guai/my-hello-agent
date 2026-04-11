from typing import Optional

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

from chapter7.base_agent import BaseAgent
from chapter7.config import Config
from chapter7.message import Message

DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请直接给出可交付的完整答案，不要只给分析过程。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请按以下结构输出：
1. 质量评估（简要）
2. 问题列表（如果有）
3. 改进建议（可执行）
4. 结论（只能是“需要改进”或“无需改进”）
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请输出改进后的最终答案（直接可用，不要只写评审意见）。
"""
}

USER_PROMPT = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)"


class MyReflectionAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        custom_prompts: Optional[dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_rounds: int = 3,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_rounds = max_rounds
        self.prompts = self._merge_prompts(custom_prompts)
        print(f"✅ {name} 初始化完成，最大反思轮次: {self.max_rounds}")

    @staticmethod
    def _merge_prompts(custom_prompts: Optional[dict[str, str]]) -> dict[str, str]:
        merged = DEFAULT_PROMPTS.copy()
        if not custom_prompts:
            return merged
        for key in ("initial", "reflect", "refine"):
            value = custom_prompts.get(key)
            if isinstance(value, str) and value.strip():
                merged[key] = value
        return merged

    def _build_messages(self, prompt: str) -> list[dict]:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt).to_dict())
        messages.append(Message(role="user", content=prompt).to_dict())
        return messages

    def _invoke_with_stream(self, prompt: str, **kwargs) -> str:
        messages = self._build_messages(prompt)
        full_response = ""
        printed_any = False
        stream_method = getattr(self.llm, "stream_invoke", None)
        if callable(stream_method):
            try:
                for chunk in stream_method(messages, **kwargs):
                    if chunk is None:
                        continue
                    text = str(chunk)
                    if not text:
                        continue
                    printed_any = True
                    full_response += text
                    # print(text, end="", flush=True)
            except Exception:
                full_response = ""
                printed_any = False
        if not full_response.strip():
            full_response = self.llm.invoke(messages, **kwargs)
            if not printed_any and full_response:
                print(full_response, end="", flush=True)
        print()
        return full_response

    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")
        current_answer = self._invoke_with_stream(self.prompts["initial"].format(task=input_text), **kwargs)
        for round_idx in range(self.max_rounds):
            feedback = self._invoke_with_stream(
                self.prompts["reflect"].format(task=input_text, content=current_answer),
                **kwargs,
            )
            if "无需改进" in feedback:
                break
            current_answer = self._invoke_with_stream(
                self.prompts["refine"].format(
                    task=input_text,
                    last_attempt=current_answer,
                    feedback=feedback,
                ),
                **kwargs,
            )
            if round_idx == self.max_rounds - 1:
                break
        self.add_message(Message(role="user", content=input_text))
        self.add_message(Message(role="assistant", content=current_answer))
        print(f"✅ {self.name} 处理完成")
        return current_answer


def run_reflection_agent(
    task: str = USER_PROMPT,
    max_rounds: int = 3,
    provider: str = "modelscope",
) -> str:
    load_dotenv()
    llm = HelloAgentsLLM(provider=provider)
    agent = MyReflectionAgent(name="反思助手", llm=llm, max_rounds=max_rounds)
    return agent.run(task)


def main() -> None:
    result = run_reflection_agent(task=USER_PROMPT)
    print("=== 最终答案 ===")
    print(result)


if __name__ == "__main__":
    main()


