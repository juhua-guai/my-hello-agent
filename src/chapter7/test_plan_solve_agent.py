# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

import ast
import re
from typing import Any, List, Protocol, Tuple, runtime_checkable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


@runtime_checkable
class LLMWithInvoke(Protocol):
    """LLM对象支持标准invoke调用"""

    def invoke(self, messages: List[dict], **kwargs: Any) -> str | None: ...


@runtime_checkable
class LLMWithThink(Protocol):
    """LLM对象支持think调用"""

    def think(self, messages: List[dict], **kwargs: Any) -> str | None: ...


# Union类型：支持invoke或think任一即可
LLMInvoker = LLMWithInvoke | LLMWithThink


class StubLLM:
    """
    内置的最小化LLM存根，用于在没有外部LLM配置时提供可执行的本地演示。
    仅支持基本的问题分解和逐步执行模拟。
    """

    def __init__(self) -> None:
        self._call_count = 0

    def invoke(self, messages: List[dict], **kwargs: Any) -> str:
        self._call_count += 1
        content = messages[0]["content"] if messages else ""

        if self._call_count == 1:
            # 首次调用：生成计划（解析问题）
            return '["理解问题：苹果销售数量", "计算周二销量：15×2=30个", "计算周三销量：30-5=25个", "汇总三天总量：15+30+25"]'
        else:
            # 后续调用：模拟逐步执行
            step_num = self._call_count - 1
            results = [
                "苹果销售问题：周一15个，周二是周一的两倍（30个），周三比周二少5个（25个）",
                "周一卖出15个苹果",
                "周二卖出30个苹果（15×2）",
                "周三卖出25个苹果（30-5）",
                "三天总共卖出70个苹果",
            ]
            return results[min(step_num, len(results) - 1)]

    def think(self, messages: List[dict], **kwargs: Any) -> str:
        return self.invoke(messages, **kwargs)


class MyPlanAndSolveAgent:
    def __init__(
        self,
        name: str,
        llm: LLMInvoker,
        planner_prompt: str = DEFAULT_PLANNER_PROMPT,
        executor_prompt: str = DEFAULT_EXECUTOR_PROMPT,
        max_plan_retries: int = 2,
        max_steps: int = 12,
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.llm = llm
        self.planner_prompt = planner_prompt
        self.executor_prompt = executor_prompt
        self.max_plan_retries = max_plan_retries
        self.max_steps = max_steps
        self.verbose = verbose
        self._history: List[dict] = []
        self._plan_retry_count = 0  # Track retry rounds for logging

    def _print_interaction(
        self, title: str, prompt: str, response: str, extra: str = ""
    ) -> None:
        """统一日志打印函数，保证格式统一"""
        if not self.verbose:
            return
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[{title}]")
        print(f"{sep}")
        print(f"--- Prompt ---")
        print(prompt)
        print(f"--- Response (raw) ---")
        print(response)
        if extra:
            print(f"--- Extra ---")
            print(extra)
        print(f"{sep}")

    def run(self, question: str, **kwargs) -> str:
        plan = self._generate_plan(question, **kwargs)
        final_answer = self._execute_plan(question, plan, **kwargs)
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": final_answer})
        return final_answer

    def _generate_plan(self, question: str, **kwargs) -> list[str]:
        last_error = None
        for retry_round in range(self.max_plan_retries + 1):
            self._plan_retry_count = retry_round
            prompt = self.planner_prompt.format(question=question)
            raw = self._invoke_llm(prompt, **kwargs)
            self._print_interaction(
                title=f"规划阶段 (重试轮次: {retry_round})",
                prompt=prompt,
                response=raw,
            )
            try:
                plan = self._parse_plan(raw)
                if plan:
                    self._print_interaction(
                        title=f"规划阶段 - 解析后的计划列表 (共 {len(plan)} 步)",
                        prompt="",
                        response=str(plan),
                    )
                    return plan[: self.max_steps]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise ValueError(f"无法解析规划结果: {last_error}")

    def _execute_plan(self, question: str, plan: list[str], **kwargs) -> str:
        executed_steps: List[Tuple[str, str]] = []
        total_steps = min(len(plan), self.max_steps)
        for idx, step in enumerate(plan[: self.max_steps], start=1):
            history_text = self._format_history(executed_steps)
            prompt = self.executor_prompt.format(
                question=question,
                plan=plan,
                history=history_text,
                current_step=step,
            )
            step_result = self._invoke_llm(prompt, **kwargs)
            self._print_interaction(
                title=f"执行阶段 - 步骤 {idx}/{total_steps}",
                prompt=prompt,
                response=step_result,
                extra=f"当前步骤内容: {step}",
            )
            executed_steps.append((step, step_result))
        if not executed_steps:
            return ""
        return executed_steps[-1][1]

    def _invoke_llm(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        result: str | None = None

        if hasattr(self.llm, "invoke"):
            # 优先尝试标准调用方式
            try:
                result = self.llm.invoke(messages=messages, **kwargs)
            except TypeError as exc:
                # 仅捕获参数类型不匹配导致的TypeError，尝试备用调用
                try:
                    result = self.llm.invoke(messages, **kwargs)
                except Exception as exc2:
                    raise RuntimeError(f"invoke调用失败: {exc2}") from exc2
        elif hasattr(self.llm, "think"):
            try:
                result = self.llm.think(messages, **kwargs)
            except TypeError as exc:
                try:
                    result = self.llm.think(messages)
                except Exception as exc2:
                    raise RuntimeError(f"think调用失败: {exc2}") from exc2
        else:
            raise AttributeError("llm对象既不支持invoke也不支持think")

        if result is None:
            raise ValueError("llm返回了空结果（None）")
        return str(result).strip()

    def _parse_plan(self, text: str) -> list[str]:
        code_block_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        candidate = code_block_match.group(1).strip() if code_block_match else None

        if candidate is None:
            bracket_match = re.search(r"\[[\s\S]*?\]", text)
            if bracket_match:
                candidate = bracket_match.group(0).strip()

        if not candidate:
            raise ValueError("未找到可解析的计划列表")

        parsed = ast.literal_eval(candidate)
        if not isinstance(parsed, list):
            raise ValueError("解析结果不是列表")

        cleaned = [str(item).strip() for item in parsed if str(item).strip()]
        if not cleaned:
            raise ValueError("计划列表为空")
        return cleaned

    def _format_history(self, executed_steps: list[tuple[str, str]]) -> str:
        if not executed_steps:
            return "（暂无）"
        lines = []
        for idx, (step, result) in enumerate(executed_steps, start=1):
            lines.append(f"{idx}. 步骤: {step}")
            lines.append(f"   结果: {result}")
        return "\n".join(lines)

    def get_history(self) -> list[dict]:
        return self._history

    def clear_history(self) -> None:
        self._history.clear()


PlanAndSolveAgent = MyPlanAndSolveAgent


def main():
    load_dotenv()
    llm = None
    use_stub = False

    try:
        from my_hello_agent.hello_agents_llm import HelloAgentsLLM

        llm = HelloAgentsLLM()
        print("[OK] 已加载外部LLM配置")
    except (ModuleNotFoundError, AttributeError) as exc:
        print(f"[WARN] 外部LLM不可用（{exc}），切换到内置存根模式")
        use_stub = True

    if use_stub or llm is None:
        llm = StubLLM()
        print("[INFO] 使用内置StubLLM进行演示")

    try:
        agent = MyPlanAndSolveAgent(name="我的规划执行助手", llm=llm, verbose=True)
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        print(f"\n[问题] {question}\n")
        result = agent.run(question)
        print(f"\n[结果] {result}")
        print(f"[历史] 历史消息条数: {len(agent.get_history())}")
    except Exception as exc:
        print(f"[ERROR] 运行失败（{type(exc).__name__}: {exc}）")
        raise


if __name__ == "__main__":
    main()