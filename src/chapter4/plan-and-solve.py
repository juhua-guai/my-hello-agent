import ast

from my_hello_agent.hello_agents_llm import HelloAgentsLLM

# from llm_client import HelloAgentsLLM

PLANNER_PROMPT_TEMPLATE: str = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""


class Planer:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题，拆解成一个执行计划
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        message = [{"role": "user", "content": prompt}]
        print("--- 正在生成计划 ---")
        plan_text = self.llm_client.think(message)
        try:
            print(f"✅ 计划已生成:\n{plan_text}")
            print("=========================")
            if plan_text:
                step_list = plan_text.split("```python")[1].split("```")[0].strip()
                # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
                step_list = ast.literal_eval(step_list)
                return step_list if isinstance(step_list, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {plan_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []
        return []


EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""


class Executor:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.history = []

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        """
        history = ""  # 用于存储历史步骤和结果的字符串
        print("\n--- 正在执行计划 ---")
        response_text = ""
        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i + 1}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history or "",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages)
            # 更新历史记录，为下一步做准备
            history += f"步骤 {i + 1}：{step} \n 结果：{response_text} \n\n"
            print(f"步骤 {i + 1} 已完成，结果：{response_text}")
        # 获取最终结果
        final_answer = response_text if response_text else ""
        return final_answer


class PlanAndSolveAgent:
    def __init__(self, llm_client: HelloAgentsLLM, planner: Planer, executor: Executor):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client = llm_client
        self.planner = planner
        self.executor = executor

    def run(self, question: str):
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        plan: list[str] = self.planner.plan(question)

        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        final_answer = self.executor.execute(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")


def main() -> None:
    llm_client = HelloAgentsLLM()
    planer = Planer(llm_client)
    executor = Executor(llm_client)

    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    agent = PlanAndSolveAgent(llm_client, planner=planer, executor=executor)
    agent.run(question)


if __name__ == "__main__":
    main()
