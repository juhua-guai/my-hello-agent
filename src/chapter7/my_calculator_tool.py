import ast
import math
import operator
import re


def my_calculate(expression: str) -> str:
    """简单的数学计算函数"""
    if not expression.strip():
        return "计算表达式不能为空"

    # 支持的基本运算
    operators = {
        ast.Add: operator.add,  # +
        ast.Sub: operator.sub,  # -
        ast.Mult: operator.mul,  # *
        ast.Div: operator.truediv,  # /
    }

    # 支持的基本函数
    functions = {
        'sqrt': math.sqrt,
        'pi': math.pi,
    }

    try:
        sanitized_expression = _extract_expression(expression)
        node = ast.parse(sanitized_expression, mode='eval')
        result = _eval_node(node.body, operators, functions)
        return str(result)
    except Exception:
        return "计算失败，请检查表达式格式"


def _extract_expression(text: str) -> str:
    """从自然语言中提取可计算表达式"""
    if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s_a-zA-Z,]+", text):
        return text.strip()

    allowed = set("0123456789.+-*/() ,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    extracted = "".join(ch for ch in text if ch in allowed).strip()
    if not extracted:
        raise ValueError("未提取到有效表达式")
    return extracted


def _eval_node(node, operators, functions):
    """简化的表达式求值"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        op = operators.get(type(node.op))
        return op(left, right)
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name in functions:
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            return functions[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]
    raise ValueError("不支持的表达式")
