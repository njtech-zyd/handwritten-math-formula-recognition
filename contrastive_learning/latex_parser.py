"""
LaTeX 结构解析 — 从 LaTeX 字符串中提取公式结构属性。

用于训练结构分类器的自动标注工具。
"""

import re
from typing import Optional

# 定义所有要检测的结构属性
STRUCTURE_FEATURES = [
    "has_frac",
    "has_sqrt",
    "has_greek",
    "has_superscript",
    "has_subscript",
    "has_pm",
    "has_div",
    "has_sum",
    "has_int",
    "has_prod",
    "has_log",
    "has_trig",
    "has_arrow",
    "has_limit",
    "has_infinity",
    "has_dots",
]

GREEK_PATTERN = re.compile(
    r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|"
    r"vartheta|iota|kappa|lambda|mu|nu|xi|omicron|pi|varpi|rho|"
    r"varrho|sigma|varsigma|tau|upsilon|phi|varphi|chi|psi|omega|"
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)"
)

TRIG_PATTERN = re.compile(r"\\(sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan)")


def parse_latex(latex: str) -> dict[str, bool]:
    """解析 LaTeX 字符串，输出结构属性字典。

    Args:
        latex: 纯 LaTeX 字符串（不含 $$ 包围）。

    Returns:
        {feature_name: True/False, ...}
    """
    text = latex.strip().strip("$").strip()

    props = {
        "has_frac": bool(re.search(r"\\frac", text)),
        "has_sqrt": bool(re.search(r"\\sqrt", text)),
        "has_greek": bool(GREEK_PATTERN.search(text)),
        "has_superscript": bool(re.search(r"\^", text)),
        "has_subscript": bool(re.search(r"_", text)),
        "has_pm": bool(re.search(r"\\pm", text)),
        "has_div": bool(re.search(r"\\div", text)),
        "has_sum": bool(re.search(r"\\sum", text)),
        "has_int": bool(re.search(r"\\int", text)),
        "has_prod": bool(re.search(r"\\prod", text)),
        "has_log": bool(re.search(r"\\log", text)),
        "has_trig": bool(TRIG_PATTERN.search(text)),
        "has_arrow": bool(re.search(r"\\rightarrow|\\Rightarrow|\\mapsto", text)),
        "has_limit": bool(re.search(r"\\lim", text)),
        "has_infinity": bool(re.search(r"\\infty", text)),
        "has_dots": bool(re.search(r"\\cdots|\\ldots|\\vdots|\\ddots", text)),
    }
    return props


def props_to_text(props: dict[str, bool]) -> str:
    """将结构属性转为自然语言描述。"""
    parts = []
    true_features = [f for f in STRUCTURE_FEATURES if props.get(f)]

    if not true_features:
        return "This is a simple mathematical expression."

    descriptions = {
        "has_frac": "contains a fraction (\\frac{}{})",
        "has_sqrt": "contains a square root (\\sqrt{})",
        "has_greek": "contains Greek letters",
        "has_superscript": "has superscripts (^{})",
        "has_subscript": "has subscripts (_{})",
        "has_pm": "contains plus-minus symbol (\\pm)",
        "has_div": "contains division symbol (\\div)",
        "has_sum": "contains summation (\\sum)",
        "has_int": "contains integral (\\int)",
        "has_prod": "contains product (\\prod)",
        "has_log": "contains logarithm (\\log)",
        "has_trig": "contains trigonometric functions",
        "has_arrow": "contains arrows (\\rightarrow, \\Rightarrow)",
        "has_limit": "contains limit (\\lim)",
        "has_infinity": "contains infinity symbol (\\infty)",
        "has_dots": "contains ellipsis (\\cdots, \\ldots)",
    }

    for feat in true_features:
        parts.append(f"- The formula {descriptions.get(feat, feat)}")

    return "Structural analysis of the input formula:\n" + "\n".join(parts)


def batch_parse(latex_list: list[str]) -> list[dict[str, bool]]:
    """批量解析 LaTeX 列表。"""
    return [parse_latex(ltx) for ltx in latex_list]


def props_to_vector(props: dict[str, bool]) -> list[float]:
    """将结构属性转为浮点向量（用于训练分类器）。"""
    return [float(props.get(f, False)) for f in STRUCTURE_FEATURES]


def vector_to_props(vec) -> dict[str, bool]:
    """将分类器输出向量转回结构属性字典。"""
    return {f: bool(round(v)) for f, v in zip(STRUCTURE_FEATURES, vec)}
