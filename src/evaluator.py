"""
Evaluator — 识别结果评估模块。

提供表达式级别和字符级别的多项指标，用于量化评估模型性能。
"""

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# LaTeX 标准化
# ---------------------------------------------------------------------------


def normalize_latex(latex_str: str) -> str:
    """对 LaTeX 字符串进行标准化，消除等价表达式的格式差异。

    标准化步骤：
    - 去除首尾空白
    - 统一 $...$ 和 \\(...\\) 包围
    - 移除多余空格（保留控制序列内空格如 \\log x 中的空格）
    - 统一 \\log{x} → \\log x 等

    Args:
        latex_str: 原始 LaTeX 字符串。

    Returns:
        标准化后的 LaTeX 字符串。
    """
    s = latex_str.strip()

    # 移除 $$...$$ 和 \[...\] 包围
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()

    # 移除所有空格（LaTeX 数学模式中空格不参与渲染）
    s = re.sub(r"\s+", "", s)

    # 统一上下标顺序：^{X}_{Y} → _{Y}^{X}（LaTeX 中等价）
    # 先统一单字符大括号
    s = re.sub(r"\^\{(\w)\}", r"^\1", s)
    s = re.sub(r"\_\{(\w)\}", r"_\1", s)
    # 再交换上下标顺序（多字符情况）
    s = re.sub(r"\^\{(.+?)\}_\{(.+?)\}", r"_{\2}^{\1}", s)

    # 统一等价 LaTeX 命令（ldots 和 cdots 在公式中常混用）
    s = s.replace(r"\ldots", r"\dots")
    s = s.replace(r"\cdots", r"\dots")

    return s


# ---------------------------------------------------------------------------
# 核心指标
# ---------------------------------------------------------------------------


def expression_accuracy(
    predictions: Sequence[str],
    truths: Sequence[str],
) -> float:
    """计算表达式精确匹配率（Exact Match Accuracy）。

    Args:
        predictions: 识别结果列表。
        truths: 真值列表。

    Returns:
        0~1 之间的准确率。
    """
    if len(predictions) == 0:
        return 0.0

    preds_norm = [normalize_latex(p) for p in predictions]
    truths_norm = [normalize_latex(t) for t in truths]

    correct = sum(1 for p, t in zip(preds_norm, truths_norm) if p == t)
    return correct / len(predictions)


def cer(prediction: str, truth: str, normalize: bool = True) -> float:
    """计算单个表达式的字符错误率（CER）。

    CER = (编辑距离) / max(len(truth), len(prediction))

    编辑距离使用 Levenshtein 距离（插入+删除+替换）。

    Args:
        prediction: 预测的 LaTeX 字符串。
        truth: 真实 LaTeX 字符串。
        normalize: 是否先标准化再计算（默认 True）。

    Returns:
        0~1 之间的 CER 值（越小越好）。
    """
    if normalize:
        prediction = normalize_latex(prediction)
        truth = normalize_latex(truth)

    # Levenshtein 距离 — 空间优化版（仅保留两行）
    n, m = len(truth), len(prediction)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return 1.0  # 全是插入
    if m == 0:
        return 1.0  # 全是删除

    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if truth[i - 1] == prediction[j - 1] else 1
            curr[i] = min(
                curr[i - 1] + 1,      # 插入
                prev[i] + 1,          # 删除
                prev[i - 1] + cost,   # 替换
            )
        prev = curr

    distance = prev[n]
    return distance / max(n, m)


def mean_cer(
    predictions: Sequence[str],
    truths: Sequence[str],
) -> float:
    """计算批量 CER 均值。"""
    if len(predictions) == 0:
        return 0.0
    scores = [cer(p, t) for p, t in zip(predictions, truths)]
    return float(np.mean(scores))


def _get_ngrams(seq: list[str], n: int) -> dict[tuple[str, ...], int]:
    """从 token 序列中提取 n-gram 计数。"""
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(seq) - n + 1):
        gram = tuple(seq[i: i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def bleu_score(
    prediction: str,
    truth: str,
    max_n: int = 4,
    normalize: bool = True,
) -> float:
    """计算 BLEU 分数。

    LaTeX 字符串按空格和 \\ 进行 token 化，计算 1~4-gram 匹配率，
    并施加 brevity penalty。

    Args:
        prediction: 预测的 LaTeX 字符串。
        truth: 真实 LaTeX 字符串。
        max_n: 最大 n-gram 阶数（默认 4）。
        normalize: 是否先标准化。

    Returns:
        0~1 的 BLEU 分数（越高越好）。
    """
    if normalize:
        prediction = normalize_latex(prediction)
        truth = normalize_latex(truth)

    # 简易 token 化：按空白和 \ 拆分
    pred_tokens = re.findall(r"(?:\\[a-zA-Z]+|\\[.,;:!?(){}[\]])|\S", prediction)
    ref_tokens = re.findall(r"(?:\\[a-zA-Z]+|\\[.,;:!?(){}[\]])|\S", truth)

    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    # 计算各阶 n-gram 精确率
    log_avg = 0.0
    for n in range(1, min(max_n, len(ref_tokens)) + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        matches = sum(
            min(count, ref_ngrams.get(gram, 0))
            for gram, count in pred_ngrams.items()
        )
        total = max(sum(pred_ngrams.values()), 1)
        prec = matches / total
        if prec == 0:
            return 0.0
        log_avg += (1.0 / max_n) * np.log(prec)

    # Brevity Penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

    return float(bp * np.exp(log_avg))


def mean_bleu(
    predictions: Sequence[str],
    truths: Sequence[str],
    max_n: int = 4,
) -> float:
    """计算批量 BLEU 均值。"""
    if len(predictions) == 0:
        return 0.0
    scores = [bleu_score(p, t, max_n=max_n) for p, t in zip(predictions, truths)]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 数据组织与聚合
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """单个样本的评估结果。"""

    file: str                  # 文件名
    truth_latex: str          # 真实 LaTeX
    predicted_latex: str      # 预测 LaTeX
    exact_match: bool         # 是否完全匹配
    cer: float                # 字符错误率
    bleu: float               # BLEU 分数


def evaluate_all(
    predictions: Sequence[str],
    truths: Sequence[str],
    file_names: Sequence[str] = (),
) -> dict:
    """批量评估，返回汇总统计和详细结果。

    Args:
        predictions: 预测结果列表。
        truths: 真值列表。
        file_names: 可选的文件名列表（用于详细结果）。

    Returns:
        dict 包含:
        - "accuracy": float — 精确匹配率
        - "mean_cer": float — 平均 CER
        - "mean_bleu": float — 平均 BLEU
        - "total": int — 样本总数
        - "details": list[EvaluationResult] — 详细结果
    """
    if not file_names:
        file_names = [f"sample_{i}" for i in range(len(predictions))]

    details = [
        EvaluationResult(
            file=fname,
            truth_latex=t,
            predicted_latex=p,
            exact_match=p == t,
            cer=cer(p, t),
            bleu=bleu_score(p, t),
        )
        for fname, p, t in zip(file_names, predictions, truths)
    ]

    return {
        "accuracy": expression_accuracy(predictions, truths),
        "mean_cer": mean_cer(predictions, truths),
        "mean_bleu": mean_bleu(predictions, truths),
        "total": len(predictions),
        "details": details,
    }


def print_results(results: dict) -> None:
    """打印评估结果汇总。"""
    print("=" * 50)
    print("  评估结果汇总")
    print("=" * 50)
    print(f"  样本总数:      {results['total']}")
    print(f"  精确匹配率:    {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  平均 CER:      {results['mean_cer']:.4f}")
    print(f"  平均 BLEU:     {results['mean_bleu']:.4f}")
    print("=" * 50)
