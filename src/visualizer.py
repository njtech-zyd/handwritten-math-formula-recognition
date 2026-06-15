"""
Visualizer — 识别结果可视化模块。

生成论文级别的图表，含识别对比图、混淆矩阵、错误分布图等。
"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .evaluator import EvaluationResult
from .inkml_parser import InkMLData

# matplotlib 全局设置
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
})


def _strip_latex_delimiters(latex: str) -> str:
    """去除 LaTeX 字符串的 $$ 包围标记。"""
    text = latex.strip()
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()
    elif text.startswith("$") and text.endswith("$") and text.count("$") == 2:
        text = text[1:-1].strip()
    return text


def _set_axis_text(ax, text: str, is_math: bool = True, color="black"):
    """在 axis 上设置文本，用 $...$ 包裹数学公式。"""
    display = f"${text}$" if is_math else text
    ax.text(0.5, 0.5, display, fontsize=14, ha="center", va="center",
            color=color, transform=ax.transAxes)


def plot_comparison(
    image: Image.Image,
    truth_latex: str,
    pred_latex: str,
    file_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """绘制三栏对比图：原图 | 真值 | 预测。

    Args:
        image: 渲染后的公式图像。
        truth_latex: 真实 LaTeX。
        pred_latex: 预测 LaTeX。
        file_name: 文件名标签（可选）。
        save_path: 保存路径（可选，不指定则在内存中）。

    Returns:
        matplotlib Figure 对象。
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5),
                              gridspec_kw={"width_ratios": [1, 1.2, 1.2]})

    # 左：原图
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    # 中：真值（去除 $$ 包围后再用 $...$ 渲染）
    truth_display = _strip_latex_delimiters(truth_latex) if truth_latex else "(empty)"
    _set_axis_text(axes[1], truth_display, is_math=bool(truth_latex))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # 右：预测（去除空格差异后再比较，LaTeX 中空格不影响语义）
    import re as _re
    def _norm_latex(s):
        s = _strip_latex_delimiters(s) if s else ""
        s = _re.sub(r"\s+", "", s)
        s = _re.sub(r"\^\{(\w)\}", r"^\1", s)  # ^{n} → ^n
        s = _re.sub(r"\_\{(\w)\}", r"_\1", s)  # _{k} → _k
        s = _re.sub(r"\^\{(.+?)\}_\{(.+?)\}", r"_{\2}^{\1}", s)  # ^{X}_{Y} → _{Y}^{X}
        s = s.replace(r"\ldots", r"\dots")  # ldots ≡ cdots
        s = s.replace(r"\cdots", r"\dots")
        return s
    match = _norm_latex(truth_latex) == _norm_latex(pred_latex)
    color = "green" if match else "red"
    pred_display = _strip_latex_delimiters(pred_latex) if pred_latex else "(empty)"
    _set_axis_text(axes[2], pred_display, is_math=bool(pred_latex), color=color)
    axes[2].set_title("Prediction" if not file_name else f"Prediction ({file_name})")
    axes[2].axis("off")

    fig.suptitle("Correct" if match else "Mismatch",
                 color=color, fontsize=13, y=1.02)

    try:
        plt.tight_layout()
    except Exception:
        # LaTeX 语法错误导致渲染失败时，回退到纯文本显示
        for i in [1, 2]:
            axes[i].clear()
            axes[i].axis("off")
        axes[1].text(0.5, 0.5, truth_display, fontsize=14, ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("Ground Truth")
        axes[2].text(0.5, 0.5, pred_display, fontsize=14, ha="center", va="center",
                     color=color, transform=axes[2].transAxes)
        axes[2].set_title("Prediction" if not file_name else f"Prediction ({file_name})")
        plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    errors: list[tuple[str, str]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """绘制符号级混淆矩阵（热力图）。

    Args:
        errors: (true_symbol, pred_symbol) 错误对列表。
        save_path: 保存路径。

    Returns:
        matplotlib Figure 对象。
    """
    if not errors:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No errors to display", ha="center", va="center")
        return fig

    symbols = sorted(set(s for pair in errors for s in pair))
    n = len(symbols)
    idx = {s: i for i, s in enumerate(symbols)}
    matrix = np.zeros((n, n), dtype=int)

    for true_s, pred_s in errors:
        matrix[idx[true_s], idx[pred_s]] += 1

    # 归一化为百分比
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float),
                           where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.4)))
    im = ax.imshow(normalized, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(symbols, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(symbols, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                color = "white" if normalized[i, j] > 0.5 else "black"
                ax.text(j, i, str(matrix[i, j]),
                        ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Error Rate")
    ax.set_title("Symbol-Level Confusion Matrix")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_error_distribution(
    results: list[EvaluationResult],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """绘制错误率分布图。

    包含两个子图：
    1. CER 分布的直方图
    2. CER vs 表达式长度的散点图

    Args:
        results: 单样本评估结果列表。
        save_path: 保存路径。

    Returns:
        matplotlib Figure 对象。
    """
    cers = [r.cer for r in results]
    lengths = [len(r.truth_latex) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左：CER 分布直方图
    axes[0].hist(cers, bins=20, edgecolor="white", color="steelblue")
    axes[0].axvline(np.mean(cers), color="red", linestyle="--",
                    label=f'Mean CER={np.mean(cers):.3f}')
    axes[0].set_xlabel("CER")
    axes[0].set_ylabel("Count")
    axes[0].set_title("CER Distribution")
    axes[0].legend(fontsize=8)

    # 右：CER vs 表达式长度
    axes[1].scatter(lengths, cers, alpha=0.6, s=15, c="coral")
    has_trend = False
    if len(set(lengths)) > 1 and len(set(cers)) > 1:
        try:
            z = np.polyfit(lengths, cers, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(lengths), max(lengths), 100)
            axes[1].plot(xs, p(xs), "b--", alpha=0.7, label="Trend")
            has_trend = True
        except np.linalg.LinAlgError:
            pass
    axes[1].set_xlabel("Expression Length (chars)")
    axes[1].set_ylabel("CER")
    axes[1].set_title("CER vs Expression Length")
    if has_trend:
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def create_report(
    results: list[EvaluationResult],
    images: Optional[list[Image.Image]] = None,
    inkml_data: Optional[list[InkMLData]] = None,
    output_dir: str | Path = "results/figures",
) -> dict[str, Path]:
    """自动生成完整报告 — 多图组合。

    生成以下文件到 output_dir：
    - report_comparison_*.png — 每张图像的对比图
    - confusion_matrix.png — 混淆矩阵
    - error_distribution.png — 错误分布
    - summary.txt — 文本总结

    Args:
        results: 评估结果列表。
        images: 对应的渲染图像（可选，用于生成对比图）。
        output_dir: 输出目录。

    Returns:
        已生成的 {名字: 路径} 字典。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}

    # 1. 对比图
    if images:
        for i, (res, img) in enumerate(zip(results, images)):
            # 取消上限，生成所有图片
            name = Path(res.file).stem if res.file else f"sample_{i}"
            out = output_dir / f"report_comparison_{name}.png"
            plot_comparison(
                image=img,
                truth_latex=res.truth_latex,
                pred_latex=res.predicted_latex,
                file_name=name,
                save_path=out,
            )
            plt.close()
            generated[f"comparison_{i}"] = out

    # 2. 混淆矩阵（无符号级错误数据，仅占位用符号级对比）
    if sum(1 for r in results if not r.exact_match) > 3:
        errors = []
        for r in results:
            if not r.exact_match:
                # 将整串 LaTeX 作为"符号级"占位
                errors.append((r.truth_latex[:5], r.predicted_latex[:5]))
        out = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(errors, save_path=out)
        plt.close()
        generated["confusion_matrix"] = out

    # 3. 错误分布
    out = output_dir / "error_distribution.png"
    plot_error_distribution(results, save_path=out)
    plt.close()
    generated["error_distribution"] = out

    # 4. 文本总结
    summary_path = output_dir / "summary.txt"
    accuracy = sum(1 for r in results if r.exact_match) / max(len(results), 1)
    summary_path.write_text(
        f"Evaluation Summary\n"
        f"{'=' * 40}\n"
        f"Total samples: {len(results)}\n"
        f"Accuracy:      {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
        f"Mean CER:      {np.mean([r.cer for r in results]):.4f}\n"
        f"Mean BLEU:     {np.mean([r.bleu for r in results]):.4f}\n",
        encoding="utf-8",
    )
    generated["summary"] = summary_path

    return generated
