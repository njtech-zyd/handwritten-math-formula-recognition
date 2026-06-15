"""
Image Renderer — 将 InkML 笔画轨迹渲染为 PNG 图像。

支持跨数据集的坐标归一化（CROHME 各年份坐标范围差异大），
输出标准化图像供多模态大模型进行视觉识别。
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .inkml_parser import InkMLData, Trace


def normalize_points(
    points: np.ndarray,
    img_size: int = 448,
    margin_ratio: float = 0.05,
) -> np.ndarray:
    """将坐标归一化到图像画布内。

    步骤：
    1. 所有点平移至正数区域
    2. 按最大边长等比缩放（保持宽高比）
    3. 翻转 Y 轴（InkML Y↓ → 图像 Y↑）
    4. 平移至画布居中，留边距

    Args:
        points: shape (N, 2) 的原始坐标数组。
        img_size: 正方形画布边长（像素）。
        margin_ratio: 边距占画布的比例（如 0.05 = 5%）。

    Returns:
        shape (N, 2) 的归一化坐标数组（float32）。
    """
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # 计算有效区域（剔除无效值后）
    valid = points[~(np.isnan(points).any(axis=1))]
    if len(valid) == 0:
        return np.zeros((len(points), 2), dtype=np.float32)

    # 平移至最小值为 0
    min_xy = valid.min(axis=0)
    centered = points - min_xy  # type: ignore

    # 等比缩放
    max_val = centered.max()
    if max_val > 0:
        usable = img_size * (1 - 2 * margin_ratio)
        scale = usable / max_val
        scaled = centered * scale
    else:
        scaled = centered

    # 平移至画布居中（不移 Y 轴：CROHME 数据使用 Y-down 坐标，
    # 即 Y 增加方向向下，与 PIL 图像坐标一致，无需翻转）
    margin = img_size * margin_ratio
    result = scaled.copy()
    result[:, 0] += margin                        # X: 靠左留边距
    result[:, 1] += margin                        # Y: 留边距（不移转）

    return result.astype(np.float32)


def _draw_trace(draw: ImageDraw.Draw, points: np.ndarray, width: int) -> None:
    """在 ImageDraw 上绘制一条笔画。"""
    if len(points) < 2:
        return
    pts = [(float(p[0]), float(p[1])) for p in points]
    draw.line(pts, fill=0, width=width)


def render_traces(
    traces: list[Trace],
    img_size: int = 448,
    stroke_width: int = 4,
    margin_ratio: float = 0.05,
    background: int = 255,
) -> Image.Image:
    """将 Trace 列表渲染为灰度图像。

    Args:
        traces: 笔画列表。
        img_size: 正方形画布边长（像素）。
        stroke_width: 笔画线条宽度（像素）。
        margin_ratio: 边距占画布比例。
        background: 背景灰度值（255=白底，0=黑底）。

    Returns:
        PIL.Image（模式="L"，灰度图）。
    """
    img = Image.new("L", (img_size, img_size), background)
    draw = ImageDraw.Draw(img)

    # 拼接所有坐标用于归一化
    all_points = np.vstack([t.points for t in traces if len(t) > 0])
    if len(all_points) == 0:
        return img

    # 归一化
    normed = normalize_points(all_points, img_size, margin_ratio)

    # 按各 trace 在 normed 中的对应位置拆分绘制
    offset = 0
    for trace in traces:
        n = len(trace)
        if n > 0:
            _draw_trace(draw, normed[offset: offset + n], stroke_width)
            offset += n

    return img


def render_inkml(
    data: InkMLData,
    img_size: int = 448,
    stroke_width: int = 4,
    margin_ratio: float = 0.05,
) -> Image.Image:
    """直接将 InkMLData 渲染为图像。

    等价于 render_traces(data.traces, ...)。
    """
    return render_traces(data.traces, img_size, stroke_width, margin_ratio)


def batch_render(
    dataset: list[InkMLData],
    img_size: int = 448,
    stroke_width: int = 4,
    margin_ratio: float = 0.05,
    verbose: bool = True,
) -> list[dict]:
    """批量渲染数据集。

    Args:
        dataset: InkMLData 列表。
        img_size: 渲染尺寸。
        stroke_width: 笔画宽度。
        margin_ratio: 边距比例。
        verbose: 是否显示进度。

    Returns:
        字典列表: [{"data": InkMLData, "image": PIL.Image, "file": str}, ...]
    """
    results = []
    iterator = dataset
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(dataset, desc="Rendering")

    for data in iterator:
        img = render_inkml(data, img_size, stroke_width, margin_ratio)
        results.append({
            "data": data,
            "image": img,
            "file": str(data.file_path) if data.file_path else "",
        })

    return results
