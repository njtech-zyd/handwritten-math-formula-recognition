"""
Stroke Formatter — 将 InkML 笔画轨迹转换为文本描述。

使纯文本 LLM（如 DeepSeek）能够"阅读"手写笔画的时序坐标，
通过文本理解手写数学公式。
"""

import numpy as np

from .inkml_parser import InkMLData, Trace


def _normalize_coords(
    traces: list[Trace],
    output_range: float = 50,
) -> tuple[list[np.ndarray], float, float]:
    """将笔画坐标归一化到 [0, output_range] 范围。

    Returns:
        (归一化后的点列表, 缩放因子, 偏移量)
    """
    all_points = np.vstack([t.points for t in traces if len(t) > 0])
    if len(all_points) == 0:
        return [], 1.0, 0.0

    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    range_xy = max_xy - min_xy
    range_xy[range_xy == 0] = 1  # 防止除零

    normed = []
    for t in traces:
        if len(t) == 0:
            continue
        pts = (t.points - min_xy) / range_xy * output_range
        normed.append(np.round(pts).astype(int))

    return normed, float(range_xy[0]), float(min_xy[0])


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """均匀采样，将轨迹点压缩到最多 max_points 个。"""
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def traces_to_text(
    data: InkMLData,
    max_strokes: int = 30,
    max_points_per_stroke: int = 40,
    output_range: float = 50,
) -> str:
    """将 InkML 笔画数据格式化为文本描述。

    Args:
        data: InkML 解析结果。
        max_strokes: 最多保留的笔画数（过长的截断）。
        max_points_per_stroke: 每笔最多保留的点数。
        output_range: 坐标归一化范围 [0, N]。

    Returns:
        格式化的文本字符串。
    """
    traces = data.traces[:max_strokes]
    if not traces:
        return "No stroke data."

    normed, _, _ = _normalize_coords(traces, output_range)

    lines = [
        f"This handwritten math expression consists of {len(normed)} strokes.",
    ]

    for i, pts in enumerate(normed):
        if len(pts) == 0:
            continue
        pts = _sample_points(pts, max_points_per_stroke)
        coord_str = ",".join(f"{x} {y}" for x, y in pts)
        lines.append(f"<stroke id={i} pts={coord_str} />")

    lines.append(
        "Output the LaTeX code that represents this expression. "
        "Output ONLY the LaTeX code, no explanation, no markdown, no delimiters."
    )

    return "\n".join(lines)


def traces_to_ascii_grid(
    data: InkMLData,
    grid_width: int = 28,
    grid_height: int = 14,
    stroke_char: str = "#",
    thin_char: str = ".",
) -> str:
    """将笔画轨迹渲染为 ASCII 字符网格（保留二维空间结构）。

    类似"低分辨率像素渲染"，让文本模型能"看到"手写形状。
    """
    traces = data.traces
    if not traces:
        return f"No stroke data."

    all_points = np.vstack([t.points for t in traces if len(t) > 0])
    if len(all_points) == 0:
        return "No stroke data."

    # 归一化到网格范围（留边距）
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    range_xy = max_xy - min_xy
    range_xy[range_xy == 0] = 1

    usable_w = grid_width - 2
    usable_h = grid_height - 2
    scale_x = usable_w / range_xy[0]
    scale_y = usable_h / range_xy[1]

    # 初始化空网格
    grid = [[thin_char] * grid_width for _ in range(grid_height)]

    for t in traces:
        if len(t) < 2:
            continue
        pts = t.points
        prev_grid = None
        for p in pts:
            if np.any(np.isnan(p)):
                continue
            col = int((p[0] - min_xy[0]) * scale_x) + 1
            row = int((p[1] - min_xy[1]) * scale_y) + 1
            col = max(0, min(grid_width - 1, col))
            row = max(0, min(grid_height - 1, row))

            # 用不同字符表示连续笔画（当前点 + 连线路径）
            grid[row][col] = stroke_char
            if prev_grid is not None:
                # 在前后点之间画一条简易直线
                _draw_line_on_grid(grid, prev_grid[0], prev_grid[1],
                                   col, row, stroke_char)
            prev_grid = (col, row)

    # 转为文本
    lines = [f"Handwritten math expression ({grid_width}x{grid_height} grid):"]
    lines.append("+" + "-" * grid_width + "+")
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * grid_width + "+")

    return "\n".join(lines)


def _draw_line_on_grid(
    grid: list[list[str]],
    x0: int, y0: int, x1: int, y1: int,
    char: str,
) -> None:
    """Bresenham 直线算法，在网格上画线。"""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if 0 <= y0 < len(grid) and 0 <= x0 < len(grid[0]):
            grid[y0][x0] = char
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def traces_to_text_compact(
    data: InkMLData,
    max_strokes: int = 30,
    max_points_per_stroke: int = 40,
    output_range: float = 50,
) -> str:
    """更紧凑的格式 — 用方向向量减少 token 消耗。

    将绝对坐标转换为相对位移（dx, dy），便于模型捕捉笔画走向。
    """
    traces = data.traces[:max_strokes]
    if not traces:
        return "No stroke data."

    normed, _, _ = _normalize_coords(traces, output_range)

    lines = [f"Strokes: {len(normed)}"]
    for i, pts in enumerate(normed):
        if len(pts) == 0:
            continue
        pts = _sample_points(pts, max_points_per_stroke)

        # 绝对坐标 + 方向向量混合
        segs = []
        prev = None
        for x, y in pts:
            if prev is None:
                segs.append(f"{x},{y}")
            else:
                dx, dy = x - prev[0], y - prev[1]
                segs.append(f"{dx:+d},{dy:+d}")
            prev = (x, y)
        lines.append(f"S{i}:{' '.join(segs)}")

    return "\n".join(lines)
