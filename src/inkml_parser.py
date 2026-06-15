"""
InkML Parser — 解析手写数学公式 InkML 文件。

InkML 是 W3C 数字墨水标准（XML 格式），用于描述手写笔画轨迹。
本模块支持 CROHME 2011–2014 和 ICFHR 2012 所有数据集的格式变体。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np

# InkML 命名空间
INKML_NS = "http://www.w3.org/2003/InkML"


@dataclass
class Trace:
    """单条笔画，包含时序坐标序列。"""

    id: str
    points: np.ndarray  # shape (N, 2), dtype=np.float32 — [[x1,y1], [x2,y2], ...]

    @property
    def x(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.points[:, 1]

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class TraceGroup:
    """笔画组 — 分层分组结构，每个组可代表一个符号或子表达式。"""

    id: str
    truth: Optional[str] = None  # 该组的真值标注
    trace_refs: list[str] = field(default_factory=list)  # 引用的 trace id
    children: list["TraceGroup"] = field(default_factory=list)  # 子组


@dataclass
class InkMLData:
    """InkML 文件解析结果。"""

    traces: list[Trace]
    truth_latex: Optional[str] = None  # annotation type="truth"（LaTeX）
    truth_mathml: Optional[str] = None  # annotationXML MathML
    trace_groups: list[TraceGroup] = field(default_factory=list)  # 分段信息
    annotations: dict[str, str] = field(default_factory=dict)  # 其他 annotation
    ui: Optional[str] = None  # 来源标识
    file_path: Optional[Path] = None

    @property
    def num_traces(self) -> int:
        return len(self.traces)

    @property
    def num_points(self) -> int:
        return sum(len(t) for t in self.traces)

    @property
    def has_truth(self) -> bool:
        return self.truth_latex is not None or self.truth_mathml is not None

    @property
    def has_segmentation(self) -> bool:
        return len(self.trace_groups) > 0


def _parse_trace_points(text: str) -> np.ndarray:
    """将 trace 文本内容解析为点坐标数组。

    坐标对格式为 "x y"，点之间以逗号分隔。
    例如: "10350 2248, 10338 2224, 10334 2207"
    """
    text = text.strip()
    if not text:
        return np.empty((0, 2), dtype=np.float32)

    pairs = text.split(",")
    points = []
    for pair in pairs:
        parts = pair.strip().split()
        if len(parts) >= 2:
            points.append([float(parts[0]), float(parts[1])])

    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(points, dtype=np.float32)


def _parse_trace_groups(
    element: ET.Element, ns: str
) -> list[TraceGroup]:
    """递归解析 traceGroup 结构。"""
    groups = []
    for tg_elem in element.findall(f"{{{ns}}}traceGroup"):
        group = TraceGroup(id=tg_elem.get("xml:id", tg_elem.get("id", "")))

        # 提取该组的真值标注
        for ann in tg_elem.findall(f"{{{ns}}}annotation"):
            if ann.get("type") == "truth":
                group.truth = ann.text

        # 提取 traceView 引用
        for tv in tg_elem.findall(f"{{{ns}}}traceView"):
            ref = tv.get("traceDataRef")
            if ref:
                group.trace_refs.append(ref)

        # 递归处理嵌套组
        group.children = _parse_trace_groups(tg_elem, ns)
        groups.append(group)

    return groups


def parse_inkml(file_path: str | Path) -> InkMLData:
    """解析 InkML 文件，返回统一数据结构。

    Args:
        file_path: InkML 文件路径。

    Returns:
        InkMLData: 包含所有 traces、标注和分段信息。
    """
    file_path = Path(file_path)
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 处理命名空间
    ns = INKML_NS
    tag = root.tag
    if tag.startswith("{"):
        ns = tag[1:].split("}")[0]

    # 1. 提取 traceFormat（如有）
    trace_format: dict[str, int] = {}
    tf_elem = root.find(f"{{{ns}}}traceFormat")
    if tf_elem is not None:
        for i, ch in enumerate(tf_elem.findall(f"{{{ns}}}channel")):
            trace_format[ch.get("name", "")] = i

    # 2. 提取所有 trace
    traces: list[Trace] = []
    for trace_elem in root.findall(f"{{{ns}}}trace"):
        tid = trace_elem.get("id", "")
        text = trace_elem.text or ""
        points = _parse_trace_points(text)
        traces.append(Trace(id=tid, points=points))

    # 3. 提取所有 annotation
    annotations: dict[str, str] = {}
    truth_latex: Optional[str] = None
    ui: Optional[str] = None
    for ann in root.findall(f"{{{ns}}}annotation"):
        atype = ann.get("type", "")
        if atype == "truth":
            truth_latex = ann.text
        elif atype == "UI":
            ui = ann.text
        else:
            annotations[atype] = ann.text or ""

    # 4. 提取 annotationXML（MathML 真值）
    truth_mathml: Optional[str] = None
    axml = root.find(f"{{{ns}}}annotationXML")
    if axml is not None:
        truth_mathml = ET.tostring(axml, encoding="unicode")

    # 5. 提取 traceGroup（分段信息）
    trace_groups = _parse_trace_groups(root, ns)

    return InkMLData(
        traces=traces,
        truth_latex=truth_latex,
        truth_mathml=truth_mathml,
        trace_groups=trace_groups,
        annotations=annotations,
        ui=ui,
        file_path=file_path,
    )


def load_dataset(
    data_dir: str | Path,
    file_pattern: str = "*.inkml",
    max_files: Optional[int] = None,
) -> list[InkMLData]:
    """批量加载目录下所有 InkML 文件。

    Args:
        data_dir: 数据目录路径。
        file_pattern: 文件匹配模式（默认 "*.inkml"）。
        max_files: 最大加载文件数，None 表示全部加载。

    Returns:
        InkMLData 列表。
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob(file_pattern))
    if max_files:
        files = files[:max_files]

    results = []
    for f in files:
        try:
            results.append(parse_inkml(f))
        except Exception as e:
            print(f"[WARNING] 解析失败: {f} — {e}")

    return results


def get_trace_stats(data: InkMLData) -> dict:
    """获取笔画统计信息。"""
    lengths = [len(t) for t in data.traces]
    return {
        "num_traces": len(data.traces),
        "total_points": sum(lengths),
        "min_points": min(lengths) if lengths else 0,
        "max_points": max(lengths) if lengths else 0,
        "mean_points": np.mean(lengths) if lengths else 0.0,
    }
