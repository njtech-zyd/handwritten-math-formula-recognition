"""单个样本演示：InkML → 图像 → 符号拆分"""
import re, os, glob
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import parse_inkml, render_strokes

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

DATA = r"d:\Users\admin\Desktop\手写数字识别\crohme_data"
files = glob.glob(os.path.join(DATA, "**", "*.inkml"), recursive=True)

# 找一个有 truth 标签的文件
found = None
for f in files:
    strokes, label, uid = parse_inkml(f)
    if label:
        found = (f, strokes, label, uid)
        break

fname, strokes, label, uid = found

# ===== 1. 原始数据 =====
print("=" * 50)
print("  原始数据")
print("=" * 50)
print(f"文件: {os.path.basename(fname)}")
print(f"UID:  {uid}")
print(f"笔画数: {len(strokes)}")
print(f"总点数: {sum(len(s) for s in strokes)}")
print(f"标签:  ${label}$")

print("\n前3个笔画 (前5个点):")
for i, s in enumerate(strokes[:3]):
    pts = ", ".join(f"({x:.0f},{y:.0f})" for x, y in s[:5])
    print(f"  笔画{i}: {len(s)}点 -> {pts}...")

# ===== 2. 渲染图 =====
img = render_strokes(strokes, img_size=64)

# ===== 3. LaTeX 符号拆分 =====
clean = label.strip("$").strip()
tokens = re.findall(
    r'\\[a-zA-Z]+|\\[^a-zA-Z]|[a-zA-Z0-9]+|\+|\-|\*|\=|\<|\>|\{|\}|\[|\]|\(|\)|\^|\_',
    clean
)
print(f"\n符号拆分 ({len(tokens)} tokens):")
print(f"  {' '.join(tokens)}")

token_cnt = Counter(tokens)
print(f"\n去重后 {len(token_cnt)} 种符号:")
for t, c in token_cnt.most_common():
    print(f"    {t}: {c}次")

# ===== 4. 可视化 =====
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
axes[0].imshow(img, cmap="gray_r")
axes[0].set_title("Rendered (64x64)", fontsize=11)
axes[0].axis("off")

sym_text = "\n".join(f"  {t} x{c}" for t, c in token_cnt.most_common(18))
axes[1].text(0.05, 0.95,
    f"LaTeX:\n  ${label}$\n\n"
    f"Tokens ({len(tokens)}):\n  {' '.join(tokens)}\n\n"
    f"Unique symbols ({len(token_cnt)}):\n{sym_text}",
    transform=axes[1].transAxes, fontsize=10,
    verticalalignment="top", fontfamily="monospace")
axes[1].axis("off")

fig.suptitle("Single Sample Demo: InkML -> Image + Symbol Extraction",
             fontsize=13, fontweight="bold")
plt.savefig("demo_sample.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n图表: demo_sample.png")
print("Done.")
