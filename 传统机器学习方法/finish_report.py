"""补全上一次实验：生成 fig12 综合仪表盘 + report JSON"""
import os, json, numpy as np
from datetime import datetime
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings; warnings.filterwarnings("ignore")

for _f in ["SimHei", "Microsoft YaHei", "DejaVu Sans"]:
    try:
        rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        break
    except: pass

FIG_DIR = os.path.join("experiments", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
C = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

# ---- 已知实验数据 ----
experiments = [{
    "name": "HOG(9,8,2)+LBP(8,1,uniform)",
    "hog": {"orient": 9, "ppc": 8, "cpb": 2},
    "lbp": {"P": 8, "R": 1, "method": "uniform"},
    "raw_dim": 910,
    "pca_dim": 303,
    "pca_variance": 0.9500,
    "best_kernel": "linear",
    "best_params": "{'C': 0.0391}",
    "cv_grid_best": 0.6699,
    "cv_mean": 0.6699,
    "cv_std": 0.0018,
    "test_accuracy": 0.6873,
    "time_s": 2043.0,
}]

all_results = {
    "timestamp": datetime.now().isoformat(),
    "dataset": {
        "n_total": 7874, "n_train": 5511, "n_test": 2363,
        "n_classes": 30, "img_size": 48,
        "min_samples_filter": 10, "aug_factor": 2,
        "aug_min_per_class": 60, "pca_variance_retain": 0.95,
    },
    "experiments": experiments,
}

best = experiments[0]
all_results["best"] = best

# ---- Fig 12: 综合仪表盘 ----
print("Generating summary dashboard...")
fig = plt.figure(figsize=(16, 10))
ds = all_results["dataset"]

ax1 = fig.add_subplot(2, 3, 1)
info = [
    f"Total Samples: {ds['n_total']}",
    f"Train: {ds['n_train']}",
    f"Test: {ds['n_test']}",
    f"Classes: {ds['n_classes']}",
    f"Image Size: {ds['img_size']}x{ds['img_size']}",
    f"Min Filter: >= {ds['min_samples_filter']}",
    f"Aug Factor: {ds['aug_factor']}x",
    f"PCA Variance: {ds['pca_variance_retain']}",
]
ax1.text(0.05, 0.95, "\n".join(info), transform=ax1.transAxes,
         fontsize=11, verticalalignment="top", fontfamily="monospace")
ax1.set_title("Dataset Overview", fontsize=12, fontweight="bold")
ax1.axis("off")

ax2 = fig.add_subplot(2, 3, 2)
names = [e["name"] for e in experiments]
x = np.arange(len(names))
w = 0.25
ax2.bar(x - w / 2, [e["cv_mean"] for e in experiments], w, label="CV", color=C[0])
ax2.bar(x + w / 2, [e["test_accuracy"] for e in experiments], w, label="Test", color=C[1])
ax2.set_xticks(x)
ax2.set_xticklabels([n[:14] for n in names], fontsize=7, rotation=20)
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Comparison", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8)
ax2.set_ylim(0, 0.85)

ax3 = fig.add_subplot(2, 3, 3)
ax3.bar(x - w / 2, [e["raw_dim"] for e in experiments], w, label="Raw", color=C[2])
ax3.bar(x + w / 2, [e["pca_dim"] for e in experiments], w, label="PCA", color=C[3])
ax3.set_xticks(x)
ax3.set_xticklabels([n[:14] for n in names], fontsize=7, rotation=20)
ax3.set_ylabel("Dimension")
ax3.set_title("Dimension Reduction", fontsize=12, fontweight="bold")
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(2, 3, 4)
bar_c = [C[0] if e["best_kernel"] == "linear" else C[2] for e in experiments]
ax4.bar(range(len(names)), [e["time_s"] for e in experiments], color=bar_c)
ax4.set_xticks(range(len(names)))
ax4.set_xticklabels([n[:14] for n in names], fontsize=7, rotation=20)
ax4.set_ylabel("Seconds")
ax4.set_title("Training Time (green=linear, red=rbf)", fontsize=11, fontweight="bold")

ax5 = fig.add_subplot(2, 3, 5)
ax5.bar(range(len(names)), [e["pca_variance"] for e in experiments], color=C[4])
ax5.set_xticks(range(len(names)))
ax5.set_xticklabels([n[:14] for n in names], fontsize=7, rotation=20)
ax5.set_ylabel("Cumulative Variance")
ax5.set_title("PCA Retained Variance", fontsize=12, fontweight="bold")
ax5.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
ax5.set_ylim(0, 1.0)

ax6 = fig.add_subplot(2, 3, 6)
best_lines = [
    f"Config: {best['name'][:30]}",
    f"Kernel: {best['best_kernel']}",
    f"Params: C=0.03907",
    f"CV: {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}",
    f"Test Acc: {best['test_accuracy']:.4f} (68.73%)",
    f"Raw Dim: {best['raw_dim']} -> PCA: {best['pca_dim']}",
    f"Time: {best['time_s']:.0f}s (34 min)",
]
ax6.text(0.05, 0.95, "\n".join(best_lines), transform=ax6.transAxes,
         fontsize=11, verticalalignment="top", fontfamily="monospace")
ax6.set_title("Best Model", fontsize=12, fontweight="bold")
ax6.axis("off")

fig.suptitle("CROHME Traditional ML Experiment Summary", fontsize=14, fontweight="bold", y=1.01)
path = os.path.join(FIG_DIR, "12_summary_dashboard.png")
plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"  [OK] {path}")
plt.close()

# ---- Save Report JSON ----
print("Saving report...")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = os.path.join("experiments", f"report_{ts}.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"  [OK] {path}")

print("Done. All 12 figures + report complete.")
