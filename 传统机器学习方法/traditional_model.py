"""
传统机器学习路径：HOG + LBP → StandardScaler → PCA → SVM
- 多组特征参数对比 + 全流程可视化
"""
import os, json, time, numpy as np
from datetime import datetime
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from preprocess import (
    load_filtered, augment_dataset, print_distribution, augment_image
)

# ==================== 全局绘图配置 ====================
FIG_DIR = os.path.join("experiments", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

for _font in ["SimHei", "Microsoft YaHei", "DejaVu Sans"]:
    try:
        rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass

COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [fig] {path}")
    plt.close()

# ==================== 可视化函数 ====================

def plot_sample_grid(X, y, le, top_n=20, seed=42):
    rng = np.random.RandomState(seed)
    cnt = Counter(y)
    top_classes = [lb for lb, _ in cnt.most_common(top_n)]
    n = len(top_classes)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, lb in enumerate(top_classes):
        ax = axes[i]
        idxs = [j for j, yy in enumerate(y) if yy == lb]
        idx = rng.choice(idxs)
        ax.imshow(X[idx], cmap="gray_r")
        short = str(le.inverse_transform([lb])[0])
        ax.set_title(short[:18], fontsize=7)
        ax.axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("CROHME 样本图库 (每类 1 张)", fontsize=12, y=1.01)
    save_fig("01_sample_grid.png")

def plot_augmentation_comparison(X_orig, n_examples=8, seed=42):
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(X_orig), min(n_examples, len(X_orig)), replace=False)
    fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 1.5, 3.5))
    for col, idx in enumerate(idxs):
        axes[0, col].imshow(X_orig[idx], cmap="gray_r")
        axes[0, col].set_title("Original", fontsize=8)
        axes[0, col].axis("off")
        aug = augment_image(X_orig[idx], rng)
        axes[1, col].imshow(aug, cmap="gray_r")
        axes[1, col].set_title("Augmented", fontsize=8)
        axes[1, col].axis("off")
    fig.suptitle("Data Augmentation (Original vs Augmented)", fontsize=12, y=1.02)
    save_fig("02_augmentation_comparison.png")

def plot_class_distribution(y_before, y_after, top_n=25):
    cnt_before = Counter(y_before)
    cnt_after = Counter(y_after)
    top = [lb for lb, _ in cnt_before.most_common(top_n)]
    before_vals = [cnt_before[lb] for lb in top]
    after_vals = [cnt_after.get(lb, 0) for lb in top]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(top))
    w = 0.35
    ax.bar(x - w / 2, before_vals, w, label="Before Aug", color=COLORS[0], edgecolor="white")
    ax.bar(x + w / 2, after_vals, w, label="After Aug", color=COLORS[1], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t)[:15] for t in top], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Samples")
    ax.set_title(f"Class Distribution (Top-{top_n})")
    ax.legend()
    save_fig("03_class_distribution.png")

def plot_hog_visualization(img, orientations=9, ppc=8, cpb=2):
    fd, hog_img = hog(img, orientations=orientations,
                      pixels_per_cell=(ppc, ppc),
                      cells_per_block=(cpb, cpb), visualize=True)
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
    axes[0].imshow(img, cmap="gray_r")
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(hog_img, cmap="hot")
    axes[1].set_title(f"HOG (orient={orientations}, ppc={ppc})", fontsize=10)
    axes[1].axis("off")
    save_fig("04_hog_visualization.png")

def plot_lbp_visualization(img, P=8, R=1, method="uniform"):
    img_u8 = (img * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_u8, P, R, method=method)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    axes[0].imshow(img, cmap="gray_r")
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")
    im = axes[1].imshow(lbp, cmap="nipy_spectral")
    axes[1].set_title(f"LBP (P={P}, R={R})", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    n_bins = int(lbp.max() + 1)
    axes[2].bar(range(n_bins), np.histogram(lbp, bins=n_bins, range=(0, n_bins))[0],
                color=COLORS[0], edgecolor="white")
    axes[2].set_title("LBP Histogram", fontsize=10)
    axes[2].set_xlabel("Pattern")
    save_fig("05_lbp_visualization.png")

def plot_pca_variance(pca_objects, config_names):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pca, name, c in zip(pca_objects, config_names, COLORS):
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumsum) + 1), cumsum, color=c, linewidth=1.5, label=name)
        idx95 = np.searchsorted(cumsum, 0.95)
        ax.axvline(x=idx95 + 1, color=c, linestyle="--", alpha=0.4, linewidth=0.8)
        ax.scatter([idx95 + 1], [cumsum[idx95]], color=c, s=30, zorder=5)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA Cumulative Variance (3 Feature Configs)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(200, ax.get_xlim()[1]))
    ax.grid(True, alpha=0.3)
    save_fig("06_pca_variance.png")

def plot_dimension_comparison(experiments):
    names = [e["name"] for e in experiments]
    raw_dims = [e["raw_dim"] for e in experiments]
    pca_dims = [e["pca_dim"] for e in experiments]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    w = 0.3
    ax.bar(x - w / 2, raw_dims, w, label="Raw Dim", color=COLORS[0], edgecolor="white")
    ax.bar(x + w / 2, pca_dims, w, label="PCA Dim", color=COLORS[3], edgecolor="white")
    for i, (r, p) in enumerate(zip(raw_dims, pca_dims)):
        ax.text(i - w / 2, r + 50, str(r), ha="center", fontsize=8)
        ax.text(i + w / 2, p + 50, str(p), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:20] for n in names], fontsize=8)
    ax.set_ylabel("Dimension")
    ax.set_title("Feature Dimension: Raw vs PCA")
    ax.legend()
    save_fig("07_dimension_comparison.png")

def plot_accuracy_comparison(experiments):
    names = [e["name"] for e in experiments]
    test_accs = [e["test_accuracy"] for e in experiments]
    cv_means = [e["cv_mean"] for e in experiments]
    cv_stds = [e["cv_std"] for e in experiments]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    w = 0.3
    bars1 = ax.bar(x - w / 2, cv_means, w, label="CV Mean", color=COLORS[0],
                   yerr=cv_stds, capsize=5, edgecolor="white")
    bars2 = ax.bar(x + w / 2, test_accs, w, label="Test", color=COLORS[1], edgecolor="white")
    for bar, val in zip(bars1, cv_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, test_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:22] for n in names], fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy (CV vs Test)")
    ax.legend()
    ax.set_ylim(0, max(max(cv_means), max(test_accs)) * 1.15)
    save_fig("08_accuracy_comparison.png")

def plot_kernel_time(experiments):
    names = [e["name"] for e in experiments]
    times = [e["time_s"] for e in experiments]
    kernels = [e["best_kernel"] for e in experiments]
    kernel_colors = {"linear": COLORS[0], "rbf": COLORS[2]}
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = [kernel_colors.get(k, COLORS[3]) for k in kernels]
    bars = ax.bar(range(len(names)), times, color=bar_colors, edgecolor="white")
    for bar, t, k in zip(bars, times, kernels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{t:.0f}s\n{k}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[:22] for n in names], fontsize=8)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time & Best Kernel")
    save_fig("09_kernel_time.png")

def plot_confusion_heatmap(cm, classes, top_n=15):
    diag = np.diag(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    per_class_acc = diag / row_sums.ravel()
    top_idx = np.argsort(per_class_acc)[-top_n:]
    cm_top = cm[np.ix_(top_idx, top_idx)]
    class_top = [str(classes[i])[:15] for i in top_idx]
    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.6), max(8, top_n * 0.55)))
    cm_norm = cm_top.astype("float") / cm_top.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    im = ax.imshow(cm_norm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(class_top)))
    ax.set_yticks(range(len(class_top)))
    ax.set_xticklabels(class_top, rotation=60, ha="right", fontsize=6)
    ax.set_yticklabels(class_top, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Top-{top_n} classes, normalized by row)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    save_fig("10_confusion_heatmap.png")

def plot_misclassification(err_pairs, top_n=15):
    items = err_pairs.most_common(top_n)
    labels = [f"{t[:25]} -> {p[:25]}" for (t, p), _ in items]
    counts = [c for _, c in items]
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    bars = ax.barh(range(len(labels)), counts, color=COLORS[2], edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Misclassification Count")
    ax.set_title("Top Misclassified Pairs (True -> Predicted)")
    ax.invert_yaxis()
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)
    save_fig("11_misclassification.png")

def plot_summary_dashboard(all_results, experiments):
    fig = plt.figure(figsize=(16, 10))
    ds = all_results["dataset"]

    ax1 = fig.add_subplot(2, 3, 1)
    info = [
        f"Total Samples: {ds['n_total']}",
        f"Train: {ds['n_train']}",
        f"Test: {ds['n_test']}",
        f"Classes: {ds['n_classes']}",
        f"Image Size: {ds['img_size']}x{ds['img_size']}",
        f"Min Samples Filter: >= {ds['min_samples_filter']}",
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
    ax2.bar(x - w / 2, [e["cv_mean"] for e in experiments], w, label="CV", color=COLORS[0])
    ax2.bar(x + w / 2, [e["test_accuracy"] for e in experiments], w, label="Test", color=COLORS[1])
    ax2.set_xticks(x)
    ax2.set_xticklabels([n[:14] for n in names], fontsize=7, rotation=20)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Comparison", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(x - w / 2, [e["raw_dim"] for e in experiments], w, label="Raw", color=COLORS[2])
    ax3.bar(x + w / 2, [e["pca_dim"] for e in experiments], w, label="PCA", color=COLORS[3])
    ax3.set_xticks(x)
    ax3.set_xticklabels([n[:14] for n in names], fontsize=7, rotation=20)
    ax3.set_ylabel("Dimension")
    ax3.set_title("Dimension Reduction", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 3, 4)
    bar_colors = [COLORS[0] if e["best_kernel"] == "linear" else COLORS[2] for e in experiments]
    ax4.bar(range(len(names)), [e["time_s"] for e in experiments], color=bar_colors)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n[:14] for n in names], fontsize=7, rotation=20)
    ax4.set_ylabel("Seconds")
    ax4.set_title("Training Time (green=linear, red=rbf)", fontsize=11, fontweight="bold")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.bar(range(len(names)), [e["pca_variance"] for e in experiments], color=COLORS[4])
    ax5.set_xticks(range(len(names)))
    ax5.set_xticklabels([n[:14] for n in names], fontsize=7, rotation=20)
    ax5.set_ylabel("Cumulative Variance Ratio")
    ax5.set_title("PCA Retained Variance", fontsize=12, fontweight="bold")
    ax5.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)

    ax6 = fig.add_subplot(2, 3, 6)
    best = all_results["best"]
    best_lines = [
        f"Best Config: {best['name'][:30]}",
        f"Kernel: {best['best_kernel']}",
        f"Params: {best['best_params']}",
        f"CV: {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}",
        f"Test Acc: {best['test_accuracy']:.4f}",
        f"PCA Dim: {best['pca_dim']}",
        f"Time: {best['time_s']:.0f}s",
    ]
    ax6.text(0.05, 0.95, "\n".join(best_lines), transform=ax6.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace")
    ax6.set_title("Best Model", fontsize=12, fontweight="bold")
    ax6.axis("off")

    fig.suptitle("CROHME Traditional ML Experiment Summary", fontsize=14, fontweight="bold", y=1.01)
    save_fig("12_summary_dashboard.png")


# ==================== 特征提取 ====================

def extract_hog(imgs, orientations=9, ppc=8, cpb=2):
    feats = []
    for img in imgs:
        fd = hog(img, orientations=orientations,
                 pixels_per_cell=(ppc, ppc),
                 cells_per_block=(cpb, cpb), visualize=False)
        feats.append(fd)
    return np.array(feats, dtype=np.float32)

def extract_lbp(imgs, P=8, R=1, method="uniform", n_bins=None):
    feats = []
    for img in imgs:
        img_u8 = (img * 255).astype(np.uint8)
        lbp = local_binary_pattern(img_u8, P, R, method=method)
        n_bins_ = n_bins or int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins_, range=(0, n_bins_))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        feats.append(hist)
    return np.array(feats, dtype=np.float32)

def build_feature_set(X_train, X_test, hog_cfg, lbp_cfg):
    print(f"  HOG: orient={hog_cfg[0]}, ppc={hog_cfg[1]}, cpb={hog_cfg[2]}")
    print(f"  LBP: P={lbp_cfg[0]}, R={lbp_cfg[1]}, method={lbp_cfg[2]}")
    h_train = extract_hog(X_train, *hog_cfg)
    h_test  = extract_hog(X_test, *hog_cfg)
    l_train = extract_lbp(X_train, *lbp_cfg)
    l_test  = extract_lbp(X_test, *lbp_cfg)
    train_feat = np.hstack([h_train, l_train])
    test_feat  = np.hstack([h_test, l_test])
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat)
    test_feat  = scaler.transform(test_feat)
    return train_feat, test_feat, scaler

def apply_pca(train, test, n_components=0.95):
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train)
    test_pca  = pca.transform(test)
    return train_pca, test_pca, pca

# ==================== SVM 训练（轻量版）====================

def train_svm_randomized(train_feat, y_train, cv=5, n_iter=15):
    """
    RandomizedSearchCV: 随机采样参数组合, n_iter 控制搜索量
    - 远快于 GridSearchCV，结果接近
    - LinearSVC + RBF 双核对比
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    total_t0 = time.time()
    results = {}

    # ---- LinearSVC: log-uniform C 采样 ----
    print(f"    [LinearSVC] RandomizedSearch (n_iter={n_iter})...")
    t0 = time.time()
    r1 = RandomizedSearchCV(
        LinearSVC(random_state=42, dual=False, max_iter=5000, class_weight="balanced"),
        {"C": np.logspace(-3, 3, 50)},
        n_iter=n_iter, cv=skf, scoring="accuracy", n_jobs=1, random_state=42, verbose=0,
    )
    r1.fit(train_feat, y_train)
    results["linear"] = {
        "model": r1.best_estimator_,
        "params": str(r1.best_params_),
        "cv_mean": float(r1.best_score_),
        "time_s": round(time.time() - t0, 1),
    }
    print(f"      Best C={r1.best_params_['C']:.4f}, CV={r1.best_score_:.4f}, {results['linear']['time_s']}s")

    # ---- RBF: log-uniform C + gamma 采样 ----
    print(f"    [RBF] RandomizedSearch (n_iter={n_iter})...")
    t0 = time.time()
    r2 = RandomizedSearchCV(
        SVC(kernel="rbf", random_state=42, class_weight="balanced"),
        {"C": np.logspace(-2, 3, 30), "gamma": np.logspace(-3, 1, 30)},
        n_iter=n_iter, cv=skf, scoring="accuracy", n_jobs=1, random_state=42, verbose=0,
    )
    r2.fit(train_feat, y_train)
    results["rbf"] = {
        "model": r2.best_estimator_,
        "params": str(r2.best_params_),
        "cv_mean": float(r2.best_score_),
        "time_s": round(time.time() - t0, 1),
    }
    print(f"      Best {r2.best_params_}, CV={r2.best_score_:.4f}, {results['rbf']['time_s']}s")

    if results["linear"]["cv_mean"] >= results["rbf"]["cv_mean"]:
        best = results["linear"]
        best["kernel"] = "linear"
    else:
        best = results["rbf"]
        best["kernel"] = "rbf"

    total_time = round(time.time() - total_t0, 1)
    print(f"    => Best: {best['kernel']}, CV={best['cv_mean']:.4f}, total {total_time}s")
    return best["model"], best["params"], best["cv_mean"], best["kernel"]

def cross_val_eval(clf, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=1)
    return scores.mean(), scores.std()

def save_report(results, log_dir="experiments"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"report_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {path}")
    return path


# ==================== 主程序 ====================

if __name__ == "__main__":
    DATA_ROOT = r"D:\Users\admin\Desktop\手写数字识别\crohme_data"
    IMG_SIZE = 48
    TOP_N_CLASSES = 30   # 只取样本最多的30类，确保实验可快速完成
    AUG_FACTOR = 2
    AUG_MIN = 60
    PCA_VAR = 0.95
    CV_FOLDS = 3         # 3-fold 足够稳定且快很多
    TEST_RATIO = 0.3
    SEED = 42

    # ============ 1. 加载数据 ============
    print("=" * 60)
    print("  1. Data Loading")
    print("=" * 60)
    t0 = time.time()

    # 先加载全部，再按 top-N 类别筛选
    X_all, y_all, ids_all = load_filtered(DATA_ROOT, min_samples=10, img_size=IMG_SIZE)
    print(f"  全部加载: {len(X_all)} 样本, {len(set(y_all))} 类")

    # 取 Top-N 类
    cnt = Counter(y_all)
    top_labels = {lb for lb, _ in cnt.most_common(TOP_N_CLASSES)}
    keep_idx = [i for i, lb in enumerate(y_all) if lb in top_labels]
    X = np.array([X_all[i] for i in keep_idx])
    y = [y_all[i] for i in keep_idx]

    print_distribution(y, f"Top-{TOP_N_CLASSES} 类")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 可视化 1: 样本图库
    print("\nPlot: sample grid...")
    plot_sample_grid(X, y_enc, le, top_n=20)

    # 增强
    print("\nData augmentation...")
    X_aug, y_aug = augment_dataset(X, y_enc, factor=AUG_FACTOR,
                                   min_per_class=AUG_MIN, seed=SEED)
    print(f"  Before: {len(X)} -> After: {len(X_aug)}")
    print(f"  Classes: {len(np.unique(y_aug))}")

    # 可视化 2+3
    print("Plot: augmentation comparison...")
    plot_augmentation_comparison(X, n_examples=8)
    print("Plot: class distribution...")
    plot_class_distribution(y, [le.inverse_transform([int(i)])[0] for i in y_aug], top_n=25)

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=TEST_RATIO, random_state=SEED, stratify=y_aug
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Data prep time: {time.time() - t0:.1f}s")

    # 可视化 4+5
    print("Plot: HOG visualization...")
    plot_hog_visualization(X_train[0])
    print("Plot: LBP visualization...")
    plot_lbp_visualization(X_train[0])

    # ============ 2. 多组特征实验 ============
    print("\n" + "=" * 60)
    print("  2. Feature Extraction + PCA + SVM (Multi-config)")
    print("=" * 60)

    feature_configs = [
        {
            "name": "HOG(9,8,2)+LBP(8,1,uniform)",
            "hog": (9, 8, 2),
            "lbp": (8, 1, "uniform"),
        },
    ]

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "n_total": len(X_aug),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": len(le.classes_),
            "top_n_selection": TOP_N_CLASSES,
            "img_size": IMG_SIZE,
            "aug_factor": AUG_FACTOR,
            "aug_min_per_class": AUG_MIN,
            "pca_variance_retain": PCA_VAR,
        },
        "experiments": [],
    }

    best_acc = 0
    best_info = None
    pca_objects = []
    pca_config_names = []

    for cfg in feature_configs:
        print(f"\n--- {cfg['name']} ---")
        t1 = time.time()

        train_feat, test_feat, scaler = build_feature_set(
            X_train, X_test, cfg["hog"], cfg["lbp"]
        )
        print(f"  Raw feature dim: {train_feat.shape[1]}")

        train_pca, test_pca, pca = apply_pca(train_feat, test_feat, n_components=PCA_VAR)
        print(f"  PCA dim: {train_pca.shape[1]} (variance: {pca.explained_variance_ratio_.sum():.4f})")
        pca_objects.append(pca)
        pca_config_names.append(cfg["name"])

        best_svm, best_params_str, cv_best, kernel = train_svm_randomized(
            train_pca, y_train, cv=CV_FOLDS, n_iter=8
        )

        cv_mean, cv_std = cross_val_eval(best_svm, train_pca, y_train, cv=CV_FOLDS)
        pred = best_svm.predict(test_pca)
        test_acc = accuracy_score(y_test, pred)

        exp = {
            "name": cfg["name"],
            "hog": {"orient": cfg["hog"][0], "ppc": cfg["hog"][1], "cpb": cfg["hog"][2]},
            "lbp": {"P": cfg["lbp"][0], "R": cfg["lbp"][1], "method": cfg["lbp"][2]},
            "raw_dim": int(train_feat.shape[1]),
            "pca_dim": int(pca.n_components_),
            "pca_variance": float(pca.explained_variance_ratio_.sum()),
            "best_kernel": kernel,
            "best_params": best_params_str,
            "cv_grid_best": round(float(cv_best), 4),
            "cv_mean": round(float(cv_mean), 4),
            "cv_std": round(float(cv_std), 4),
            "test_accuracy": round(float(test_acc), 4),
            "time_s": round(time.time() - t1, 1),
        }
        all_results["experiments"].append(exp)

        print(f"  Grid CV Best: {cv_best:.4f}")
        print(f"  Final CV:     {cv_mean:.4f} +/- {cv_std:.4f}")
        print(f"  Test Acc:     {test_acc:.4f}")
        print(f"  Best Kernel:  {kernel} | {best_params_str}")
        print(f"  Time: {exp['time_s']}s")

        if test_acc > best_acc:
            best_acc = test_acc
            best_info = exp
            best_clf = best_svm
            best_pca_obj = pca
            y_test_best = y_test
            pred_best = pred

    # ============ 3. 实验对比可视化 ============
    print("\nPlot: PCA variance curves...")
    plot_pca_variance(pca_objects, pca_config_names)
    print("Plot: dimension comparison...")
    plot_dimension_comparison(all_results["experiments"])
    print("Plot: accuracy comparison...")
    plot_accuracy_comparison(all_results["experiments"])
    print("Plot: kernel + time...")
    plot_kernel_time(all_results["experiments"])

    # ============ 4. 最优模型详细评估 ============
    print("\n" + "=" * 60)
    print("  3. Best Model Report")
    print("=" * 60)
    print(f"  Config:      {best_info['name']}")
    print(f"  Kernel:      {best_info['best_kernel']} / {best_info['best_params']}")
    print(f"  Test Acc:    {best_info['test_accuracy']:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test_best, pred_best,
                                target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_test_best, pred_best)
    print(f"Confusion Matrix: {cm.shape}, correct on diagonal: {np.trace(cm)} / {cm.sum()}")

    print("Plot: confusion matrix heatmap...")
    plot_confusion_heatmap(cm, le.classes_, top_n=min(15, TOP_N_CLASSES))

    errors = [(i, y_test_best[i], pred_best[i])
              for i in range(len(y_test_best)) if y_test_best[i] != pred_best[i]]
    print(f"\nMisclassified: {len(errors)} / {len(y_test_best)}")
    if errors:
        err_pairs = Counter((le.inverse_transform([t])[0],
                             le.inverse_transform([p])[0])
                            for _, t, p in errors)
        print("Top-10 errors (True -> Pred):")
        for (t, p), c in err_pairs.most_common(10):
            print(f"  {t[:40]} -> {p[:40]}: {c}")
        print("Plot: misclassification analysis...")
        plot_misclassification(err_pairs, top_n=min(15, len(err_pairs)))

    # ============ 5. 综合仪表盘 ============
    all_results["best"] = best_info
    print("\nPlot: summary dashboard...")
    plot_summary_dashboard(all_results, all_results["experiments"])

    save_report(all_results)

    print(f"\nAll figures saved to: {os.path.abspath(FIG_DIR)}")
    print("Done.")
