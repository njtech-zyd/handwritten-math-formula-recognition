"""
快速测试：CROHME 2011 数据 → HOG+LBP → PCA → SVM
Top-10 类 + 轻量 RandomizedSearch，几分钟出结果
"""
import os, json, time, warnings, numpy as np
from datetime import datetime
from collections import Counter

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
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import load_from_folder, augment_dataset, augment_image

# ============ 配置 ============
DATA_ROOT = r"D:\Users\admin\Desktop\手写数字识别\crohme_data"  # 遍历全部有 truth 标签的数据
FIG_DIR  = os.path.join("experiments", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("experiments", exist_ok=True)

for _f in ["SimHei", "Microsoft YaHei", "DejaVu Sans"]:
    try:
        rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass

C = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
SEED = 42
np.random.seed(SEED)

def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [OK] {path}")
    plt.close()

# ============ 特征 ============
def get_hog(imgs, o=9, p=8, b=2):
    return np.array([hog(im, orientations=o, pixels_per_cell=(p, p),
                         cells_per_block=(b, b), visualize=False)
                     for im in imgs], dtype=np.float32)

def get_lbp(imgs, P=8, R=1, method="uniform"):
    f = []
    for im in imgs:
        u8 = (im * 255).astype(np.uint8)
        lbp = local_binary_pattern(u8, P, R, method=method)
        n = int(lbp.max() + 1)
        h, _ = np.histogram(lbp, bins=n, range=(0, n))
        f.append(h.astype(np.float32) / (h.sum() + 1e-6))
    return np.array(f, dtype=np.float32)

# ============ 主流程 ============
print("=" * 60)
print("  CROHME 2011 快速测试")
print("=" * 60)
t_total = time.time()

# ---------- 1. 加载 ----------
print("\n[1/6] 加载 2011 年数据...")
X_all, y_all, _ = load_from_folder(DATA_ROOT, img_size=64)
print(f"  总样本: {len(X_all)}, 类别: {len(set(y_all))}")

# Top-10 类
cnt = Counter(y_all)
top_labels = {lb for lb, _ in cnt.most_common(10)}
idx = [i for i, lb in enumerate(y_all) if lb in top_labels]
X = np.array([X_all[i] for i in idx])
y = [y_all[i] for i in idx]
print(f"  Top-10 类: {len(X)} 样本")

le = LabelEncoder()
y_enc = le.fit_transform(y)

# ---------- 2. 增强 ----------
print("\n[2/6] 数据增强...")
X_aug, y_aug = augment_dataset(X, y_enc, factor=2, min_per_class=60, seed=SEED)
print(f"  {len(X)} → {len(X_aug)} 样本")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_aug, y_aug, test_size=0.3, random_state=SEED, stratify=y_aug
)
print(f"  训练: {len(X_tr)}, 测试: {len(X_te)}")

# ---------- 3. 特征 ----------
print("\n[3/6] HOG + LBP 特征提取...")

# 只用一组最稳定的配置
h_tr = get_hog(X_tr, 9, 8, 2)
h_te = get_hog(X_te, 9, 8, 2)
l_tr = get_lbp(X_tr, 8, 1, "uniform")
l_te = get_lbp(X_te, 8, 1, "uniform")

print(f"  HOG 维度: {h_tr.shape[1]}, LBP 维度: {l_tr.shape[1]}")

# 拼接 + 标准化
train_feat = np.hstack([h_tr, l_tr])
test_feat  = np.hstack([h_te, l_te])
sc = StandardScaler()
train_feat = sc.fit_transform(train_feat)
test_feat  = sc.transform(test_feat)
print(f"  总维度: {train_feat.shape[1]}")

# ---------- 4. PCA ----------
print("\n[4/6] PCA 降维...")
pca = PCA(n_components=0.95, random_state=SEED)
train_pca = pca.fit_transform(train_feat)
test_pca  = pca.transform(test_feat)
print(f"  降维后: {train_pca.shape[1]} (保留方差: {pca.explained_variance_ratio_.sum():.4f})")

# ---------- 5. SVM ----------
print("\n[5/6] SVM 调参 (RandomizedSearchCV, n_iter=8, cv=3)...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

t0 = time.time()
print("  [LinearSVC] ...")
r1 = RandomizedSearchCV(
    LinearSVC(random_state=SEED, dual=False, max_iter=5000, class_weight="balanced"),
    {"C": np.logspace(-3, 3, 30)},
    n_iter=8, cv=skf, scoring="accuracy", n_jobs=1, random_state=SEED,
)
r1.fit(train_pca, y_tr)
lin_best = r1.best_estimator_
print(f"    Best C={r1.best_params_['C']:.4f}, CV={r1.best_score_:.4f}, {time.time()-t0:.0f}s")

t0 = time.time()
print("  [RBF] ...")
r2 = RandomizedSearchCV(
    SVC(kernel="rbf", random_state=SEED, class_weight="balanced"),
    {"C": np.logspace(-2, 3, 20), "gamma": np.logspace(-3, 1, 20)},
    n_iter=8, cv=skf, scoring="accuracy", n_jobs=1, random_state=SEED,
)
r2.fit(train_pca, y_tr)
rbf_best = r2.best_estimator_
print(f"    Best {r2.best_params_}, CV={r2.best_score_:.4f}, {time.time()-t0:.0f}s")

# 选最优
if r1.best_score_ >= r2.best_score_:
    best_clf, best_kernel, best_params = lin_best, "linear", str(r1.best_params_)
else:
    best_clf, best_kernel, best_params = rbf_best, "rbf", str(r2.best_params_)
print(f"  => 最优: {best_kernel} | {best_params}")

# CV 评估
cv_scores = cross_val_score(best_clf, train_pca, y_tr, cv=skf, scoring="accuracy", n_jobs=1)
cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
print(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")

# 测试
pred = best_clf.predict(test_pca)
test_acc = accuracy_score(y_te, pred)
print(f"  测试准确率: {test_acc:.4f}")

# ---------- 6. 可视化 + 报告 ----------
print("\n[6/6] 生成可视化...")

# Fig 1: 样本图库
print("  样本图库...")
n_cls = len(le.classes_)
cols = min(5, n_cls)
rows = int(np.ceil(n_cls / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
axes = axes.flatten() if rows * cols > 1 else [axes]
for i in range(n_cls):
    idx_i = [j for j, yy in enumerate(y_enc) if yy == i]
    axes[i].imshow(X[idx_i[0]], cmap="gray_r")
    axes[i].set_title(str(le.classes_[i])[:15], fontsize=6)
    axes[i].axis("off")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
fig.suptitle("CROHME 2011 Top-10 样本", fontsize=11, y=1.01)
save_fig("01_sample_grid.png")

# Fig 2: 增强对比
print("  增强对比...")
rng = np.random.RandomState(SEED)
idxs = rng.choice(len(X), min(6, len(X)), replace=False)
fig, axes = plt.subplots(2, 6, figsize=(9, 3.5))
for col, ii in enumerate(idxs):
    axes[0, col].imshow(X[ii], cmap="gray_r")
    axes[0, col].set_title("Original", fontsize=7)
    axes[0, col].axis("off")
    axes[1, col].imshow(augment_image(X[ii], rng), cmap="gray_r")
    axes[1, col].set_title("Augmented", fontsize=7)
    axes[1, col].axis("off")
fig.suptitle("Data Augmentation", fontsize=11, y=1.02)
save_fig("02_augmentation.png")

# Fig 3: 类别分布
print("  类别分布...")
cnt_b = Counter(y)
cnt_a = Counter([le.inverse_transform([int(i)])[0] for i in y_aug])
top = [lb for lb, _ in cnt_b.most_common(10)]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(top))
w = 0.35
ax.bar(x - w / 2, [cnt_b[lb] for lb in top], w, label="Before Aug", color=C[0])
ax.bar(x + w / 2, [cnt_a.get(lb, 0) for lb in top], w, label="After Aug", color=C[1])
ax.set_xticks(x); ax.set_xticklabels([str(t)[:12] for t in top], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Samples"); ax.set_title("Class Distribution"); ax.legend()
save_fig("03_class_distribution.png")

# Fig 4: HOG 可视化
print("  HOG 可视化...")
_, hog_img = hog(X_tr[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5, 2.8))
a1.imshow(X_tr[0], cmap="gray_r"); a1.set_title("Original", fontsize=9); a1.axis("off")
a2.imshow(hog_img, cmap="hot"); a2.set_title("HOG", fontsize=9); a2.axis("off")
save_fig("04_hog.png")

# Fig 5: LBP 可视化
print("  LBP 可视化...")
u8 = (X_tr[0] * 255).astype(np.uint8)
lbp = local_binary_pattern(u8, 8, 1, "uniform")
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7, 2.8))
a1.imshow(X_tr[0], cmap="gray_r"); a1.set_title("Original", fontsize=9); a1.axis("off")
im = a2.imshow(lbp, cmap="nipy_spectral"); a2.set_title("LBP", fontsize=9); a2.axis("off")
plt.colorbar(im, ax=a2, fraction=0.046)
a3.bar(range(10), np.histogram(lbp, bins=10, range=(0, 10))[0], color=C[0])
a3.set_title("LBP Histogram", fontsize=9)
save_fig("05_lbp.png")

# Fig 6: PCA 累计方差
print("  PCA 方差曲线...")
fig, ax = plt.subplots(figsize=(7, 3.5))
cs = np.cumsum(pca.explained_variance_ratio_)
ax.plot(range(1, len(cs) + 1), cs, color=C[1], linewidth=1.5)
idx95 = np.searchsorted(cs, 0.95)
ax.axvline(x=idx95 + 1, color="gray", linestyle="--", alpha=0.5)
ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
ax.scatter([idx95 + 1], [cs[idx95]], color=C[2], s=40, zorder=5)
ax.set_xlabel("Components"); ax.set_ylabel("Cumulative Variance")
ax.set_title(f"PCA (95% at {idx95 + 1} components)"); ax.grid(True, alpha=0.3)
save_fig("06_pca.png")

# Fig 7: Confusion Matrix Heatmap
print("  混淆矩阵热力图...")
cm = confusion_matrix(y_te, pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
ax.set_xticklabels([str(c)[:12] for c in le.classes_], rotation=60, ha="right", fontsize=6)
ax.set_yticklabels([str(c)[:12] for c in le.classes_], fontsize=6)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix (normalized)")
plt.colorbar(im, ax=ax, fraction=0.046)
save_fig("07_confusion.png")

# Fig 8: 模型对比
print("  模型对比...")
fig, ax = plt.subplots(figsize=(6, 4))
kernels = ["LinearSVC", "RBF"]
cv_vals = [r1.best_score_, r2.best_score_]
test_vals = [accuracy_score(y_te, lin_best.predict(test_pca)),
             accuracy_score(y_te, rbf_best.predict(test_pca))]
x = np.arange(2); w = 0.3
ax.bar(x - w / 2, cv_vals, w, label="CV", color=C[0])
ax.bar(x + w / 2, test_vals, w, label="Test", color=C[1])
for i, (cv, ts) in enumerate(zip(cv_vals, test_vals)):
    ax.text(i - w / 2, cv + 0.01, f"{cv:.3f}", ha="center", fontsize=10)
    ax.text(i + w / 2, ts + 0.01, f"{ts:.3f}", ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(kernels)
ax.set_ylabel("Accuracy"); ax.set_title("Kernel Comparison"); ax.legend()
save_fig("08_kernel_comparison.png")

# Report
print("\n" + "=" * 60)
print("  测试结果")
print("=" * 60)
print(f"  数据: CROHME 2011, Top-10 类")
print(f"  样本: {len(X_aug)} (训练 {len(X_tr)}, 测试 {len(X_te)})")
print(f"  特征: HOG(9,8,2) + LBP(8,1,uniform)")
print(f"  特征维度: {train_feat.shape[1]} → PCA: {train_pca.shape[1]}")
print(f"  最佳核: {best_kernel}")
print(f"  最佳参数: {best_params}")
print(f"  CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  总耗时: {time.time() - t_total:.0f}s")
print(f"\n  图表目录: {os.path.abspath(FIG_DIR)}")

print("\n--- Classification Report ---")
print(classification_report(y_te, pred, target_names=le.classes_, zero_division=0))

# 保存 JSON
report = {
    "timestamp": datetime.now().isoformat(),
    "data": "CROHME 2011 Top-10",
    "n_samples": len(X_aug),
    "n_train": len(X_tr),
    "n_test": len(X_te),
    "n_classes": n_cls,
    "feature_dim_raw": int(train_feat.shape[1]),
    "feature_dim_pca": int(train_pca.shape[1]),
    "best_kernel": best_kernel,
    "best_params": best_params,
    "cv_mean": float(cv_mean),
    "cv_std": float(cv_std),
    "test_accuracy": float(test_acc),
    "linear_cv": float(r1.best_score_),
    "rbf_cv": float(r2.best_score_),
    "time_s": round(time.time() - t_total, 1),
}
path = os.path.join("experiments", f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n报告: {path}")
print("Done.")
