"""预测效果可视化：原始图片 + 真实公式 + 预测公式"""
import os, warnings, numpy as np
from collections import Counter
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from preprocess import load_from_folder, augment_dataset
from skimage.feature import hog, local_binary_pattern

for _f in ["SimHei", "Microsoft YaHei", "DejaVu Sans"]:
    try:
        rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        break
    except: pass

# ============ 配置 ============
DATA_ROOT = r"D:\Users\admin\Desktop\手写数字识别\crohme_data"
TOP_N = 15
IMG_SIZE = 48
SEED = 42

# ============ 1. 加载数据 ============
print("Loading data...")
X_all, y_all, _ = load_from_folder(DATA_ROOT, img_size=IMG_SIZE)
cnt = Counter(y_all)
top_labels = {lb for lb, _ in cnt.most_common(TOP_N)}
idx = [i for i, lb in enumerate(y_all) if lb in top_labels]
X = np.array([X_all[i] for i in idx])
y = [y_all[i] for i in idx]
print(f"Top-{TOP_N}: {len(X)} samples")

le = LabelEncoder()
y_enc = le.fit_transform(y)

# 数据增强
X_aug, y_aug = augment_dataset(X, y_enc, factor=2, min_per_class=60, seed=SEED)

# 划分
X_tr, X_te, y_tr, y_te = train_test_split(
    X_aug, y_aug, test_size=0.3, random_state=SEED, stratify=y_aug
)
y_tr = np.asarray(y_tr, dtype=int)
y_te = np.asarray(y_te, dtype=int)
print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

# ============ 2. 特征提取 ============
print("Extracting features...")
def get_hog(imgs):
    return np.array([hog(im, orientations=9, pixels_per_cell=(8,8),
                         cells_per_block=(2,2), visualize=False)
                     for im in imgs], dtype=np.float32)

def get_lbp(imgs):
    feats = []
    for im in imgs:
        u8 = (im * 255).astype(np.uint8)
        lbp = local_binary_pattern(u8, 8, 1, "uniform")
        h, _ = np.histogram(lbp, bins=10, range=(0,10))
        feats.append(h.astype(np.float32) / (h.sum() + 1e-6))
    return np.array(feats, dtype=np.float32)

h_tr, h_te = get_hog(X_tr), get_hog(X_te)
l_tr, l_te = get_lbp(X_tr), get_lbp(X_te)

train_feat = np.hstack([h_tr, l_tr])
test_feat  = np.hstack([h_te, l_te])
sc = StandardScaler()
train_feat = sc.fit_transform(train_feat)
test_feat  = sc.transform(test_feat)

pca = PCA(n_components=0.95, random_state=SEED)
train_pca = pca.fit_transform(train_feat)
test_pca  = pca.transform(test_feat)
print(f"Features: {train_feat.shape[1]} -> PCA: {train_pca.shape[1]}")

# ============ 3. 训练（用已知最优参数）============
print("Training LinearSVC...")
clf = LinearSVC(C=0.039, random_state=SEED, dual=False, max_iter=5000, class_weight="balanced")
clf.fit(train_pca, y_tr)

pred = clf.predict(test_pca)
acc = (pred == y_te).mean()
print(f"Test Accuracy: {acc:.4f}")

# ============ 4. 可视化预测结果 ============
print("Creating prediction visualization...")

# 每个类别选2个：1个正确预测 + 1个错误预测（如果有）
n_classes = TOP_N
samples_per_class = []

for cls in range(n_classes):
    cls_idx = np.where(y_te == cls)[0]
    if len(cls_idx) == 0:
        continue
    correct = cls_idx[pred[cls_idx] == cls]
    wrong   = cls_idx[pred[cls_idx] != cls]
    chosen = []
    if len(correct) > 0:
        chosen.append(correct[0])
    if len(wrong) > 0:
        chosen.append(wrong[0])
    samples_per_class.append((cls, chosen))

# 绘图
n_cols = 6  # 每行 3 对（每对 = 原图 | 标签信息）
n_rows_per_pair = 2  # 原图行 + 标签行
total_items = sum(len(chosen) for _, chosen in samples_per_class)
items_per_row = n_cols // 2

fig = plt.figure(figsize=(n_cols * 2.2, total_items * 1.8))

plot_idx = 0
for cls, chosen in samples_per_class:
    for idx in chosen:
        true_label = le.inverse_transform([y_te[idx]])[0]
        pred_label = le.inverse_transform([pred[idx]])[0]
        is_correct = (pred[idx] == y_te[idx])
        status = "CORRECT" if is_correct else "WRONG"
        color = "#2ecc71" if is_correct else "#e74c3c"

        # 左：原图
        ax_img = plt.subplot(total_items, 2, 2*plot_idx + 1)
        ax_img.imshow(X_te[idx], cmap="gray_r")
        ax_img.set_title(f"[{status}]", fontsize=10, color=color, fontweight="bold")
        ax_img.axis("off")

        # 右：标签对比
        ax_label = plt.subplot(total_items, 2, 2*plot_idx + 2)
        ax_label.axis("off")
        text = (
            f"True:\n  ${true_label}$\n\n"
            f"Pred:\n  ${pred_label}$"
        )
        if is_correct:
            text = f"True & Pred:\n  ${true_label}$"
        ax_label.text(0.05, 0.5, text,
                      transform=ax_label.transAxes, fontsize=10,
                      verticalalignment="center", fontfamily="monospace",
                      color=color)

        plot_idx += 1

fig.suptitle(f"CROHME Prediction Results (Top-{TOP_N}, Acc={acc:.2%})",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
os.makedirs("experiments/figures", exist_ok=True)
path = "experiments/figures/13_prediction_samples.png"
plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {path}")
plt.close()

# ============ 5. 补充：正确/错误统计图 ============
per_class_acc = []
for cls in range(n_classes):
    mask = y_te == cls
    if mask.sum() > 0:
        per_class_acc.append(pred[mask].sum() / mask.sum())

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(per_class_acc))
bars = ax.bar(x, per_class_acc, color=["#2ecc71" if a >= 0.5 else "#e74c3c" for a in per_class_acc],
              edgecolor="white")
ax.axhline(y=acc, color="gray", linestyle="--", alpha=0.6, label=f"Overall {acc:.2%}")
ax.set_xticks(x)
ax.set_xticklabels([f"{le.classes_[i][:20]}" for i in range(len(per_class_acc))],
                   rotation=60, ha="right", fontsize=7)
for bar, a in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{a:.0%}", ha="center", fontsize=8)
ax.set_ylabel("Accuracy")
ax.set_title(f"Per-Class Accuracy (Top-{TOP_N})")
ax.set_ylim(0, 1.1)
ax.legend()
path2 = "experiments/figures/14_per_class_accuracy.png"
plt.savefig(path2, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {path2}")
plt.close()

print("Done.")
