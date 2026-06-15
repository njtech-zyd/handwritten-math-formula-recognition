import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter

NS = {"ink": "http://www.w3.org/2003/InkML"}

# ==================== InkML 解析 ====================

def parse_inkml(filepath):
    """解析 InkML 文件，返回 (笔画列表, 标签, uid)"""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception:
        return [], "", ""

    uid = root.get("documentID", os.path.basename(filepath))

    strokes = []
    for trace in root.findall(".//ink:trace", NS) + root.findall(".//trace"):
        text = (trace.text or "").strip()
        if not text:
            continue
        points = []
        for token in text.split(","):
            coords = token.strip().split()
            if len(coords) < 2:
                continue
            try:
                x, y = float(coords[0]), float(coords[1])
                points.append((x, y))
            except ValueError:
                continue
        if len(points) >= 2:
            strokes.append(points)

    label = ""
    for ann in root.findall(".//ink:annotation", NS) + root.findall(".//annotation"):
        if ann.get("type") == "truth":
            text = (ann.text or "").strip()
            if text.startswith("$") and text.endswith("$"):
                text = text[1:-1]
            label = text
            break

    return strokes, label, uid


# ==================== 渲染 ====================

def render_strokes(strokes, img_size=64, canvas_size=256, padding=10, thickness=2):
    """
    将笔画渲染为灰度图像。
    - 自适应缩放 + 居中，保持宽高比
    - 抗锯齿线条
    """
    if not strokes:
        return np.zeros((img_size, img_size), dtype=np.float32)

    all_pts = [p for s in strokes for p in s]
    xs = np.array([p[0] for p in all_pts])
    ys = np.array([p[1] for p in all_pts])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w, h = x_max - x_min, y_max - y_min
    if w < 1e-6 and h < 1e-6:
        return np.zeros((img_size, img_size), dtype=np.float32)

    scale = (canvas_size - 2 * padding) / max(w, h)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    for s in strokes:
        pts = np.array([
            [int((x - cx) * scale + canvas_size / 2),
             int((y - cy) * scale + canvas_size / 2)]
            for x, y in s
        ], dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), 255, thickness, cv2.LINE_AA)

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


# ==================== 数据增强 ====================

def _rand_float(lo, hi, rng):
    return rng.random() * (hi - lo) + lo


def augment_image(img, rng=None):
    """
    单张图像随机增强：
    - ±10° 旋转
    - ±15% 缩放
    - ±2px 平移
    - 随机 elastic deformation（小幅度）
    返回增强后的图像 (float32, 0~1)
    """
    if rng is None:
        rng = np.random.RandomState()

    h, w = img.shape
    img_u8 = (img * 255).astype(np.uint8)

    # 随机旋转
    angle = _rand_float(-10, 10, rng)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 随机缩放
    scale = _rand_float(0.85, 1.15, rng)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 随机平移
    dx = int(_rand_float(-2, 2, rng) * w / 64)
    dy = int(_rand_float(-2, 2, rng) * h / 64)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 轻微 elastic deformation
    if rng.random() < 0.5:
        alpha = img_u8.shape[0] * _rand_float(0.5, 1.5, rng)
        sigma = img_u8.shape[0] * _rand_float(0.04, 0.06, rng)
        dx_field = rng.randn(*img_u8.shape).astype(np.float32) * sigma
        dy_field = rng.randn(*img_u8.shape).astype(np.float32) * sigma
        dx_field = cv2.GaussianBlur(dx_field, (0, 0), sigmaX=sigma, sigmaY=sigma)
        dy_field = cv2.GaussianBlur(dy_field, (0, 0), sigmaX=sigma, sigmaY=sigma)
        x, y = np.meshgrid(np.arange(img_u8.shape[1]), np.arange(img_u8.shape[0]))
        map_x = (x + alpha * dx_field).astype(np.float32)
        map_y = (y + alpha * dy_field).astype(np.float32)
        img_u8 = cv2.remap(img_u8, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return img_u8.astype(np.float32) / 255.0


def augment_dataset(images, labels, factor=2, min_per_class=50, seed=42):
    """
    对数据集的每一类做增强。
    - factor: 每类最少样本数 = max(min_per_class, 当前样本数 * factor)
    - 保持类别平衡
    返回 (augmented_images, augmented_labels)
    """
    rng = np.random.RandomState(seed)
    label_to_imgs = {}
    for img, lb in zip(images, labels):
        label_to_imgs.setdefault(lb, []).append(img)

    aug_imgs, aug_labs = list(images), list(labels)
    for lb, imgs in label_to_imgs.items():
        target = max(min_per_class, len(imgs) * factor)
        need = target - len(imgs)
        for _ in range(need):
            idx = rng.randint(0, len(imgs))
            aug_imgs.append(augment_image(imgs[idx], rng))
            aug_labs.append(lb)

    return np.array(aug_imgs), aug_labs


# ==================== 数据加载 ====================

def load_from_folder(folder, img_size=64):
    """递归遍历文件夹，解析所有 .inkml 并渲染为图像"""
    X, y, ids = [], [], []
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.endswith(".inkml"):
                continue
            path = os.path.join(root, f)
            strokes, label, uid = parse_inkml(path)
            if not label:
                continue
            img = render_strokes(strokes, img_size=img_size)
            X.append(img)
            y.append(label)
            ids.append(uid)
    return np.array(X), y, ids


def load_filtered(folder, min_samples=5, img_size=64):
    """加载数据并过滤掉样本数 < min_samples 的类别"""
    X, y, ids = load_from_folder(folder, img_size)
    if len(X) == 0:
        return np.array([]), [], []

    cnt = Counter(y)
    keep_labels = {lb for lb, c in cnt.items() if c >= min_samples}
    keep_idx = [i for i, lb in enumerate(y) if lb in keep_labels]

    X_f, y_f, ids_f = [], [], []
    for i in keep_idx:
        X_f.append(X[i])
        y_f.append(y[i])
        ids_f.append(ids[i])
    return np.array(X_f), y_f, ids_f


def print_distribution(y, title="类别分布"):
    cnt = Counter(y)
    print(f"\n--- {title}（{len(y)} 样本, {len(cnt)} 类）---")
    for lb, c in cnt.most_common():
        print(f"  {lb:>8s}: {c}")

# ==================== 自测 ====================

if __name__ == "__main__":
    DATA = r"D:\Users\admin\Desktop\手写数字识别\crohme_data"
    print("加载原始数据...")
    X, y, ids = load_filtered(DATA, min_samples=5, img_size=64)
    print_distribution(y, "原始分布")

    if len(X) > 0:
        print("数据增强...")
        X_aug, y_aug = augment_dataset(X, y, factor=2, min_per_class=60)
        print_distribution(y_aug, "增强后分布")
        print(f"增强前: {len(X)} → 增强后: {len(X_aug)}")
