"""
结构分类器训练 — 在对比学习编码器特征上训练轻量分类头。

分类器输入：编码器输出的 128 维特征
分类器输出：16 个二值结构属性（有无分式、根号、希腊字母等）
分类器架构：Linear(128 → 64 → 16) + Sigmoid

训练数据自动来自 LaTeX 解析（无需人工标注）。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inkml_parser import load_dataset
from contrastive_learning.latex_parser import (
    STRUCTURE_FEATURES, parse_latex, props_to_vector,
)
from contrastive_learning.extract_features import extract_encoder_features
from contrastive_learning.train_encoder import Encoder
from src.image_renderer import render_inkml
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class StructureClassifier(nn.Module):
    """轻量结构分类头：128维特征 → 16个结构属性。"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def build_training_data(
    data_dir: str,
    contrastive_ckpt: str,
    max_samples: int = 500,
    img_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """提取编码器特征 + 解析 LaTeX 结构标签。

    Returns:
        (features: [N, 128], labels: [N, 16])
    """
    dataset = load_dataset(data_dir, max_files=max_samples)
    print(f"加载 {len(dataset)} 个样本")

    # 加载对比学习编码器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Encoder(output_dim=128).to(device)
    encoder.load_state_dict(torch.load(contrastive_ckpt, map_location=device))
    encoder.eval()
    print(f"对比学习编码器已加载 ({sum(p.numel() for p in encoder.parameters()):,} 参数)")

    features, label_vecs = [], []
    for i, data in enumerate(dataset):
        # 提取特征
        img = render_inkml(data, img_size=img_size, stroke_width=4)
        if img.mode != "L":
            img = img.convert("L")
        from torchvision.transforms import functional as F
        tensor = F.to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = encoder(tensor).cpu().numpy().flatten()
        features.append(feat)

        # 解析结构标签
        latex = (data.truth_latex or "").strip("$").strip()
        props = parse_latex(latex)
        label_vecs.append(props_to_vector(props))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(dataset)}]")

    return np.stack(features).astype(np.float32), np.array(label_vecs, dtype=np.float32)


def train(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
) -> StructureClassifier:
    """训练结构分类器。"""
    X = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    n = len(X)
    print(f"训练数据: {n} 样本, {X.shape[1]} 维特征, {y.shape[1]} 个结构属性")

    # 训练/验证分割
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = StructureClassifier()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    # 统计每个属性在训练集中的比例，用于处理类别不均衡
    pos_ratio = y_train.mean(dim=0)
    pos_weight = (1.0 / (pos_ratio + 1e-6)).clamp(max=10.0)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0

        for start in range(0, len(X_train), batch_size):
            idx = perm[start:start + batch_size]
            bx, by = X_train[idx], y_train[idx]
            pred = model(bx)
            loss = bce(pred, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        train_loss = total_loss / len(X_train)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = bce(val_pred, y_val).item()
            val_acc = ((val_pred > 0.5) == y_val.bool()).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch [{epoch+1}/{epochs}]  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    print(f"训练完成! 最佳验证 loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="训练结构分类器")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/encoder_final.pt")
    parser.add_argument("--output", type=str, default="checkpoints/classifier.pt")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    features, labels = build_training_data(
        data_dir=args.data_dir,
        contrastive_ckpt=args.encoder_ckpt,
        max_samples=args.max_samples,
        img_size=args.img_size,
    )

    model = train(features, labels, epochs=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"分类器已保存: {output_path}")


if __name__ == "__main__":
    main()
