"""
对比学习训练 — 训练编码器使同一公式的图像向量相近，不同公式的向量相远。

训练完成后，编码器输出的 128 维向量可用于检索式识别：
测试图像 → 编码器 → 向量 → 在训练集中找最近邻 → 返回其 LaTeX

用法:
    python -m contrastive_learning.train_encoder \\
        --data_dir data/raw/archive/CROHME_training_2011 \\
        --epochs 50 --batch_size 32 --lr 0.001
"""

import argparse
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inkml_parser import load_dataset
from src.image_renderer import render_inkml


# ─────────────────────────────────────────────
# 1. 数据增强
# ─────────────────────────────────────────────

class FormulaAugmentation:
    """对已渲染的公式图像做随机增强，产生同一公式的不同"视角"。"""

    def __init__(self, img_size: int = 448):
        self.img_size = img_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.copy()

        # 随机旋转（±8°）
        angle = random.uniform(-8, 8)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

        # 随机缩放（0.85～1.15 倍）
        scale = random.uniform(0.85, 1.15)
        new_size = int(self.img_size * scale)
        img = img.resize((new_size, new_size), Image.BILINEAR)
        # 裁切或填充回 img_size
        if new_size > self.img_size:
            left = (new_size - self.img_size) // 2
            img = img.crop((left, left, left + self.img_size, left + self.img_size))
        elif new_size < self.img_size:
            padded = Image.new("L", (self.img_size, self.img_size), 255)
            offset = (self.img_size - new_size) // 2
            padded.paste(img, (offset, offset))
            img = padded

        # 随机水平平移（±5%）
        shift = random.randint(-self.img_size // 20, self.img_size // 20)
        img = img.transform(img.size, Image.AFFINE,
                            (1, 0, shift, 0, 1, 0), fillcolor=255)

        # 随机橡皮擦（遮盖 2～8% 的区域）
        if random.random() < 0.3:
            erase_size = random.randint(
                int(self.img_size * 0.02),
                int(self.img_size * 0.08),
            )
            x = random.randint(0, self.img_size - erase_size)
            y = random.randint(0, self.img_size - erase_size)
            draw = ImageDraw.Draw(img)
            draw.rectangle([x, y, x + erase_size, y + erase_size], fill=255)

        # 转 Tensor，归一化到 [0,1]
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]


# ─────────────────────────────────────────────
# 2. 编码器（小 CNN）
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """轻量 CNN 编码器：448x448 灰度图 → 128 维向量。

    架构：4 层卷积 + 全局平均池化 + 投影头。
    约 1.5M 参数，CPU 可训。
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            # 448 → 224 → 112
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 112 → 56 → 28
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 28 → 14
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 14 → 7
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # → [B, 256, 1, 1]
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)            # [B, 256, 7, 7]
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, 256]
        x = self.proj(x)             # [B, output_dim]
        return F.normalize(x, dim=1)  # L2 归一化


# ─────────────────────────────────────────────
# 3. 对比损失 (NT-Xent / InfoNCE)
# ─────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss。

    输入: [2*B, D] — 每对正样本在位置 (i, i+B)。
    对每个样本，正样本是其配对视图，其他 2B-2 个为负样本。
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0] // 2
        if B == 0:
            return torch.tensor(0.0, device=z.device)

        # 余弦相似度矩阵 [2B, 2B]
        sim = z @ z.T / self.temperature  # 已 L2 归一化，点积=余弦

        # 去掉对角线（自身对自身的相似度）
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim[mask].view(2 * B, -1)  # [2B, 2B-1]

        # 正样本位置：每个样本的配对在 (i+B) if i<B else (i-B)
        pos = torch.arange(2 * B, device=z.device)
        pos = torch.where(pos < B, pos + B, pos - B)

        # 在 mask 后的 sim 中，正样本所在的列索引需要重新计算
        # 对于样本 i，它的正样本是 pos[i]。
        # 在 mask 后的矩阵中，如果 pos[i] > i，则列号 = pos[i]-1
        pos_col = torch.where(pos > torch.arange(2 * B, device=z.device),
                              pos - 1, pos)

        pos_sim = sim[torch.arange(2 * B, device=z.device), pos_col]
        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        return loss.mean()


# ─────────────────────────────────────────────
# 4. 数据集
# ─────────────────────────────────────────────

class InkMLContrastiveDataset(Dataset):
    """从 InkML 数据加载公式，每次返回两个增强视图。"""

    def __init__(
        self,
        data_list: list,
        img_size: int = 224,
        stroke_width: int = 4,
    ):
        self.data_list = data_list
        self.img_size = img_size
        self.stroke_width = stroke_width
        self.aug = FormulaAugmentation(img_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data_list[idx]
        img = render_inkml(data, img_size=self.img_size,
                           stroke_width=self.stroke_width)
        view1 = self.aug(img)
        view2 = self.aug(img)
        return view1, view2


# ─────────────────────────────────────────────
# 5. 训练
# ─────────────────────────────────────────────

def train_epoch(
    model: Encoder,
    loader: DataLoader,
    loss_fn: NTXentLoss,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_idx, (views1, views2) in enumerate(loader):
        # views1, views2: [B, 1, H, W]   → 拼接成 [2B, 1, H, W]
        x = torch.cat([views1, views2], dim=0).to(device)

        optimizer.zero_grad()
        z = model(x)                # [2B, 128]
        loss = loss_fn(z)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f"    batch [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}")

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="训练对比学习编码器")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--img_size", type=int, default=224,
                        help="训练图像尺寸（224 比 448 快 4x）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cpu":
        print("  （CPU 训练较慢，建议减少 --img_size 或 --max_samples）")

    # 数据
    print(f"加载数据: {args.data_dir}")
    dataset = load_dataset(args.data_dir, max_files=args.max_samples)
    if not dataset:
        print("[ERROR] 数据为空")
        return
    print(f"  共 {len(dataset)} 个样本")

    train_ds = InkMLContrastiveDataset(
        dataset, img_size=args.img_size, stroke_width=4,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0,  # Windows 上多进程可能不稳定
    )

    # 模型
    model = Encoder(output_dim=args.output_dim).to(device)
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"从 {args.resume} 恢复 (epoch {start_epoch})")

    loss_fn = NTXentLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    print(f"\n开始训练: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  编码器参数量: {sum(p.numel() for p in model.parameters()):,}")
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"epoch [{epoch+1}/{args.epochs}] loss={loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.1f}s)")

        # 每 10 个 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"encoder_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": loss,
            }, ckpt_path)
            print(f"  checkpoint 已保存: {ckpt_path}")

    total_time = time.time() - t_start
    print(f"\n训练完成! 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")

    # 保存最终模型
    final_path = output_dir / "encoder_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"最终模型已保存: {final_path}")
    print(f"  使用命令测试检索: python -m contrastive_learning.evaluate "
          f"--train_dir {args.data_dir} --test_dir {args.data_dir} "
          f"--max_train 200 --max_test 50")


if __name__ == "__main__":
    main()
