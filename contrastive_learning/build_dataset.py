"""
构建训练数据集 — 加载 InkML、渲染图像、提取编码器特征、建立词汇表。

输出目录结构:
  data_cache/
  ├── features/          # [样本ID].npy — 编码器特征 [49, 256]
  ├── meta.csv           # 样本ID, LaTeX, 文件名
  └── vocab.json          # 词汇表

用法:
    python -m contrastive_learning.build_dataset \
        --data_dir data/raw/archive/TrainINKML_2013 \
        --output data_cache
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src.inkml_parser import load_dataset
from src.image_renderer import render_inkml
from contrastive_learning.train_encoder import Encoder
from contrastive_learning.latex_tokenizer import build_vocab


class FeatureExtractor:
    """从对比学习编码器提取中间特征图 [B, 256, 7, 7] → [B, 49, 256]。"""

    def __init__(self, ckpt: str, device: str = "cpu"):
        self.device = device
        self.encoder = Encoder(output_dim=128).to(device)
        self.encoder.load_state_dict(torch.load(ckpt, map_location=device))
        self.encoder.eval()
        print(f"[特征提取器] 加载编码器: {ckpt} ({sum(p.numel() for p in self.encoder.parameters()):,} 参数)")

    def extract(self, image) -> np.ndarray:
        """提取特征图 [49, 256]。

        Args:
            image: PIL Image (灰度图)。

        Returns:
            [49, 256] numpy array。
        """
        if image.mode != "L":
            image = image.convert("L")
        tensor = F.to_tensor(image).unsqueeze(0).to(self.device)  # [1,1,H,W]

        with torch.no_grad():
            # 获取卷积层输出 [1, 256, 7, 7]
            feat_map = self.encoder.conv(tensor)
            # 重塑为 [49, 256]
            b, c, h, w = feat_map.shape
            features = feat_map.view(b, c, h * w).permute(0, 2, 1)  # [1, 49, 256]

        return features.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="构建训练数据集")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data_cache")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/encoder_final.pt")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    feat_dir = output_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print(f"[1/4] 加载数据: {args.data_dir}")
    dataset = load_dataset(args.data_dir, max_files=args.max_samples)
    print(f"      共 {len(dataset)} 个样本")

    # 2. 渲染 + 提取特征
    print(f"[2/4] 渲染图像 + 提取编码器特征...")
    extractor = FeatureExtractor(args.encoder_ckpt)
    meta = []
    t0 = time.time()

    for i, data in enumerate(dataset):
        img = render_inkml(data, img_size=args.img_size, stroke_width=4)
        latex = (data.truth_latex or "").strip("$").strip()
        features = extractor.extract(img)

        # 保存特征
        np.save(feat_dir / f"{i}.npy", features)

        meta.append({
            "id": i,
            "latex": latex,
            "file": str(data.file_path) if data.file_path else "",
        })

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(dataset) - i - 1) / rate
            print(f"      [{i+1}/{len(dataset)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"      完成! {len(dataset)} 样本耗时 {elapsed:.0f}s ({elapsed/len(dataset):.3f}s/样本)")

    # 3. 建立词汇表
    print(f"[3/4] 建立 LaTeX 词汇表...")
    all_latex = [m["latex"] for m in meta]
    vocab = build_vocab(all_latex, min_freq=2)
    print(f"      词汇表大小: {len(vocab)} (含 {len([k for k in vocab if k.startswith('\\')])} 个 LaTeX 命令)")

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 4. 保存元数据
    print(f"[4/4] 保存元数据...")
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n数据集构建完成!")
    print(f"  特征: {feat_dir}/  ({len(dataset)} 个 .npy 文件, 每个 [49,256])")
    print(f"  词汇表: {output_dir/'vocab.json'} ({len(vocab)} tokens)")
    print(f"  元数据: {output_dir/'meta.json'}")


if __name__ == "__main__":
    main()
