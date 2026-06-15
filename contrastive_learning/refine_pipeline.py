"""
特征增强流水线 — 对比学习提纯特征 → 结构分析 → DeepSeek 增强识别。

流程：
1. 对比学习编码器提取输入图像的"提纯"特征 (128维)
2. 结构分类头预测公式结构属性（分式、根号、希腊字母等）
3. 将结构分析 + 笔画数据组装为增强 Prompt
4. DeepSeek 基于"更好的原料"生成 LaTeX

与 OCR 流水线的区别：
- OCR:       图像 → LaTeXify → (DeepSeek校正) → LaTeX
- 增强流水线: 图像 → 编码器提纯 → 结构分析 → 增强 Prompt → DeepSeek → LaTeX
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contrastive_learning.train_encoder import Encoder
from contrastive_learning.train_classifier import StructureClassifier
from contrastive_learning.latex_parser import (
    STRUCTURE_FEATURES, props_to_text,
)
from src.stroke_formatter import traces_to_text as strokes_to_text
from src.inkml_parser import load_dataset
from src.image_renderer import render_inkml
from dotenv import load_dotenv
from src.api_recognizer import StrokeTextRecognizer
from src.evaluator import evaluate_all, print_results
from src.visualizer import create_report


_STRUCTURE_PROMPT = (
    "You are a handwritten math formula recognizer. You receive:\n"
    "1) Structural hints about the formula\n"
    "2) Ink stroke data as coordinate sequences\n\n"
    "Use the structural hints to guide recognition of the stroke data. "
    "Output ONLY the raw LaTeX. No $$, no markdown."
)


def extract_feature(
    encoder: Encoder,
    image: Image.Image,
    device: str,
) -> np.ndarray:
    """用对比学习编码器提取特征。"""
    from torchvision.transforms import functional as F
    if image.mode != "L":
        image = image.convert("L")
    tensor = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = encoder(tensor).cpu().numpy().flatten()
    return vec.astype(np.float32)


def predict_structure(
    classifier: StructureClassifier,
    feature: np.ndarray,
) -> dict[str, bool]:
    """从特征预测结构属性。"""
    x = torch.from_numpy(feature).unsqueeze(0)
    classifier.eval()
    with torch.no_grad():
        probs = classifier(x).squeeze().numpy()
    return {f: bool(round(p)) for f, p in zip(STRUCTURE_FEATURES, probs)}


def build_enhanced_prompt(
    stroke_text: str,
    structure: dict[str, bool],
) -> str:
    """组装增强 Prompt：结构分析 + 笔画数据。"""
    analysis = props_to_text(structure)
    return (
        f"{analysis}\n\n"
        f"Stroke coordinates:\n"
        f"{stroke_text}"
    )


class StructureEnhancedRecognizer:
    """特征增强式识别器。

    三步：提纯特征 → 结构分析 → 增强 Prompt → DeepSeek。
    """

    def __init__(
        self,
        encoder_ckpt: str,
        classifier_ckpt: str,
        prompt: str = _STRUCTURE_PROMPT,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 1. 对比学习编码器
        self.encoder = Encoder(output_dim=128).to(device)
        self.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
        self.encoder.eval()

        # 2. 结构分类器
        num_classes = len(STRUCTURE_FEATURES)
        self.classifier = StructureClassifier(num_classes=num_classes)
        self.classifier.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))
        self.classifier.eval()

        # 3. DeepSeek 文本 API
        self.recognizer = StrokeTextRecognizer(
            base_url="https://api.deepseek.com",
            prompt=prompt,
        )
        print(f"[增强流水线] 编码器: {encoder_ckpt}")
        print(f"[增强流水线] 分类器: {classifier_ckpt}")
        print(f"[增强流水线] 设备: {device}")

    def recognize(self, image: Image.Image, stroke_text: str) -> str:
        """识别单张图像。

        Args:
            image: 渲染后的灰度图像。
            stroke_text: 格式化的笔画文本。

        Returns:
            LaTeX 字符串。
        """
        # Step 1: 提取特征
        feat = extract_feature(self.encoder, image, self.device)

        # Step 2: 预测结构
        structure = predict_structure(self.classifier, feat)
        print(f"  [结构分析] {structure}", flush=True)

        # Step 3: 组装增强 Prompt
        enhanced_prompt = build_enhanced_prompt(stroke_text, structure)

        # Step 4: DeepSeek 识别（纯文本，直接用 prompt+stroke_data）
        latex = self.recognizer.recognize(enhanced_prompt)
        return latex


def main():
    parser = argparse.ArgumentParser(description="特征增强式流水线")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/encoder_final.pt")
    parser.add_argument("--classifier_ckpt", type=str, default="checkpoints/classifier.pt")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="results/enhanced")
    args = parser.parse_args()

    # 加载 .env 环境变量（API Key 等）
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    recognizer = StructureEnhancedRecognizer(
        encoder_ckpt=args.encoder_ckpt,
        classifier_ckpt=args.classifier_ckpt,
    )

    # 加载数据
    print(f"\n加载数据: {args.data_dir}")
    dataset = load_dataset(args.data_dir, max_files=args.max_samples)
    if not dataset:
        print("[ERROR] 数据为空")
        return
    print(f"共 {len(dataset)} 个样本")

    predictions = []
    truths = []
    file_names = []
    images = []

    for i, data in enumerate(dataset):
        img = render_inkml(data, img_size=args.img_size, stroke_width=4)
        stroke_text = strokes_to_text(data)
        print(f"  [{i+1}/{len(dataset)}] 正在识别...", flush=True)
        latex = recognizer.recognize(img, stroke_text)

        truth = (data.truth_latex or "").strip("$").strip()
        fname = str(data.file_path) if data.file_path else f"sample_{i}"

        predictions.append(latex)
        truths.append(truth)
        file_names.append(fname)
        images.append(img)

        print(f"  [{i+1}/{len(dataset)}] {Path(fname).stem}: {truth} → {latex}", flush=True)

    # 评估
    print(f"\n{'='*50}")
    print(f"  评估结果")
    print(f"{'='*50}")
    results = evaluate_all(
        predictions=predictions,
        truths=truths,
        file_names=file_names,
    )
    print_results(results)

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import csv
    pred_csv = output_dir / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "truth", "prediction", "exact_match", "cer", "bleu"])
        for detail in results["details"]:
            writer.writerow([
                detail.file, detail.truth_latex, detail.predicted_latex,
                detail.exact_match, f"{detail.cer:.4f}", f"{detail.bleu:.4f}",
            ])
    print(f"预测结果已保存: {pred_csv}")

    # 可视化
    try:
        report = create_report(
            results=results["details"],
            images=images,
            output_dir=output_dir / "figures",
        )
        for name, path in report.items():
            print(f"  {name}: {path.name}")
    except Exception as e:
        print(f"[visualize] {e}")

    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
