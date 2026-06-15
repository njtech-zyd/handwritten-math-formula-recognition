"""
LaTeXify 优化流水线 — 本地 TrOCR + DeepSeek 文本校正。

用法:
    python run_latexify.py --data_dir data/raw/archive/CROHME_training_2011 --max_samples 50
    python run_latexify.py --data_dir data/raw/archive/CROHME_training_2011 --max_samples 200 --refine
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from openai import OpenAI
from src.inkml_parser import load_dataset
from src.image_renderer import render_inkml
from src.evaluator import evaluate_all, print_results


_REFINE_PROMPT = (
    "You are a math formula OCR post-processor. Fix errors in LaTeX OCR output.\n\n"
    "Common errors to fix:\n"
    "- Missing backslashes: sin(x) -> \\sin(x), cos -> \\cos, etc.\n"
    "- Wrong characters that are clearly OCR mistakes: d -> 2, e -> c, l -> 1\n"
    "- Extra decimal points: 9.97 -> 97, 5.0 -> 50 (when not meaningful)\n"
    "- Missing curly braces in subscripts/superscripts: n_1 -> n_{1}\n"
    "- Wrong ellipsis: ... -> \\dots or \\cdots\n"
    "- Missing operators: =, \\pm, \\div, \\times, \\neq, \\leq, \\geq\n"
    "- Document wrappers like \\documentclass{article}\n\n"
    "IMPORTANT: Only fix CLEAR errors. If uncertain, keep the original.\n"
    "Output ONLY the corrected LaTeX. No $$, no explanations."
)


def refine_latex(client, latex, timeout=15):
    """DeepSeek 文本 API 校正 LaTeX。"""
    if not latex.strip():
        return latex
    try:
        r = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a LaTeX OCR post-processor."},
                {"role": "user", "content": f"{_REFINE_PROMPT}\n\nLaTeX to fix:\n{latex}"},
            ],
            timeout=timeout,
        )
        result = r.choices[0].message.content.strip().strip("$").strip()
        return result if result else latex
    except Exception as e:
        print(f"    [timeout, keeping original]")
        return latex


def main():
    parser = argparse.ArgumentParser(description="LaTeXify 优化流水线")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--output_dir", type=str, default="results/latexify")
    parser.add_argument("--refine", action="store_true", help="启用 DeepSeek 文本校正")
    parser.add_argument("--batch_refine", action="store_true", help="批量校正（所有结果一起发）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 LaTeXify
    print("加载 LaTeXify...")
    t0 = time.time()
    processor = TrOCRProcessor.from_pretrained("tjoab/latex_finetuned")
    model = VisionEncoderDecoderModel.from_pretrained("tjoab/latex_finetuned")
    model.eval()
    print(f"  完成 ({time.time()-t0:.1f}s, {sum(p.numel() for p in model.parameters()):,} params)")

    # DeepSeek 客户端
    client = None
    if args.refine:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("[WARNING] DEEPSEEK_API_KEY 未设置，跳过校正")
        else:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            print("DeepSeek 文本校正: ON")

    # 加载数据
    print(f"加载数据: {args.data_dir}")
    dataset = load_dataset(args.data_dir, max_files=args.max_samples)
    print(f"  共 {len(dataset)} 个样本")

    # 识别
    print(f"\n开始识别 ({len(dataset)} 样本)...")
    predictions = []
    truths = []
    file_names = []
    n_correct_raw = 0
    n_correct_ref = 0

    for i, data in enumerate(dataset):
        img = render_inkml(data, img_size=args.img_size, stroke_width=4).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=128)
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        truth = (data.truth_latex or "").strip("$").strip()
        raw_match = pred.replace(" ", "") == truth.replace(" ", "")
        if raw_match:
            n_correct_raw += 1

        # DeepSeek 校正
        refined = pred
        if client:
            refined = refine_latex(client, pred)

        ref_match = refined.replace(" ", "") == truth.replace(" ", "")
        if ref_match:
            n_correct_ref += 1

        predictions.append(refined)
        truths.append(truth)
        file_names.append(str(data.file_path) if data.file_path else f"sample_{i}")

        raw_mark = "[OK]" if raw_match else "[NO]"
        ref_mark = "[OK]" if ref_match else "[NO]"
        print(f"  [{i+1:2d}/{len(dataset)}] {raw_mark}{ref_mark} "
              f"t={truth[:35]:35s} p={pred[:35]:35s}", end="")
        if client and refined != pred:
            print(f" ref={refined[:35]}")
        else:
            print()

        if (i + 1) % 10 == 0:
            print(f"  原始正确率: {n_correct_raw}/{i+1} ({n_correct_raw/(i+1)*100:.1f}%)"
                  + (f" 校正后: {n_correct_ref}/{i+1} ({n_correct_ref/(i+1)*100:.1f}%)" if client else ""))

    # 评估
    print(f"\n{'='*50}")
    tag = "latexify_raw" if not args.refine else "latexify_refined"
    results = evaluate_all(
        predictions=predictions,
        truths=truths,
        file_names=file_names,
    )
    print_results(results)

    # 保存 CSV
    csv_path = output_dir / f"predictions_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "truth", "prediction", "exact_match", "cer", "bleu"])
        for d in results["details"]:
            w.writerow([d.file, d.truth_latex, d.predicted_latex, d.exact_match, f"{d.cer:.4f}", f"{d.bleu:.4f}"])
    print(f"结果保存: {csv_path}")

    # 保存摘要
    summary_path = output_dir / f"summary_{tag}.txt"
    accuracy = sum(1 for r in results["details"] if r.exact_match) / max(len(results["details"]), 1)
    summary_path.write_text(
        f"LaTeXify {'+ DeepSeek Refine' if args.refine else 'Raw'} Evaluation\n"
        f"{'='*50}\n"
        f"Total: {len(results['details'])}\n"
        f"Exact Match: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
        f"Mean CER: {results['summary']['cer']:.4f}\n"
        f"Mean BLEU: {results['summary']['bleu']:.4f}\n"
    )

    print(f"\n{'='*50}")
    print(f"  完成！结果目录: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
