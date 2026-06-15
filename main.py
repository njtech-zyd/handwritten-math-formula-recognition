#!/usr/bin/env python3
"""
多模态数学公式识别 — 主流水线。

串联数据加载、图像渲染、API 识别、评估和可视化的完整流程。

用法:
    # 完整流水线（渲染 + 识别 + 评估）
    python main.py --mode pipeline --data_dir data/raw/archive/CROHME_test_2011

    # 仅渲染并保存图像
    python main.py --mode render --data_dir data/raw/archive/CROHME_test_2011 --max_samples 50

    # 仅评估（已有预测结果时）
    python main.py --mode evaluate --pred_file results/predictions/predictions.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多模态数学公式识别 — 主流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode", type=str, default="pipeline",
        choices=["pipeline", "stroke_pipeline", "ocr_pipeline", "decoder", "decoder_refine", "lora", "lora_refine", "render", "evaluate"],
        help="运行模式 (default: pipeline)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="",
        help="InkML 数据目录路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="结果输出目录 (default: results)",
    )
    parser.add_argument(
        "--img_size", type=int, default=448,
        help="渲染图像尺寸 (default: 448)",
    )
    parser.add_argument(
        "--stroke_width", type=int, default=4,
        help="笔画渲染宽度 (default: 4)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="最大处理样本数 (调试用)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (default: 42)",
    )
    parser.add_argument(
        "--pred_file", type=str, default="",
        help="评估模式：预测结果 CSV 路径",
    )
    parser.add_argument(
        "--encoder_ckpt", type=str, default="",
        help="编码器 checkpoint 路径 (decoder 模式使用)",
    )
    parser.add_argument(
        "--decoder_ckpt", type=str, default="",
        help="解码器 checkpoint 路径 (decoder 模式使用)",
    )
    parser.add_argument(
        "--data_cache", type=str, default="",
        help="特征缓存目录路径 (decoder 模式使用)",
    )
    parser.add_argument(
        "--lora_ckpt", type=str, default="checkpoints/latexify_gpu_v2_ep8",
        help="LoRA 权重路径 (lora 模式使用)",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="启用 DeepSeek 文本 API 校正 (lora 模式使用)",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    """解析相对/绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    return p.resolve()


def mode_render(args: argparse.Namespace) -> Path:
    """仅渲染模式。"""
    from src.inkml_parser import load_dataset
    from src.image_renderer import render_inkml

    data_dir = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] 加载数据: {data_dir}")
    dataset = load_dataset(data_dir, max_files=args.max_samples)
    print(f"[render] 共 {len(dataset)} 个样本")

    render_dir = output_dir / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(dataset):
        img = render_inkml(data, img_size=args.img_size, stroke_width=args.stroke_width)
        fname = data.file_path.stem if data.file_path else f"sample_{i}"
        out = render_dir / f"{fname}.png"
        img.save(out)

    print(f"[render] 图像已保存到: {render_dir}")
    return render_dir


def mode_pipeline(args: argparse.Namespace) -> None:
    """完整流水线：加载 → 渲染 → 识别 → 评估 → 可视化。"""
    import numpy as np
    from src.inkml_parser import load_dataset
    from src.image_renderer import batch_render
    from src.api_recognizer import DeepSeekRecognizer
    from src.evaluator import evaluate_all, print_results
    from src.visualizer import create_report

    np.random.seed(args.seed)

    data_dir = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print(f"[step 1/5] 加载数据: {data_dir}")
    dataset = load_dataset(data_dir, max_files=args.max_samples)
    if len(dataset) == 0:
        print("[ERROR] 数据为空，退出。")
        sys.exit(1)
    print(f"         共 {len(dataset)} 个样本")

    # 2. 渲染图像
    print(f"[step 2/5] 渲染图像 ({args.img_size}x{args.img_size})")
    rendered = batch_render(dataset, img_size=args.img_size, stroke_width=args.stroke_width)
    images = [r["image"] for r in rendered]
    truths = [(r["data"].truth_latex or "").strip("$").strip() for r in rendered]
    file_names = [r["file"] for r in rendered]

    # 统计有真值的样本
    has_truth = sum(1 for t in truths if t)
    print(f"         有真值标注: {has_truth}/{len(truths)}")

    # 3. API 识别
    print(f"[step 3/5] 调用 DeepSeek API 识别...")
    recognizer = DeepSeekRecognizer()
    predictions = recognizer.batch_recognize(images)
    print(f"         识别完成: {len(predictions)} 个预测")

    # 4. 评估
    print(f"[step 4/5] 计算评估指标")
    results = evaluate_all(
        predictions=predictions,
        truths=truths,
        file_names=file_names,
    )
    print_results(results)

    # 保存预测结果
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = pred_dir / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "truth", "prediction", "exact_match", "cer", "bleu"])
        for detail in results["details"]:
            writer.writerow([
                detail.file, detail.truth_latex, detail.predicted_latex,
                detail.exact_match, f"{detail.cer:.4f}", f"{detail.bleu:.4f}",
            ])
    print(f"         预测结果已保存: {pred_csv}")

    # 5. 可视化
    print(f"[step 5/5] 生成可视化报告")
    report_files = create_report(
        results=results["details"],
        images=images,
        output_dir=output_dir / "figures",
    )
    for name, path in report_files.items():
        print(f"         {name}: {path.name}")

    print(f"\n{'=' * 50}")
    print(f"  流水线完成。输出目录: {output_dir}")
    print(f"{'=' * 50}")


def mode_stroke_pipeline(args: argparse.Namespace) -> None:
    """笔画文本流水线：加载 → 格式化笔画文本 → 识别 → 评估 → 可视化。"""
    import numpy as np
    from src.inkml_parser import load_dataset
    from src.image_renderer import batch_render
    from src.stroke_formatter import traces_to_ascii_grid
    from src.api_recognizer import StrokeTextRecognizer
    from src.evaluator import evaluate_all, print_results
    from src.visualizer import create_report

    np.random.seed(args.seed)

    data_dir = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print(f"[step 1/5] 加载数据: {data_dir}")
    dataset = load_dataset(data_dir, max_files=args.max_samples)
    if len(dataset) == 0:
        print("[ERROR] 数据为空，退出。")
        sys.exit(1)
    print(f"         共 {len(dataset)} 个样本")

    # 2. 格式化笔画为 ASCII 网格 + 渲染图像（可视化用）
    print(f"[step 2/5] 转换笔画数据为 ASCII 网格...")
    stroke_texts = []
    truths = []
    file_names = []
    for data in dataset:
        text = traces_to_ascii_grid(data)
        stroke_texts.append(text)
        truths.append((data.truth_latex or "").strip("$").strip())
        file_names.append(str(data.file_path) if data.file_path else "")

    has_truth = sum(1 for t in truths if t)
    print(f"         有真值标注: {has_truth}/{len(truths)}")

    # 顺便渲染图像（可视化报告需要）
    rendered = batch_render(dataset, img_size=args.img_size, stroke_width=args.stroke_width)
    images = [r["image"] for r in rendered]

    # 3. DeepSeek 文本识别（用 OpenAI 端点，不是 Anthropic）
    print(f"[step 3/5] 调用 DeepSeek API (text) 识别笔画...")
    recognizer = StrokeTextRecognizer(base_url="https://api.deepseek.com")
    predictions = recognizer.batch_recognize(stroke_texts)
    print(f"         识别完成: {len(predictions)} 个预测")

    # 4. 评估
    print(f"[step 4/5] 计算评估指标")
    results = evaluate_all(
        predictions=predictions,
        truths=truths,
        file_names=file_names,
    )
    print_results(results)

    # 保存预测结果
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = pred_dir / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "truth", "prediction", "exact_match", "cer", "bleu"])
        for detail in results["details"]:
            writer.writerow([
                detail.file, detail.truth_latex, detail.predicted_latex,
                detail.exact_match, f"{detail.cer:.4f}", f"{detail.bleu:.4f}",
            ])
    print(f"         预测结果已保存: {pred_csv}")

    # 5. 可视化
    print(f"[step 5/5] 生成可视化报告")
    report_files = create_report(
        results=results["details"],
        images=images,
        output_dir=output_dir / "figures",
    )
    for name, path in report_files.items():
        print(f"         {name}: {path.name}")

    print(f"\n{'=' * 50}")
    print(f"  流水线完成。输出目录: {output_dir}")
    print(f"{'=' * 50}")


def mode_lora(args: argparse.Namespace) -> None:
    """GPU LoRA 微调流水线：加载 → 渲染 → GPU LoRA 识别 → 评估 → 对比图。"""
    import numpy as np
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from peft import PeftModel
    from src.inkml_parser import load_dataset
    from src.image_renderer import batch_render
    from src.evaluator import evaluate_all, print_results
    from src.visualizer import create_report

    np.random.seed(args.seed)
    data_dir = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lora] 设备: {device}")

    # 1. 加载数据
    print(f"[step 1/5] 加载数据: {data_dir}")
    dataset = load_dataset(data_dir, max_files=args.max_samples)
    if len(dataset) == 0:
        print("[ERROR] 数据为空")
        sys.exit(1)
    print(f"         共 {len(dataset)} 个样本")

    # 2. 渲染
    print(f"[step 2/5] 渲染图像 ({args.img_size}x{args.img_size})")
    rendered = batch_render(dataset, img_size=args.img_size, stroke_width=args.stroke_width)
    images = [r["image"] for r in rendered]
    truths = [(r["data"].truth_latex or "").strip("$").strip() for r in rendered]
    file_names = [r["file"] for r in rendered]

    # 3. 加载 GPU LoRA 模型
    print(f"[step 3/5] 加载 GPU LoRA 模型: {args.lora_ckpt}")
    processor = TrOCRProcessor.from_pretrained("tjoab/latex_finetuned")
    base = VisionEncoderDecoderModel.from_pretrained("tjoab/latex_finetuned")
    model = PeftModel.from_pretrained(base, args.lora_ckpt)
    model.to(device).eval()

    # 4. 识别
    print(f"[step 4/5] 识别 {len(images)} 个样本...")
    predictions = []
    for i, img in enumerate(images):
        pv = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values=pv, max_length=128)
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
        predictions.append(pred)
        if (i + 1) % 10 == 0:
            print(f"         [{i+1}/{len(images)}]")

    # 4b. DeepSeek 文本校正（可选）
    if args.refine:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            print(f"         DeepSeek 文本校正中 ({len(predictions)} 个样本)...")
            refine_prompt = (
                "You are a math formula OCR post-processor. Fix LaTeX recognition errors.\n"
                "Common fixes needed:\n"
                "- Wrong digits or variables\n"
                "- Missing or extra parentheses\n"
                "- Wrong operators (\\pm, \\div, \\times, \\neq, \\leq, \\geq)\n"
                "- Add missing backslashes to functions (sin → \\sin)\n\n"
                "Output ONLY the corrected LaTeX. No explanations."
            )
            for i, pred in enumerate(predictions):
                if pred.strip():
                    try:
                        r = client.chat.completions.create(
                            model="deepseek-v4-flash",
                            messages=[
                                {"role": "system", "content": "You are a LaTeX OCR post-processor."},
                                {"role": "user", "content": f"{refine_prompt}\n\nLaTeX to correct:\n{pred}"},
                            ],
                            timeout=15,
                        )
                        refined = r.choices[0].message.content.strip().strip("$").strip()
                        if refined:
                            predictions[i] = refined
                    except Exception:
                        pass
                if (i + 1) % 10 == 0:
                    print(f"         refine [{i+1}/{len(predictions)}]")
            print(f"         DeepSeek 校正完成")
        else:
            print(f"         [WARNING] DEEPSEEK_API_KEY 未设置，跳过校正")

    # 5. 评估 + 可视化
    print(f"[step 5/5] 评估...")
    results = evaluate_all(predictions=predictions, truths=truths, file_names=file_names)
    print_results(results)

    import csv
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = pred_dir / "predictions_lora.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "truth", "prediction", "exact_match", "cer", "bleu"])
        for d in results["details"]:
            w.writerow([d.file, d.truth_latex, d.predicted_latex, d.exact_match, f"{d.cer:.4f}", f"{d.bleu:.4f}"])
    print(f"         结果保存: {pred_csv}")

    report_files = create_report(results["details"], images, output_dir=output_dir / "figures")
    for name, path in report_files.items():
        print(f"         {name}: {path.name}")
    print(f"\n输出目录: {output_dir}")


def mode_decoder(args: argparse.Namespace, use_refine: bool = False) -> None:
    """编码器-解码器流水线：加载 → 渲染 → 编码器 → Decoder → (DeepSeek校正) → 评估。"""
    import json
    import numpy as np
    import torch
    from src.inkml_parser import load_dataset
    from src.image_renderer import batch_render
    from src.evaluator import evaluate_all, print_results
    from src.visualizer import create_report
    from contrastive_learning.decoder import LaTeXDecoder, load_decoder
    from contrastive_learning.latex_tokenizer import detokenize

    np.random.seed(args.seed)

    data_dir = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print(f"[step 1/5] 加载数据: {data_dir}")
    dataset = load_dataset(data_dir, max_files=args.max_samples)
    if len(dataset) == 0:
        print("[ERROR] 数据为空，退出。")
        sys.exit(1)
    print(f"         共 {len(dataset)} 个样本")

    # 2. 渲染
    print(f"[step 2/5] 渲染图像 ({args.img_size}x{args.img_size})")
    rendered = batch_render(dataset, img_size=args.img_size, stroke_width=args.stroke_width)
    images = [r["image"] for r in rendered]
    truths = [(r["data"].truth_latex or "").strip("$").strip() for r in rendered]
    file_names = [r["file"] for r in rendered]

    # 3. 加载模型
    print(f"[step 3/5] 加载 Encoder-Decoder 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_ckpt = args.encoder_ckpt or "checkpoints/encoder_final.pt"
    decoder_ckpt = args.decoder_ckpt or "checkpoints/decoder_v2.pt"

    # 加载词汇表
    data_cache = args.data_cache or "data_cache_full"
    vocab_path = Path(data_cache) / "vocab.json"
    if not vocab_path.exists():
        # 回退到 data_cache
        vocab_path = Path("data_cache") / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)

    # 编码器
    from contrastive_learning.train_encoder import Encoder
    encoder = Encoder(output_dim=128).to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    encoder.eval()

    # 解码器
    decoder = load_decoder(decoder_ckpt, len(vocab), device=device)

    from torchvision.transforms import functional as F
    from contrastive_learning.latex_tokenizer import SPECIAL as T_SPECIAL

    # 4. 识别
    print(f"[step 4/5] 解码器识别 {'+ DeepSeek 校正' if use_refine else ''}...")
    if use_refine:
        from src.api_recognizer import StrokeTextRecognizer
        refiner = StrokeTextRecognizer(
            base_url="https://api.deepseek.com",
            prompt="Fix any LaTeX syntax errors in the following formula. Output ONLY corrected LaTeX. No explanations.",
        )

    predictions = []
    for i, img in enumerate(images):
        # 提取编码器特征
        if img.mode != "L":
            img = img.convert("L")
        tensor = F.to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_map = encoder.conv(tensor)
            b, c, h, w = feat_map.shape
            features = feat_map.view(b, c, h * w).permute(0, 2, 1)  # [1, 49, 256]

        # 解码器生成
        with torch.no_grad():
            generated = decoder.generate(features, max_len=128, temperature=0.6)
        pred_latex = detokenize(generated[0].tolist(), vocab)

        # DeepSeek 校正（可选）
        if use_refine and pred_latex.strip():
            try:
                refined = refiner.recognize(pred_latex)
                if refined.strip() and len(refined) > 2:
                    pred_latex = refined
            except Exception as e:
                print(f"  [refine error] {e}")

        predictions.append(pred_latex)

        if (i + 1) % 10 == 0:
            print(f"         [{i+1}/{len(images)}] {truths[i][:30] if truths[i] else '?'} → {pred_latex[:30]}")

    # 5. 评估
    print(f"[step 5/5] 评估...")
    results = evaluate_all(predictions=predictions, truths=truths, file_names=file_names)
    print_results(results)

    # 保存
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    import csv
    pred_csv = pred_dir / f"predictions_decoder{'_refine' if use_refine else ''}.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "truth", "prediction", "exact_match", "cer", "bleu"])
        for detail in results["details"]:
            writer.writerow([
                detail.file, detail.truth_latex, detail.predicted_latex,
                detail.exact_match, f"{detail.cer:.4f}", f"{detail.bleu:.4f}",
            ])
    print(f"         结果已保存: {pred_csv}")

    # 6. 可视化
    report_files = create_report(
        results=results["details"],
        images=images,
        output_dir=output_dir / "figures",
    )
    for name, path in report_files.items():
        print(f"         {name}: {path.name}")

    print(f"\n{'=' * 50}")
    print(f"  解码器流水线完成。输出目录: {output_dir}")
    print(f"{'=' * 50}")


def mode_ocr_pipeline(args: argparse.Namespace) -> None:
    """OCR 流水线：加载 → 渲染 → pix2tex OCR → (DeepSeek 校正) → 评估 → 可视化。"""
    import numpy as np
    from src.inkml_parser import load_dataset
    from src.image_renderer import batch_render
    from src.ocr_recognizer import OcrRecognizer
    from src.evaluator import evaluate_all, print_results
    from src.visualizer import create_report

    np.random.seed(args.seed)

    data_dir = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print(f"[step 1/5] 加载数据: {data_dir}")
    dataset = load_dataset(data_dir, max_files=args.max_samples)
    if len(dataset) == 0:
        print("[ERROR] 数据为空，退出。")
        sys.exit(1)
    print(f"         共 {len(dataset)} 个样本")

    # 2. 渲染图像
    print(f"[step 2/5] 渲染图像 ({args.img_size}x{args.img_size})")
    rendered = batch_render(dataset, img_size=args.img_size, stroke_width=args.stroke_width)
    images = [r["image"] for r in rendered]
    truths = [(r["data"].truth_latex or "").strip("$").strip() for r in rendered]
    file_names = [r["file"] for r in rendered]

    has_truth = sum(1 for t in truths if t)
    print(f"         有真值标注: {has_truth}/{len(truths)}")

    # 3. OCR 识别（LaTeXify + DeepSeek 校正）
    print(f"[step 3/5] LaTeXify OCR 识别...")
    recognizer = OcrRecognizer(use_deepseek_refine=True)
    predictions = recognizer.batch_recognize(images)
    print(f"         识别完成: {len(predictions)} 个预测")

    # 4. 评估
    print(f"[step 4/5] 计算评估指标")
    results = evaluate_all(
        predictions=predictions,
        truths=truths,
        file_names=file_names,
    )
    print_results(results)

    # 保存预测结果
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = pred_dir / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "truth", "prediction", "exact_match", "cer", "bleu"])
        for detail in results["details"]:
            writer.writerow([
                detail.file, detail.truth_latex, detail.predicted_latex,
                detail.exact_match, f"{detail.cer:.4f}", f"{detail.bleu:.4f}",
            ])
    print(f"         预测结果已保存: {pred_csv}")

    # 5. 可视化
    print(f"[step 5/5] 生成可视化报告")
    report_files = create_report(
        results=results["details"],
        images=images,
        output_dir=output_dir / "figures",
    )
    for name, path in report_files.items():
        print(f"         {name}: {path.name}")

    print(f"\n{'=' * 50}")
    print(f"  OCR 流水线完成。输出目录: {output_dir}")
    print(f"{'=' * 50}")


def mode_evaluate(args: argparse.Namespace) -> None:
    """仅评估模式（从已有预测文件读取）。"""
    from src.evaluator import evaluate_all, print_results

    pred_file = _resolve(args.pred_file)
    if not pred_file.exists():
        print(f"[ERROR] 预测文件不存在: {pred_file}")
        sys.exit(1)

    predictions, truths, file_names = [], [], []
    with open(pred_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_names.append(row.get("file", ""))
            truths.append(row.get("truth", ""))
            predictions.append(row.get("prediction", ""))

    results = evaluate_all(predictions=predictions, truths=truths, file_names=file_names)
    print_results(results)

    # 保存详细结果
    out = pred_file.parent / "evaluation_results.json"
    import json
    serializable = {
        "accuracy": results["accuracy"],
        "mean_cer": results["mean_cer"],
        "mean_bleu": results["mean_bleu"],
        "total": results["total"],
        "details": [
            {
                "file": d.file,
                "truth_latex": d.truth_latex,
                "predicted_latex": d.predicted_latex,
                "exact_match": d.exact_match,
                "cer": d.cer,
                "bleu": d.bleu,
            }
            for d in results["details"]
        ],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"  详细结果已保存: {out}")


def main() -> None:
    args = parse_args()

    # 加载 .env
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if args.mode == "render":
        mode_render(args)
    elif args.mode == "pipeline":
        mode_pipeline(args)
    elif args.mode == "stroke_pipeline":
        mode_stroke_pipeline(args)
    elif args.mode == "ocr_pipeline":
        mode_ocr_pipeline(args)
    elif args.mode == "lora":
        mode_lora(args)
    elif args.mode == "lora_refine":
        args.refine = True
        mode_lora(args)
    elif args.mode == "decoder":
        mode_decoder(args, use_refine=False)
    elif args.mode == "decoder_refine":
        mode_decoder(args, use_refine=True)
    elif args.mode == "evaluate":
        mode_evaluate(args)
    else:
        print(f"[ERROR] 未知模式: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
