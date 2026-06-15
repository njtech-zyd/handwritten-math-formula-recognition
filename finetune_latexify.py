"""
LoRA 微调 LaTeXify — 让 TrOCR 适应手写 CROHME 公式。

用法:
    python finetune_latexify.py --train_dir data/raw/archive/TrainINKML_2013 --max_samples 1000 --epochs 5
    python finetune_latexify.py --eval_dir data/raw/archive/CROHME_training_2011 --max_samples 50 --do_eval
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType

from src.inkml_parser import load_dataset
from src.image_renderer import render_inkml
from src.evaluator import evaluate_all, print_results


class CROHMEDataset(Dataset):
    """CROHME 训练数据集：图像 → LaTeX。"""

    def __init__(self, data_dir, processor, max_samples=None, offset=0, img_size=448):
        self.processor = processor
        self.img_size = img_size
        dataset = load_dataset(data_dir, max_files=offset + max_samples if max_samples else None)
        dataset = dataset[offset:] if offset else dataset
        self.data = []
        for d in dataset:
            latex = (d.truth_latex or "").strip("$").strip()
            if latex:
                self.data.append((d, latex))
        print(f"  加载 {len(self.data)} 个样本 (offset={offset})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, latex = self.data[idx]
        img = render_inkml(data, img_size=self.img_size, stroke_width=4).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            latex, return_tensors="pt", padding="max_length", max_length=128, truncation=True
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel_values, "labels": labels}


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/raw/archive/CROHME_training_2011")
    parser.add_argument("--eval_dir", type=str, default="data/raw/archive/CROHME_training_2011")
    parser.add_argument("--output", type=str, default="checkpoints/latexify_lora")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--eval_offset", type=int, default=0, help="评估数据偏移量，避免与训练集重叠")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--resume", type=str, default="", help="已训练的 LoRA 权重路径")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，只做评估")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = timestamp()
    results_dir = Path(f"results/finetune_{ts}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"时间戳: {ts}")
    print(f"设备: {device}, 结果: {results_dir}")

    # 加载模型和处理器
    print("\n[1/4] 加载 LaTeXify...")
    t0 = time.time()
    processor = TrOCRProcessor.from_pretrained("tjoab/latex_finetuned")
    model = VisionEncoderDecoderModel.from_pretrained("tjoab/latex_finetuned")
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or 0
    model.config.pad_token_id = processor.tokenizer.pad_token_id or 0
    print(f"  完成 ({time.time()-t0:.1f}s, {sum(p.numel() for p in model.parameters()):,} params)")

    if args.resume or args.skip_train:
        # 直接加载已训练的 LoRA 权重
        from peft import PeftModel
        ckpt = args.resume or args.output
        model = PeftModel.from_pretrained(model, ckpt)
        model.to(device)
        print(f"  已加载 LoRA 权重: {ckpt}")
    else:
        # 应用 LoRA
        print("\n[2/4] 应用 LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "v_proj"],  # RoBERTa decoder attention layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.to(device)

    if not args.do_eval and not args.skip_train:
        # === 训练 ===
        print("\n[3/4] 加载训练数据...")
        train_ds = CROHMEDataset(args.train_dir, processor, max_samples=args.max_samples)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_batches = len(train_loader)

        print(f"\n开始训练: {args.epochs} epochs, {total_batches} batches/epoch")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            t_epoch = time.time()

            for batch_idx, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"  epoch [{epoch+1}/{args.epochs}] batch [{batch_idx+1}/{total_batches}] loss={loss.item():.4f}")

            avg_loss = total_loss / total_batches
            elapsed = time.time() - t_epoch
            print(f"  >>> epoch [{epoch+1}/{args.epochs}] avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")

        # 每轮保存 LoRA 权重
        epoch_dir = Path(f"{args.output}_ep{epoch+1}")
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        print(f"  >>> 权重已保存: {epoch_dir}")

    # === 加载已保存权重（如果 skip_train） ===
    if args.skip_train:
        from peft import PeftModel
        ckpt_path = args.resume or args.output
        model = PeftModel.from_pretrained(model, ckpt_path)
        model.to(device)
        print(f"  已加载 LoRA 权重: {ckpt_path}")

    # === 评估 ===
    print("\n[4/4] 评估...")
    model.eval()

    eval_ds = CROHMEDataset(args.eval_dir, processor, max_samples=args.eval_samples, offset=args.eval_offset)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)

    # 获取真值（使用偏移量避免训练集重叠）
    dataset = load_dataset(args.eval_dir, max_files=args.eval_offset + args.eval_samples)
    truths = [(d.truth_latex or "").strip("$").strip() for d in dataset[args.eval_offset:args.eval_offset + len(eval_ds)]]
    file_names = [str(d.file_path) if d.file_path else f"sample_{i}" for i, d in enumerate(dataset[:len(eval_ds)])]

    predictions = []
    for i, batch in enumerate(eval_loader):
        pixel_values = batch["pixel_values"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(pred)

        truth = truths[i] if i < len(truths) else ""
        match = "[OK]" if pred.replace(" ", "") == truth.replace(" ", "") else "[NO]"
        print(f"  [{i+1}/{len(eval_ds)}] {match} t={truth[:40]} p={pred[:40]}")

    # 评估指标
    results = evaluate_all(predictions=predictions, truths=truths, file_names=file_names)
    print_results(results)

    # 保存 CSV
    csv_path = results_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "truth", "prediction", "exact_match", "cer", "bleu"])
        for d in results["details"]:
            w.writerow([d.file, d.truth_latex, d.predicted_latex, d.exact_match, f"{d.cer:.4f}", f"{d.bleu:.4f}"])
    print(f"\n结果保存: {csv_path}")

    # 保存摘要
    accuracy = sum(1 for r in results["details"] if r.exact_match) / max(len(results["details"]), 1)
    summary_path = results_dir / "summary.txt"
    summary_path.write_text(
        f"LoRA Fine-tuning Results ({ts})\n"
        f"{'='*50}\n"
        f"Total: {len(results['details'])}\n"
        f"Exact Match: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
        f"Mean CER: {results['summary']['cer']:.4f}\n"
        f"Mean BLEU: {results['summary']['bleu']:.4f}\n"
        f"Train samples: {args.max_samples}, Epochs: {args.epochs}\n"
    )

    print(f"摘要保存: {summary_path}")


if __name__ == "__main__":
    main()
