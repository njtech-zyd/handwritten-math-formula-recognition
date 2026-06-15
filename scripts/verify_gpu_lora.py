"""
GPU LoRA 验证脚本 — 运行测试并生成对比图。

用法:
    python scripts/verify_gpu_lora.py
"""

import sys, os, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from src.inkml_parser import load_dataset
from src.image_renderer import render_inkml
from src.evaluator import evaluate_all, print_results
from src.visualizer import plot_comparison
from pathlib import Path
import matplotlib.pyplot as plt

ts = time.strftime("%Y%m%d_%H%M%S")
out = Path(f"results/verify_{ts}")
out.mkdir(parents=True, exist_ok=True)
fig_dir = out / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

print(f"GPU LoRA 验证 {ts}")
device = "cuda"

# 加载模型
print("加载模型...")
processor = TrOCRProcessor.from_pretrained("tjoab/latex_finetuned")
base = VisionEncoderDecoderModel.from_pretrained("tjoab/latex_finetuned")
model = PeftModel.from_pretrained(base, "checkpoints/latexify_gpu_v2_ep8")
model.to(device)
model.eval()
print("模型加载完成 (GPU)\n")

# 加载测试数据
dataset = load_dataset("data/raw/archive/CROHME_training_2011")
test = dataset[700:720]  # 20个未见过的样本
print(f"测试 {len(test)} 个样本 (ID 700-719)...\n")

preds, truths, files, images = [], [], [], []
correct, total = 0, 0

for i, d in enumerate(test):
    img = render_inkml(d, img_size=448, stroke_width=4).convert("RGB")
    pv = processor(images=img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel_values=pv, max_length=128)
    pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
    truth = (d.truth_latex or "").strip("$").strip()
    match = pred.replace(" ", "") == truth.replace(" ", "")
    if match:
        correct += 1
    total += 1
    preds.append(pred)
    truths.append(truth)
    files.append(str(d.file_path))
    images.append(img.convert("L"))
    mark = "[OK]" if match else "[NO]"
    print(f"  {mark} #{i:2d} t={truth[:50]:50s} p={pred[:50]}")

print(f"\n{'='*50}")
print(f"  结果: {correct}/{total} = {correct/total*100:.1f}%")
print(f"{'='*50}\n")

# 生成对比图
print(f"生成对比图到 {fig_dir}/ ...")
for i in range(len(test)):
    fname = Path(files[i]).stem
    fig = plot_comparison(
        images[i], truths[i], preds[i],
        file_name=fname,
        save_path=fig_dir / f"compare_{i:02d}.png",
    )
    plt.close(fig)

print(f"完成！共 {len(test)} 张对比图")
print(f"查看方式：打开 {fig_dir}/ 目录查看图片")
