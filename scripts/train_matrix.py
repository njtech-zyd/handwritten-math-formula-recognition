"""训练矩阵专用 LoRA 模型"""
import os, sys, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType
from src.inkml_parser import load_dataset
from src.image_renderer import render_inkml

print("加载矩阵数据...")
dataset = load_dataset("data/raw/archive/MatricesTrain2014")
train, val = dataset[:420], dataset[420:]
print(f"训练: {len(train)}, 验证: {len(val)}")

print("加载模型...")
proc = TrOCRProcessor.from_pretrained("tjoab/latex_finetuned")
base = VisionEncoderDecoderModel.from_pretrained("tjoab/latex_finetuned")
model = get_peft_model(base, LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","out_proj"],
    lora_dropout=0.1, task_type=TaskType.SEQ_2_SEQ_LM
))
model.to("cuda")
model.print_trainable_parameters()

opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
for ep in range(20):
    model.train()
    loss_sum = 0
    t0 = time.time()
    for i, d in enumerate(train):
        img = render_inkml(d, img_size=448, stroke_width=4).convert("RGB")
        pv = proc(images=img, return_tensors="pt").pixel_values.to("cuda")
        latex = (d.truth_latex or "").strip("$").strip()
        labels = proc.tokenizer(latex, return_tensors="pt", padding="max_length", max_length=256, truncation=True).input_ids.to("cuda")
        loss = model(pixel_values=pv, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        loss_sum += loss.item()
        if (i + 1) % 50 == 0:
            print(f"  ep{ep+1} [{i+1}/{len(train)}] loss={loss_sum/(i+1):.4f}")
            sys.stdout.flush()
    print(f">>> epoch {ep+1}: avg_loss={loss_sum/len(train):.4f} ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

model.save_pretrained("checkpoints/lora_matrix")
print("矩阵模型已保存: checkpoints/lora_matrix/")
