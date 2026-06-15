# 对比学习 — 检索式手写公式识别

## 思路

使用预训练编码器（LaTeXify 的 ViT）提取图像特征向量，
然后通过**最近邻检索**从训练集中找到最相似的公式，直接返回其 LaTeX。

这与 OCR 流水线形成对比：

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **OCR（端到端）** | 图像 → 模型生成 LaTeX | 能处理未见过的公式 | 需要大量训练，生成可能不稳定 |
| **检索（对比学习）** | 图像 → 特征 → 匹配最相似的训练样本 | 简单可靠，可解释 | 只能返回已知的 LaTeX |

这两种方法本质上是**生成式 vs 检索式**的对比，是课设报告的绝佳素材。

## 文件结构

```
contrastive_learning/
├── __init__.py          # 模块说明
├── extract_features.py  # 用 ViT 编码器提取特征
├── index.py             # 检索索引（余弦相似度最近邻）
├── evaluate.py          # 完整评估流水线
└── README.md            # 本文档
```

## 用法

```bash
# 提取训练集特征并评估（推荐使用 .inkml 数据）
python -m contrastive_learning.evaluate \
    --train_dir data/raw/archive/CROHME_training_2011 \
    --test_dir data/raw/archive/CROHME_training_2011 \
    --max_train 200 \
    --max_test 50
```

参数说明：
- `--train_dir`：建索引用训练集
- `--test_dir`：评估用测试集
- `--max_train` / `--max_test`：限制样本数
- `--top_k`：Top-K 评估（默认 3）
- `--feature_cache`：复用缓存的 .npy 特征

## 与 OCR 流水线的对比实验

```bash
# 1. OCR 流水线（端到端生成）
python main.py --mode ocr_pipeline --data_dir data/raw/archive/CROHME_training_2011 --max_samples 50

# 2. 检索式识别（对比学习）
python -m contrastive_learning.evaluate \
    --train_dir data/raw/archive/CROHME_training_2011 \
    --test_dir data/raw/archive/CROHME_training_2011 \
    --max_train 200 --max_test 50
```

## 对比学习扩展

当前实现使用 LaTeXify 的预训练编码器（不做对比学习训练），
等同于"零样本"检索。要真正训练对比学习模型，可以：

1. 初始化一个 ViT 编码器
2. 对每个公式做数据增强 → 正样本对
3. 用 InfoNCE 损失训练（拉近正样本，推远负样本）
4. 用训练好的编码器替换 LaTeXify 的编码器

这样模型能学出对旋转/缩放不变性的特征，提升检索准确率。
