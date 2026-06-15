# 多模态大模型在手写数学公式识别中的应用

本项目基于 **DeepSeek + LoRA 微调** 构建了一套混合识别流水线，用于识别 CROHME 数据集中的手写数学公式，最终准确率达到 **85.2%**。

---

## ⚠️ 使用前必读：缺失文件说明

由于文件大小和隐私限制，以下内容**未包含在 GitHub 仓库中**，需要手动配置：

### 1. 数据集（未上传）

| 数据集 | 位置 | 获取方式 |
|:-------|:-----|:---------|
| CROHME_training_2011 | `data/raw/archive/CROHME_training_2011/` | 从 CROHME 官网下载 |
| CROHME_test_2011 | `data/raw/archive/CROHME_test_2011/` | 同上 |
| MatricesTrain2014 | `data/raw/archive/MatricesTrain2014/` | 同上 |
| MatricesTest2014 | `data/raw/archive/MatricesTest2014/` | 同上 |


数据集目录结构应为：
```
data/raw/archive/
├── CROHME_training_2011/     # 921 个 .inkml 文件
├── CROHME_test_2011/         # 348 个 .inkml 文件
├── MatricesTrain2014/        # 512 个 .inkml 文件
└── MatricesTest2014/         # 244 个 .inkml 文件
```

### 2. DeepSeek API Key（隐私，未上传）

在项目根目录创建 `.env` 文件：
```
DEEPSEEK_API_KEY=sk-你的API密钥
```

申请地址：https://platform.deepseek.com/

### 3. 模型权重（文件过大，未上传）

| 权重文件 | 作用 | 获取方式 |
|:---------|:-----|:---------|
| `checkpoints/latexify_gpu_v2_ep8/adapter_model.safetensors` | ★ LoRA 最佳权重（71%→85%） | 自行训练 或 联系作者 |
| `checkpoints/lora_matrix/adapter_model.safetensors` | 矩阵识别模型（45.9%） | 自行训练 或 联系作者 |
| `checkpoints/encoder_final.pt` | 对比学习编码器（未在最终方案使用） | 自行训练 |
| `checkpoints/classifier.pt` | 结构分类器（未在最终方案使用） | 自行训练 |

**训练 LoRA 权重（使用 GPU，约 7 分钟）**：
```powershell
.venv_cuda\Scripts\python.exe finetune_latexify.py ^
    --train_dir "data/raw/archive/CROHME_training_2011" ^
    --max_samples 700 --epochs 8 --batch_size 8
```

### 4. 虚拟环境 + CUDA PyTorch（未上传）

需要自行创建：
```powershell
# 需要 Python 3.12（不支持 3.14，PyTorch CUDA 不兼容）
py -3.12 -m venv .venv_cuda

# 安装 CUDA PyTorch（如果有无 GPU，可用 CPU 版本但速度较慢）
.venv_cuda\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖
.venv_cuda\Scripts\python.exe -m pip install transformers peft sentencepiece pillow python-dotenv openai matplotlib
```

### 5. 结果文件（未上传）

`results/` 目录包含所有评估结果和 921 张对比图，因数量过多未上传。运行后会自动生成。

### 6. 特征缓存文件（未上传）

`data_cache_full/` 目录包含 17K InkML 文件的预提取特征，用于对比学习训练。运行 `build_dataset.py` 可重新生成。

---

## 🚀 快速开始

### 1. 环境准备
```powershell
# 配置 API Key
echo DEEPSEEK_API_KEY=sk-your-key > .env
```

### 2. 运行混合识别流水线（推荐）
```powershell
.venv_cuda\Scripts\python.exe main.py --mode lora_refine --data_dir "data/raw/archive/CROHME_training_2011" --max_samples 20
```

### 3. 仅 LoRA（不调用 DeepSeek，更快）
```powershell
.venv_cuda\Scripts\python.exe main.py --mode lora --data_dir "data/raw/archive/CROHME_training_2011" --max_samples 20
```

### 4. 矩阵识别
```powershell
.venv_cuda\Scripts\python.exe main.py --mode lora --lora_ckpt checkpoints/lora_matrix --data_dir "data/raw/archive/MatricesTest2014"
```

### 5. 全量评估
```powershell
.venv_cuda\Scripts\python.exe scripts\train_matrix.py     # 训练矩阵模型
.venv_cuda\Scripts\python.exe scripts\verify_gpu_lora.py  # 生成对比图
```

---

## 📁 完整项目文件结构及功能说明

```
multimodal_math_recognition/
│
├── 📄 项目入口
│   ├── main.py                      # 主流水线入口（~26KB）
│   │   └── 支持7种模式：pipeline / stroke_pipeline / ocr_pipeline /
│   │       lora / lora_refine / decoder / render
│   ├── finetune_latexify.py         # LoRA 微调脚本（~9KB）
│   │   └── TrOCR + LoRA (rank=8) 在 CROHME 上微调，7分钟训练
│   └── run_latexify.py              # LaTeXify 快速运行脚本（~7KB）
│
├── 📦 核心源码模块（src/）
│   ├── inkml_parser.py        # InkML 解析器（~7KB）
│   │   └── 解析XML→提取笔画坐标轨迹+LaTeX真值标签
│   ├── image_renderer.py      # 图像渲染引擎（~5KB）
│   │   └── 笔画坐标→448×448灰度图像，自适应缩放居中
│   ├── api_recognizer.py      # DeepSeek API 封装（~13KB）
│   │   └── DeepSeekRecognizer / StrokeTextRecognizer / OcrRecognizer
│   ├── ocr_recognizer.py      # TrOCR 模型封装（~4KB）
│   │   └── HuggingFace TrOCR 统一识别接口
│   ├── evaluator.py           # 评估指标体系（~8KB）
│   │   └── Exact Match / CER / BLEU + LaTeX归一化（5类等价差异）
│   ├── visualizer.py          # 对比图可视化（~10KB）
│   │   └── 三栏对比图 / 混淆矩阵 / CER分布 / 报告自动生成
│   └── stroke_formatter.py    # 笔画数据格式化（~6KB）
│       └── 坐标序列→文本描述，用于DeepSeek文本API输入
│
├── 🔬 对比学习模块（contrastive_learning/）
│   ├── train_encoder.py       # 对比学习编码器训练（~13KB）
│   │   └── 4层CNN：448×448→128维，InfoNCE损失
│   ├── train_classifier.py    # 结构分类器训练（~6KB）
│   │   └── 预测公式结构属性（分式/根号/矩阵等）
│   ├── decoder.py             # LaTeX 解码器（~4KB）
│   │   └── Transformer解码器，4.3M参数
│   ├── latex_tokenizer.py     # LaTeX 分词器（~3KB）
│   │   └── 122个LaTeX token词表
│   ├── latex_parser.py        # LaTeX 结构解析器（~4KB）
│   ├── build_dataset.py       # 特征缓存构建（~5KB）
│   │   └── 预提取17K InkML的ViT特征
│   ├── refine_pipeline.py     # 特征增强流水线（~8KB）
│   │   └── 编码器提纯→结构分析→增强Prompt→DeepSeek
│   └── README.md              # 对比学习说明文档
│
├── 🛠️ 工具脚本（scripts/）
│   ├── train_matrix.py        # 矩阵模型训练（~2KB）
│   │   └── rank=16，含k_proj/out_proj，420样本20轮
│   └── verify_gpu_lora.py     # GPU LoRA验证（~3KB）
│       └── 全量评估→921张对比图+summary
│
├── 📐 传统机器学习路线
│   ├── preprocess.py          # 数据预处理（~8KB）
│   │   └── InkML→48×48图像→数据增强→Top-N频率过滤
│   ├── traditional_model.py   # 主训练脚本（~26KB）
│   │   └── HOG+LBP→PCA→SVM + 14张可视化图生成
│   ├── quick_test.py          # 快速验证（~12KB）
│   │   └── Top-10类，~8分钟快速跑通
│   ├── visualize_predictions.py # 预测可视化（~7KB）
│   │   └── 每类展示正确和错误的预测样本
│   ├── demo_one_sample.py     # 单样本演示（~2KB）
│   ├── finish_report.py       # 实验报告补全（~5KB）
│   ├── generate_report_docx.py # Word报告生成（~19KB）
│   └── experiments/
│       ├── figures/           # 14张可视化分析图
│       ├── 个人工作报告.md     # 实验报告（Markdown）
│       ├── 个人工作报告.docx   # 实验报告（Word）
│       └── report_*.json      # 实验数据
│
├── 📄 配置与文档
│   ├── README.md              # 本文件
│   ├── requirements.txt       # Python 依赖清单（~288B）
│   └── .gitignore             # Git 忽略规则
│
└── 📂 排除文件（.gitignore，需自行准备）
    ├── .env                   # DeepSeek API Key
    ├── data/                  # CROHME 数据集（~22K InkML文件）
    ├── checkpoints/           # 模型权重（LoRA适配器~4MB）
    ├── data_cache_full/       # 特征缓存
    └── results/               # 评估结果（921张对比图+CSV）
```

## 📊 预期结果

| 方案 | 准确率 | 说明 |
|:-----|:------:|:-----|
| LoRA 推理 | **80.8%** | 921 样本全量 |
| **+ DeepSeek 校正** | **85.2%** | 921 样本全量 |
| 矩阵识别 | **45.9%** | 244 样本 |

---

## 📧 联系方式

如有问题请联系项目作者。
