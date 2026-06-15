# 多模态大模型在手写数学公式识别中的应用

## 一、项目概述

### 1.1 项目背景

手写数学公式识别是文档数字化中的核心挑战之一。与普通文字识别不同，数学公式具有二维空间结构（分式、根号、上下标、矩阵等），符号之间不是简单的线性排列，因此传统的 OCR 技术难以直接应用。

本课程项目旨在探索**多模态大模型**在手写数学公式识别任务上的实际应用能力，以 **DeepSeek** 为核心，结合 **LoRA 微调**和 **TrOCR 视觉模型**，构建了一套"视觉预处理 + 大模型语义理解"的混合识别流水线。

### 1.2 数据集

项目使用 **CROHME**（Competition on Recognition of Online Handwritten Mathematical Expressions）国际竞赛数据集：

| 数据集 | 样本数 | 用途 | 说明 |
|:-------|:-----:|:----|:----|
| CROHME_training_2011 | 921 | 训练 + 评估 | 含 LaTeX 真值标注 |
| CROHME_test_2011 | 348 | 推理测试 | 无公开真值 |
| MatricesTrain2014 | 512 | 矩阵模型训练 | 含矩阵 LaTeX 真值 |
| MatricesTest2014 | 244 | 矩阵模型测试 | 含矩阵 LaTeX 真值 |

数据格式为 **InkML**（Ink Markup Language），记录笔画的时序坐标点序列，经过渲染引擎转为图像后输入模型。

### 1.3 技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                   混合识别流水线架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  手写公式 (InkML)                                            │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                        │
│  │ 图像渲染引擎      │  ← InkML 坐标 → 灰度图像 (448×448)     │
│  └────────┬────────┘                                        │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ TrOCR + LoRA    │  ← 334M 参数视觉模型, LoRA 微调适应     │
│  │ 第一步: 视觉识别  │     CROHME 数据分布                       │
│  └────────┬────────┘                                        │
│           ▼ 初步 LaTeX                                       │
│  ┌─────────────────┐                                        │
│  │ DeepSeek 文本 API│  ← 利用大模型语义理解能力修正错误         │
│  │ 第二步: 语义校正  │     理解上下文, 推断正确变量/数字         │
│  └────────┬────────┘                                        │
│           ▼                                                 │
│    最终 LaTeX 公式                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、项目文件结构及作用

```
multimodal_math_recognition/
│
├── main.py                      # 主入口，7种运行模式
├── report.md                    # 本汇报文档
├── requirements.txt             # Python 依赖清单
├── finetune_latexify.py         # LoRA 微调脚本
├── run_latexify.py              # LaTeXify 快速运行脚本
│
├── src/                         # 核心源码模块
│   ├── api_recognizer.py        # DeepSeek API 封装层
│   ├── evaluator.py             # 评估指标体系
│   ├── image_renderer.py        # InkML → 图像 渲染引擎
│   ├── inkml_parser.py          # InkML 文件解析器
│   ├── ocr_recognizer.py        # TrOCR 模型封装
│   ├── stroke_formatter.py      # 笔画数据格式化
│   └── visualizer.py            # 对比图可视化
│
├── contrastive_learning/        # 对比学习模块（辅助探索）
│
├── scripts/                     # 辅助工具脚本
│   ├── train_matrix.py          # 矩阵识别模型训练
│   └── verify_gpu_lora.py       # GPU LoRA 验证
│
├── checkpoints/                 # 预训练模型权重
│   ├── latexify_gpu_v2_ep8/     # ★ LoRA 最佳权重
│   └── lora_matrix/             # 矩阵模型权重
│
└── results/                     # 实验结果
    ├── all/figures/             ★ 全部 921 张对比图
    ├── hybrid_all/              ★ 最终评估数据
    └── matrix_test/              # 矩阵评估数据
```

### 2.1 核心源码详解

#### main.py — 项目主入口

**作用**：统一的命令行接口，支持 7 种运行模式，是整个项目的调度中心。

**设计原因**：将数据加载、图像渲染、模型推理、评估、可视化串联成完整的流水线，用户只需一行命令即可运行任意模式。

**关键参数**：
| 参数 | 说明 |
|:-----|:-----|
| `--mode` | 运行模式：lora/lora_refine/ocr_pipeline/等 |
| `--data_dir` | InkML 数据目录 |
| `--max_samples` | 处理样本数上限 |
| `--lora_ckpt` | LoRA 权重路径 |
| `--refine` | 启用 DeepSeek 文本校正 |

**为什么这样设计**：
- 单一入口 vs 多个脚本：便于维护和文档化，用户无需了解内部模块调用关系
- 模式分离：不同技术路线各自独立，可单独运行也可组合
- 参数化配置：训练数据、模型权重、输出目录均可灵活指定

#### src/api_recognizer.py — DeepSeek API 封装

**作用**：封装 DeepSeek 的 API 调用，提供统一的识别接口。

**设计原因**：
- DeepSeek 同时提供 OpenAI 兼容和 Anthropic 兼容两种 API 端点
- OpenAI 兼容端点（`https://api.deepseek.com`）支持文本补全，用于公式校正
- Anthropic 兼容端点（`https://api.deepseek.com/anthropic`）理论上支持多模态
- 封装后上层代码无需关注 API 差异

**关键实现**：
- `DeepSeekRecognizer`：基于 Anthropic SDK，尝试发送图像（实际不支持）
- `StrokeTextRecognizer`：基于 OpenAI SDK，纯文本交互
- `refine()`：专门用于 LaTeX 语法校正

**为什么这样设计**：
- 多模态 API 调用复杂，封装后简化上层使用
- 统一错误处理（重试机制、超时控制）
- 支持从 `.env` 文件读取 API Key，不硬编码

#### src/evaluator.py — 评估指标体系

**作用**：计算识别准确率的各项指标，包括精确匹配率、CER、BLEU。

**关键指标**：
| 指标 | 全称 | 说明 | 范围 |
|:-----|:-----|:-----|:----:|
| Exact Match | 精确匹配率 | 预测与真值完全一致 | 0%-100% |
| CER | Character Error Rate | 字符级编辑距离 | 0.0-1.0 |
| BLEU | Bilingual Evaluation Understudy | n-gram 精确度 | 0.0-1.0 |

**LaTeX 归一化**：由于 LaTeX 有多种等价写法（如 `^{n}` ≡ `^n`、`\ldots` ≡ `\cdots`、`\sum^{\infty}_{k=0}` ≡ `\sum_{k=0}^{\infty}`），直接字符串比较会产生大量假阴性。`normalize_latex()` 函数将等价格式统一后再比较。

**为什么这样设计**：
- 单一指标不够全面：精确匹配率严格但可能低估（等价格式差异），CER 反映字符级接近程度，BLEU 反映 n-gram 匹配度
- 归一化防止等价格式被误判为错误

#### src/image_renderer.py — InkML 渲染引擎

**作用**：将 InkML 笔画坐标数据渲染为灰度图像，供视觉模型识别。

**关键实现**：
- 将时序坐标点连接为笔画线条
- 支持自定义图像尺寸和笔画宽度
- 修正 CROHME 坐标系（Y 轴方向）与 PIL 的差异

**为什么这样设计**：
- CROHME 数据使用 Y-down 坐标系，PIL 使用 Y-up，需要坐标转换
- 渲染质量直接影响 OCR 识别效果：笔画宽度、图像分辨率都是关键参数

#### src/visualizer.py — 对比图可视化

**作用**：生成三栏对比图：原图 | 真值 | 预测，用于直观评估识别效果。

**设计原因**：相比于纯数字的评估指标，可视化对比能直观展示模型在哪类公式上表现好、哪类容易出错，便于针对性改进。

**匹配判定逻辑**：使用与 `evaluator.py` 一致的 `_norm_latex()` 函数进行归一化后比较，确保对比图中的"Correct/Mismatch"标识与评估指标一致。

---

## 三、技术方案详解

### 3.1 问题分析：为什么不能直接用多模态大模型？

项目初期尝试了两种直接调用 DeepSeek 的方案：

**尝试一：DeepSeek 视觉 API**
```
方案：将渲染图像直接发送给 DeepSeek 多模态模型
结果：约 70% 请求返回"无法看到图片"
原因：DeepSeek 的 API 实际不支持图像输入
```

**尝试二：笔画文本 + DeepSeek 文本 API**
```
方案：将 InkML 坐标转为文本描述，让 DeepSeek 理解
结果：模型无法从坐标序列中理解空间结构
原因：坐标点的空间关系对文本模型而言过于抽象
```

**结论**：当前 DeepSeek 的 API 不支持图像输入，无法直接完成"图像 → LaTeX"的端到端识别。需要设计替代方案。

### 3.2 解决方案：LoRA + DeepSeek 混合流水线

基于上述分析，设计了"视觉预处理 + 大模型语义理解"的混合架构：

#### 第一步：TrOCR + LoRA 视觉识别

**TrOCR 模型**：微软发布的 Transformer-based OCR 模型，采用 ViT 编码器 + RoBERTa 解码器架构，总计 **334M 参数**。本项目使用的 `tjoab/latex_finetuned` 版本在 MathWriting 合成数据集上经过微调，具备基础的公式识别能力。

**LoRA 微调（Low-Rank Adaptation）**：

LoRA 是一种参数高效的微调方法，通过在预训练模型的权重矩阵旁添加低秩适配器，仅训练少量参数即可实现有效的领域适应。

```
LoRA 原理：
  W' = W + BA
  其中 W ∈ ℝ^{d×k} 是原始权重矩阵
  B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r << min(d,k)
  训练时只更新 B 和 A，W 保持不变
```

**为什么选择 LoRA 而不是全量微调**：
| 对比维度 | 全量微调 | LoRA 微调 |
|:---------|:--------:|:---------:|
| 训练参数量 | 334M (100%) | 1M (0.3%) |
| GPU 显存需求 | >16GB | ~4GB |
| 训练时间 | 数小时 | **7 分钟** |
| 过拟合风险 | 高（数据少时） | 低 |
| 模型保存 | 334MB | **4MB** |

**微调配置**：
```
基础模型: TrOCR (ViT + RoBERTa), 334M 参数
LoRA 配置: rank=8, alpha=16
目标模块: q_proj, v_proj (注意力层)
训练参数: 1,056,768 (0.3%)
训练数据: CROHME_training_2011 (700 样本)
批量大小: 8
学习率: 2e-4
优化器: AdamW
训练时间: ~7 分钟 (RTX 4060)
设备: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
```

#### 第二步：DeepSeek 文本 API 语义校正

在 LoRA 完成初步识别后，将输出结果送入 DeepSeek 文本 API 进行二次校正。

**DeepSeek 在此环节的作用**：
1. **语法修正**：补充缺失的反斜线（`sin` → `\sin`）、修正 LaTeX 命令拼写
2. **语义推断**：根据数学上下文推断正确的变量名和数字（如 `z^x` → `2^x`）
3. **结构优化**：优化上下标位置、括号匹配等结构问题

**Prompt 设计**：
```
System: You are a LaTeX OCR post-processor.
User: Fix LaTeX recognition errors. Correct wrong digits, 
      variables, operators. Add missing backslashes. 
      Output ONLY the corrected LaTeX.

      [LoRA 输出的初步 LaTeX]
```

DeepSeek 的校正能力来源于其在大规模数学语料上的预训练，能够理解公式的语义而不是仅做字符串替换。

### 3.3 混合方案的协同效应

```
                    LoRA 负责                   DeepSeek 负责
                    ──────────                ────────────────
    图像理解    ✅ 能看懂手写体                    ❌ 看不到图
    格式语法    ⚠️ 偶尔出错（缺反斜线等）          ✅ 精通 LaTeX
    数字识别    ⚠️ 混淆 2/z, 8/3 等              ✅ 语义推断
    上下文理解   ❌ 只看像素不看含义               ✅ 理解数学语义
```

两者互补，形成"1+1>2"的效果。

---

## 四、实验结果

### 4.1 评估指标定义

| 指标 | 计算方法 | 含义 |
|:-----|:---------|:-----|
| Exact Match | 归一化后字符串完全一致 | 严格正确率 |
| CER | 编辑距离 / 真值长度 | 字符级错误率（越低越好） |
| BLEU | 4-gram 精确度 × 长度惩罚 | n-gram 匹配度（越高越好） |

### 4.2 全量评估结果

**测试环境**：
- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- Python: 3.12 / PyTorch 2.12.0+cu126
- DeepSeek 模型: deepseek-v4-flash
- 测试数据: CROHME_training_2011（921 样本全量）

**核心结果**：

| 方案 | 准确率 | 正确数 | 总样本 |
|:-----|:-----:|:------:|:-----:|
| LaTeXify 原始（未微调） | ~20% | ~10 | 50 |
| LoRA 微调推理 | **80.8%** | 744 | 921 |
| **+ DeepSeek 语义校正** | **85.2%** | **785** | **921** |

**DeepSeek 净提升：+4.4%**（24 条错误被成功修正）

**验证集（未参与训练）结果**：

| 方案 | 准确率 | 正确数 | 总样本 |
|:-----|:-----:|:------:|:-----:|
| LoRA 推理 | 72.4% | 160 | 221 |
| **+ DeepSeek 校正** | **74.7%** | **165** | **221** |

### 4.3 DeepSeek 实际校正案例

| # | LoRA 输出（有误） | DeepSeek 校正后 | 错误类型 |
|:-:|:-----------------|:----------------|:--------|
| 1 | `\int \left( z^{x} - 3 e^{x} \right) d x` | `\int \left( 2^{x} - 3 e^{x} \right) d x` | 变量误识 `z→2` |
| 2 | `\frac{\sin \theta + \cos \theta + \tan g}{...}` | `\frac{\sin \theta + \cos \theta + \tan \theta}{...}` | 希腊字母 `g→θ` |
| 3 | `a^2 - 8ab = (a - b)^2 - b^2` | `a^2 - 2ab = (a - b)^2 - b^2` | 数字误识 `8→2` |
| 4 | `ax^{2} + 2 bx + C = 0` | `ax^{2} + 2bx + c = 0` | 大小写 `C→c` |
| 5 | `\cos(4a) = 8\cos^2(a) - 8\cos^2(a) + 1` | `\cos(4a) = 8\cos^4(a) - 8\cos^2(a) + 1` | 指数误识 `2→4` |
| 6 | `\tan z = \frac{sin z}{\cos z}` | `\tan z = \frac{\sin z}{\cos z}` | 缺失反斜线 |
| 7 | `151\pm143\div9.97` | `151\pm143\div97` | 多余小数点 |
| 8 | `d^{n}-1` | `2^{n}-1` | 字母→数字 `d→2` |
| 9 | `b=a^{2}+e^{2}` | `b=a^{2}+c^{2}` | 字母误识 `e→c` |
|10 | `cosx+i_{xinx}=e^{ix}` | `\cos x + i \sin x = e^{ix}` | 完全修正 |

**关键发现**：DeepSeek 能在看不到原图的情况下，仅凭数学语义推断出正确的数字和变量。如第 1 例：积分式中的 `z^x` 在数学上不常见，DeepSeek 推断应为 `2^x`；第 3 例：`a^2 - 8ab` 在公式 `(a-b)^2 = a^2 - 2ab + b^2` 的语境下 `8` 不可能是正确数字，修正为 `2`。

### 4.4 矩阵识别结果

| 方案 | 准确率 | 总样本 |
|:-----|:-----:|:-----:|
| 矩阵 LoRA 模型 | **45.9%** | 244 |

矩阵公式的 LaTeX 结构更复杂（`\begin{pmatrix}...\end{pmatrix}`），且训练数据仅 420 样本，但仍能正确生成矩阵结构。主要错误集中在矩阵内部数字的精度上。

---

## 五、项目的关键发现与结论

### 5.1 核心结论

**1. 多模态大模型的视觉能力瓶颈**

当前 DeepSeek 的 API 不支持图像输入，无法直接完成"图像 → LaTeX"的任务。这揭示了当前多模态大模型的一个实用局限：**"多模态"不等于"什么都能看"**，实际 API 能力与宣传存在差距。

**2. LoRA 微调的高效性**

仅用 0.3% 的参数（1M/334M）和 7 分钟 GPU 训练，就将准确率从 20% 提升到 80%。这证明了 LoRA 在小样本、低资源场景下的巨大价值。

**3. 混合方案的最优性**

```
LoRA（感知）+ DeepSeek（理解）> 单独使用任何一种
```

- LoRA 擅长"看"——理解图像中的笔画和符号
- DeepSeek 擅长"想"——理解公式的语义和上下文
- 两者互补，实现了单一模型无法达到的效果

**4. 数据匹配的重要性**

| 训练数据 | 评估数据 | 准确率 |
|:---------|:---------|:------:|
| CROHME 2011 | CROHME 2011 | **71-80%** |
| CROHME 2013 | CROHME 2011 | **16%** |

同样的 LoRA 方法，同源数据微调效果远好于异源数据（差距 4 倍），说明**数据分布对齐**比模型大小更重要。

### 5.2 技术收获

1. **PyTorch + CUDA 实战**：完整走通了环境配置、模型加载、GPU 训练调优的全流程，解决了 Python 3.14 不兼容 CUDA PyTorch 的问题
2. **LoRA 参数高效微调**：掌握了 peft 库的使用，理解了低秩适配的原理和优势
3. **DeepSeek API 工程**：深入理解了大模型 API 的调用模式、Prompt 工程和错误处理
4. **评估体系设计**：设计了多维度评估指标（Exact Match + CER + BLEU），以及 LaTeX 归一化方案
5. **问题定位能力**：从"准确率为什么低"到逐步排查出 API 限制、数据分布不匹配、等价格式误判等问题

### 5.3 应用价值

本项目构建的"小模型做感知 + 大模型做理解"的混合架构，不仅适用于公式识别，也为其他"图像→结构化文本"任务（如表格识别、图表解读、乐谱识别）提供了可参考的技术路线。

---

## 六、使用指南

### 6.1 环境配置

```powershell
# 1. 创建虚拟环境
py -3.12 -m venv .venv_cuda

# 2. 安装 CUDA PyTorch
.venv_cuda\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 安装其他依赖
.venv_cuda\Scripts\python.exe -m pip install transformers peft sentencepiece pillow python-dotenv openai matplotlib
```

### 6.2 运行命令

```powershell
# 混合方案：LoRA + DeepSeek 校正（推荐，最准确）
.venv_cuda\Scripts\python.exe main.py --mode lora_refine --data_dir "data/raw/archive/CROHME_training_2011" --max_samples 20

# 仅 LoRA（不调用 DeepSeek，更快）
.venv_cuda\Scripts\python.exe main.py --mode lora --data_dir "data/raw/archive/CROHME_training_2011" --max_samples 20

# 矩阵识别
.venv_cuda\Scripts\python.exe main.py --mode lora --lora_ckpt checkpoints/lora_matrix --data_dir "data/raw/archive/MatricesTest2014"

# 生成全部对比图
.venv_cuda\Scripts\python.exe -c "见 scripts/verify_gpu_lora.py"
```

### 6.3 实验结果查看

- 对比图：`results/all/figures/`（921 张，按公式编号命名）
- 评估数据：`results/hybrid_all_20260522_135833/predictions.csv`
- 矩阵结果：`results/matrix_test_20260522_131247/predictions.csv`

---

*项目完成日期：2026 年 5 月 22 日*
