# Project 5: 多模态情感分类

**学号**：10235501402  
**姓名**：李则言  
**GitHub 仓库**：[https://github.com/ZyanNo1/Project5-of-Modern-AI](https://github.com/ZyanNo1/Project5-of-Modern-AI)

---

## 项目概述

本项目基于 **CLIP (Contrastive Language-Image Pre-training)** 构建多模态情感分类模型，对配对的文本与图像数据进行三分类预测（Positive / Neutral / Negative）。核心创新包括：

- **两阶段渐进微调策略**：先冻结 CLIP 训练分类头，再轻量微调文本编码器最后 1 层
- **多种融合机制对比**：Concat / Gated / FiLM / Cross-Attention 的系统性探索
- **类别不均衡处理**：平方根平滑的类别加权损失函数
- **完整消融实验**：验证多模态融合相比单模态的优势

最终模型在内部测试集上达到 **76.3% Accuracy** 和 **64.2% Macro F1**。

---

## 目录结构

```
Project5/
├── data/                          # 数据目录（需自行解压）
│   ├── data/                      # 图像和文本文件
│   ├── train.txt                  # 训练集标签（guid,tag）
│   └── test_without_label.txt     # 测试集（无标签）
├── src/                           # 源代码
│   ├── train.py                   # 训练脚本
│   ├── evaluate.py                # 评估脚本（内部测试集）
│   ├── infer.py                   # 推理脚本（生成提交文件）
│   ├── ablations.py               # 消融实验自动化脚本
|   ├── analyze_runs.py            # 汇总实验结果
│   ├── model.py                   # 模型定义（Late Fusion + 多种融合机制）
│   ├── datasets.py                # 数据集与 DataLoader
│   ├── transform.py               # 图像/文本预处理
│   └── utils.py                   # 工具函数（日志、指标、绘图等）
├── outputs/                       # 训练输出（自动生成）
│   ├── run_YYYYMMDD_HHMMSS/       # 每次训练的独立目录
│   │   ├── hparams.json           # 超参数配置
│   │   ├── best_checkpoint.pth    # 最优模型权重
│   │   ├── splits/                # 固化的数据划分
│   │   │   ├── train_split.csv
│   │   │   ├── val_split.csv
│   │   │   └── test_split.csv
│   │   ├── eval/                  # 评估结果
│   │   │   └── internal_test_metrics.json
│   │   └── training_curves.png    # 训练曲线
│   └── run_summary.csv            # 所有实验的汇总表
├── requirements.txt               # 依赖列表
├── README.md                      # 本文件
└── report.pdf                     # 实验报告
```

---

## 环境配置

### 1. 系统要求

- **操作系统**：Linux / macOS / Windows（推荐 Linux）
- **GPU**：NVIDIA GPU with CUDA 11.8+ （推荐，CPU 可运行但速度慢）
- **显存**：至少 4GB
- **硬盘空间**：约 5GB（模型权重 + 数据集）

### 2. 克隆仓库

```bash
git clone https://github.com/ZyanNo1/Project5-of-Modern-AI.git
cd Project5-of-Modern-AI
```

### 3. 安装依赖

**Python 版本**：3.8+（推荐 3.9）

```bash
pip install -r requirements.txt
```

**核心依赖**：

- `torch>=2.0.0`（支持 CUDA 11.8+）
- `transformers>=4.30.0`（CLIP 模型）
- `Pillow>=9.0.0`（图像处理）
- `pandas`, `numpy`, `scikit-learn`（数据处理与评估）
- `matplotlib`, `seaborn`（可视化）

### 4. 数据准备

将 `实验五数据.zip` 解压到项目根目录：

```bash
unzip 实验五数据.zip
```

解压后目录结构：
```
data/
├── data/
│   ├── 0a0b1c2d.jpg
│   ├── 0a0b1c2d.txt
│   └── ...
├── train.txt
└── test_without_label.txt
```

---

## 快速开始

### 训练主模型（Concat + 两阶段）

```bash
python src/train.py \
  --data_dir ./data/data \
  --train_txt ./data/train.txt \
  --clip_model openai/clip-vit-base-patch32 \
  --fusion concat \
  --two_stage \
  --stage1_epochs 5 \
  --stage2_epochs 5 \
  --unfreeze_text \
  --unfreeze_text_layers 1 \
  --lr_head 3e-4 \
  --lr_clip 5e-6 \
  --hidden_dim 256 \
  --hidden_dim2 64 \
  --dropout 0.5 \
  --weight_decay 0.05 \
  --batch_size 16 \
  --seed 42 \
  --early_stop_patience 3
```

**训练输出**：模型权重和日志保存在 `outputs/run_YYYYMMDD_HHMMSS/`

---

### 评估模型（内部测试集）

```bash
python src/evaluate.py \
  --checkpoint ./outputs/run_20260118_162451/best_checkpoint.pth \
  --split_dir ./outputs/run_20260118_162451/splits \
  --data_dir ./data/data \
  --batch_size 16
```

**输出示例**：
```
=== Internal Test Evaluation ===
Accuracy:      0.7625
Macro F1:      0.6424
Precision:     0.6625
Recall:        0.6300
Per-class F1:  [0.7304, 0.3612, 0.8356]
```

结果保存在 `outputs/run_YYYYMMDD_HHMMSS/eval/internal_test_metrics.json`

---

### 生成测试集预测

```bash
python src/infer.py \
  --checkpoint ./outputs/run_20260118_162451/best_checkpoint.pth \
  --test_txt ./data/test_without_label.txt \
  --data_dir ./data/data \
  --output_path ./submission.txt \
  --batch_size 16
```

**输出格式**（`submission.txt`）：
```
guid,tag
0a0b1c2d,positive
1e2f3g4h,negative
5i6j7k8l,neutral
...
```

---

## 消融实验

### 自动运行三组消融实验（multimodal / text_only / image_only）

```bash
python src/ablations.py \
  --base_run_dir ./outputs/run_20260118_162451 \
  --do_eval
```

**说明**：
- 自动复用主实验的数据划分（`splits/`）和超参数（`hparams.json`）
- 仅改变 `--input_mode` 参数
- 结果保存在 `outputs/run_20260118_162451/ablations/{multimodal,text_only,image_only}/`

**手动运行单个消融实验**：

```bash
# Text-only
python src/train.py \
  --split_dir ./outputs/run_20260118_162451/splits \
  --run_dir ./outputs/run_20260118_162451/ablations/text_only \
  --input_mode text_only \
  --two_stage \
  --stage1_epochs 5 \
  --stage2_epochs 5 \
  --lr_head 3e-4 \
  --batch_size 16

# Image-only
python src/train.py \
  --split_dir ./outputs/run_20260118_162451/splits \
  --run_dir ./outputs/run_20260118_162451/ablations/image_only \
  --input_mode image_only \
  --two_stage \
  --stage1_epochs 5 \
  --stage2_epochs 5 \
  --lr_head 3e-4 \
  --batch_size 16
```

---

## 融合机制对比实验

### Concat（默认）
```bash
python src/train.py --fusion concat --two_stage ...
```

### Gated Fusion
```bash
python src/train.py --fusion gated --two_stage ...
```

### FiLM
```bash
python src/train.py --fusion film_concat --two_stage --lr_clip 3e-6 ...
```

### Cross-Attention（需独立学习率+类别加权）
```bash
python src/train.py \
  --fusion cross_attn \
  --two_stage \
  --lr_head 3e-4 \
  --lr_attn 2e-5 \
  --lr_clip 5e-6 \
  --class_weights \
  --stage1_epochs 8 \
  --stage2_epochs 4 \
  ...
```

---

## 关键参数说明

### 训练策略
- `--two_stage`：启用两阶段训练（强烈推荐）
- `--stage1_epochs`：Stage 1 训练轮数（冻结 CLIP）
- `--stage2_epochs`：Stage 2 训练轮数（轻量微调）
- `--freeze_clip`：完全冻结 CLIP（单阶段时使用）
- `--unfreeze_text`：Stage 2 解冻文本编码器
- `--unfreeze_text_layers`：解冻最后 N 层（默认 1）

### 融合方式
- `--fusion concat`：简单拼接（最稳定，推荐）
- `--fusion gated`：门控加权
- `--fusion film_concat`：条件归一化
- `--fusion cross_attn`：跨模态注意力（参数量大，需精细调优）

### 学习率
- `--lr_head`：分类头学习率（默认 3e-4）
- `--lr_clip`：CLIP 微调学习率（默认 5e-6，过大会过拟合）
- `--lr_attn`：Cross-Attention 模块独立学习率（默认 1e-4）

### 类别加权
- `--class_weights`：启用平方根平滑的类别加权损失

### 复现性
- `--seed`：随机种子（默认 42）
- `--split_dir`：复用已有的数据划分（确保公平对比）

---

## 实验管理与分析

### 汇总所有实验结果

使用 `analyze_runs.py` **自动扫描** `outputs/` 目录，生成 `run_summary.csv` 汇总表：

```bash
python src/analyze_runs.py --outputs ./outputs
```

**功能说明**：

1. **自动扫描**：遍历 `outputs/run_*` 目录，提取超参数和评估指标

2. **生成汇总表**：包含测试集 Macro F1、Accuracy、每类 F1（Negative/Neutral/Positive）、验证集最优 F1 等

3. **显示 Top-K 实验**：按测试集 Macro F1 排序，展示最优的 10 个实验配置

4. **保存 CSV**：自动保存到 `outputs/run_summary.csv`（后续运行会直接加载，除非使用 `--regenerate`）

###  分析功能

#### **按超参数分组统计**

```bash
python src/analyze_runs.py \
  --outputs ./outputs \
  --group-by fusion
```

**输出**：对比不同融合方式的平均 F1、标准差和最优值。

#### 超参数相关性分析

```bash
python src/analyze_runs.py --outputs ./outputs
```

**输出**：自动计算 `lr_clip`、`dropout`、`hidden_dim` 等超参数与测试 F1 的 Pearson 相关系数。

#### 生成 LaTeX 表格（用于报告）

```bash
python src/analyze_runs.py \
  --outputs ./outputs \
  --latex ./report/top_runs_table.tex \
  --top-k 10
```

**输出**：生成可直接插入 LaTeX 文档的表格代码。

#### 强制重新生成汇总表

```bash
python src/analyze_runs.py --outputs ./outputs --regenerate
```

**场景**：新增了实验，需要更新 `run_summary.csv`。

---

## 实验结果

### 主模型（Concat + 两阶段）

| 数据集     | Accuracy  | Macro F1  | Pos F1 | Neu F1 | Neg F1 |
| ---------- | --------- | --------- | ------ | ------ | ------ |
| 内部测试集 | **76.3%** | **64.2%** | 73.0%  | 36.1%  |        |

### 消融实验对比

| 输入模态 | 测试集 Accuracy | 测试集 Macro F1 | Neu F1 |
|----------|----------------|----------------|--------|
| **Multimodal** | **75.3%** | **624%** | **34.4%** |
| Text-only | 69.3% | 49.5% | 12.5% |
| Image-only | 71.8% | 50.2% | 4.2% |

**结论**：多模态融合相比单模态 Macro F1 提升 **14.7个百分点**，Neutral F1 提升 **2.9-8.6倍**。

### 融合机制对比

| 融合方式 | 测试 F1 | Neu F1 | 参数量 |
|----------|---------|--------|--------|
| **Concat** | **64.2%** | **36.1%** | +13K |
| Gated | 56.3% | 24.7% | +0.5K |
| FiLM | 60.1% | 28.9% | +3.1K |
| Cross-Attn | 61.2% | 34.6% | +1.2M |

**结论**：简单拼接在小样本场景下最稳定，复杂融合机制易过拟合。

---

## 代码实现亮点

### 1. 模块化设计
- **`model.py`**：统一接口支持 5 种融合方式，易扩展
- **`datasets.py`**：自动处理多编码格式（UTF-8/GBK/Latin-1）
- **`utils.py`**：完整的日志、指标计算、可视化工具

### 2. 实验管理
- 自动导出 `hparams.json`（所有超参数）
- 固化数据划分（`splits/`），确保消融实验公平对比
- 生成 `run_summary.csv`，方便批量实验分析

### 3. 鲁棒性
- 三级 fallback 读取策略解决文本编码不一致
- Early stopping 防止过拟合
- ReduceLROnPlateau 自动调整学习率

### 4. 可复现性
- `--seed` 固定所有随机性（Python/NumPy/PyTorch/CUDA）

- `--split_dir` 复用数据划分

- Checkpoint 包含完整训练状态（optimizer/scheduler）

---

## 常见问题

### Q1：如何修改模型架构？

**A**：编辑 `src/model.py` 中的 `LateFusionClassifier`，在 `__init__` 中添加新的融合层。

### Q2：如何添加新的数据增强？

**A**：修改 `src/transform.py` 中的 `build_image_transforms` 函数。

### Q3：如何使用自己的数据集？

**A**：准备 `train.txt`（格式：`guid,tag`）和对应的 `guid.jpg`/`guid.txt` 文件，修改 `--data_dir` 和 `--train_txt` 路径。

### Q4：训练中断如何恢复？

**A**：暂不支持断点续训，建议使用 `--early_stop_patience` 避免过长训练。

### Q5：为什么 Cross-Attention 需要独立学习率？

**A**：Attention 模块（1.2M 参数）与分类头（13K 参数）的梯度尺度差异大，共用学习率会导致 attention 训练不充分。



---

## 参考文献

### 学术论文

1. Radford, Alec, et al. "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*.  
   https://arxiv.org/abs/2103.00020

2. Poria, Soujanya, et al. "A Comprehensive Survey on Multimodal Sentiment Analysis." *IEEE TAC 2025*.  
   https://ieeexplore.ieee.org/document/10862066

3. Ye, Han-Jia, et al. "Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations." *ICML 2023*.  
   https://proceedings.mlr.press/v206/ye23a/ye23a.pdf

4. Zhou, Kaiyang, et al. "Learning to Prompt for Vision-Language Models." *CVPR 2022*.  
   https://arxiv.org/pdf/2109.01134

5. Cui, Yin, et al. "Class-Balanced Loss Based on Effective Number of Samples." *CVPR 2019*.

6. He, Kaiming, et al. "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022*.

### 开源代码

7. OpenAI CLIP 官方仓库：https://github.com/openai/CLIP
8. Hugging Face Transformers：https://huggingface.co/docs/transformers/model_doc/clip
9. scikit-learn：https://scikit-learn.org/stable/
10. PyTorch 官方文档：https://pytorch.org/docs/stable/

---



