# Project 5 多模态情感分类

<div style="text-align: right; font-size: 18px;">
10235501402  &nbsp;李则言   
</div>
## 概述

本实验利用文本与图像的配对数据，对样本的情感倾向进行三分类预测。基于提供的匿名数据集构建了一个视觉文本融合模型，并自行划分验证集，完成模型选择与超参数调优。实验中尝试了多种样本增强手段、训练方法和融合架构，克服样本类别不均衡问题，并做了消融实验以体现文本-图像互补必要性。

实验仓库：[ZyanNo1/Project5-of-Modern-AI: Project 5 of Modern AI, Autumn, 2025](https://github.com/ZyanNo1/Project5-of-Modern-AI)



## 架构设计

本实验以预训练模型 **CLIP** 作为双模态 backbone，分别使用其 text encoder 与 vision encoder 提取文本与图像的语义向量，采用 **late fusion（拼接 + MLP）** 作为融合策略。训练使用**两阶段训练策略**，结合先冻结CLIP训练分类头，再轻量微调文本编码器，以取得预训练知识和任务适配的平衡。

### 基础架构

选择OpenAI 的 clip-vit-base-patch32 作为多模态编码器的原因在于其在4亿图文对上进行的对比学习预训练已建立了强大的跨模态对齐能力。与从零训练的 ResNet 加 BERT 的双塔编码器相比，CLIP在小样本场景(本实验总共仅4000可用样本)下能显著减少过拟合风险。

多模态融合主要有两种范式：Early Fusion在输入层拼接原始特征，Late Fusion在特征层拼接编码后的embedding。Early Fusion需要引入大量可训练参数，但现有数据集量有限。相比之下，**Late Fusion仅需训练轻量分类头**，可充分利用CLIP预训练的特征提取能力,在小样本场景下更稳定。最终选择Late Fusion作为主架构，在后续探索中也验证了简单拼接在当前场景下优于更复杂融合机制 (FiLM、Gated、Cross-Attention) 的结论。

### 分类头设计

CLIP双模态编码器各输出512维特征,拼接后得到1024维向量。初期分类头设计为两层MLP [1024->512->128->3]，但在验证集上出现震荡(F1在0.65-0.70间波动)，经分析认为512维隐层过大拟合能力过强导致**过拟合**。秉持隐层维度应与任务复杂度相当的原则，将隐层维度减半为**[1024->256->64->3]**，该调整显著改善稳定性，验证集与测试集F1 gap从0.1445缩小至0.0562，测试F1从54.6%提升至62.4%，泛化能力明显提升。

小样本场景下正则化也至关重要。Dropout从初始0.1逐步提升至 **0.5**，Weight Decay从0.01增至 **0.05**，配合BatchNorm 与 ReLU激活函数，效果良好。实验发现Dropout<0.3时模型在20 epochs内显著过拟合，>0.5时收敛过慢且欠拟合，0.5为该任务的经验最优值。



## 训练策略

**阶段一：完全冻结 CLIP**

最初采用**单阶段训练**策略，冻结CLIP全部参数(freeze_clip=True)，仅训练分类头。该策略稳定性高、训练快，稳定收敛，但是上限提升困难，因为CLIP的预训练任务是图文匹配，和情感分类任务语义空间不完全对齐，完全冻结导致模型无法学习细粒度的语义偏移。

**阶段二：直接解冻**

为适配任务语义，尝试**直接解冻CLIP文本编码器全部12层**，用 `lr_clip = 1e-5` 联合训练。结果出现严重过拟合，验证集F1冲至70.3%，测试集却崩溃至55.1%，gap达15.2%。分析发现高学习率快速破坏了CLIP预训练权重，模型迅速退化为随机初始化网络在小数据上的记忆，遗忘现象超级明显。

**阶段三：两阶段渐进微调**

ICML 2023 *[Freeze then Train](https://proceedings.mlr.press/v206/ye23a/ye23a.pdf)* 和  CVPR 2022 *[CoOp](https://arxiv.org/pdf/2203.05557v1)* 都提及了先冻结再解冻微调的两阶段训练可以显著提升小数据集上的稳定性与泛化能力，因此借鉴他们的想法将训练拆分为**两阶段**：

- **Stage 1 (约5-12 epochs)**:冻结CLIP全部参数，用 `lr_head=3e-4` 训练分类头至收敛(验证集F1约65-68%)，建立稳定的任务特定决策边界。此阶段相当于用CLIP既有特征探路，找到情感分类的初步映射规则。

- **Stage 2(约5-10 epochs)**:从Stage 1最优checkpoint出发，**仅解冻CLIP文本编码器最后1层**，用极小学习率lr_clip=5e-6 (仅为分类头的1/60) 轻量微调。此时模型可以适配情感任务的语义特征，同时通过极低学习率避免破坏预训练权重。



## 数据处理

**文本预处理**采用轻量化清洗策略保留情感关键信息。首先移除URL链接(`http\S+`)和用户提及(`@username`),避免噪声干扰；将话题标签拆分为普通词汇(如`#HappyDay`->`HappyDay`)，保留其语义内容;统一转为小写并规范化空白字符。**实验中刻意保留了标点符号(如`!`和`?`)与emoji表情符号**，因为其携带强烈的情感信号清洗后的文本通过CLIP tokenizer编码(最大长度77)，确保与预训练阶段的输入格式一致。

**图像预处理**遵循CLIP标准流程，使用ImageNet归一化参数(`mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]`)。训练阶段采用轻量数据增强，短边缩放至256像素后随机裁剪至224×224，以0.5概率水平翻转,并施加小幅 ColorJitter。验证与测试阶段均使用确定性预处理，缩放至256像素后中心裁剪224×224，不进行随机变换。

**数据集划分**采用**分层抽样**保证三类情感在训练/验证/测试集中的比例严格一致,避免随机划分导致某类样本在验证集中过少影响早停策略。具体流程为:首先读取 train.txt获取全部4000个guid及其标签,利用`sklearn.model_selection.train_test_split` 按8:1:1比例分层抽样划分训练集3200样本、验证集400样本、内部测试集400样本。划分时固定随机种子 seed=42，并将划分结果固化为CSV文件(`train_split.csv/val_split.csv/test_split.csv`)存储于每个run的 `splits/` 目录下,确保消融实验与主模型使用完全一致的数据划分,满足公平对比的科学性要求。后续所有实验均复用同一份splits,通过`--split_dir`参数加载,避免因数据划分不同导致的性能差异。

数据集存在显著的**类别不均衡**：positive占41.5%(1660), negative占33.2%(1329), neutral仅占5.3%(211)，训练样本中neutral与positive的比例接近1:8。这种不均衡导致模型在初期训练中几乎不预测neutral(F1仅0.12)，全部误判为多数类。为此在损失函数中引入**类别加权**(`class_weights=True`)，根据inverse frequency计算权重 `weight = total_samples / class_count`，得到[1.93, 2.41, 15.17]。结果neutral虽被预测 (F1升至0.35)，但整体macro F1反降至57.3%，感觉因为极高权重(15.17)导致模型过度关注neutral，positive/negative准确率下降。后参考何凯明的Focal Loss思想，对inverse frequency做**平方根平滑**并归一化，避免neutral权重过大导致其他类崩溃。



## 消融实验

在固定数据集划分、超参数配置与训练策略的前提下，系统对比了**仅文本(text-only)**、**仅图像(image-only)**与**多模态融合(multimodal)**三种输入模态的性能表现。所有消融实验均复用主模型的 `splits/` 目录，确保训练、验证和测试集划分完全一致。

#### 实验设置

为纯粹对比输入模态的作用，三组消融实验均采用**单阶段训练策略**：完全冻结CLIP，仅训练分类头（融合维度->256->64->3），训练10 epochs，学习率 `lr_head=3e-4`，Dropout=0.5，Weight Decay=0.05，Batch Size=16。数据增强与预处理策略均采用轻量ColorJitter + 水平翻转，固定seed=42确保可复现性。唯一差异在于输入模态：multimodal将图像与文本特征拼接后输入分类头（输入维度1024），text-only仅使用文本特征并将图像输入置为None（输入维度512），image-only反之。

#### 性能对比

| 输入模态       | Accuracy  | Macro F1  | Pos F1    | Neu F1    | Neg F1    |
| -------------- | --------- | --------- | --------- | --------- | --------- |
| **Multimodal** | **0.753** | **0.624** | **0.697** | **0.344** | **0.831** |
| Text-only      | 0.693     | 0.495     | 0.571     | 0.125     | 0.790     |
| Image-only     | 0.715     | 0.499     | 0.645     | **0.042** | 0.812     |

| 模态       | Neutral->Pos | Neutral->Neu | Neutral->Neg | Neutral 召回率 |
| ---------- | ------------ | ------------ | ------------ | -------------- |
| Multimodal | 11 (26%)     | **11 (26%)** | 20 (48%)     | 26%            |
| Text-only  | 8 (19%)      | **3 (7%)**   | 31 (74%)     | 7%             |
| Image-only | 17 (40%)     | **1 (2%)**   | 24 (57%)     | 2%             |

从指标上来看，**多模态融合显著优于单模态**。相比text-only,multimodal的macro F1提升**13.0个百分点**，相比image-only提升**12.5个百分点**(0.499->0.624)，验证了文本与图像特征在情感分类任务中的互补性。具体到各类别，positive类F1提升最显著(text-only 0.571->multimodal 0.697,提升22%)，说明多模态融合能更准确捕捉积极情感的综合表达(如文本中的"happy"配合图像中的笑脸)；negative类在所有模态下表现均较好(F1>0.79)，因该类样本数最多(239个)且视觉特征明显(暗色调、皱眉表情)，单模态已有较强区分度;**neutral类是最大受益者**,multimodal F1达0.344，是text-only(0.125)的2.75倍，image-only(0.042)的8.2倍，充分体现了多模态融合对困难类别的提升作用。

Text-only在neutral类上F1仅0.125，混淆矩阵显示42个neutral样本中仅3个被正确识别,31个误判为negative,8个误判为positive。分析发现,文本中"stunned"、"confused"等中性词汇的语义边界模糊，CLIP文本编码器在预训练时更多关注明确的情感极性词汇(如"love"对应positive,"hate"对应negative)，对微妙的中性情感区分度不足。此外,文本清洗后标点符号与emoji被保留，但单纯依赖文本特征时，模型难以从符号推断出困惑(neutral)与惊喜(positive)的差异，导致误判率高。

Image-only在neutral类上表现更差,F1仅0.042,42个neutral样本中仅1个被正确识别,17个误判为positive,24个误判为negative。**这说明neutral情感几乎无法通过纯视觉线索识别**。neutral标签对应的图像往往缺乏明显的情感特征,如墨镜自拍、风景照等,这些图像的色调、构图与positive/negative样本存在大量重叠，视觉编码器无法仅凭图像区分"震惊"(neutral)与"兴奋"(positive)。相比之下,positive样本常包含明亮色调、笑脸表情，negative样本多为暗色调、皱眉，视觉特征明确，因此image-only在这两类上F1仍达0.645与0.812。

**多模态融合的互补**能明显体现在neutral类上。对于neutral样本，文本提供语义线索(如"stunned"、"???")，图像提供情境信息(如自拍墨镜),两者结合后模型能推断出"困惑/震惊"的中性情感；对于positive/negative样本，文本的情感极性词汇(如"love"、"hate")与图像的色调或表情相互验证，降低误判率。混淆矩阵显示，multimodal对neutral的召回率(11/42=26%)虽仍不高，但相比单模态已有显著提升(text-only 7%,image-only 2%)，且误判分布更均衡(11个误判为positive，20个误判为negative)，说明模型在利用多模态信息时能更谨慎地预测neutral，而非一味偏向多数类。



## 模型调优

### 从单阶段到两阶段

单阶段训练测试集上达到62.4% macro F1，虽然训练稳定，但完全冻结限制了模型对任务特定特征的捕捉能力。为此引入**两阶段训练策略**，Stage 1冻结CLIP训练分类头5 epochs，Stage 2解冻文本编码器最后1层用极小学习率(`lr_clip=5e-6`)微调5 epochs。该策略在测试集上达到**64.2% macro F1**，相比单阶段**提升1.8个百分点**，验证集F1也从68.0%升至70.9%。两阶段的成功在于平衡了预训练知识保留与任务适配:Stage 1建立稳定基线，Stage 2通过轻量微调使CLIP逐步适配情感语义，同时避免忘记预训练内容。

在 Stage 2的训练中，**学习率至关重要**。对比实验显示，`lr_clip=1e-5` 时模型严重过拟合(验证F1达70.3%但测试F1崩至55.1%，gap高达15.2%)，因高学习率快速破坏预训练权重导致遗忘；`lr_clip=2e-6`则微调不足，测试F1仅61.6%；最终选定**5e-6**作为平衡点，既能适配任务又不破坏预训练知识。

此外，选择**文本层解冻**是因为尝试解冻 vision后发现测试 F1 降至0.589，低于冻结全CLIP的 0.624。分析认为情感标签的区分度更依赖文本语义，而 CLIP 的视觉特征(表情、色调等)预训练已足够，解冻视觉层过度微调反而引入噪声。

![image-20260123203450358](./assets/image-20260123203450358.png)

### 融合机制对比

Late Fusion的核心在于如何组合CLIP提取的512维图像特征与512维文本特征。最简单的 Concat 直接拼接两模态向量送入分类头，虽然稳定但缺乏显式的跨模态交互，可能丢失图文之间的细粒度关联。为探索更强的融合方式，按 **注意力机制复杂度由轻到重**的顺序，依次尝试了Gated Fusion、FiLM 和 Cross-Attention 三种方法，每种方式进行调优，尝试找到最优解。

**Concat** 直接将图像特征与文本特征拼接为1024维向量，送入MLP分类头。该方法无显式交互，完全依赖分类头学习两模态的线性组合模式。采用之前最优两阶段配置(`lr_head=3e-4`, `lr_clip=5e-6`, `stage1=5, stage2=5`)，测试F1达64.2%，验证集gap仅6.7%，作为后续方法的对照基线。

**Gated fusion** 引入可学习的门控向量，让模型自适应地调整图像与文本的贡献权重，期望在不同样本上动态侧重更具判别力的模态。其计算门控权重 $g = \sigma(W[f_{img}; f_{txt}] + b)$，其中 $\sigma$ 为sigmoid函数，$W \in \mathbb{R}^{1 \times 1024}$。融合特征为 $f = g \cdot f_{img} + (1-g) \cdot f_{txt}$，即通过标量门控动态加权两模态。经过调优，测试F1仅 56.3%，反而低于基线7.9个百分点。训练曲线显示门控参数在验证集上快速过拟合(验证F1达0.68)但测试集泛化弱，说明门控机制引入的约0.5K参数的额外自由度在小数据上容易记忆验证集模式，未能带来性能增益。

**FiLM** (Feature-wise Linear Modulation)通过文本特征对图像特征做逐维度的仿射变换，引入更丰富的条件调制。其用文本特征生成缩放因子 $\gamma = W_\gamma f_{txt}$ 和偏移因子 $\beta = W_\beta f_{txt}$，对图像特征做调制 $f'_{img} = \gamma \odot f_{img} + \beta$。调制后的图像特征与原文本特征拼接送入分类头，共引入约3.1K参数。FiLM 对学习率比较敏感，于是进行了网格搜索。最优配置为`lr_clip=3e-6`，此时测试F1达60.1%，F1 gap 减小到7.6%，但仍高于baseline。虽然 FiLM 在验证集上表现突出(F1约0.69)，但该优势在测试集上未能保持，说明条件归一化在小数据上易过拟合特定的图文关联模式。

| FiLM配置(lr_clip) | 验证F1 | 测试F1 | Gap   | 说明                     |
| ----------------- | ------ | ------ | ----- | ------------------------ |
| 2e-6              | 0.683  | 0.572  | 0.111 | 学习率过低，微调不足     |
| **3e-6**          | **0.676** | **0.601** | **0.076** | **调优后最优**           |
| 4e-6              | 0.665  | 0.548  | 0.117 | 开始过拟合               |
| 5e-6              | 0.689  | 0.599  | 0.090 | 验证集虚高，泛化能力下降 |

**Cross-Attention机制**实现了最深层的跨模态交互(约1.2M参数)，理论上是能力最强的融合方法，针对其训练难度进行了最复杂的调优，包括attention模块独立学习率搜索(`lr_attn` ∈ {1e-5, 2e-5, 4e-5, 5e-5})和类别加权策略探索。初期采用两阶段训练(完全冻结CLIP)测试F1仅57.3%，后为attention模块设置独立的`lr_attn`，与分类头解耦。参数搜索后 F1 提升至58.9%( `lr_attn=5e-5`)，但仍低于Concat。Cross-Attention 对类别不均衡极度敏感，很容易被多数类主导，尝试引入类别加权 (class_weight)。无加权时模型几乎完全忽略 neutral(F1仅0.19)，引入平方根平滑的类别加权后测试F1跃升至 **61.2%**( `lr_attn=2e-5, class_weights=True`)，neutral F1从0.19提升至0.35(**+80%**)，但整体性能仍略低于concat的64.2%。

| 配置                 | lr_attn  | class_wt | 验证F1    | 测试F1    | Gap       | **Neu F1** |
| -------------------- | -------- | -------- | --------- | --------- | --------- | ---------- |
| 冻结CLIP(无独立lr)   | -        | F        | 0.693     | 0.573     | 0.120     | **0.192**  |
| +独立lr(5e-5)        | 5e-5     | F        | 0.690     | 0.589     | 0.101     | **0.250**  |
| +独立lr(2e-5,调低)   | 2e-5     | F        | 0.678     | 0.562     | 0.116     | **0.179**  |
| **+独立lr+类别加权** | **2e-5** | T        | **0.682** | **0.612** | **0.070** | **0.346**  |

**融合机制总结**

| 方法               | 参数量 | 验证F1    | 测试F1    | **Neu F1** | Gap       |
| ------------------ | ------ | --------- | --------- | ---------- | --------- |
| **Concat**         | 0      | **0.709** | **0.642** | **0.361**  | **0.067** |
| Gated              | +0.5K  | 0.682     | 0.563     | 0.247      | 0.119     |
| FiLM(lr_clip=3e-6) | +3.1K  | 0.676     | 0.601     | 0.289      | 0.076     |
| Cross-Attn(无权重) | +1.2M  | 0.693     | 0.573     | **0.192**  | 0.120     |
| Cross-Attn(加权)   | +1.2M  | 0.682     | 0.612     | 0.346      | 0.070     |

在3200训练样本规模下，最简单的Concat反而最优，而理论上最强的Cross-Attention即使经过复杂调优仍落后3个百分点。复杂融合引入的额外参数很容易成为过拟合的温床，Gated/FiLM/Cross-Attention在验证集上均有亮眼表现(F1 0.68-0.70)，但测试集掉点明显(gap 7-12%)，说明它们学到的是验证集特有的模式而非通用规律。所以在小样本场景下，模型容量与类别均衡学习的权衡至关重要。

最终根据综合性能和稳定性，采用 **Concat两阶段** 作为最终提交模型。最终预测来自基于最优参数，使用全量数据训练的模型。



## 遇到的困难

1. 文本编码不一致：数据集中的 `.txt` 文件编码不统一，部分使用 UTF-8，部分使用 GBK，直接读取时频繁抛出 `UnicodeDecodeError`。最终在读入时采用三级fallback。

2. 贯穿整个实验的过拟合：通过冻结预训练参数、强化正则化、降低学习率等等方法改善，在前文中已经详细提及。全部的训练都设有早停。

3. 类别不均衡，Neutral 类几乎不被预测：Concat 使用两阶段训练与数据增强自然缓解不均衡，其他融合策略使用Inverse Frequency 加权 以及 平方根平滑 + 归一化。

4.  实验管理，配置多样导致混乱：每次run是一个文件夹，训练前自动导出配置到 json，评估时自动读取配置，并导出指标保存回源目录。同时固化数据划分，将 train/val/test split 保存为 CSV 文件至 `splits/` 目录。

**参考文献**

1. Radford, Alec, et al. "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. (CLIP) [Link](https://github.com/openai/CLIP)
2. Poria, Soujanya, et al. "A Comprehensive Survey on Multimodal Sentiment Analysis." *IEEE Transactions on Affective Computing*, 2025. [Link](https://ieeexplore.ieee.org/document/10862066) 
3. Ye, Han-Jia, et al. "Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations." *ICML 2023*. [Link](https://proceedings.mlr.press/v206/ye23a/ye23a.pdf) 
4. Zhou, Kaiyang, et al. "Learning to Prompt for Vision-Language Models." *CVPR 2022*. [Link](https://arxiv.org/pdf/2109.01134)
5. Cui, Yin, et al. "Class-Balanced Loss Based on Effective Number of Samples." *CVPR 2019*. [Link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) 
6. He, Kaiming, et al. "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022*. [Link](https://arxiv.org/abs/2111.06377) 
7. OpenAI CLIP 官方仓库  [Link](https://github.com/openai/CLIP)
8. PyTorch Image Models (timm) - 数据增强参考 [Link](https://github.com/huggingface/pytorch-image-models) 