# 004 · Joint Pruning with a Learned Utility Scorer

> 状态：待实现、待实验。
> 本文预先记录实验目的、方法、评价口径与结果判定标准；不包含尚未运行的结果。

## 1. Objective

实验 003 将解决“视觉与文本分数是否可比”，但图文语义相关性不一定等于 token 对最终答案的作用。一个 chunk 可能与问题语义相关却没有提供答案证据，也可能与另一个 chunk 高度冗余。

本实验训练一个轻量级 utility scorer，直接预测删除文本或视觉 chunk 对答案概率造成的边际影响。scorer 为两种模态输出同一标度的 utility scalar，而不要求 Qwen 原始视觉和文本 embedding 本身有可比余弦距离。

本轮回答：

1. 用答案似然度变化构造的 utility label 是否稳定、可学；
2. 轻量 scorer 能否在未见样本上预测 chunk 的相对价值；
3. learned utility 是否比 raw Qwen cosine、外部 aligned relevance 和 random selection 更好地保留答案质量；
4. 标签构造的一次性成本与 scorer 的在线开销是否可接受。

本实验的研究假设是：

$$
\text{Pruning Utility}\neq\text{Semantic Similarity},
$$

而用完整模型的反事实删除效应作为监督信号，能更直接对齐剪枝目标。

## 2. Setup

### 2.1 数据划分与泄漏控制

必须建立样本 ID 不相交的 train/validation/test 划分：

| Split | 用途 | 允许用于什么 |
| --- | --- | --- |
| Train | 生成 utility label，训练 scorer | 参数更新 |
| Validation | early stopping、有限超参数选择和阈值检查 | 模型选择 |
| Test | 最终 pruning 和答案评测 | 仅一次最终评价 |

实验 002/003 的固定 50 样本作为 test set，不得用于：

- 构造 scorer 训练标签；
- 拟合归一化参数；
- 挑选 checkpoint 或 early stopping；
- 根据最终 Accuracy 反复修改特征和 loss。

如果 MMCoQA 中存在同一对话、文档或图像派生的多个问题，必须按源文档/对话/图像 group 划分，不能只按问题行随机拆分，避免近重复内容跨 split 泄漏。

### 2.2 冻结教师模型与 scorer

| 组件 | 设定 |
| --- | --- |
| Teacher / answer model | 冻结的 Qwen2.5-VL-7B-Instruct |
| 基础特征 | 冻结的 Qwen 输入特征，可选实验 003 aligned relevance |
| Utility scorer | 共享的轻量 MLP/ranking head |
| 训练参数 | 仅 scorer 和必要的小型 feature projection |
| 输出 | 每个 chunk 一个实数 utility score |

主实验不微调 Qwen，避免将“选择器是否更好”与“生成模型是否经过额外训练”混为一个变量。

### 2.3 实验条件

主预算为 75% 和 50%，25% 作为压力测试。所有剪枝方法使用同一严格联合总预算。

| 方法 | 目的 |
| --- | --- |
| No Pruning | 任务性能基线 |
| Random Joint, 3 seeds | 随机选择基线 |
| Fixed Joint | 固定模态预算基线 |
| Raw Qwen Joint | 实验 002 方法 |
| Aligned Encoder Joint | 实验 003 主方法 |
| **Learned Utility Joint** | 本实验主方法 |
| Oracle Utility | 仅小样本，估计 scorer 上限 |

scorer 训练包含随机初始化、样本次序和可能的负样本采样，因此主结果使用至少 3 个训练 seed，报告均值、标准差和逐 seed 结果。每个 seed 必须使用相同数据划分与评测样本。

### 2.4 评价与记录

分为三层评价：

1. **Label 质量**：utility 分布、正负比例、不同模态分布、重复前向的数值稳定性；
2. **Scorer 质量**：Spearman/Kendall 排名相关、pairwise ranking accuracy、top-k utility recall；
3. **下游质量**：Accuracy、robust Accuracy、配对变化、严格预算、端到端速度和显存。

记录必须包含 split manifest、数据版本、scorer checkpoint、seed、训练曲线、最佳 validation checkpoint、特征配置、label 生成命令和逐样本预测。

## 3. Method

### 3.1 以 Chunk 为学习单元

沿用实验 002 的文本句子 chunk 和视觉空间窗口：

$$
\mathcal G=\mathcal G_t\cup\mathcal G_v.
$$

首轮不为每个 token 单独构造反事实标签，原因是单 token 删除噪声大且标签成本与 token 数线性增长。scorer 先预测 chunk utility，然后将预算分给高 utility chunk；chunk 内部选择暂时沿用实验 003 的对齐相关性或使用空间均匀保留。

### 3.2 Leave-one-chunk-out Utility Label

对包含标准答案 $y=(y_1,\ldots,y_M)$ 的训练样本，先在 teacher forcing 下计算完整输入 $X$ 的平均答案 NLL：

$$
\ell(X,y)=-\frac1M\sum_{k=1}^M\log p_\theta(y_k\mid X,y_{<k}).
$$

对每个 chunk $g$ 构造删除后输入 $X\setminus g$，定义：

$$
u_g=\ell(X\setminus g,y)-\ell(X,y).
$$

- $u_g>0$：删除该 chunk 使答案 NLL 变差，应优先保留；
- $u_g\approx0$：该 chunk 对当前答案影响小；
- $u_g<0$：删除后 NLL 反而降低，可能是干扰或数值噪声。

标签生成只计算 gold answer token 的 loss，prompt、padding 和图像占位 token 均必须 mask 掉。文本和视觉删除仍使用项目已验证的序列重建与原始 mRoPE 位置。

### 3.3 冗余与交互的扩展标签

Leave-one-out 只估计“其他内容都存在”时的边际影响，无法充分表示高压缩率下的互补和冗余。如果第一阶段可学但 50%/25% 表现差，再在每个样本上采样少量子集 $S$，计算：

$$
u(g\mid S)=\ell(S\setminus g,y)-\ell(S,y).
$$

这是 Monte Carlo marginal contribution/Shapley 近似，不作为首轮必需步骤，避免在验证基础 label 前引入过高成本。

### 3.4 Scorer 输入与结构

对每个 chunk $g$ 构造：

$$
z_g=[p_g,q,r_g,d_g,m_g,c_g].
$$

其中：

- $p_g$：chunk 内 Qwen 特征的 attention/mean pooling；
- $q$：问题 pooled representation；
- $r_g$：可选的实验 003 aligned relevance；
- $d_g$：chunk 与同模态其他 chunk 的最大/平均冗余度；
- $m_g$：可学习的 modality embedding；
- $c_g$：长度、文本顺序或图像 ID 与空间坐标。

主结构使用一个共享轻量 MLP：

$$
\hat u_g=f_\phi(z_g)\in\mathbb R.
$$

可以使用分模态的小型输入 projection，但最后的 utility head 必须共享，以便在同一标尺上排序。不以“语义相同的视觉和文本 chunk 必须得分相同”作为硬约束，因为冗余度不同时，它们对答案的边际效用可以不同。

### 3.5 训练目标

主 loss 使用样本内 pairwise ranking，对 $u_i>u_j$ 的 chunk 对：

$$
\mathcal L_{rank}
=\log\left(1+\exp[-(\hat u_i-\hat u_j)]\right).
$$

辅以对稳健归一化 label 的 Huber regression：

$$
\mathcal L=
\mathcal L_{rank}+\lambda\mathcal L_{Huber}.
$$

ranking 是主目标，因为 pruning 更关心同一样本内的保留顺序，而 raw NLL delta 的绝对大小会受答案长度、基础难度和教师置信度影响。配对采样应在文本—文本、视觉—视觉和视觉—文本三类中保持合理比例，否则共享标尺无法得到充分训练。

### 3.6 从 Utility 到剪枝计划

首先按 $\hat u_g$ 计算 chunk 价值，然后在总 token 预算 $B$ 下选择 chunk/token。主算法应考虑 chunk 价值与长度，避免一个长 chunk 仅因总 utility 较高就占用过多预算。可使用：

$$
\operatorname{density}(g)=
\frac{\max(\hat u_g,0)}{|g|^\gamma},
$$

其中 $\gamma$ 只能在 validation set 上选择。若选中 chunk 后仍有剩余预算，按实验 002 的整数再分配逻辑精确填满；若 chunk 超出剩余容量，在 chunk 内按 aligned relevance 或空间/语序均匀规则选择。

不先用独立模态 softmax 分配 $B_t,B_v$；主实验让两种模态的 chunk 在统一 utility density 上直接竞争。为防止早期 scorer 崩溃后删空某个模态，可保留实验 002 的最低模态预算作 safety constraint，并设置“启用/不启用”消融。

### 3.7 标签生成与训练加速

utility label 生成是本实验的主要成本。对含 $G$ 个 chunk 的样本，基础 leave-one-out 需要 $G+1$ 次 teacher-forced scoring，但不需要自回归生成完整答案。

优先采用：

- 每个样本的 Qwen 视觉 encoder/projector 只运行一次，缓存视觉 embedding 和原始位置；
- 将多个删除变体按压缩后长度 bucketing 并成 batch；
- 使用 teacher forcing 而不是 `generate()`，避免串行 decode；
- 使用 BF16、inference mode 和已验证的 FlashAttention 2；
- 多卡按样本分片生成 label，每个 shard 原子写入并支持断点续跑；
- 特征提取完成后训练 scorer 时不再加载 7B teacher。

多卡只用于样本并行，不必为小型 scorer 引入 DDP。正式生成全量 label 前，必须通过少量样本实测单 chunk 前向耗时、并发 batch 上限、预计 GPU-hours 和中间特征存储量。

## 4. Results & Analysis & Conclusion

### 4.1 标签正确性门槛

训练前先验证：

1. 完整输入的 teacher-forced token NLL 可重复，无 label mask 偏移；
2. 删除后输入长度、保留索引与 mRoPE 位置通过现有 invariant；
3. 手工构造的唯一答案 chunk 被删除时，utility 明显高于无关 chunk；
4. 同一删除变体重复前向的 NLL delta 在可接受数值误差内一致；
5. train/validation/test 的 group ID 无交集。

任意一项失败都不应启动全量 label 生成。

### 4.2 Scorer 本身的成功标准

必须同时报告总体、文本、视觉和跨模态 pair 的排名指标。若总体指标较高但跨模态 pair 接近随机，则不能认为“统一 utility 标尺”已学到。

关键检查包括：

- validation ranking 指标明显高于随机；
- 三个 seed 的方向一致，不依赖单一最佳 checkpoint；
- 高分 chunk 实际删除后造成的 NLL 恶化高于低分 chunk；
- modality-only、position-only 和 length-only 弱基线无法解释主要收益。

### 4.3 下游主结果判定

实验的核心证据不是 scorer 的 regression loss，而是在严格预算下的未见 test 结果。

继续该路线的正向证据为：

1. Learned Utility 在 75% 和 50% 上均高于 Random Joint 三种子均值；
2. 相对实验 003 呈现一致的配对改善，说明学到的不只是语义相关性；
3. 多个训练 seed 的差异小于方法相对主对照的收益；
4. scorer 在线开销计入后，在至少一个不明显降低 Accuracy 的预算上保持端到端加速。

在 50 样本上仍应报告置信区间、配对胜负和精确配对检验。若只有 oracle 好而 learned scorer 差，说明 label 存在信号但特征或模型不足；若 oracle 也不优于 random，应优先怀疑单次 leave-one-out 标签与目标预算不匹配。

### 4.4 风险与 fallback

| 风险 | 影响 | Fallback |
| --- | --- | --- |
| leave-one-out 无法捕捉交互 | 高压缩率下排名失效 | 采样子集边际贡献，做 Monte Carlo Shapley 近似 |
| gold-answer NLL 与自由生成 Accuracy 不一致 | scorer 优化代理目标 | 加入 sequence-level preference/是否答对信号，但保留 NLL 主口径 |
| utility label 噪声大 | 训练不稳定 | 用 pairwise loss、Huber loss、忽略近零差值 pair |
| scorer 学到位置/模态捷径 | 分布外泛化差 | 报告弱基线、做 feature ablation，增加跨模态 pair |
| label 生成过慢 | GPU-hours 过高 | 先采样部分 chunk，batch teacher forcing，优先高变异样本 |
| MMCoQA 训练数据过小 | scorer 过拟合 | 降低模型容量，增加其他长上下文 benchmark 训练数据，保持 benchmark-held-out 评价 |

## 5. Next

1. 先固化 group-aware train/validation/test manifest，对近重复文档、对话和图像做泄漏检查；
2. 在 3–5 个训练样本上生成 leave-one-out label，用手工案例检验符号、mask 和 mRoPE；
3. 实测标签生成的单样本时间、batch 峰值显存、缓存大小和 GPU-hours，再决定正式标签规模；
4. 先训练最小共享 MLP，证明 validation ranking 可学，再增加 redundancy、aligned relevance 和子集边际标签；
5. 使用 3 个 seed 训练并锁定 checkpoint，最后一次性运行固定 50 样本 test；
6. 若 scorer 显著优于 aligned relevance，再扩展至 MMLongBench-Doc 等文本/视觉 token 比例更多样的数据。

