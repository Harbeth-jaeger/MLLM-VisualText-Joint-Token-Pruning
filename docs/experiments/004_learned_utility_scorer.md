# 004 · Joint Pruning with a Learned Utility Scorer

> 状态：正式实验已完成。120/30 个 train/validation 样本用于标签生成与训练，固定 50 样本 test 已完成全部主对照评测。
> 结论：首版 scorer 学到了一定跨模态排序信号，但总体排序能力弱且出现过拟合；下游结果未达到预先定义的继续路线标准。

## 1. Objective

实验 003 原计划解决“视觉与文本分数是否可比”，但图文语义相关性不一定等于 token 对最终答案的作用。当前执行顺序调整为先完成实验 004，实验 003 后续作为独立对照、chunk 内 token 选择辅助方法及可选 scorer 特征。一个 chunk 可能与问题语义相关却没有提供答案证据，也可能与另一个 chunk 高度冗余。

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

### 4.1 运行完整性与数据划分

正式流程于 2026-10-17 20:59:06 CST 启动，23:36:58 CST 正常结束，总墙钟时间为 2 小时 37 分 52 秒。运行使用单张 GPU、BF16 和 FlashAttention 2。最终产物包括：

- 120 个 train 标签文件和 30 个 validation 标签文件；
- 3 个独立训练 seed 的最佳 checkpoint 与完整训练历史；
- 固定 50 样本 test 上 25 个条件、共 1250 条逐样本结果；
- 1250 条记录的 `(sample_id, method, retention, seed)` 均唯一，且全部记录 `flash_attention_2`；
- 所有剪枝条件的 `budget_deviation=0`，实际候选保留率分别为 0.2498、0.4999 和 0.7498。

划分 seed 为 2026。MMCoQA 这 200 个样本在当前按共享图片连通的分组规则下形成 200 个独立 group，最终严格划分为 120/30/50；固定 test manifest 与实验 002/003 一致，没有 test-connected 样本进入训练或验证。这里的泄漏检查只覆盖当前可获得的图片路径关联，不能排除数据集中未显式标注的语义近重复。

本轮实际完成的下游方法为 No Pruning、Fixed Joint、Random Joint（3 seeds）、Raw Qwen Joint 和 Learned Utility Joint（3 个训练 seeds）。预注册中的 Aligned Encoder Joint 依赖尚未运行的实验 003，Oracle Utility 也未实现，因此二者不应被视为本轮已完成对照。

### 4.2 Utility label 分布与成本

| Split | 样本 | Chunk 总数 | 平均/样本 | Text / Vision chunk | Utility 均值 ± 标准差 | 正 / 近零 / 负比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 120 | 17,817 | 148.5 | 1,009 / 16,808 | 0.0007 ± 0.1435 | 40.6% / 11.7% / 47.6% |
| Validation | 30 | 5,023 | 167.4 | 256 / 4,767 | 0.0198 ± 0.1469 | 46.7% / 7.7% / 45.5% |

“近零”定义为 $|u_g|\le 0.001$，与训练时忽略近零 pair 的阈值一致。Train utility 的中位数为 -0.0004，1%/99% 分位数为 -0.2312/0.3123，完整范围为 -5.2754 到 8.9714。标签并未退化为全零或单一符号，说明 leave-one-out NLL 至少产生了可用于排序的非平凡监督信号。

模态分布极不均衡：train 中视觉 chunk 占 94.3%。文本 utility 的标准差为 0.5280，约为视觉 utility 标准差 0.0707 的 7.5 倍；文本中位数为 0.0063，视觉中位数为 -0.0005。这意味着共享标尺必须同时处理数量不平衡和明显不同的长尾尺度，也增加了 scorer 依赖模态捷径的风险。

标签生成的实际开销如下：

| Split | 总 GPU 时间 | 平均/样本 | 中位数/样本 | P95/样本 | 峰值显存 | 磁盘 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 84.98 min | 42.49 s | 34.01 s | 103.02 s | 20.68 GB | 125 MB |
| Validation | 36.67 min | 73.34 s | 47.28 s | 200.95 s | 21.14 GB | 35 MB |

两部分合计约 121.65 GPU-min 和 160 MB 标签缓存。样本耗时随 chunk 数显著增长；validation 平均更慢主要来自其平均 chunk 数更高及少数长样本，不能解释为 split 本身的系统差异。

### 4.3 Scorer 训练结果

checkpoint 只由 validation macro pairwise accuracy 选择。三个 seed 的最佳结果为：

| Seed | 最佳 epoch | 训练轮数 | 最佳 val macro pairwise acc. | 跨模态 pairwise acc. | Val MAE | 首轮 → 末轮 train loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 14 | 52.85% | 64.39% | 0.0603 | 0.9803 → 0.6135 |
| 1 | 1 | 14 | 53.06% | 64.40% | 0.0604 | 0.9792 → 0.6258 |
| 2 | 4 | 17 | 53.29% | 64.69% | 0.0604 | 0.9794 → 0.5848 |

训练 loss 稳定下降约 36%–40%，所以 smoke test 中“loss 能否下降”的问题在正式数据上得到肯定答案。但 validation 总体排序仅略高于 50% 随机水平；最佳 epoch 很早，之后 train loss 继续下降而 validation 排名和 MAE 多数变差，表现出明显过拟合。跨模态准确率稳定在约 64.5%，说明 scorer 确实学到部分图文间标尺，但这没有转化为足够强的样本内细粒度排序。

当前实现没有输出预注册的 Spearman、Kendall、top-k utility recall，也没有运行 modality-only、position-only 和 length-only 弱基线。因此，本轮不能确认 64.5% 的跨模态结果来自语义 utility，而不是模态、长度或位置等捷径；这是报告结论的明确边界。

### 4.4 Test 主结果

以下均为固定 50 个未参与训练与模型选择的 test 样本。`±` 为 3 个 seed 的总体标准差；单 seed 方法不报告标准差。

| 方法 | 保留率 | Accuracy | Robust Accuracy | 平均输入 token | 平均 Text / Vision budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Pruning | 100% | 70.0% | 70.0% | 2645.5 | 458.9 / 2103.6 |
| Fixed Joint | 75% | 64.0% | 66.0% | 2004.6 | 417.2 / 1504.4 |
| Random Joint | 75% | 64.0% ± 3.3% | 66.0% ± 3.3% | 2004.6 | seed-dependent |
| Raw Qwen Joint | 75% | **66.0%** | 68.0% | 2004.6 | 454.0 / 1467.5 |
| **Learned Utility Joint** | **75%** | 64.7% ± 2.5% | 66.7% ± 2.5% | 2004.6 | 441.0 / 1480.6 |
| Fixed Joint | 50% | 62.0% | 62.0% | 1364.0 | 362.4 / 918.6 |
| Random Joint | 50% | **62.7% ± 4.1%** | **64.0% ± 4.3%** | 1364.0 | seed-dependent |
| Raw Qwen Joint | 50% | **66.0%** | **66.0%** | 1364.0 | 434.4 / 846.6 |
| **Learned Utility Joint** | **50%** | 60.0% ± 1.6% | 61.3% ± 2.5% | 1364.0 | 420.1 / 860.9 |
| Fixed Joint | 25% | 42.0% | 42.0% | 723.3 | 264.9 / 375.3 |
| Random Joint | 25% | 47.3% ± 0.9% | 48.7% ± 0.9% | 723.3 | seed-dependent |
| Raw Qwen Joint | 25% | 46.0% | 46.0% | 723.3 | 349.0 / 291.2 |
| **Learned Utility Joint** | **25%** | 47.3% ± 2.5% | 49.3% ± 2.5% | 723.3 | 333.9 / 306.4 |

Learned Utility 的逐 seed Accuracy/Robust Accuracy 为：75% 下 64/66、68/70、62/64；50% 下 62/64、60/62、58/58；25% 下 48/50、50/52、44/46。75% 的最佳 seed 接近无剪枝基线，但其他 seed 没有复现；50% 三个 seed 都不超过 Raw Qwen；25% 的均值与 Random 完全相同。

No Pruning 的 Accuracy 为 35/50，Wilson 95% CI 为 [56.2%, 80.9%]。Learned Utility 各 seed 的区间较宽：75% 为 [50.1%, 75.9%]、[54.2%, 79.2%]、[48.2%, 74.1%]；50% 为 [48.2%, 74.1%]、[46.2%, 72.4%]、[44.2%, 70.6%]；25% 为 [34.8%, 61.5%]、[36.6%, 63.4%]、[31.2%, 57.7%]。50 样本只能识别较大的效果差异，当前 1–3 个样本量级的差别不能作强结论。

相对 No Pruning，Learned Utility 的原始指标“由对变错 / 由错变对”分别为：75% 的 4/1、2/1、5/1；50% 的 5/1、7/2、9/3；25% 的 12/1、11/1、14/1。双侧精确 McNemar 检验在 75% 和 50% 均不显著（所有 $p\ge0.146$），25% 三个 seed 均显著差于无剪枝（$p=0.0034,0.0063,0.0010$）。Learned 与相同 seed Random、以及 Learned 与 Raw Qwen 的逐 seed 配对比较全部 $p\ge0.289$；没有证据表明观察到的小差异超过这 50 个样本上的随机波动。

### 4.5 速度、显存与计时边界

| 条件 | Planner ms | Prefill ms | 相对 No Pruning 的 prefill 降幅 | Decode ms | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Pruning | 32.2 | 363.2 | — | 623.0 | 16.43 GB |
| Learned 75%（3-seed mean） | 7.6 | 313.9 | 13.6% | 445.8 | 16.23 GB |
| Learned 50%（3-seed mean） | 10.1 | 205.1 | 43.5% | 472.9 | 16.04 GB |
| Learned 25%（3-seed mean） | 7.4 | 124.1 | 65.8% | 491.9 | 15.84 GB |

严格减少输入 token 确实带来了清晰的 prefill 缩短，显存也小幅下降。但当前 `planning_ms` 只覆盖预算规划器，不包括 Qwen relevance、utility 特征构造、scorer 前向、CPU 预处理和 I/O；`vision_encode_ms` 单独记录，且 decode 时间还受各条件输出长度影响。因此这些数字不能被表述为完整在线端到端加速，预注册成功标准中的“计入 scorer 在线开销后仍加速”尚未被当前 instrumentation 证明。

### 4.6 对预注册成功标准的逐项判定

| 成功标准 | 结果 | 判定 |
| --- | --- | --- |
| 75% 和 50% 均高于 Random 三种子均值 | 75%：64.7% vs 64.0%；50%：60.0% vs 62.7% | **失败** |
| 相对实验 003 有一致配对改善 | 实验 003 尚未运行，Aligned Encoder 对照缺失 | **未评估** |
| seed 波动小于方法收益 | 75%/25% seed 标准差 2.5%，而相对 Random 差值仅 +0.7%/0%；50% 为 -2.7% | **失败** |
| 计入 scorer 在线开销后保持端到端加速 | prefill 明显下降，但完整 scorer/特征开销未计时 | **未证明** |

### 4.7 结论

首版实验 004 **没有验证 Learned Utility Joint 优于现有选择方法**。最可信的正向结果是：leave-one-out 标签具有非退化分布，三个 seed 都能学到稳定的跨模态粗排序信号；最可信的负向结果是：总体 validation pairwise accuracy 仍接近随机，训练很早过拟合，并且 50% 主预算下持续弱于 Raw Qwen，75% 也没有稳定优势。

这组结果更支持“当前标签/特征与目标预算不完全匹配”，而不是“MLP 容量不足”。单 chunk leave-one-out 描述的是其余内容完整时的局部边际效用，无法直接刻画同时删除大量相似或互补 chunk 后的效用；同时，视觉 chunk 数量占绝对多数、文本 label 方差远高于视觉，也使统一回归与排序更困难。现阶段不应仅靠扩大 MLP 或增加 epoch 延续路线。

由于 Oracle Utility、弱特征基线和实验 003 aligned 对照尚未运行，本轮还不能区分以下三种解释：标签本身与高压缩目标错位、当前输入特征不足、或 scorer 主要学到了模态/位置捷径。后续实验应以区分这三者为目标。

### 4.8 结果产物

本报告中的汇总结果来自以下可复核产物：

- 主配置：`configs/experiments/004_learned_utility_scorer.yaml`；
- 划分与审计：`data/interim/experiment_004/splits/`；
- 标签、checkpoint、训练历史、逐样本结果和完整日志：`/data2/wenhaozhang/runs/mllm_prune/experiment_004/`；
- 自动汇总：`outputs/experiment_004/summary.json`。

Accuracy、预算和阶段耗时取自自动汇总；label 分位数、Wilson 区间及双侧精确 McNemar $p$ 值由上述原始 `.pt`/JSONL 重新计算。运行环境未安装 SciPy，因此精确配对检验使用二项分布公式直接计算，没有依赖近似检验或外部统计库。

## 5. Next

1. 按既定安排完成实验 003，把外部 encoder 作为独立对照和 chunk 内 token 选择辅助方法；暂不把 test 结果用于调参。
2. 在 validation 上补齐 Oracle leave-one-out 排序。如果 Oracle 在 50%/25% 仍不优于 Random，应优先改为预算条件化的子集边际标签，而不是改 scorer。
3. 补 modality-only、position/coordinate-only、length-only 和 Qwen-relevance-only 弱基线，并输出同模态 pairwise、Spearman/Kendall 与 top-k recall，判断当前 64.5% 跨模态准确率是否来自捷径。
4. 若 Oracle 有明显上限，再尝试更小模型、模态平衡采样/加权、稳健 label 归一化，以及在 50%/25% 子集上采样 conditional marginal utility；所有选择只在 validation 上完成。
5. 为特征构造、scorer 前向和完整请求墙钟时间增加独立 CUDA 同步计时，再评价真实在线收益。
6. 当前固定 test 已经被查看；后续方法迭代应新增独立 held-out 样本或新 benchmark，避免继续用这 50 个样本进行模型选择。

