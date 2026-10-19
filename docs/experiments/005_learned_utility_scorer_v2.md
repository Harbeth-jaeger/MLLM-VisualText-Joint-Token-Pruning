# 005 · Aligned Prefilter + Conditional Utility Scorer

> 状态：实验设计完成，尚未实现与运行。
>
> 本实验是实验 004 的第二版。核心思路是把一次联合剪枝拆成两次：先用外部图文对齐 encoder 做保守初筛，再在初筛后的上下文中学习 conditional utility 并完成细筛。最终提交物仍是冻结的 aligned encoder、训练好的 utility scorer 与剪枝配置；推理时不需要标准答案，也不重新生成 utility 标签。

## 1. Objective

实验 004 证明了 leave-one-chunk-out utility 可以生成非平凡监督信号，但 learned scorer 没有稳定优于 Random 或 Raw Qwen。探索分析显示，主要问题集中在两个层面：

1. 原始 utility

   $$
   u_g=\ell(X\setminus g,y)-\ell(X,y)
   $$

   描述的是“其余所有 chunk 都存在时，单独删除 $g$ 的影响”，而真实 pruning 会同时删除一批 chunk。若两个 chunk 提供相同证据，单独删除任意一个都可能几乎没有损失，但同时删除二者会损失关键信息。
2. 实验 004 主要依赖 chunk hidden state 的均值池化。视觉证据经常只集中在 chunk 内少数 token 上，均值可能稀释局部强信号；同时，chunk 内 token 的最终选择仍使用未经专门对齐的 Qwen similarity。
3. 实验 004 的 loss 虽然包含 ranking，但视觉 chunk 数量很大、utility 差异普遍很小。有限 pair 若大量落在低信息视觉 pair 上，训练信号会接近噪声。
4. 实验 004 推理时使用 utility density，即以 chunk 长度惩罚预测 utility。这不符合本轮的 chunk 级价值定义：一个长视觉 chunk 即使只有少数 token 是关键证据，也不应因为其余 token 无用而被整体降权。

本实验的目的，是验证下面的两阶段假设：

> 问题相关性适合做低风险的粗粒度排除；答案条件下的边际效用适合在缩小后的候选集中做精细选择。先用 aligned relevance 删除最明显的无关内容，再学习初筛后上下文中的 conditional utility，可能比直接在完整上下文上学习 leave-one-out utility 更符合实际批量剪枝场景。

具体回答以下问题：

1. aligned encoder 初筛能否在删除一部分候选 token 的同时，高召回地保留真正高 utility 的 chunk；
2. conditional utility 是否比完整上下文上的 leave-one-out utility 更能表示实际 pruning 中的 chunk 价值；
3. 加入 aligned token similarity 统计后，scorer 能否学会视觉 chunk 内的稀疏局部证据；
4. 更有信息量的 pair 采样能否显著改善 Vision–Vision 排序；
5. “Aligned Prefilter + Conditional Utility”能否在相同严格总预算下稳定优于 Random、Aligned-only、Raw Qwen 和实验 004；
6. 方法的收益来自初筛、conditional label、输入特征还是训练 pair 设计。

本轮不声称两阶段流程已经解决所有冗余交互。若两个相似且可互相替代的 chunk 都通过初筛，它们的 conditional utility 仍可能同时较低，并在细筛中一起被删除。该问题作为明确限制和后续优化方向保留。

## 2. Setup

### 2.1 数据与划分

数据集继续使用 MileBench/MMCoQA，并复用实验 004 冻结的 group-aware 划分：

| Split | 当前规模 | 用途 |
| --- | ---: | --- |
| Train | 120 | 运行初筛、生成 conditional utility 标签、训练 scorer |
| Validation | 30 | 选择超参数、early stopping、checkpoint 和阈值 |
| Test | 50 | 固定最终评测；不得参与训练或方案选择 |

沿用实验 004 的 sample/group/image 泄漏检查。若后续扩大训练集，必须重新生成有版本号的 manifest，并保证固定 test 所关联的文档、对话、图像及重复内容不进入 train/validation。

本轮首先使用现有 120/30/50 划分，使实验 004 与 005 可直接配对比较。若 120 个训练样本不足，应在方法链路成立后扩大 train/validation，而不是通过反复查看固定 test 调参。

### 2.2 环境与模型

| 组件 | 设定 |
| --- | --- |
| Answer model / teacher | 冻结的 Qwen2.5-VL-7B-Instruct，与实验 004 保持相同 revision |
| Aligned encoder | 实验 003 选定并通过 dense text–patch alignment 检查的冻结 encoder；具体 checkpoint/revision 在实现前冻结并记录 |
| Utility scorer | 轻量共享 MLP；允许文本/视觉具有独立输入 projection |
| 可训练参数 | scorer、问题/chunk projection、modality embedding；Qwen 与 aligned encoder 均冻结 |
| 数值与生成配置 | 与实验 004 保持一致；BF16、FlashAttention 2 仅在兼容并实际启用时使用 |
| 最终部署产物 | aligned encoder 标识与配置、训练好的 scorer、特征归一化参数、预算与选择配置 |

推理阶段只有 question、context 和 images，不可使用 gold answer、teacher NLL 或真实 conditional utility。

### 2.3 候选与预算定义

沿用实验 004 的文本 chunk 和视觉空间 chunk。问题、system prompt、assistant prefix、图像边界及必要特殊 token 始终保护。

设原始可剪枝 token 总数为：

$$
N=N_t+N_v.
$$

最终候选保留率为 $\rho\in\{0.75,0.50,0.25\}$，严格 token 预算为：

$$
B=\lfloor\rho N\rfloor.
$$

总删除比例是 $1-\rho$。设初筛承担总删除量的比例为 $\alpha$，则：

$$
D_1=\alpha(1-\rho)N,
$$

$$
D_2=(1-\alpha)(1-\rho)N.
$$

$D_1$ 是第一次 pruning 删除的 token 数，$D_2$ 是第二次 pruning 继续删除的 token 数。两次删除后最终仍严格保留 $B$ 个候选 token。

例如 $\rho=0.5$：

- $\alpha=0.5$ 时，初筛先删除原始候选的 25%，细筛再删除 25%；
- $\alpha=0.25$ 时，初筛先删除 12.5%，细筛再删除 37.5%。

这里的 $\alpha$ 只描述“总删除任务有多少交给初筛”，不是最终保留率。

### 2.4 需考虑的超参数

为避免在小 validation 上过度搜索，超参数分为主实验网格和诊断项。

| 类别 | 超参数 | 首轮候选/默认值 | 作用 |
| --- | --- | --- | --- |
| 最终预算 | $\rho$ | 0.75, 0.50；0.25 为压力测试 | 控制最终压缩强度 |
| 初筛比例 | $\alpha$ | 0.25, 0.50 | 控制 aligned prefilter 的激进程度 |
| chunk aligned 聚合 | $k$ 或 top fraction | 实验 003 validation 确定 | 由 chunk 内最高相似度 token 表示局部相关性 |
| scorer similarity 统计 | top-$k$ mean、max、mean、std | 主方案全部记录 | 表示强局部证据与整体分布 |
| pair 上限 | $P$ | 256, 512, 1024/type/sample | 控制每样本每类 pair 的训练量 |
| hard-pair 比例 | $r_{hard}$ | 0.5, 0.75 | 在大 utility gap pair 与随机有效 pair 之间平衡 |
| 最小 label gap | $\epsilon$ | 由 train label 噪声诊断确定 | 过滤次序不可靠的近 tie pair |
| regression 权重 | $\lambda$ | 0, 0.1, 0.25 | ranking 为主，回归为辅 |
| scorer seed |  | 至少 3 个 | 评估训练稳定性 |

主结果不能通过在固定 test 上挑选 $\alpha,k,P,r_{hard},\epsilon,\lambda$ 得到。若组合过多，先用 validation 的 prefilter recall 和 scorer ranking 缩小候选，再运行下游 test。

### 2.5 对照与消融设定

所有方法使用完全相同的候选集定义、最终 token 预算、Qwen 生成配置和 test 样本。

#### 2.5.1 主对照

| 方法 | 目的 |
| --- | --- |
| No Pruning | 完整输入性能上限与速度基线 |
| Random Joint，至少 3 seeds | 无信息选择基线 |
| Fixed Joint | 固定模态预算基线 |
| Raw Qwen Joint | 实验 002 方法 |
| Learned Utility v1 | 实验 004 完整方法 |
| Aligned-only | 只使用实验 003 aligned score，一次剪到最终预算 |
| **Aligned Prefilter + Conditional Utility** | 实验 005 主方法 |

#### 2.5.2 机制消融

| 消融 | 与主方法的唯一区别 | 回答的问题 |
| --- | --- | --- |
| No prefilter + conditional training | $\alpha=0$ | 初筛本身是否必要 |
| Prefilter + original LOO label | 第二阶段仍使用完整上下文 $u_g$ | 收益是否来自 conditional label |
| No aligned feature | scorer 的 $X_g$ 不含 aligned similarity 统计 | similarity 特征是否帮助学习局部视觉证据 |
| Random prefilter + conditional utility | 初筛随机删同样数量 | 收益来自“先缩小集合”还是 aligned relevance |
| Uniform/random pair sampling | 不优先大 gap pair | hard/random 混合采样是否有效 |
| Utility density | 第二阶段改回 $\hat u_g/|g|^\gamma$ | 不除长度是否更合理 |

Oracle 只在 validation 小样本运行：使用真实 conditional utility 完成第二阶段选择，估计标签本身的下游上限。Oracle 读取 gold answer，因此不是可部署方法，也不能与在线速度直接比较。

### 2.6 需记录的指标

#### 初筛质量

- 删除比例与严格预算偏差；
- 原始高 utility chunk 的 recall@remaining-budget；
- 高 utility token/chunk 被初筛误删的比例；
- 分模态 recall，以及 Text/Vision 保留量；
- 初筛前后 chunk 数、token 数和冗余度分布；
- aligned-only 下游 Accuracy。

初筛最重要的指标是高价值证据召回率，而不是它独立取得多高的 Accuracy。初筛错误不可逆，因此允许保留额外假阳性，但应尽量避免删除真正关键的 chunk。

#### Label 与 scorer 质量

- conditional utility 的均值、方差、分位数、正/近零/负比例；
- 与实验 004 原始 utility 的 Spearman/Kendall 相关及 rank 变化；
- 总体和 Text–Text、Vision–Vision、Text–Vision 的 tie-aware pairwise accuracy；
- 总体和分模态 Spearman/Kendall；
- top-$k$ utility recall、NDCG 或 regret；
- train/validation loss、最佳 epoch、early stopping 情况与 seed 波动；
- scorer 分数方差、exact/near ties，避免只看单一 pairwise accuracy。

#### 下游质量与代价

- Accuracy 与 robust Accuracy；
- 相对 No Pruning、Random、Aligned-only、v1 的 paired win/loss/tie 与 McNemar 检验；
- 每个 retention 和 seed 的逐样本结果；
- 严格 token budget deviation；
- Text/Vision 实际保留 token 数；
- 初筛、scorer、planner、prefill、decode 和端到端时间；
- 峰值显存与离线标签生成成本。

### 2.7 和上一版本相比的优化

实验 005 不是简单扩大实验 004 的训练规模，而是针对上一版本暴露出的监督目标、输入特征、pair 采样和推理决策问题，对完整链路进行了修改。核心变化如下。

1. **从单阶段 pruning 改为“aligned 初筛 + learned utility 细筛”。**

   实验 004 直接让 learned scorer 在全部 chunk 中完成最终选择。实验 005 先利用冻结的图文对齐 encoder 删除总删除量中的一小部分，再让 learned scorer 处理剩余的困难候选。初筛对 chunk 内相似度最高的一部分 token 做均值聚合，而不是平均整个 chunk；其目标是保守排除明显无关内容，同时尽量保留局部强证据。

2. **从完整上下文 LOO utility 改为 conditional utility。**

   实验 004 的标签

   $$
   u_g=\ell(X\setminus g,y)-\ell(X,y)
   $$

   只衡量“其他 chunk 全部存在时，单独删除 $g$”的影响，与实际一次删除多个 chunk 的场景存在错位。实验 005 先由初筛得到剩余集合 $S$，再计算

   $$
   u(g\mid S)=\ell(S\setminus g,y)-\ell(S,y).
   $$

   scorer 因而学习的是：在一部分内容已经被删除、当前只剩 $S$ 的条件下，继续删除 $g$ 会造成多大损失。这样可以减轻冗余证据在完整上下文中互相替代、导致每个 chunk 的单独 utility 都偏低的问题。

3. **为 scorer 补充局部 aligned-similarity 特征。**

   实验 004 主要依赖 chunk hidden state 的均值池化 $p_g$ 和问题表示 $q$。均值池化会把“只有一两个 token 很关键”的局部信号稀释。实验 005 保留 $p_g$、$q$、投影后的点乘与差值，同时加入 chunk 内 aligned token similarity 的 max、mean、std 和 top-fraction mean。新特征同时表达整体语义、问答匹配和局部强证据，而不是用 similarity 完全替代 Qwen 表示。

4. **把 pair 数量和 hard/random 比例显式作为超参数。**

   实验 004 每类每样本最多只使用 256 个 pair。视觉 chunk 很多，而大量 utility 近似相同的视觉 pair 几乎不提供有效排序信号。实验 005 提高候选 pair 上限，并把每类 pair 上限 $P$、大 utility-gap pair 比例 $r_{hard}$ 和最小有效 gap $\epsilon$ 纳入 validation 选择。大 gap pair 提供更可靠的 rank 监督，随机有效 pair 则防止模型只学习 utility 排名两端而忽略中间区域。

5. **第二阶段直接按预测 utility 选择 chunk，不再除以 chunk 长度。**

   实验 004 使用 utility density $\hat u_g/|g|^\gamma$。但视觉 chunk 中常见“多数 token 无关、少数 token 极其关键”的情况，用整个 chunk 长度作分母会系统性压低这类 chunk。实验 005 将 chunk 继续作为价值判断的最小单位，直接按 $\hat u_g$ 排序；只有最终预算落在某个边界 chunk 内时，才进入 token 级选择。

6. **边界 chunk 内的 token 选择改用外部图文对齐 encoder。**

   实验 004 使用 Qwen 内部 similarity 选择边界 chunk 中保留哪些 token，但文本与视觉 token 的相似度尺度并不稳定。实验 005 使用冻结 aligned encoder 的 calibrated similarity 完成 chunk 内 token 选择，使这一步建立在专门训练过的图文公共表征空间上。

整体而言，实验 005 将上一版本的单一问题拆成两层：aligned encoder 负责低风险的相关性初筛，conditional utility scorer 负责依赖当前剩余集合的价值细筛；同时保留 chunk 级决策的稳定性，只在严格预算边界处进行 token 级裁剪。

## 3. Method

### 3.1 Overview

整个方法包含三个阶段：

```text
question + text chunks + vision chunks
                │
                ▼
  Stage 1: aligned encoder 保守初筛
  chunk 内 top-k token similarity 聚合为 chunk relevance
  先删除总删除量中的 α 部分，得到剩余集合 S
                │
                ▼
  Stage 2: conditional utility scorer 细筛
  离线训练：用 gold answer 构造 u(g | S)
  在线推理：训练好的 scorer 预测 û(g | S)
  按原始预测 utility 排序，继续删到最终预算 B
                │
                ▼
  若边界 chunk 只能部分保留
  用 aligned token similarity 选择其中 token
                │
                ▼
       保留原始 Qwen feature/mRoPE
                │
                ▼
          Qwen 正常生成答案
```

第一阶段回答“这个 chunk 与问题是否明显无关”，第二阶段回答“在当前剩余证据已经存在的条件下，删除这个 chunk 会对正确答案造成多大损失”。二者不是重复打分，而是承担不同风险与目标。

### 3.2 相比实验 004 的优化项

| 实验 004 | 实验 005 | 设计内涵 |
| --- | --- | --- |
| 在完整上下文 $X$ 上计算单 chunk LOO utility | 先初筛得到 $S$，再计算 $u(g\mid S)$ | 标签更接近真实批量 pruning 时的上下文状态 |
| 一次 learned-utility pruning | aligned 初筛 + utility 细筛 | 用低成本 relevance 排除明显无关项，把昂贵学习能力用于困难候选 |
| 输入主要依赖 mean-pooled Qwen chunk feature | 增加 chunk 内 aligned token similarity 分布与 top-$k$ 聚合 | 防止少数关键 token 被均值稀释 |
| 每类最多 256 pair，半数大 gap、半数随机 | pair 数与 hard/random ratio 成为 validation 超参数 | 增强稀疏有效视觉监督，同时保留非头部 pair 的覆盖 |
| 以 utility density 选择 chunk | 主方案直接按原始 $\hat u$ 排序 | chunk 是价值判断单位，不用无关 token 数稀释关键证据价值 |
| 边界 chunk 内使用 Raw Qwen similarity | 使用 aligned encoder token similarity | 使用专门训练的图文公共空间选择局部证据 |

### 3.3 详细展开

#### 3.3.1 Chunk 与保护范围

沿用已验证的 CandidateLayout。文本按句子/语义边界形成 macro chunk，视觉按 Qwen merger 后的空间网格形成视觉 chunk。所有 chunk 共同组成：

$$
\mathcal G=\mathcal G_t\cup\mathcal G_v.
$$

问题本身不参与 pruning，因为它规定任务；system、模板、图像边界和必要特殊 token 不参与竞争，因为删除它们会改变输入协议而不是内容预算。

第一、二阶段都以 chunk 为价值判断单位。只有为了严格满足 token 预算而出现边界 chunk 时，才在 chunk 内进一步选 token。这使训练标签、scorer 输出与主要选择单位保持一致。

#### 3.3.2 第一阶段：Aligned Prefilter

冻结的 aligned encoder 分别将问题、文本局部窗口和视觉 patch/token 映射到经过验证的公共空间。对 chunk $g$ 内第 $i$ 个 token/patch，计算与问题的相似度：

$$
s_{g,i}=\operatorname{sim}(e_{g,i},e_q).
$$

将 chunk 内相似度从高到低排列，取 top-$k$ 或 top fraction 的均值：

$$
r_g^{align}=\frac{1}{|K_g|}\sum_{i\in K_g}s_{g,i},
\qquad
K_g=\operatorname{TopK}\{s_{g,i}:i\in g\}.
$$

不用全 chunk mean 的原因是：一个视觉 chunk 中可能只有写着 “Alice” 或 “won the race” 的少数局部 token 与问题直接相关，其余大面积背景不应把这些强证据平均掉。相比单纯 max，top-$k$ mean 又不容易被一个偶然高相似度 token 支配。

根据 $r_g^{align}$ 从低到高删除 chunk，直到删除约 $D_1$ 个 token，得到剩余集合：

$$
S=\operatorname{Prefilter}(\mathcal G,D_1).
$$

若最后一个 chunk 跨越 $D_1$ 边界，使用其内部 aligned token score 精确删除所需 token；同时记录该 chunk 的实际保留 mask。后续条件标签与推理必须基于这一真实的 $S$，不能假设边界 chunk 完整存在。

初筛采用保守比例 $\alpha$，因为被删除的证据无法被第二阶段恢复。它的首要目标是高 recall，而不是单独完成全部剪枝。

#### 3.3.3 第二阶段标签：Conditional Utility

训练/验证样本先运行与推理完全相同、参数已冻结的 aligned prefilter，得到 $S$。对 gold answer $y=(y_1,\ldots,y_M)$，计算：

$$
\ell(S,y)
=-\frac1M\sum_{m=1}^{M}\log p_\theta(y_m\mid S,y_{<m}).
$$

然后对每个仍在 $S$ 中的 chunk $g$ 构造删除变体 $S\setminus g$：

$$
u(g\mid S)=\ell(S\setminus g,y)-\ell(S,y).
$$

解释如下：

- $u(g\mid S)>0$：在当前剩余上下文中删除 $g$ 会让正确答案更难，应该保留；
- $u(g\mid S)\approx0$：在当前 $S$ 中，$g$ 的边际作用小或可被其他证据替代；
- $u(g\mid S)<0$：删除 $g$ 反而提高 teacher 对 gold answer 的置信度，$g$ 可能是干扰项或标签噪声。

conditional 的含义来自条件集合 $S$：同一个 chunk 在完整输入和已经删除部分内容后的上下文中可以具有不同 utility。例如两个 chunk 都说明 Alice 获胜；完整输入中分别删除任意一个都影响很小。如果初筛删掉其中一个，那么留下的另一个在 $S$ 中变成唯一证据，其 conditional utility 会明显增大。

这里仍然是单 chunk conditional marginal，不是完整的联合子集 utility。如果两个重复 chunk 都留在 $S$，二者仍可能同时得到低 utility。因此本设计只“减轻”而不宣称“解决”重复证据的交互问题。

#### 3.3.4 Scorer 输入 $X_g$

对每个 $g\in S$ 构造：

$$
X_g=[p_g',q',p_g'\odot q',|p_g'-q'|,a_g,d_g,c_g,m_g,b].
$$

各部分含义如下。

**1. Chunk 表示 $p_g'$ 与问题表示 $q'$**

从冻结 Qwen 特征得到 chunk pooled representation $p_g$ 和 question pooled representation $q$，再分别投影到共同维度：

$$
p_g'=W_m p_g,\qquad q'=W_q q.
$$

$W_m$ 可按文本/视觉分开，$W_q$ 为问题投影；它们属于 scorer，并随 scorer 一起训练。投影的作用不是篡改 Qwen，而是把来源和统计尺度不同的高维表示压缩到 scorer 更容易比较的空间。

均值池化仍保留，因为它提供 chunk 的整体语义与上下文概貌，但不再独自承担“发现局部关键 token”的任务。

**2. 交互项 $p_g'\odot q'$ 与 $|p_g'-q'|$**

- 逐元素乘积表示 chunk 与问题在哪些语义维度同时激活，可理解为“相似处”；
- 绝对差表示两者在哪些维度不一致，可理解为“相差处”。

它们给浅层 MLP 提供显式匹配信号。理论上足够大的网络可自行学出交互，但显式构造有助于在小数据下学习。是否确有收益需通过消融验证。

**3. Aligned similarity 特征 $a_g$**

至少包括：

$$
a_g=[\max_i s_{g,i},\operatorname{mean}_i s_{g,i},
\operatorname{std}_i s_{g,i},
\operatorname{TopKMean}_i(s_{g,i}),
\operatorname{TopKValues}_i(s_{g,i})].
$$

其中 top-$k$ values 可固定长度并做 padding/mask，或只保留多个 top fraction 的聚合统计。它们直接告诉 scorer：“这个 chunk 中是否存在少量与问题高度对齐的 token”，用于弥补 $p_g$ 均值池化的信息稀释。

**4. 冗余特征 $d_g$**

只在初筛后的集合 $S$ 内重新计算，而不复用完整集合的冗余度。它描述 $g$ 与当前剩余其他 chunk 的最大/平均相似度，使 scorer 知道同类证据是否仍然存在。

**5. 结构特征 $c_g$、模态 $m_g$ 与预算条件 $b$**

$c_g$ 包含 chunk 实际剩余 token 数、相对顺序、图像 ID/空间位置、模态占比等；$m_g$ 是可学习的 modality embedding。预算条件建议编码 $\rho$、$\alpha$、初筛后剩余比例与当前模态比例：

$$
b=[\rho,\alpha,|S|/N,N_t^S/|S|,N_v^S/|S|].
$$

不同 $\rho$ 和 $\alpha$ 会产生不同条件集合 $S$，同一 chunk 的价值可能随状态改变，因此 scorer 应显式知道当前剪枝状态。另一种更简单但维护成本更高的做法，是为每组 $(\rho,\alpha)$ 单独训练 checkpoint；二者作为可比较选项。

#### 3.3.5 Scorer 与预测目标

共享 scorer 输出一个实数：

$$
\hat u(g\mid S)=f_\phi(X_g).
$$

文本和视觉可使用独立输入 projection，但最终 MLP/head 共享，使两种模态输出在同一尺度上竞争。

训练标签 $u(g\mid S)$ 只在离线 label generation 中由 gold answer 和冻结 teacher 产生。在线推理时，scorer 只根据 $X_g$ 预测 $\hat u(g\mid S)$。因此部署成本不包含 leave-one-out teacher scoring，也不要求用户自己训练模型。

#### 3.3.6 Loss 与 Pair 构造

对同一样本中满足：

$$
u_i-u_j>\epsilon
$$

的 pair $(i,j)$，使用：

$$
\mathcal L_{rank}
=\operatorname{softplus}[-(\hat u_i-\hat u_j)].
$$

辅助回归项使用样本内稳健归一化后的 conditional utility：

$$
\mathcal L
=\mathcal L_{rank}+\lambda\mathcal L_{Huber}.
$$

ranking 是主目标，因为细筛最关心 chunk 的相对保留顺序；Huber 让模型保留 utility 强弱信息，并限制极端 NLL delta 的影响。

pair 分为 Text–Text、Vision–Vision 和 Text–Vision。每个样本、每种 pair 类型最多选择 $P$ 个有效 pair：

1. 将满足最小 gap 的 pair 按 $|u_i-u_j|$ 排序；
2. 取 $r_{hard}P$ 个大 gap pair；
3. 从其余有效 pair 中随机采样 $(1-r_{hard})P$ 个；
4. 不足 $P$ 时使用全部有效 pair，不重复填充。

大 gap pair 的排序标签更可靠、学习价值更高；随机部分避免模型只认识最显著的头部差异，而忽略中等或低排名 chunk。$P$ 和 $r_{hard}$ 是 validation 超参数，不能固定认为 256 一定足够。

这里“hard”沿用当前项目语境，实际指 utility gap 大、监督方向更可靠的 pair；若后续引入模型当前最易混淆的 hard-negative mining，应另行命名和区分。

#### 3.3.7 第二阶段选择与边界 Chunk

对 $S$ 中所有 chunk 一次预测 $\hat u(g\mid S)$，主方案直接按预测 utility 从高到低排序：

$$
g_{(1)},g_{(2)},\ldots
=\operatorname{SortDesc}_{g\in S}\hat u(g\mid S).
$$

不除以 chunk 长度。原因是 $\hat u$ 表示删除整个 chunk 对答案的预测损失；若 64-token 视觉 chunk 只有 2 个 token 是关键证据，长度惩罚会再次把关键价值稀释掉。

planner 按排序依次保留 chunk，直到接近最终预算 $B$：

- 能完整放入剩余预算的 chunk 全部保留；
- 第一个超出剩余预算的边界 chunk，只保留其中 aligned similarity 最高的 token；
- 若后面还有更短 chunk，主方案不因“更容易塞满”而跳过高 utility chunk，剩余位置由当前边界 chunk 精确补齐；
- 最终验证保留候选 token 数严格等于 $B$。

该规则把 chunk 作为排序单位，同时允许严格 token 预算。应额外报告 partial chunk 的频率和来自其中的 token 数，以判断边界规则是否实际影响很大。

#### 3.3.8 Chunk 内 Token 选择

当初筛或细筛只能部分删除/保留一个 chunk 时，按 token 级 aligned similarity 选择，而不是 Raw Qwen cosine：

$$
K_g^*=\operatorname{TopK}_{i\in g}s_{g,i}.
$$

对文本，token 分数由实验 003 的局部文本窗口映射得到；对视觉，token 分数由经过验证的 dense image–text alignment 和网格映射得到。aligned encoder 只决定选择，不替换 Qwen 的输入 embedding。

#### 3.3.9 重建与正常推理

两次 pruning 得到最终 token mask 后：

1. gather 被保留 token 的原始 Qwen embedding；
2. gather 对应的原始 mRoPE position IDs；
3. 保持原始相对顺序和必要边界；
4. 将压缩后的序列输入冻结 Qwen；
5. 使用与所有对照完全一致的生成配置得到答案。

因此方法只改变“保留哪些 token”，不改变 Qwen 参数、保留 token 的表示或答案生成规则。

### 3.4 完整闭环

#### 3.4.1 训练闭环

对每个 train/validation 样本：

1. 构造文本/视觉 chunk，并保护 question 和特殊 token；
2. 用冻结 aligned encoder 计算 token similarity 与 chunk top-$k$ relevance；
3. 按给定 $(\rho,\alpha)$ 运行初筛，得到真实剩余集合 $S$；
4. 用冻结 Qwen 和 gold answer 计算 $\ell(S,y)$；
5. 对每个 $g\in S$ 计算 $\ell(S\setminus g,y)$，得到 $u(g\mid S)$；
6. 从 $S$ 构造 Qwen pooled feature、问题交互、aligned similarity、冗余、结构、模态和预算特征；
7. 按模态类型、utility gap、$P$ 和 $r_{hard}$ 采样 pair；
8. 用 ranking + Huber loss 更新 scorer；
9. 只依据 validation ranking、top utility recall 和下游小规模验证选择 checkpoint；
10. 冻结 scorer。固定 test 不参与上述任何选择。

#### 3.4.2 推理闭环

对一个从未见过、没有 gold answer 的样本：

1. 用相同 chunk 规则和 aligned encoder 完成第一次 pruning；
2. 在剩余集合 $S$ 上构造 scorer 输入；
3. 用冻结 scorer 一次得到所有 $\hat u(g\mid S)$；
4. 按原始预测 utility 保留 chunk，边界 chunk 内按 aligned similarity 保留 token；
5. 严格保留 $B$ 个候选 token；
6. Qwen 正常生成答案。

推理时不会计算 $u(g\mid S)$，因为没有 gold answer；真实 utility 只是训练监督，$\hat u$ 才是部署时使用的分数。

#### 3.4.3 实际 Case

假设一个样本的问题是：

> Who won the race?

候选内容共有 400 个 token，最终保留率 $\rho=0.5$，因此最终预算 $B=200$。取 $\alpha=0.5$，初筛和细筛各负责删除 100 个 token。

主要 chunk 如下：

| Chunk | 内容 | token 数 | 直观作用 |
| --- | --- | ---: | --- |
| $T_1$ | “Alice won the race.” | 40 | 直接文本证据 |
| $V_1$ | 终点牌局部写着 “Winner: Alice” | 64 | 直接视觉证据，只有少数 patch 含文字 |
| $T_2$ | 重复描述 Alice 获胜 | 40 | 与 $T_1$ 冗余 |
| $V_2$ | 观众与赛场背景 | 64 | 与问题主题相关但不含答案 |
| 其余 | 其他赛前介绍、无关图片与文本 | 192 | 大多无关 |

**第一阶段：aligned 初筛。**

$V_1$ 的全 chunk mean 可能不高，因为只有少数 patch 包含 “Winner: Alice”；但这些 patch 与问题中的 “who/won/race” 高度对齐，因此 top-$k$ mean 很高。$T_1$、$T_2$ 也较高；背景 $V_2$ 可能有 “race” 相关性，但缺少 “winner/person” 证据，分数中等；无关介绍最低。

初筛从低分 chunk 开始删除 100 token。假设无关介绍被删掉，$T_1,T_2,V_1,V_2$ 都暂时留下，形成 $S$。此时初筛完成的是保守排除，不负责判断 $T_1$ 与 $T_2$ 谁更有边际价值。

**第二阶段训练标签。**

训练时已知 gold answer 是 “Alice”。冻结 Qwen 分别比较：

$$
\ell(S,\text{Alice})
$$

和删除每个 chunk 后的答案 NLL。可能得到：

| Chunk | conditional utility（示意） | 解释 |
| --- | ---: | --- |
| $T_1$ | 0.05 | $T_2,V_1$ 仍提供 Alice，单独删除影响小 |
| $T_2$ | 0.03 | 与 $T_1$ 重复，边际价值小 |
| $V_1$ | 0.42 | 视觉终点牌强化并定位正确答案 |
| $V_2$ | -0.01 | 背景没有答案作用，甚至略有干扰 |

这些数值只用于说明机制，不是实验结果。scorer 使用各 chunk 的 $X_g$ 学习预测上述相对顺序。$V_1$ 的 Qwen mean feature 即使被背景稀释，aligned top-$k$ similarity 仍能提示其中存在少量强证据。

**第二阶段在线细筛。**

对 test 样本没有 “Alice” 这个 gold label。冻结 scorer 只看问题、chunk 特征、aligned similarity 和当前 $S$，预测例如：

$$
\hat u_{V_1}>\hat u_{T_1}>\hat u_{T_2}>\hat u_{V_2}.
$$

planner 按这个顺序保留 chunk，继续删除 100 token 直到总预算为 200。若只剩 20 个 token 容量可分给 $V_1$，就在 $V_1$ 内保留 aligned similarity 最高的 20 个 patch/token，因而优先保留 “Winner: Alice” 区域，而不是平均或随机选择背景。

**正常推理。**

最终 mask 对应的原始 Qwen embedding 和 mRoPE 被 gather，Qwen 在压缩上下文上生成 “Alice”。整个在线过程没有运行 leave-one-out，也没有使用标准答案。

**该 case 暴露的剩余失败模式。**

如果 $T_1$ 和 $T_2$ 都通过初筛，二者的 conditional utility 仍可能都很低；当 $V_1$ 也未被 scorer 正确识别时，细筛可能把重复文本一起删掉。因此 conditional LOO 只利用了初筛带来的状态变化，尚未直接学习“同时删除一个集合”的风险。

### 3.5 潜在可修改项

以下内容不是首版同时加入的复杂机制，而是在对应诊断证据出现后考虑的下一版本选项。

#### 3.5.1 初筛

- 若高 utility recall 低：减小 $\alpha$、增大 top-$k$ 覆盖，或为负相关/否定/数字/OCR 证据加入保护规则；
- 若两个高相关重复 chunk 总是一起进入 $S$：加入 MMR/多样性约束，使 relevance 与冗余共同决定初筛；
- 若视觉 dense alignment 不可靠：更换 encoder，或只将 aligned score 用于文本/全图，不强行使用未验证的 patch feature；
- 若不同模态分数仍不可比：使用实验 003 的冻结 calibration artifact，而不是在单个样本内强行归一化成相同分布。

#### 3.5.2 Conditional Label

- 若 conditional oracle 仍无明显收益：说明单 chunk marginal 与批量删除目标不匹配，可改为 pair/group deletion utility；
- 若 utility 对不同初筛状态变化大：为同一样本采样多个 $S$，训练 state-conditioned scorer；
- 若标签噪声大：重复 teacher scoring、使用 token-level NLL diagnostics，或过滤低置信/短答案异常样本；
- 若冗余问题突出：近似 Shapley、subset marginal、iterative deletion/re-scoring 可作为更强但更贵的标签。

#### 3.5.3 输入特征

- 将 top-$k$ scalar statistics 改为 attention pooling 或小型 set encoder，直接读取 chunk 内 token feature；
- 使用 aligned token feature 与 Qwen token feature 的轻量 cross-attention，而不只输入相似度；
- 加入 OCR、数字、实体、否定词或答案类型等任务特征；
- 重新设计 redundancy，使其不仅比较语义相似，还比较是否提供同一答案证据。

#### 3.5.4 Loss 与 Pair

- 使用全 pair、listwise loss、LambdaRank 或 differentiable top-$k$，减少采样偏差；
- 用模型当前预测最容易颠倒的 pair 做真正的 hard-negative mining；
- 按 label uncertainty 对 pair 加权，而不是仅用固定 $\epsilon$ 过滤；
- 对 Vision–Vision 增加配额，但必须先确认额外 pair 不是大量无意义近 tie。

#### 3.5.5 第二阶段选择

- 若 raw utility 总偏向长 chunk，可比较 raw utility、density 和严格 budget knapsack；
- 若一次预测无法反映连续删除后的状态变化，可小批量删除并重新计算 scorer 特征；
- 若冗余 chunk 被成组误删，可加入保留多样性、覆盖约束或 group-level safety rule；
- 若 partial chunk 频繁出现且影响明显，可学习 chunk 内 token scorer，而不再只用 aligned similarity。

## 4. Results & Analysis & Conclusion

> 状态：主流程已完成。以下结论来自冻结的 120/30/50 train/validation/test 划分、6 个 $(\rho,\alpha)$ 状态、3 个 scorer seed 和固定 50 样本 test。Conditional Oracle 与预注册的机制消融未运行，因此本轮能够判断“完整 v2 是否有效”，但不能完全分解每个组件的独立因果贡献。

### 4.1 结果是否完善 / 实验是否正常结束

第一次运行在 calibration 样本 21 遇到纯空白文本窗口后停止；修复方式是保留该 token 的覆盖关系，但不将空字符串送入 CLIP，而将其 raw relevance 设为余弦下界 $-1$。回归测试和原失败样本通过后，正式流程于 2026-10-18 23:44:07 CST 重新启动，并于 2026-10-19 04:38:45 CST 正常结束，成功运行墙钟为 **4 小时 54 分 38 秒**。第一次失败发生在正式 artifact 写出前，不计入成功运行耗时。

完整性审计如下：

| 产物 | 预期 | 实际 | 状态 |
| --- | ---: | ---: | --- |
| Aligned cache：train / validation / test | 120 / 30 / 50 | 120 / 30 / 50 | 完整 |
| Conditional-label record：train | $120\times6=720$ | 720 | 完整 |
| Conditional-label record：validation | $30\times6=180$ | 180 | 完整 |
| Scorer checkpoint / history | 3 / 3 | 3 / 3 | 完整 |
| Test 条件 | 37 | 37 | 完整 |
| Test 逐样本记录 | $37\times50=1850$ | 1850，全部唯一 | 完整 |
| 严格预算 | 所有 pruning row 偏差为 0 | 全部通过 | 完整 |

37 个 test 条件由 1 个 No Pruning、每个 retention 下的 Fixed、Raw Qwen、Aligned-only、3 个 Random seed，以及 $2$ 个 $\alpha\times3$ 个 scorer seed 的 v2 组成。所有 test 条件均有 50 个样本，没有 duplicate key、缺失条件、NaN、OOM 或静默 attention fallback。

本轮**已完成**主对照：No Pruning、Fixed、Random、Raw Qwen、Aligned-only 和完整 v2。实验 004 的 v1 结果来自同一固定 test，可作历史配对比较。

本轮**未完成**：Conditional Oracle、No-prefilter conditional、original LOO label、去 aligned feature、random prefilter、uniform/random pair、utility density 六项机制消融。因此不能把结果差异唯一归因于 conditional label、aligned feature 或 raw-utility planner 中的某一个组件。

数据纪律方面，aligned empirical-CDF calibration 使用 50 个不含答案的输入样本；它与固定 test 的 sample ID 交集为 0。其与 train/validation 分别有 40/10 个输入样本重叠，这不会把 test answer 引入训练，但意味着 calibration 不是一个独立于 scorer train/validation 的第四 split；解释 validation score 时应保留这一边界。

产物总大小约 946 MB：aligned cache 5.7 MB、conditional labels 826.2 MB、checkpoint 36.8 MB、prediction JSONL 120.3 MB。

### 4.2 最优超参数 / 实验所用超参数汇总

本轮没有用 test 选择单一“最优” $\alpha$；两个预注册值都完整报告。$\rho$ 是部署预算而非待选择超参数，也全部报告。

| 项目 | 实际设定 |
| --- | --- |
| Answer model / teacher | Qwen2.5-VL-7B-Instruct，冻结，BF16，FlashAttention 2 |
| Aligned encoder | `clip-vit-base-patch16`，revision `main`，冻结，BF16 |
| Aligned calibration | 分模态 empirical CDF；50 个无 test-ID 样本 |
| Text aligned window | 16 tokens，stride 8，重叠取 mean |
| Chunk | text 16–64 tokens；vision $4\times4$ merger-token window |
| Final retention $\rho$ | 0.75 / 0.50 / 0.25 |
| Prefilter share $\alpha$ | 0.25 / 0.50，两个值均进入训练和 test |
| Token/chunk top fraction | 0.20 / 0.20 |
| Aligned feature top fraction | 0.20；max/mean/std/top-mean 四维统计 |
| Scorer | modality-specific 3584→256 projection；shared hidden 256；modality embedding 16；dropout 0.10 |
| Pair | TT/VV/TV 每类每样本最多 512；50% 大 gap + 50% random valid pair |
| Gap threshold | $\epsilon=0.001$ |
| Loss | pairwise softplus + $0.25\times$ Huber |
| Optimizer | AdamW，lr $3\times10^{-4}$，weight decay $10^{-4}$ |
| Training | 最多 100 epochs，patience 12，seeds 0/1/2 |
| Planner | raw predicted utility，不除 chunk 长度；边界 chunk 按 calibrated aligned token score |

checkpoint 只按 validation macro pairwise accuracy 选择：

| Seed | 最佳 epoch | 实际训练轮数 | 最佳 validation macro pairwise | 首轮 → 末轮 train loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 25 | 52.63% | 1.014 → 0.679 |
| 1 | 10 | 23 | 52.49% | 1.011 → 0.687 |
| 2 | 15 | 28 | 52.61% | 1.009 → 0.662 |

训练 loss 明显下降，但 validation 排名长期停留在约 52.5%，仍表现为拟合训练数据而泛化排序能力弱。

### 4.3 各对照组指标结果汇总

#### 4.3.1 初筛质量

本轮没有预先保存 test oracle utility，因此初筛诊断使用 validation 上实验 004 的完整上下文 LOO label：把每个样本 utility 最高的 20% chunk 视为“高 utility”，测量它们经过 aligned prefilter 后仍存在的比例。该指标不是 conditional oracle，只用于判断 aligned relevance 是否至少比无信息删除更善于保留 v1 高价值 chunk。

| $\rho$ | $\alpha$ | 初筛后理论保留率 $B_1/N$ | Overall top-20 recall | Text recall | Vision recall | 同预算 Random 期望 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 75% | 0.25 | 93.75% | 91.62% | 83.89% | 92.73% | 93.75% |
| 75% | 0.50 | 87.50% | 83.67% | 81.11% | 84.64% | 87.50% |
| 50% | 0.25 | 87.50% | 83.67% | 81.11% | 84.64% | 87.50% |
| 50% | 0.50 | 75.00% | 73.63% | 67.39% | 75.30% | 75.00% |
| 25% | 0.25 | 81.25% | 78.81% | 76.83% | 80.07% | 81.25% |
| 25% | 0.50 | 62.50% | 61.15% | 54.44% | 62.80% | 62.50% |

六个状态的 overall recall 都没有超过其同预算 Random 期望，且 text recall 始终更低。这个结果与 Aligned-only 的下游表现一致：其 Accuracy 在 75%/50%/25% 下分别为 62%/48%/48%，没有形成可靠的独立基线。

因此，aligned prefilter 并未证实“语义相关性能够高召回地保护 utility 证据”。它可能仍能保留某些局部视觉线索，但就 v1 utility 排名而言，没有比随机保留更好。

#### 4.3.2 Conditional Label

每个 $(sample,\rho,\alpha)$ 状态单独生成 conditional label。总体分布均非退化：近零比例约 6.2%–12.2%，正负 utility 都大量存在。

| Split | $\rho$ | $\alpha$ | Chunk 数 | 平均 chunk/样本 | Utility mean ± std | 正 / 近零 / 负 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 75% | 0.25 | 16,629 | 138.6 | 0.0017 ± 0.1522 | 40.9% / 12.2% / 46.9% |
| Train | 75% | 0.50 | 15,523 | 129.4 | 0.0019 ± 0.1628 | 44.4% / 10.8% / 44.7% |
| Train | 50% | 0.25 | 15,527 | 129.4 | 0.0039 ± 0.1609 | 45.6% / 10.3% / 44.1% |
| Train | 50% | 0.50 | 13,256 | 110.5 | 0.0087 ± 0.1942 | 45.9% / 9.6% / 44.5% |
| Train | 25% | 0.25 | 14,395 | 120.0 | 0.0030 ± 0.1596 | 47.9% / 9.2% / 42.9% |
| Train | 25% | 0.50 | 11,074 | 92.3 | 0.0076 ± 0.1948 | 46.9% / 9.0% / 44.1% |

validation 的 conditional utility 与同一 surviving chunk 的实验 004 原始 LOO utility，样本内 Spearman 只有：

| $\rho$ | $\alpha=0.25$ | $\alpha=0.50$ |
| ---: | ---: | ---: |
| 75% | 0.234 | 0.214 |
| 50% | 0.225 | 0.180 |
| 25% | 0.194 | 0.126 |

这说明 state 改变后标签排序确实发生了明显变化，conditional label 并非简单复制 v1；但低相关既可能代表成功捕获条件依赖，也可能代表更高的标签噪声。由于没有运行 Conditional Oracle，本轮无法区分这两种解释。

#### 4.3.3 Scorer 排序质量

下表中的总体 pairwise 为训练时按样本 macro 平均的 checkpoint 选择指标；TT/VV/TV 为报告阶段在全部 validation 有效 pair 上的 micro accuracy。Spearman/Kendall 和 top-20 recall 为样本 macro 平均。

| Seed | Overall pairwise | Text–Text | Vision–Vision | Text–Vision | Spearman | Kendall | Top-20 utility recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 52.63% | 57.94% | 50.70% | 66.95% | 0.072 | 0.049 | 27.38% |
| 1 | 52.49% | 59.31% | 51.01% | 66.62% | 0.066 | 0.046 | 25.91% |
| 2 | 52.61% | 57.99% | 51.02% | 67.39% | 0.071 | 0.049 | 26.93% |

三个 seed 均没有 prediction exact tie，但 Vision–Vision 仍接近随机，整体相关性甚至略低于 v1 探索分析中的 0.086–0.120。跨模态约 67% 说明模型仍能区分一部分文本/视觉尺度，但没有学会占绝大多数候选的视觉内部细粒度顺序。加入 aligned statistics、状态特征和更多 pair，并未解决 v1 的核心 scorer 问题。

#### 4.3.4 Test 下游主结果

固定 50 test 样本上的 Accuracy：

| 方法 | 75% | 50% | 25% |
| --- | ---: | ---: | ---: |
| Fixed Joint | 64.0% | 62.0% | 42.0% |
| Random Joint，3-seed mean ± std | 64.0% ± 3.3% | **62.7% ± 4.1%** | 47.3% ± 0.9% |
| Raw Qwen Joint | **66.0%** | **66.0%** | 46.0% |
| Aligned-only | 62.0% | 48.0% | 48.0% |
| Learned Utility v1，3-seed mean ± std | 64.7% ± 2.5% | 60.0% ± 1.6% | 47.3% ± 2.5% |
| v2，$\alpha=0.25$，3-seed mean ± std | 56.0% ± 1.6% | 58.7% ± 1.9% | **54.7% ± 0.9%** |
| v2，$\alpha=0.50$，3-seed mean ± std | 58.7% ± 0.9% | 58.0% ± 0.0% | 52.7% ± 6.8% |

No Pruning 为 70.0%。对应 Robust Accuracy：

| 方法 | 75% | 50% | 25% |
| --- | ---: | ---: | ---: |
| Random Joint | 66.0% ± 3.3% | 64.0% ± 4.3% | 48.7% ± 0.9% |
| Raw Qwen Joint | **68.0%** | **66.0%** | 46.0% |
| Aligned-only | 64.0% | 50.0% | 50.0% |
| Learned Utility v1 | 66.7% ± 2.5% | 61.3% ± 2.5% | 49.3% ± 2.5% |
| v2，$\alpha=0.25$ | 58.0% ± 1.6% | 60.7% ± 1.9% | **56.7% ± 0.9%** |
| v2，$\alpha=0.50$ | 60.7% ± 0.9% | 60.0% ± 0.0% | 53.3% ± 6.2% |

v2 的逐 seed Accuracy：

| $\rho$ | $\alpha$ | Seed 0 | Seed 1 | Seed 2 |
| ---: | ---: | ---: | ---: | ---: |
| 75% | 0.25 | 58% | 54% | 56% |
| 75% | 0.50 | 58% | 60% | 58% |
| 50% | 0.25 | 60% | 60% | 56% |
| 50% | 0.50 | 58% | 58% | 58% |
| 25% | 0.25 | 56% | 54% | 54% |
| 25% | 0.50 | 62% | 46% | 50% |

主要观察：

1. **75%：v2 明显退步。** 两个 $\alpha$ 均低于 Raw Qwen、Random 和 v1。$\alpha=0.25$ 的 seed 1 相对 Raw Qwen 为 0 个由错变对、6 个由对变错，精确 McNemar $p=0.031$；相对同 seed Random 为 0/7，$p=0.016$。这不是稳定正结果。
2. **50%：v2 没有改善。** 两个 $\alpha$ 均约 58%–59%，低于 Raw Qwen 66% 和 Random mean 62.7%，与 v1 的 60% 接近。所有 v2-v1 配对检验 $p=1.0$ 或更高，差异不足以区分。
3. **25%：$\alpha=0.25$ 有稳定方向但证据仍弱。** 三个 seed 均为 54%–56%，平均比 Raw Qwen 高 8.7 percentage points、比 Random/v1 高 7.3 points。逐 seed 相对 Raw Qwen 的 win/loss 为 7/2、7/3、7/3，对应 $p=0.180,0.344,0.344$；样本只有 50 个，尚不显著。
4. **25%、$\alpha=0.50$ 不稳定。** seed 0 达 62%，但 seed 1/2 只有 46%/50%，标准差 6.8 points。seed 0 相对 Raw Qwen 的 10/2 达 $p=0.039$，但其他 seed 不复现，不能据此选择 seed 0 或宣称方法成立。
5. **不存在一个预先固定的 $\alpha$ 在三个 retention 上都好。** 在 test 后观察到 75% 偏向 $\alpha=0.50$，50%/25% 偏向 $\alpha=0.25$，但这只能用于解释，不能作为无偏超参数选择。

#### 4.3.5 速度、显存与离线成本

Aligned score cache 在 200 个样本上的纯 external text/image encoding + mapping 平均约 47–56 ms/样本；正式 test 使用预缓存，因此主结果中的 planning time不含这部分。实际部署若不能预缓存，应将约 56 ms/sample 加回在线成本。

| 条件 | Planning ms | Prefill ms | 相对 No Pruning 的 prefill 降幅 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| No Pruning | 33.0 | 263.8 | — | 16.43 GB |
| v2 75%，所有 $\alpha$/seed 范围 | 36.7–40.1 | 198.1–198.2 | 24.9% | 16.23 GB |
| v2 50%，所有 $\alpha$/seed 范围 | 33.4–38.1 | 137.7–137.9 | 47.8% | 16.03 GB |
| v2 25%，所有 $\alpha$/seed 范围 | 28.0–34.4 | 76.3–76.4 | 71.1% | 15.84 GB |

decode 时间受答案长度和生成内容影响，不能只归因于 pruning；本轮观察到的均值约为 360–427 ms。在线 scorer/planner 开销为几十毫秒，若再加 external encoder 约 56 ms，仍远小于离线标签成本，但在 75% 的温和剪枝下会抵消相当一部分 prefill 节省。

Conditional label 生成的累计 GPU 时间约 **7.51 GPU-hours**：train 约 5.72 GPU-hours，validation 约 1.79 GPU-hours。单 state 的平均样本耗时随剩余 chunk 数变化，train 为 16.8–37.4 s，validation 为 21.3–46.6 s；峰值显存分别约 20.37/20.80 GB。两卡样本并行使完整成功运行的墙钟控制在约 4.9 小时。

### 4.4 结论

本轮的直接结论是：**实验 005 的完整方法没有成为稳定优于简单基线的主方案。**

1. **Aligned prefilter 假设没有得到支持。** 它对 v1 top-20 utility chunk 的 recall 不高于同预算 Random 期望，Aligned-only 在 50% 下甚至只有 48% Accuracy。当前 CLIP dense/token relevance 与答案 utility 明显错位。
2. **Conditional label 确实改变了监督目标，但尚不知是否变得更正确。** 它与原始 LOO 排名相关很低，说明状态条件发挥了作用；然而缺少 Conditional Oracle，无法判断变化是有价值的交互信息还是更大的 NLL 噪声。
3. **scorer 的核心问题仍未解决。** Vision–Vision pairwise 仍约 51%，整体 Spearman 约 0.07。更多 pair、aligned statistics 和 budget conditioning 没有让模型可靠学习视觉内部排序。
4. **主要预算下没有收益。** 75% 明显差于 Raw Qwen/Random/v1；50% 也没有超过它们。只有 25%、$\alpha=0.25$ 在三个 seed 上给出一致正向趋势，但 50 样本配对检验不显著，不能升级为肯定结论。
5. **更激进的初筛放大 seed 不稳定性。** 25%、$\alpha=0.50$ 从 46% 到 62%，说明 scorer 与不可逆 prefilter 的组合对训练随机性敏感。
6. **工程链路可行，但研究信号不足。** 缓存、两阶段严格预算、conditional label、scorer 训练和在线推理均正常；失败更可能是方法/监督问题，而不是流程没有跑通。

因此，不建议立即扩大训练集或继续在同一结构上增加 MLP 容量。先验证 Conditional Oracle 和初筛替代方案，才能判断下一步应修 label、修 scorer，还是放弃 learned utility 路线。

## 5. Next

下一步按信息价值排序：

1. **先补 Conditional Oracle，不训练新 scorer。** 在 validation 上直接用真实 $u(g\mid S)$ 走相同 raw-utility planner，并与 Original LOO Oracle、Aligned-only 和 Random 配对比较。这一步能回答 conditional label 本身是否具有更高下游上限。
2. **分析 prefilter 误删 case。** 优先检查 validation 中 aligned score 删除了 v1 高 utility chunk 的样本，区分文本窗口、视觉 patch alignment、CDF calibration、top-20% 聚合和语义相关/答案效用错位。
3. **检查 25%、$\alpha=0.25$ 的稳定增益来自哪里。** 对三个 seed 共同由错变对、共同由对变错的 case 做可视化，比较 v1/v2/aligned/raw 的 chunk 和边界 token 选择。目标是判断它是否来自 conditional utility，还是仅来自不同的模态预算。
4. **若 Conditional Oracle 有效而 scorer 无效**，再考虑扩大独立训练样本、token/set encoder、listwise/top-budget loss，或按 state 单独训练；当前 120 个样本展开成 720 条相关 state record，并不等于 720 个独立样本。
5. **若 Conditional Oracle 无效**，停止优化 MLP，转向 pair/group deletion utility、MMR/coverage 或 iterative state-aware pruning，直接处理“两个替代 chunk 同时被删”的组合风险。
6. **不要继续用固定 50 test 调参。** 该 test 已用于实验 004、005 的结果分析；下一版超参数和机制选择只能使用 train/validation。若形成新主方案，应准备新的未见 test 或更大 benchmark 做最终确认。
