建议把实验 004 实现为四个彼此解耦的阶段：group-aware 数据划分、冻结 Qwen 的反事实 utility 标签与特征生成、离线轻量 scorer 训练、加载 scorer 的联合剪枝与评测。

本实验不以实验 003 的完成为前提。首版 scorer 只依赖冻结的 Qwen 输入特征、问题特征、模态与位置等结构特征；实验 003 的 aligned relevance 仅作为可选输入特征和后续消融。这样即使不运行实验 003，也能完整验证“答案 utility 是否比语义相似度更适合剪枝”这一核心假设。

Qwen2.5-VL-7B-Instruct 在标签生成和最终答案评测中始终冻结。训练阶段只更新轻量 scorer 及必要的小型投影层。这是本轮的主要归因边界：若 learned utility 优于现有方法，收益应来自更好的 token/chunk 选择，而不是答案模型被额外微调。

## 1. Problem

### 1.1 要解决的问题

沿用实验 002 的可剪枝候选和严格联合预算。对样本 $x$，文本和视觉 chunk 集合为：

$$
\mathcal G(x)=\mathcal G_t(x)\cup\mathcal G_v(x).
$$

问题、system prompt、图像边界和必要特殊 token 均受保护。若文本和视觉候选 token 数分别为 $N_t,N_v$，给定保留率 $\rho$，总预算为：

$$
B=\left\lfloor\rho(N_t+N_v)\right\rfloor.
$$

实验 002 使用问题与候选 Qwen embedding 的 cosine 作为 relevance；实验 003 计划使用外部图文对齐空间解决跨模态数值不可比问题。但即使文本和视觉 relevance 已可比较，语义相关仍不等价于对答案的实际贡献：

* 与问题同主题的 chunk 可能不包含答案证据；
* 两个高相关 chunk 可能提供重复信息；
* 某个表面相关性较低的数字、否定词或局部视觉区域可能决定答案；
* chunk 的价值会随其他上下文是否存在而变化。

本轮用冻结 teacher 的 leave-one-chunk-out 反事实效应定义监督信号。对 gold answer $y=(y_1,\ldots,y_M)$，完整输入的平均答案 NLL 为：

$$
\ell(X,y)=-\frac{1}{M}\sum_{k=1}^{M}\log p_\theta(y_k\mid X,y_{<k}).
$$

删除 chunk $g$ 后的 utility 为：

$$
u_g=\ell(X\setminus g,y)-\ell(X,y).
$$

$u_g>0$ 表示删除该 chunk 会降低 teacher 对正确答案的置信度，应该优先保留；$u_g<0$ 表示该 chunk 可能是干扰项或标签存在数值噪声。

学习目标是构造一个共享 scorer：

$$
\hat u_g=f_\phi(z_g),\qquad g\in\mathcal G_t\cup\mathcal G_v,
$$

使文本和视觉 chunk 的预测 utility 可以在同一总预算下直接竞争。算法必须满足：

* train/validation/test 按源文档、对话和图像 group 隔离；
* 实验 002/003 的固定 50 样本只用于最终 test；
* utility 只计算 gold answer token 的 NLL，不包含 prompt、padding 或图像占位 token；
* 反事实删除复用已有序列重建、保护 token 和原始 mRoPE 位置逻辑；
* scorer 不读取 gold answer 文本或 teacher loss 作为在线输入；
* 最终选择严格使用总预算 $B$，不因模态不同放宽预算；
* 不改变 Qwen 权重、生成参数和被保留 token 的原始特征。

### 1.2 输入与输出

数据划分阶段输入：

```text
SplitPreparationInput
├─ MMCoQA sample index
├─ source document / conversation / image identifiers
├─ fixed 50-sample test manifest
├─ duplicate-image fingerprints
└─ split seed and target proportions
```

输出冻结的 manifest：

```text
SplitArtifact
├─ format_version
├─ train / validation / test sample IDs
├─ group IDs and split assignment
├─ source-data fingerprint
├─ duplicate and leakage audit
├─ split seed and construction config
└─ fixed-test-manifest fingerprint
```

标签与特征生成阶段输入：

```text
UtilityLabelInput
├─ frozen split manifest
├─ MMCoQA images, context, question and gold answer
├─ frozen Qwen model + processor revision
├─ existing CandidateLayout and chunk definitions
├─ pooling / structural feature configuration
└─ deletion-variant batching configuration
```

每个样本输出一个可断点续跑的 shard record：

```text
UtilitySampleRecord
├─ sample_id / group_id / split
├─ model, processor and code fingerprints
├─ complete-input answer NLL and answer-token count
├─ chunk IDs, modality, capacity and original spans/grid cells
├─ pooled chunk/question features
├─ structural and redundancy features
├─ leave-one-out answer NLL and utility labels
├─ numerical, timing and memory diagnostics
└─ optional aligned relevance
```

训练阶段输入所有 train/validation records，输出：

```text
ScorerArtifact
├─ format_version
├─ scorer architecture and state_dict
├─ feature schema and normalization statistics
├─ split / data / teacher fingerprints
├─ training seed and optimization config
├─ best validation epoch and metrics
├─ train/validation curves
└─ compatibility metadata
```

推理阶段输出：

```text
LearnedUtilityResult
├─ predicted utility per chunk
├─ utility density and selected chunks/tokens
├─ text/vision retained counts
├─ strict-budget diagnostics
├─ prediction and answer metrics
└─ scorer, planning, prefill, decode and end-to-end timing
```

### 1.3 建议的核心接口

teacher-forced gold-answer scoring：

```python
score_gold_answer_nll(
    model: Qwen2_5_VLForConditionalGeneration,
    encoded: EncodedSample,
    answer_token_ids: Tensor,
    *,
    reduction: Literal["mean"] = "mean",
) -> AnswerNLL
```

其中 `AnswerNLL` 至少包含 `mean_nll`、`sum_nll`、`num_answer_tokens` 和可选的逐 token NLL。API 必须显式接收答案边界，不能依赖“最后若干 token”之类的隐含假设。

删除变体构造：

```python
build_chunk_deletion_variant(
    encoded: EncodedSample,
    layout: CandidateLayout,
    chunk: Chunk,
    answer_token_ids: Tensor,
) -> TeacherForcedVariant
```

批量标签生成：

```python
generate_utility_labels(
    model: Qwen2_5_VLForConditionalGeneration,
    encoded: EncodedSample,
    layout: CandidateLayout,
    chunks: Sequence[Chunk],
    answer_token_ids: Tensor,
    config: UtilityLabelConfig,
) -> UtilitySampleRecord
```

scorer 保持独立于 Qwen runner：

```python
class JointUtilityScorer(nn.Module):
    def forward(
        self,
        chunk_features: Tensor,       # [G, D]
        question_features: Tensor,    # [G, D_q] or [D_q]
        modality_ids: Tensor,         # [G]
        structural_features: Tensor,  # [G, D_s]
        chunk_mask: Tensor,           # [G]
    ) -> Tensor:                      # [G]
        ...
```

训练与推理共用同一 feature builder：

```python
build_utility_features(
    encoded: EncodedSample,
    layout: CandidateLayout,
    chunks: Sequence[Chunk],
    *,
    aligned_scores: AlignedRelevance | None = None,
) -> UtilityFeatures
```

从预测分数生成计划：

```python
plan_learned_utility_pruning(
    layout: CandidateLayout,
    predicted_utilities: Tensor,
    retention: float,
    config: UtilityPlanningConfig,
    *,
    token_relevance: RelevanceScores | None = None,
) -> PruningPlan
```

上述接口应复用 `joint_pruning.py` 已验证的保护 token、整数预算、序列恢复和 mRoPE gather，不复制一套新的 Qwen 解码路径。

## 2. Algorithm

### 2.1 Group-aware 数据划分与冻结

先确定固定 50 样本 test 的 sample ID，并追溯每个样本的源文档、对话、图像和可用数据集 group 字段。划分单位是 group，而不是问题行。

推荐流程：

```text
load full MMCoQA index
    → normalize source/group identifiers
    → compute image/content duplicate fingerprints
    → reserve every group touching the fixed 50 test samples
    → split remaining groups into train/validation
    → audit exact and near duplicates across splits
    → write and freeze manifests + fingerprints
```

必须检查：

* sample ID 三个 split 两两无交集；
* group ID 三个 split 两两无交集；
* 同一路径、同一内容 hash 的图像不跨 split；
* 可识别的同一对话/文档派生问题不跨 split；
* test 50 的顺序和内容与实验 002 保持一致；
* train/validation 的规模、问题类型、文本长度、图像数和候选数分布被记录。

若数据缺少可靠 group ID，应优先从原始元数据、对话 ID、共享图像集合和文本 fingerprint 重建。无法排除的潜在近重复必须记录为限制，不能退化成纯随机行划分后声称无泄漏。

split manifest 一旦用于生成标签就冻结。改变 split、过滤规则或 test 集都必须产生新的 artifact version，旧 checkpoint 不得静默复用。

### 2.2 Teacher-forced 答案 NLL

对每个训练或验证样本，构造与推理一致的 prompt，并将 gold answer 追加到 assistant 区域。设完整 token 序列为：

$$
[x_1,\ldots,x_L,y_1,\ldots,y_M].
$$

因 causal LM 的 logits 存在一位偏移，标签 mask 应满足：只有预测 $y_k$ 的 logit 位置参与 loss。实现不直接依赖 Transformers 默认的全序列 loss，而应先验证 shift 后显式抽取 answer logits 与 targets：

$$
\ell(X,y)=\frac{1}{M}\sum_{k=1}^{M}
\operatorname{CE}(\operatorname{logit}_{L+k-1}, y_k).
$$

必须明确处理：

* assistant 模板 token 是否属于 prompt，而不是答案；
* EOS 是否计入主标签。主方案只计算标准答案内容 token，EOS 可单独记录为消融；
* tokenizer 自动添加的 BOS/EOS 和 chat template 边界；
* 多 token 数字、Unicode、前导空格和换行；
* padding side 与 batch 内不同序列长度；
* 空答案、无法 tokenization 的异常记录显式失败。

完整输入只计算一次 baseline NLL：

$$
\ell_{full}=\ell(X,y).
$$

该值在所有 chunk utility 中复用，不为每个删除变体重复计算。

### 2.3 Leave-one-chunk-out 删除变体

对每个 chunk $g$，从完整候选序列中删除其 token，但保留：

* system、question、assistant prefix 和 answer token；
* 图像起止边界与必要 placeholder；
* 其他文本和视觉候选的原始相对顺序；
* 所有保留 token 对应的原始 Qwen embedding；
* 从完整输入 gather 的原始 mRoPE position IDs。

删除后的 teacher-forced 输入记作 $X\setminus g$。若一个视觉 chunk 覆盖若干 merger token，只删除这些视觉 token，不重新运行低分辨率视觉编码；若一个文本 chunk 被删除，不应把剩余文本重新 tokenize 成不同 token 序列。

utility 标签为：

$$
u_g=\ell(X\setminus g,y)-\ell_{full}.
$$

同时保留以下诊断值：

```text
full_mean_nll
deleted_mean_nll
utility_delta
answer_token_count
deleted_candidate_count
remaining_sequence_length
modality / chunk_id / original positions
```

平均 NLL 是主口径，用于消除答案长度的直接尺度影响；sum NLL 可记录但不用于主标签。不得对负 utility 直接截断后再存储，否则无法分析干扰 chunk 和数值噪声。

### 2.4 标签生成的批处理与缓存

单样本包含 $G$ 个 chunk 时，理论上需要一次完整前向和 $G$ 次删除前向。推荐执行顺序：

```text
load sample/images
    → Qwen vision encoder/projector once
    → construct full multimodal embeddings and mRoPE once
    → compute full-answer NLL once
    → build all deletion metadata
    → bucket variants by compressed length
    → pad and batch teacher-forced LLM forwards
    → compute utility deltas on CPU FP32
    → atomically write one sample record
```

视觉 encoder/projector 输出、文本 embedding、完整 position IDs、问题特征和 chunk 特征必须在样本内复用。删除变体只重组已有 embedding 与位置，不重复进行图像编码。

不同删除变体长度可能不同。batch collator 必须同时构造：

* padded `inputs_embeds`；
* attention mask；
* 兼容 Qwen mRoPE 形状的 position IDs；
* gold answer target 和精确 answer mask；
* 每条变体到 chunk ID 的映射。

优先按序列长度 bucketing，避免最长变体支配所有 padding。batch 大小由 smoke test 的显存 profile 决定，不在代码中硬编码。

每个样本单独原子写入，cache key 至少包含：

* sample content fingerprint；
* teacher model、processor 与 tokenizer revision；
* prompt/template fingerprint；
* chunking config；
* feature schema；
* dtype 和 NLL/EOS 口径；
* 相关代码版本。

断点续跑时只跳过 fingerprint 完整匹配且通过 schema 校验的 record；损坏或旧版本产物不得静默当作成功。

### 2.5 Scorer 输入特征

对每个 chunk 构造：

$$
z_g=[p_g,q,p_g\odot q,|p_g-q|,d_g,m_g,c_g,r_g].
$$

首版各部分定义如下：

* $p_g$：chunk 内冻结 Qwen 输入特征的 mean pooling；
* $q$：question token 的 mean pooling；
* $p_g\odot q$ 与 $|p_g-q|$：低成本显式交互；
* $d_g$：与同模态其他 chunk 的最大和平均 cosine 冗余度；
* $m_g$：可学习的 text/vision modality embedding；
* $c_g$：结构特征，包括 chunk token 数、归一化顺序、图像 ID、归一化空间中心/宽高及总 chunk 数；
* $r_g$：可选 aligned relevance，首版默认缺失并带显式 feature-present mask。

Qwen hidden dimension 较大，应先通过分模态小型 projection 降维：

$$
\tilde p_g=P_{m_g}(p_g),\qquad \tilde q=P_q(q).
$$

文本与视觉可以使用不同输入 projection，以适应两类原始特征分布；投影后的 feature、共享交互层和最终 utility head 必须让两种模态共同训练，不能变成完全独立的两个 scorer 后直接比较未校准输出。

主结构保持轻量：

```text
Qwen pooled feature
    → modality-specific projection
    → concatenate question/interactions/structure/modality
    → shared LayerNorm + 2-layer MLP + GELU + dropout
    → shared scalar utility head
```

特征标准化参数只能由 train split 拟合并写入 checkpoint。validation/test 只应用冻结统计量。modality、position、length-only 弱基线使用相同 split 和训练口径，用于检查 scorer 是否只学到捷径。

### 2.6 Pairwise ranking 与回归目标

剪枝主要依赖样本内排序，因此主 loss 使用样本内 pairwise logistic ranking。对有效 pair $(i,j)$ 且 $u_i>u_j$：

$$
\mathcal L_{rank}(i,j)=
\log\left(1+\exp[-(\hat u_i-\hat u_j)]\right).
$$

当 $|u_i-u_j|<\delta_{pair}$ 时，该 pair 可能主要是数值噪声，首版从 ranking loss 中忽略；$\delta_{pair}$ 只能根据非 test 的重复前向误差或 validation 设定。

为保留 utility 的符号和相对幅度，对 train label 做稳健归一化后加入 Huber loss：

$$
\tilde u_g=\frac{u_g-\operatorname{median}_{train}}
{\operatorname{IQR}_{train}+\epsilon},
$$

$$
\mathcal L=\mathcal L_{rank}+\lambda\mathcal L_{Huber}(\hat u_g,\tilde u_g).
$$

主实现需满足：

* pair 只在同一样本内构造；
* text-text、vision-vision、text-vision 三类 pair 均有覆盖；
* 跨模态 pair 不得因数量少而长期缺席；
* 一个 chunk 数很多的样本不能凭 pair 数量平方级主导训练；
* 每个样本的 loss 先归一化，再在 batch 内平均；
* 记录有效 pair 数、忽略近零 pair 数和三类 pair 比例。

pair 较多时采用分层采样：每个样本分别从三类 pair 中采固定上限，并优先覆盖 label gap 较大的 pair；validation 指标则尽可能在所有有效 pair 上计算，避免采样造成指标波动。

### 2.7 训练、模型选择与 checkpoint 冻结

scorer 训练不加载 7B teacher，只读取离线 feature/label records。每个训练 seed 使用相同 split 和特征配置，但独立初始化、样本顺序和 pair sampling RNG。

推荐训练循环：

```text
load and schema-check train/validation records
    → fit train-only feature/label normalization
    → train scorer
    → evaluate validation ranking/regression metrics each epoch
    → early stop on preregistered validation ranking metric
    → save best checkpoint + optimizer-independent artifact
    → run final validation diagnostics
    → freeze checkpoint before test
```

主模型选择指标建议使用 validation 的宏平均 pairwise ranking accuracy，并将跨模态 pairwise accuracy 作为门槛或并列诊断。不能只按总体 MSE 选模型，因为总体指标可能被单一模态或长样本主导。

每个 seed 必须记录：

* train/validation loss 曲线；
* Spearman、Kendall、pairwise accuracy 和 top-k utility recall；
* 总体、文本、视觉和跨模态分项；
* 最佳 epoch、early-stopping 依据和 checkpoint hash；
* modality-only、position-only、length-only 弱基线结果。

任何结构、$\lambda$、dropout、隐藏维度、$\delta_{pair}$ 或 density 参数只能依据 train/validation 决定。查看固定 50 test 的答案 Accuracy 后不得回头挑 checkpoint。

### 2.8 从预测 Utility 到严格联合预算

scorer 为每个 chunk 输出 $\hat u_g$。由于 chunk 长度不同，主 planner 使用 utility density：

$$
D_g=\frac{\max(\hat u_g,0)}{|g|^\gamma}.
$$

$\gamma$ 在 validation 上从少量预注册候选中选择；若不做 sweep，首版使用 $\gamma=1$，表示单位 token 的预测 utility。必须同时保留不除长度的 $\gamma=0$ 作为解释性消融，而不能在不同 test 样本间根据结果切换。

规划过程：

1. 计算所有文本和视觉 chunk 的 $D_g$；
2. 在同一列表中按 density 排序，不先做分模态 softmax；
3. 在容量 $B$ 下依次分配 chunk；
4. 若完整 chunk 超过剩余容量，对该 chunk 做部分保留；
5. 使用已有整数再分配逻辑精确填满 $B$；
6. 恢复原序列顺序并 gather 原始 mRoPE position IDs。

chunk 内 token 选择的首版规则不依赖实验 003：

* 文本 chunk：优先均匀覆盖并保持首尾，或复用 raw Qwen token relevance；
* 视觉 chunk：空间均匀保留，或复用 raw Qwen token relevance；
* 若后续已有可靠 aligned relevance，再作为独立消融替换 chunk 内选择。

需要避免所有预测 utility 为负时 planner 无法排序。主方案的 `max(.,0)` 只用于 density；若所有 density 都为 0，则使用预测 utility 的原始排序和确定性 tie-break 填满预算。原始 $\hat u_g$ 始终保留用于分析。

实验 002 的最低模态预算可以作为 safety constraint 消融，但主条件不启用。否则无法判断 scorer 是否真的学会跨模态竞争。若启用，必须报告被约束改变计划的样本数和 token 数。

### 2.9 Oracle Utility 与交互扩展

Oracle Utility 仅在少量样本或已有真实 leave-one-out label 的范围内运行，用真实 $u_g$ 替换 $\hat u_g$，其作用是估计当前标签与 planner 的上限，而不是可部署方法。

若 oracle 在 75%/50% 也不优于 random，应优先检查：

* utility 符号或 answer mask 是否错误；
* leave-one-out 的“其他 chunk 全存在”边际价值是否不适配目标预算；
* chunk utility 与 token 长度/density 转换是否错误；
* chunk 内部分保留是否破坏 oracle 优势。

只有基础 label 正确、可学且 oracle 在温和预算有效，但 50%/25% 明显失效时，才扩展条件边际标签。对采样子集 $S$：

$$
u(g\mid S)=\ell(S\setminus g,y)-\ell(S,y).
$$

子集采样需显式记录保留率、随机种子和包含的 chunk，并保证 $g\in S$。首版不实现完整 Shapley 枚举；Monte Carlo 扩展属于第二阶段，不应阻塞基础 leave-one-out 验证。

### 2.10 最终推理与主比较

对固定 test 样本，先用冻结 Qwen 构造与训练一致的 feature，再加载每个训练 seed 的 scorer 预测 utility。推理绝不需要 gold answer 或 leave-one-out teacher scoring。

主比较条件：

```text
No Pruning
Random Joint × 3 selection seeds
Fixed Joint
Raw Qwen Joint
Learned Utility Joint × at least 3 training seeds
Oracle Utility on a small declared subset
```

实验 003 未运行时，`Aligned Encoder Joint` 标记为未获得结果，不构成启动实验 004 的阻塞。若以后补跑实验 003，应在相同 test、预算和生成配置下追加该对照，不改动已经冻结的实验 004 checkpoint。

75% 和 50% 是主预算，25% 是压力测试。所有 joint 条件严格共享相同 $B$。相同 sample、retention、scorer seed 的 utility 只计算一次，并在可复用的规划/诊断中缓存。

## 3. 文件分工

建议增量复用实验 002/003 已有的数据、Qwen、budget 和评测代码，不将标签生成、scorer 训练和最终大模型推理塞进同一个脚本。

| 文件 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| `src/mllm_prune/data/splits.py` | group-aware 划分、fingerprint 和泄漏检查 | MMCoQA 索引、固定 test manifest | `SplitArtifact` |
| `src/mllm_prune/models/qwen25vl.py` | 增加 gold-answer teacher forcing、答案边界与批量 deletion variant 前向 | prompt、answer、缓存 embedding/mRoPE | answer NLL、冻结特征 |
| `src/mllm_prune/algorithms/utility_labels.py` | 删除变体、utility delta、label record 与 schema | encoded sample、chunks、answer | `UtilitySampleRecord` |
| `src/mllm_prune/algorithms/utility_features.py` | chunk/question pooling、冗余度和结构特征 | Qwen features、layout | `UtilityFeatures` |
| `src/mllm_prune/models/utility_scorer.py` | modality projection、共享 MLP 和 scalar head | `UtilityFeatures` | predicted utility |
| `src/mllm_prune/algorithms/utility_training.py` | pair 构造、ranking/Huber loss、指标与 checkpoint | train/validation records | `ScorerArtifact` |
| `src/mllm_prune/algorithms/joint_pruning.py` | utility density、容量选择、部分 chunk 和严格预算 | predicted utility、layout、retention | `PruningPlan` |
| `scripts/prepare_exp004_splits.py` | 固化 train/validation/test manifests 与泄漏报告 | 数据索引、test 50 | manifests、audit report |
| `scripts/generate_utility_labels.py` | 多卡样本分片、断点续跑、原子写 label/features | config、manifest、Qwen | label/feature shards |
| `scripts/train_utility_scorer.py` | 单卡/CPU 小模型训练和 validation 选模 | records、seed、config | scorer checkpoints、curves |
| `scripts/run_learned_utility_pruning.py` | test 特征、scorer 推理、联合剪枝和生成 | Qwen、checkpoint、test manifest | prediction JSONL shards |
| `scripts/summarize_learned_utility.py` | 标签、scorer、下游、配对与性能汇总 | records、checkpoints、predictions | JSON/CSV/Markdown summary |
| `configs/experiments/004_learned_utility_scorer.yaml` | split、标签、特征、模型、训练、预算和运行参数 | 人工配置 | 全流程 config |
| `tests/test_utility_labels.py` | answer mask、shift、delta 符号和删除 invariant | fake logits/小模型 | 精确 NLL/utility 断言 |
| `tests/test_utility_features.py` | pooling、冗余度、结构特征与 schema | synthetic chunks | shape/value 断言 |
| `tests/test_utility_training.py` | pair sampling、loss、mask、指标和 checkpoint | synthetic records | 排名与训练断言 |
| `tests/test_exp004_pipeline.py` | record→scorer→plan 的集成回归 | fixture/fake scorer | 严格预算与序列断言 |

建议配置至少包含：

```yaml
experiment: 004_learned_utility_scorer
split:
  manifest_path_env: EXPERIMENT_004_SPLIT_MANIFEST
  fixed_test_manifest_env: MMCOQA_TEST_SUBSET_PATH
  seed: 2026
teacher:
  model_path_env: QWEN_MODEL_PATH
  dtype: bfloat16
  answer_loss_reduction: mean
  include_eos_in_label: false
label_generation:
  variant_batch_size: 8
  length_bucket: true
  atomic_per_sample: true
vision_window: 4
features:
  chunk_pooling: mean
  question_pooling: mean
  projection_dim: 256
  include_redundancy: true
  include_aligned_relevance: false
scorer:
  hidden_dims: [256, 128]
  dropout: 0.1
training:
  seeds: [0, 1, 2]
  pair_gap_threshold: ...
  max_pairs_per_type_per_sample: ...
  huber_weight: ...
  early_stopping_patience: ...
planning:
  density_gamma: 1.0
  minimum_modality_budget: false
retentions: [0.75, 0.50, 0.25]
random_selection_seeds: [0, 1, 2]
```

路径放入 `.env`；会影响实验定义、标签或训练结果的参数放入 YAML。首轮 GPU smoke 后固定使用 `variant_batch_size=8` 和 $4\times4$ merger-token 视觉 utility chunk；若正式运行前因硬件变化调整，必须产生新的配置与 record fingerprint，不能静默复用旧标签。

## 4. Verification

### 4.1 Split 与泄漏检查

必须自动验证：

* train/validation/test sample ID 无交集；
* group ID 无交集；
* test manifest 与固定 50 样本 fingerprint 完全一致；
* exact image/content hash 不跨 split；
* 同一源文档、对话或共享图像集合不跨 split；
* 重复运行相同 seed 得到相同 manifest；
* 修改源数据或 test manifest 后旧 artifact 被拒绝。

泄漏检查报告应列出 group 数、样本数、被合并/排除的重复项和无法判定的风险项，而不是只输出一个布尔值。

### 4.2 Gold-answer NLL 正确性

使用人工 logits 和短 token 序列精确验证 causal shift 与 mask：

* 只有 gold answer 内容 token 参与 loss；
* prompt、assistant prefix、padding 和视觉 placeholder 全部被排除；
* batch padding 与逐条前向的 NLL 一致；
* answer token 数与 tokenizer 结果一致；
* mean NLL 乘 token 数能复原 sum NLL；
* 相同输入重复前向在允许误差内一致；
* BF16 与 FP32 的 utility 排名在小样本上稳定；
* 空答案、全 mask 和 NaN/Inf 显式失败。

必须用 3–5 个非 test 真实样本人工查看 prompt/answer 边界及逐 token loss。只看 Transformers 返回的 scalar loss 不足以证明 mask 正确。

### 4.3 删除变体与 mRoPE invariant

对文本和视觉 chunk 分别检查：

* 删除 token 数等于 chunk capacity；
* 保护 token 和 answer token 完整保留；
* 其他 token embedding 与完整输入逐值一致；
* 剩余 token 相对顺序不变；
* position IDs 是完整输入对应位置的 gather；
* 多图 token 不串图，图像边界 token 不被误删；
* batch 与逐变体前向 NLL 在容差内一致；
* 完整输入作为“删除空集”时 NLL 完全回归。

手工构造唯一证据 chunk 和无关 chunk：删除唯一证据后 utility 应显著更高。该测试用于验证符号和数据流，不把单个案例当作模型普遍有效的证据。

### 4.4 Label 数值与统计门槛

正式全量生成前，在少量训练样本上报告：

* utility min/max/mean/std 和分位数；
* 正、近零、负 label 比例；
* 文本/视觉分项分布；
* utility 与 chunk 长度、位置、模态的相关性；
* 重复前向的 delta 误差；
* 每个样本 chunk 数、单样本耗时和峰值显存；
* 不同 variant batch size 的吞吐与数值一致性。

若 utility 的主要尺度与重复数值误差相当，先修正精度/口径或提高 pair gap threshold，不启动全量训练。若几乎所有 label 完全相同，也应先检查 answer mask、删除是否真正生效及 teacher 对 gold answer 的置信度。

### 4.5 Feature 与 Scorer 单元验证

需要验证：

* 每个 chunk 恰有一行 feature，顺序和 chunk ID 稳定；
* text/vision projection 输入维度和 mask 正确；
* mean pooling 不包含其他 chunk 或 padding；
* redundancy 不将 chunk 自身计入最大相似度；
* 单 chunk 模态的冗余度有明确默认值和 valid flag；
* train normalization 不读取 validation/test；
* save/load 后同一输入预测逐值一致；
* artifact 的 feature schema、split 和 teacher fingerprint 不匹配时拒绝加载。

用合成数据训练一个极小 scorer，确认 label 可分时 ranking loss 下降、预测顺序正确；标签全部相同时不应产生非法 pair 或 NaN。

### 4.6 Scorer 评价

validation 至少报告：

* Spearman 和 Kendall；
* pairwise ranking accuracy；
* text-text、vision-vision、text-vision pairwise accuracy；
* top-k utility recall；
* Huber/MAE 等回归诊断；
* 逐样本宏平均和全 chunk 微平均。

总体指标高但跨模态 pair 接近随机时，不能声称统一 utility 标尺已经学到。三个训练 seed 应分别报告，不只保留最佳 seed。

还需比较 full-feature scorer 与 modality-only、position-only、length-only 弱基线。若弱基线解释了大部分 validation 表现，应在进入最终 test 前检查数据偏置和特征泄漏。

### 4.7 联合剪枝回归

继续满足实验 002 invariant：

* chunk 完整覆盖候选 token 且互不重叠；
* `kept_text + kept_vision == B`；
* 每个 chunk 分配不超过容量；
* 保护 token 永不删除；
* 压缩序列长度等于 `protected_count + B`；
* 保留视觉 token 的图像 ID 和 mRoPE 坐标正确；
* 相同输入、checkpoint、配置和 seed 得到相同计划；
* $\rho=1$ 时 keep mask、生成结果和 No Pruning 一致。

特别测试所有 utility 为负、全为零、存在 tie、一个 chunk 大于总预算、某模态只有一个 chunk和某模态无候选等边界情况。

### 4.8 下游评测与性能验证

任务结果同时保留 MileBench/MMCoQA 原口径和 robust Accuracy，并保存逐样本 prediction、answer、两种判定、保留计划与 utility。

主比较在相同 test、预算和生成配置下报告：

* Accuracy 与 robust Accuracy；
* 相对 No Pruning、Random、Fixed、Raw Qwen 的配对变化；
* 正确→错误、错误→正确与持平数量；
* 置信区间和适合 50 个配对样本的精确检验；
* Random selection seed 与 scorer training seed 的均值、标准差和逐 seed 结果；
* 文本/视觉保留比例和严格预算检查。

性能计时至少拆分：

```text
feature extraction / pooling
utility scorer forward
utility planning
Qwen vision encoding
LLM prefill
decode
end-to-end
```

标签生成另行拆分视觉编码、完整 NLL、删除变体构造、batched variant forwards 和写盘。GPU 计时需要预热和 synchronize。

部署主口径中，Learned Utility 必须计入在线 feature/scorer/planner 开销；一次性的 label 生成与 scorer 训练成本单独报告，不混入单次在线推理延迟。

## 5. Risk 与 fallback

| 风险 | 表现 | 优先处理 | Fallback |
| --- | --- | --- | --- |
| answer mask/causal shift 错误 | 所有 label 异常或符号相反 | 人工 logits 与逐 token NLL 检查 | 暂停标签生成，修正后使旧 cache 失效 |
| split 近重复泄漏 | validation 很高、test 不泛化 | group/image/content fingerprint 审计 | 合并可疑 group，重新划分和训练 |
| 删除变体位置错误 | NLL 波动大、视觉前向异常 | 检查 embedding、顺序和 mRoPE gather | 复用已验证单条件剪枝重建函数 |
| BF16 delta 噪声接近 utility | 大量 pair 顺序不稳定 | 重复前向、FP32 loss 计算、设 pair gap | 标签关键段提高精度或忽略近零 pair |
| leave-one-out 忽略互补/冗余 | 75% 有效、50%/25% 失效 | 先看 oracle 与预算分层结果 | 采样 conditional marginal / Monte Carlo Shapley |
| gold NLL 与自由生成 Accuracy 不一致 | scorer 指标好但答案无提升 | 配对检查 NLL 与生成变化 | 加 sequence-level preference 辅助信号，保留 NLL 主口径 |
| scorer 学到模态/位置/长度捷径 | 弱基线接近 full model | 弱基线和 feature ablation | 降低结构特征、重采样或增加数据 |
| 跨模态 pair 太少 | 总体排名好但跨模态随机 | 分层 pair sampling | 对跨模态 pair 加权，仍按样本归一化 |
| chunk 数导致 pair 爆炸 | 内存和训练时间平方增长 | 每样本每类型有界采样 | listwise/soft rank 作为后续替代 |
| label 生成过慢 | GPU-hours 超预算 | 缓存视觉特征、batch variants、多卡样本并行 | 先采部分 chunk/样本验证可学性 |
| variant batch OOM | 单样本长序列失败 | length bucket、减 batch、记录峰值 | 单变体前向断点续跑，不静默 CPU 全量 |
| scorer 过拟合 | train 高、validation 低且 seed 波动大 | 小模型、dropout、early stopping | 减特征/容量，增加 group-disjoint 训练数据 |
| 负 utility 被 planner 丢失 | 全零 density 或计划退化 | 明确 fallback 排序与 tie-break | validation 比较 signed/shifted density |
| 实验 003 未运行 | 缺少 aligned 对照/特征 | 将其标记为未获得，不阻塞 004 | 后续同口径补跑，不重选 004 checkpoint |

其中增加 conditional marginal、sequence-level 标签或其他 benchmark 都会改变训练监督或数据范围，必须作为后续阶段单独版本化，不能在看到 test 结果后悄悄并入主方法。

## 6. 时间与显存分析

### 6.1 成本分解

对样本 $s$，设 chunk 数为 $G_s$。基础 leave-one-out 需要：

$$
1+G_s
$$

次 teacher-forced LLM scoring，但视觉 encoder/projector 只需运行一次。若每批平均容纳 $K_s$ 个 deletion variants，则 LLM forward 批次数近似为：

$$
1+\left\lceil\frac{G_s}{K_s}\right\rceil.
$$

这不代表计算量减少到原来的 $1/K_s$；batch 主要提高 GPU 并行度并减少调度开销，总 token FLOPs 仍与所有变体长度之和相关。

总标签时间可用 smoke 实测外推：

$$
T_{label}\approx
\frac{1}{N_{gpu}}
\sum_s
\left(T_{vision,s}+T_{full,s}+T_{variants,s}+T_{io,s}\right).
$$

正式估算必须基于 chunk 数、序列长度、图像数有代表性的样本，不能只测最短样本。

### 6.2 首轮正确性 smoke

第一轮只使用 3–5 个非 test 的训练样本，并包含：

* 至少一个文本 chunk 较多样本；
* 至少一个视觉 chunk 较多或多图样本；
* 至少一个答案证据可以人工定位的样本；
* 完整 NLL 和每个 chunk 的 leave-one-out utility；
* batch size 1 与小批量变体的数值对照；
* 重复前向稳定性、峰值显存和分阶段计时。

该 smoke 只验证标签链路，不解读 scorer 或最终 Accuracy。五项标签正确性门槛通过后，才扩大标签生成。

### 6.3 可学性 pilot

在全量标签前，先对一个明确记录的小型 train/validation 子集生成完整 label，并训练最小 scorer。pilot 的目标是回答：

* label 是否有非平凡分布；
* validation 排名是否高于随机；
* 跨模态 pair 是否存在且可学；
* oracle utility 在少量 held-out 样本上是否优于 random；
* 单样本和单 chunk 的实际成本是多少。

只有 pilot 表明标签正确且至少存在可学习信号，才值得扩大全量 label。pilot 样本不能进入固定 test，且不得因为 pilot 指标不好反复搜索大量结构超参。

### 6.4 多卡标签生成

utility label 生成按样本独立，优先采用数据并行：每张 GPU 加载一个冻结 Qwen，领取互斥 sample shard，输出独立目录。无需 DDP 或梯度通信。

多卡 runner 必须保证：

* manifest 分片无重叠、无遗漏；
* 每个 sample ID 只有一个最终成功 record；
* 单样本失败不会让其他 shard 被标记完成；
* 统计 completed/skipped/failed 及失败原因；
* 中断后只重跑缺失或 fingerprint 不匹配样本；
* 不同时写同一个 JSONL/临时文件。

正式启动前，根据 smoke 报告每卡峰值显存、推荐 batch、样本吞吐、预计 GPU-hours 和存储量，并按项目规范取得长耗时任务确认。

### 6.5 Scorer 训练与最终推理成本

特征提取完成后，scorer 训练不加载 Qwen 7B。对于几百至几千个样本、每样本几十个 chunk 的规模，小型 MLP 通常可在单 GPU 甚至 CPU 上训练；但具体时间和内存以实际 record 数、特征维度和 pair sampling 为准，不预先承诺固定耗时。

在线 scorer 成本包括：

* 从已存在 Qwen embedding 做 pooling；
* 计算 redundancy/structural feature；
* 一次小型 MLP forward；
* utility density 排序和整数规划。

这些成本应远小于重复 teacher scoring；最终推理绝不能在线生成 leave-one-out label。是否获得端到端加速仍需把 scorer 开销与剪枝后的 prefill/decode 节省一起实测。

### 6.6 显存与存储

标签生成的峰值显存主要由以下部分组成：

```text
Qwen BF16 weights
cached vision/text input embeddings
batched deletion-variant inputs
LLM activations/logits for teacher forcing
temporary attention workspace
```

虽然使用 inference mode 不保存反向图，但 batched 长序列 logits 仍可能很大。实现应只保留 answer 位置所需的 loss/诊断，及时释放完整 logits 引用，并通过 length bucket 和 batch profile 控制峰值。

存储每个 chunk 的完整 3584 维或同量级 FP32 feature 可能迅速膨胀。首版建议：

* label 与元数据使用 JSONL/Parquet 等可检查格式；
* 高维 feature 使用分片 tensor 文件，label record 保存索引和 fingerprint；
* 在确认投影方式后，可保存 FP16/BF16 pooled feature并在训练时转 FP32；
* 不保存所有 deletion variant 的完整 embedding 或 logits；
* checkpoint、全量 feature 和运行日志放仓库外 runs 路径。

正式生成前按 pilot 的平均 chunk 数和 feature 大小估算：

$$
S_{feature}\approx
N_{chunk}\times D\times \text{bytes per value}
+S_{metadata}.
$$

### 6.7 推荐的实现与优化顺序

1. **固化 split 与泄漏审计**  
   先确定训练、验证和固定 test 的边界，避免后续标签和 checkpoint 作废。
2. **实现并验证 gold-answer NLL**  
   用人工 logits、短序列和真实样本确认 causal shift 与 answer mask。
3. **复用删除与 mRoPE 路径**  
   先做单 chunk、batch size 1，确保 utility 符号和序列重建正确。
4. **样本内缓存视觉与输入特征**  
   避免每个删除变体重复 vision encoder/projector。
5. **按长度 batch deletion variants**  
   从小 batch profile，验证与逐条前向数值一致后再扩大。
6. **原子分片与断点续跑**  
   在多卡前先保证单卡中断可恢复、旧 cache 可检测。
7. **训练最小共享 MLP**  
   先证明 validation 排名信号，再加入 redundancy 或更复杂交互。
8. **多卡样本并行生成全量 label**  
   仅在正确性、成本估算和资源确认后启动。
9. **锁定 3 个 seed checkpoint 后运行固定 test**  
   test 只用于最终评价，不反向调整特征、loss 或 checkpoint。
10. **最后评估 conditional marginal 扩展**  
    只有基础方法在高压缩率暴露明确交互问题时再增加标签成本。

实验 004 的首个实现里程碑不是“训练脚本能运行”，而是：split 无泄漏、gold-answer NLL mask 正确、删除变体保持原始特征与 mRoPE、utility 重复稳定，并且小型 validation 上的跨模态排序高于随机。满足这些条件后，正式 learned utility 剪枝结果才具备可解释性。
