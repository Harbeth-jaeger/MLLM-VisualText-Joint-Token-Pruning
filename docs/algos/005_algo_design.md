# 005 · Aligned Prefilter + Conditional Utility Scorer 算法设计

> 对应实验设计：`docs/experiments/005_learned_utility_scorer_v2.md`
>
> 状态：算法设计完成，尚未实现。本文件描述代码边界、接口、不变量、验证方式、风险与资源估计；研究动机、对照组和结果表以实验设计文档为准。

## 1. Problem

### 1.1 要解决的问题

实验 005 要把实验 003 的 aligned relevance 与实验 004 的 learned utility 串成一个可训练、可部署、可严格复现的两阶段 pruning 系统：

1. 用冻结的图文对齐 encoder 给文本和视觉候选 token 打分，并用 chunk 内 top-$k$ similarity 聚合得到 chunk relevance；
2. 用 relevance 先删除总删除量的一部分，得到条件集合 $S$；
3. 在 train/validation 上计算：
   
   $$
   u(g\mid S)=\ell(S\setminus g,y)-\ell(S,y),
   $$
   
   作为第二版 scorer 的监督标签；
4. 训练 scorer 根据 $S$ 中的 Qwen 特征、aligned similarity、冗余、结构与预算状态预测：
   
   $$
   \hat u(g\mid S);
   $$
5. 在线推理时，先做相同初筛，再按原始 $\hat u(g\mid S)$ 对 chunk 排序；只有边界 chunk 在内部按 aligned token similarity 选 token；
6. 两次 pruning 后严格保留：
   
   $$
   B=\lfloor\rho(N_t+N_v)\rfloor
   $$
   
   个候选 token，并复用 Qwen 原始 embedding 与 mRoPE 正常生成答案。

本算法不重新实现 Qwen 编码、候选布局、teacher-forced NLL 或压缩生成，而是复用实验 002–004 已验证的公共路径。新增代码只负责：aligned 缓存、初筛状态、conditional feature/label、v2 scorer、第二阶段 planner 和实验入口。

### 1.2 核心不变量

实现必须始终满足：

1. **无测试泄漏**：test 不参与 aligned calibration、scorer 训练、归一化、early stopping 或超参数选择；
2. **无答案泄漏**：gold answer 只进入 train/validation 的 conditional label generation；scorer 输入和 test 推理不包含 answer/NLL/utility label；
3. **状态一致**：同一 $(sample,\rho,\alpha)$ 的训练标签和在线推理必须使用相同 prefilter 规则得到 $S$；
4. **索引稳定**：初筛和细筛全程保存相对原始 CandidateLayout 的 token 索引，不在删减后重新编号并丢失映射；
5. **严格预算**：最终选择数必须恰好为 $B$，即 `budget_deviation == 0`；
6. **两种模态共用总预算**：不预先固定 Text/Vision 比例；
7. **不改 Qwen 表示**：选择使用 external score，但送入 Qwen 的仍是原始 feature 和原始 mRoPE；
8. **v1 可复现**：不破坏实验 004 的 record、checkpoint、入口和测试；v2 使用新的 artifact version 与独立输出目录；
9. **缓存可验证**：aligned cache 必须绑定 sample、layout、encoder、calibrator 和 scoring config 的 fingerprint；不兼容时直接报错；
10. **可断点续跑**：长流程以样本为原子记录，已有且 fingerprint 一致的记录跳过，不一致的旧记录拒绝静默覆盖。

### 1.3 总体 I/O

#### 1.3.1 离线训练流水线输入

```text
Exp005TrainingInput
├─ experiment 004 frozen train/validation/test manifests
├─ MMCoQA text, images, question and train/validation gold answer
├─ frozen Qwen2.5-VL model + processor
├─ frozen aligned encoder + processor
├─ frozen experiment 003 calibration artifact
├─ experiment 005 YAML config
└─ code/model/data fingerprints
```

#### 1.3.2 中间产物

```text
AlignedScoreCache                      # 每个样本只计算一次
├─ raw/calibrated text token scores
├─ raw/calibrated vision token scores
├─ chunk/layout metadata
├─ encoder/calibrator/scoring fingerprints
└─ timings and numerical diagnostics

ConditionalUtilityRecord              # 每个 sample × state 一条
├─ state_id = (retention, prefilter_fraction)
├─ prefilter token mask and surviving chunks
├─ conditional baseline NLL l(S, y)
├─ deleted NLLs and u(g | S)
├─ v2 scorer features
├─ original↔state index mappings
├─ timing/memory diagnostics
└─ all dependency fingerprints
```

#### 1.3.3 训练输出

```text
UtilityScorerV2Artifact
├─ format_version = 2
├─ feature schema and dimensions
├─ model state_dict
├─ feature/label normalization statistics
├─ supported budget-state encoding
├─ train/validation manifest fingerprints
├─ aligned cache/config fingerprints
├─ seed, best epoch and validation metrics
└─ training history
```

#### 1.3.4 在线推理输入与输出

```text
Exp005InferenceInput
├─ question, context and images
├─ final retention rho and prefilter fraction alpha
├─ frozen Qwen
├─ frozen aligned encoder + calibrator
├─ frozen scorer-v2 checkpoint
└─ experiment 005 config

Exp005InferenceOutput
├─ aligned token/chunk scores
├─ prefilter state and removed indices
├─ predicted conditional utility per surviving chunk
├─ final strict-budget PruningPlan
├─ generated answer
├─ correctness metrics when gold is available for evaluation
└─ stage-wise timings and peak memory
```

### 1.4 建议的公共接口

#### 1.4.1 Aligned cache

放在：`src/mllm_prune/algorithms/aligned_scoring.py`

```python
@dataclass(frozen=True)
class AlignedScoreArtifact:
    sample_id: int
    text_scores: Tensor              # [N_t], calibrated
    vision_scores: Tensor            # [N_v], calibrated
    raw_text_scores: Tensor | None
    raw_vision_scores: Tensor | None
    layout_fingerprint: str
    scoring_fingerprint: str
    timing: dict[str, float]


def save_aligned_score_artifact(path: Path, artifact: AlignedScoreArtifact) -> None: ...


def load_aligned_score_artifact(
    path: Path,
    *,
    expected_layout: CandidateLayout,
    expected_fingerprint: str,
) -> AlignedScoreArtifact: ...
```

缓存保存 calibrated token scores，因为初筛、scorer feature 与边界 token 选择都应使用同一套可比较分数。raw score 可选保存，供诊断使用，但不能在主流程中混用 raw/calibrated score。

#### 1.4.2 Prefilter state

新增：`src/mllm_prune/algorithms/cascaded_utility.py`

```python
@dataclass(frozen=True)
class ConditionalPruningState:
    retention: float
    prefilter_fraction: float
    original_budget: int
    prefilter_keep_budget: int
    final_budget: int
    selected_text_indices: tuple[int, ...]      # 原始候选索引
    selected_vision_indices: tuple[int, ...]
    text_chunk_masks: tuple[tuple[int, ...], ...]
    vision_chunk_masks: tuple[tuple[int, ...], ...]
    surviving_chunk_ids: tuple[int, ...]         # 全局 chunk 顺序
    partial_chunk_ids: tuple[int, ...]
    state_layout: CandidateLayout

    def validate(self, original_layout: CandidateLayout) -> None: ...


def build_aligned_prefilter_state(
    layout: CandidateLayout,
    aligned_relevance: RelevanceScores,
    *,
    retention: float,
    prefilter_fraction: float,
) -> ConditionalPruningState: ...
```

预算使用整数定义，避免浮点累计误差：

```python
final_budget = floor(retention * total_capacity)
total_delete = total_capacity - final_budget
prefilter_delete = round_or_floor(prefilter_fraction * total_delete)
prefilter_keep_budget = total_capacity - prefilter_delete
```

具体使用 `floor` 还是“最近整数”必须在配置中冻结。建议使用 `floor` 删除量，使初筛更保守；所有地方只能调用同一个预算函数。

`state_layout` 只包含初筛后仍至少保留一个 token 的 chunk。若边界 chunk 被部分保留，其 `token_indices` 仍引用原始模态候选索引。`ConditionalPruningState` 负责验证这些索引不重复、不越界，并与 selected indices 完全一致。

#### 1.4.3 V2 feature

新增：`src/mllm_prune/algorithms/conditional_utility.py`

```python
@dataclass(frozen=True)
class ConditionalUtilityFeatures:
    chunk_features: Tensor             # [G_S, D_qwen]
    question_feature: Tensor           # [D_qwen]
    modality_ids: Tensor               # [G_S]
    structural_features: Tensor        # [G_S, D_struct]
    aligned_features: Tensor           # [G_S, D_align]
    budget_features: Tensor            # [G_S, D_budget]
    original_chunk_ids: Tensor         # [G_S]
    chunk_capacities: tuple[int, ...]

    def validate(self, state: ConditionalPruningState) -> None: ...


def build_conditional_utility_features(
    original_layout: CandidateLayout,
    state: ConditionalPruningState,
    text_features: Tensor,
    vision_features: Tensor,
    question_features: Tensor,
    aligned_scores: AlignedScoreArtifact,
) -> ConditionalUtilityFeatures: ...
```

特征构造必须基于初筛后真实 token mask：

- chunk mean 只平均仍在 $S$ 中的 token；
- aligned max/mean/std/top-$k$ 只统计仍在 $S$ 中的 token；
- chunk capacity 使用实际剩余数；
- redundancy 只在 $S$ 的同模态 chunk 中重算；
- budget feature 对每个 chunk 广播相同的 $[\rho,\alpha,|S|/N,N_t^S/|S|,N_v^S/|S|]$；
- `original_chunk_ids` 保证 label、feature、prediction 和最终 plan 使用同一顺序。

#### 1.4.4 Conditional label

继续复用 `models/qwen25vl.py::score_gold_answer_variants`，但新增 state 到 deletion spec 的适配函数：

```python
def conditional_deletion_specs(
    original_layout: CandidateLayout,
    state: ConditionalPruningState,
) -> list[ConditionalDeletionSpec]: ...


def conditional_utility_from_nll(
    state_nll: float,
    deleted_nlls: Sequence[float],
) -> Tensor: ...
```

Qwen 侧需要支持一个基础保留 mask 加一个待删除 chunk mask：

```python
@dataclass(frozen=True)
class CandidateSelection:
    selected_text_indices: tuple[int, ...]
    selected_vision_indices: tuple[int, ...]


def score_gold_answer_selections(
    model,
    encoded: EncodedSample,
    answer_token_ids: Tensor,
    selections: Sequence[CandidateSelection],
    *,
    variant_batch_size: int,
) -> list[AnswerNLL]: ...
```

不应通过“先物理压缩 encoded，再调用旧删除接口”实现，因为这会增加二次索引和 mRoPE 错位风险。每个 variant 直接用原始索引表达最终保留集合：第一个 selection 是 $S$，后续 selection 是 $S\setminus g$。

#### 1.4.5 V2 scorer

新增：`src/mllm_prune/models/utility_scorer_v2.py`

```python
@dataclass(frozen=True)
class UtilityScorerV2Config:
    qwen_input_dim: int
    structural_dim: int
    aligned_dim: int
    budget_dim: int
    projection_dim: int = 256
    hidden_dim: int = 256
    modality_dim: int = 16
    dropout: float = 0.1


class JointUtilityScorerV2(nn.Module):
    def forward(
        self,
        chunk_features: Tensor,
        question_feature: Tensor,
        modality_ids: Tensor,
        structural_features: Tensor,
        aligned_features: Tensor,
        budget_features: Tensor,
    ) -> Tensor: ...                         # [G_S]
```

v2 继续构造 projected chunk、projected question、逐元素乘积和绝对差，然后拼接 aligned、structural、budget 与 modality embedding。单独建 v2 类而不是原地改变 v1，可避免旧 checkpoint 因 feature shape 变化而不可加载。

artifact 使用 `format_version=2`，加载时严格检查 feature schema、维度、aligned fingerprint 与支持的 state encoding。

#### 1.4.6 Pair sampling

修改：`src/mllm_prune/algorithms/utility_training.py`

保留当前 `stratified_pairs()` 供实验 004 使用，新增：

```python
def stratified_pairs_v2(
    labels: Tensor,
    modality_ids: Tensor,
    *,
    gap_threshold: float,
    max_per_type: int,
    high_gap_fraction: float,
    seed: int,
) -> PairBatch: ...
```

`PairBatch` 至少记录 `high`、`low`、`stratum`、`absolute_gap` 和采样统计。`high_gap_fraction` 必须满足 $[0,1]$；相同 seed、labels 和配置必须产生完全相同 pair。不能先构造所有 Vision–Vision 组合再排序，因为 150 个视觉 chunk 已有约 11k pair，扩大数据后会浪费内存；实现可按 stratum 以 bounded heap/partial selection 保留大 gap 候选，并对剩余有效 pair reservoir sample。

首版样本规模下，简单枚举 pair 仍可接受，但接口不应把完整 pair 列表写入 record。record 只保存 label，训练时按 epoch/sample/seed 确定性重建 pair。

#### 1.4.7 第二阶段 planner

放在：`src/mllm_prune/algorithms/cascaded_utility.py`

```python
def plan_conditional_utility_pruning(
    original_layout: CandidateLayout,
    state: ConditionalPruningState,
    predicted_utilities: Tensor,             # state chunk 顺序
    aligned_relevance: RelevanceScores,
    *,
    seed: int = 0,
) -> PruningPlan: ...
```

planner 按原始预测 utility 降序，不取 `clamp_min(0)`，也不除以长度：

```text
sort surviving chunks by (-predicted_utility, original_chunk_id)
remaining = final_budget
for chunk in order:
    keep = min(actual_state_capacity(chunk), remaining)
    if keep == full capacity:
        keep full chunk
    else:
        keep top aligned-score tokens inside this boundary chunk
    remaining -= keep
    stop when remaining == 0
```

稳定 tie-breaker 使用原始 chunk ID。输出的 `PruningPlan.selected_*_indices` 仍是原始候选索引，便于直接交给现有 `generate_from_plan()`。

### 1.5 文件路径与职责

#### 1.5.1 新增文件

| 路径 | 职责 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `configs/experiments/005_learned_utility_scorer_v2.yaml` | 005 唯一正式配置 | 人工冻结超参数 | 供所有入口加载的 config |
| `src/mllm_prune/algorithms/cascaded_utility.py` | prefilter state、预算、v2 planner | layout、aligned relevance、utility predictions | state、严格预算 plan |
| `src/mllm_prune/algorithms/conditional_utility.py` | state-aware feature、label 与 record schema | Qwen feature、state、aligned cache、NLL | v2 feature/label record |
| `src/mllm_prune/models/utility_scorer_v2.py` | v2 网络与 artifact I/O | v2 feature tensors | utility prediction/checkpoint |
| `scripts/cache_aligned_scores.py` | 独立运行外部 encoder 并缓存 | manifest、images、encoder、calibrator | 每样本 aligned cache |
| `scripts/generate_conditional_utility_labels.py` | 用 Qwen 生成 $u(g\mid S)$ | manifest、aligned cache、Qwen、states | train/validation conditional records |
| `scripts/train_utility_scorer_v2.py` | 多状态、多 seed 训练 | conditional records | scorer-v2 checkpoints/history |
| `scripts/run_learned_utility_pruning_v2.py` | 主评测与消融 | test、Qwen、encoder/cache、checkpoints | JSONL predictions/metrics |
| `scripts/summarize_learned_utility_v2.py` | 完整性与统计汇总 | records/checkpoints/predictions | Markdown/JSON/CSV summary |
| `tests/test_cascaded_utility.py` | prefilter/planner 单元测试 | 合成 layout/scores | budget/index/tie 行为验证 |
| `tests/test_conditional_utility.py` | feature/label/schema 单元测试 | 合成 state/features/NLL | shape、mask、mapping 验证 |
| `tests/test_utility_scorer_v2.py` | v2 scorer/artifact 单元测试 | 小 tensor/checkpoint | forward/save-load 验证 |

#### 1.5.2 修改文件

| 路径 | 修改内容 | 不允许破坏的旧行为 |
| --- | --- | --- |
| `src/mllm_prune/algorithms/aligned_scoring.py` | 增加 cache artifact、fingerprint 和 save/load | 实验 003 raw score 计算结果 |
| `src/mllm_prune/algorithms/utility_training.py` | 增加 v2 pair sampler 与 tie-aware 指标 | 实验 004 sampler/loss |
| `src/mllm_prune/models/qwen25vl.py` | 增加任意候选 selection 的 batched teacher scoring | v1 deletion variants、baseline generation |
| `.env.example` | 增加 005 cache/label/checkpoint/output 路径变量 | 003/004 现有变量 |
| `docs/PROJECT_STATUS.md` | 实现/运行后更新真实状态 | 不提前写成已完成 |

原则上不修改 `joint_pruning.py` 的基础数据类语义；如果 `Method` literal 限制新方法名，可只做向后兼容扩展。实验 004 文件不重命名、不复用 005 输出目录。

## 2. Algorithm

### 2.1 配置冻结与 state 列表

`005_learned_utility_scorer_v2.yaml` 至少包含：

```yaml
experiment: 005_learned_utility_scorer_v2
retentions: [0.75, 0.50, 0.25]
prefilter_fractions: [0.25, 0.50]
max_pixels: 401408
max_new_tokens: 32

text_chunk:
  min_tokens: 16
  max_tokens: 64
vision_window: 4

aligned_encoder:
  revision: main
  dtype: bfloat16
  text_batch_size: 64
  image_batch_size: 8
text_scoring:
  window_tokens: 16
  stride_tokens: 8
  overlap_reduce: mean
aligned_features:
  top_fraction: 0.20
  include_max: true
  include_mean: true
  include_std: true
  include_top_mean: true

label_generation:
  variant_batch_size: 8
  prefilter_delete_rounding: floor

features:
  projection_dim: 256
  hidden_dim: 256
  modality_dim: 16
  dropout: 0.10

training:
  seeds: [0, 1, 2]
  epochs: 100
  learning_rate: 0.0003
  weight_decay: 0.0001
  huber_weight: 0.25
  pair_gap_threshold: 0.001
  max_pairs_per_type_per_sample: 512
  high_gap_fraction: 0.50
  patience: 12

planning:
  use_utility_density: false
  boundary_token_score: aligned_calibrated
```

正式实现时还要写入 aligned model ID/path 的外部环境变量与 artifact fingerprint。YAML 不硬编码个人绝对路径或 GPU 编号。

训练一个统一 scorer 时，每个 train/validation sample 对每个 state $(\rho,\alpha)$ 产生一条 record。若资源不足，先固定一个核心 state，例如 $(0.5,0.5)$ 验证 conditional oracle 和 scorer，再决定是否扩展全部六个 state。

### 2.2 Phase A：缓存 Aligned Scores

入口：`scripts/cache_aligned_scores.py`

流程：

```text
load manifest and frozen aligned config
    → load aligned encoder + calibrator
    → for each sample:
        resolve question/text/images
        encode only enough Qwen metadata to reproduce CandidateLayout
        compute raw aligned text/vision token scores
        transform with frozen calibrator
        validate lengths and finiteness
        write one atomic cache artifact
    → write shard summary and fingerprint
```

这里有一个实现约束：aligned token score 必须映射到与实验 005 完全相同的 Qwen CandidateLayout。当前 `compute_aligned_raw_scores()` 接收 `EncodedSample`，而构造它会加载 Qwen。为了不让两个模型同时常驻 GPU，有两种实现路径：

1. **推荐路径**：新增轻量 `CandidateMetadata` 构造，只用 Qwen processor/tokenizer 和图像 grid 规则得到 layout、字符 span、image grid，不加载 Qwen 7B；
2. **首版 fallback**：让 Qwen 在 CPU 或 GPU 上逐样本只完成 layout 后卸载，再加载 aligned encoder；该方式启动慢、实现简单，但不适合全量反复运行。

如果无法证明轻量 layout 与 `encode_sample()` 完全一致，就不能使用推荐路径；先用真实 Qwen 对一批样本做逐字段等价测试。另一个安全 fallback 是把 Phase A 分成：Qwen metadata cache 与 aligned scoring 两个独立进程。

每个 aligned cache 应至少记录：

- `sample_id`；
- 原始 text/vision token 数和 chunk token indices；
- calibrated text/vision token scores；
- 可选 raw scores；
- encoder ID/revision、processor、calibrator hash；
- text window、stride、vision resize/map、chunk config；
- manifest/data/layout fingerprint；
- text encode、vision encode、mapping 和总耗时。

同一样本的 aligned score 对所有 $(\rho,\alpha)$ 复用。

### 2.3 Phase B：构造 Aligned Prefilter State

输入原始 layout 和 calibrated relevance。先计算：

$$
N=N_t+N_v,
$$

$$
B=\lfloor\rho N\rfloor,
$$

$$
D=N-B,
$$

$$
D_1=\lfloor\alpha D\rfloor,
$$

$$
B_1=N-D_1.
$$

用已有 `build_relevance_from_token_scores()` 按 top fraction 计算 chunk aligned relevance。随后按：

```text
(-chunk_relevance, original_global_chunk_id)
```

从高到低保留，直到 $B_1$。这与“从低到高删除 $D_1$”等价，但更容易复用严格预算 planner。

若边界 chunk 容量超过剩余 $B_1$，只保留该 chunk 内 calibrated aligned score 最高的 token。输出 state 时必须同时保存：

- 原始 token selected indices；
- 每个原始 chunk 的实际剩余 token mask；
- 完全删除、完整保留和部分保留的 chunk IDs；
- state layout 与 original-to-state mapping；
- 原始/初筛/最终预算。

状态验证包括：

```text
|selected_text| + |selected_vision| == B1
selected indices unique and in range
state chunks partition exactly the selected indices
surviving chunk count == feature/label target count
B <= B1 <= N
alpha == 0  => B1 == N and state equals original layout
retention == 1 => B == B1 == N
```

### 2.4 Phase C：生成 Conditional Utility Labels

入口：`scripts/generate_conditional_utility_labels.py`

流程：

```text
load Qwen only
    → load train/validation sample and aligned cache
    → encode sample once with Qwen
    → verify encoded layout matches cache fingerprint
    → extract frozen Qwen text/vision/question features
    → for each configured state (rho, alpha):
        build deterministic prefilter state S
        build features from tokens surviving in S
        create selection for S
        create one selection S\g for each surviving chunk g
        batched teacher-forced score all selections
        compute u(g | S)
        validate record and save atomically
```

每个 state 的 scoring 需要 $G_S+1$ 个 teacher-forced variants，但可以按压缩后长度 bucketing/batching。基础 NLL $\ell(S,y)$ 在该 state 内只计算一次。

#### 2.4.1 删除 partially retained chunk

若初筛仅保留原 chunk 的一部分，则这个 state 中的 $g$ 定义为“该 chunk 当前仍存在的全部 token”。计算 $S\setminus g$ 时删除的是这些剩余 token，而不是回到原始 chunk 再删除已不在 $S$ 的 token。

#### 2.4.2 Record 顺序

统一顺序固定为：state 中存活的 text chunks 按原始 text chunk ID，随后 vision chunks 按原始 vision chunk ID。以下字段第一维必须严格同序：

- conditional labels；
- deleted NLLs；
- chunk/question/structural/aligned/budget features；
- modality IDs；
- original chunk IDs；
- chunk capacities。

#### 2.4.3 Record schema

建议 `format_version=2`：

```text
ConditionalUtilityRecordV2
├─ experiment/sample/split/group
├─ state_id/rho/alpha/B/B1/N
├─ selected original token indices
├─ original and state chunk mappings
├─ state_mean_nll/deleted_mean_nlls/utility_labels
├─ features
├─ answer token count (not answer text unless existing policy requires)
├─ Qwen/aligned/config/manifest fingerprints
├─ variant batch size/attention backend
└─ total/stage timing and peak GPU memory
```

不把完整 Qwen token features重复保存六次。推荐每个 sample 保存一个共享 Qwen feature artifact，再让多个 state record 引用它；如果为了训练读取简单而内嵌 pooled state feature，也应确认总存储仍可接受。

### 2.5 Phase D：Conditional Oracle Gate

在完整训练前，先对少量 validation 样本运行：

1. Random；
2. Aligned-only；
3. Original LOO Oracle；
4. Aligned Prefilter + Conditional Oracle。

Conditional Oracle 直接使用真实 $u(g\mid S)$ 调用同一个 v2 planner。它必须通过与 learned scorer 完全相同的排序、边界 chunk 和生成路径，唯一差别是 utility 来源。

若 Conditional Oracle 在关键 retention 上仍没有比 Aligned-only/Original LOO 更好的 paired trend，则不启动全量 scorer sweep。因为此时继续优化网络无法突破标签目标本身的上限。

### 2.6 Phase E：训练 Scorer V2

入口：`scripts/train_utility_scorer_v2.py`

#### 2.6.1 数据加载

启动时执行：

- 检查 train/validation record fingerprint 各自一致；
- 检查两个 split 的 sample/group 不相交；
- 检查 state 覆盖完整或显式报告缺失；
- 仅用 train 拟合连续 feature 的 center/scale 和 label center/scale；
- validation 只应用 train statistics；
- test record 永远不能被该入口加载。

训练单位为一条 `(sample_id, state_id)` record；一个 batch 可先保持为一条 record，因为每条 chunk 数不同。若使用 gradient accumulation，应按 record 数记录 effective batch，而不是误写成 chunk 数。

#### 2.6.2 Forward

```text
Qwen chunk feature
    → modality-specific projection p'_g
Qwen question feature
    → shared projection q'

concat(
    p'_g,
    q',
    p'_g ⊙ q',
    |p'_g - q'|,
    aligned features,
    state-recomputed structural/redundancy features,
    budget features,
    modality embedding
)
    → shared MLP
    → predicted normalized conditional utility
```

#### 2.6.3 Pair sampling

对 TT/VV/TV 分层，忽略 $|u_i-u_j|\le\epsilon$ 的 near tie。每层最多 $P$ 个：

- `ceil(P * high_gap_fraction)` 来自最大 utility gap；
- 其余从剩余有效 pair 中按训练 seed、epoch、sample/state 派生 seed 后随机采样；
- validation pair 集在训练前固定，不能每个 epoch 改变，否则 early stopping 指标不可比；
- train pair 可以逐 epoch 重采样以扩大覆盖，但必须可复现。

#### 2.6.4 Loss 与 checkpoint

复用 ranking + Huber：

$$
\mathcal L
=\operatorname{mean}_{(i,j)}\operatorname{softplus}[-(\hat u_i-\hat u_j)]
+\lambda\operatorname{Huber}(\hat u,\tilde u).
$$

checkpoint 主选择指标建议改为 validation macro pairwise accuracy 与 top-utility recall 的预注册组合，或明确只用 macro pairwise。不能训练后查看 test 再决定。每个 epoch 保存：

- train total/rank/Huber loss；
- validation overall 与 TT/VV/TV pairwise；
- Spearman/Kendall；
- top utility recall/NDCG；
- prediction variance/tie rate；
- 各 state 指标，而不只汇总平均。

early stopping 与实验 004 一样只看 validation。最佳 epoch 和实际训练轮数必须分别记录。

### 2.7 Phase F：在线/测试推理

入口：`scripts/run_learned_utility_pruning_v2.py`

对每个样本：

```text
encode sample with Qwen
compute/load aligned token scores
    → build relevance
for each (rho, alpha):
    → build prefilter state S
    → build v2 features from S
    → scorer forward once per seed/checkpoint
    → sort chunks by raw predicted utility
    → fill final budget B
    → boundary chunk uses aligned token scores
    → generate_from_plan()
    → append one JSONL row and flush
```

正式 test 有两种资源模式：

1. **缓存模式，推荐用于实验**：预先缓存固定 test 的 aligned scores，评测时只加载 Qwen；便于避免显存冲突并使所有方法共享相同 score；
2. **端到端部署模式**：aligned encoder 与 Qwen 顺序运行，必要时在 CPU/GPU 间卸载；用于测量真实端到端 latency，但不能把预计算时间排除在部署指标之外。

同一个样本只做一次 Qwen encode 和一次 aligned score；不同 retention、alpha 和 scorer seed 复用。不同方法生成答案仍需分别运行 prefill/decode。

### 2.8 Phase G：汇总

入口：`scripts/summarize_learned_utility_v2.py`

汇总器首先做完整性校验，再计算指标：

1. 预期 key 集合：`(sample_id, method, rho, alpha, seed)`；
2. 检查重复、缺失、失败、fingerprint 混杂；
3. 检查所有 pruning row 的 `budget_deviation == 0`；
4. 汇总 prefilter recall、label 分布、scorer ranking、下游 Accuracy/robust Accuracy；
5. 对相同样本做 paired win/loss/tie 和 McNemar；
6. 按 seed、retention、alpha 和 modality 分层；
7. 汇总 stage-wise time、GPU-hours、显存与磁盘；
8. 输出 machine-readable JSON/CSV 和可写入实验报告的 Markdown 表。

## 3. Verification

“脚本能 run”只说明控制流没有立刻崩溃，不能证明算法正确。本实验按六层验证。

### 3.1 纯函数与预算单元测试

#### 3.1.1 Prefilter 预算

使用手工 layout：text capacities `(3, 2)`、vision capacities `(4, 1)`，总数 $N=10$。枚举多个 $\rho,\alpha$，验证：

$$
|S|=N-\lfloor\alpha(N-\lfloor\rho N\rfloor)\rfloor,
$$

并验证最终：

$$
|P_{final}|=\lfloor\rho N\rfloor.
$$

覆盖：`rho=1`、`alpha=0`、chunk 恰好填满、必须部分保留、单一模态为空、所有 utility 相同。

#### 3.1.2 排序方向

构造 chunk relevance `[0.9, 0.2, 0.7]`，预算只允许两个 chunk，验证初筛删除 0.2 而不是 0.9。构造 predicted utility `[3, -1, 1]`，验证细筛顺序是 `3 > 1 > -1`，没有 `clamp_min(0)` 导致退化，也没有暗中除长度。

#### 3.1.3 边界 chunk

设置最高 utility chunk 容量 5、剩余预算 2、内部 aligned scores `[0.1,0.8,0.3,0.9,0.2]`，验证只保留 score 0.9 和 0.8 对应原始索引，并且输出索引最终排序满足 Qwen 输入顺序。

#### 3.1.4 Deterministic ties

当 chunk utility 或 token score 完全相等时，使用原始 ID/index 决定次序。同样输入运行两次应逐位相同；改变无关 Python hash seed 不应改变计划。

### 3.2 状态与索引映射验证

这是最容易产生“能跑但删错 token”的位置。

对合成和真实小样本验证：

- original layout flatten 后覆盖 `0..N_m-1`；
- prefilter selected indices 与 state chunks flatten 后完全相同；
- partial chunk 的 mask 是原 chunk token set 的真子集；
- label 第 $j$ 项对应的 original chunk ID 与 feature/prediction 第 $j$ 项相同；
- $S\setminus g$ 相对 $S$ 恰好少该 state chunk 的实际 token 数；
- text/vision index 空间不混合；
- gather 后保留 token 的 embedding 和 mRoPE 与未剪枝输入相同位置逐元素一致。

建议测试中给每个 token feature 填入其原始 index，任何错位都会直接暴露，而不是依赖随机向量碰巧不相等。

### 3.3 Aligned cache 一致性验证

#### 3.3.1 缓存与在线重算等价

选 2–3 个真实样本，在同一 encoder/config 下比较：

- token score shape；
- raw/calibrated score；
- chunk top-$k$ score；
- prefilter state；
- 最终 selected indices。

缓存加载结果与在线重算应在预先定义的数值容差内相等，离散选择必须完全一致。若 BF16 在边界 tie 附近造成选择变化，应保存 FP32 score 或定义稳定量化/tie 规则。

#### 3.3.2 Fingerprint 拒绝旧缓存

分别改变 encoder revision、calibrator、text window、vision window、max pixels、chunk config 或 manifest，验证 loader 明确拒绝旧 artifact，而不是继续运行。

### 3.4 Conditional label 正确性验证

#### 3.4.1 `alpha=0` 回归到实验 004

当 `prefilter_fraction=0` 且其他 chunk/config 一致时：

$$
S=X,
$$

因此：

$$
u(g\mid S)=u_g^{v1}.
$$

对同一真实样本比较 v1 与 v2：

- baseline answer token IDs；
- full/state NLL；
- 每个 chunk deleted NLL；
- utility labels。

允许的差异只来自明确记录的数值精度；否则说明 selection、answer mask 或位置索引有 bug。这是 conditional label 最强的回归 oracle。

#### 3.4.2 手算 NLL delta

用小型 mock model 构造某个 token 被保留时 gold answer NLL 固定降低 2，验证对应 utility 为 `+2`；构造干扰 token 使删除后 NLL 降低，验证 utility 为负。确认公式方向没有写反。

#### 3.4.3 Batched/unbatched 等价

对同一 $S$，使用 `variant_batch_size=1` 与 `>1` 比较所有 NLL。结果应在容差内一致，variant 顺序不能因 length bucketing 改变。

#### 3.4.4 Conditional 行为 case

构造或选取两个可替代 chunk A/B：完整输入中删除 A、B 的影响均小；预先从 $S$ 删除 A 后，B 的 utility 应上升。真实模型不保证每个 case 都符合，但至少要保存典型样本验证 conditional label 确实能随 state 改变，而不是因实现错误始终复制 v1 label。

### 3.5 Feature 与 scorer 验证

#### 3.5.1 Feature 形状与语义

验证所有 feature 第一维等于 $G_S$，finite，且：

- 修改被初筛删除 token 的 feature 不应改变 state pooled feature；
- 修改一个存活 token 应只影响对应 chunk pooled/aligned statistics 及相关 redundancy；
- 相同 $S$ 的 budget feature 一致；不同 $(\rho,\alpha)$ 的 budget feature 可区分；
- aligned top mean 与手工 top-$k$ 结果相同；
- 单 token chunk 的 std 定义为 0，不产生 NaN。

#### 3.5.2 Overfit tiny batch

取 2–4 条 record，关闭 dropout 或固定 seed，训练足够步数。期望：

- total/ranking loss 明显下降；
- 训练 pairwise accuracy 接近 100%；
- prediction 不是常数；
- save/load 后预测逐元素一致。

若 tiny batch 都无法过拟合，应先检查 label 顺序、forward、loss 与 optimizer，而不是启动全量训练。

#### 3.5.3 Pair sampler

用手工 labels 验证：

- 每个 pair 的 high label 严格大于 low；
- gap 均大于 $\epsilon$；
- TT/VV/TV 分类正确；
- 每类不超过 $P$；
- high-gap 数量满足配置或受候选不足限制；
-相同 seed 可复现，不同 epoch seed 能改变随机尾部但不改变固定 validation pairs。

### 3.6 端到端 smoke

建议依次运行：

1. 1 个 train + 1 个 validation 样本缓存 aligned scores；
2. 对一个 state 生成 conditional labels；
3. tiny-overfit 并保存 checkpoint；
4. 对固定 test 的 1 个样本运行 baseline、aligned-only、v2；
5. 验证生成不为空、无 NaN、预算严格、artifact 可重载；
6. 运行 `alpha=0` 回归，与实验 004 同路径结果比较；
7. 用同一条件重新运行一次，确认断点续跑不产生重复 JSONL row。

### 3.7 根据结果判断“算法是否工作”

除程序正确外，至少满足以下证据链才值得启动完整主实验：

| 环节 | 需要观察的结果 | 如果失败说明什么 |
| --- | --- | --- |
| Aligned 初筛 | validation 高 utility chunk recall 明显高于同预算 Random，尤其视觉高 utility recall 不崩溃 | encoder/映射/top-$k$/calibration 或 $\alpha$ 有问题 |
| Conditional label | label 非全零、非全同；相对 v1 有可解释 rank 变化 | state 没真正进入标签，或 conditional 假设没有产生作用 |
| Conditional Oracle | 在小 validation 下优于或至少显示稳定 paired trend 超过 Aligned-only/Original LOO | 标签目标上限不足，不应先怪 scorer |
| Scorer tiny overfit | loss 下降、train ranking 接近满分 | 实现/特征/loss bug |
| Scorer validation | Vision–Vision 明显高于约 50% 且 seed 较稳定；top utility recall 提升 | 视觉细粒度仍未学会或数据不足 |
| 下游 test | 多 seed、主要 retention 下稳定优于 Random、Aligned-only 与 v1 | 若 ranking 好而下游差，检查 planner/metric 对齐 |

这些不是为了用 validation/test “保证正结果”，而是为了区分：实现错误、上游 encoder 失败、label 目标失败、scorer 学习失败和下游选择失败。

### 3.8 建议验证命令类别

实现后至少运行：

```bash
pytest -q \
  tests/test_joint_pruning.py \
  tests/test_aligned_scoring.py \
  tests/test_utility.py \
  tests/test_cascaded_utility.py \
  tests/test_conditional_utility.py \
  tests/test_utility_scorer_v2.py
```

再分别提供 CPU 算法 smoke 与真实 GPU 1-sample smoke。这里暂不写最终 CLI 参数，以实际实现后的 `--help` 为准，避免设计文档中的命令先于接口失效。

## 4. Risk

### 4.1 Aligned encoder 的 dense feature 不可靠

**风险**：CLIP/SigLIP 的全图 pooled feature 与文本对齐，不代表任意 patch feature 都具备可靠 dense alignment。若 patch score 无意义，初筛会不可逆地误删关键视觉证据。

**检测**：实验 003 的代码路径检查、局部区域可视化、合成图定位、validation 高 utility recall。

**Fallback**：

1. 更换经过 dense grounding/region-text 训练的 encoder；
2. 降低 $\alpha$，让 aligned score 只做极保守过滤；
3. 对视觉只按图级/大区域初筛，chunk 内保留空间覆盖；
4. 若视觉 dense alignment 始终失败，首版仅初筛文本，视觉全部交给 utility scorer。

### 4.2 Prefilter false negative 不可恢复

**风险**：第一阶段删掉的 chunk 不会进入 scorer，任何后续模型都无法恢复。

**检测**：高 utility recall、误删案例、分模态 recall、不同 $\alpha$ 曲线。

**Fallback**：减小 $\alpha$；设置小型 modality safety floor；保护数字/OCR/实体/否定词；用 relevance 与 uncertainty 联合决定是否允许删除。

### 4.3 Conditional LOO 仍不能处理成组交互

**风险**：若两个冗余 chunk 都在 $S$ 中，分别删除仍可能 utility 很低，细筛可能把两者都删掉。

**检测**：成组由对变错 case、冗余度与错误相关性、pair/group oracle。

**Fallback**：MMR/coverage constraint、pair/group deletion label、多个随机 $S$、近似 Shapley、分批删除后重新预测。

### 4.4 一个统一 scorer 面对多个 state 可能容量不足

**风险**：不同 $\rho,\alpha$ 产生的 $S$ 分布差异较大，120 个 train 样本却被展开为相关的多 state record；记录数增加不等于独立样本增加。

**检测**：按 state 报告 validation；观察某些 state 系统性失败；比较统一 scorer 与单-state scorer。

**Fallback**：先固定主 state；为每组 state 单独训练；或增强 budget/state conditioning。不能把六倍 record 当作六倍独立数据声称样本量扩大。

### 4.5 Aligned calibration 与 task 分布失配

**风险**：实验 003 的 calibration 分布或 encoder config 与 005 不一致，导致跨模态分数失衡。

**检测**：artifact fingerprint、分模态 score quantile、初筛模态占比、不同 split drift。

**Fallback**：仅用无泄漏 calibration set 重新拟合 artifact；不在 test 样本内分别归一化到相同分布；必要时将 raw 与 calibrated 作为消融而非混用。

### 4.6 缓存 layout 与真实 Qwen layout 不一致

**风险**：tokenizer、max pixels、vision window 或 processor 改变后，旧 aligned score 映射到错误 token，shape 可能相同但语义已错位。

**检测**：layout fingerprint 不只包含数量，还包含 chunk/token span、image grid、processor/config hash；真实小样本 cache-vs-online 等价测试。

**Fallback**：拒绝加载并重建缓存。不能通过 truncate/pad 让 shape 勉强匹配。

### 4.7 Qwen 与 aligned encoder 同卡 OOM

**风险**：实验 004 标签生成峰值约 20.7–21.1GB；在 24GB GPU 上再常驻外部 encoder 很可能 OOM。

**Fallback**：主方案分阶段缓存 aligned score，使标签生成只加载 Qwen；端到端测试采用顺序加载/CPU offload，或将两个模型放不同 GPU。减小 batch 只能缓解 activation，不能消除双模型权重常驻。

### 4.8 Variant batch 因长度差异 OOM

**风险**：某些样本 chunk 多、序列长，`variant_batch_size=8` 可能超过显存；初筛比例较小时尤其接近实验 004。

**检测**：记录每样本长度、$G_S$、batch、峰值显存；smoke 覆盖 P95 长样本而不只随机短样本。

**Fallback**：按压缩长度 bucketing；OOM 时仅对当前样本自适应减半 batch 并记录，最低降到 1；不静默转 CPU 跑全量。

### 4.9 Pair 数增加导致 CPU 时间和内存上升

**风险**：视觉 chunk 数多时 pair 为 $O(G_v^2)$；提高 $P$ 不应意味着保存全部 pair。

**Fallback**：streaming/reservoir sampling、partial top-k、每 epoch 重建而不落盘；若训练仍被 pair 构造限制，预先缓存紧凑 pair index，但必须绑定 label/config fingerprint。

### 4.10 Raw utility 偏好长 chunk

**风险**：不除长度符合当前价值定义，但长 chunk 的整体删除效应可能天然更大，导致预算效率下降。

**检测**：预测/真实 utility 与 capacity 的相关性、长短 chunk 选择率、raw-vs-density 消融。

**Fallback**：严格 budget knapsack、学习 per-token opportunity cost，或重新定义固定大小 chunk；只有结果支持时再恢复 density，不能先验断言任一规则必然正确。

### 4.11 小 validation 上超参数过拟合

**风险**：30 个 validation 样本不足以同时搜索 $\alpha$、top-$k$、$P$、ratio、$\epsilon$、$\lambda$ 等大量组合。

**Fallback**：分阶段 gate：先定 aligned/prefilter，再定 label/scorer；每阶段只改变少数参数；保存全部候选结果；固定 test 只运行冻结配置。如需扩大选择空间，先扩大 train/validation 而非查看 test。

### 4.12 长任务中断或 stale record 混入

**风险**：多卡 shard、中断重启和 config 改动可能产生缺失、重复或新旧 record 混合。

**Fallback**：原子写入、fingerprint、显式 `--overwrite`、按 sample/state 唯一 key、汇总前完整性审计。不同 shard 只能写不同目录/文件，结束后再 merge。

## 5. Analysis

### 5.1 复杂度

设：

- $N$：原始候选 token 数；
- $G$：原始 chunk 数；
- $N_S,G_S$：初筛后 token/chunk 数；
- $M$：gold answer token 数；
- $K$：训练 state 数；
- $P$：每种 pair 类型上限。

#### Aligned scoring

外部 encoder 的主要成本来自文本窗口和图像 patch encoding。每样本只执行一次并跨 $K$ 个 state 复用。token-to-chunk top-$k$ 聚合可用 `topk`，总体约为：

$$
O(N\log k)
$$

或小 chunk 下近似线性。

#### Prefilter 与 planner

chunk 排序：

$$
O(G\log G),
$$

边界 chunk token 排序最多：

$$
O(|g_b|\log |g_b|).
$$

相对模型前向通常可忽略。

#### Conditional labels

每个 state 需要 $G_S+1$ 个 teacher-forced variants。Transformer 成本随压缩后序列长度强烈增长，粗略写为：

$$
O\left(K(G_S+1)\cdot C_{Qwen}(N_S+M)\right).
$$

因此它是本实验的主要离线成本。初筛会同时减少 $G_S$ 和序列长度，单 state 通常比实验 004 完整输入 LOO 更便宜；但若一次生成 6 个 state，总成本仍明显高于实验 004。

#### Pair 构造

朴素枚举为：

$$
O(G_S^2),
$$

最终参与 loss 的 pair 至多约为 $3P$。首版 $G\approx150$ 时枚举仍小，但应避免为全部样本和 epoch长期保存所有组合。

#### Scorer

scorer 对所有 chunk 一次前向，成本近似：

$$
O(G_S d h),
$$

相对 Qwen 很小。其显存主要是模型参数和每条 record feature，通常不是瓶颈。

### 5.2 运行时间估计

以下以实验 004 的实测数据为基线，而不是承诺值：

- 120 train + 30 validation 的完整上下文 LOO 标签合计约 **121.65 GPU-min**；
- 单样本平均约 42.5s（train）和 73.3s（validation），长样本 P95 可超过 100–200s；
- 完整 004 pipeline 单卡墙钟约 2h38m，包括标签、训练和 test 评测。

005 每个 conditional state 的 $N_S,G_S$ 小于完整输入，因此单 state 标签大致可能是 v1 的 **45%–90%**，具体取决于 $(\rho,\alpha)$、batch padding 和序列长度。由此给出保守规划：

| 工作 | 单卡估计 | 4 张独立 GPU 样本分片估计 | 说明 |
| --- | ---: | ---: | --- |
| aligned cache，150 train/val | 约 10–40 min | 约 3–15 min | 强依赖 encoder、图片数与是否已有 metadata cache |
| 一个核心 state 的 conditional labels | 约 1–2 h | 约 20–45 min | 先用 $(\rho=0.5,\alpha=0.5)$ smoke/gate |
| 全 6 states conditional labels | 约 6–11 GPU-h | 约 1.5–3.5 h 墙钟 | 不是 6 倍独立样本，仅是 6 个条件状态 |
| 3-seed scorer training | 约数分钟到 30 min | 可并行到单 seed 时长 | 小模型，通常 CPU/I/O 也可能占主导 |
| 固定 50 test 主评测 | 约 1–3 h | 约 20–60 min | 取决于方法/seed/state 条件数和 decode |
| 全流程 | 约 8–15 GPU-h | 约 2–5 h 墙钟 | 未计排队、失败重跑和大范围超参数 sweep |

这些范围必须在实现后的代表性 smoke 上更新。建议分别选普通样本和 chunk 数/P95 长度样本计时，再按真实 $G_S$ 外推。不能只用一个短样本估算全量。

### 5.3 显存压力

#### Phase A：aligned cache

只加载 aligned encoder 时，显存取决于 checkpoint 和 image batch；预期显著低于 Qwen 7B。主要峰值来自多图 batch 和 dense patch feature。可通过 image batch size 调整。

#### Phase C：conditional labels

实验 004 已实测约 20.7–21.1GB 峰值。005 若只加载 Qwen、保持 BF16/FlashAttention 2，初筛后的序列更短，理论上 activation 压力不高于相同 batch 的 v1；但 variant padding、长 answer 和部分 state 接近完整输入，仍应按 **约 18–22GB** 规划。

因此：

- 24GB 卡可能可运行 Qwen-only label generation，但余量较小；
- Qwen 与 aligned encoder 不应默认同卡常驻；
- 48GB/80GB 卡更宽裕，可提高 variant batch，但要实测吞吐而不是盲目放大；
- scorer training 通常低于 1–2GB，不值得用 DDP；多个 seed 更适合独立并行。

### 5.4 时间优化方案

按收益和风险排序：

1. **aligned score 单次缓存并跨 state/seed/method 复用**；
2. **Qwen 每样本只 encode 一次**，同一进程内依次生成多个 state label；
3. **先跑核心 state 和 Conditional Oracle gate**，没有上限收益就不生成六个 state；
4. **多卡按样本 shard**，每卡独立加载模型、输出独立目录，最后审计 merge；
5. **variants 按压缩长度 bucketing**，减少 padding；
6. **动态 variant batch**：对短样本提高 batch，对长样本降低；
7. **BF16 + 已验证的 FlashAttention 2**，若不兼容则明确 fallback 到 SDPA；
8. **缓存共享 Qwen feature**，避免多 state record 重复写 3584 维 pooled feature；
9. **test 中同一样本共享 encode/aligned score/features**，多个方法只重复必要的 generation；
10. **3 个 scorer seed 独立并行**，而不是为小 MLP 引入 DDP 通信。

不建议首版使用模型量化改变 teacher/scoring 数值，因为这会改变 utility label 定义；如为部署测试量化，应作为独立实验条件。

### 5.5 显存优化方案

1. 将 aligned cache 与 Qwen label generation 拆为不同进程；
2. 缩小 `variant_batch_size`，必要时按样本自适应；
3. 使用 `torch.inference_mode()`、`model.eval()` 和 BF16；
4. 及时释放只在单样本使用的大 tensor，避免把 GPU tensor放入长期 Python record；
5. cache/record 写盘前统一 detach 到 CPU，Qwen pooled feature可用 BF16，score/label 保持 FP32；
6. 对 test 端到端模式，让 aligned encoder 和 Qwen 顺序上卡，或放到不同 GPU；
7. 如果 24GB 仍 OOM，优先减 variant batch，而不是改变 max pixels、vision window 或序列长度；后者会改变研究设定。

梯度检查点对 inference-mode label generation没有收益；FSDP/DDP 也不适合单样本独立 teacher scoring。多卡最自然的策略仍是样本并行。

### 5.6 I/O 与存储

实验 004 的 150 个 label record 约 160MB。005 若简单复制六个 state，可能达到约 **0.5–1GB**，仍可接受，但会重复大量 Qwen feature。推荐：

- aligned cache 每样本一份；
- Qwen shared feature 每样本一份；
- state record 只保存 state mask、pooled feature、aligned statistics、NLL 和 label；
- 原子写入单样本/单 state 文件；
- 汇总时流式读取，不一次把所有大 tensor常驻内存。

若实现复杂度超过收益，首版允许 record 自包含，但必须在 smoke 后实测平均文件大小并估算全量存储，而不是凭感觉决定。

### 5.7 推荐执行顺序

实现与运行按以下 gate 推进：

```text
纯算法数据结构与单元测试
    ↓
aligned metadata/cache 等价验证
    ↓
1 train + 1 validation + 1 test GPU smoke
    ↓
P95 长样本显存/耗时 smoke
    ↓
少量 validation 的 prefilter recall + Conditional Oracle
    ↓ 只有上限结果支持时
一个核心 state 的完整 train/validation labels
    ↓
3-seed scorer + validation 分析
    ↓ 只有 scorer 确实学到时
扩展其他 state / 正式固定 test
```

任何预计超过约 10 分钟的环境配置、模型下载、全量标签、训练或评测，在实际执行前都应给出命令、GPU 选择、耗时与显存估计，由用户在长程终端中启动；不由交互会话持续监控。
