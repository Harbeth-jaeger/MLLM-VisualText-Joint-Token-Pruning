建议把实验 003 实现为三层：外部对齐 encoder 适配层负责产生可解释的 raw relevance，校准层使用独立 calibration set 将文本和视觉分数转换到可比的模态内分位数，已有联合剪枝层继续负责 chunk 价值、严格总预算、选择、mRoPE 保留和解码。

该实验只替换 **scorer**，不替换 Qwen2.5-VL 的视觉 encoder、projector 或 LLM。这是本轮的主要归因边界：如果结果变化，应首先归因于候选 token 的打分与选择，而不是答案模型本身发生了变化。

还有一个需要在选型阶段确认的前提：许多 CLIP/SigLIP checkpoint 只显式对齐全图 pooled embedding 和整句 pooled embedding，不能因为能取出中间 hidden states，就默认每个 patch/token 也经过了可靠的 dense alignment。正式实现前必须用 checkpoint 文档、前向路径和少量可视化同时验证这一点。

## 1. Problem

### 1.1 要解决的问题

对于一个 MMCoQA 样本，仍使用实验 002 定义的可剪枝候选集：

$$
\mathcal C=T_c\cup V,
$$

其中 $T_c$ 为文本上下文 token，$V$ 为 Qwen vision merger 后的视觉 token。问题 $T_q$、system prompt、图像边界和特殊 token 全部保护。

对给定候选保留率 $\rho$，总预算仍为：

$$
B=\left\lfloor\rho(N_t+N_v)\right\rfloor.
$$

实验 002 的关键问题是，它直接比较 Qwen LLM text embedding 与 vision projector embedding 的 cosine：

$$
r_i=\max_j\cos(h_i,h_{q,j}),
$$

但这两种 embedding 没有被训练成具有可比 cosine 几何的公共检索空间，导致文本分数远高于视觉分数，动态预算大多退化为“先保留文本”。

本轮要计算两类新分数：

$$
r_i^t=\operatorname{AlignedTextScore}(x_i^t,q),
$$

$$
r_i^v=\operatorname{AlignedVisionScore}(x_i^v,q),
$$

并用预先拟合的分模态校准器转换为：

$$
s_i^m=F_m(r_i^m),\qquad m\in\{t,v\}.
$$

最终 $s_i^t,s_i^v\in(0,1)$ 作为已有 joint planner 的 token relevance。算法需要同时满足：

* 外部 encoder 完全冻结，不使用 test answer；
* calibration 与 test 按样本及源数据 group 隔离；
* 文本与视觉分数都来自外部对齐空间；
* 分位数映射是全局预先拟合的，不在每个 test 样本内重新排名；
* 不强制当前样本的文本/视觉总分相等；
* 总保留数严格等于 $B$；
* 不改变 Qwen 的输入 embedding、原始 mRoPE 位置和生成参数。

### 1.2 输入与输出

校准阶段输入：

```text
CalibrationInput
├─ calibration sample manifest
├─ MMCoQA images and text contexts
├─ frozen aligned encoder + processor
├─ text micro-window configuration
├─ external vision preprocessing/grid metadata
└─ sampling/weighting configuration
```

输出一个可版本化的校准产物：

```text
CalibrationArtifact
├─ format_version
├─ encoder model ID and revision
├─ processor/config fingerprint
├─ calibration manifest fingerprint
├─ text raw-score support and cumulative weights
├─ vision raw-score support and cumulative weights
├─ fit sample/candidate counts
├─ score quantiles and diagnostics
└─ fitting configuration
```

推理阶段输入：

```text
AlignedScoringInput
├─ question text
├─ original context text and Qwen token spans
├─ original images
├─ Qwen CandidateLayout and vision-grid metadata
├─ frozen aligned encoder
└─ frozen CalibrationArtifact
```

输出：

```text
AlignedRelevance
├─ text_raw_scores          [N_t]
├─ vision_raw_scores        [N_v]
├─ text_calibrated_scores   [N_t]
├─ vision_calibrated_scores [N_v]
├─ text_chunk_scores        [G_t]
├─ vision_chunk_scores      [G_v]
├─ text/vision modality scores
├─ external/Qwen grid mappings
└─ timing and numerical diagnostics
```

在相同样本上，`AlignedRelevance` 对所有 retention 共享，不应为 75%/50%/25% 重复计算外部 encoder。

### 1.3 建议的核心接口

外部 encoder 的模型无关协议：

```python
class AlignedEncoder(Protocol):
    def encode_text(self, texts: Sequence[str]) -> Tensor:
        """Return L2-normalized pooled embeddings with shape [B, D]."""

    def encode_image_patches(
        self,
        images: Sequence[Image.Image],
    ) -> Sequence[DenseImageFeatures]:
        """Return projected, L2-normalized dense features and grid metadata."""
```

```text
DenseImageFeatures
├─ features       [H_e, W_e, D]
├─ valid_mask     [H_e, W_e]
├─ grid_height / grid_width
├─ resize/crop/pad transform
└─ original image width / height
```

原始分数接口：

```python
compute_aligned_raw_scores(
    encoder: AlignedEncoder,
    question: str,
    context: str,
    encoded_qwen: EncodedSample,
    images: Sequence[Image.Image],
    config: AlignedScoringConfig,
) -> RawAlignedScores
```

校准器：

```python
class EmpiricalCDFCalibrator:
    @classmethod
    def fit(
        cls,
        text_scores: Sequence[WeightedScores],
        vision_scores: Sequence[WeightedScores],
        metadata: CalibrationMetadata,
    ) -> "EmpiricalCDFCalibrator": ...

    def transform(
        self,
        text_scores: Tensor,
        vision_scores: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "EmpiricalCDFCalibrator": ...
```

从外部分数构造已有 planner 可用的 relevance：

```python
build_relevance_from_token_scores(
    layout: CandidateLayout,
    text_token_scores: Tensor[N_t],
    vision_token_scores: Tensor[N_v],
    *,
    token_top_fraction: float,
    chunk_top_fraction: float,
) -> RelevanceScores
```

应在现有 `joint_pruning.py` 中增加这个通用入口，而不让 planner 直接依赖某个 CLIP/SigLIP 类。`plan_joint_pruning(..., relevance=...)` 可继续复用，避免重写预算和选择逻辑。

## 2. Algorithm

### 2.1 数据划分与校准产物生命周期

从 MMCoQA 中固化 calibration/test manifest。实验 002 的 50 个样本全部归入 test。calibration 必须排除：

* 相同 sample ID；
* 相同源文档、对话或问题 group；
* 相同或可高置信判定为重复的图像。

拟合校准器不需要 answer，只需要 $(q,T_c,I)$。建议将流程分成两个显式命令：

```text
prepare calibration manifest
    → fit score calibrator once
    → freeze artifact and fingerprint
    → run test with the frozen artifact
```

每次正式运行必须校验 artifact 中的 encoder revision、processor fingerprint 和当前配置一致。不匹配时报错，不得静默复用旧分布。

### 2.2 文本对齐分数

外部 text encoder 通常对齐的是整句 pooled embedding，不能将其每个 hidden token 直接当作对齐 embedding。为了产生与 Qwen context token 一一对应的分数，主方案使用局部文本微窗口。

先取问题 pooled embedding：

$$
u_q=\operatorname{normalize}(E_T(q)).
$$

对实验 002 的每个文本 macro chunk，按 Qwen token 边界构造重叠 micro windows：

$$
W_{g,k}=T_c[a_{g,k}:b_{g,k}],
$$

默认可从 `window_tokens=16, stride_tokens=8` 开始，但它们是需要在 calibration/validation 数值检查后固定的实验参数。不跨越文档、对话轮次、图像占位符或 macro chunk 边界。

批量编码后：

$$
u_{g,k}=\operatorname{normalize}(E_T(W_{g,k})),
$$

$$
a_{g,k}=u_{g,k}^{\top}u_q.
$$

一个 Qwen 文本 token $i$ 可能被多个微窗口覆盖，其 raw score 定义为所有覆盖窗口的平均：

$$
r_i^t=
\frac{1}{|\mathcal W(i)|}
\sum_{W_{g,k}\in\mathcal W(i)} a_{g,k}.
$$

平均比 max 更不容易被一个偶然高分窗口支配；`max` 可作为消融。如果某个 token 未被任何窗口覆盖，应视为映射错误，不得用 0 静默填充。

必须从原始文本字符区间反向映射 Qwen token，而不用 `decode(token_ids)` 重建后再假设字符边界不变。否则空格、Unicode 和 tokenizer normalization 可能使 micro-window 错位。

### 2.3 视觉对齐分数

对每张原始图像 $I_k$ 分别运行外部 vision encoder：

$$
F_k^e=E_I(I_k)\in\mathbb R^{H_e\times W_e\times D_e}.
$$

只有在确认 patch feature 经过与 text embedding 对应的 projection 后，才计算：

$$
U_k^e=\operatorname{normalize}(P_I(F_k^e)),
$$

$$
R_k^e[a,b]=(U_k^e[a,b])^\top u_q.
$$

外部 processor 可能对图像执行 resize、center crop 或 padding。`DenseImageFeatures` 必须显式保留这些几何变换，将外部 patch cell 映射回原图归一化坐标 $[0,1]^2$。Qwen merger 后每个视觉 token 同样表示为原图中的一个矩形 cell。

第 $i$ 个 Qwen vision token 的 raw score 是所有相交 external cells 的面积加权平均：

$$
r_i^v=
\frac{
\sum_{a,b}\operatorname{area}(C_i^q\cap C_{a,b}^e)R_k^e[a,b]
}{
\sum_{a,b}\operatorname{area}(C_i^q\cap C_{a,b}^e)
}.
$$

这比将 score map 简单 bilinear resize 更符合“一个 token 覆盖一块图像区域”的语义。实现上可使用等价的 area interpolation，但必须通过合成 grid 测试证明等价。

约束：

* 多图分别映射，不跨图插值；
* padding 区域不参与加权；
* 原图 crop 掉的内容无法恢复，因此主选 encoder 应优先支持保留完整画面的 resize/pad；
* 本轮不处理视频；发现 $T>1$ 时应明确报错或走单独已测试的帧聚合路径；
* 映射只用于 score，不改写 Qwen 视觉 embedding 和 mRoPE。

### 2.4 Dense Alignment 资格检查

在固定 external checkpoint 前，先进行三级检查：

1. **代码路径检查**：patch feature 与 pooled image embedding 是否使用了同一 image projection，映射后维度是否与 text projection 一致。
2. **全局一致性检查**：对 projected patch 做模型定义的 pooling 后，能否在数值误差内复现官方 pooled image embedding/logit。
3. **局部语义检查**：对若干含明确局部目标的非 test 图像，检查问题相关 score map 是否在目标区域达到更高排名。

第 1 级失败则不能使用该 checkpoint 做 patch scorer。第 2/3 级失败时，应更换支持 dense retrieval/grounding 的 checkpoint；区域 crop 重复编码只用于估计上限或 fallback，因为它会显著增加计算。

### 2.5 分模态经验 CDF 拟合

从 calibration set 收集文本和视觉 raw token score：

$$
\mathcal R_t=\{r_{s,i}^t\},\qquad
\mathcal R_v=\{r_{s,j}^v\}.
$$

不能把所有 token 无权堆叠，因为高分辨率图像和长文本样本会因 token 更多而过度影响分布。主方案使每个样本、每个存在的模态总权重相同：

$$
w_{s,i}^m=\frac{1}{|S_m|N_{s,m}},
$$

其中 $|S_m|$ 是包含模态 $m$ 的 calibration 样本数，$N_{s,m}$ 是样本 $s$ 中该模态的候选数。

对每个模态建立加权经验 CDF。为减少 tie 和有限样本导致的 0/1 饱和，使用 mid-rank：

$$
F_m(r)=
\sum_{x<r}w_x+
\frac12\sum_{x=r}w_x.
$$

实现时将结果 clamp 到 $[\epsilon,1-\epsilon]$，$\epsilon$ 由有效 calibration 样本量决定，不作为 test 调参项。对于两个 support value 之间的新分数，主方案使用 `searchsorted` 定位的阶梯 CDF；线性插值只作为消融。

必须保留以下诊断：

* raw score 的 min/max/mean/std 与 1/5/25/50/75/95/99 分位数；
* 每个模态的样本数、token 数和有效总权重；
* 重复值比例和 calibration 后饱和率；
* 按文本/视觉 token 数分桶后的分布；
* calibration 与 test raw score 的 drift，但不使用 test answer 调整映射。

### 2.6 校准变换

对一个 test 样本，先产生 raw score，然后批量查表：

$$
s_i^t=F_t(r_i^t),\qquad
s_j^v=F_v(r_j^v).
$$

该步骤必须是单调的：

$$
r_a^m<r_b^m\Longrightarrow F_m(r_a^m)\le F_m(r_b^m).
$$

因此校准不会改变同一模态内的排名，只会改变视觉与文本之间的数值对比。

需要同时保留两组 relevance：

```text
raw relevance          -> aligned_uncalibrated condition
calibrated relevance   -> aligned_calibrated condition
```

两个条件使用完全相同的 chunk、预算和生成路径，才能将差异归因于 calibration。

### 2.7 从 Token Score 构造 Chunk 与模态价值

对文本和视觉使用与实验 002 一致的 top-fraction mean：

$$
r_g=
\operatorname{mean}
\left(
\operatorname{TopK}_{\lceil\beta|G_g|\rceil}
\{s_i:i\in G_g\}
\right),
$$

$$
R_m=
\operatorname{mean}
\left(
\operatorname{TopK}_{\lceil\gamma|\mathcal G_m|\rceil}
\{r_g:g\in\mathcal G_m\}
\right).
$$

主配置沿用实验 002 的 $\beta=\gamma=0.2$。不根据实验 003 test Accuracy 重新调整，否则会同时改变 scorer 和 planner 两个变量。

为避免平均分位数因候选数不同而产生不直观的偏移，应同时记录 $R_t,R_v$、top chunk 分数、两模态候选数和最终预算，不只保存 $B_t,B_v$。

### 2.8 严格联合预算与选择

直接复用实验 002 已验证的：

* 模态 softmax 及最低预算；
* capacity clamp 与 overflow reallocation；
* chunk 内带容量整数 water filling；
* chunk 内 top-k token 选择；
* 原始序列顺序恢复；
* 完整 mRoPE position IDs 按 keep mask gather；
* 自定义 greedy prefill/decode。

不再把 visual-only/text-only 作为主比较，因为当被强制完整保留的模态容量已经大于 $B$ 时，它们无法同时满足“单模态剪枝”和“统一总预算”。

主对照为：

```text
No Pruning
Random Joint x 3 seeds
Fixed Joint
Raw Qwen Joint
Aligned Uncalibrated Joint
Aligned Calibrated Joint
```

所有 joint 条件必须严格使用同一 $B$。

### 2.9 缓存与高效执行

单个样本的推荐执行顺序：

```text
read sample and images
    → Qwen encode once
    → external question/text windows encode in batches
    → external images encode once
    → map raw scores to Qwen candidates
    → CDF lookup once
    → construct raw/calibrated relevance once each
    → reuse them for every retention and condition
    → generate and write records
```

实现要点：

* 在运行前对 external encoder 的 text/image batch 上限做短 profile；
* 文本微窗口按 tokenized length bucketing，减少 padding；
* 多张图像尺寸兼容时批处理，否则分组批处理；
* raw external features 只在当前样本 GPU 上保留，结果转为 CPU FP32 score；
* 长期缓存默认保存 raw score 而不是 dense feature，便于重算 calibration 且节省空间；
* cache key 至少包含 image/text content hash、encoder revision、processor fingerprint 和 scoring config；
* Qwen 与 external encoder 均使用 `eval()` 和 inference mode；
* Qwen 继续使用已验证的 FlashAttention 2；external encoder 优先使用其支持的 SDPA/FlashAttention，不兼容时独立回退；
* 多卡使用样本并行，每个 GPU 一个 Qwen 和一个 external encoder 实例，不需要 DDP 通信。

如果 Qwen 与 external encoder 同时常驻造成显存压力，fallback 顺序为：减小 external batch；将 external score 离线缓存后卸载 scorer；最后才考虑使用更小 external encoder。不应先改变 Qwen 量化或图像分辨率，因为这会改变实验设定。

## 3. 文件分工

建议在现有分层上增量实现，不复制实验 002 的 Qwen 解码和预算代码。

| 文件 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| `src/mllm_prune/models/aligned_encoder.py` | 外部 encoder 协议、checkpoint 适配、text pooling、projected dense patch 提取 | text/image batch、model config | normalized text embedding、`DenseImageFeatures` |
| `src/mllm_prune/algorithms/aligned_scoring.py` | 文本微窗口、token 回填、视觉面积映射和 raw score 构造 | encoder 输出、Qwen layout | `RawAlignedScores` |
| `src/mllm_prune/algorithms/calibration.py` | 加权经验 CDF 拟合、查表、序列化与 fingerprint 验证 | calibration raw scores | `CalibrationArtifact`、calibrated scores |
| `src/mllm_prune/algorithms/joint_pruning.py` | 增加从预计算 token score 构造 `RelevanceScores` 的通用入口 | layout、text/vision scores | `RelevanceScores` |
| `src/mllm_prune/models/qwen25vl.py` | 仅按需补充 Qwen token 字符区间和原图 grid cell metadata | prompt/processor 输出 | 可对齐的 span/grid metadata |
| `scripts/prepare_exp003_calibration.py` | 构建 group-disjoint calibration manifest，输出泄漏检查摘要 | MileBench 原始索引、test manifest | calibration manifest/report |
| `scripts/fit_score_calibration.py` | 分片提取 calibration raw score，合并后拟合 CDF | config、manifest、images | raw-score shards、calibration artifact |
| `scripts/run_aligned_pruning.py` | 实验 003 条件编排、断点续跑和分 shard 写出 | `.env`、config、test subset | 逐样本 JSONL |
| `scripts/summarize_aligned_pruning.py` | 完整性、双评测口径、分数/预算分布、配对结果和效率汇总 | result shards | JSON/CSV/Markdown summary |
| `configs/experiments/003_aligned_encoder_scoring.yaml` | encoder、校准、微窗口、条件和性能参数 | 人工配置 | runner config |
| `tests/test_calibration.py` | CDF 单调性、权重、tie、边界和序列化测试 | 人工分数 | 精确期望分位数 |
| `tests/test_aligned_scoring.py` | 文本窗口覆盖与合成视觉 grid 映射 | fake encoder/features | token score 与映射断言 |
| `tests/test_aligned_encoder.py` | 真实 checkpoint 小型集成测试 | 1–2 个本地 fixture | shape、norm、projection/grid 断言 |
| `tests/test_exp003_pipeline.py` | raw/calibrated relevance 到 pruning plan 的集成测试 | 小样本或 fake model | 预算、序列与回归断言 |

建议核心配置至少包含：

```yaml
experiment: 003_aligned_encoder_scoring
aligned_encoder:
  model_path_env: ALIGNED_ENCODER_PATH
  revision: ...
  dtype: bfloat16
  text_batch_size: ...
  image_batch_size: ...
  require_dense_projection: true
text_scoring:
  window_tokens: 16
  stride_tokens: 8
  overlap_reduce: mean
vision_scoring:
  grid_reduce: area
calibration:
  artifact_path_env: EXPERIMENT_003_CALIBRATION_PATH
  sample_weighting: equal_sample_within_modality
  tie_method: midrank
  interpolation: step
retentions: [0.75, 0.50, 0.25]
methods:
  - fixed_joint
  - random_joint
  - raw_qwen_joint
  - aligned_uncalibrated
  - aligned_calibrated
random_seeds: [0, 1, 2]
```

个人路径、模型路径、calibration artifact 与输出路径全部放在 `.env`，不写入 YAML 或源码。精确 checkpoint revision 和算法超参写入 YAML，便于复现。

## 4. Verification

### 4.1 纯校准算法正确性

使用可人工计算的小数组验证：

* 输出严格落在 $(0,1)$ 的预期边界内；
* raw score 增大时 percentile 不下降；
* 相同 raw score 获得相同 mid-rank；
* 两个样本即使 token 数相差很大，对 CDF 的总权重仍相同；
* text CDF 不会意外应用到 vision score，反之亦然；
* 低于/高于所有 support 的输入被稳定 clamp；
* save/load 前后的输出逐值一致；
* artifact version、encoder revision 或 processor fingerprint 不匹配时拒绝运行；
* 输入 NaN/Inf 显式报错，不参与排序。

除手写例子外，对随机有限数组检查单调性和加权和，但不必为这个有限算法引入额外大型依赖。

### 4.2 文本映射正确性

构造短文本和 fake encoder，使每个 micro window 返回可预测分数，检查：

* 每个 context token 至少被一个窗口覆盖；
* 边界 token 不被映射到相邻文档/对话轮次；
* 重叠 token 的 mean/max 与手算值一致；
* Unicode、多空格、换行和标点不导致字符—token span 错位；
* 超过 external text length 上限时显式分窗，不静默截断；
* 批处理与逐条编码的 embedding/score 在允许误差内一致。

### 4.3 视觉 Grid 映射正确性

使用合成 score map：

* 单点高分：高分应只进入与该 cell 重叠的 Qwen token；
* 左右/上下梯度：映射后方向不翻转、不转置；
* 常数图：任意目标 grid 上仍为同一常数；
* 不整除 grid：面积加权结果与手算一致；
* resize + padding：padding 不贡献分数；
* 多图：第 $k$ 张图的高分不泄漏到第 $k+1$ 张图；
* 映射后 token 数严格等于 `layout.num_vision_tokens`。

对 3–5 个非 test 真实样本保存 score-map overlay，人工检查图像方向、crop/pad 和问题相关区域。可视化只是结构验证，不用 test answer 挑选能获得最高 Accuracy 的 checkpoint。

### 4.4 外部 Encoder 集成正确性

对固定 checkpoint 验证：

* text/image projection 后 hidden dimension 一致；
* 归一化后 norm 约为 1；
* raw cosine 落在 $[-1,1]$ 且没有 NaN/Inf；
* processor 实际 resize/crop/pad 与 metadata 一致；
* 对同一批输入，batch 大小改变不改变结果；
* dtype 回退时 FP32/BF16 排名高度一致；
* 所谓 dense feature 确实经过对应 projection，而不是未投影中间状态。

这些测试中需要真实权重的部分应标为 integration/GPU test，纯算法单元测试不应被大模型下载或 GPU 可见性阻塞。

### 4.5 联合剪枝回归

必须继续满足实验 002 的 invariant：

* `text_scores.shape == [N_t]`；
* `vision_scores.shape == [N_v]`；
* chunk 完整分割候选 token；
* `text_budget + vision_budget == B`；
* 每个 chunk budget 不超过容量；
* 保护 token 不被删除；
* 压缩序列长度等于 `protected_count + B`；
* 视觉 token 保留原始图像 ID 与 mRoPE $(t,h,w)$；
* 相同输入、artifact 与 seed 产生相同 plan。

$\rho=1$ 是硬回归：即使 external scorer 仍被运行，Aligned 条件的最终 keep mask、生成 token 和评分必须与 No Pruning 一致。性能测量时 No Pruning 不应计入无意义的 external scoring 开销，而剪枝条件必须计入。

### 4.6 分数及预算行为验证

在 calibration 和 test 上分别报告：

* raw text/image score 分布；
* calibrated text/image score 分布；
* 两种模态的 0/1 附近饱和率；
* $R_t,R_v$、$\alpha_t,\alpha_v$和 $B_t,B_v$ 分布；
* 完整保留某个模态的样本数；
* raw 与 calibrated 条件的预算改变数；
* calibration-to-test drift，并按候选 token 数和问题类型分桶。

校准后两个模态的全局边际分布接近是 CDF 方法的数学结果，不是算法有效的充分证据。更重要的是：

* 同模态排名保持；
* 动态预算不再因原始数值尺度长期饱和；
* 与问题明确相关的文本和视觉区域保持高排名；
* 最终答案质量高于 random/fixed/raw Qwen 主对照。

### 4.7 评测与性能验证

任务结果同时保留：

* 已有 MileBench/MMCoQA 口径；
* robust adapter 口径；
* 逐样本原始 prediction、answer 和两种判定。

主比较使用相同 test 样本和严格总预算，报告 Accuracy、与 baseline 一致率、正确→错误/错误→正确、配对置信区间与精确配对检验。Random Joint 使用 3 个 seed 报告均值和标准差。

性能计时至少拆分：

```text
CPU image/text preprocessing
external text encoding
external vision encoding
grid mapping + CDF lookup
Qwen vision encoding
joint planning
LLM prefill
decode
end-to-end
```

GPU 阶段计时需要预热和 synchronize。主要部署口径是：No Pruning 不包含 external scorer，Aligned 剪枝包含 external scorer 的端到端耗时。可额外报告“已有 score cache”的离线研究口径，但不能用它取代主口径。

## 5. Risk 与 fallback

| 风险 | 表现 | 优先处理 | Fallback |
| --- | --- | --- | --- |
| checkpoint 只对齐 pooled image | patch heatmap 无局部意义 | 检查 projection/pooling 代码与局部样例 | 换 dense retrieval/grounding encoder；或 region-crop 上限 |
| text hidden token 未被对齐 | token cosine 不可靠 | 使用 pooled micro-window 回填 | 将文本选择粒度改为整 chunk |
| external crop 丢失原图边缘 | Qwen 边缘 token 无 score | 优先 resize+pad 的 processor/checkpoint | 对原图分 tile 打分后融合 |
| grid 方向或坐标错配 | 高分区域翻转/偏移 | 合成梯度和 overlay 测试 | 暂时上调为整图/大区域 score |
| CDF 被长样本主导 | 校准随 token 数偏移 | 每样本每模态等权 | 每样本等量 reservoir sampling |
| calibration/test 分布偏移 | 分位数在 test 饱和 | 增大并分层构造 calibration set | 报告 drift，改用冻结 z-score/robust scaling 消融 |
| CDF 抹去真实全局模态差异 | 预算过度平衡 | 对比 uncalibrated 并检查样本级模态依赖 | 使用有条件/类别校准，或进入 learned utility |
| 相关性不等于 utility | score map 合理但答案不改善 | 与 random/fixed 做配对失败分析 | 进入实验 004 学习 deletion utility |
| external scorer 增加太多延迟 | prefill 收益被抵消 | batch、缓存、更小 scorer、长序列条件启用 | 离线 score cache；或放弃该路线 |
| 双模型常驻造成 OOM | 模型加载/峰值失败 | 减小 external batch、及时释放 dense feature | scorer 离线预计算后卸载，再加载 Qwen |
| 多卡 shard 重复/缺失 | 汇总样本数异常 | 独立 JSONL 和唯一 condition key | 单卡断点续跑 |
| 新 scorer 引入额外超参 | 在 50 test 上过拟合 | 预注册主配置，只在 calibration/validation 做结构检查 | 保留单一主配置，消融另行报告 |

其中“使用更小 encoder”和“只对长输入启用”会改变部署策略，必须作为独立条件报告，不能在单个方法内按结果静默切换。

## 6. 时间与显存分析

### 6.1 复用实验 002 的已知部分

实验 002 在当时的 4 张 RTX A6000 上已证明：

* Qwen2.5-VL-7B-Instruct BF16 可在单卡稳定工作；
* FlashAttention 2 实际被选中；
* 50 样本、950 条生成记录可使用 4 卡样本并行快速完成；
* baseline 平均完整输入约 2,646 tokens；
* Qwen 视觉编码平均约 272 ms；baseline prefill 约 372 ms；
* 峰值显存约 16.4 GB；
* 75% 剪枝的 prefill 收益会被剪枝开销抵消，所以新 scorer 必须计入端到端口径。

实验 003 的 Qwen encode、预算、压缩和 decode 都应复用该路径。新增成本只应来自 external text/image encoding、grid mapping 和 CDF lookup。

### 6.2 Calibration 成本

对 $S_c$ 个 calibration 样本，每个样本只需要：

* 问题 text embedding 一次；
* 所有 context micro windows 的批量 text encoding；
* 每张图像的 vision encoding 一次；
* CPU 上的 raw score 汇总与 CDF 拟合。

这是一次性成本，不需要加载 Qwen 7B，除非为了取得与 Qwen tokenizer/grid 一致的 layout metadata。实现上应尽量让 layout 构造只依赖 processor 和已有数据逻辑，避免为 CDF 拟合长时间常驻 7B 模型。

总耗时主要取决于未固化的 external checkpoint、图像数、分辨率和微窗口数，因此在选型前不给出虚假的固定分钟数。正式拟合前必须在 5–10 个代表样本上实测吞吐，再用：

$$
T_{cal}\approx S_c\times\bar T_{external}/G,
$$

估算 $G$ 张可用 GPU 下的总时间。

### 6.3 Smoke test

建议第一轮只运行：

* 3–5 个非 test 样本的 dense alignment/grid overlay；
* 5 个 test 样本；
* baseline；
* 75% 的 raw Qwen、aligned uncalibrated、aligned calibrated；
* 一个 Random seed；
* `--verify-baseline`。

该 smoke 的主要目标是验证 scorer 输出、grid 映射、CDF artifact、严格预算和 100% 回归，不解读 5 样本 Accuracy。

在当前 Qwen 路径上，新增 external encoding 每个样本只执行一次并在条件间复用。因此条件数增加不会线性重复 scorer 成本，但会线性增加 Qwen 生成次数。

### 6.4 正式 50 样本实验

若每个保留率包含：

* Fixed Joint；
* Raw Qwen Joint；
* Aligned Uncalibrated；
* Aligned Calibrated；
* Random Joint $\times3$；

再加一次 baseline，则每个样本有：

$$
1+3\times(1+1+1+1+3)=22
$$

个生成条件，总计：

$$
50\times22=1100
$$

条生成记录。

相对实验 002 的 950 条记录，Qwen 生成次数约增加 15.8%，另加 50 个样本的 external scoring。4 卡样本并行仍是首选，但具体 wall time 应使用 smoke profile 外推，不直接假设与实验 002 按比例一致。

如果只关心主比较，可在不改变预注册结论的前提下将 25% 作为后续压力测试，先运行 75%/50%。这会减少首轮生成量，但必须在运行前确定，不能根据已看到的 25% 结果选择是否报告。

### 6.5 显存与存储

Qwen BF16 在实验 002 的峰值约 16.4 GB。external encoder 的额外显存包括：

* scorer 权重；
* text/image batch activation；
* dense patch feature；
* 映射前的 score map。

对 48 GB A6000，中小型 encoder 与 Qwen 同卡常驻大概率可行，但这不能在 checkpoint 未固化前当作已验证事实。需在 smoke 中分别记录：

```text
Qwen only peak
external encoder only peak
both resident peak
external text batch peak
external image batch peak
full sample end-to-end peak
```

不应缓存所有样本的 GPU dense feature。如果每个候选只保存 raw/calibrated FP32 scalar，50 样本的 score cache 通常很小；而缓存所有 patch embedding 会放大存储和版本管理成本，首版没有必要。

### 6.6 推荐的优化顺序

1. **按样本复用外部 score**
   只计算一次 raw/calibrated relevance，复用给所有 retention 和对照。
2. **批量文本微窗口**
   先 profile batch 大小，再用 length bucketing 减少 padding。
3. **批量多图编码**
   仅在 processor 能正确保留每图几何 metadata 时启用。
4. **向量化 grid 映射和 CDF 查表**
   面积权重可按 grid shape 缓存，CDF 使用 `searchsorted`，避免 Python token 循环。
5. **分片多 GPU 样本并行**
   每卡独立模型和输出 shard，检查 seed、去重和失败汇总。
6. **双模型高效 attention backend**
   Qwen 复用 FlashAttention 2；external encoder 按兼容性选择 FlashAttention/SDPA，与 eager 做最小正确性对照。
7. **可选离线 score cache**
   当同一 encoder 需要重复运行多个消融时启用；主端到端性能仍计入在线 scorer 成本。
8. **最后才评估 `torch.compile`**
   只在 profile 证明 encoder 计算是主瓶颈、shape 足够稳定且编译预热可被全量运行摊销时使用。

选型、安装或下载 external checkpoint 是后续实现阶段的独立决策。在 checkpoint 未固化前，本文的时间和显存部分只提供测量方法与资源上界检查，不宣称尚未实测的加速收益。
