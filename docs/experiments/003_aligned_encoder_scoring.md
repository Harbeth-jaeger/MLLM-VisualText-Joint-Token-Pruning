# 003 · Joint Pruning with an External Aligned Encoder

> 状态：正式实验已完成。
> 固定 50 样本、22 个条件的主实验于 2026-10-19 完成；本文同时保留预注册设计、完整性边界和正式结果。

## 1. Objective

实验 002 直接比较 Qwen2.5-VL projector 后的视觉 token 与 LLM 文本 token embedding，观察到严重的文本偏置：文本模态分数均值为 0.881，视觉模态分数均值为 0.083。这两类特征虽然具有相同 hidden size，但并未被当作可直接比较余弦相似度的共同度量空间训练。

本实验引入一个已经进行图文对齐预训练的外部 encoder，仅用于估计问题与视觉区域、文本 chunk 之间的相关性，不替换、微调或修改 Qwen2.5-VL 的视觉编码器与生成路径。

本轮回答以下问题：

1. 外部对齐空间是否能消除实验 002 的数量级文本偏置；
2. 经过按模态校准后，视觉和文本分数是否能支持有效的动态预算分配；
3. 在相同严格总预算下，新评分是否稳定优于实验 002 的 raw Qwen score、固定分配和随机选择；
4. 外部 encoder 的时间、显存与预处理开销是否能被 LLM prefill 的加速抵消。

本实验的核心假设是：实验 002 的主要缺陷来自评分空间不可比，而不是 chunk 化、统一总预算或动态分配思路本身。

## 2. Setup

### 2.1 数据与划分

主评测仍使用 **MileBench / MMCoQA**，并保留实验 002 的固定 50 样本作为 test set，以支持逐样本配对比较。

新增一个与这 50 个样本 ID 完全不相交的 calibration set，只用于估计不同模态分数的位置与尺度参数。calibration set 不用于挑选能够提高 test Accuracy 的配置，也不允许根据 test 答案反复调整校准方法。

| 数据部分 | 用途 | 是否使用答案 |
| --- | --- | ---: |
| Calibration | 拟合分模态分数分布，检查数值稳定性 | 否 |
| Test | 与实验 002 进行配对评测 | 仅用于最终评分 |

### 2.2 模型

| 角色 | 模型 | 权重状态 |
| --- | --- | --- |
| 答案模型 | Qwen2.5-VL-7B-Instruct | 完全冻结 |
| 打分模型 | CLIP/SigLIP 系列图文对齐 encoder | 完全冻结 |

具体外部 encoder 需在实现前固化到配置文件，记录精确的 model ID、revision、processor 配置、输入分辨率、最大文本长度和输出层。选择时必须确认：

- 图像和文本表示由同一对比目标对齐；
- 能稳定取得 patch-level 视觉特征，而不只是全图 pooled embedding；
- 视觉 patch 特征确实经过与文本空间匹配的 projection；
- 允许 BF16/FP16 推理并与当前 PyTorch/CUDA 环境兼容。

如果某个 checkpoint 只对齐全局 pooled image embedding，不应默认其中间 patch token 也具有可靠的 dense alignment；此时应更换 checkpoint，或把“区域 crop 重复编码”作为低效 fallback 单独记录。

### 2.3 实验条件

统一使用候选 token 总保留率：

$$
\rho\in\{1.00,0.75,0.50,0.25\}.
$$

25% 作为压力测试；75% 和 50% 为主比较。根据前期结论，不再把无法严格满足联合总预算的 visual-only/text-only 作为主对照。

| 方法 | 预算 | 打分方式 | 目的 |
| --- | --- | --- | --- |
| No Pruning | 100% | 无 | 性能与效率基线 |
| Random Joint | 严格统一 | 随机，3 seeds | 检验 selector 是否优于随机 |
| Fixed Joint | 严格统一 | 固定模态比例 | 检验动态分配价值 |
| Raw Qwen Joint | 严格统一 | 实验 002 score | 隔离打分空间的影响 |
| Aligned, uncalibrated | 严格统一 | 外部 encoder raw cosine | 检验对齐空间本身 |
| **Aligned, calibrated** | 严格统一 | 外部 encoder + 分模态校准 | 主方法 |

Raw Qwen Joint 必须使用与实验 002 相同的公式与参数，不应为迎合新实验重新调参。除 Random Joint 外的推理应使用确定性解码和固定 seed。

### 2.4 评价与记录

任务性能同时保留两种口径：

- MileBench 原始/已有指标，用于与历史结果对齐；
- 新增的 robust evaluation adapter，处理大小写、标点、冠词、数字与英文数词等等价形式。

新 adapter 不覆盖或改写原始评测结果。两种口径必须并列报告，并保存逐样本判定。

除 Accuracy 外，至少记录：

- 与 baseline 的答案一致率、正确→错误和错误→正确数；
- 文本/视觉 raw score 及 calibrated score 的均值、方差、分位数和饱和率；
- 每个样本的模态预算、实际保留数和严格预算 invariant；
- 视觉 patch 到 Qwen visual token 的坐标映射信息；
- 外部 encoder、校准、剪枝、Qwen 视觉编码、prefill 和 decode 的分阶段耗时；
- 端到端耗时、峰值显存和实际 attention backend。

## 3. Method

### 3.1 冻结的旁路 scorer

原始 Qwen 生成路径保持不变：

```text
图像/文本
├─ Qwen2.5-VL 路径 → 应用 pruning plan → 生成答案
└─ 外部对齐 encoder → 产生 score ──────┘
```

该设计保证实验只替换 selector，不把视觉 encoder 替换对生成质量的影响混入结果。

### 3.2 文本 Chunk 打分

沿用实验 002 的文本 chunk 边界。对问题 $q$ 和第 $j$ 个 chunk $c_j$ 使用外部 text encoder：

$$
u_q=\operatorname{norm}(E_T(q)),\qquad
u_j^t=\operatorname{norm}(E_T(c_j)).
$$

原始文本相关性为：

$$
r_j^t=(u_j^t)^\top u_q.
$$

若 chunk 超过外部 encoder 的文本上限，必须继续分段后池化，不得静默截断。问题模板、前缀及 tokenizer 配置需固化。

### 3.3 视觉区域打分与坐标映射

外部 vision encoder 产生 patch grid：

$$
F^e\in\mathbb R^{H_e\times W_e\times D_e}.
$$

将每个 patch feature 投影到对齐空间并归一化：

$$
u_{a,b}^v=\operatorname{norm}(P_e(F^e_{a,b})),\qquad
r_{a,b}^v=(u_{a,b}^v)^\top u_q.
$$

Qwen 和外部 encoder 的 patch grid 通常不同。对于每张图像，将两者都表示为归一化图像坐标，使用 area pooling 将外部 score map 映射到 Qwen merger 后的视觉网格：

$$
R^q=\operatorname{AreaPool}(R^e,H_q,W_q).
$$

不允许跨图像插值或组成视觉 chunk。所有保留的 Qwen token 仍使用其原始 mRoPE $(t,h,w)$ 位置。

### 3.4 分模态校准

“处于同一对齐空间”并不自动保证 image-text cosine 和 text-text cosine 具有相同分布。因此在 calibration set 上分别估计两个经验分布 $F_v,F_t$，将 raw score 转换为 percentile：

$$
s_i^m=F_m(r_i^m),\qquad m\in\{v,t\}.
$$

$s_i^m\in[0,1]$ 表示该 token/chunk 在同模态背景分布中的相对相关程度。主方法使用经验 CDF/分位数校准，因为它对长尾和离群值比均值—方差 z-score 更稳健。z-score 仅作为消融。

校准不应强制每个样本给两种模态相同的总分，否则会抹去样本间真实的模态依赖差异。

### 3.5 预算与选择

校准后的 token score 代入实验 002 已实现的：

```text
token score
  → chunk value
  → modality value
  → strict joint budget
  → within-modality chunk budget
  → token selection
  → sequence reconstruction with original mRoPE
```

除 selector 外，chunk 划分、总预算、最低预算、整数分配、序列重建和解码配置必须与实验 002 保持一致，确保差异可归因于 score。

### 3.6 高效执行

- Qwen 和外部 encoder 均使用 `eval()`、inference mode 和兼容的 BF16/FP16；
- 一张图像的外部视觉特征只计算一次，对不同 retention 和对照共享缓存；
- 问题和文本 chunk 批量编码，按长度 bucketing，避免无效 padding；
- 多卡首选样本并行，每个 worker 保持独立输出文件，最后检查重复和缺失；
- 优先复用现有 FlashAttention 2；若外部 encoder 不兼容，仅对其回退 SDPA，不影响 Qwen 后端。

## 4. Results & Analysis & Conclusion

### 4.1 完整性门槛

在解读 Accuracy 前，必须同时满足：

1. 100% retention 与 No Pruning 的输入布局、生成 token 和评分一致；
2. 所有剪枝条件均严格保留 $B=\lfloor\rho(N_t+N_v)\rfloor$ 个候选 token；
3. 图像边界、Qwen grid、外部 grid 和 mRoPE 坐标映射无跨图、越界或错序；
4. calibration/test 样本 ID 不相交；
5. 每个条件无重复、缺失或静默失败样本。

### 4.2 分数可比性判定

不以“两种模态均值完全相等”作为成功标准。重点检查：

- calibrated score 是否不再出现数量级差异或大量饱和；
- 模态预算是否不再在绝大多数样本上退化为“先保留所有文本”；
- 问题中存在明确文本或视觉线索时，相应区域的分数排名是否合理；
- 分数分布在 calibration 与 test 之间是否出现明显 drift。

### 4.3 主结果判定

主比较是 **Aligned, calibrated vs. Random Joint / Fixed Joint / Raw Qwen Joint**，不是与不满足总预算的单模态条件比较。

若同时观察到以下结果，则支持继续该路线：

1. 在 75% 和 50% 两个主预算上，主方法都高于 Random Joint 的三种子均值，而非只在单一预算偶然占优；
2. 相对 Raw Qwen Joint 呈现一致的配对改善，且改善不是评测器形式差异导致；
3. 相对 Fixed Joint 有正向信号，说明 sample-wise 预算分配产生了价值；
4. 外部打分开销计入后，至少在某个不明显降低 Accuracy 的预算上获得端到端加速。

小样本结果必须报告配对胜负数、置信区间和配对检验，不仅比较 Accuracy 点估计。若只修正了分数分布，但未超过 Random Joint，则结论应是“跨模态标度问题被缓解，但语义相关性仍不足以作为 pruning utility”。

### 4.4 预期风险与 fallback

| 风险 | 影响 | Fallback |
| --- | --- | --- |
| patch feature 未经 dense alignment | 视觉局部分数不可靠 | 换用支持 dense retrieval 的 checkpoint，或用 region crop 编码验证上限 |
| 外部与 Qwen grid 不一致 | 细粒度区域错配 | 用面积池化、保留坐标可视化，必要时改为更大视觉 chunk |
| text-text 与 image-text 仍不同分布 | 动态预算再次偏移 | 使用独立 calibration set 上的 CDF/z-score，不在 test 上调参 |
| 相关性不等于回答效用 | 分数合理但 Accuracy 无提升 | 进入实验 004，直接学习 marginal utility |
| 外部 encoder 开销过大 | 端到端无加速 | 特征缓存、batch、更小 encoder，或只在长序列上启用 |

### 4.5 结果完整性与运行情况

正式实验使用 GPU 6、7 做两路样本并行，于 2026-10-19 10:17:11 CST 启动，10:23:36 CST 完成，墙钟为 **6 分 25 秒**。两个 shard 各处理 25 个样本；Qwen 和外部 encoder 均为 BF16，Qwen 实际 attention backend 为 FlashAttention 2。

| 检查项 | 预期 | 实际 | 状态 |
| --- | ---: | ---: | --- |
| Test 样本 | 50 | 50 | 完整 |
| 条件数 | 22 | 22 | 完整 |
| 逐样本记录 | $50\times22=1100$ | 1100 | 完整 |
| 唯一 `(sample, method, retention, seed)` | 1100 | 1100 | 无重复 |
| 严格预算偏差 | 全部为 0 | 全部为 0 | 通过 |
| Attention backend | FlashAttention 2 | FlashAttention 2 | 通过 |
| NaN / OOM / 静默失败 | 0 | 0 | 通过 |

calibration artifact 来自 50 个无答案样本，记录了 15,144 个文本候选和 95,586 个视觉候选；与固定 50 test 的 sample ID 无交集。正式结果文件约 82.2 MB，summary 和两个 shard 均正常落盘。

有一项预注册完整性门槛需要单独说明：smoke 中 100% retention 的候选集合、输入长度和首个生成位置 logits 与 Transformers 原生 `generate()` 完全一致，但 sample 110 在后续 BF16 greedy decode 中出现 token-level 分叉：自定义路径不输出答案引号，原生路径输出引号，两者均得到正确答案 `A Midsummer Night's Dream`。因此没有满足“所有样本逐 token 完全一致”的最强回归标准。正式实验的所有方法都使用同一个自定义 mRoPE decoder，方法间比较口径一致，但 No Pruning 不应表述为与当前 Transformers `generate()` 已证明逐位等价。

### 4.6 分数空间与校准效果

下表报告固定 test 上全部 token 的总体分布：

| 分数 | Text mean ± std | Vision mean ± std | 观察 |
| --- | ---: | ---: | --- |
| Raw aligned cosine | 0.713 ± 0.087 | 0.231 ± 0.032 | 仍存在明显跨模态尺度差异 |
| Calibrated percentile | 0.546 ± 0.277 | 0.510 ± 0.293 | 两种模态已进入近似可比的 $[0,1]$ 分位数尺度 |

校准后的最大值饱和比例很低：text 为 0.32%，vision 为 0.009%；没有出现大面积压到 0 或 1 的退化。test text 均值略高于 0.5，说明存在轻微 calibration/test drift，但远小于 raw cosine 的模态差异。

校准也明显改变了模态预算。平均保留 token 数如下：

| Retention | Raw/Uncalibrated Text | Raw/Uncalibrated Vision | Calibrated Text | Calibrated Vision |
| ---: | ---: | ---: | ---: | ---: |
| 75% | 454 | 1,468 | 388 | 1,533 |
| 50% | 434 | 847 | 336 | 945 |
| 25% | 349 | 291 | 241 | 399 |

因此，实验的第一个问题可以肯定回答：**外部 aligned space 本身仍有 text–vision raw-score 偏移，但 empirical CDF 成功消除了数量级不可比，并避免预算退化为优先保留几乎全部文本。**

这里有一个实现口径需要澄清：`Random Joint` 只随机化模态内部的 token/chunk 分数，模态总预算仍使用当前传入的 aligned-calibrated modality score。因此它与 Aligned Calibrated 在每个样本上的 Text/Vision 总预算完全相同；二者比较隔离的是“同一模态预算下，aligned 排序是否优于随机排序”，不是一个连模态预算也完全随机的无信息基线。

### 4.7 下游主结果

固定 50 test 样本上的 Accuracy：

| 方法 | 75% | 50% | 25% |
| --- | ---: | ---: | ---: |
| Fixed Joint | 60% | 50% | 48% |
| Random Joint，3-seed mean ± std | 63.3% ± 2.5% | 53.3% ± 2.5% | 41.3% ± 0.9% |
| Raw Qwen Joint | **68%** | **58%** | **54%** |
| Aligned Uncalibrated | 60% | **58%** | 48% |
| **Aligned Calibrated** | 62% | 48% | 46% |

No Pruning Accuracy 为 70%。对应 Robust Accuracy：

| 方法 | 75% | 50% | 25% |
| --- | ---: | ---: | ---: |
| Fixed Joint | 62% | 52% | 50% |
| Random Joint，3-seed mean ± std | 64.7% ± 1.9% | 54.7% ± 3.4% | 42.7% ± 0.9% |
| Raw Qwen Joint | **68%** | 58% | **54%** |
| Aligned Uncalibrated | 62% | **60%** | 50% |
| **Aligned Calibrated** | 64% | 50% | 48% |

Aligned Calibrated 的 Accuracy 及 95% Wilson interval 分别为：

| Retention | Accuracy | 95% CI |
| ---: | ---: | ---: |
| 75% | 62% | 48.2%–74.1% |
| 50% | 48% | 34.8%–61.5% |
| 25% | 46% | 33.0%–59.6% |

配对结果进一步说明没有稳定收益：

- 75% 下，Aligned Calibrated 相对 Raw Qwen 为 1 个由错变对、4 个由对变错、45 个不变，exact McNemar $p=0.375$；相对三个 Random seed 分别为 2/4、2/3、3/2，均不显著。
- 50% 下，相对 Raw Qwen 为 2/7，$p=0.180$；相对 Aligned Uncalibrated 为 3/8，$p=0.227$。这是主预算上方向最明确的负信号。
- 25% 下，相对 Raw Qwen 为 2/6，$p=0.289$；虽然高于 Random 的三 seed 均值 4.7 percentage points，但逐 seed 配对检验均不显著，而且仍低于 Fixed 和 Uncalibrated。

所以，校准解决了“分数能否跨模态比较”，却没有解决“相似度是否代表回答效用”。Aligned Calibrated 没有在 75% 和 50% 两个预注册主预算上超过 Random mean，更没有超过 Raw Qwen；主假设不成立。

### 4.8 速度、显存与离线成本

外部 encoder 每个样本只计算一次，并在全部 retention 和条件间复用。平均额外成本为：

| 阶段 | 平均耗时 |
| --- | ---: |
| External text encoding | 20.1 ms |
| External vision encoding | 33.6 ms |
| Grid mapping | 0.6 ms |
| **External scorer 合计** | **54.3 ms/sample** |

Aligned Calibrated 的部署侧 scorer + planner + prefill 为：

| Retention | External + planner + prefill | No Pruning prefill | Prefill-only 降幅 | 峰值显存 |
| ---: | ---: | ---: | ---: | ---: |
| 75% | 269.7 ms | 262.6 ms | 24.7% | 16.48 GB |
| 50% | 200.9 ms | 262.6 ms | 47.6% | 16.28 GB |
| 25% | 139.2 ms | 262.6 ms | 71.0% | 16.08 GB |

75% 的 prefill 节省约 64.8 ms，几乎被 54.3 ms external scoring 和 17.6 ms planning 抵消，部署侧反而比 No Pruning prefill 多约 7 ms。50% 和 25% 仍分别保留约 62 ms 和 123 ms 的 prefill-side 净节省，但 Accuracy 已明显下降。decode 均值约 431–503 ms，受生成内容和长度影响较大，不宜将变化直接归因于 pruning。

完整两卡实验墙钟 6 分 25 秒，峰值显存不超过 16.68 GB。工程上 external encoder 成本可控，问题主要不在速度或显存，而在选择质量。

### 4.9 结论

实验 003 得到一个清晰的拆分结论：

1. **跨模态标度问题被解决。** empirical CDF 将 text/vision 分数变为近似可比的分位数，并实质改变了模态预算。
2. **下游 pruning 问题没有被解决。** Aligned Calibrated 在 75%/50%/25% 上分别为 62%/48%/46%，三个预算均低于 Raw Qwen 的 68%/58%/54%。
3. **失败不主要来自 external encoder 太慢。** 54 ms/sample 的额外打分成本可接受，50% 和 25% 仍有 prefill-side 净节省；真正瓶颈是相关性排序与答案 utility 错位。
4. **CDF 不是免费收益。** Uncalibrated 在 50% 和 25% 分别比 Calibrated 高 10 和 2 percentage points，说明强行修正模态尺度虽然让预算更平衡，却可能丢掉某些对当前数据有用的文本偏置。
5. **当前 CLIP patch/text similarity 不适合作为独立主 selector。** 它可以作为局部辅助特征或边界 token 选择器，但证据不支持直接替换 Raw Qwen scorer。

因此，本实验不支持继续扩大同一 Aligned Calibrated 方案。更准确的研究结论是：**“对齐空间 + 分模态校准”修复了可比性，却没有证明 semantic relevance 等价于 deletion utility。**

## 5. Next

1. 不扩大当前 CLIP + CDF 主 selector；优先对 Aligned Calibrated 相对 Raw Qwen 的 50% 预算 2/7 paired case 做区域和 chunk 可视化，确认失败来自 dense patch alignment、文本窗口还是 relevance/utility 错位。
2. 后续若继续使用 external encoder，将其定位为辅助组件：提供 scorer similarity 特征，或只负责边界 chunk 内 token 选择，而不是独立决定所有 chunk 的去留。
3. 若要重新挑战独立 aligned selector，应换用明确支持 dense retrieval/grounding 的 checkpoint，并先在独立 validation 上验证局部区域召回，再进入 50 test。
4. 下一次随机对照需区分两种口径：`random within calibrated modality budget` 与 `fully random joint budget + selection`，避免把模态预算信息混入“无信息”基线。
5. 在后续正式实验前处理 100% retention 与当前 Transformers 原生 `generate()` 的 token-level decode 差异，或把基线定义明确固定为项目自定义 decoder，并冻结软件版本。

