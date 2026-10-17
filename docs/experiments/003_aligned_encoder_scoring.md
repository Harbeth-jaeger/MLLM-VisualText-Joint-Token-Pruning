# 003 · Joint Pruning with an External Aligned Encoder

> 状态：待实现、待实验。
> 本文预先记录实验目的、方法、评价口径与结果判定标准；不包含尚未运行的结果。

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

## 5. Next

1. 先用 3–5 个非 test 样本验证 patch feature 的投影、grid 映射和显著区域可视化；
2. 固化 external encoder checkpoint 与 calibration split，再实现缓存和多卡样本并行；
3. 先运行 5 样本的 100% 回归与严格预算 smoke test；
4. 通过后运行固定 50 样本，保留 raw score，使任何校准结果均可重算；
5. 若 aligned score 稳定优于 random，再扩大数据和 benchmark；若仅改善分布却不改善答案，直接转入实验 004。

