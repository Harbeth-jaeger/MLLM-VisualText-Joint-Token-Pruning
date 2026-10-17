建议把实验 002 实现为两层：上层是与模型无关的"分组—评分—预算—选择"算法，下层是 Qwen2.5-VL 专用适配器，负责提取 projector 后视觉 embedding、保留 mRoPE 位置并执行压缩后的 greedy decode。

这里有一个必须提前解决的关键点：Qwen2.5-VL 不能简单删除 `<|image_pad|>` 后继续把原始 `image_grid_thw` 传回模型。官方实现默认每张图仍是完整规则网格，会据此检查视觉 token 数量并生成二维位置编码。任意视觉 token 剪枝后需要手动重构 embedding 和 position IDs，否则即使程序能运行，空间位置也可能是错的。

## 1. Problem

### 1.1 要解决的问题

给定一个 MMCoQA 样本：

* 系统/任务指令；
* 多轮文本上下文 $T_c$；
* 当前问题 $T_q$；
* 多张图片经过 vision encoder 和 merger 后的视觉 token $V$；

在统一候选预算

$$
B=\left\lfloor\rho(N_t+N_v)\right\rfloor
$$

下，从 $T_c\cup V$ 中保留恰好 $B$ 个 token，同时：

* system、问题、生成提示和特殊 token 始终保留；
* 每个样本动态决定文本和视觉预算；
* 保留 token 的原始顺序和视觉空间位置；
* 不训练 selector；
* 尽量保持答案正确率；
* 真正缩短 LLM prefill 序列和 KV cache。

最终压缩后的总序列长度为：

$$
L'=N_{\mathrm{protected}}+B
$$

需要同时记录"候选 token 保留率"和"完整输入保留率"，避免只报告 $B/(N_t+N_v)$ 却忽略不可剪 token。

### 1.2 建议的核心接口

模型无关算法：

```
plan_joint_pruning(
    layout: CandidateLayout,
    text_embeddings: Tensor[N_t, D],
    vision_embeddings: Tensor[N_v, D],
    query_embeddings: Tensor[N_q, D],
    config: JointPruningConfig,
    generator: torch.Generator | None = None,
) -> PruningPlan
```

主要输出：

```
PruningPlan
├── total_budget
├── text_budget
├── vision_budget
├── selected_text_indices
├── selected_vision_indices
├── text_chunks[]
├── vision_chunks[]
├── token_scores
├── chunk_scores
├── modality_scores
├── overflow/reallocation statistics
└── diagnostics
```

Qwen 适配接口：

```
encode_qwen_sample(
    model,
    processor,
    sample,
    image_dir,
) -> EncodedMultimodalSample
```

输出至少包含：

```
EncodedMultimodalSample
├── input_ids                    [1, L]
├── input_embeddings             [1, L, D]
├── full_position_ids            [3, 1, L]
├── attention_mask               [1, L]
├── text/query/vision/protected masks
├── vision image/grid/spatial coordinates
├── candidate-to-sequence mappings
└── original token statistics
```

压缩与生成：

```
compress_and_generate(
    model,
    encoded: EncodedMultimodalSample,
    plan: PruningPlan,
    generation_config,
) -> GenerationResult
```

实验层：

```
run_condition(
    sample,
    condition,
    retention,
    seed,
) -> ExperimentRecord
```

## 2. Algorithm

### 2.1 样本解析和 token 对齐

先构造完整、未剪枝的 Qwen 输入。

1. 读取任务指令、长上下文、最终问题和图片路径。
2. 将 `{image#k}`、`{table#k}` 替换为对应图片内容块。
3. 应用 Qwen chat template 和 processor。
4. 得到完整 `input_ids`、`pixel_values`、`image_grid_thw`。
5. 标记四类序列位置：
   * `protected`：模板、问题、特殊 token、视觉边界；
   * `text_candidate`：可剪文本上下文；
   * `vision_candidate`：图像内部的 `<|image_pad|>`；
   * `other`：不参与预算但保留的控制 token。

文本位置不能仅靠 token ID 判断，因为问题和上下文使用相同词表。建议在 prompt 构造阶段保留字符区间，再利用 fast tokenizer 的 offset mapping 映射到 token；同时用"最终问题 token 子序列位于末尾"的检查作为第二重验证。

### 2.2 一次性提取特征

文本和问题：

```
H_t = embedding_layer(text_candidate_ids)   [N_t, D]
H_q = embedding_layer(question_ids)         [N_q, D]
```

视觉：

```
image_embeddings = model.get_image_features(
    pixel_values,
    image_grid_thw,
)
```

Qwen2.5-VL 的视觉输出已经经过 vision encoder 和 merger，维度与 LLM hidden size 相同。对每张图：

$$
N_{v,k}= \frac{T_kH_kW_k}{s^2},
$$

其中 `spatial_merge_size = 2`。

每张图的 LLM 视觉网格是：

$$
T_k\times(H_k/2)\times(W_k/2).
$$

需要为每个视觉 token 记录：

```
(image_index, temporal_index, row, column, sequence_index)
```

不能跨图片构造视觉 chunk。

### 2.3 Token 评分

所有 embedding 先进行 FP32 的 $L_2$ 归一化，避免 BF16 在范数和相似度计算中损失过大：

$$
\hat H = \operatorname{normalize}(H.float()).
$$

过滤问题中的：

* 特殊 token；
* 视觉占位符；
* padding；
* 纯标点。

相关性为：

$$
s_i=\max_j\hat h_i^\top\hat h_{q,j}.
$$

建议保留完整相似度统计，但只保存每个候选 token 的最大值及对应问题 token，不长期保存完整 $N_c\times N_q$ 矩阵。

### 2.4 Chunk 构造

文本 chunk：

* 不跨图片占位符、段落、文档或历史轮次；
* 优先按换行和句末标点切分；
* 长句按 token 顺序切到最多 64；
* 相邻短句在不超过 64 的前提下合并；
* 目标最小长度 16，但不强行跨结构边界合并。

视觉 chunk：

* 在 merger 后的 LLM 视觉网格上分组；
* 默认 $2\times2$；
* 图像边缘允许不完整窗口；
* 不跨图片；
* 暂不处理视频。

Chunk 分数：

$$
r_g=\operatorname{mean}\left( \operatorname{TopK}_{\lceil0.2|G_g|\rceil} \{s_i:i\in G_g\} \right).
$$

### 2.5 模态价值和动态预算

文本和视觉模态价值分别取本模态最高 20% chunk 的均值：

$$
R_m=\operatorname{mean}\left( \operatorname{TopK}_{\lceil0.2|\mathcal G_m|\rceil} \{r_g\} \right).
$$

使用数值稳定的 softmax：

$$
\alpha_m= \operatorname{softmax}\left( [R_t,R_v]/\tau_m \right)_m.
$$

然后加入最低预算：

$$
B_{\mathrm{free}}=B-b_t^{\min}-b_v^{\min}.
$$

$$
B_t=b_t^{\min} +\operatorname{round}(\alpha_tB_{\mathrm{free}}), \qquad B_v=B-B_t.
$$

必须执行容量限制和溢出再分配：

```
0 ≤ B_t ≤ N_t
0 ≤ B_v ≤ N_v
B_t + B_v = B
```

最低预算建议暂时使用"候选模态容量的一定比例"，而不是写死绝对 token 数，例如：

```
b_t_min = min(N_t, round(0.05 * B))
b_v_min = min(N_v, round(0.10 * B))
```

但具体值需要在独立 calibration 样本上只观察预算稳定性，不根据正式 50 条的最终 Accuracy 反复调参。

### 2.6 Chunk 内预算

对每个模态计算：

$$
p_g=\operatorname{softmax}(r_g/\tau_g).
$$

然后使用"带容量上限的整数 water filling"：

1. 计算连续预算 $p_gB_m$；
2. 向下取整；
3. 将达到容量的 chunk 固定；
4. 对未饱和 chunk 重新分配剩余预算；
5. 最后按小数余量分配整数；
6. 同分时按原始 chunk index 决定，保证确定性。

这个函数应作为纯函数实现并进行穷举式小规模测试。

### 2.7 Chunk 内选择

首版严格遵循实验 002 文档：

* chunk 内按 $s_i$ 选择 top-k；
* 最终按原序列位置排序；
* 不做 token merging；
* 不改变 embedding。

文本 top-k 可能破坏语法，这是已知风险。建议把"连续 span 扩展"保留为后续消融，不在首版同时引入。

### 2.8 压缩序列与 mRoPE

这是实现中最关键的一步。

先在未剪枝完整输入上调用 Qwen 的 `get_rope_index()`，得到：

```
full_position_ids: [3, 1, L]
```

随后对最终 keep mask 同时 gather：

```
compressed_input_ids
compressed_inputs_embeds
compressed_position_ids
compressed_attention_mask
```

这样：

* 文本 token 保留原始相对位置；
* 视觉 token 保留原始 $(t,h,w)$ 空间位置；
* 删除视觉 token 后不再伪造一个完整规则网格；
* 不再把剪枝后的视觉 token 交给标准 placeholder scatter；
* 问题位于输入末尾，因此 decode 起始位置仍可由最后保留位置可靠确定。

不能在剪枝后再次调用标准 `get_rope_index(image_grid_thw=原网格)`。

### 2.9 Greedy decode

建议首版实现一个受控的 batch-size-1 greedy decoder，而不是强行修改 Transformers 内部模型源码：

1. 用 `compressed_inputs_embeds`、显式 `compressed_position_ids` 完成一次 prefill；
2. 开启 `use_cache=True`；
3. 从最后一个位置 logits 取 `argmax`；
4. 逐 token decode；
5. 每一步：
   * 传入上一步生成 token；
   * 复用 KV cache；
   * 三个 mRoPE 维度使用相同递增文本位置；
   * 更新 attention mask 和 cache position；
   * 遇到任意合法 EOS 或达到 32 token 时停止。

必须先证明该 decoder 在 $\rho=1$ 时与原始 `model.generate(do_sample=False)` 输出一致。若不一致，不进入正式剪枝实验。

### 2.10 对照条件需要修正的一点

实验文档当前要求所有方法使用相同总预算，但 Text-only 在当前数据上通常不可行。

平均：

```
N_v ≈ 2104
N_t ≈ 448
```

当 $\rho=0.75$ 时：

$$
B\approx0.75(2104+448)=1914<N_v.
$$

仅保留全部视觉 token 已经超过总预算，因此 Text-only 无法满足统一预算。$\rho=0.50$ 和 $0.25$ 更不可能。

建议正式区分：

公平的统一预算主对照：

* No Pruning
* Visual-only
* Fixed Joint
* Random Joint × 3 seeds
* Proposed Joint

历史参考但不宣称预算公平：

* Experiment 001 Text-only 50%/25%
* 或只在满足 $B\ge N_v$ 的样本和保留率上加入 Text-only

否则"所有方法相同总预算"的结论不成立。

## 3. 文件分工

建议新增或调整以下文件。


| 文件                                 | 作用                                                              | 输入                          | 输出                        |
| ------------------------------------ | ----------------------------------------------------------------- | ----------------------------- | --------------------------- |
| `src/mllm_prune/mmcoqa.py`           | 从实验 001 提取共享的数据解析、prompt、评分逻辑                   | MMCoQA sample、图片目录       | messages、标准化评分        |
| `src/mllm_prune/joint_pruning.py`    | 模型无关的分组、评分、预算和选择                                  | embedding、layout、配置       | `PruningPlan`               |
| `src/mllm_prune/qwen25vl_pruning.py` | Qwen 特征提取、视觉网格映射、mRoPE gather、压缩 prefill 和 decode | processor 输出、`PruningPlan` | prediction、tensor/时间统计 |
| `scripts/run_joint_pruning.py`       | 实验编排、断点续跑、条件循环和 JSONL 写出                         | `.env`、config、固定子集      | 逐样本 JSONL                |
| `scripts/summarize_joint_pruning.py` | 完整性、Accuracy、配对变化、预算、时间和显存汇总                  | JSONL                         | JSON/CSV summary            |
| `configs/experiment_002.yaml`        | 可版本化实验条件与超参数                                          | 人工配置                      | runner 配置                 |
| `tests/test_joint_pruning.py`        | 纯算法单元测试                                                    | 人工小张量                    | 预算/选择断言               |
| `tests/test_qwen25vl_layout.py`      | tokenizer、占位符、视觉网格和 mRoPE 对齐测试                      | 一个真实样本或小 fixture      | shape/index 断言            |
| `tests/test_qwen25vl_decode.py`      | $\rho=1$ 回归与短 decode 测试                                     | 1 个真实样本                  | logits/输出一致性           |

不建议直接修改 Transformers 安装目录，也不建议把所有逻辑塞进一个大型 runner。

核心配置建议包含：

```
retentions: [1.0, 0.75, 0.5, 0.25]
text_chunk_min_tokens: 16
text_chunk_max_tokens: 64
vision_window: 2
token_value_top_fraction: 0.2
modality_value_top_fraction: 0.2
modality_temperature: ...
chunk_temperature: ...
text_min_budget_fraction: ...
vision_min_budget_fraction: ...
random_seeds: [0, 1, 2]
max_new_tokens: 32
attention_backend: flash_attention_2
```

## 4. Verification

### 4.1 纯算法正确性

预算函数必须验证：

* 任意合法输入下 `Bt + Bv == B`；
* 每个 chunk 分配不超过容量；
* chunk 预算和等于模态预算；
* 无空模态、单 token、极小预算、全部容量饱和等边界错误；
* 相同输入与随机种子输出完全一致；
* tie-break 稳定；
* 输入顺序不被 top-k 打乱。

可以对小容量进行穷举测试，而不只是几个手写例子。

### 4.2 结构正确性

每个真实样本检查：

$$
N_v=\sum_k\frac{T_kH_kW_k}{s^2}
$$

并验证：

* 完整视觉 embedding 数等于原始 `<|image_pad|>` 数；
* 每个视觉 token 都有唯一的图片、行、列和序列位置；
* 不存在跨图片 chunk；
* protected token 全部保留；
* 压缩序列长度等于 `protected_count + B`；
* position IDs 与 keep mask 一一对应；
* attention mask 长度正确；
* embedding 中没有 NaN/Inf；
* 实际保留率与目标预算完全一致。

### 4.3 $\rho=1$ 回归是硬门槛

同一个样本、相同模型和解码配置下比较：

1. 现有 `model.generate()`；
2. 新的自定义 prefill/decode；
3. 新路径下的 Proposed Joint，$\rho=1$。

应验证：

* prefill 最后位置 logits 在允许的数值误差内接近；
* 首个生成 token 一致；
* 完整 greedy prediction 一致；
* 5 个 smoke 样本的评分与实验 001 baseline 一致。

如果 $\rho=1$ 都不能复现，不能通过"最终 Accuracy 恰好相同"来放行。

### 4.4 人工可预测测试

构造小 embedding：

* 某个文本 token 与问题完全相同；
* 某个视觉 token 与问题完全相同；
* 其他 token 正交或负相关。

检查：

* 高相关 token 被优先选择；
* 调高某一模态 $R_m$ 后其预算不下降；
* 降低 $\tau_m$ 后模态预算差距增大；
* 增加 chunk 分数后其预算不下降。

这些测试能验证算法含义，而不只是"没有报错"。

### 4.5 结果层面的合理性检查

正式结果至少检查：

* $\rho$ 下降时，总保留 token 必须单调下降；
* 相同总预算的所有主对照实际 token 数完全一致；
* Random Joint 三个种子报告均值和标准差；
* Proposed 应和 Random、Fixed、Visual-only 做逐样本配对；
* 报告正确→错误、错误→正确；
* 报告预算分配分布，而不只报告均值；
* 检查 $\alpha_t$ 是否塌缩到 0 或 1；
* 检查某种模态是否由于分数尺度偏差长期获得几乎全部预算；
* 分别报告 pruning、vision encoder、prefill、decode 和端到端时间。

50 个样本中一道题等于 2 percentage points，因此一题差异只能视作信号，不能作为稳定提升结论。可以补充 paired bootstrap 区间或 McNemar exact test，但不应夸大小样本显著性。

性能收益不能只用 token 数推断。合理信号应是：

* prefill 时间随完整输入长度下降；
* KV cache/峰值显存不增加；
* 剪枝开销显著小于节省的 LLM 时间；
* 端到端时间在预热和同步后确实下降。

## 5. Risk 与 fallback


| 风险                                | 表现                           | 优先处理                                         | Fallback                                            |
| ----------------------------------- | ------------------------------ | ------------------------------------------------ | --------------------------------------------------- |
| 视觉剪枝破坏 mRoPE                  | shape 错误或空间位置错误       | 完整输入先算 position IDs，再按 keep mask gather | 暂停视觉 token 级剪枝，只做整图/规则区域选择        |
| 自定义 decode 与`generate()` 不一致 | $\rho=1$ 输出变化              | 对齐 EOS、cache、position IDs、attention mask    | 改为 Qwen 子类适配`GenerationMixin`，不进入正式实验 |
| 文本/视觉余弦分数不可比             | 预算长期偏向一个模态           | 记录两种分数分布并在独立 calibration 样本检查    | Fixed Joint；或做无标签的模态尺度校准               |
| 输入 embedding 语义较浅             | 选中词面相似但推理无用的 token | 失败案例分析、与 random/fixed 对照               | 尝试浅层 hidden state scorer，作为后续消融          |
| 文本 token top-k 破坏语法           | 输出退化、残缺文本             | 保留原序并记录可读案例                           | chunk 全保留或连续 span 选择                        |
| 最低预算不合理                      | 某一模态被挤出或过度保护       | calibration 只看预算和数值稳定性                 | 固定最低比例或 Fixed Joint                          |
| 小 chunk softmax 偏置               | 大量 chunk 获得 0 token        | 带容量 water filling                             | chunk 全选式 knapsack                               |
| FlashAttention 不兼容自定义路径     | 运行错误或静默回退             | 确认实际后端并做同输入对照                       | 显式切换 SDPA                                       |
| 视觉编码重复计算                    | 全量实验时间过长               | 每个样本只编码一次，复用所有条件                 | 缓存 CPU BF16 embedding 或逐样本顺序运行            |
| 多进程写同一输出                    | JSONL 损坏或重复               | 每个 GPU 独立 shard                              | 单 GPU；结束后确定性合并                            |
| OOM                                 | 峰值突然增加                   | batch size 1、inference mode、及时释放中间张量   | SDPA、降低 smoke 图像分辨率；正式比较必须统一配置   |
| 50 样本方差过大                     | 一两题决定结论                 | 配对结果与置信区间                               | 扩展样本或加入 MMLongBench-Doc                      |

其中"降低图像分辨率"会改变视觉 token 数和研究设置，只能用于调试；不能只对某个实验条件使用。

## 6. 时间与显存分析

### 6.1 当前已有依据

实验 001 的 50 样本 baseline 实测：

* 平均输入 token：约 2,640；
* 平均视觉 token：约 2,104；
* 平均单次生成：约 1.21 秒；
* 峰值显存：约 16.36 GB。

这来自当时的 RTX A6000 48 GB、原 attention 后端和现有图像分辨率。FlashAttention 已安装但尚未在当前可见 GPU 上验证，因此不能直接假设新实验会更快多少。

### 6.2 Smoke test

若运行：

* 5 个样本；
* baseline 1 个条件；
* 3 个压缩率；
* 每个压缩率包括 Visual-only、Fixed、Proposed、Random × 3；

共：

$$
5\times(1+3\times6)=95
$$

次生成。

按实验 001 的生成速度，纯推理约 2 分钟。加上模型加载、首次 kernel、图像处理和调试日志，建议预计 3–8 分钟。

### 6.3 正式 50 样本主实验

同一条件网格为：

$$
50\times19=950
$$

次生成。

按 1.1–1.3 秒/次，纯生成约 17–21 分钟。考虑：

* 模型加载；
* 视觉编码；
* pruning；
* 文件 I/O；
* CUDA 预热；
* FlashAttention/自定义 decode 的实际差异；

单 GPU 建议保守预计 25–45 分钟。

如果再加入 Text-only、窗口消融、最低预算消融和温度 sweep，运行次数会快速翻倍，不应和主实验一次全部启动。

### 6.4 显存压力

单卡 batch size 1 时预计不会明显超过实验 001 的约 16.4 GB：

* 模型 BF16 权重约 16.6 GB 文件规模；
* 单样本视觉 embedding 大约：

$$
2104\times3584\times2\text{ bytes}\approx15\text{ MB};
$$

* 文本/问题相似度矩阵通常只有几 MB；
* 压缩后 LLM activation 和 KV cache 应下降。

因此一张 48 GB A6000 有充足余量。主要风险不是单样本 embedding，而是：

* 同一 GPU 上启动多个模型副本；
* 意外保留所有层 hidden states；
* 缓存所有 50 样本的 GPU embedding；
* 多进程重复加载模型。

### 6.5 推荐的优化顺序

1. **每个样本只做一次视觉编码**
   样本在外层，所有条件复用 projector 后视觉 embedding。这是最直接且不改变语义的优化。
2. **只保留当前样本的 GPU cache**
   处理完全部条件后释放。若需要跨进程重用，可缓存 CPU BF16，但首版没有必要。
3. **向量化相似度**
   用一次矩阵乘法：$S=\hat H_c\hat H_q^\top$，而不是 Python 双循环。
4. **FlashAttention 与 SDPA 对照后启用**
   先验证 $\rho=1$ 正确性，再比较端到端耗时。不能仅以安装成功作为启用依据。
5. **多 GPU 按样本分片**
   如果 `nvidia-smi` 显示多张明确可用 GPU，每卡一个模型进程，每个进程写独立 JSONL：

   ```
   results.rank0.jsonl
   results.rank1.jsonl
   ...
   ```

   最后按样本、条件和种子排序合并。独立样本几乎不需要 GPU 间通信，比模型并行更适合本实验。
6. **预处理与 GPU 重叠**
   CPU 可以预读下一样本图片，但先确认图像读取确实是瓶颈，再增加 worker/prefetch，避免复杂度无收益。
7. **避免全量 logits**
   prefill 只需要最后位置 logits，使用等价的 `logits_to_keep=1` 路径。
8. **先主实验，后消融**
   主链路通过后再运行 $w\in\{2,4\}$、最低预算和温度消融，避免错误实现放大为大量无效计算。

目前当前会话的 `nvidia-smi` 仍无法连接驱动，因此这些时间估计来自实验 001，而不是 FlashAttention 和新 decoder 的重新实测。正式编码完成后，应先恢复 GPU 可见性，再执行 1–5 样本的 correctness/benchmark smoke test。
