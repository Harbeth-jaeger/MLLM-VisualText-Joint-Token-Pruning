# 001 · Text-only Pruning 探究实验

> 归档说明：该实验属于早期探索，baseline 会经过文本分句和重组，现已确认不适合作为后续实验的严格基线。实验记录与历史输出保留用于追溯；专用算法、运行脚本、汇总脚本和测试已于 2026-10-14 从当前代码库移除，下文中的旧路径与命令仅表示当时的运行方式，不再可执行。

## 1. Objective

本实验是 visual-text token 联合剪枝研究的前置探究：固定图像、问题和模型，仅压缩输入中的上下文文本，检验文本侧是否存在可利用的冗余。

核心问题：

1. 能否删除一定比例的上下文文本，同时基本保持任务性能？
2. 根据问题相关性选择保留内容，是否优于随机选择和均匀选择？
3. 文本压缩能带来多大的整体输入缩减，以及实际效率收益？

正向结果支持进一步开展联合剪枝实验，但不能直接证明两种模态同时剪枝可行，也不能证明联合决策优于独立剪枝。

## 2. Setup

### 2.1 模型与数据

| 项目 | 设定 |
|---|---|
| 模型 | Qwen2.5-VL-7B-Instruct |
| 模型权重 | `.env` 中的 `QWEN_MODEL_PATH` |
| 数据集 | MileBench / MMCoQA（长文本、多图、自由回答） |
| 数据路径 | `.env` 中的 `MILEBENCH_ROOT` / `MMCOQA_IMAGE_DIR` |
| 正式实验规模 | 50 个样本，11 个条件，共 550 次生成 |
| 样本选择 | 在上下文不少于 128 words 的样本中，以 seed 42 随机抽取并固化 |
| 评价方式 | MMCoQA 的归一化答案子串匹配 Accuracy |

所选 50 个样本的图片均存在且可读取，没有样本因模型上下文长度被排除，也没有启用未记录的自动截断。

### 2.2 固定配置

| 参数 | 设定 |
|---|---|
| 解码方式 | Greedy decoding，`do_sample=False` |
| `max_new_tokens` | 32 |
| Batch size | 1 |
| 图像处理 | 所有组保留全部图像及原始顺序；`min_pixels=256×28×28`，`max_pixels=512×28×28` |
| Prompt template | MileBench task instruction + 原始图文交错 context；Qwen chat template |
| Tokenizer / Processor | Qwen2.5-VL 对应 tokenizer / processor，Transformers 4.57.6（默认 fast image processor） |
| Attention backend | `.env` 中的 `MLLM_ATTN_IMPLEMENTATION`；当前为 `flash_attention_2` |
| Chunk 大小 | 目标约 64 个文本 token，优先保持句子完整 |
| 主方法 scorer | BM25 |
| 文本保留率 | $r \in \{1.0, 0.5, 0.25\}$ |
| 随机对照种子 | 0、1、2 |
| 硬件与软件环境 | NVIDIA RTX A6000 48 GB（GPU 3）；driver 550.54.14；PyTorch 2.6.0+cu124；Transformers 4.57.6 |

输出长度上限固定，同时记录实际输出 token 数，避免将生成长度变化误认为输入剪枝的加速。

## 3. Method

### 3.1 输入表示 $X$

将输入表示为：

$$
X=[S,\operatorname{Interleave}(C,V),Q,F]
$$

其中：

- $S$：系统指令；
- $C$：可剪枝的上下文文本；
- $V$：图像输入及其对应的视觉 token；
- $Q$：最终问题及选项；
- $F$：输出格式要求。

`Interleave` 表示上下文文本与图像可能交错出现，实际排列遵循原始样本和模型模板。

本实验只修改 $C$。系统指令、图像、图像占位符、问题、选项及必要的格式标记全部保留。剪枝后保留内容仍按原始顺序排列。

### 3.2 Prune 对象：根据问题相关性选择上下文 chunk

先将可剪上下文划分为：

$$
C=\{C_1,C_2,\ldots,C_m\}
$$

划分规则：

- 每个 chunk 目标约 64 个文本 token；
- 优先按句子或段落边界切分；
- 不跨图像占位符；
- 图像编号、边界标记等必要结构信息单独保护。

使用问题及选项作为查询，通过 BM25 评估每个 chunk 的重要性：

$$
s_j=\operatorname{BM25}(Q,C_j)
$$

分数越高，表示该 chunk 与问题的词汇相关性越强。按分数从高到低选择保留内容，优先删除低分 chunk。

BM25 只是本轮的简单重要性近似，不能保证识别所有语义相关或图像相关信息。后续可用 embedding cosine 替换 scorer，作为扩展实验。

### 3.3 Prune 时机：进入模型前

在模型的 tokenizer / processor 构造最终输入之前，对原始上下文文本执行 chunk 选择，再重建图文输入。

流程：

1. 分离受保护内容和可剪上下文；
2. 划分、评分并选择 chunk；
3. 按原始位置恢复保留 chunk，保持图像顺序与对应关系；
4. 使用目标模型的 tokenizer / processor 构造输入；
5. 执行推理并评分。

本轮属于输入级文本压缩，不修改 Transformer 内部 hidden states、attention mask 或 KV cache。

### 3.4 Prune 程度：保留率 sweep

设原始可剪上下文包含 $N_C$ 个文本 token，目标保留率为 $r$，则保留预算为：

$$
B_r=\lfloor rN_C\rfloor
$$

首轮测试：

| 目标保留率 $r$ | 目标剪枝率 $1-r$ |
|---:|---:|
| 1.00 | 0% |
| 0.50 | 50% |
| 0.25 | 75% |

按重要性排序依次加入完整 chunk；若某个 chunk 超过剩余预算，则跳过并继续检查后续 chunk。最终按原始顺序拼接，不裁切 chunk 来凑预算。

由于 chunk 长度不同，且拼接后重新分词可能改变 token 数，实际保留率未必恰好等于目标值。因此记录：

$$
r_{\mathrm{actual}}
=
\frac{N_{C,\mathrm{kept}}}{N_C}
$$

同时分别报告：

$$
P_C=1-\frac{N_{C,\mathrm{kept}}}{N_C}
$$

$$
P_{\mathrm{all}}
=
1-\frac{N_{\mathrm{input,kept}}}
{N_{\mathrm{input,original}}}
$$

其中 $P_C$ 是可剪上下文的剪枝率，$P_{\mathrm{all}}$ 是整个多模态输入的缩减率。

本轮 sweep 用于观察性能随压缩程度的变化，不据此宣称获得最优阈值。后续若需选定固定保留率，应使用独立校准集选择，并在测试集验证。

### 3.5 对照组：改变 chunk 的选择方式

各剪枝组共用相同的 chunk 划分、保护规则和 token 预算，仅改变保留对象的选择方式。

| 方法 | Chunk 选择方式 | 用途 |
|---|---|---|
| No Pruning | 保留完整上下文 | 未剪枝 baseline |
| Question-aware BM25 | 按问题相关性从高到低选择 | 主方法 |
| Random | 随机打乱 chunk 后按预算选择 | 检验相关性评分是否有效 |
| Uniform | 在原始 chunk 序列中均匀分布地选择 | 检验均匀覆盖上下文的效果 |

Uniform 根据预算估算保留 chunk 数，在原始序列上等间隔选择；若超过预算，减少保留数量。记录实际保留率，避免预算偏差影响比较。

Random 使用 3 个种子，报告均值及标准差。这里随机和均匀选择的单位均为完整 chunk，与主方法保持一致。

| 方法 | 保留 100% | 保留 50% | 保留 25% |
|---|---:|---:|---:|
| No Pruning | ✓ | — | — |
| Question-aware BM25 | — | ✓ | ✓ |
| Random，3 seeds | — | ✓ | ✓ |
| Uniform | — | ✓ | ✓ |

## 4. Result & Analysis

### 4.1 性能与压缩结果

实验于 2026-10-07 完成。50 个样本各产生 11 条记录，共 550 条；完整性检查未发现缺失或重复。MMCoQA 使用归一化答案子串匹配 Accuracy，一道题对应 2 个百分点，因此小幅差异只作趋势判断。

$$
\Delta \mathrm{Score}
=
\mathrm{Score}_{\mathrm{pruned}}
-
\mathrm{Score}_{\mathrm{baseline}}
$$

答案一致率：

$$
\mathrm{Agreement}
=
\frac{1}{M}
\sum_{i=1}^{M}
\mathbb{1}
\left(
\hat y_i^{\mathrm{pruned}}
=
\hat y_i^{\mathrm{baseline}}
\right)
$$

使用同一套答案抽取和规范化规则计算一致率。一致不代表正确，因此必须与任务分数一起报告。

| 方法 | 目标保留率 | 实际上下文保留率 | 总输入缩减率 | Accuracy | 相对 baseline | 答案一致率 |
|---|---:|---:|---:|---:|---:|---:|
| No Pruning | 100% | 100.0% | 0.0% | 72.0% | 0.0 pp | 100.0% |
| BM25 | 50% | 47.4% | 8.8% | **74.0%** | **+2.0 pp** | 36.0% |
| BM25 | 25% | 22.9% | 13.0% | 60.0% | -12.0 pp | 22.0% |
| Random，均值 ± 标准差 | 50% | 47.2% ± 0.6% | 8.9% | 59.3% ± 4.6% | -12.7 pp | 33.3% ± 1.2% |
| Random，均值 ± 标准差 | 25% | 23.0% ± 0.3% | 13.0% | 55.3% ± 3.1% | -16.7 pp | 18.0% ± 6.0% |
| Uniform | 50% | 47.2% | 8.8% | 58.0% | -14.0 pp | 24.0% |
| Uniform | 25% | 22.8% | 13.0% | 50.0% | -22.0 pp | 22.0% |

这里的总输入缩减率按各条件的平均模型输入 token 相对 baseline 的变化计算。baseline 平均输入 2640 tokens，其中视觉 token 平均 2104，可剪上下文文本平均仅 448 tokens。因而即使删除约一半上下文文本，总输入也只减少约 8.8%；删除约四分之三上下文文本，总输入只减少约 13.0%。

主要观察：

- **文本有冗余，但必须选择性删除。** BM25 保留约 47% 上下文时，Accuracy 为 74%，与 72% baseline 基本持平；50 个样本不足以将 +2 pp 解释成真实提升，但至少没有看到明显性能损失。
- **激进裁剪开始伤害性能。** BM25 保留约 23% 时下降 12 pp，说明只保留四分之一文本对当前简单 scorer 已经过强。
- **相关性选择明显有用。** 在相同预算下，BM25 比 Random 均值高 14.7 pp（50% 档）和 4.7 pp（25% 档），比 Uniform 高 16 pp 和 10 pp。50% 档的信号尤其清楚。
- **输出文本很不稳定。** BM25 50% 虽保持 Accuracy，逐字符串答案一致率只有 36%。其中 baseline 与 BM25 同时答对 34 题、同时答错 11 题、baseline 对而 BM25 错 2 题、baseline 错而 BM25 对 3 题。生成式答案可发生表述变化，因此一致率只作辅助指标。

### 4.2 效率结果

在同一张 RTX A6000 上串行运行，模型生成前后调用 CUDA synchronize。当前实现记录的是 `model.generate` 的端到端耗时（包含 prefill 与 decode），没有将 prefill 单独拆出，因此不报告虚假的 prefill 数值。Random 为三个种子的均值。

| 方法 | 目标保留率 | 平均输入 tokens | 剪枝耗时 / ms | `generate` 耗时 / ms | 相对加速 | 峰值显存 / GB | 输出 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Pruning | 100% | 2640 | 4.64 | 1208 | — | 16.36 | 19.36 |
| BM25 | 50% | 2407 | 4.14 | 1087 | 10.0% | 16.33 | 16.48 |
| BM25 | 25% | 2296 | 4.09 | 1075 | 11.0% | 16.32 | 16.46 |
| Random | 50% | 2406 | 3.80 | 1103 | 8.7% | 16.33 | 17.00 |
| Random | 25% | 2297 | 3.70 | 1127 | 6.7% | 16.32 | 18.63 |
| Uniform | 50% | 2407 | 3.67 | 1130 | 6.4% | 16.33 | 18.16 |
| Uniform | 25% | 2296 | 3.68 | 1126 | 6.8% | 16.32 | 18.40 |

剪枝开销约 4 ms，相对约 1.1 s 的生成耗时很小。端到端耗时还受实际输出长度影响，所以本次 6%–11% 的变化只能视为粗略效率印象；显存峰值几乎不变。完整 550 次生成的累计模型推理时间约 10.3 分钟。

### 4.3 代表性案例

| Baseline | 剪枝后 | 检查重点 |
|---|---|---|
| 正确 | 正确 | 保留了哪些信息，删除了哪些冗余？ |
| 正确 | 错误 | 是否误删关键证据或破坏图文对应关系？ |
| 错误 | 正确 | 是否删除了干扰信息？ |
| 错误 | 错误 | 是否存在任务困难、模型能力不足或输入错误？ |

以 BM25 50% 为例：

- **正确 → 正确（sample 51）：** 标准答案 `Timo Glock`，baseline 与裁剪后均输出 `Timo Glock.`。
- **正确 → 错误（sample 100）：** 标准答案 `yes`；baseline 正确回答 Kyrie Irving 出演过 *Uncle Drew*，裁剪后却回答成 Stephen Curry 未出演。说明词面相关性仍可能保错人物或删掉关系证据。
- **错误 → 正确（sample 27）：** 标准答案 `Paul Menard`；baseline 生成冗长且没有给出人名，裁剪后直接回答正确。删除干扰信息有时能改善输出。
- **错误 → 错误（sample 199）：** 标准答案 `New England Patriots`，两组都回答了其他球队，属于当前模型或证据定位仍未解决的样本。

这些案例只用于帮助理解机制，不构成额外统计证据。

### 4.4 可复现文件

- 固定样本：`data/interim/experiment_001/mmcoqa_50.json`
- 原始逐次结果：`outputs/experiment_001/results.jsonl`
- 聚合结果：`outputs/experiment_001/summary.json`
- 样本准备：`scripts/prepare_mmcoqa_subset.py`
- 推理脚本：`scripts/run_text_pruning.py`
- 汇总脚本：`scripts/summarize_text_pruning.py`

关键运行命令：

```bash
source .env
PYTHONPATH=src:. "$MLLM_PRUNE_ENV_PATH/bin/python" scripts/run_text_pruning.py
```

数据、模型、子集和输出路径由 `.env` 统一提供，各 CLI 路径参数仍可用于临时覆盖。
FlashAttention 从源码编译时需复用现有 PyTorch/CUDA，安装依赖应使用
`pip install --no-build-isolation -r requirements.txt`。若特定调试或模型操作不兼容，可用
`--attn-implementation sdpa` 明确回退。

## 5. Conclusion

这次小实验给出的直观结论是：**text token 确实存在可利用的冗余，且“删哪些”比“删多少”更重要。** 在只保留约 47% 上下文文本时，BM25 将 Accuracy 从 72% 保持在 74%，而 Random 和 Uniform 已下降到约 59% 和 58%。这支持继续研究 question-aware token pruning，而不是无差别压缩。

但本实验也揭示了实际瓶颈：MMCoQA 在当前图像分辨率下平均约 2104 个视觉 token，可剪文本仅约 448 个 token。砍掉一半文本只将整个输入缩短 8.8%，生成耗时约降低 10%，显存几乎没有变化。因此，text-only pruning 可以作为联合方法的一部分，却不太可能单独带来很大的总体压缩；若目标是显著提速，下一步应同时研究 visual token pruning。

本结论仅限 Qwen2.5-VL-7B-Instruct、MMCoQA 的固定 50 样本以及当前 64-token chunk + BM25 方法。74% 对 72% 只是 1 道题的差异，不能宣称裁剪提高了准确率。下一步最有价值的验证是保留约 50% 文本的 BM25 设置，扩展样本数后与 visual-only、独立 text+visual pruning 和真正的联合决策进行对比；25% 文本保留率暂不适合作为默认配置。
