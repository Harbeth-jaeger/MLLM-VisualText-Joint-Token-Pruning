# 002 · Chunk-based Visual-Text Joint Token Pruning

> 状态：实验已实现并完成首轮 50 样本评测（2026-10-14）。
> 本文记录实验设计、实现验证、正式结果及基于当前证据能够支持的结论。

## 1. Objective

验证一种 sample-wise、training-free、query-aware 的视觉—文本联合 token 剪枝方法：在统一总预算下，以 chunk 为结构单元评估文本与视觉信息的价值，按样本动态分配模态预算，并在降低 LLM 输入长度与推理开销的同时尽量保持任务性能。

本轮是初步可行性探索，重点回答：该方法能否正确运行、能否严格满足联合预算，以及相比只剪单一模态或固定分配预算是否呈现值得继续研究的信号。

## 2. Setup

### 2.1 数据集

首轮采用 **MileBench / MMCoQA**，并复用实验 001 已固化的 50 个样本：


| 项目      | 设定                                               |
| --------- | -------------------------------------------------- |
| Benchmark | MileBench                                          |
| 任务      | MMCoQA（长文本、多图、自由回答）                   |
| 原始数据  | `.env` 中的 `MILEBENCH_ROOT` / `MMCOQA_IMAGE_DIR`  |
| 固定子集  | `.env` 中的 `MMCOQA_SUBSET_PATH`                   |
| 首轮规模  | 先用 5 个样本 smoke test，再在固定 50 个样本上运行 |
| 样本顺序  | 沿用实验 001 的固定顺序，不重新采样                |
| 主指标    | MMCoQA 归一化答案子串匹配 Accuracy                 |

选择 MMCoQA 的原因：

- 输入同时包含多张图片、较长文本上下文和问题，能够同时构造 $T_c$ 与 $V$；
- 与项目面向的长多模态上下文和 Visual RAG 场景较接近，证据分散在图像与文本中；
- 实验 001 已完成数据读取、图文交错 prompt、Qwen2.5-VL 推理及评分链路，可减少本轮新增变量；
- 复用相同样本可与 text-only 结果进行配对比较。

该数据并不能完整代表最终目标场景。实验 001 中，这 50 个样本的平均视觉 token 数约为 2,104，可剪文本 token 数约为 448，视觉仍占明显多数。因此，本轮只用于机制验证，不能据此证明方法已经适用于视觉与文本长度接近的长文档。若先导结果成立，后续应加入 **MMLongBench-Doc** 等多页长文档 benchmark，并按文本/视觉 token 比例分桶报告结果。

### 2.2 模型


| 项目             | 设定                                        |
| ---------------- | ------------------------------------------- |
| 模型             | Qwen2.5-VL-7B-Instruct                      |
| 本地权重         | `.env` 中的 `QWEN_MODEL_PATH`               |
| 推理精度         | BF16                                        |
| 推理方式         | 开源权重本地推理，不训练模型或额外 selector |
| Batch size       | 1                                           |
| 解码             | Greedy decoding，`do_sample=False`          |
| `max_new_tokens` | 32，与实验 001 保持一致                     |

选择该模型是因为权重已经完整存在，实验 001 已验证其在 MMCoQA 上的推理链路；其配置使用 `Qwen2_5_VLForConditionalGeneration`，视觉编码器输出经 projector 映射到 LLM hidden size，适合在 projector 之后、LLM prefill 之前插入联合剪枝。

实现前必须通过一次无剪枝回归，确认新推理入口在相同输入和解码配置下复现实验 001 的 baseline 输出或至少复现其评分结果。本轮默认权重由 `.env` 中的 `QWEN_MODEL_PATH` 指定。

### 2.3 运行环境

复用 `.env` 中 `MLLM_PRUNE_ENV_PATH` 指定的现有 Conda/micromamba 环境。

```bash
source .env
"$MLLM_PRUNE_ENV_PATH/bin/python" --version
```

已核实的核心版本：


| 组件            |     当前版本 | 判断                    |
| --------------- | -----------: | ----------------------- |
| Python          |      3.11.17 | 可用                    |
| PyTorch         |  2.6.0+cu124 | 可用                    |
| TorchVision     | 0.21.0+cu124 | 可用                    |
| Transformers    |       4.57.6 | 可用                    |
| Accelerate      |       1.15.0 | 可用                    |
| `qwen-vl-utils` |       0.0.14 | 可用                    |
| `python-dotenv` |        1.2.4 | 可用                    |
| FlashAttention  |  2.8.3.post1 | 已安装，待 GPU 实跑验证 |
| `einops`        |        0.8.2 | 可用                    |
| `ninja`         |       1.13.2 | 可用                    |
| Pillow          |       12.3.0 | 可用                    |
| NumPy           |        2.4.6 | 可用                    |
| Pytest          |        9.1.1 | 可用                    |

结论：现有环境足以开始实现和进行正确性 smoke test，应直接在该环境上增量补充真正缺失的包，不新建另一套大型环境。当前项目 `requirements.txt` 已覆盖 Qwen2.5-VL 推理、图像处理、BM25 和结果处理所需的主要依赖。

注意事项：

- 已安装 FlashAttention，并在 `.env` 中将 `MLLM_ATTN_IMPLEMENTATION` 设为 `flash_attention_2`。正式实验前必须在 GPU 可见时确认 kernel 实际运行，并与 SDPA 做正确性和端到端速度对照；不兼容时通过配置明确回退到 `sdpa`。
- 当前环境未安装 `datasets`、`evaluate`、`sentencepiece`、`bitsandbytes`、`decord` 和 `opencv-python`；本轮不应预先全部安装，只有代码实际依赖时才增量加入。
- 不直接安装 MileBench 官方 `requirements.txt`。该文件固定了较旧的 PyTorch、Transformers、Accelerate 与 NumPy 版本，并包含大量本轮无关包，可能破坏已工作的 Qwen2.5-VL 环境。
- 本次检查时 `nvidia-smi` 无法与驱动通信，因此尚未重新确认可用 GPU。正式运行前必须再次检查 GPU、显存和占用情况；不得在 GPU 不可用时静默回退到 CPU 执行全量实验。

### 2.4 实验条件与对照

首轮统一使用总保留率：

$$
\rho\in\{1.00, 0.75, 0.50, 0.25\}.
$$

其中 $\rho=1.00$ 为无剪枝 baseline。建议至少比较以下条件：


| 方法           | 文本剪枝 | 视觉剪枝 | 模态预算                               | 目的                         |
| -------------- | -------: | -------: | -------------------------------------- | ---------------------------- |
| No Pruning     |       否 |       否 | 全部保留                               | 性能与效率基线               |
| Text-only      |       是 |       否 | 视觉全保留                             | 与实验 001 衔接              |
| Visual-only    |       否 |       是 | 文本全保留                             | 检验联合方法是否优于只剪视觉 |
| Fixed Joint    |       是 |       是 | 固定 50%/50%，受容量约束               | 检验动态模态预算的价值       |
| Random Joint   |       是 |       是 | 与主方法相同预算，chunk/token 随机选择 | 检验相关性评分的价值         |
| Proposed Joint |       是 |       是 | 由$R_t,R_v$ 动态分配                   | 主方法                       |

Random Joint 至少使用 3 个随机种子并报告均值与标准差。所有方法必须使用完全相同的样本、模型、图像预处理、解码配置和总 token 预算；若由于容量约束产生实际预算偏差，应记录而不能静默忽略。

### 2.5 超参数与消融

主配置先使用方案默认值：


| 参数                     |          主配置 |
| ------------------------ | --------------: |
| 文本 chunk 最小长度      |       16 tokens |
| 文本 chunk 最大长度      |       64 tokens |
| 视觉窗口$w$              | 2，即$2\times2$ |
| Chunk 价值的 top 比例    |      20% tokens |
| 模态价值的 top 比例      |      20% chunks |
| 模态温度$\tau_m$         |            0.10 |
| Chunk 温度$\tau_g$       |            0.10 |
| 文本最低预算$b_t^{\min}$ |     总预算的 5% |
| 视觉最低预算$b_v^{\min}$ |    总预算的 10% |

不得用正式 50 样本的最终分数反复选择 $\tau_m$、$\tau_g$ 或最低预算。首轮可在独立的少量 calibration 样本上确认数值稳定性和预算分布；若暂时没有独立校准集，应把参数 sweep 标为探索性结果，不宣称选出的配置具有泛化性。

本轮实际直接采用上述预设值，没有根据正式 50 样本 Accuracy 调参，也没有运行温度或窗口大小 sweep。因此本轮结果只评价这一组预注册式主配置，不能据此比较其他温度或视觉窗口。

建议的最小消融为：$w\in\{2,4\}$、是否启用最低模态预算，以及动态预算对比固定预算。其余消融在主链路跑通后再决定，避免首轮条件过多。

### 2.6 评价与记录

任务性能：

- Accuracy；
- 相对 No Pruning 的分数变化（percentage points）；
- 与 No Pruning 的答案一致率；
- 正确→错误、错误→正确的配对计数。

压缩与预算：

- 原始与保留的文本、视觉、总 token 数；
- 实际总保留率和分模态保留率；
- $R_t,R_v$、$alpha_t,\alpha_v$、$B_t,B_v$；
- 每个 chunk 的长度、价值、分配预算和最终保留数；
- 容量溢出与再分配次数。

效率：

- 剪枝算法耗时；
- visual encoder、projector、LLM prefill 和 decode 耗时，能够拆分时分别报告；
- 端到端样本耗时；
- 峰值显存；
- 输入 token 吞吐与生成 token 数。

先进行预热，再用 CUDA synchronize 包围计时区间。由于本方法在 visual projector 之后剪枝，visual encoder 和 projector 的计算不会减少；主要预期收益来自 LLM prefill、后续 attention/FFN 计算以及 KV cache。端到端收益必须包含剪枝开销，不能仅用理论 token 缩减代替实测加速。

## 3. Method

### 3.1 输入与剪枝对象

MLLM 的输入序列表示为：

$$
X=[T_{\mathrm{sys}},T_q,T_c,V]
$$

其中：

- $T_{\mathrm{sys}}$：system prompt；
- $T_q$：用户问题；
- $T_c$：文本上下文；
- $V$：视觉 token。

仅对 $T_c$ 和 $V$ 进行剪枝：

$$
\mathcal C=T_c\cup V
$$

以下内容始终保留：

- $T_{\mathrm{sys}}$；
- $T_q$；
- BOS、EOS 等特殊 token；
- 图像起止与模态分隔 token。

剪枝在 visual projector 之后、LLM 正式执行 prefill 之前进行。总保留预算固定为：

$$
B=\left\lfloor\rho(N_t+N_v)\right\rfloor
$$

其中 $N_t=|T_c|$，$N_v=|V|$，$
ho\in(0,1]$ 为总保留比例。

### 3.2 Token 分组

#### 3.2.1 文本分组

将文本上下文 $T_c$ 按句子边界划分为若干文本 chunk：

$$
\mathcal G_t=\{G_{t,1},G_{t,2},\ldots,G_{t,M}\}
$$

每个文本 chunk 的长度限制为：

$$
16\le |G_{t,i}|\le64
$$

具体处理规则如下：

1. 根据句号、问号、感叹号、分号和换行符识别句子边界；
2. 长度超过 64 tokens 的句子按照语序切分为多个不超过 64 tokens 的 chunk；
3. 长度不足 16 tokens 的句子与相邻短句合并；
4. 如果合并后超过 64 tokens，则保留为独立短 chunk；
5. 不跨越文档、段落或对话轮次边界进行合并。

#### 3.2.2 视觉分组

假设 visual encoder 输出的视觉 token 可以还原为空间特征网格：

$$
V\in\mathbb R^{H_v\times W_v\times D_v}
$$

经过 visual projector 后：

$$
\tilde V=P(V)\in\mathbb R^{H_v\times W_v\times D}
$$

其中 $D$ 为 LLM 的 hidden dimension。将空间相邻的视觉 token 划入同一个固定窗口：

$$
\mathcal G_v=\{G_{v,1},G_{v,2},\ldots,G_{v,N}\}
$$

每个窗口大小为：

$$
w\times w,\qquad w\in\{2,4\}
$$

默认使用 $2\times2$ 窗口。对于位于图像边缘、无法构成完整窗口的视觉 token，将其组成较小的不完整窗口，不进行填充。

对于多图或 multi-crop 输入，分别在每张图片或每个 tile 内部划分窗口，不跨图片或 tile 组成视觉 chunk。

### 3.3 特征提取与归一化

所有重要性计算均在 LLM 的输入表示空间中完成。对于文本上下文和问题 token，使用 LLM embedding layer 输出：

$$
H_t=E(T_c)\in\mathbb R^{N_t\times D}
$$

$$
H_q=E(T_q)\in\mathbb R^{N_q\times D}
$$

对于视觉 token，使用 visual projector 输出：

$$
H_v=P(E_v(I))\in\mathbb R^{N_v\times D}
$$

对所有 token 表示进行 $L_2$ 归一化：

$$
\hat h_i=\frac{h_i}{\|h_i\|_2},\qquad
\hat h_{q,j}=\frac{h_{q,j}}{\|h_{q,j}\|_2}
$$

归一化后，文本 token、视觉 token 和问题 token 均位于维度为 $D$ 的 LLM 输入空间。

### 3.4 Token 重要性计算

对于任意候选 token $x_i\in T_c\cup V$，分别计算它与每个问题 token 的余弦相似度：

$$
a_{i,j}=\cos(h_i,h_{q,j})=\hat h_i^\top\hat h_{q,j}
$$

使用候选 token 与所有问题 token 之间的最大相似度作为其重要性：

$$
s_i=\max_{1\le j\le N_q}a_{i,j}
$$

该分数表示候选 token 与问题中最相关语义成分之间的匹配程度。为避免特殊符号和低信息问题 token 干扰匹配，以下问题 token 不参与最大值计算：

- BOS、EOS 等特殊 token；
- 标点符号；
- 模态分隔符；
- padding token。

### 3.5 Chunk 价值计算

对于任意 chunk $G_g$，首先按照 token 重要性 $s_i$ 从高到低排序：

$$
s_{g,(1)}\ge s_{g,(2)}\ge\cdots\ge s_{g,(|G_g|)}
$$

取 chunk 内重要性最高的 20% token：

$$
K_g^{\mathrm{eval}}=\max\left(1,\left\lceil0.2|G_g|\right\rceil\right)
$$

chunk 价值定义为这些 token 重要性的平均值：

$$
r_g=\frac{1}{K_g^{\mathrm{eval}}}\sum_{k=1}^{K_g^{\mathrm{eval}}}s_{g,(k)}
$$

分别得到文本和视觉 chunk 价值：

$$
\{r_{t,1},r_{t,2},\ldots,r_{t,M}\},\qquad
\{r_{v,1},r_{v,2},\ldots,r_{v,N}\}
$$

### 3.6 模态价值计算

对于模态 $m\in\{t,v\}$，将其 chunk 价值从高到低排序：

$$
r_{m,(1)}\ge r_{m,(2)}\ge\cdots\ge r_{m,(|\mathcal G_m|)}
$$

取价值最高的 20% chunk：

$$
K_m^{\mathrm{eval}}=\max\left(1,\left\lceil0.2|\mathcal G_m|\right\rceil\right)
$$

模态整体价值为：

$$
R_m=\frac{1}{K_m^{\mathrm{eval}}}\sum_{k=1}^{K_m^{\mathrm{eval}}}r_{m,(k)}
$$

由此分别得到 $R_t$ 和 $R_v$。其中 $R_t$ 表示文本上下文对当前问题的整体价值，$R_v$ 表示视觉输入对当前问题的整体价值。

### 3.7 模态间预算分配

#### 3.7.1 初始分配比例

通过带温度参数的 softmax 将模态价值转换为预算比例：

$$
\alpha_t=\frac{\exp(R_t/\tau_m)}{\exp(R_t/\tau_m)+\exp(R_v/\tau_m)}
$$

$$
\alpha_v=\frac{\exp(R_v/\tau_m)}{\exp(R_t/\tau_m)+\exp(R_v/\tau_m)}
$$

满足 $\alpha_t+\alpha_v=1$。其中 $\tau_m>0$ 为模态预算分配温度：较小的 $\tau_m$ 产生更激进的预算差异，较大的 $\tau_m$ 使预算更加均衡。

#### 3.7.2 最低预算约束

分别设置文本和视觉模态的最低预算 $b_t^{\min}$、$b_v^{\min}$，并满足：

$$
b_t^{\min}+b_v^{\min}\le B
$$

将剩余预算定义为：

$$
B_{\mathrm{free}}=B-b_t^{\min}-b_v^{\min}
$$

最终模态预算为：

$$
B_t=b_t^{\min}+\operatorname{round}(\alpha_tB_{\mathrm{free}})
$$

$$
B_v=B-B_t
$$

同时施加容量约束 $B_t\le N_t$、$B_v\le N_v$。如果某一模态的预算超过其 token 总数，则将溢出的预算重新分配给另一模态。

### 3.8 模态内 Chunk 预算分配

对于模态 $m$ 中的 chunk $G_{m,g}$，根据其价值计算预算权重：

$$
p_{m,g}=\frac{\exp(r_{m,g}/\tau_g)}{\sum_{j=1}^{|\mathcal G_m|}\exp(r_{m,j}/\tau_g)}
$$

其中 $\tau_g>0$ 为 chunk 预算分配温度。每个 chunk 的连续预算为：

$$
\tilde B_{m,g}=p_{m,g}B_m
$$

施加 chunk 容量约束：

$$
0\le B_{m,g}\le |G_{m,g}|
$$

#### 3.8.1 整数预算分配

首先对连续预算向下取整：

$$
B_{m,g}^{(0)}=\min\left(|G_{m,g}|,\left\lfloor\tilde B_{m,g}\right\rfloor\right)
$$

计算尚未分配的预算：

$$
B_m^{\mathrm{remain}}=B_m-\sum_gB_{m,g}^{(0)}
$$

按照小数余量：

$$
d_{m,g}=\tilde B_{m,g}-\left\lfloor\tilde B_{m,g}\right\rfloor
$$

从大到小依次向尚未达到容量上限的 chunk 追加一个预算，直至：

$$
\sum_gB_{m,g}=B_m
$$

如果某个 chunk 达到容量上限，则跳过该 chunk，继续向其他 chunk 分配剩余预算。

### 3.9 Chunk 内 Token 选择

对于每个 chunk $G_{m,g}$，按照 token 重要性 $s_i$ 从高到低排序，并保留前 $B_{m,g}$ 个 token：

$$
S_{m,g}=\operatorname{TopK}\left(\{s_i\mid i\in G_{m,g}\},B_{m,g}\right)
$$

分别得到文本和视觉保留集合：

$$
S_t=\bigcup_{g=1}^{M}S_{t,g},\qquad
S_v=\bigcup_{g=1}^{N}S_{v,g}
$$

满足：

$$
|S_t|=B_t,\qquad |S_v|=B_v,\qquad |S_t|+|S_v|=B
$$

### 3.10 序列重构

Top-$K$ 只用于确定保留 token 集合，不改变 token 在原始输入中的排列顺序。对选中的文本 token，根据其在 $T_c$ 中的原始位置重新排序：

$$
T_c'=\operatorname{SortByPosition}(S_t)
$$

对选中的视觉 token，根据其原始图像、tile 和空间位置重新排序：

$$
V'=\operatorname{SortByPosition}(S_v)
$$

最终压缩后的输入序列为：

$$
X'=[T_{\mathrm{sys}},T_q,T_c',V']
$$

若原始模型模板中视觉 token 位于文本 token 之前，则按照原模型的输入模板重构，仅替换其中的 $T_c$ 和 $V$，不改变模态边界及特殊 token 位置。

### 3.11 推理过程

压缩后的序列 $X'$ 被送入 LLM 执行完整 prefill 和 decode：

$$
H'=\operatorname{LLM}_{\mathrm{prefill}}(X')
$$

$$
Y=\operatorname{LLM}_{\mathrm{decode}}(H')
$$

整个剪枝过程：

- 不更新 MLLM 参数；
- 不训练额外 selector；
- 不使用额外标注；
- 不执行 token merging 或信息回收；
- 不在 decode 阶段继续剪枝；
- 对每个样本独立计算模态价值及预算分配。

因此，该方法属于 sample-wise、training-free、query-aware 的 visual-text joint token pruning 方法。

## 4. Results & Analysis & Conclusion

### 4.1 运行与数据完整性

正式实验使用 4 张 NVIDIA RTX A6000 按样本分片运行，实际 attention 后端为 `flash_attention_2`。原始结果位于：

```text
outputs/experiment_002/results_shard_0.jsonl
outputs/experiment_002/results_shard_1.jsonl
outputs/experiment_002/results_shard_2.jsonl
outputs/experiment_002/results_shard_3.jsonl
outputs/experiment_002/summary.json
```

实验运行时项目工作树包含尚未提交的实验 002 实现，因此当前没有可唯一引用的 Git commit。复现需同时保留本报告对应的源码、配置文件 `configs/experiments/002_chunk_based_joint_pruning.yaml` 和四个原始 JSONL。实际条件未包含 Text-only；原因见 4.4 节的统一预算不可行性说明。

完整性检查结果：


| 检查项                                        |    结果 |
| --------------------------------------------- | ------: |
| 样本数                                        |      50 |
| 每个样本的实验条件数                          |      19 |
| 总记录数                                      |     950 |
| 重复条件键                                    |       0 |
| 缺失字段/空记录                               |       0 |
| 使用 FlashAttention 2 的记录                  | 950/950 |
| 非 visual-only 条件的预算不一致               |       0 |
| chunk 预算、选择数量或压缩长度 invariant 错误 |       0 |

smoke test 还使用 `--verify-baseline` 比较了自定义 mRoPE/greedy decode 与 Transformers 原始 `generate`；两者生成 token 序列完全一致。正式数据中每个样本均完成 baseline、visual-only、fixed joint、3 个随机种子的 random joint 和 proposed joint。

50 个样本平均包含 458.88 个可剪文本 token、2,103.62 个视觉 token 和 83.04 个保护 token，平均完整输入长度为 2,645.54。视觉 token 在候选集合中的平均占比为 78.23%，但不同样本差异较大，范围为 22.21%–94.09%。这说明 MMCoQA 总体仍偏视觉，但确实包含部分文本占主导的样本。

### 4.2 任务性能

下表中的 Random Joint Accuracy 是 3 个随机种子的均值与种子间标准差；其他方法只运行确定性的单个 seed。`Δ` 为相对本轮 66% baseline 的百分点变化。


| 方法               | 候选保留率 |      Accuracy |         Δ | 与 baseline 答案一致率 |
| ------------------ | ---------: | ------------: | ---------: | ---------------------: |
| No Pruning         |       100% |           66% |       0 pp |                   100% |
| Visual-only        |        75% |           68% |      +2 pp |                    60% |
| Fixed Joint        |        75% |           64% |      -2 pp |                    56% |
| Random Joint       |        75% | 64.0% ± 2.0% |      -2 pp |                  58.0% |
| **Proposed Joint** |    **75%** |       **68%** |  **+2 pp** |                **62%** |
| Visual-only        |        50% |           60% |      -6 pp |                    40% |
| Fixed Joint        |        50% |           52% |     -14 pp |                    34% |
| Random Joint       |        50% | 58.7% ± 2.3% |    -7.3 pp |                  40.7% |
| **Proposed Joint** |    **50%** |       **56%** | **-10 pp** |                **42%** |
| Visual-only        |        25% |           62% |      -4 pp |                    22% |
| Fixed Joint        |        25% |           48% |     -18 pp |                    16% |
| Random Joint       |        25% | 47.3% ± 1.2% |   -18.7 pp |                  22.0% |
| **Proposed Joint** |    **25%** |       **54%** | **-12 pp** |                **20%** |

主要观察如下：

1. **75% 是当前唯一没有观察到 Accuracy 下降的 proposed 配置。** Proposed Joint 在 75% 时答对 34/50，baseline 为 33/50；配对变化为 3 个错误→正确、2 个正确→错误。
2. **更强压缩会明显损伤性能。** Proposed Joint 在 50% 和 25% 时分别下降 10 pp 和 12 pp，说明当前 selector 无法在激进预算下稳定保留充分证据。
3. **动态方法相对 Fixed Joint 呈一致但较小的正向信号。** Proposed 在 25%/50%/75% 分别比 Fixed 高 6/4/4 pp，配对胜负为 5:2、2:0、2:0；但样本量只有 50，这些差异的双侧精确配对检验均不显著（最小 $p=0.453$）。
4. **当前方法没有稳定优于随机选择。** 50% 时 Random Joint 的三种子平均 Accuracy 为 58.7%，高于 Proposed 的 56%；25% 和 75% 时 Proposed 分别高 6.7 pp 和 4 pp。结果随预算变化，不能支持“相关性评分稳定优于随机”的结论。
5. **50 样本的不确定性较大。** Proposed 在 25%/50%/75% 的 95% Wilson 区间分别约为 40.4%–67.0%、42.3%–68.8%、54.2%–79.2%；baseline 为 52.2%–77.6%。所有 Proposed 与 baseline 的配对差异均未达到显著性（双侧精确检验 $p\ge0.210$）。因此 75% 的 +2 pp 只能视为可继续验证的信号，不能解释为已证明提升。

### 4.3 动态预算实际行为

Proposed 的模态分数和预算呈现明显的文本偏置：


| 保留率 | 平均文本预算占比 | 文本预算占比范围 | 完整保留全部文本的样本 |
| -----: | ---------------: | ---------------: | ---------------------: |
|    25% |           62.19% |   23.66%–90.19% |                  34/50 |
|    50% |           40.97% |   11.82%–90.03% |                  43/50 |
|    75% |           28.75% |    7.88%–89.80% |                  49/50 |

文本模态分数均值为 0.881，视觉模态分数均值仅为 0.083；21/50 个样本的文本模态分数恰好为 1.0。原因是文本候选与问题使用同一个静态 LLM token embedding 空间，只要 chunk 中出现与问题相同的词元，最大余弦相似度就可能达到 1；视觉 projector embedding 与文本 embedding 的分布和标度则没有经过校准。在当前 $\tau_m=0.1$ 下，softmax 会进一步放大这种差异。

因此，本轮“动态分配”大多数时候退化为“尽量保留全部文本，再把剩余预算给视觉”。它与 visual-only 的预测一致率在 25%/50%/75% 时分别达到 80%、90%、98%；两者预算完全相同的样本分别为 34、43、49 个。Visual-only 相比 Proposed 在 25% 和 50% 各多答对 4 个和 2 个样本，主要来自 visual-only 在文本容量冲突时使用了更多实际预算，而不是证明视觉选择策略更优。

这一结果说明 sample-wise 预算机制确实运行了，但当前跨模态分数**不可直接比较**。在解决分数校准前，不能把本实验解释为动态模态分配已经有效。

### 4.4 Visual-only 的预算可行

Visual-only 必须保留全部文本。当文本数量本身超过目标总预算时，它无法同时满足“只剪视觉”和“严格统一总预算”。本轮显式记录了这类偏差：


| 目标保留率 | 发生预算上调的样本 | 平均实际保留率 | 最大实际保留率 | 最大额外 token |
| ---------: | -----------------: | -------------: | -------------: | -------------: |
|        25% |              14/50 |         30.46% |         77.79% |          1,236 |
|        50% |               3/50 |         50.82% |         77.79% |            651 |
|        75% |               1/50 |         75.04% |         77.79% |             66 |

因此，尤其在 25% 条件下，Visual-only 的 62% Accuracy 不能与严格 25% 预算的 joint 方法作完全公平的横向比较。它更适合作为“保持所有文本时能达到什么效果”的方法边界。Text-only 对照因视觉 token 数通常已经超过低保留率总预算，本轮未运行；强行加入会产生更严重的同类问题。

### 4.5 效率与显存

平均压缩后完整 LLM 输入长度从 baseline 的 2,645.54 降为：


| 候选保留率 | 平均完整输入 token | Proposed 平均 prefill | Prefill 加速 | LLM 阶段加速 | 峰值显存 |
| ---------: | -----------------: | --------------------: | -----------: | -----------: | -------: |
|       100% |           2,645.54 |             371.94 ms |       1.00× |       1.00× | 16.39 GB |
|        75% |           2,004.56 |             284.19 ms |       1.31× |       1.07× | 16.19 GB |
|        50% |           1,364.04 |             196.20 ms |       1.90× |       1.20× | 16.00 GB |
|        25% |             723.26 |             111.48 ms |       3.34× |       1.34× | 15.80 GB |

Prefill 加速与序列缩短方向一致，说明剪枝确实减少了 LLM prefill 工作量，而不只是理论 FLOPs 下降。整体 LLM 阶段收益更小，因为 decode 时间还受到输出长度影响：baseline 平均生成 17.4 token，而 Proposed 三档平均生成约 18.9–19.5 token。

若采用更严格的可部署口径——baseline 不计不必要的 pruning，而剪枝方法计入 relevance、planning、vision encoder、prefill 和 decode——Proposed 在 25%/50%/75% 的已测流水线加速约为 1.12×、1.05×、0.97×。也就是说，75% 虽然 prefill 更快，但剪枝开销在当前实现和输入规模下尚未摊平，端到端反而略慢。该口径仍不包含 CPU 图像预处理、模型加载、JSON 序列化和磁盘 I/O，因此不能代替完整应用延迟。

显存只下降约 0.20–0.60 GB，因为 7B 模型权重和视觉编码器占据主要显存，batch size 又为 1；缩短 KV cache 对本设置峰值显存的影响有限。视觉编码平均约 271.96 ms，且本方法在 projector 后剪枝，因此这部分没有减少。

### 4.6 与实验 001 baseline 的口径差异

实验 001 保存的 baseline Accuracy 为 72%，平均输入 2,639.98 token；本轮为 66% 和 2,645.54 token。两轮平均视觉 token 均为 2,103.62，差异来自文本处理，而不是图像分辨率。

进一步核对表明，两轮当前代码生成的原始 chat prompt 字符串完全一致；但实验 001 的所谓 baseline 仍调用了 `prune_context(..., retention=1.0)`，该函数会对文本分句、去除部分原始空白并重新拼接，因此并非严格的原始输入。实验 002 baseline 直接使用未重写的原始 context，并通过了与 Transformers `generate` 的 token 级回归。故本报告以内轮 66% baseline 作为所有配对比较基准，不把实验 001 的 72% 与本轮剪枝结果直接比较。

### 4.7 代表性配对变化

75% Proposed 相对 baseline 共改变 5 个样本的正确性：


| Sample | Answer            | Baseline                               | Proposed 75%            | 变化                                               |
| -----: | ----------------- | -------------------------------------- | ----------------------- | -------------------------------------------------- |
|     84 | Win               | `Giorgos Karagounis scored 2 goals...` | `...achieved a win...`  | 错→对                                             |
|     40 | Chicago Bears     | 回答称文本未提供信息                   | 明确回答`Chicago Bears` | 错→对                                             |
|     72 | four              | `4`                                    | `...has four leaves.`   | 错→对；也暴露字符串评测未把`4` 与 `four` 视为等价 |
|    182 | Sacramento Kings  | `Sacramento Kings`                     | `Houston Rockets`       | 对→错                                             |
|     69 | Manchester United | `Manchester United`                    | `Birmingham City`       | 对→错                                             |

样本 72 说明当前归一化子串指标存在数字表达等价性缺陷；剪枝后的长答案被判对，而语义正确的 baseline `4` 被判错。因此 Accuracy 变化中包含少量评测器敏感性，不应全部归因于模型推理能力变化。

### 4.8 结论

本轮得到三个可信结论：

1. **机制可行。** projector 后联合剪枝、mRoPE 位置保留、精确预算和压缩后自回归生成均正确运行；25%–50% 保留率能显著缩短 prefill。
2. **75% 保留率值得继续验证，但尚无性能提升证据。** Proposed 在 50 样本上达到 68%，比 baseline 高 2 pp，但差异不显著，且端到端未获得净加速。
3. **当前动态预算公式存在关键缺陷。** 文本与视觉静态 embedding 相似度未校准，使预算几乎总是优先满足文本；当前结果不能证明动态联合分配优于 visual-only、fixed 或 random。

因此，本实验支持继续研究“联合预算”问题，但不支持把当前算法作为最终方法。下一轮的优先级应是修复跨模态评分可比性和评测口径，而不是直接扩大 benchmark 或继续细调同一温度。

## 5. Next

建议按以下顺序推进：

1. **修正评测器。** 增加数字—英文数词等价、冠词和常见格式归一化；保留当前指标结果，同时报告修正版结果，避免覆盖历史口径。
2. **校准跨模态分数。** 在独立 calibration 样本上比较模态内 z-score/percentile、去除问题停用词与重复 token、分别设置文本/视觉温度，或使用浅层 contextual hidden state；先检查预算分布是否仍饱和，再跑 Accuracy。
3. **增加一个“容量匹配 visual-only”分析。** 对预算冲突样本单独报告，或只在目标预算可行的配对子集比较 visual-only，避免把额外 token 带来的收益误认为方法优势。
4. **复跑最小验证矩阵。** 优先使用 baseline、75%/50% proposed、对应 random 和 visual-only；若校准后 Proposed 仍不能稳定超过 random，则应重新考虑 token 评分而非扩大消融。
5. **在信号成立后扩展到长文档 benchmark。** MMCoQA 平均仍有 78.23% 的候选 token 来自视觉，不能充分代表文本与视觉长度接近的目标场景；后续再加入 MMLongBench-Doc 等数据。
