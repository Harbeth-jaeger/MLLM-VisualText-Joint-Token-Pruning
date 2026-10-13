# LLM Visual-Text Joint Pruning：可行性调研

## 0. Idea 落点

**研究方向：MLLM Visual-Text Joint Token Pruning**

核心研究问题：

> 给定有限的 token/computation budget，如何在 visual tokens 和 text tokens 之间动态分配预算，从而实现更优的效率—性能权衡？

即 **Cross-modal Adaptive Token Budget Allocation**。

相比传统 visual-only pruning，不仅研究两种模态分别剪哪些 token，还要研究**两种模态各剪多少，以及是否存在更优的联合分配策略**。

初步定位：Input-level、training-free、task-aware、sample-adaptive joint pruning。

## 1. 基本认识

1. **Text token 存在 redundancy**：部分文本语义重复、与当前任务无关，因此理论上可以剪枝。
2. **Text token 的剪枝风险通常更高**：与 visual token 相比，text token 往往数量更少、语义密度更高，删除关键指令、约束或关系词可能显著影响模型性能。但具体敏感程度依赖任务。
3. **不同模态的可剪冗余不对称**：visual token 数量多、冗余通常更高，因此现有 MLLM pruning 研究主要集中于 visual token。

核心问题不是 text 能不能剪，而是：**在什么场景下，剪 text 比继续剪 visual 更划算？**

## 2. 研究价值

### 2.1 Related Work：其他学者做得多不多？

目前已有大量 text-guided visual pruning 工作，但真正对两种模态都进行压缩的工作相对较少。以下两篇值得重点关注。

**① Dynamic-LLaVA — ICLR 2025**

[论文链接](https://proceedings.iclr.cc/paper_files/paper/2025/hash/aeafb73dfed3007ec5299be1604d6f99-Abstract-Conference.html)

* **研究问题**：现有 visual-only pruning 主要减少 prefill 计算，但随着 decode 阶段不断生成文本，输出 text context 逐渐增长，仅剪 visual 的相对收益减弱。
* **技术方案**：通过两个 learned predictors，分别稀疏化 prefill 阶段的 visual context，以及 decode 阶段生成的 text context；针对有无 KV cache 设计不同推理策略。
* **结果/结论**：报告 prefill 计算量降低约 75%，无 KV cache 的 decode 计算量降低约 50%，有 KV cache 时 GPU 显存开销降低约 50%，同时保持接近原模型的任务性能。

**与本 idea 的区别**：Dynamic-LLaVA 主要针对 input visual tokens + generated text context；本 idea 针对 input visual tokens + input text tokens，并重点研究两种模态的预算分配。

**② VLZip — arXiv 2026**

[论文链接](https://arxiv.org/abs/2608.08630)

* **研究问题**：超长、图文交错的多模态上下文同时包含大量 visual 和 text tokens，只压缩视觉信息难以充分解决长上下文计算与显存问题。
* **技术方案**：对视觉和文本分别进行分块、编码与特征压缩，将信息蒸馏为 compact soft prefixes，并将其注入不同 Transformer layers，从而缩短注意力序列。
* **结果/结论**：支持约 120K tokens 的训练上下文（报告为基线的 6 倍），并展示超过 280K tokens 的推理能力，同时保持较强的长上下文多模态推理性能。

**与本 idea 的区别**：VLZip 属于需要训练的 learned compression，将原始信息压缩为新的表示；本 idea 暂定为 training-free token pruning，直接选择并删除低价值 token，不额外训练大型压缩模块。

**调研结论**

* Visual-only pruning：研究较多，相对成熟。
* Visual-text joint compression：已有相关工作，并非全新方向。
* Input visual-text joint hard pruning + adaptive cross-modal budget allocation：目前未发现成熟、直接解决这一完整问题的代表性方案，但尚不能断言完全没有相关研究。

因此，**“两种模态都剪”不足以单独构成创新；真正的研究空间在于如何判断不同模态的相对重要性，以及如何自适应分配预算。**

另一个值得关注的相关方向是 **VL-Cache（ICLR 2025）**，它已经研究 modality-aware KV-cache scoring 和 layer-adaptive budget allocation。因此预算分配本身也不是全新概念，需要进一步突出 input-level token pruning 和跨模态联合决策的区别。

### 2.2 做出来以后意义大不大？

**① 潜在效率收益**

Input token pruning 可以影响 Prefill 和 Decode 两个阶段。


| 阶段     | 优化收益                                                           |
| -------- | ------------------------------------------------------------------ |
| Prefill  | 减少 attention、FFN 等计算                                         |
| Decode   | 减少 KV cache 显存与访问开销，以及历史 token 参与的 attention 计算 |
| 最终效果 | 潜在降低 TTFT、decode latency、峰值显存和端到端推理成本            |

注意：上述收益需要以真实 latency、memory、FLOPs 实验验证，不能直接用 token reduction ratio 代替实际加速比。

**② 暂定适用场景**

由于普通 VQA 的 visual tokens 往往远多于 text tokens，直接剪 text 的效率收益可能很有限。

因此重点考虑两类场景：

* **Aggressive visual pruning**：当 visual tokens 已被大量剪除后，text token 的相对占比上升，进一步进行 text pruning 的潜在价值增加。
* **Long-text MLLM**：长文本指令、多轮图文对话、多模态 ICL、图文交错长上下文等场景，text tokens 本身占比较高，可能存在更多可利用的冗余。

需要特别注意：visual pruning 后 text 占比上升，并不必然意味着 text pruning 的绝对收益变大。必须检验额外延迟收益、性能损失以及不同模态间的相互影响。

**③ 待验证的研究假设（Probe Experiments）**


| 假设                                        | 实验设计                                                      | 希望回答的问题                        |
| ------------------------------------------- | ------------------------------------------------------------- | ------------------------------------- |
| H1：Text 存在可剪冗余                       | Text-only pruning，sweep text pruning ratio                   | 能剪多少，质量损失多大？              |
| H2：Visual 大量剪除后，联合剪枝具有额外价值 | 在不同 visual pruning ratio 下继续剪 text，对比性能与 latency | 联合剪枝是否优于 visual-only？        |
| H3：最优剪枝比例依赖 sample/task            | 对不同 sample/task 进行二维 ratio sweep                       | 是否需要 adaptive budget allocation？ |

其中 H1 已经纳入此前设计的 text-only pruning 探究性实验，可直接沿用该方案；但在获得实际实验结果前，还不能视为已完成验证。

**研究价值的关键判据**：

若实验表明，在相同质量约束和计算预算下，某些非平凡的 visual-text 联合配置稳定优于 visual-only，并且最优配置随 sample/task 明显变化，则可以较有力地支持自适应联合预算分配的必要性。

## 3. 研究成本与初步技术方案

目前暂定采用较轻量、training-free 的路线，优先验证想法，不急于设计复杂网络。

### 3.1 Prune 对象：如何评判重要性？

% 思考更合适、更新颖、更有效的 metric 来评价联合 prune 对象，可考虑训练 selector

同时针对 visual/text tokens，考虑两类信息。

**① Importance：对任务输出有多重要？**

基本思想：删除 token 后，模型预测质量下降越大，该 token 越重要。

对于有参考答案的校准样本，可定义：

$$
I_i=\mathcal L(X\setminus x_i,Y)-\mathcal L(X,Y)
$$

其中 $\mathcal L$ 为任务损失。

这是理想的 token importance 定义，但逐个删除、重复推理的成本较高，因此实际可以使用 attention、query relevance、embedding 等代理指标。

**② Redundancy：信息是否已被其他 token 覆盖？**

基本思想：如果某个 token 的信息已经可以从其他 token 中获得，那么删除它的风险可能较低。

例如使用 hidden-state similarity：

$$
R_i=\max_{j\ne i}\cos(h_i,h_j)
$$

相似度越高，潜在冗余越大。但语义相似不一定代表功能等价，尤其需要警惕文本中的否定、数字和逻辑关系。

**初步思路**：将 importance 和 redundancy 结合，分别得到 visual/text token scores，再进行选择。跨模态的分数尺度需要校准，不能直接认为两个 scorer 的数值可比较。

### 3.2 Prune 时机：什么时候剪？

% prefill阶段、decode阶段之前：0层之前，或者0-10层的位置，不要在太深层进行 prune

暂定在**视觉编码器与 projector 处理完成后、进入 LLM 主干之前**进行 pruning。

$$
X=[V,T]\in\mathbb R^{B\times L\times D}
$$

$$
X'=[V',T']\in\mathbb R^{B\times L'\times D}
$$

其中 $L'<L$，即直接删除部分 tokens，降低序列长度。

这样可以减少 LLM 主干 Prefill 的计算，并在后续 Decode 中减少 KV cache 占用。

暂不研究 layer-wise pruning 或生成阶段的动态 pruning。

### 3.3 Prune 程度：Visual/Text 各剪多少？

这是本 idea 最核心、目前尚无明确方案的研究空间。

定义：

$$
r_V=\text{visual pruning ratio}
$$

$$
r_T=\text{text pruning ratio}
$$

目标是在质量约束下寻找更低成本的预算分配：

$$
\min_{r_V,r_T} C(r_V,r_T)
$$

$$
\text{s.t.}\quad D(r_V,r_T)\le\epsilon
$$

其中：

* $C$：推理计算/延迟/显存成本，实验中需明确主优化目标；
* $D$：与 full model 相比的质量损失；
* $\epsilon$：允许的最大质量偏移。

初期先用 calibration set 对 $(r_V,r_T)$ 进行二维 grid search，观察不同预算分配的效果及其最优区域。

若确认最优比例具有 sample/task dependence，再设计轻量级 adaptive allocator，根据两种模态的 token 数量、importance 分布、score gap 等信息，动态预测预算。

### 3.4 Evaluation：如何判断方案好不好？

**① 质量指标**

使用 full model 和 pruned model 在相同参考输出条件下的 next-token 概率分布，计算：

$$
D_{\mathrm{KL}}
(P_{\mathrm{full}}\mid P_{\mathrm{pruned}})
$$

要求：

$$
D_{\mathrm{KL}}\le\epsilon
$$

KL divergence 衡量的是预测分布的变化，适合做校准代理指标，但不能完全替代真实任务性能。

因此还需要验证：

* Accuracy / EM / F1 等任务指标；
* 不同任务类型、不同 pruning ratio 下的质量变化。

**② 效率指标**

* Token retention ratio；
* Prefill latency / TTFT；
* Decode latency / tokens per second；
* KV cache memory；
* End-to-end latency。

**③ 最核心的方案评价：与 Oracle 和 Baseline 比较**

通过二维 sweep 获得 calibration set 上近似最优的 $(r_V^*,r_T^*)$，将其作为参考。

将提出的 budget allocator 与以下方法比较：


| Baseline           | 作用                                   |
| ------------------ | -------------------------------------- |
| Full model         | 原始性能参考                           |
| Visual-only        | 验证是否有必要剪 text                  |
| Text-only          | 验证 text pruning 本身的价值           |
| Fixed-ratio joint  | 验证 adaptive allocation 的必要性      |
| Grid-search oracle | 衡量当前分配策略距离最佳可选配置有多远 |

特别注意：如果 oracle 是通过 calibration set sweep 得到的，就应在独立 test set 上进行最终评估，避免把校准集上的最优结果当作泛化性能。

最终重点看：**在相同质量损失或相同计算预算下，联合自适应剪枝能否形成比现有 baseline 更优的效率—性能 Pareto frontier。**

## 4. 初步结论与下一步

**研究价值：有潜力，但必要性仍待验证。**

* Novelty：联合压缩已有相关论文，单纯 joint pruning 不足以保证创新；自适应跨模态预算分配可能是更有价值的切入点。
* Necessity：短文本 VQA 场景价值有限；aggressive visual pruning 与 long-text multimodal context 更值得关注。
* Feasibility：training-free 的初步 probe 方案实现门槛相对较低，但真正获得推理加速、实现稳定的动态预算分配仍有工程与方法难度。

**下一步建议**：先完成已设计的 text-only pruning 实验，再扩展为 visual-text 二维 ratio sweep，得到 performance–latency sensitivity map。

若发现联合剪枝存在明显优势且最优配置随任务或样本变化，再正式投入 Cross-modal Adaptive Token Budget Allocation 的方法设计；反之则应及时调整研究方向或应用场景。

**一句话总结：这个 idea 的真正价值，不是证明 visual 和 text 都能剪，而是研究有限预算究竟应该如何在两种模态之间分配。**
