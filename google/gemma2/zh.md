# Gemma 2

- **作者 / 机构**：Gemma Team，Google DeepMind
- **发表时间**：2024-08
- **链接**：[论文 (arXiv:2408.00118)](https://arxiv.org/abs/2408.00118) · [模型 (HuggingFace)](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) · [代码 (keras-hub / transformers)](https://github.com/keras-team/keras-hub)

## 一句话总结

Gemma 2 是 Google DeepMind 的第二代开源模型系列，分为 2B、9B、27B 三个规格。27B 从头训练；2B 和 9B 采用**知识蒸馏**——用 27B 教师模型的软概率分布代替硬 token 标签进行训练，以更小的参数量传递更多能力。架构上有两处 infra 层面值得关注的改动：**logit 软截断**（一种低成本的训练稳定技术，在 BF16 下无需 loss scaling 即可防止 logit 爆炸）和**交替局部/全局注意力**（每隔一层使用滑动窗口注意力，将 KV cache 增长量压缩约一半）。结果：Gemma 2 27B 在参数量只有 Llama 3.1 70B 一半的情况下性能相当；Gemma 2 9B 可与 13B–40B 区间的模型竞争。

## 背景与动机

### 开源模型的效率差距

2024 年中，Llama 3 系列已确立强势基准：8B 和 70B 稠密模型在约 15T token 上训练，指令微调版本媲美 GPT-3.5 级别。开源生态的实际问题是：能否在更小参数量下获得相近能力？

答案取决于两点：(1) 训练数据的质量与数量；(2) 数据信息向模型权重转化的效率。Gemma 2 直接攻克 (2)——对 2B 和 9B 模型，用**教师的完整概率分布**取代（或辅助）硬 next-token prediction 目标，每个 token 携带的信息量大幅提升。

### 为何蒸馏在小模型前沿如此关键

硬标签只说"下一个 token 是 X"。教师的软分布则说："下一个 token 是 X 的概率为 0.6，但 Y 也合理（0.2），Z 有一定可能（0.1）……"软分布中的熵编码了不确定性、近义词、合理续写和推理路径——这些信息在硬标签中全部丢失。对于容量有限的小模型，教师在每个训练步骤提供的引导是重要的学习信号放大器。

基础设施含义：蒸馏训练要求教师模型在整个学生训练过程中常驻（或可访问）。对于 27B 教师 + 9B 学生，这意味着需要同时加载两个模型，对每个训练 batch 执行教师前向传播，并将生成的 logit 分布作为训练目标。

## 核心方法

### 架构：相较于 Gemma 1 的变化

Gemma 2 保留了 Gemma 1 的基础（带 RoPE 位置编码、RMSNorm、GeGLU FFN 激活函数、全程 GQA 的 Transformer），并增加了三项修改：

**1. 交替局部/全局注意力**

注意力层在两种模式间交替：
- **局部（滑动窗口）**：每个 token 仅关注前 4,096 个 token 的窗口。每层 KV cache 上限固定为 `2 × window_size × d_head × n_kv_heads`，与序列长度无关。
- **全局（全量）**：对完整序列做标准注意力。KV cache 随序列长度正常增长。

一半层为局部，一半为全局（交替排列）。对于长度为 L 的序列，总 KV cache 近似为：

```
KV cache ≈ n_layers/2 × (2 × 4096 × d_kv)   [局部层，上限固定]
          + n_layers/2 × (2 × L × d_kv)       [全局层，随 L 增长]
```

在 Gemma 2 训练的 8,192 上下文长度下，局部层与全局层贡献大致相当。更长的序列中，局部层贡献保持不变，全局层线性增长——相比全程全局注意力，KV cache 开销约减半。

全局层负责长距离依赖；局部层负责局部连贯性。交替排列意味着每个 token 距离最近的全局注意力计算最多两跳。

**2. Logit 软截断**

在两处应用：注意力 softmax 之前的 attention logits，以及最终输出 softmax 之前的 output logits。公式：

```
logits_capped = cap × tanh(logits / cap)
```

attention logits 的 `cap = 50.0`，output logits 的 `cap = 30.0`。`tanh` 将数值平滑压缩至 ±`cap` 区间——极端 logit 被拉回，小值几乎不受影响（因为小 x 时 `tanh(x) ≈ x`）。

动机：BF16 训练容易出现 logit 爆炸——一个正反馈循环，过大的 softmax 输入产生极尖锐的分布，产生过大的梯度，进一步推高 logit。Loss scaling 在梯度路径上缓解这一问题，但对前向传播无效。软截断直接在前向传播中处理，消除了对激活裁剪启发式方法的需求。计算代价极低：每次 attention 和 output softmax 多一次 `tanh`，几乎可忽略不计。

**3. 后归一化（Post-norm，在预归一化基础上叠加）**

Gemma 2 在每个子层**之前**（标准 pre-norm）和**之后**各应用一次 RMSNorm。Post-norm 随深度增加保持残差流有界。与软截断结合，即使在 BF16 下也能在 27B 规模实现显著稳定的训练动态，无需大量 loss scaling 调试。

### 知识蒸馏

对于 2B 和 9B 模型，每个训练步骤计算两项损失的组合：

1. **标准交叉熵损失**（对真实 next-token 标签）：
   ```
   L_CE = -log p_student(x_{t+1} | x_{≤t})
   ```

2. **对教师 logit 的 KL 散度**：
   ```
   L_KD = KL(p_teacher(· | x_{≤t}) || p_student(· | x_{≤t}))
        = Σ_v p_teacher(v) log [p_teacher(v) / p_student(v)]
   ```

总损失为加权组合：`L = α · L_CE + (1 − α) · L_KD`。KL 项在完整词汇表分布上计算——不只是 top-k token——保留了教师完整的不确定性信号。

教师（Gemma 2 27B）在学生训练全程保持冻结。教师与学生看到相同的 token 序列；教师 logit 通过前向传播计算，按 batch 缓存。由于教师是 9B 学生的约 3 倍大，教师推理约增加 33% 的单步计算开销。

**为何不用 9B 蒸馏 2B？** 使用更强的教师可产生更好的学生——教师的不确定性估计更精准，合理替代项上的概率质量更大。用 27B 同时作为 2B 和 9B 的教师，使两者的学习信号均最大化。

### 模型规格与训练规模

| 规格 | 参数量 | 层数 | 训练 token | 上下文 |
|---|---|---|---|---|
| 2B | 2.6B | 26 | ~2T | 8,192 |
| 9B | 9.2B | 42 | ~8T | 8,192 |
| 27B | 27.2B | 46 | ~13T | 8,192 |

27B 是唯一纯 next-token prediction 训练的模型。2B 和 9B 的训练流程在每次学生前向传播的同时运行 27B 教师推理。Token 预算少于 Llama 3（后者 8B 模型训练了 15T token），用数据规模换取蒸馏信号。

## 工程权衡

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 2B / 9B 蒸馏训练 | 每参数能力显著提升；教师不确定性信号降低了对数据规模的依赖 | 单步训练计算增加约 33%（教师前向传播）；教师需常驻或可被推理访问；学生训练进度与教师可用性耦合 |
| 交替局部/全局注意力 | 训练上下文长度下 KV cache 减少约 50%；局部层更快（每次 attention 序列长度有界） | 全局层仍随 L 线性增长；长上下文质量依赖全局层分布是否合理；局部窗口超过 4,096 token 的依赖关系不可见 |
| Logit 软截断 | BF16 训练稳定，无需 loss scaling 调参；防止 attention logit 爆炸 | 破坏标准点积注意力 kernel 优化（分数必须经过 tanh 才能做 softmax，无法使用将 softmax 与输出累加融合的标准 FlashAttention kernel） |
| Post-norm + Pre-norm | 深层残差流稳定；更好的梯度流 | 每层多一次 RMSNorm；约 2% 的小额计算开销 |
| 全程 GQA | 相比 MHA 减少 KV cache；decode 速度更快 | KV head 数更少意味着 key/value 表示更粗粒度；极少 KV head 数时存在质量折中 |

关于软截断与 FlashAttention 的交互：attention 路径中的 `tanh` 阻止了直接使用标准 FlashAttention 融合 kernel（假设 scaled dot-product 可直接做 softmax）。下游服务框架需要自定义 kernel（TPU 上的 Pallas、将 tanh 融合进 attention score 路径的 CUDA kernel）才能实现完整效率，这是不可忽视的实现成本。

## 实验与结果

**跨规格质量（相对参数量）：**
- Gemma 2 27B 在 MMLU、GSM8K、HumanEval 及多项推理 benchmark 上与 Llama 3.1 70B 持平或更优——参数量仅为后者的约 38%。
- Gemma 2 9B 在多项 benchmark 上与 Llama 3 70B 相当，明显优于 Llama 3 8B——体现了蒸馏增益。
- Gemma 2 2B 发布时是该参数量级最强模型，在多项任务上与上一代 7B 模型相当。

**蒸馏消融（来自技术报告）：**
- 同等 token 预算下，带蒸馏训练的 9B 模型性能显著优于不带蒸馏的 9B（纯 next-token prediction）。
- 增益在推理任务（GSM8K、MATH）上最大——教师对合理解题路径的不确定性提供了硬标签遗漏的密集信号。

**推理效率：**
- batch size 为 1 时，交替注意力设计使局部层的 KV cache 在典型序列长度下可容纳于 GPU 的 L2/L3 缓存，减少这些层的 HBM 读取。
- 27B 的吞吐量与 Llama 3.1 70B 相当，尽管参数更少——服务更快，因为每 token 的权重带宽更低。

## 可复现性说明

- 所有三个规格的 Gemma 2 权重（基础版 + 指令微调版）已在 HuggingFace 以 Gemma 许可证公开发布（在一定规模以下可用于研究和商业用途）。
- 训练代码未公开。技术报告对蒸馏流程（α 权重、教师缓存策略、训练课程）有高层次描述，但不足以从头精确复现。
- **HuggingFace Transformers** 原生支持 Gemma 2（`AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")`），软截断已内置实现。
- **vLLM 和 SGLang** 均支持 Gemma 2 推理。attention 软截断需要修改后的 attention kernel；两个框架均已实现。在典型 batch size 下，吞吐量在无截断 attention 的 10–15% 以内。
- **服务注意事项**：交替局部/全局注意力模式要求 KV cache 管理器正确处理——局部层的 cache 可按窗口大小驱逐/限定，但该优化并非所有服务框架均已实现。朴素实现将所有层视为全局层，导致内存节省无法落地。

## 评注

Gemma 2 最持久的贡献在于证明了：**对小模型而言，用强教师蒸馏比单纯增加训练 token 更能有效利用训练计算量**。9B 模型远超其参数量级，正是因为它从 27B 模型的完整概率分布中学习，而非仅从硬数据标签学习。这并非新思想（Hinton 等，2015 年《蒸馏神经网络中的知识》），但在这种规模下——在线进行、覆盖完整词汇表分布、历经数十亿训练步骤——证明了它作为高效开源模型开发实用方案的可行性。

Logit 软截断技术被低估了。它在架构上开销极低（每次 attention 和 output softmax 多一次 tanh），解决了真实存在的 BF16 训练脆弱性，而代价（破坏标准 FlashAttention 融合）通过自定义 kernel 可以管理。预计未来在不依赖大量 loss scaling 基础设施推进 BF16 训练的模型中会看到这种或类似的稳定化技术。

交替局部/全局注意力是更具争议的设计选择。KV cache 节省是真实存在的，但 4,096 token 的窗口形成了硬边界——单跳需要关注超过 4,096 token 的任务（超长文档推理、检索、扩展代码上下文）完全依赖全局层。Mistral 7B v0.1（2023）采用了相同设计；Mistral 7B v0.2 将其移除改为全上下文，表明这一权衡对任务分布敏感。Gemma 2 的 8k 训练上下文意味着对其目标用例而言，这在实践中影响有限。

对从业者而言：在需要 7B–13B 占用且对质量要求较高的部署场景中，Gemma 2 9B 目前是最佳选择之一——蒸馏优势可量化，与 70B 相比的服务效率提升显著。指令微调版本在其参数量级中对齐质量尤为出色。

## 参考文献

- [1] Gemma Team. _Gemma 2: Improving Open Language Models at a Practical Size._ arXiv:2408.00118, 2024.
- [2] Gemma Team. _Gemma: Open Models Based on Gemini Research and Technology._ arXiv:2403.08295, 2024.
- [3] Hinton et al. _Distilling the Knowledge in a Neural Network._ arXiv:1503.02531, 2015.
- [4] Ainslie et al. _GQA: Training Generalized Multi-Query Transformer Models._ arXiv:2305.13245, 2023.
- [5] Beltagy et al. _Longformer: The Long-Document Transformer._ arXiv:2004.05150, 2020.（滑动窗口注意力）
- [6] Jiang et al. _Mistral 7B._ arXiv:2310.06825, 2023.（交替局部/全局注意力的先例）
- [7] Zhang et al. _RMS Norm._ arXiv:1910.07467, 2019.
