# Switch Transformers & GShard — 将 MoE 扩展至万亿参数

- **作者 / 机构**：Fedus、Zoph、Shazeer 等（Google Brain）—— Switch Transformers；Lepikhin 等（Google）—— GShard
- **发表时间**：GShard：arXiv:2006.16668，2020；Switch Transformers：arXiv:2101.03961，JMLR 2022
- **链接**：[Switch Transformers](https://arxiv.org/abs/2101.03961) | [GShard](https://arxiv.org/abs/2006.16668)

## 一句话总结

GShard（2020）在 TPU 集群上训出首个 600B 参数 MoE 模型，采用 top-2 路由、局部组分发和容量因子来处理专家溢出。Switch Transformers（2022）进一步将路由简化为 top-1（每个 token 只路由到一个专家），证明在谨慎初始化和 FP32 路由器精度下 top-1 训练稳定，并将规模推至 1.6T 参数（Switch-C，2048 个专家），在相同算力预算下比 T5-XXL 快 7 倍。这两篇论文共同确立了此后所有 MoE 系统（DeepSeekMoE、Mixtral、Llama 4）沿用的工程词汇体系。

## 背景与动机

稠密 Transformer 的参数量和算力同步增长——模型翻倍，每个 token 的 FLOPs 也翻倍。MoE（混合专家）打破了这一耦合：每个 token 只激活一部分参数，打开了一个全新的扩展维度。这个想法可追溯到 Jacobs et al.（1991）和 Shazeer et al. 2017 年的稀疏门控 MoE，但 2017 年的工作存在训练不稳定问题，从未进入大规模生产部署。

到 2020 年，高带宽互联的 TPU Pod 已可容纳数百个设备——MoE 所需的硬件时机成熟。阻碍 MoE 实用化的两大问题是：

1. **负载不均**：路由呈现"富者愈富"的崩溃现象，少数专家吸引大多数 token，其余专家闲置，分布式训练中出现尾部卡顿（straggler）。
2. **训练不稳定**：路由网络的 argmax 不可微，路由决策翻转时产生大梯度更新，容易导致训练发散。

GShard 在 600B 规模下解决了 top-2 路由的这两个问题。Switch Transformers 重新审视了前提假设，发现 top-1 路由能绕开许多不稳定性，同时保持竞争力。

## GShard：top-2 路由训练 600B 模型

### 架构与 TPU 布局

GShard 在每隔一层的 Transformer FFN 层替换为 MoE。模型分片使每个 TPU 设备恰好承载每个 MoE 层的一个专家。以 2048 个 TPU、2048 个专家为例，每个专家精确驻留在一个设备上——这是专家并行最自然的形态。

关键设计是**局部组分发（local group dispatching）**。批次中的 token 被划分为多个组（如 4096 个 token → 32 组，每组 128 个）。路由计算在每个组内本地完成：argmax 仍面向全部 2048 个专家，但容量统计在组内进行。这将跨设备通信的范围限制在组内路由决策所覆盖的范围，而非全局排序。

### 容量因子与 token 丢弃

每个专家有硬性容量上限：

```
专家容量 C = (每组 token 数 / 专家数) × 容量因子
```

`容量因子 = 1.0` 且路由完全均衡时，每个专家恰好分到其份额。实际路由不均衡，因此 GShard 默认 `容量因子 = 1.25`，保留 25% 余量。若某 token 的第一选择专家已满，则派发到第二选择；若第二选择也满，该 token 被**丢弃**：绕过专家 FFN，通过残差流直接传递（恒等旁路）。丢弃的 token 仍通过非 MoE 层参与模型输出。

GShard 引入**辅助负载均衡损失**以防止路由崩溃，在组级别惩罚不均衡，推动路由器趋向均匀利用。

### token 丢弃的工程现实

token 丢弃不是理论上的边界情况——在生产批量下，`容量因子 = 1.0` 时，训练早期丢弃率可达 5–15%。关键洞察在于丢弃是**优雅降级**：模型仍输出合法结果，只是经过较少专家处理。推理阶段，对质量敏感的应用通常将容量因子调高至 1.5–2.0，以接受额外的填充开销换取更少的丢弃。

## Switch Transformers：top-1 路由扩展至 1.6T 参数

### top-1 简化的背后逻辑

Switch Transformers 提出一个出人意料的主张：每个 token 只路由到**一个**专家。先前工作认为 top-2 是稳定性所必需的（第二个专家作为备份）。Switch 证明并非如此，且 top-1 带来直接收益：

- **all-to-all 通信量减半**（每个 token 只穿越一次设备边界）。
- **容量统计更简单**：token 要么进入其专家，要么被丢弃，无需二次派发逻辑。
- **路由开销更低**：路由器对 `n` 个专家做 softmax 后取 argmax，top-1 退化为简单的 `max`，无额外调度逻辑。

路由器为 token `x` 计算专家概率分布：

```
h(x) = W_r · x          （线性投影，路由器权重）
p_i(x) = softmax(h(x))_i
route(x) = argmax_i p_i(x)
```

token 被发送至 `route(x)` 对应的专家，其输出乘以 `p_{route(x)}(x)` 后加回残差流。

### 辅助负载均衡损失

防止路由崩溃的核心训练机制：

```
L_aux = α · n · Σᵢ fᵢ · Pᵢ
```

其中：
- `n` = 专家数
- `fᵢ` = 当前批次中被派发到专家 `i` 的 token 比例（直通估计，不可微）
- `Pᵢ` = 路由器分配给专家 `i` 的总概率质量比例（可微）
- `α` = 损失系数（通常为 1e-2；过大会使路由均匀化，过小允许路由崩溃）

乘积 `fᵢ · Pᵢ` 在 token 比例和概率质量均匀分布时最小。`fᵢ` 是停止梯度的计数——提供哪些专家过载的信号，但计数本身不参与梯度传播。`Pᵢ` 是可微的调节手柄：路由器学会将概率质量从拥挤的专家重新分配。

### Switch 的专家容量计算

```
C = (批次 token 数 / 专家数) × 容量因子
```

`容量因子 = 1.0`：完全均衡时零浪费，任何不均衡都导致丢弃。
`容量因子 = 1.25`：每个专家缓冲区增加 25% 内存开销，提供分配余量。
`容量因子 = 2.0`：余量充足，缓冲内存开销高。

以 128 个专家、批次 512 个 token 为例：`C = (512/128) × 1.25 = 5`。若专家 0 吸引了 9 个 token，则 4 个被丢弃。

### 训练稳定性修复

Switch 之前的 MoE 模型频繁发散。Switch 识别出两个根本原因并逐一修复：

1. **路由器精度**：即使主体模型使用 BF16，路由器线性投影保持 FP32。对数百个专家做 softmax 在数值上很敏感；BF16 在 logit 差异小时产生下溢，导致路由崩溃。代价可忽略——路由器参数量相对于专家权重极小。

2. **权重初始化**：路由器权重以更小的比例初始化（默认值的 0.1 倍）。初始路由权重过大 → 初始路由概率差异大 → 少数专家从第一步起就占据主导 → 辅助损失面对已固化的不均衡状态。

### 规模：Switch-C

Switch-C 使用 2048 个专家，总参数 1.6T，每个 token 激活约 1B 参数。与同等 FLOP 预算下的 T5-XXL（11B 稠密模型）相比，实现了 **7 倍训练加速**。在 SuperGLUE 基准上质量具有竞争力，尽管在某些设置下稀疏模型需要更多步骤才能达到稠密模型等效的困惑度。

"叠加假说"在 Switch-C 中隐然成立：更多专家 = 更多独立参数组 = 可存储更多不同的知识模式。每个专家学会处理特定的 token 类型、领域或句法模式；路由路径的组合空间使模型能存储远多于同等 FLOP 稠密模型的关联。

## 工程权衡

| 决策 | 收益 | 代价 |
|---|---|---|
| top-1 路由（Switch）vs. top-2（GShard） | all-to-all 通信减半；调度逻辑更简单 | 失去 token 丢弃时的二次选择；专家数少时与 top-2 有小幅质量差距 |
| 容量因子 > 1.0 | 路由不均衡时优雅处理；token 丢弃减少 | 每个专家额外缓冲内存；专家负载长尾处存在未充分利用的容量 |
| token 丢弃（旁路连接） | 无论不均衡程度如何，专家负载有界；无尾部等待 | 被丢弃的 token FFN 处理质量下降；推理时难以直接观测的软性失效 |
| FP32 路由器 + BF16 权重 | 多专家路由 softmax 数值稳定 | 双精度管理；路由激活值小幅内存开销 |
| 辅助负载均衡损失 | 训练期间防止路由崩溃 | `α` 调参繁琐；过大 → 均匀路由（专家无差异化）；过小 → 路由崩溃 |
| 专家并行（每设备一个专家） | 总参数随设备数近线性扩展 | 每个 MoE 层的 all-to-all 通信开销；网络带宽成为新瓶颈 |

## 实验与结果

**GShard（600B）：**
- 在 100+ 语言的神经机器翻译上训练的 600B MoE 模型。
- 容量因子 1.25 的 top-2 路由；2048 个专家分布在 2048 个 TPU 核上。
- 在 WMT 多语言基准上超越 100B 稠密模型；质量提升接近线性直到 600B。

**Switch Transformers：**
- Switch-Base（7B 总参数，12 个专家）：以 1/7 的每 token FLOPs 匹配 T5-Base。
- Switch-Large（26B 总参数，128 个专家）：收敛更快；以约 1/4 的算力成本接近 T5-Large 质量。
- Switch-C（1.6T 总参数，2048 个专家）：vs T5-XXL 训练速度提升 7 倍；当时 SuperGLUE 最高分。
- 消融实验：top-1 vs top-2 在下游任务上差距在 1% 以内，top-1 因通信减半始终更快。

## 评述

这两篇论文最持久的贡献不是特定的路由算法，而是**工程词汇体系**：容量因子、token 丢弃、辅助损失、专家并行。2021 年后构建的每个 MoE 系统都沿用这些术语和调节旋钮。其中容量因子是一个被低估的概念：它将一个难解的组合分配问题（将 token 分配给专家且不溢出）转化为一个软性工程权衡（选择为余量烧多少内存 vs. 丢弃多少 token）。

从系统角度看，最重要的洞察是 **token 丢弃是可以接受的**。Switch 之前，普遍假设每个 token 必须获得完整处理；丢弃被视为正确性违反。Switch 通过实验证明，推理阶段 1–5% 的丢弃率对下游指标几乎无影响。这解锁了一整类有界成本路由算法，否则这些算法要么需要动态可变长度批处理，要么需要代价高昂的负载重平衡过程。

辅助损失调参问题是 Switch/GShard 方案的长期局限：`α` 这个标量必须针对每个新的模型规模重新调整——过小导致路由崩溃（收敛但泛化差），过大导致均匀路由（模型失去专业化）。DeepSeek 后续通过偏置项实现的无辅助损失路由（见 [DeepSeekMoE](../../deepseek/moe/zh.md)）正是对这一脆弱性的直接回应。[Mixtral](../../mistral/mixtral/zh.md) 和 [Llama 4](../../meta/llama4/zh.md) 都继承了这一基本框架，但以不同方式调整平衡——Mixtral 使用无丢弃的 top-2，Llama 4 使用带交错 MoE 层的 top-2。

## 参考文献

- [1] Lepikhin et al. _GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding._ arXiv:2006.16668，2020。
- [2] Fedus, Zoph, Shazeer. _Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity._ JMLR，2022。arXiv:2101.03961。
- [3] Shazeer et al. _Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer._ ICLR 2017。
- [4] DeepSeekMoE：见 [../../deepseek/moe/zh.md](../../deepseek/moe/zh.md)
- [5] Mixtral of Experts：见 [../../mistral/mixtral/zh.md](../../mistral/mixtral/zh.md)
- [6] Llama 4：见 [../../meta/llama4/zh.md](../../meta/llama4/zh.md)
