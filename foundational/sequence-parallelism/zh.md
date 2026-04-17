# 序列并行 — Ring Attention、DeepSpeed Ulysses 与 Megatron-CP

- **核心论文**：Liu 等（2023）、Jacobs 等（2023）、NVIDIA Megatron-Core（2024）
- **发表时间**：Ring Attention — 2023-10 · DeepSpeed Ulysses — 2023-09 · Megatron-CP — 集成于 Megatron-Core 2024
- **链接**：[Ring Attention (arXiv:2310.01889)](https://arxiv.org/abs/2310.01889) · [DeepSpeed Ulysses (arXiv:2309.14509)](https://arxiv.org/abs/2309.14509) · [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)

## 一句话总结

单 GPU 注意力的上限由 HBM 容量决定：在 128k 个 token、BF16 精度下，单个注意力层的 KV 张量就需要数十 GB，还不计查询、输出和激活值。序列并行（SP）将序列维度分布到多个 GPU 上，使长上下文训练的 GPU 数量线性增长，而非序列长度平方增长。目前涌现了三种架构上截然不同的方案：**Ring Attention**，将 KV 块在设备环上传递；**DeepSpeed Ulysses**，通过全到全通信在头维度上转置问题；**Megatron-CP（上下文并行）**，加入因果负载均衡并与 Megatron 现有 TP/PP/DP 并行机制干净集成。最佳选择取决于序列长度、注意力头数量以及已使用的并行度配置。

## 为何需要序列并行

注意力计算复杂度在计算上为 $O(S^2)$，Q、K、V 张量的显存占用为 $O(S)$，但相关常数很大。对于 $d_{\text{model}} = 8192$、$n_{\text{heads}} = 64$、$n_{\text{kv\_heads}} = 8$、BF16 精度的模型：

**每层每 token 的 KV 缓存大小**：2（K 和 V）× $d_{\text{head}}$ × $n_{\text{kv\_heads}}$ × 2 字节 = 2 × 128 × 8 × 2 = 4096 字节 = 4 KB。

**训练时单层、序列长度 $S = 1$M 的 KV 激活值**：

$$\text{每层 KV 显存} = 2 \times S \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{字节} = 2 \times 10^6 \times 8 \times 128 \times 2 \approx 4 \text{ GB}$$

对于 80 个 Transformer 层，单条训练样本的 KV 激活值高达 320 GB——超过四块 H100 的 HBM 总量——还未计查询张量、注意力分数、前馈激活或优化器状态。

FlashAttention（参见 [../flash-attention/zh.md](../flash-attention/zh.md)）通过在反向传播中重计算注意力分块，消除了 $O(S^2)$ 的激活值显存，但并未减少 Q、K、V 本身所需的 $O(S)$ 显存。在主流模型上，当 $S \geq 128$k token 时，即使有 FlashAttention，仅 Q/K/V 存储就会耗尽单 GPU 的 HBM。因此序列并行不是优化手段，而是长上下文训练的硬性需求。

需要注意，Megatron-LM v3 的序列并行（参见 [../megatron-lm/zh.md](../megatron-lm/zh.md)）是相关但不同的技术：它在张量并行组内将非矩阵乘激活值（dropout、LayerNorm 输出）沿序列轴分片。该方案减少了非注意力操作的激活值显存，但并不分布注意力计算本身。本词条描述的三种方案都直接针对注意力计算。

## 方案一：Ring Attention

Ring Attention（详见 [../ring-attention/zh.md](../ring-attention/zh.md)）使用环形拓扑将序列分布到 $P$ 个设备上。设备 $i$ 持有 Q、K、V 的第 $i$ 个块，各含 $S/P$ 个 token。为计算完整注意力，每个设备最终需要处理其 Q 块与所有 K/V 块。

算法进行 $P$ 轮旋转：
1. 设备 $i$ 使用带在线 softmax 的逐块 FlashAttention，计算本地 Q 与当前缓冲区中 K/V 块的偏注意力，累积偏结果。
2. 同时，异步地将当前 K/V 缓冲区发送给设备 $(i+1) \bmod P$，并从设备 $(i-1) \bmod P$ 接收下一个 K/V 缓冲区。
3. 经过 $P$ 轮后，每个 Q 块已关注所有 K/V 块；在线 softmax 归一化产生精确的完整注意力。

**显存分析**：每个设备持有 $S/P$ 个 token 的 Q、K、V——每 GPU 总计 $O(S/P)$，随并行度线性扩展。

**通信量**：每步发送一个大小为 $2 \times (S/P) \times n_{\text{kv\_heads}} \times d_{\text{head}}$ 的 K/V 块至相邻设备。经过 $P$ 步，每个设备发送的总数据量为 $2 \times S \times n_{\text{kv\_heads}} \times d_{\text{head}}$——与完整序列 KV 成正比，而非本地块。这是点对点发送/接收，不是真正的集合通信，但总量与 KV 张量上的全到全通信相当。

**因果注意力与负载均衡**：使用因果掩码时，持有最后序列块的设备 $P-1$ 需要对所有 $P$ 个 K/V 块计算注意力，而持有第一块的设备 0 只需一个块。这造成严重的负载不均衡。Striped Attention 和 NVIDIA 的上下文并行通过交错分配 token 位置（奇偶交错）解决这一问题：设备 $i$ 持有 token $\{i, i+P, i+2P, \ldots\}$ 而非连续块 $[iS/P, (i+1)S/P)$。这样每个设备在每轮环操作时承担等量的因果计算，使因果模型在长上下文下的有效吞吐量翻倍。

**重叠需求**：Ring Attention 的核心性能特性是计算与通信重叠。若单个 K/V 块的计算时间约等于传输时间，环操作以计算瓶颈速度运行。这要求：
- 每设备序列长度足以使一个块使张量核心饱和。
- 网络带宽足以在下一计算步开始前完成 K/V 块传输（NVLink 900 GB/s 双向，或 InfiniBand HDR 200 Gb/s，各在不同块大小下有不同权衡）。

## 方案二：DeepSpeed Ulysses

Ulysses（来自 [../../microsoft/deepspeed/zh.md](../../microsoft/deepspeed/zh.md)）采用根本不同的方式：不是分布序列维度后旋转 K/V，而是通过全到全集合通信将数据布局转置，使每个 GPU 对头的子集执行完整注意力。

**配置**：将 $N$ 个注意力头分配到 $P$ 个设备上，每个设备拥有 $N/P$ 个头。约束条件为 $P \leq N$（头数是 Ulysses 并行度的上限）。

**前向传播**：
1. 注意力之前，每个设备持有完整序列，但仅持有其 TP 分片对应的 Q、K、V 头投影子集。一次**全到全集合通信**重新分配张量：全到全之后，每个设备持有所有 $N$ 个头，但仅持有 Q、K、V 的 $S/P$ 个 token。
2. 每个设备使用标准 FlashAttention，在其 $S/P$ token 切片上对所有 $N$ 个头计算完整自注意力。
3. 第二次**全到全集合通信**完成转置还原：每个设备恢复持有完整序列 $S$，但仅持有输出的 $N/P$ 个头。

**每次全到全的通信量**：被重新分配的张量为 $S \times d_{\text{model}}$（Q、K 或 V 投影输出）。每次全到全总字节数 = $S \times d_{\text{model}} \times \text{字节数}$。每个注意力层两次全到全（一前一后），单层总通信量为 $2 \times S \times d_{\text{model}} \times \text{字节数}$。

对于 $S = 128$k、$d_{\text{model}} = 8192$、BF16：$2 \times 128000 \times 8192 \times 2 \approx 4.2$ GB/层——数量可观，但明确且与并行度 $P$ 无关（总量不变，仅分配方式改变）。

**重叠**：Ulysses 的全到全是全局同步集合通信，不是点对点传输，比 Ring 的异步发送/接收更难与计算重叠。实践中实现方案通过将全到全与 Q/K/V 投影流水化实现部分重叠。

**核心约束**：$P \leq n_{\text{heads}}$。对于使用 GQA 且 KV 头数少的模型（如 Llama 3 的 $n_{\text{kv\_heads}} = 8$），若仅对 KV 头应用 Ulysses，有效约束为 $P \leq n_{\text{kv\_heads}} = 8$。这限制了 Ulysses 在激进 KV 头缩减 GQA 模型上的可扩展性。

**显存分析**：第一次全到全之后，每个设备以完整头宽度处理 $S/P$ 个 token。注意力层的激活值显存为 $O(S/P \times d_{\text{model}})$——与 Ring Attention 每 token 显存效率相同，但全到全缓冲区需要同时持有完整序列 Q/K/V，形成瞬时峰值。

## 方案三：Megatron-CP（上下文并行）

Megatron-CP 将序列并行作为 Megatron-Core 四维并行层级的一等公民：张量并行（TP）× 流水线并行（PP）× 上下文并行（CP）× 数据并行（DP）。CP 轴以能与其他轴干净组合的方式处理注意力的序列维度。

**算法**：Megatron-CP 使用全收集 + 规约散射通信模式，而非纯环旋转。注意力计算流程如下：
1. 每个设备持有 $S/P$ 个 token（其 CP 分片）的 Q、K、V。
2. 一次**全收集**将完整 K、V 张量（大小 $S$）聚合到每个设备。每个设备此时有本地 Q（$S/P$ 个 token）和完整 K、V（$S$ 个 token）。
3. 每个设备使用 FlashAttention 计算 $\text{Attention}(Q_{\text{local}}, K_{\text{full}}, V_{\text{full}})$，产生 $S/P$ 个输出 token。
4. 由于每个设备直接产生其 $S/P$ 输出切片，输出无需规约散射。

这在逻辑上等价于 Ring Attention，但使用显式全收集而非旋转——代价是全收集期间每个设备需要临时持有 $S$ 个 token 的 K、V，产生 $P$ 倍的 K/V 显存峰值。实践中，通过将 K/V 全收集与 FlashAttention 核融合（使用 K/V 分块），完整 $S$ 长度的 K/V 不会同时物化，从而管理这一峰值。

**因果负载均衡**：Megatron-CP 实现了 token 位置的奇偶交错，与 Striped Attention 类似。设备 $i$ 拥有 token $\{i, P+i, 2P+i, \ldots\}$ 和 $\{S-1-i, S-1-P-i, \ldots\}$（序列的前向与后向两半交错），确保在因果掩码下每个设备计算等量的 FLOPs。

**与四维并行的集成**：在 CP 组内，所有设备共享同一 TP 组。TP 对 QKV 投影和注意力输出投影的全规约模式在 CP 组内运行，不受影响。PP 跨 CP 组操作（每个 PP 阶段在一个 CP 度下运行）。DP 是最外层复制轴。Llama 3 405B（参见 [../../meta/llama3/zh.md](../../meta/llama3/zh.md)）在长上下文训练阶段使用 $\text{TP}=8, \text{PP}=16, \text{CP}=2, \text{DP}=128$，共 $8 \times 16 \times 2 \times 128 = 32768$ 块 GPU。

**CP 并行度 $N$ 下的显存分析**：

$$\text{每层 KV 缓存} = 2 \times \frac{S}{N} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{每值字节数}$$

$$\text{所有层总 KV 显存} = N_{\text{layers}} \times 2 \times \frac{S}{N} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{字节数}$$

对于 Llama 3 405B（$N_{\text{layers}} = 126$、$n_{\text{kv\_heads}} = 8$、$d_{\text{head}} = 128$、BF16、$S = 128\text{k}$、$N = 2$）：

$$126 \times 2 \times \frac{128000}{2} \times 8 \times 128 \times 2 \approx 41 \text{ GB/GPU}$$

若不使用 CP（$N=1$），这将是 82 GB——在不计模型权重、优化器状态或前馈激活的情况下，已超过单块 H100 的 80 GB HBM 预算。CP=2 使其可行；更高的 CP 度支持更长的上下文。

## 通信对比

| 维度 | Ring Attention | DeepSpeed Ulysses | Megatron-CP |
|---|---|---|---|
| 集合通信类型 | 点对点发送/接收（环步骤） | 每注意力层两次全到全 | 每注意力层一次 K/V 全收集 |
| 每层消息量 | $2 \times S \times n_{\text{kv\_heads}} \times d_{\text{head}}$（总量，分摊至 $P$ 步） | $2 \times S \times d_{\text{model}}$（每次全到全） | $S \times n_{\text{kv\_heads}} \times d_{\text{head}} \times 2$（全收集） |
| 头数约束 | 无（$P$ 可超过 $n_{\text{heads}}$） | $P \leq n_{\text{heads}}$（GQA 下 $P \leq n_{\text{kv\_heads}}$） | 无（$P$ 可超过 $n_{\text{heads}}$） |
| 因果负载均衡 | 需要 Striped Attention 变体 | 按设计均衡（基于头，非序列位置） | 内置奇偶交错 |
| 重叠策略 | 异步 K/V 发送与计算重叠 | 全到全与 QKV 投影重叠 | 全收集与 FlashAttention 分块融合 |
| 集成复杂度 | 独立；需要自定义通信内核 | 独立；与任意框架兼容 | 与 Megatron-Core TP/PP/DP 轴紧密耦合 |

## 选择建议

**使用 DeepSpeed Ulysses** 的场景：
- 序列长度中等（32k–256k token），且头数足以支持所需 $P$（即 $n_{\text{heads}} \geq P$ 或 GQA 下 $n_{\text{kv\_heads}} \geq P$）。
- 已在使用 DeepSpeed 进行 ZeRO 或 MoE 训练，希望最小化集成开销。
- 倾向于全局同步集合通信（比异步环通信更易推理正确性）。
- 模型未使用激进 GQA（KV 头数少会将 $P$ 上限压得过低）。

**使用 Ring Attention / Megatron-CP** 的场景：
- 序列长度极端（≥ 512k token），基于头数的并行度不足。
- 需要 $P > n_{\text{heads}}$，Ulysses 无法提供。
- 已处于 Megatron 生态系统，希望 CP 与现有 TP/PP/DP 配置无缝组合而无需重写训练循环。
- 因果模型是主要目标（两者均支持奇偶交错；在极端长度下负载均衡至关重要）。

**方案组合**：Ulysses 与 Ring Attention 互补，可以复合——Ulysses 负责头维度，Ring 负责剩余序列维度。总并行度为 $P = P_{\text{Ulysses}} \times P_{\text{Ring}}$，适合头数达 64+ 且序列长度达百万 token 的场景。DeepSpeed 已发布混合 Ulysses + Ring 的实验结果。

## 与四维并行的集成

实践中的长上下文训练需要同时使用所有四个并行轴：

```
total_gpus = TP × PP × CP × DP
```

CP 是解决序列显存瓶颈的轴；其他三个轴由模型规模（TP/PP）和吞吐量（DP）决定。通信模式设计为互不干扰：TP 全规约在节点内（NVLink）进行，PP 发送/接收跨节点组流水线边界，CP 环或全收集操作在 CP 组内通信（通常也在节点内或跨少量节点），DP 梯度全规约在整个 DP 轴上进行。

FlashAttention（参见 [../flash-attention/zh.md](../flash-attention/zh.md)）是任何序列并行注意力实现的前提：没有分块级注意力和在线 softmax，即使将 $S$ 跨设备分片后，完整序列的 $O(S^2)$ 注意力矩阵仍需物化，从而抵消分片的意义。本词条描述的三种 SP 方案均假设 FlashAttention 或等价的 IO-aware 内核可用。

## 可复现性说明

- Ring Attention 在开源 EasyContext 库和基于 JAX 的 LongContext 训练仓库中实现；核心算法直接，但因果负载均衡需要 Striped Attention。
- DeepSpeed Ulysses 在 DeepSpeed 中实现（参见 [../../microsoft/deepspeed/zh.md](../../microsoft/deepspeed/zh.md)），可通过 `deepspeed.sequence_parallel` 使用。
- Megatron-CP 在 Megatron-Core 中实现（参见 [../megatron-lm/zh.md](../megatron-lm/zh.md)），需要 Megatron 的并行配置。NVIDIA 为 Llama 系列模型发布了验证过的配置方案。
- 三种方案均需要 FlashAttention；FA2 是稳定训练的最低要求，H100 上推荐 FA3。
- 在某些 token 块比其他块承担明显更多因果计算的序列长度下，因果负载均衡的奇偶交错不是可选的——若不启用，部分 GPU 将停滞等待其他 GPU，有效吞吐量骤降。

## 参考文献

- [1] Liu 等。_Ring Attention with Blockwise Transformers for Near-Infinite Context._ arXiv:2310.01889，2023。
- [2] Brandon 等。_Striped Attention: Faster Ring Attention for Causal Transformers._ arXiv:2311.09431，2023。
- [3] Jacobs 等。_DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models._ arXiv:2309.14509，2023。
- [4] NVIDIA Megatron-Core。Context Parallelism 实现。github.com/NVIDIA/Megatron-LM，2024。
- [5] Meta AI。_The Llama 3 Herd of Models._ arXiv:2407.21783，2024。
- [6] 相关词条：[Ring Attention](../ring-attention/zh.md)、[Megatron-LM](../megatron-lm/zh.md)、[FlashAttention](../flash-attention/zh.md)、[Llama 3](../../meta/llama3/zh.md)、[DeepSpeed](../../microsoft/deepspeed/zh.md)
