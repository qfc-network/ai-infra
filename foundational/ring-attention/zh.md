# Ring Attention（与上下文并行 CP）

- **作者 / 机构**：Liu、Zaharia、Abbeel（UC Berkeley）
- **发表时间**：2023-10
- **链接**：[Ring Attention (arXiv:2310.01889)](https://arxiv.org/abs/2310.01889) · [Blockwise Transformers (arXiv:2305.19370)](https://arxiv.org/abs/2305.19370) · [Striped Attention](https://arxiv.org/abs/2311.09431)

## 一句话总结

超长上下文下，**即便有 FlashAttention**，attention 的显存也塞不进单卡。Ring Attention 沿**序列轴**把张量切到多卡，K/V 块在 ring 上传递：每卡对当前手上的 K/V 切片算一次局部 attention，再把切片转给下一个、同时接收上一卡的切片。配合 blockwise softmax，最终算出**完全精确**的 attention，算力与 ring 通信完全重叠。Gemini 1.5 的百万 token 上下文、NVIDIA 的 Context Parallelism、Llama 3 的 CP 轴，以及现今多数长上下文训练栈都采用了这条路线（及其变体）。

## 背景与动机

FlashAttention 解决了单卡上 attention 的带宽瓶颈。当 `N` 到 128k、1M+ 时又出现两个新问题：

1. **激活显存仍随 `N` 增长**——即便有在线 softmax，单卡也存不下 Q、K、V 的全序列切片。
2. **单序列内缺乏并行**——Megatron 的 TP 切的是 hidden 维而非序列；FSDP 切的是参数而非激活。两者都帮不上长序列。

此前的办法（blockwise 计算、梯度 checkpoint）只降显存但不"分布"它。Ring Attention 是第一种干净的方案，可以**把一条序列跨多卡并行**，同时保持精确 attention。

## 核心方法

### 序列切分 + ring 旋转

序列切成 `P` 等份，chunk `i` 放到 device `i`。每卡持有自己的 Q、K、V 切片。算 attention 时，每卡最终需要看到完整序列的 K、V。

- **环形拓扑**：`0 → 1 → ... → P-1 → 0`。
- 第 `s` 步时，device `i` 一直持有自己的 Q，手上的 K/V 块最初属于 device `(i − s) mod P`。
- Device `i` 用 **blockwise FlashAttention + 在线 softmax** 对自己的 Q 与当前 K/V 块算局部 attention。
- 同时**异步**把当前 K/V 块发给 `i+1`，**接收**来自 `i-1` 的下一块。
- `P` 步后，每个 Q 都看过所有 K/V；跨步的在线 softmax 归一化给出精确 attention。

### 重叠

每块计算时间 ≈ 每块通信时间时，ring 接近纯算力 bound，与 `P` 无关。这是 Ring Attention 的定义特性：**加设备线性延长上下文，wall time 不增**。

### 因果 mask 与变体

- 朴素环 + 因果 mask 会让序列后半段的卡大量空转（上三角块为零）。
- **Striped Attention**（同作者后续）把序列位置置换，使每个 ring 步在各卡上工作量均衡。因果模型长上下文下吞吐约 2×。
- **NVIDIA Context Parallelism**（Megatron-Core、Llama 3 所用）：生产级 ring，带 Striped 式均衡，已与 TP/PP/FSDP 集成。

### 与其他并行的组合

CP 与 TP/PP/DP/FSDP **正交**，并行总量：

```
total_gpus = TP × PP × CP × DP
```

Llama 3 在长上下文阶段用 `CP=2`；前沿实验室据称走得更远。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 序列跨卡切分 | 上下文随 `P` 扩展；单卡分片显存恒定 | 短序列下也要 `P` 卡（通常只在长上下文阶段启用） |
| Ring 通信 | 调好后与算力完全重叠 | 对不均衡敏感；朴素因果 mask 会腰斩效率 |
| 精确 attention | 无质量近似 | 与稀疏注意力技巧组合需谨慎 |
| 与 TP/PP/DP 正交 | 可作为第 4 并行轴插入 | 拓扑与调度与其他轴相互作用，配置更复杂 |
| Striped 置换 | 因果场景工作量均衡 | 位置 ID 需重排；部分 kernel 要改写 |

## 实验与结果

- 原论文：32 张 TPU v4 pod 上跑**百万级 token**的精确 attention；随卡数近线性延长序列。
- Striped Attention：因果 LM 在 128k–1M 上相对朴素环约 2×。
- Gemini 1.5 Pro 报告 1M（后为 10M）上下文；架构细节未公开，但 Ring/CP 是被业界默认的机制。
- Llama 3 405B：8k→128k 上下文扩展阶段使用 `CP=2`。

## 复现要点

- 参考实现：原论文 JAX 代码、`ring-flash-attention`（PyTorch，Zilin Zhu 版本）、NVIDIA Megatron-Core CP。
- 正确性验证：短序列上与单卡 FlashAttention 对齐，FP16/BF16 容差内数值一致。
- 生产部署嵌在大型训练框架里（Megatron-Core、MaxText、torchtitan）。

## 个人评注

Ring Attention 是对一个看似算法上无解的问题的优雅回答：**在不近似 attention 的前提下，把一条序列并行到多张 GPU**。关键 insight——在线 softmax 对 K/V 块是可结合的，所以块可以乱序到达——FlashAttention 其实已经给了；Ring Attention 做的是把这个性质配上环形拓扑。这一技巧悄悄地成了必需品：2024–2025 间任何可信的 "1M 上下文" 声明背后都离不开 CP。向前看，有意思的前沿在**异构 ring 拓扑**（NVLink 快跳 + 跨节点慢跳混合）、**长上下文解耦部署**（大 CP 环做 prefill，decode 放别处），以及与 MLA 这类长上下文友好架构的组合。训练或 serving 超过 32k 上下文的，都该理解它——绕不过去。

## 参考

- [1] Liu, Zaharia, Abbeel. _Ring Attention with Blockwise Transformers for Near-Infinite Context._ arXiv:2310.01889, 2023.
- [2] Liu, Abbeel. _Blockwise Parallel Transformer for Large Context Models._ arXiv:2305.19370, 2023.
- [3] Brandon et al. _Striped Attention._ arXiv:2311.09431, 2023.
- [4] NVIDIA. _Context Parallelism in Megatron-Core._ Docs, 2024.
