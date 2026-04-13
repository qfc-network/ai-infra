# FlashAttention（1 / 2 / 3）

- **作者 / 机构**：Tri Dao 等（Stanford → Princeton → Together AI）
- **发表时间**：FA1 — 2022-05 · FA2 — 2023-07 · FA3 — 2024-07
- **链接**：[FA1 (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) · [FA2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) · [FA3 (arXiv:2407.08608)](https://arxiv.org/abs/2407.08608) · [代码](https://github.com/Dao-AILab/flash-attention)

## 一句话总结

注意力是**带宽瓶颈，不是算力瓶颈**——完整的 `N×N` 注意力矩阵会被写入 HBM 再读回，HBM 带宽是真正的卡点。FlashAttention 把计算重排成**分块、融合、IO-aware 的 kernel**，全程不物化完整矩阵：Q、K、V 分块流过片上 SRAM，softmax 用数值稳定的在线算法合并，backward 用重算换显存。FA1 让长上下文训练变得可行；FA2 改进了工作划分与并行度；FA3 围绕 Hopper 的 async TMA / WGMMA 重写并加入 FP8。如今几乎所有前沿实验室的注意力实现都用它。

## 背景与动机

标准注意力：

```
S = Q · K^T        # [N, N]
P = softmax(S)     # [N, N]
O = P · V          # [N, d]
```

中间 `S`、`P` 各占 `O(N²)` 显存，更关键是 `O(N²)` 次 HBM 读写。现代 GPU 的算力/带宽比严重失衡，标准注意力只能跑到峰值 FLOPs 的很小一部分——A100 上大约 3%。

此前的"高效注意力"（稀疏、线性、低秩）都在改数学以减少 FLOPs。那是找错了问题。FlashAttention **保持数学精确**，直接改攻存储层次。

## 核心方法

### FlashAttention 1（2022）

- **将 Q、K、V 切成能装进 SRAM（shared memory / 寄存器）的小块。**
- 对每个 Q 块，依次扫过 K/V 块；在 SRAM 内算注意力；用**在线 softmax**（Milakov & Gimelshein 2018）跨 K/V 块正确合并部分结果，全程不物化 `S` 与 `P`。
- **Backward**：不存大块的 `P`，只存输出 `O` 与每行的 softmax 统计量 `(m, ℓ)`（很小），反向时逐块重算 `S, P`。FLOPs 多了一点，HBM 流量大幅下降。

效果：**精确注意力，HBM 访问从 `O(N²)` 降到 `O(N)`**。A100 上 `N=8k` 有 3× 加速、10–20× 显存节省，序列越长增益越大。

### FlashAttention 2（2023）

FA1 在长序列上 GPU 利用率不高，因为并行维度是 `batch × heads`，两者都小时并行度不够。FA2 增加：

- **在序列维度上并行**，不只是 batch / heads。
- **改进 warp 间工作划分**——一个 warp 独占一个 Q 块扫描 K/V，省掉跨 warp 的 softmax 归并。
- **减少非 matmul 操作**——重排 scaling、masking 的位置。

结果：A100/H100 上较 FA1 快 ~2×，峰值 FP16 利用率 50–70%。

### FlashAttention 3（2024）

围绕 Hopper 特性彻底重写：

- **Async TMA**（Tensor Memory Accelerator）把全局→共享的拷贝与计算重叠。
- **WGMMA**（warpgroup matmul）取代 MMA；tile 更大、异步行为更好。
- **Ping-pong 调度**：把一块的 softmax 与下一块的 matmul 重叠——softmax 走 CUDA 核心，matmul 走 tensor 核心，硬件互不干扰。
- **FP8 支持**，配合 incoherent processing 抑制 outlier。

结果：H100 上 FP16 约 75% 峰值（~740 TFLOPs/s），FP8 约 1.2 PFLOPs/s，较 FA2 再快 1.5–2×。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 分块 + 在线 softmax | 精确注意力，`O(N)` HBM | kernel 复杂度；跨块 softmax 的数值稳定性需仔细处理 |
| Backward 重算 | 显存大幅下降，长上下文训练可行 | 反向 attention FLOPs 多一倍（但仍是带宽瓶颈，wall time 近乎免费） |
| 按架构手调 kernel | 接近峰值利用率 | 每代 GPU 都要大改写；FA3 初期仅支持 Hopper |
| FP8（FA3） | 再 2× 吞吐 | outlier 处理麻烦；精度验证不平凡 |
| 融合成单 kernel | 性能最佳 | shape 支持僵硬；head dim、mask 类型、dropout 变体各需一条代码路径 |

## 实验与结果

- FA1：BERT 精确注意力快 3×，GPT-2 端到端快 2.4×；单张 A100 上可训 16k 上下文。
- FA2：较 FA1 快 ~2×；A100 FP16 峰值 72%。
- FA3：H100 FP16 约 75% 峰值（≈740 TFLOPs/s），FP8 约 1.2 PFLOPs/s。
- 采用度：PyTorch 默认注意力（`scaled_dot_product_attention`）、vLLM、TensorRT-LLM，基本覆盖所有主流训练栈。

## 复现要点

- 完全开源。FA2 在 Ampere / Hopper 上是生产级；FA3 主攻 Hopper，仍在演进。
- 通过 PyTorch SDPA 集成很方便；自定义 mask / bias 需要下沉到底层 API。
- AMD MI300 上有对应工作在推进（Composable Kernel FlashAttention 等）。

## 个人评注

FlashAttention 是近年 ML 系统里最干净的一个例子——"**算法没问题，实现错了**"。注意力的平方显存被当成数学问题研究了很多年（因此有线性注意力、Performer、Linformer 等），而真正的瓶颈只是没人把 kernel 写对。这个教训可推广：**当一个 workload 跑得远低于 roofline 时，先问存储层次在干什么，再去改数学**。FA1→FA2→FA3 也是一个范例，说明 **kernel 软件需要每代 GPU 重写**；预期会有 FA4 for Blackwell，且这条线还会继续复利，不会很快收敛。做严肃 LLM infra 的人至少应完整读一遍 FA2 论文——那是目前关于 transformer GPU kernel 设计最清晰的单一文本。

## 参考

- [1] Dao et al. _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness._ arXiv:2205.14135, 2022.
- [2] Dao. _FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning._ arXiv:2307.08691, 2023.
- [3] Shah et al. _FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision._ arXiv:2407.08608, 2024.
- [4] Milakov & Gimelshein. _Online normalizer calculation for softmax._ arXiv:1805.02867, 2018.
