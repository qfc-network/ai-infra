# Blackwell / B200 架构 Primer

- **作者 / 机构**: NVIDIA
- **发表时间**: GTC 2024 年 3 月（架构发布）· GB200 NVL72 系统：2024 年底 / 2025 年供货
- **链接**: [NVIDIA Blackwell 架构白皮书](https://resources.nvidia.com/en-us-blackwell-architecture) · [GB200 NVL72 产品页](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

## TL;DR

Blackwell 是 NVIDIA 继 [Hopper](../hopper-h100/zh.md)（H100）之后的 GPU 架构，于 GTC 2024 发布。B200 GPU 和 GB200 超级芯片（同一封装上两颗 B200 GPU + 一颗 Grace CPU）是核心计算单元。相比 Hopper 的主要新增：**FP4 张量核心**（再次将稠密 FLOPS 翻倍）、**NVLink 5**（1.8 TB/s 对分带宽），以及**GB200 NVL72** 机架级系统——36 个 Grace-Blackwell 超级芯片通过完全无阻塞的 NVLink 互联，对软件呈现为一个 130 TB/s 的平坦内存系统。本 primer 覆盖计算原语（FP4、新 GEMM tile 形状）、内存层次（HBM3e、NVLink 5、NVSwitch 4），以及系统级 GB200 NVL72 架构，并将各项特性映射到关心它们的推理和训练工作负载。

## 背景：从 Hopper 到 Blackwell 的演进

理解 Blackwell 的变化，从 [Hopper](../hopper-h100/zh.md) 的优势和局限开始：

- **Hopper 计算峰值**（H100 SXM）：989 TFLOPS FP16，1,979 TFLOPS FP8（含稀疏）。
- **Hopper 显存**：HBM2e/HBM3，80 GB，3.35 TB/s。
- **Hopper 互联**：NVLink 4，每 GPU 总 NVLink 带宽 900 GB/s。

Blackwell 的设计决策直接源于 H100 在生产 LLM 工作负载中的瓶颈：

1. **计算/内存带宽差距**：推理解码是带宽瓶颈；若带宽不成比例增长，额外 FLOPS 无用。
2. **FP8 普及**：H100 FP8 在训练和推理中显示出重大收益；下一步是 FP4，再次将内存带宽减半。
3. **纵向扩展 vs 横向扩展**：900 GB/s NVLink 4 有时是 8 GPU 张量并行 all-reduce 的瓶颈；1.8 TB/s NVLink 5 解决了这个问题。
4. **机架级一致性**：解耦 prefill/decode 架构需要低延迟跨节点通信；NVL72 平坦互联直接针对此场景。

## 计算：FP4 张量核心

### 新精度

Blackwell 为张量核心引入 **FP4（E2M1）** 新计算精度：

- 每个值 4 位：1 位符号位，2 位指数位，1 位尾数位。
- 动态范围：±6.0（vs FP8 E4M3 的 ±448）。
- 用途：推理权重和 KV cache 量化（训练后量化可吸收精度损失）。

FLOPS 扩展：与 H100 FP8 相同的芯片面积，操作数为 FP4 时 FLOPS 再翻倍。

| 精度 | B200 FLOPS（稠密） | B200 FLOPS（2:4 稀疏） |
|---|---|---|
| FP32 | ~20 TFLOPS | — |
| BF16 / FP16 | ~4.5 PFLOPS | ~9 PFLOPS |
| FP8 | ~9 PFLOPS | ~18 PFLOPS |
| FP4 | ~18 PFLOPS | ~36 PFLOPS |

（数字为近似值；精确规格请参阅官方白皮书。）

### 扩展注意事项

模型权重使用 FP4 意味着相比 FP8 显存占用减半。70B 模型 FP8 需要约 70 GB；FP4 约 35 GB。单颗 B200（192 GB HBM3e）可在 FP4 下存放一个 70B 模型，并为 KV cache 留出充足空间。

FP4 量化的质量损失不可忽视：比 FP8 更敏感，需要逐组缩放（类似[混合精度训练](../mixed-precision/zh.md)中的 FP8 缩放方案）或类似 [AWQ](../weight-quantization/zh.md) 的激活感知方法。截至 2025 年，这仍是活跃研究领域。

### GEMM Tile 形状

Blackwell 保留了 Hopper 的 `wgmma` 异步 warpgroup GEMM 指令（见 [Hopper primer](../hopper-h100/zh.md)），并新增原生支持 FP4 操作数的**第五代张量核心**。Tile 形状和调度模型与 Hopper 一致——针对 H100 wgmma 的现有 Triton 和 CUTLASS kernel 需要更新以支持 FP4，但基本编程模型不变。

## 内存：HBM3e

B200 GPU 使用 **HBM3e**（第三代高带宽内存，扩展版）：

- **容量**：每颗 B200 192 GB（vs H100 SXM 80 GB）。
- **带宽**：每颗 B200 约 8 TB/s（vs H100 SXM 3.35 TB/s）。

容量提升（80 GB → 192 GB）是对 serving 最直接的影响。以 Llama 3 405B 为例：

- BF16：810 GB → 需要 11 颗 H100（张量并行）或 5 颗 B200。
- FP8：405 GB → 需要 6 颗 H100 或 3 颗 B200。
- FP4：约 202 GB → 可放入 **2 颗 B200**。

这直接影响所需的最小张量并行度，进而影响 all-reduce 通信开销。

## 互联：NVLink 5 与 NVSwitch 4

### NVLink 5

NVLink 5 将每链路带宽相比 NVLink 4 翻倍：

| 世代 | 每 GPU 总 NVLink 带宽 | 备注 |
|---|---|---|
| NVLink 4（H100） | 900 GB/s | 18 条链路 × 50 GB/s |
| NVLink 5（B200） | 1.8 TB/s | 18 条链路 × 100 GB/s |

对于 all-reduce 的张量并行：在 8 GPU TP 组中，每颗 GPU 每次 all-reduce 发送/接收 `(TP-1)/TP × hidden_size × batch_size × 2 字节`。NVLink 带宽翻倍使 all-reduce 延迟减半，直接提升大隐藏维度（如 Llama 3 405B 的 16384 维）下的 TP 扩展效率。

### NVSwitch 4

NVSwitch 4 提供连接 NVL72 系统内多颗 GPU 的交换结构。与前代一样，它以**全 NVLink 带宽支持全对全通信**——而非仅点对点。这对 MoE 专家并行 all-to-all 至关重要：72 颗 GPU 可同时以全速交换数据。

## GB200 NVL72：机架级系统

GB200 NVL72 是关键的系统级单元：

```
GB200 NVL72
├── 36 个 Grace-Blackwell 超级芯片
│   └── 每个：1 颗 Grace CPU（72 核 Arm Neoverse V2）+ 2 颗 B200 GPU
│                            通过 NVLink-C2C（900 GB/s，缓存一致性）连接
│
├── 9 颗 NVSwitch 4 芯片（提供交换结构）
└── 共 72 颗 B200 GPU，完全无阻塞 NVLink 互联
```

### 内存容量

总统一 GPU 内存：`72 × 192 GB = 13.8 TB`。通过 NVSwitch 结构，任意 GPU 可以 NVLink 带宽访问其他任意 GPU 的内存。从编程模型角度看，72 颗 GPU 呈现为一个 **130 TB/s** 的内存池（所有交换机的聚合 NVLink 带宽）。

参考：FP8 下 671B DeepSeek V3 需要约 671 GB。单个 NVL72 可同时存放约 20 个副本。

### NVLink-C2C：CPU-GPU 一致性

每个超级芯片上的 Grace CPU 和两颗 B200 GPU 通过 **NVLink-C2C** 连接——一种 900 GB/s 双向（每方向 450 GB/s）的一致性 CPU-GPU 互联。这支持：

- **CPU-GPU 统一内存**：Grace CPU 可以 NVLink 带宽直接访问 GPU HBM，GPU 也可以相同速度访问 CPU DRAM。对于将 KV cache 或优化器状态卸载到 CPU 内存，无需承受 PCIe 的 64 GB/s 带宽限制。
- **缓存一致性**：CPU 缓存和 GPU L2 保持一致性，支持细粒度 CPU-GPU 协作，无需显式数据拷贝。

这是与 x86+PCIe 连接系统的关键区别：在 H100 DGX 系统中，CPU-GPU 带宽受限于 PCIe 5.0（约 128 GB/s）。NVLink-C2C 的 900 GB/s 使 CPU 内存成为 GPU 内存的实用扩展。

## 与 LLM 工作负载的关联

| 工作负载 | B200 的关键优势 | 原因 |
|---|---|---|
| 解码（延迟敏感） | 8 TB/s HBM3e | 解码是带宽瓶颈；2.4× 带宽提升直接缩短每 token 解码时间 |
| 大模型 serving（单节点） | 192 GB HBM3e | 405B FP8 用 3 颗 B200 vs 6 颗 H100；更低 TP 度 → 更少 all-reduce 开销 |
| FP4 推理 | 18 PFLOPS FP4 稠密 | vs FP8 吞吐翻倍；更小 batch 也能达到 MFU 目标 |
| MoE 专家并行 all-to-all | NVSwitch 4 + NVLink 5 | 每 GPU 1.8 TB/s 全对全全速通信；减少 EP 通信瓶颈 |
| 解耦 prefill/decode | NVL72 平坦互联 | prefill 和 decode 节点共处同一 NVL72 互联；以 NVLink 速度跨节点传输 KV |
| 训练（大型 MoE） | 整机架作为一个系统 | 671B+ MoE 训练无需 NVL72 内的跨节点 RDMA |
| KV cache 扩展至 CPU | NVLink-C2C 900 GB/s | 多级 KV cache（GPU HBM → Grace DRAM）带宽高到可与重计算竞争 |

## 工程权衡

| 决策 | 收益 | 代价 |
|---|---|---|
| FP4 张量核心 | vs FP8 FLOPS 翻倍，显存节省翻倍 | 量化误差更大；需要逐组缩放或激活感知方法 |
| 192 GB HBM3e（vs 80 GB） | 更多模型可放单 GPU；更低 TP 度 | 每 GPU 成本更高；功耗更大 |
| NVLink 5（1.8 TB/s） | all-reduce 更快；TP 扩展更好 | NVSwitch 成本更高；H100 软件不能自动移植 |
| NVLink-C2C CPU-GPU 一致性 | CPU DRAM 作为快速内存层；缓存一致性 | 仅 Arm（Grace CPU）；绑定 NVIDIA SoC 设计 |
| GB200 NVL72（72 GPU 机架单元） | 机架级平坦内存互联；NVL72 内无需 RDMA | 必须以 72 GPU 为单位购买/部署；灵活性低于单独 GPU SKU |
| 向后兼容 Hopper SM90 ISA | 现有 CUDA/Triton/CUTLASS kernel 可移植 | FP4 和新 tile 形状需要新 kernel；不是自动加速 |

## 对比：B200 vs H100

| 特性 | H100 SXM | B200 |
|---|---|---|
| HBM | 80 GB，3.35 TB/s | 192 GB，约 8 TB/s |
| FP8 峰值（稠密） | 1,979 TFLOPS | ~9,000 TFLOPS |
| FP4 峰值（稠密） | — | ~18,000 TFLOPS |
| NVLink 带宽 | 900 GB/s | 1.8 TB/s |
| TDP | 700W | 1,000W |
| 系统单元 | DGX H100（8 GPU） | GB200 NVL72（72 GPU） |

## 深度解析

GB200 NVL72 最重要的特性是架构性的，而非数字性的：它消除了最多 72 颗 GPU 之间"节点内"与"节点间"的区别。这直接针对解耦 prefill/decode 模式——如果 prefill 和 decode 是独立进程（如 [DistServe](../distserve/zh.md) 和 [Mooncake](../../moonshot/mooncake/zh.md)），它们之间的 KV 传输现在可以以 NVLink 速度进行，而非 RDMA InfiniBand 速度。

每 GPU 192 GB HBM3e 改变了大模型所需的最小 TP 度。Llama 3 405B FP8 从 6 颗降至 3 颗 GPU TP，all-reduce 次数减半——可能比原始带宽提升带来更大的实际加速。

FP4 是最大的开放问题：对于 70B+ 前沿模型，训练后量化能否在 4 位下保持可接受的质量？早期结果喜忧参半。答案将决定 B200 FP4 是生产工具还是基准测试数字。

对于规划基础设施的团队：GB200 NVL72 的 72 GPU 粒度意味着这是数据中心级别的承诺。标准 DGX 风格配置的单颗 B200 SXM GPU 将随后推出；NVL72 适用于机架级一致性物有所值的大规模推理集群和训练任务。

## 参考文献

- [1] NVIDIA. _NVIDIA Blackwell Architecture Technical Brief._ GTC 2024.
- [2] NVIDIA. _GB200 NVL72 Product Overview._ 2024.
- [3] NVIDIA. _NVLink 5 and NVSwitch 4 Architecture._ 内部白皮书，2024.
- [4] Hopper 前代：[Hopper / H100 架构 Primer](../hopper-h100/zh.md)。
- [5] 互联背景：[GPU 互联 Primer](../gpu-interconnect/zh.md)。
- [6] FP4/FP8 量化背景：[混合精度训练](../mixed-precision/zh.md)、[KV Cache 量化](../kv-cache-quantization/zh.md)。
