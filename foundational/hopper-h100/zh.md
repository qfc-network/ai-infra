# Hopper / H100 架构 Primer

_参考手册，非论文解析。H100（Hopper 微架构，SM90）是本仓库 FlashAttention-3、DeepGEMM、DualPipe、DeepEP 及 DeepSeek-V3 FP8 训练方案共同假设的硬件基础。本文命名关键原语并解释它们对上述系统的具体意义。_

- **机构**：NVIDIA
- **发布时间**：2022-03 公布；2023 年批量供货
- **链接**：[Hopper 架构白皮书](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) · [H100 数据手册](https://www.nvidia.com/en-us/data-center/h100/) · [CUDA PTX ISA (SM90)](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## 一句话总结

H100（SM90 / Hopper）引入了四项原语，从根本上改变了高性能 CUDA kernel 的编写方式：**wgmma**（warpgroup 矩阵乘累加，异步）、**TMA**（张量内存加速器，带张量描述符的异步 DMA）、**线程块集群（Thread Block Clusters）**（同一 GPC 内多 SM 共享内存）以及**原生 FP8 Tensor Core**（E4M3 与 E5M2）。它们共同实现了数据搬运、计算与同步并发执行的编程模型——这是相较于 Ampere 同步 mma.sync 的根本转变。FlashAttention-3 的 warp 专化、DeepGEMM 的 JIT tile、DeepEP 的细粒度分发均依赖这些原语。理解它们可以让本仓库约 6 个条目无需查阅外部文档就能读懂。

## 背景：相较于 Ampere (A100) 的变化

| 特性 | A100 (SM80) | H100 (SM90) | 影响 |
|---|---|---|---|
| Tensor Core 指令 | `mma.sync`（单 warp，阻塞） | `wgmma.mma_async`（warpgroup，异步） | 在 kernel 内实现计算与数据搬运重叠 |
| 异步数据搬运 | `cp.async`（无描述符） | TMA（张量描述符，无寄存器压力） | 消除索引计算开销与寄存器溢出 |
| SM 协调范围 | 仅线程块内 | 线程块集群（跨 SM 共享 SMEM） | 允许同一 GPC 内细粒度负载共享 |
| FP8 支持 | 无 | E4M3 + E5M2 Tensor Core | 支持高吞吐细粒度 per-tile FP8 GEMM |
| HBM 带宽 | 2.0 TB/s（HBM2e） | 3.35 TB/s（HBM3） | 更高计算/带宽比；更多寄存器复用以分摊代价 |
| NVLink 带宽 | 600 GB/s | 900 GB/s（NVLink 4） | MoE 专家并行每步 all-to-all 成本降低 |
| SM 数量 | 108 | 132 | 原始算力提升 |

设计哲学的转变：A100 是"给程序员强大的同步构建块"；H100 是"给程序员强大的*异步*构建块与显式同步点"。性能上限提高——但编程模型更难，将 A100 kernel 朴素移植到 H100 通常只能实现峰值的 50–70%。

## SM 层次结构

理解 Hopper 需要理解芯片内部的三级并行：

```
H100 GPU
└── GPC（GPU 处理集群）× 8
    └── TPC（纹理处理集群）× 每 GPC 约 8 个
        └── SM（流式多处理器）× 132 个
            ├── Warpgroup（4 个 warp = 128 线程）
            │   └── Warp（32 线程）
            ├── 共享内存：228 KB（可配置与 L1 的分割）
            ├── 寄存器文件：256 KB
            └── 第 4 代 Tensor Core（FP8 / FP16 / BF16 / TF32 / INT8）
```

**线程块集群（Thread Block Clusters）** 是线程块之上的新抽象。一个集群将最多 8 个线程块调度到**同一 GPC 内**的 SM 上。集群中的线程块可通过分布式共享内存访问彼此的 SMEM（`cluster.map_shared_memory`）。这是协作加载的硬件机制：一个线程块可在自己的 SMEM 中计算的同时，另一个线程块将数据预取到其 SMEM——跨越 SM 边界。

## wgmma — Warpgroup 矩阵乘累加

### 是什么

`wgmma.mma_async` 是一条新的 PTX 指令，针对整个 **warpgroup**（4 个 warp，128 线程）而非单个 warp。它向 Tensor Core 流水线异步发出矩阵乘累加指令后立即返回——warpgroup 线程可以立即发出下一条 wgmma 或执行其他工作，而硬件并行执行乘法。

Tile 尺寸远大于 Ampere 的 mma：
- FP16/BF16：单 warp 64×16×16 → warpgroup 最小 64×64×16；最大可达 64×256×16
- FP8：warpgroup 最小 64×64×32；等效算术强度约为 FP16 的两倍

异步性由 **wgmma fence** 控制：`wgmma.fence.sync.aligned` 在发出 wgmma 前标记边界；`wgmma.commit_group.sync.aligned` 提交一组；`wgmma.wait_group.sync.aligned N` 等待至多 N 组在飞。这使 kernel 可以流水线化多个 tile。

### 为何重要

在 A100 上，`mma.sync` 是**阻塞**的：warp 停滞直到 Tensor Core 结果写入寄存器。程序员可以在 mma.sync 调用之间插入内存加载，但计算与加载仍争用 warp 执行槽。

在 H100 上，wgmma 是**非阻塞**的：Tensor Core 流水线独立于 warpgroup 指令流执行。Kernel 可以发出 wgmma，然后为下一个 tile 发出 TMA 加载，然后等待两者完成——计算与数据搬运真正并行。

这正是 FlashAttention-3"warp 专化"所利用的：warpgroup 中部分 warp 被指定为"生产者"（发出 TMA 加载），其余为"消费者"（发出 wgmma），通过共享内存 barrier 通信。硬件在同一 SM 内并发运行它们。

## TMA — 张量内存加速器

### 是什么

TMA 是一个片上 DMA 引擎，在全局内存（HBM）与共享内存之间搬运 tile。程序员创建一个**张量描述符**（描述张量形状、步长、元素类型和复制 tile 大小的结构体），并用单条 PTX 指令发出复制：

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global [smem_ptr], [desc_ptr], [coords], [barrier];
```

一个线程发出此指令，TMA 引擎异步执行复制，完成后触发 `mbarrier`。发出指令的线程——以及 warpgroup 其余线程——可以立即继续执行其他工作。

### 为何重要

在 A100 上，将一个 tile 从 HBM 加载到共享内存需要：
- 计算每个元素的全局地址（消耗寄存器）
- 为每个元素或向量发出 `cp.async`
- 管理来自地址计算的寄存器压力

对于大 tile（如 128×128 BF16），地址计算消耗大量寄存器文件，导致寄存器溢出。寄存器溢出降低占用率，而占用率是隐藏内存延迟的主要杠杆。

TMA 消除了所有这些。描述符只需设置一次（在 CPU 或 kernel 启动时）；PTX 指令只需提供坐标。无需逐元素地址算术，加载路径无寄存器压力。这就是 DeepGEMM 能在高占用率下使用超大 tile（128×128 或 128×256）的原因，也是 FlashAttention-3 能在不因寄存器压力限制 tile 大小的情况下流水线化 Q/K/V 加载与 wgmma 计算的原因。

### mbarrier — 同步原语

TMA 和 wgmma 均在完成时触发 `mbarrier` 对象（内存侧 barrier）。mbarrier 有一个 phase（0 或 1）和一个到达计数；当到达计数达到阈值时翻转 phase。生产者（TMA 加载、集群对端）到达 barrier；消费者（读取已加载数据的 wgmma 计算）等待 barrier。这种通过 mbarrier 管理的解耦生产者-消费者模型是所有异步 H100 kernel 的基础。

## FP8 Tensor Core

H100 原生支持两种 FP8 格式：
- **E4M3**：4 位指数，3 位尾数。范围：±448。精度更高；用于前向传播激活值和权重（值域更受限）。
- **E5M2**：5 位指数，2 位尾数。范围：±57,344。范围更大；用于反向传播梯度（训练时出现较大值）。

wgmma 指令支持 FP8 操作数：
```ptx
wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
```
操作数 A（来自寄存器）和 B（来自 SMEM）均为 E4M3；累加器为 F32。Tensor Core 硬件内部处理格式转换。

**为何需要 per-tile scaling**：FP8 有限的动态范围（总共 8 位）无法用单个全局缩放因子表示权重矩阵的完整范围。DeepSeek-V3 使用 per-tile（激活值 1×128，权重 128×128）缩放因子。每个 tile 在 wgmma 前被缩放到 FP8 范围内；per-tile 缩放因子在累加时乘回。TMA 的 tile 级粒度与此直接对应：每次 TMA 复制搬运一个 tile，该 tile 的缩放因子随之获取。DeepGEMM 通过利用 tile 大小、TMA 与 wgmma 之间的这种对齐，以不到 300 行代码实现了这一功能。

## 共享内存配置

每个 SM 有 **228 KB** 的 L1 数据缓存与共享内存组合（A100 为 192 KB）。程序员配置分割比例；典型高性能 kernel 将 164–228 KB 用作共享内存，L1 极小。

228 KB 足以对两个 128×128 BF16 tile 进行双缓冲（2 × 128 × 128 × 2 字节 = 64 KB），同时为累加器和元数据留有空间。双缓冲是 Hopper 的标准模式：wgmma 计算缓冲区 A 的同时，TMA 填充缓冲区 B；然后交换。更大的 SMEM 预算使得在 H100 tile 尺寸下无寄存器溢出的双缓冲成为可能。

## L2 缓存与 HBM

- **L2 缓存**：50 MB（A100 为 40 MB）。足以在大多数配置下容纳单个注意力层的权重 tile，实现真正的权重复用。
- **HBM3 带宽**：3.35 TB/s（A100 为 2.0 TB/s）。注意力和 decode 阶段推理仍受内存带宽限制；这 1.67 倍的带宽提升直接转化为 serving 中 KV cache 读取的吞吐提升。
- **HBM 容量**：80 GB（SXM5）——与 A100 80 GB 相同，但带宽更高，通过 PCIe 5 / NVLink 4 的 CPU-GPU 传输更快。

## NVLink 4 与 Fabric

详细内容见 [GPU 互联 primer](./gpu-interconnect/)。H100 关键数字：
- **NVLink 4**：每 GPU 双向 900 GB/s（A100 为 600 GB/s）
- **NVSwitch 第 3 代**：支持 NVL72（72 张 H100 在一个机架内，以完整 NVLink 速度 all-to-all）
- **IBGDA**：InfiniBand GPU Direct Async——GPU 无需 CPU 介入即可发出 RDMA；DeepEP 的低延迟路径（`notify_dispatch`）使用 IBGDA 进行细粒度专家分发时序控制

相较于 A100，NVLink 带宽提升 50%，直接降低了 MoE 专家并行每步 all-to-all 的成本。对于 V3 的 256 专家配置，每个 token 的分发会跨越多个节点；每字节每次 NVLink 跳转成本更低，意味着 DualPipe 中的流水线停顿时间更短。

## 这些原语如何组合：Hopper Kernel 范式

所有高性能 H100 kernel 遵循相同模式：

```
以线程块集群启动 kernel
│
├── 生产者 warp：
│   ├── 创建 TMA 描述符（一次）
│   └── 循环：
│       ├── cp.async.bulk.tensor [TMA 加载 tile A → smem_A]
│       ├── cp.async.bulk.tensor [TMA 加载 tile B → smem_B]
│       └── mbarrier.arrive（信号：tile 已加载）
│
└── 消费者 warp：
    └── 循环：
        ├── mbarrier.wait（等待 tile）
        ├── wgmma.mma_async [smem_A × smem_B → 寄存器累加器]
        ├── wgmma.commit_group
        ├── wgmma.wait_group 1   （保持 1 组在飞）
        └── [epilogue：缩放、通过 TMA 或寄存器存储结果]
```

核心不变式：**生产者与消费者并发运行**。mbarrier 解耦它们。TMA 引擎和 Tensor Core 流水线与 warp 指令流并行执行。正确使用 `wgmma.fence`、`wgmma.commit_group`、`wgmma.wait_group` 和 `mbarrier.wait` 决定了计算是否真正重叠，还是意外串行化。

FlashAttention-3 将其生产者 warp 命名为"加载 warp"，消费者 warp 命名为"数学 warp"。DeepGEMM 使用相同模式，但通过 JIT（Python → PTX）生成 kernel，tile 大小参数在运行时选择。DualPipe 的计算-通信重叠在更粗的粒度上使用相同的异步原则：一个流水线阶段计算时，下一阶段的 NVLink/IB 传输并行进行。

## 工程权衡

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| wgmma（异步，warpgroup 级） | 计算/内存重叠；更大的有效 tile；更高峰值 TFLOPS 利用率 | 编程复杂度高；fence 放置错误会静默串行化；调试异步计算困难 |
| TMA（基于描述符的 DMA） | 加载路径零寄存器压力；在高占用率下支持大 tile | 描述符设置开销（每 kernel 摊销）；仅限连续张量布局（步长限制） |
| FP8 Tensor Core | 吞吐约为 FP16/BF16 的 2 倍；支持细粒度量化的 per-tile scaling | 8 位范围需要精细 scaling；累加器必须是 F32（无 FP8 累加）；需要自定义 kernel——cuBLAS FP8 灵活性不如手调 kernel |
| 228 KB 共享内存 | 大型双缓冲 tile；降低每 tile HBM 流量 | 若一个线程块占用所有 SM 资源，限制占用率（每 SM 线程块数更少） |
| 线程块集群 | 跨 SM 细粒度共享内存；GPC 内更好的负载均衡 | 限制线程块放置（必须在同一 GPC 内共定位）；每集群最多 8 个线程块；问题规模较小时硬件支持有限 |

## 对本仓库已有条目的意义

| 条目 | 使用的 H100 原语 | 方式 |
|---|---|---|
| [FlashAttention-3](../flash-attention/) | wgmma、TMA、mbarrier | Warp 专化：生产者 warp TMA 加载 Q/K/V tile；消费者 warp wgmma 计算注意力；mbarrier 解耦两者 |
| [DeepGEMM](../../deepseek/open-source-week/deep-gemm/) | wgmma（内联 PTX）、TMA、FP8 Tensor Core | JIT 生成使用 wgmma.mma_async.e4m3 的 PTX；per-tile FP8 scaling 与 TMA tile 粒度对应 |
| [DualPipe](../../deepseek/open-source-week/dualpipe/) | 异步 CUDA 流、NVLink 4 | 双向流水线将计算隐藏在 NVLink 传输之后；NVLink 4 的 900 GB/s 减少每次传输的停顿 |
| [DeepEP](../../deepseek/open-source-week/deep-ep/) | IBGDA、NVLink 4、异步流 | 低延迟路径使用 IBGDA 无需 CPU 发出 RDMA；普通路径通过异步流将 IB 传输与计算重叠 |
| [混合精度训练](../mixed-precision/) | FP8 Tensor Core、E4M3/E5M2 | H100 的按格式累加 + F32 累加器是使 V3 FP8 训练可行的硬件基础 |
| [Triton](../triton/) | wgmma（通过 `tl.dot`）、TMA（通过带缓存修饰符的 `tl.load`） | Triton 在其 MLIR 后端中针对 SM90；wgmma 通过块级 `tl.dot` 抽象暴露 |

## 可复现性说明

- H100 SXM5（数据中心版）和 H100 PCIe 使用相同的 Hopper 芯片，但内存和连接配置不同。SXM5 拥有完整 NVLink 4 + 900 GB/s；PCIe 版无 NVLink。
- H800（面向中国的出口管制版本）：相同 SM90 芯片，相同计算原语（wgmma、TMA、FP8），但 NVLink 带宽上限约 400 GB/s。DeepSeek 的 V3 和 OSW kernel 在 H800 上开发——片内计算原语完全相同；节点间通信是 H800 的短板。
- **CUDA 12.0+** 是 wgmma 和 TMA 的必要条件。大多数生产框架（PyTorch 2.1+、vLLM、SGLang）在 H100 上需要 CUDA 12 才能获得完整性能。
- 学习资源：NVIDIA [Hopper 架构白皮书](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)、[CUDA PTX ISA 指南（SM90 章节）](https://docs.nvidia.com/cuda/parallel-thread-execution/)，以及 FlashAttention-3 论文附录（在真实 kernel 中对 wgmma/TMA 最易读的解释）。

## 参考文献

- [1] NVIDIA. _NVIDIA H100 Tensor Core GPU Architecture._ Whitepaper, 2022.
- [2] NVIDIA. _Parallel Thread Execution ISA Version 8.x (SM90)._ developer.nvidia.com, 2024.
- [3] Shah et al. _FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision._ arXiv:2407.08608, 2024.
- [4] DeepSeek-AI. _DeepGEMM._ github.com/deepseek-ai/DeepGEMM, 2025.
- [5] DeepSeek-AI. _DeepEP._ github.com/deepseek-ai/DeepEP, 2025.
- [6] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024.（FP8 训练方案）
- [7] Luo et al. _CUTLASS._ github.com/NVIDIA/cutlass, 2024.（SM90 kernel 底层的 CuTe 布局代数）
