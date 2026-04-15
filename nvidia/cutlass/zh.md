# CUTLASS —— CUDA 线性代数子程序模板库

- **作者 / 机构**：Andrew Kerr、Duane Merrill、Julien Demouth、John Tran 等（NVIDIA）
- **发表时间**：2017-11（初始版本）；持续更新至 CUTLASS 3.x（2023–2024）
- **链接**：[代码](https://github.com/NVIDIA/cutlass) · [博客](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) · [文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cutlass_3x_design.md)

## 一句话总结

CUTLASS 是 NVIDIA 开源的 C++ 模板库，专为 CUDA GPU 上高性能 GEMM（通用矩阵乘法）及相关操作而设计。它将 cuBLAS 内部使用的层次化分解——threadblock tile、warp tile、指令级 fragment——以可组合的 C++ 模板形式暴露出来，让你能自定义 epilogue、融合操作，并针对 Hopper 专有硬件指令进行目标编译，而无需重新发明底层机制。Triton 提供生产力（Python DSL、编译器管理内存）；CUTLASS 提供控制力（C++ 模板、手动调度的内存流水线、WGMMA/TMA 指令）。FlashAttention-2/3、DeepGEMM 以及大多数生产推理 kernel，要么直接基于 CUTLASS 构建，要么与其设计模式高度一致——理解 CUTLASS 是必备的前置知识。

## 背景与动机

cuBLAS 在标准矩阵规格上能达到峰值 FLOP 利用率，但它是黑盒——无法自定义 epilogue，无法融合自定义激活，也无法控制内存访问模式。要写一个从零开始能与 cuBLAS 竞争的原始 CUDA GEMM，需要同时深入掌握以下内容：

- 共享内存 tiling，以掩盖 HBM 延迟（tile 必须装进 SMEM；tile 大小决定 occupancy）
- 寄存器分配，避免 spilling（大型累加器 tile 会占用大量寄存器文件）
- Warp 级矩阵指令（Volta 的 WMMA、Ampere 的 MMA、Hopper 的 WGMMA）
- 双缓冲（软件流水），让计算与内存读取重叠
- 指令调度，保证 tensor core 在内存操作间隙持续满负荷

常见失败模式：kernel 只搞定其中某几项，最终落在 cuBLAS 的 40–60% 而不是 95%+。CUTLASS 的解法是：把这些复杂性分解成可复用、可组合的 C++ 模板，按 cuBLAS 内部相同的层次结构组织——区别在于你能读、能扩展、能修改每一层。

CUTLASS 2.x（Ampere 时代，2020–2022）确立了四级模板层次，成为生产 GEMM 定制的主流范式。CUTLASS 3.x（Hopper，2023）围绕 **CuTe** 重新架构——这是一个嵌入 C++ 的张量代数 DSL——让索引运算通过构造正确，而不是依赖约定。3.x 重设计还原生支持 Hopper 异步硬件单元（WGMMA、TMA），这对达到 H100 峰值吞吐至关重要。

## 核心方法

### GEMM 的层次化分解

基本问题：计算 **C = A × B + C**，其中 A ∈ ℝ^{M×K}，B ∈ ℝ^{K×N}，C ∈ ℝ^{M×N}。

CUTLASS 跨四个硬件层级进行分解：

1. **Grid 层** —— 将 (M, N) 分配给各 threadblock。每个 threadblock 拥有大小为 (BlockM × BlockN) 的输出 tile，并在整个 K 维度上迭代。
2. **Threadblock 层** —— 从 A 加载 (BlockM × BlockK) tile、从 B 加载 (BlockK × BlockN) tile 到共享内存。以 BlockK 为步长遍历 K，累加到寄存器中的 (BlockM × BlockN) fragment。
3. **Warp 层** —— 每个 warp 拥有 threadblock C tile 的一个 (WarpM × WarpN) fragment，发出 tensor core 指令（Volta：WMMA；Ampere：MMA；Hopper：WGMMA）。
4. **线程 / 指令层** —— 单次 tensor core 操作处理小 tile，例如 Ampere FP16 的 `m16n8k16` MMA 指令处理 16×8×16 个元素。

吞吐的关键路径是以足够快的速度向 tensor core 供料使其始终饱和。在每个层级，tile 大小控制并行度、寄存器压力和 SMEM 用量之间的权衡。CUTLASS 将这些作为编译期常量模板化——改变 (BlockM, BlockN, BlockK, WarpM, WarpN) 一步改变 occupancy、寄存器文件使用和 SMEM 占用。

### 软件流水（双缓冲）

隐藏 HBM 延迟的经典技巧是双缓冲：warpgroup 在处理第 K 块的同时，异步预取第 K+1 块到第二个共享内存缓冲区。伪代码如下：

```
分配 smem_buf[2][BlockM][BlockK]  // 两个乒乓缓冲区
异步拷贝(A_tile[k=0] → smem_buf[0])
等待拷贝完成()

for k in range(0, K, BlockK):
    异步拷贝(A_tile[k+1] → smem_buf[(stage+1) % 2])
    执行MMA计算(smem_buf[stage % 2])   // tensor core 读 SMEM
    等待拷贝完成()
    stage += 1
```

这需要为 A tile（B 同理）使用 2× SMEM，但能将 HBM 延迟完全掩盖在计算后面。CUTLASS 2.x 将其实现为 `PipelinedGemmKernel`；CUTLASS 3.x 将其泛化为 `PipelineAsync`，stage 深度可配置（视 SMEM 预算和重叠需求，可设 1–5+ 级）。

### CuTe：构造正确的索引代数

CUTLASS 3.x 引入 **CuTe** 作为张量布局描述与操作的基础层。CuTe 中的 `Layout` 是一个从多维整数坐标到线性内存偏移的函数：

```cpp
// 行主序 M×K 矩阵：
auto layout = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));

// 为 threadblock tiling 分区布局：
auto blk_layout = zipped_divide(layout, make_shape(BlockM, BlockK));
// blk_layout(tile_m, tile_k, inner_m, inner_k) → 正确偏移
```

CuTe layout 通过 `coalesce`、`zipped_divide`、`tiled_divide`、`logical_divide` 组合——这些代数操作为层次结构的每一级分区张量，无需手动索引运算。收益在于：描述 A 全局内存布局的同一套 layout 操作，同时描述其 tile 如何落入共享内存、warp 的 fragment 如何读取它——由类型组合而非约定来强制保证。

这消除了一整类 bug：全局内存、共享内存、寄存器布局之间的 stride 不匹配。在 CUTLASS 2.x 中，此类不匹配需要仔细的文档维护，是定制 kernel 时的常见正确性问题来源。

### Hopper 专属：WGMMA + TMA

Hopper（H100）引入了两个硬件单元，需要 CUTLASS 3.x 的模式才能充分利用：

**WGMMA（Warpgroup 矩阵乘累加）** —— 跨一个 warpgroup（4 个 warp，128 线程）的异步 tensor core 指令。与 Ampere 从寄存器读取的同步 MMA 不同，WGMMA 直接从共享内存读取，使寄存器文件可以专用于累加器 tile。在相同寄存器预算下，每个 warpgroup 可实现的累加器规模大约翻倍。

**TMA（Tensor Memory Accelerator）** —— 一个硬件 DMA 单元，将 HBM 中的矩形 tile 拷贝到共享内存，地址生成完全从 warp 卸载。与仍消耗 warp 发射带宽的 `cp.async` 指令不同，TMA 由单个描述符触发，独立运行。Warp 发出 TMA 指令后执行其他工作，然后在 barrier 处等待。CUTLASS 3.x 将 TMA 封装为 `SM90_TMA_LOAD` 和 `SM90_TMA_STORE`，并与 WGMMA 流水，以在 H100 上实现接近理论峰值的 FP8/FP16 吞吐。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| C++ 模板 vs cuBLAS 黑盒 | 完全可定制：epilogue 融合、自定义布局、FP8 逐块缩放、任意输出类型 | 编译时间长（一套 CUTLASS kernel 编几分钟）；模板报错出了名的晦涩——类型不匹配可产生 100+ 行报错 |
| CUTLASS vs Triton | 细粒度内存控制；Hopper WGMMA/TMA 访问；峰值吞吐；手动调度指令序列的能力 | Triton 在研究原型阶段生产力高得多；CUTLASS 需要 C++ 专业知识和对 PTX 级硬件细节的了解 |
| CuTe 布局代数（3.x） | 类型系统强制保证可组合、正确的索引运算；消除层次各级手动 stride 运算 | 学习曲线陡峭；CuTe 类型（`Tensor<Engine, Layout>`、layout functor）对多数 CUDA 程序员陌生；文档即源码 |
| 双缓冲（每缓冲区 2× SMEM） | 完全掩盖 HBM 延迟；大多数矩阵规格上接近 roofline 吞吐 | 可用于 tile 大小的 SMEM 减半；限制 BlockM × BlockN 的可选范围，可能降低 occupancy 或被迫使用更小的 tile |
| WGMMA 异步（仅 Hopper） | warpgroup 粒度的计算与 SMEM 加载重叠；解锁 H100 FP8 峰值 | 不可移植到 Ampere（需要独立代码路径）；warpgroup 同步编程模型比逐 warp MMA 复杂 |
| 纯头文件 C++ 库 | 无需单独编译库；易于内嵌或修改 | 用户自行承担编译成本；无预编译二进制可链接；头文件中大量 CUDA 设备代码会给部分构建系统带来麻烦 |

## 实验与结果

**标准 GEMM 吞吐**：CUTLASS 在调优形状（M=N=K=4096，FP16，A100）上达到 cuBLAS 吞吐的 95–98%。剩余 2–5% 的差距反映了 cuBLAS 使用未公开 PTX 内置指令和 NVIDIA 内部离线 profiling 的优势。

**FlashAttention-2**：在 Ampere attention kernel 中使用 CUTLASS MMA 抽象。FA2 在典型序列长度上 A100 达到 70%+ MFU，其中 CUTLASS 的 kernel 贡献了绝大部分性能。自定义 epilogue（跨 K 维的 softmax 归一化）正是 cuBLAS 无法提供的部分。

**FlashAttention-3**：Hopper 版本显式使用 CUTLASS 3.x WGMMA 和 TMA。在 H100 FP16 上达到 75%+ MFU，FP8 注意力接近峰值——相比之下，朴素注意力实现约 35% MFU。

**DeepGEMM**（DeepSeek）：以 CUTLASS 3.x 为基础实现逐块缩放的 FP8 GEMM。通过将 CUTLASS 的 WGMMA 流水与 JIT 编译的 tile 配置结合，达到单张 H100 超过 90% 的 FP8 峰值（约 1.9 PFLOP/s）。

**Epilogue 融合**：CUTLASS 的 epilogue 框架允许 bias 加法、激活函数（ReLU、GELU）和输出缩放在与 GEMM 相同的 kernel pass 中执行，消除每层单独启动 1–2 个内存带宽瓶颈 kernel 的开销。在 transformer 规模下（数十亿参数，每训练步数千个 layer），这是端到端时间中不可忽视的比例。

## 复现要点

CUTLASS 完全开源于 [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)。它是**纯头文件 C++ 库**——无需单独编译库本体，只需 include 头文件并用 `nvcc` 编译你的 kernel。

**环境要求**：
- CUTLASS 2.x：CUDA 11.4+，sm_70（Volta）及以上
- CUTLASS 3.x / CuTe：CUDA 12.0+；WGMMA/TMA 需 sm_90（Hopper）；其余 3.x 特性支持 sm_80（Ampere）

**Profiler 工具**：`tools/profiler` 可对任意 kernel 配置（问题规模、数据类型、tile 形状）做基准测试，无需写 host 代码。这是为特定 GPU 找最优 (BlockM, BlockN, BlockK) 的最快途径。

**Python 绑定**：`cutlass-py`（`pip install nvidia-cutlass`）提供 Python 接口，以编程方式生成 CUTLASS kernel，适合快速探索而无需直接写 C++ 模板。

**构建**：CMake 加 `-DCUTLASS_NVCC_ARCHS=90a`（或 80、89 等）以定向特定 GPU 世代。调试构建加 `-DCUTLASS_ENABLE_TESTS=ON`。

**测试**：`test/unit/` 目录有详尽的正确性测试；`test/perf/` 有各形状、各 GPU 的参考性能数字——这是"我的修改有没有破坏什么"的权威依据。

## 个人评注

CUTLASS 占据独特的生态位：它是 **cuBLAS 使用但从未暴露的技术的有文档、开源实现**。每一个非平凡的 LLM 训练与推理 GPU kernel，最终要么直接使用 CUTLASS，要么重新发明 CUTLASS 的模式，要么构建在 FlashAttention、DeepGEMM 这些已经做了前两件事之一的库之上。

正确的心智模型是一条**抽象层次谱系**：

`torch.compile` → Triton → CUTLASS 3.x + CuTe → 裸 PTX / SASS

每向下走一级，性能潜力提升约 5–15%，工程成本乘以 3–10 倍。CUTLASS 所在的层次让你以单条缓存行和指令流水的粒度控制内存访问，但不需要写汇编。

Triton（另见专项文章）是更高生产力的替代——对于需要快速原型化的研究 kernel，Triton 在"写到正确"的时间上赢了。但对于 Hopper 专属硬件至关重要的场景——从 SMEM 读取的 WGMMA、无地址生成开销的 TMA DMA、带逐块 E4M3/E5M2 缩放的 FP8——CUTLASS 3.x 是参考实现。两者的关系是**互补而非竞争**：Triton 用于研究 kernel，CUTLASS 用于生产 kernel——当最后 10% 的吞吐至关重要，且操作形状稳定到值得投入工程资源时。

3.x 的 CuTe 重架构在软件工程层面是真正的进步，不只是性能故事。旧 CUTLASS 2.x 对 stride 如何跨层级组合有大量隐式约定；CuTe 将这些约定显化到类型系统中。未来的硬件目标——Blackwell、后 Hopper 世代——将因为抽象层次更干净而更容易正确地添加支持。

2026 年做生产 AI infra 的人：如果你在写自定义注意力、量化 GEMM、MoE 路由，或任何 cuBLAS/cuDNN 不覆盖的 kernel，你都会与 CUTLASS 打交道。投入时间学习 CuTe 的布局模型——这是理解最慢、一旦内化收益最大的部分。

## 参考

- [1] Kerr et al. 2017. "CUTLASS: Fast Linear Algebra in CUDA C++." NVIDIA Developer Blog. https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
- [2] NVIDIA 2023. "CUTLASS 3.0: NVIDIA CUTLASS Design Guide." https://github.com/NVIDIA/cutlass/blob/main/media/docs/cutlass_3x_design.md
- [3] NVIDIA 2023. "CuTe: A Domain-Specific Language for CUDA Tensor Operations." 见 CUTLASS 仓库 `include/cute/`。
- [4] Dao et al. 2023. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691.
- [5] Shah et al. 2024. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608.
- [6] DeepSeek 2025. "DeepGEMM: Clean and Efficient FP8 GEMM Kernels with Fine-grained Scaling." https://github.com/deepseek-ai/DeepGEMM
