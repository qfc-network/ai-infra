# Triton —— GPU kernel 的语言与编译器

- **作者 / 机构**：Philippe Tillet 等，Harvard → OpenAI
- **发表时间**：MAPL '19（原始论文）；现代 Triton 由 OpenAI 自 2021 年起发布
- **链接**：[论文 (MAPL '19)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) · [代码](https://github.com/triton-lang/triton) · [文档](https://triton-lang.org)

## 一句话总结

Triton 是 **嵌入在 Python 里的 DSL + 基于 MLIR 的编译器**，让你以**块（block）**为单位写 GPU kernel，而不是以线程为单位。你描述 kernel 为对显存块的操作——加载这块、做这次 matmul、存回那块——线程映射、共享内存分配、软件流水、指令选择都交给编译器。结果：一份有竞争力的 FP16 matmul 或融合注意力，从原本几百行 CUDA，缩到**几十行 Python**。Triton 支撑了 FlashAttention 的 PyTorch 参考实现、vLLM 的自定义 kernel、Liger Kernels、Unsloth、`torch.compile` 大量内部代码，以及 CUTLASS / CUDA C++ 阵营之外几乎所有现代开源 GPU kernel 工作。

## 背景与动机

Triton 之前，写一个有竞争力的 GPU kernel 只能在以下几条路里挑：

1. **CUDA C++**——手编线程布局、共享内存 tiling、warp shuffle。控制力最大，写起来慢，换硬件就重写。
2. **CUTLASS** 模板。比裸 CUDA 的 matmul 形状问题快，但学习曲线陡，不容易扩到非标准 op（融合注意力、自定义 epilogue）。
3. **厂商库**（cuBLAS、cuDNN）。黑盒；不可扩展；不可洞察。
4. **PyTorch eager** 配合 `torch.compile` / TVM / XLA 的自动融合。能拿到峰值的 ~70%；剩下 30% 还是要手调。

缺的是：**一个研究员能在一下午里写出与厂商库竞争的 kernel 的高产语言**。Triton 填了这个坑。Tillet MAPL '19 给出块级抽象；OpenAI 重写（2021+）让它在真实性能上变得可用、激进。

## 核心方法

### 块级编程模型

一个 Triton kernel 看起来是这样：

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))
```

关键概念动作：

- **每个 kernel 实例拥有输出的一个*块***，而不是单个线程。
- **存储操作（`tl.load`、`tl.store`）作用在 tile 上**，不是标量。编译器负责向量化与合并。
- **计算（`tl.dot`）也在 tile 上**。编译器选 Tensor Core MMA 形状、布局共享内存、分配 warp。
- **控制流**（`for k` 循环）驱动 K 轴分块；编译器为它做软件流水以掩盖延迟。

程序员永远不用说"线程 17 拥有 tile 的第 4 行"。这个映射是编译器的活。

### 编译器都做了什么

Triton 基于 **MLIR**。编译流水线（简化）：

1. Triton IR —— Python AST → 高层 tiled IR（含 `triton.dot`、`triton.load` 等）。
2. 优化 pass：layout 选择、软件流水、masked load 合并、冗余 load 消除。
3. lower 到 GPU dialect（NVIDIA：PTX → ptxas → SASS；AMD：ROCDL → LLVM AMDGPU）。
4. 落到厂商汇编器。

非平凡的 pass：

- **Layout 选择** —— 给 `tl.dot` 选 MMA 指令变体（Ampere 的 m16n8k16、m16n8k8 等；Hopper 的 WGMMA），以及对应的寄存器/共享内存 tile 布局。
- **软件流水** —— 让下一轮的 `tl.load` 与本轮的 `tl.dot` 重叠。等价于手写 cp.async 流水线，但是生成的。
- **共享内存分配** —— 编译器决定哪些数据走 SMEM、计算地址并避免 bank 冲突。

可以这样理解 Triton：**CUDA kernel 写作里无聊、易错的部分自动化；有意思的部分暴露给你**。

### 自动调优

Triton 自带 autotuner：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        # ...
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...): ...
```

声明搜索空间；新 (M, N, K) 形状首次出现时 Triton 跑各配置 benchmark 选最优、缓存。冷启动延迟换持续吞吐——和 DeepGEMM 是同一种模式，但内置在语言里。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 块级抽象 | 几十行就能写 kernel；迭代快 | 一些 warp 级 / 指令级技巧不沉到 PTX inline 就表达不了 |
| Python 嵌入 | 研究员能用；与 PyTorch 集成 | 每个 kernel 签名有编译时间（缓存可缓解） |
| 编译器自动调度 | 不必手工管线程/warp/SMEM | 调度选错时调试很难 |
| MLIR 后端 | 多 GPU 目标（NVIDIA、AMD、Intel） | 编译器复杂度；后端 bug 会暴露给 kernel 作者 |
| 内置 autotune | 对形状变化覆盖容易 | 首次调用 wall-clock 成本；开发期需要代表性形状 |
| 仅 JIT（无 AOT） | 与 DeepGEMM 同款 shape-specialization 故事 | 生产部署要预热缓存或持久化磁盘 cache |

## 实验与结果

- 原 MAPL '19：手写 Triton matmul 在 V100 上达到 cuBLAS 的 ~90%；FFT、卷积在部分形状上与 cuDNN 接近。
- 现代 Triton（2024）：FlashAttention-2 的**参考实现就是 Triton**；Hopper 多形状 70%+ 的 cuBLAS 等效峰值。FA3 部分 Triton、部分手写 CUTLASS。
- vLLM、SGLang、FlashInfer 都出 Triton kernel 实现 paged attention、采样、RMSNorm、RoPE 等。
- Hopper 上足够大 M/N/K 时 Triton matmul 与 cuBLAS 性能可比。差距逐年收窄。

## 复现要点

- 完全开源。`pip install triton` 装 OpenAI 版。
- 持续开发：`triton-lang/triton` 频繁发布；Hopper / Blackwell 支持有 NVIDIA 大量贡献；AMD CDNA 支持 2025 年已与 rocBLAS 竞争。
- 调试工具：`TRITON_INTERPRET=1` 用纯 Python 跑 kernel 做正确性检查；`triton.testing.do_bench` 做干净的性能 benchmark。
- 性能因形状与 Triton 版本变化大。生产里要锁版本。

## 个人评注

Triton 是现代 GPU kernel 工作里**最大单一生产力放大器**。"一下午用 Triton 复现 FlashAttention" 这种 2023 年成为标准的范式，在只有 CUDA C++ 时不可想象。块级抽象选对了——它正好契合高性能 kernel 实际的结构（tiled、软件流水、基于 MMA），同时不让程序员写 bookkeeping。

诚实的局限：在最调过的形状上，Triton 仍落后手写 CUTLASS / CUDA **5–15%**。对每一个百分点都重要的生产 matmul，你还是会沉到 CUTLASS / CuTe。除此之外——融合注意力、采样、归一化、自定义 op 研究——Triton 是默认。**DeepGEMM 用 CUTLASS 写 matmul，但 DeepGEMM 没覆盖的，开源社区都用 Triton 写。**

更深的启示是**抽象的胜出原则**：Triton 选了正确的层（块，不是线程）、嵌在正确的语言里（Python）、保留了正确的逃生口（PTX inline）。许多 GPU DSL 失败要么因为飞太高（Halide 的多面体模型）要么因为飞太低（无表达力的 kernel autogen）。Triton 落在生产力的中段。作为对照，**PyTorch `torch.compile` 内部把许多 op lower 到 Triton**——意味着编译器/DSL 的界限模糊了，Triton 实际成了 PyTorch 的 GPU IR。

2026 年做 infra 的人：如果你的 kernel 能用 Triton 表达，就用——天数级出活，性能通常逼近专家手调。只在 profiling 证明 Triton 留下了实质性能时才下沉到 CUTLASS / CUDA。

## 参考

- [1] Tillet et al. _Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations._ MAPL '19.
- [2] Triton 仓库：https://github.com/triton-lang/triton
- [3] Triton 文档：https://triton-lang.org
- [4] FlashAttention Triton 参考实现：在 FlashAttention 仓库内
- [5] OpenAI 介绍博客（2021）：https://openai.com/blog/triton/
