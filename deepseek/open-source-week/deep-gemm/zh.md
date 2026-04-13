# DeepGEMM —— 源码走读

- **仓库**：[deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- **首次发布**：2025-02（Open Source Week 第三天）
- **配套笔记**：[`deepseek/v3-tech-report/`](../../v3-tech-report/) —— V3 的 FP8 细粒度 scaling 配方
- **配套笔记**：[`deepseek/open-source-week/deep-ep/`](../deep-ep/) —— 推理中 DeepEP 的输出直接喂给 DeepGEMM

## 这个仓库是什么

一个 **JIT 编译的 GEMM 库**，针对 Hopper（SM90）和 Blackwell（SM100），主攻 FP8（2025 年加了 BF16），专为 DeepSeek 实际使用的矩阵形状优化。刻意**不做 CUTLASS 替代**——README 明确目标是 "clean and efficient"，只实现有限的核心 kernel。

性能声明：**H800 上最高 1550 TFLOPS**（见 PR #74、#78、#81、#86）。在覆盖形状上持平或胜过专家调优库。

三类 GEMM 覆盖 V3 训练 + 推理整栈：

| API | 用途 |
|---|---|
| `fp8_gemm_{nt,nn,tn,tt}` | Dense 层（attention 投影、MLA 上/下投影、LayerNorm 邻近 matmul） |
| `m_grouped_fp8_gemm_*_contiguous` | MoE 训练 / prefill —— M 轴分组，专家拼接 |
| `m_grouped_fp8_gemm_nt_masked` | MoE 推理 decode —— CUDA graph 友好、带 mask，直接吃 DeepEP LL 输出 |
| `k_grouped_fp8_gemm_tn_contiguous` | MoE 权重反向 |
| `fp8_mqa_logits`、`fp8_paged_mqa_logits` | V3.2 Lightning Indexer (DSA) 评分 |

## "clean" 为什么重要

CUTLASS 是 NVIDIA 的官方 matmul 库，但也是 ~100 万行重模板 C++，覆盖数千种形状。在生产里用它要么 (a) 接受默认模板（吃不到峰值），要么 (b) 成为 CUTLASS 专家（学习曲线极长）。DeepGEMM 的 README 说得直白：

> DeepGEMM 借鉴了 CUTLASS 与 CuTe 的概念，但避免重依赖其模板或代数。库被刻意设计成简单的，只有少量核心 kernel。因此也是学习 NVIDIA GPU kernel 优化的干净教材。

设计取舍是 **"kernel 更少、手工调优、从上到下读完"**，而非"覆盖所有形状"。DeepSeek 做得了这个选择，因为它只需要自己模型的形状。

## 目录结构

```
DeepGEMM/
├── deep_gemm/
│   ├── __init__.py
│   ├── include/deep_gemm/
│   │   ├── common/                   # 设备侧原语
│   │   └── impls/                    # 设备侧 kernel 实现
│   ├── legacy/                       # 2025-07 重构前版本
│   ├── testing/
│   └── utils/
├── csrc/
│   ├── python_api.cpp                # PyTorch 绑定面
│   ├── apis/                         # 主机侧各 kernel 家族入口
│   ├── jit/                          # JIT cache、NVCC/NVRTC 驱动
│   ├── jit_kernels/
│   │   ├── heuristics/               # 形状 → 配置选择
│   │   └── impls/                    # 主机侧 launcher
│   │       ├── sm90_fp8_gemm_1d1d.hpp
│   │       ├── sm90_fp8_gemm_1d2d.hpp        # V3 细粒度配方
│   │       ├── sm100_fp8_gemm_1d1d.hpp
│   │       ├── sm90_bf16_gemm.hpp
│   │       ├── sm100_bf16_gemm.hpp
│   │       ├── smxx_fp8_mqa_logits.hpp       # V3.2 indexer（非分页）
│   │       ├── smxx_fp8_paged_mqa_logits.hpp # V3.2 indexer（分页）
│   │       ├── sm{90,100}_bmk_bnk_mn.hpp     # batched 变体
│   │       ├── epilogue.hpp
│   │       └── runtime_utils.hpp
│   ├── indexing/
│   └── utils/
├── third-party/                      # CUTLASS submodule（CuTe 原语）
├── scripts/、tests/
└── setup.py
```

注意：`sm90_fp8_gemm_1d1d.hpp` 与 `sm90_fp8_gemm_1d2d.hpp` 是独立文件，不是同一个模板。任取一个从上读到下都是可行的。

## 1d1d vs 1d2d（V3 细粒度 scaling 的代码形态）

FP8 训练必须 scale 以控制 outlier。V3 的配方是**激活按 1×128 tile、权重按 128×128 block** 分别 scale。DeepGEMM 把这一点落成两种 kernel：

- **`fp8_gemm_1d1d`** —— 两端都按 1-D tile scale。更简单，部分形状/场景用。
- **`fp8_gemm_1d2d`** —— LHS（激活）按 1-D tile scale，RHS（权重）按 2-D block scale。这是 V3 论文的配方。

scale 的数据布局按架构不同：

- **SM90**：scale 用 FP32，要求 TMA 对齐、转置（`get_mn_major_tma_aligned_tensor`）。
- **SM100**：scale 打包成 **UE8M0**（4 个一 `int32`），用 Blackwell 原生的 micro-scaling 格式。

UE8M0 是 Blackwell 的 micro-scaling 格式——8 位指数、0 位尾数，本质上是 2 的幂次乘子——由硬件在 FP8 matmul 中直接作为 scale 使用。每 int32 打 4 个 scale 摊销读取成本。

## JIT 编译——不寻常的那一部分

与 cuBLAS / CUTLASS / FlashAttention 不同，**DeepGEMM 不预编译 kernel**，也没有 kernel 的 `.so`。流程是：

1. 用户调 `fp8_gemm_nt(A, B, ...)`。
2. Python 侧看形状，走启发式选配置（`csrc/jit_kernels/heuristics/`）。
3. C++ JIT 模块（`csrc/jit/`）拼出 kernel 源码，用 NVCC（或 `DG_JIT_USE_NVRTC=1` 时用 NVRTC）编译。
4. 编译产物落盘缓存（默认 `$HOME/.deep_gemm`，可用 `DG_JIT_CACHE_DIR` 覆盖）。
5. 后续同形状调用命中缓存。

为什么 JIT？从代码和 news 能看到三个原因：

1. **形状专属 kernel**：每个 (M, N, K) 的 block tiling、pipeline 深度、TMA 布局各自一份二进制。AOT 得编几千种变体。
2. **安装简单**：`pip install` 不走长时间编译，首次调用摊销编译成本。
3. **按集群调编译器**：`DG_JIT_NVCC_COMPILER` 可指定 NVCC 路径。在 NVCC 12.9 之前，编译器差异直接影响 SASS 质量。

NVRTC 于 2025-05（PR #94）加入，"编译快达 10×"，但"部分场景可能损失性能"。用户可在编译时间和运行时性能间做权衡。

## FFMA 交错的故事

DeepGEMM 最被引用的一项优化是 **FFMA 指令交错**——编译后 SASS 级重排乘加指令以改善延迟掩盖。2025-04 公告"H800 上 1550 TFLOPS"就来自这里。

2025-07 的 news 把这事闭环了：

> NVCC 12.9 已自动完成 FFMA 交错，所有后处理优化不再支持。

两点值得记：

- DeepGEMM 做过 SASS 级后处理。这在开源 GPU 代码里极少见——你在字面意义上重写汇编器的输出。
- 几个月后 NVCC 补齐，hack 就过时了。**很多看起来永恒的 kernel 魔法都是编译器暂时落后的产物。** 读前沿 GPU 代码时值得记住这一点。

## 三种 MoE 布局

V3 的 256 路由专家让 MoE kernel 必须支持分组。DeepGEMM 暴露三种：

### 连续布局（Contiguous）—— 训练 / prefill

各专家 **N、K 相同**（权重形状一致），但 `M_i`（每个专家的 token 数）不同。DeepGEMM 把所有专家的 token 沿 M 拼起来，用一个 kernel 按 per-group offset 识别边界：

```
[expert 0 的 tokens | expert 1 的 tokens | ... | expert 255 的 tokens]
```

每组 `M_i` 要对齐到 GEMM M block size（`get_mk_alignment_for_contiguous_layout()`）。一次 launch 处理整块拼接张量；N 和 K 在单次 launch 中固定，符合 MoE 所有专家同形假设。

### 掩码布局（Masked）—— 推理 decode

Decode 为低延迟做 CUDA graph。CPU 在 launch 时不知道 `M_i`（由运行时路由决定）。DeepGEMM 的答案：**按最坏 `M_max` 跑 kernel，用 mask 标记无效位置**。mask 张量告诉 kernel 每组哪些 M 位置有效。

这与 **DeepEP 的 LL 输出**无缝对接：DeepEP 给出带 mask 的路由 token，DeepGEMM 消费同一个 mask 在同一 CUDA graph 里算出专家输出。两个库是协同设计的。

### K-grouped —— MoE 权重反向

MoE 的权重梯度要按专家 `e` 算 `dW_e = X_e^T @ dY_e`。此时 M、N 固定（权重形状），K 变（每专家 token 数）。`k_grouped_fp8_gemm_tn_contiguous` 是 M-contiguous kernel 在这种情况下的对偶。

## V3.2 Lightning Indexer kernel

2025-09（PR #200）加入，不在主 GEMM 家族。V3.2 论文引入 **DeepSeek Sparse Attention (DSA)**：轻量 indexer 给 token-pair 打分，取 top-k，只对被选中的 KV 做 attention。打分的 kernel 形状特殊：

```
# 对应 README 里的伪代码
kv_j = kv[0][j, :] * kv[1][j]          # 用 scale 反量化
out_ij = q[i, :, :] @ kv_j             # per-head logit
out_ij = out_ij.relu() * weights[i, :]  # 加权 ReLU
out_ij = out_ij.sum()                  # 标量 logit
```

`fp8_mqa_logits`（prefill，非分页）、`fp8_paged_mqa_logits`（decode，分页 KV）实现之。ReLU 激活**融进了 matmul epilogue**——便宜，但对 indexer 那种低"算力/字节比"的形状来说很关键。

## 启发式与调优

`csrc/jit_kernels/heuristics/` 按 (M, N, K, arch) 选 block 大小、pipeline 深度、TMA multicast 形状。roadmap 有：

> Better `get_best_configs` modeling

说明还在改进。目前 `DG_PRINT_CONFIGS=1` 可打印每个形状选到的配置——排查性能回退很好用。

三个工具函数与栈其他部分联动：

- **`set_num_sms(n)` / `get_num_sms()`** —— 限定 GEMM 使用的 SM（与 DeepEP 的 `Buffer.set_num_sms` 对称，两个库共享 SM 预算）。
- **`set_tc_util(r)`** —— 给启发式一个近似 Tensor Core 利用率目标；可用峰值吞吐换更低功耗/更好并存。
- **`transform_sf_into_required_layout`** —— 把用户持有的 scale 转成 kernel 要求的 TMA 对齐、转置布局。README 明示"可能慢"——提醒**真正的融合应在上游 kernel 里做**。

## 哪些能搬走

- **1d1d / 1d2d FP8 scaling 配方**是 V3 风格细粒度 FP8 最清晰的开源实现。任何愿采用该布局的训练栈都可直接用。
- **掩码分组 GEMM + DeepEP LL 组合**是 **CUDA graph 兼容的 MoE decode** 模板。任何规模化 MoE 推理最终都会重新实现这对；DeepGEMM 的版本够短、好移植。
- **带磁盘缓存的 JIT GEMM** 对"形状空间大但每个调用点形状稳定"的场景都是合适模式——推理 server、动态形状训练。
- **Blackwell 上的 UE8M0 scaling** —— DeepGEMM 是目前开源库中少数有生产路径的，SM100 采纳者值得参考。

## 哪些不能

- 只出 **FP8 和 BF16**，没 FP16、没 INT8。刻意选择——V3 用不到——但意味着不是老栈的 drop-in 替代。
- 形状覆盖是**"DeepSeek 需要的那些"**，不是通用。奇怪形状可能能编译但性能打折。
- JIT 带**每形状首次调用的编译延迟**。低延迟 serving 且形状动态时要规划缓存预热。
- 部分 SASS 级 trick 依赖 NVCC 12.9 之前的编译器。新工具链下不需要后处理器就有同等 SASS。
- Ampere 在 roadmap 但未交付；SM80 用户暂不在射程内。

## 个人评注

DeepGEMM 是"我们只要自己的形状，所以库也只做这些形状"这种立场在开源 GPU 代码里最清晰的一次表态。跟 V3 技术报告一起读，你可以把报告里**关于 FP8 scaling 的每条描述都映射到这里的某个 kernel 变体**——1d2d 是那份配方，`transform_sf_into_required_layout` 工具是 bookkeeping，UE8M0 路径说明 Blackwell 该在哪嵌入。最值得研究的是与 DeepEP 的协同（掩码分组 GEMM ↔ DeepEP LL 输出）——**MoE 推理不是 GEMM 问题，也不是 all-to-all 问题，而是一条流水线**，DeepGEMM + DeepEP 是这条流水线的两端。FFMA 交错的故事则给出另一层启示：**有些 kernel 优化是真正的算法胜利，另一些是半衰期 6 个月的"编译器滞后套利"**。区分两者得读 commit 记录，而不是 README。

## 参考

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM
- V3 技术报告（FP8 scaling 配方）：arXiv:2412.19437
- V3.2 论文（DSA、indexer）：https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- CUTLASS：https://github.com/NVIDIA/cutlass
- NVIDIA PTX UE8M0 文档：https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats
