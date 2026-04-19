# NVIDIA TransformerEngine

- **机构**：NVIDIA
- **首次发布**：2022-11（随 H100 正式上市同步开源）
- **链接**：[GitHub](https://github.com/NVIDIA/TransformerEngine) · [文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)

## 一句话总结

TransformerEngine（TE）是 NVIDIA 开源的 Python/C++ 库，将 Hopper 架构的 FP8 张量核指令（`wgmma`）封装成可直接替换 PyTorch 标准模块的高层接口，位于硬件（`cuBLAS`/`cuDNN` FP8 GEMM）与训练框架（Megatron-Core、NeMo、FSDP）之间。其核心抽象是融合模块——`te.Linear`、`te.LayerNormLinear`、`te.MultiheadAttention`、`te.TransformerLayer`——每个模块在单个 kernel 内完成 FP8 量化、GEMM 与反量化，配合 `DelayedScaling` 配方维护逐张量的幅度历史，从而在不强制每步同步更新 scale 的前提下保障量化的数值安全。H100 上端到端训练吞吐量相比 BF16 基线可提升 1.3–1.6×。

## 背景与动机

H100 引入了新指令族 `wgmma`（warpgroup matrix multiply-accumulate），支持 FP8 输入（E4M3 和 E5M2）并以 FP32 累加。FP8 GEMM 的峰值吞吐为 3958 TFLOPS；BF16 GEMM 为 1979 TFLOPS。理论上 2× 的差距是 FP8 训练要争取的预算。

障碍在于数值范围。FP8 E4M3 的可表示范围约为 [−448, 448]，比 BF16 窄约 11 个数量级。直接将大型权重或激活张量 cast 到 FP8 会立即溢出或下溢。需要两个条件：

1. 一个逐张量的 scale，在 GEMM 之前将每个张量的实际幅度映射到 FP8 可表示区间，并在 GEMM 之后施加反 scale。
2. 一个以低开销维护这些 scale 的机制——每步都精确计算逐张量最大值需要额外的规约 kernel，以及在前向传播开始前的设备到主机同步。

TransformerEngine 同时解决了这两个问题：用融合的 cast+GEMM+dequant kernel 包裹每个 FP8 操作，用 `DelayedScaling` 将 scale 更新开销分摊到多步之间。

## 核心架构

### 模块层次

TransformerEngine 暴露四个主要模块：

- `te.Linear` — 替代 `nn.Linear`；输入与权重量化为 FP8，GEMM 在 FP8 下执行，输出反量化为 BF16/FP32。
- `te.LayerNormLinear` — 将 LayerNorm 融合进 cast 路径；省去一次规范化激活的 HBM 往返。
- `te.LayerNormMLP` — 融合 LayerNorm + 两个 Linear（上投影 + 下投影）与 GELU 门控，合并为单次 kernel 启动。
- `te.MultiheadAttention` — 封装 QKV 投影、注意力计算（可选调用 FlashAttention-3 后端）以及输出投影；所有 Linear 层均使用 FP8。
- `te.TransformerLayer` — 完整的 pre-norm Transformer 块（自注意力 + MLP）；Megatron-Core 直接将其用作层实现。

所有模块共享同一套 scale 基础设施：每个模块为输入、权重和梯度输出分别维护 `amax_history`、`scale` 与 `scale_inv` 缓冲。

### FP8 格式：E4M3 与 E5M2

TE 使用两种具有不同数值特性的 FP8 数据类型：

| 格式   | 指数位 | 尾数位 | 最大值 | 动态范围 |
|--------|--------|--------|--------|----------|
| E4M3   | 4      | 3      | 448    | ~8 个数量级 |
| E5M2   | 5      | 2      | 57344  | ~24 个数量级 |

E4M3 尾数精度更高，适用于前向传播张量（权重、激活），因为这些张量上的微小数值误差会累积。E5M2 以牺牲尾数分辨率换取更大动态范围，适合梯度张量——其幅度可能在训练过程中跨越多个数量级。TE 的 `DelayedScaling` 配方默认在前向传播的输入和权重上使用 E4M3，在梯度输出上使用 E5M2。

### FP8 量化 + GEMM + 反量化模式

对于权重矩阵 W 和输入 X 的 Linear 层，计算流程如下：

```
# Scale 选取（每 N 步执行一次，而非每步）
scale_x   = fp8_max(E4M3) / amax_history_x.max()   = 448   / amax_x
scale_w   = fp8_max(E4M3) / amax_history_w.max()   = 448   / amax_w

# 前向传播（单个融合 kernel）
X_fp8     = cast_to_fp8(X * scale_x,  dtype=E4M3)
W_fp8     = cast_to_fp8(W * scale_w,  dtype=E4M3)
Y_fp32    = wgmma(X_fp8, W_fp8)                    # 以 FP32 累加
Y_out     = Y_fp32 * (1 / scale_x) * (1 / scale_w) # 反量化，转回 BF16

# 反向传播（梯度采用独立的 E5M2 配方）
scale_dY  = fp8_max(E5M2) / amax_history_dY.max()  = 57344 / amax_dY
dY_fp8    = cast_to_fp8(dY * scale_dY, dtype=E5M2)
dX        = wgmma(dY_fp8, W_fp8^T) * (1/scale_dY) * (1/scale_w)
dW        = wgmma(X_fp8^T, dY_fp8) * (1/scale_x)  * (1/scale_dY)
```

关键在于：量化、GEMM 和反量化不是三个独立 kernel——它们合并为一个融合 kernel，以预计算的 scale 作为参数传入。这样就避免了在量化与 GEMM 之间将 FP8 张量作为中间结果写入 HBM。

## DelayedScaling 配方

`DelayedScaling` 是 TE 中默认且唯一生产可用的 scale 配方。它维护一个逐张量绝对最大值的滑动窗口（amax 历史），每隔 `amax_history_len` 步更新一次 scale，而不是每步都更新。

```python
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling(
    margin=0,                    # fp8_max 以下的 log2 余量；出现溢出时增大
    interval=1,                  # scale 更新间隔步数（1 = 每步更新）
    fp8_format=Format.HYBRID,    # 前向 E4M3，反向 E5M2
    amax_history_len=16,         # amax 滑动窗口长度
    amax_compute_algo="max",     # "max" 或 "most_recent"
)

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    loss = model(batch)
```

`fp8_autocast` 是一个上下文管理器，将 FP8 配方安装到其范围内的所有 TE 模块。在该上下文之外，模块以 BF16 运行，因此同一模型无需修改代码即可在全精度下评估。

`margin` 参数提供二的幂次余量：scale 计算中实际使用的 fp8_max 为 `fp8_max / 2^margin`。设置 `margin=1` 将可用范围减半，降低溢出风险，但代价是损失部分可表示精度。生产配方通常保持 `margin=0` 并监控 loss spike。

### Scale 更新机制

在每次前向传播中，TE 从实际张量值计算逐张量 amax（设备上的规约 kernel），并存入 amax 历史缓冲。Scale 使用滑动窗口最大值更新：

```
scale = fp8_max / (amax_history[最近 N 步].max() * 2^margin)
```

由于 scale 基于历史数据而非当前步的张量，它可能落后 `amax_history_len` 步之多。若某层激活突然出现峰值——例如训练初期或学习率预热跳跃之后——陈旧的 scale 可能过大而导致溢出。`amax_history_len` 参数在响应速度与稳定性之间做取舍：更短的窗口反应更快，但噪声更大。

### 逐张量 scale 与逐通道 scale 的对比

TE 采用逐张量 scale：每个张量一个 scale，不论形状。另一种方式是逐通道（或逐行）scale：为权重矩阵的每个输出通道或激活的每个 token 计算一个独立 scale。逐通道 scale 在精度上更优，因为它能单独处理离群通道而不影响整个张量，但它需要更多的元数据，并且要求在反量化时支持按通道操作——这是 H100 `wgmma` 对 FP8 原生不支持的功能。TE 因此接受精度上的代价，换取硬件兼容性与实现简洁性。

## 与训练框架的集成

### Megatron-Core

Megatron-Core 的 `TransformerLayer` 通过 `transformer_layer_spec` 参数接受 `te.TransformerLayer` 作为直接替换。启用 TE 后，Transformer 块中的每个 Linear 均使用 FP8。激活检查点（Activation Checkpointing）对用户透明：TE 模块实现了自定义的 `torch.autograd.Function` 子类，前向传播时只保存 FP8 量化后的激活（而非 BF16 原始值），将检查点显存减少高达 50%。

Megatron 的流水线并行（PP）在各阶段之间传递 BF16 张量；TE 不尝试将跨阶段激活保持为 FP8，因为接收方需要发送方的 scale 元数据。

### FSDP

与 TE 模块配合使用时，FSDP（FullyShardedDataParallel）需要设置 `use_orig_params=True`，这与 `torch.compile` 的要求相同。不设置该选项时，FSDP 在 all-gather 时将参数重建为平坦张量，TE 注册在每个 `nn.Module` 上的逐参数 FP8 scale/amax 元数据缓冲会与聚合后的参数张量失去对应关系。

设置 `use_orig_params=True` 后，FSDP 在对权重参数分片的同时也对 FP8 scale 和 amax 缓冲进行分片，在 all-gather 后保持对齐。scale 缓冲很小（每个张量一个 float），通信开销可忽略不计。

### 与 FlashAttention-3 的关系

TE 的 `MultiheadAttention` 模块以 FP8 处理 QKV 投影和输出投影（均为 Linear 层）。对于注意力得分计算本身——`softmax(QK^T / sqrt(d)) V`——TE 在 Hopper 上可用时将其分派给 FlashAttention-3 后端。FA3 原生使用 `wgmma` 对内部 QK 和 PV 矩阵乘法进行 FP8 计算，并处理注意力的 IO 效率问题（分块 SRAM 计算，不物化完整注意力矩阵）。两者相辅相成：FA3 负责注意力得分 kernel，TE 负责其外围的 Linear 投影层。联合使用可最大限度捕获 FP8 收益。

## 性能分析

H100 SXM5 密集 GEMM 峰值 FLOPs：

| 数据类型 | 峰值 TFLOPS | 备注                        |
|----------|-------------|------------------------------|
| FP32     | 67          | CUDA core                    |
| TF32     | 989         | Tensor Core                  |
| BF16     | 1979        | Tensor Core（wgmma）         |
| FP8      | 3958        | Tensor Core（wgmma，Hopper） |

BF16 与 FP8 之间 2× 的比值是硬件上限，而非实际训练加速比。端到端加速更低，原因如下：

1. 注意力 FLOPs 不会翻倍（对于长序列，FA3 FP8 比 FA3 FP16 快约 1.5–2×；注意力仅占总 FLOPs 的一部分）。
2. Embedding 层、LayerNorm 及其他非 GEMM 操作受带宽限制，不受 FP8 影响。
3. 通信（AllReduce、AllGather、ReduceScatter）不受益。
4. Scale 更新和 amax 规约 kernel 引入额外开销。

生产环境中观测到的端到端训练加速：对 GPT 类模型（大隐藏维度）为 **BF16 基线的 1.3–1.6×**。注意力占比较高的模型（FFN 比例小）在低端；FFN 规模大的模型（如高隐藏维度 MoE）在高端。

## 工程权衡

| 量化方案 | 相对 BF16 的精度 | 实现复杂度 | 硬件要求 | 训练稳定性 |
|---|---|---|---|---|
| DelayedScaling（FP8 E4M3+E5M2） | LLM 上 ppl 差距约 0.1–0.3，通常在噪声范围内 | 中——需要 amax 历史缓冲、跨 FSDP 分片的 scale 同步 | H100 / GH200（需要 Hopper wgmma） | 良好（调好 margin 后）；loss 急速下降期有溢出风险 |
| 静态逐张量 FP8 | 强依赖预校准 scale 的准确性 | 低——无运行时状态 | H100 | 脆弱——任何分布偏移均导致溢出/下溢 |
| 逐通道（逐行）FP8 量化 | 优于逐张量；接近 INT8 逐通道 | 高——反量化需逐通道，wgmma 原生不支持 | 需自定义 kernel 或未来硬件 | 对离群值密集模型更稳定 |
| BF16（基线） | 参考基准 | 最低——标准混合精度 | A100 及更新 | 极稳定；loss landscape 行为已充分验证 |
| INT8（训练后量化，仅推理） | 推理可接受，训练会退化 | 低（推理）/ 高（训练） | Ampere 及更新 | 不用于训练 |

## 故障排查

**训练启动时 loss spike**：amax 历史为冷启动（全零），首次 scale 更新基于错误的参考值。缓解方法：在 FP8 训练前用一次校准前向传播预填充 amax 历史，或在前几百步设置较大的初始 `margin`。

**反向传播溢出**：E5M2 范围（57344）在训练早期仍可能被梯度幅度超越。可通过 `NVTE_FP8_DFP_AMAX_REDUCE_DEBUG=1` 排查。增大 `margin` 或延长 `amax_history_len`。

**跨数据并行 rank 的 scale 同步**：每个 rank 从其数据分片计算自己的 amax。若不同 rank 看到不同的输入分布，scale 会出现偏差。TE 的 `fp8_autocast` 提供 `scale_factor_reduction` 选项，在更新 scale 之前跨 DP rank 对 amax 进行 AllReduce。

**FSDP + compile 冲突**：将 `torch.compile` 与 FSDP 和 TE 配合使用需要设置 `use_orig_params=True` 和 `dynamic=False`。FSDP 延迟参数聚合与 TE 自定义 autograd 函数之间的交互是最常见的静默正确性错误来源。

## 工程师视角评注

TransformerEngine 是将 Hopper 硅片算力转化为训练吞吐量最直接的路径。`wgmma` 指令配合 FP8 使 GEMM 的硬件吞吐预算翻倍；TE 是让这笔预算在无需重写每个训练循环的情况下从 Python 层可访问的关键。`DelayedScaling` 配方并不神奇——它是一种务实的近似，在激活幅度分布平滑时表现良好，在分布剧烈波动时会出问题。知道何时放宽 margin、何时延长历史窗口，以及何时 FP8 路径根本不适用（例如训练非常早期，或非常敏感的微调任务），才是真正的工程技能所在。随着 Blackwell 引入 FP4 和更宽的 wgmma 变体，同样的模式将再次上演：新的低精度格式、新的硬件指令、新的融合 kernel、新的 scale 配方。TE 的架构足够通用，可以吸收这些演进。

## 相关条目

- [`../hopper-h100/`](../hopper-h100/) — `wgmma` 指令、FP8 张量核硬件、H100 存储层次
- [`../../foundational/mixed-precision/`](../../foundational/mixed-precision/) — FP16/BF16/FP8 训练演进；loss scaling 历史
- [`../../foundational/flash-attention/`](../../foundational/flash-attention/) — FA3 使用 `wgmma` 处理注意力；TE 处理其外围 Linear 层
- [`../../foundational/zero-fsdp/`](../../foundational/zero-fsdp/) — FSDP 集成；`use_orig_params=True` 要求
- [`../megatron-core/`](../megatron-core/) — Megatron-Core 通过 `transformer_layer_spec` 将 TE 用作层实现

## 参考资料

- [1] NVIDIA TransformerEngine GitHub. https://github.com/NVIDIA/TransformerEngine
- [2] NVIDIA. _H100 Tensor Core GPU Architecture._ White paper, 2022.
- [3] Micikevicius et al. _FP8 Formats for Deep Learning._ arXiv:2209.05433, 2022.
- [4] Shah et al. _FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision._ arXiv:2407.08608, 2024.
- [5] Shoeybi et al. _Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism._ arXiv:1909.08053, 2019.
