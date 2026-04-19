# Megatron-Core（mcore）：现代大规模训练背后的模块化库

- **机构**：NVIDIA
- **从 Megatron-LM 中提取时间**：约 2023 年（首次作为 `megatron/core/` 出现在 Megatron-LM 代码仓库中）
- **链接**：[GitHub — NVIDIA/Megatron-LM (core/)](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) · [NeMo](https://github.com/NVIDIA/NeMo) · [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)

## 一句话总结

Megatron-LM（2019–2022 年的论文系列）是一套整体式的 GPT 训练脚本。Megatron-Core——即 mcore——是从中重构提取出来的库：可独立导入的 Python 模块集合，实现了张量并行、流水线并行、上下文并行、FP8 路由以及流水线调度。NeMo、NeMo-Aligner、Nemotron、xAI 的 Grok 训练栈以及众多外部研究团队都将 mcore 作为库依赖来消费，而非 fork 原始训练脚本。若 2024–2025 年的论文或系统报告中提到"Megatron 风格 TP"或"4D 并行"，其实际运行的实现几乎必然是 mcore，而非早期的原始脚本。

## 背景与动机

原始 Megatron-LM 代码仓库将 GPT 模型的端到端训练打包在一起。其并行逻辑正确且经过充分测试，但深度嵌入在模型定义、数据加载和优化器循环的单一代码库中。若要在新架构（如混合专家模型或 RLHF 训练循环）中仅复用张量并行线性层，就必须 fork 整个仓库，这会不断偏离上游并制造维护负担。

Megatron-Core 通过将并行原语提取到具有清晰 API 的结构化 Python 包（`megatron/core/`）中来解决这一问题。关键设计决策：进程组管理、模型构建和调度执行是各自独立的关注点，分别由独立模块负责。外部框架只需导入所需部分。

这对生态的实际影响是：NeMo 不再自己实现张量并行线性层，而是封装 mcore 的实现。NeMo-Aligner 的 GRPO 和 DPO 训练循环构建于 mcore 的 DDP 和调度模块之上。当 NVIDIA 发布 Nemotron 模型权重时，训练配置以 mcore 的 `TransformerConfig` 对象来表达。`../../foundational/megatron-lm/` 中的论文条目描述的是*算法*；本条目描述的是在生产规模下*执行*这些算法的*库*。

## 核心抽象

### ParallelState（`megatron/core/parallel_state.py`）

`ParallelState` 是判断哪个 GPU 在哪个并行维度中占据哪个 rank 的唯一权威来源。在任务启动时，`initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, ...)` 通过计算 `world_size` 个 rank 的组合布局，为 TP、PP、DP 和 CP 分别创建 NCCL 进程组。此后的每次通信调用——all-reduce、reduce-scatter、all-gather——都通过 `get_tensor_model_parallel_group()` 或 `get_data_parallel_group()` 等函数查询进程组，而非手动传递组句柄。

这种设计的意义在于：由于进程组是全局单例，调用栈中的任何模块——transformer 块深处的线性层、自定义 MoE 路由器、优化器——都能在不通过每个函数签名传递组句柄的情况下获取正确的进程组。它本质上是一个进程组注册表。其缺点是引入了全局可变状态，这使测试复杂化，也令同进程内多任务的场景处理起来颇为别扭。

### TransformerConfig

`TransformerConfig` 是整个 mcore transformer 栈的核心配置数据类。关键字段示例如下：

```python
TransformerConfig(
    num_layers=96,
    hidden_size=12288,
    num_attention_heads=96,
    num_query_groups=8,               # GQA：8 个 KV head
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=16,
    context_parallel_size=2,          # 用于长上下文的 CP
    sequence_parallel=True,
    use_flash_attention=True,
    fp8=True,                         # 路由至 TransformerEngine
    fp8_recipe=te.recipe.DelayedScaling(...),
    normalization="RMSNorm",
    activation_func=F.silu,           # SwiGLU
)
```

这个单一对象会被传入 `GPTModel`、`TransformerBlock`、`TransformerLayer`，并一路向下传递到各个注意力和 MLP 模块。每个组件从中读取所需的配置项。其实际效果是：修改并行度、精度策略或归一化类型只需编辑一个对象，而无需将新参数串联穿透数十个调用层级。当 `fp8=True` 时，层的构建路径会路由到 TransformerEngine 的 `te.TransformerLayer`，而非 mcore 自身的实现——mcore 成为编排层，TE 提供实际的 FP8 内核。

### TransformerLayer 与注意力后端（`megatron/core/transformer/`）

`TransformerLayer` 在多个维度上可配置：

- **注意力后端**：Flash Attention（通过 `FlashAttentionCore`）、融合注意力（NVIDIA 自定义融合内核）、点积注意力（参考实现）。由 `config.use_flash_attention` 和运行时能力检查共同决定选择哪个。
- **归一化方式**：`LayerNorm` 或 `RMSNorm`，由 `config.normalization` 选择。
- **MLP 变体**：标准稠密 MLP、SwiGLU，或混合专家 MLP（`config.num_moe_experts`）。
- **偏置项**：可按投影逐一控制。

这种可插拔性对实验研究至关重要：从 LayerNorm 切换到 RMSNorm，或从标准注意力切换到 GQA，完全不需要触碰任何并行代码。并行的连线由更底层的 `MegatronModule` 和列/行并行线性层负责处理。

### MegatronModule 与并行线性层

`MegatronModule` 是所有分片模块的基类。其两个主要子类是 `ColumnParallelLinear` 和 `RowParallelLinear`，它们实现了原始 Megatron-LM v1 论文中的核心 TP 通信模式：

```
# 单个线性层对的 TP 前向传播（简化版）
# 列并行：输入 X 广播，输出 Y 按 TP rank 切分
Y_local = X @ W_col_local      # 无通信
# 行并行：输入已切分，输出需要 all-reduce
Z = Y_local @ W_row_local
Z_full = all_reduce(Z)          # 跨 tensor_model_parallel_group
```

`ColumnParallelLinear` 接受 `gather_output=False` 参数，当其输出直接馈入 `RowParallelLinear` 时，可跳过 all-gather，从而减少每个矩阵乘法对中一次通信操作。因此，mcore 中 TP 通信的开销对 MLP 内部为零，对每个注意力输出投影和每个 MLP 输出投影各为一次 all-reduce——与原始论文中的计数完全一致。

### DistributedDataParallel（mcore DDP）

mcore 自带 DDP 实现（`megatron/core/distributed/`），与 `torch.nn.parallel.DistributedDataParallel` 分离。原因如下：PyTorch DDP 将梯度分桶并将 all-reduce 与反向传播重叠，但它不理解张量并行分片。一个跨 TP rank 做列并行的参数，其梯度*不应*在 TP rank 间做 all-reduce（这些 rank 已经持有互补的分片），而应只在 DP rank 间规约。mcore 的 DDP 使用 `ParallelState` 为每个参数查找正确的规约进程组，并实现了梯度分桶，与反向计算流显式重叠。

## mcore 中的上下文并行

上下文并行（CP）是 mcore 并行分解中的第四个维度，用于支持序列长度超出单个 TP 组激活显存容量的场景。GPU 总量变为：

```
world_size = TP × PP × CP × DP
```

mcore 的 CP 实现采用**all-gather 加因果负载均衡**。序列通过奇偶交错方式分布到 CP rank 上：rank 0 获取 token [0, 2, 4, ...]，rank 1 获取 [1, 3, 5, ...]，以此类推。这种交错方式确保每个 CP rank 处理大致相同数量的因果相关 token（序列末尾的 token 有更长的因果上下文，是注意力计算中开销最大的部分；通过交错分配，这部分工作得以均摊）。在注意力前向传播过程中，每个 rank 从所有 CP 对等方 all-gather K 和 V 张量，计算其分配到的那部分注意力，然后进行规约。

这正是 `../../foundational/sequence-parallelism/` 中详细描述的 Megatron-CP 变体。Llama 3 405B 在 128k 序列长度下使用了 CP=2，使每个 GPU 的有效窗口为 64k token，同时保持完整的因果注意力正确性。

## 流水线调度模块（`megatron/core/pipeline_parallel/schedules.py`）

调度模块实现了实际的流水线执行逻辑——决定每个 GPU 在每个时间步应执行前向传播、反向传播还是等待的代码：

- **非交错 1F1B**（`forward_backward_pipelining_without_interleaving`）：来自 Megatron-LM v2 的基线调度。bubble 占比 `(PP-1)/(m+PP-1)`。
- **交错 1F1B**（`forward_backward_pipelining_with_interleaving`）：虚拟流水线阶段；每个 GPU 持有多个不连续的层组。bubble 占比 `(PP-1)/(m·v+PP-1)`，其中 `v` 是每个 GPU 的虚拟阶段数。要求 `m % PP == 0`。
- **双管道（Dual-pipe）风格调度**：较新的新增内容，实现了 DualPipe 论文背后的思想——利用成对的前向/反向传播，在 PP 阶段间重叠计算与通信，将 bubble 比例压低至 1F1B 以下。

调度模块是 mcore 流水线实现实际所在之处——DualPipe 论文描述的是算法，而 `schedules.py` 是 NeMo 和其他消费方调用的运行代码。这一区分在调试流水线 bubble 效率低下问题时至关重要：问题几乎必然出在调度变体选择或 micro-batch 数量设置上，修复也发生在这个模块中。

## 通过 TransformerEngine 集成 FP8

当设置 `TransformerConfig(fp8=True)` 时，mcore 的 `TransformerLayer` 构造函数会检查 TransformerEngine 的可用性，若存在，则构建 `te.TransformerLayer` 而非自身的实现。`TransformerConfig` 的字段 `fp8_recipe`（一个 `te.recipe.DelayedScaling` 或 `te.recipe.MXFP8BlockScaling` 对象）和 `fp8_wgrad` 会被转发给 TE。

这意味着：mcore 不自己实现 FP8 内核。它提供*配置界面*和*上下文管理*（在前向传播周围设置 `te.fp8_autocast` 上下文），而 TE 提供 cuBLAS FP8 GEMM、带 FP8 I/O 的融合注意力以及 amax 历史跟踪。用户只需在 `TransformerConfig` 中设置一个标志即可启用 FP8 训练；内核选择和缩放因子管理均在 mcore API 层以下自动处理。TE 内部机制详见 `../transformer-engine/`。

## 工程权衡

| 维度 | Megatron-Core | PyTorch FSDP（原生） | DeepSpeed | Nanotron |
|---|---|---|---|---|
| 张量并行 | 原生支持，生产级（TP 通过 NVLink 最高达 8） | 不原生支持，需自定义分片 | 通过 DS-Tensor 支持，测试覆盖较少 | 支持，设计与 mcore 一致 |
| 流水线并行 | 原生支持；1F1B + 交错 + 双管道 | 不支持 | GPipe 风格；粒度较粗 | 支持基本 1F1B |
| 上下文并行 | 原生支持（生产中 CP=2–8） | 不支持 | 不支持 | 不支持 |
| 模块化程度 | 高：可作为库导入；NeMo、Aligner 均消费 | 框架原生；与 PyTorch DDP 高度耦合 | 插件式；运行时开销较重 | 轻量；单文件设计，刻意为之 |
| 外部采用 | NeMo、NeMo-Aligner、xAI Grok、Nemotron | torchtitan、llama-recipes、大多数 PyTorch 原生训练栈 | HuggingFace Transformers、众多学术场景 | 极少；主要用于研究/教学 |
| 维护开销 | 高：NVIDIA 维护，活跃开发，API 变动频繁 | PyTorch 核心；稳定但功能迭代较慢 | 代码库庞大；ZeRO + TP + MoE 集于一框架 | 低：代码库小，依赖极少 |

## 与原始 Megatron-LM 条目的关系

`../../foundational/megatron-lm/` 中的条目覆盖的是算法：张量并行（v1）、1F1B 流水线调度（v2）、序列并行加选择性激活重算（v3）。那些算法是正确的、描述清晰的，值得单独理解。

但那个条目没有涉及的是：*当 NeMo 在 16,000 张 H100 上训练 Llama 3 时，这些算法实际上是如何被调用的*。答案是：通过 mcore 的 `ParallelState`、`TransformerConfig`、`TransformerLayer` 和 `schedules.py`。论文条目是算法规范；本条目是实现。当 2024–2025 年的技术报告引用"Megatron 风格 4D 并行"时，它们指的是这里描述的 mcore API 层。

## 交叉参考

- `../../foundational/megatron-lm/` — mcore 所实现和扩展的论文三部曲
- `../transformer-engine/` — 通过 `TransformerConfig(fp8=True)` 接入；mcore 编排，TE 执行
- `../../foundational/sequence-parallelism/` — mcore 的 CP 即此处描述的 Megatron-CP 变体
- `../../meta/llama3/` — Llama 3 405B 训练使用了 mcore 的 4D 并行（TP=8, PP=16, CP=2, DP=512）
- `../../foundational/zero-fsdp/` — 替代数据并行方案；可与 mcore 的 TP/PP 正交组合

## 参考文献

- [1] Shoeybi et al. _Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism._ arXiv:1909.08053, 2019.
- [2] Narayanan et al. _Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM._ arXiv:2104.04473, 2021.
- [3] Korthikanti et al. _Reducing Activation Recomputation in Large Transformer Models._ arXiv:2205.05198, 2022.
- [4] Dubey et al. _The Llama 3 Herd of Models._ arXiv:2407.21783, 2024.（第 3.3 节：使用 mcore 4D 并行的训练基础设施。）
- [5] NVIDIA Megatron-Core 源码：`github.com/NVIDIA/Megatron-LM/tree/main/megatron/core`
