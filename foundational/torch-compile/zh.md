# torch.compile 与 Inductor 后端

- **机构**：PyTorch / Meta
- **发布时间**：PyTorch 2.0（2023 年 3 月）；Inductor 在 2.1 稳定版中正式落地
- **链接**：[PyTorch 2.0 论文 (arXiv:2311.13608)](https://arxiv.org/abs/2311.13608) · [Inductor 博客](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) · [torch.compile 文档](https://pytorch.org/docs/stable/torch.compiler.html)

## 一句话总结

`torch.compile` 是 PyTorch 的生产级编译通道，从 Python 字节码一路向下直到 Triton kernel 或 C++/OpenMP。三个组件各司其职：**TorchDynamo** 在不破坏用户代码的前提下从活跃 Python 执行中捕获计算图；**AOTAutograd** 将捕获的图提升为前向-反向联合单元，让反向传播也被纳入编译范围；**Inductor** 将结果图下降到 Triton（GPU）或 C++/OpenMP（CPU），并在此过程中做算子融合、内存布局优化、归约/逐点融合。净效果是：内存带宽受限的操作——LayerNorm、GELU、残差加法、RMSNorm——被合并成更少的 CUDA kernel 启动次数，"急切 PyTorch"与"手写 Triton"之间的性能差距收缩到训练端到端约 15–30%、推理约 20–50%。

## 背景与动机

急切模式的 PyTorch 为每个算子支付一次独立的调度开销：每个 `torch.add`、`torch.nn.functional.layer_norm`、注意力投影都会触发一次单独的 CUDA kernel 启动。对于计算密集型操作（大矩阵乘），这无关紧要——kernel 本身运行数毫秒。但现代 Transformer 中充满了内存受限操作：每个 LayerNorm、每个 GELU 激活、每个残差加法都需要往返 HBM。以 4096 维隐藏状态的 LayerNorm 为例，每次 kernel 启动都要读取一次输入并写回一次输出。如果 LayerNorm、GELU、残差加法是三个独立 kernel，HBM 流量就是实际需要的 3 倍。

解法在 CUDA 时代就已熟知：**算子融合**——将它们合并成一个 kernel，从 HBM 读取输入一次，在寄存器中依次完成三个操作，再将结果写回一次。`torch.compile` 之前，这需要手写 Triton 或 CUDA kernel，或者依赖 nvFuser。`torch.compile` 将这个过程自动化。

## 三层编译栈

### 第一层：TorchDynamo（前端）

TorchDynamo 在 `CPython` 层面拦截 Python 字节码。当 `torch.compile` 包装一个函数时，Dynamo 安装字节码评估钩子：它不执行指令，而是**追踪**它们，构建出所遇 PyTorch 操作的 FX（函数式变换）图。

关键在于，Dynamo 做这件事**不要求用户编写静态模型**。控制流、循环、Python 层面的副作用都通过**守卫（guard）**处理：追踪时对张量形状、数据类型、Python 值做出的假设。只要守卫成立，编译产物就被缓存复用。如果守卫失败——例如批次大小改变——Dynamo 对该次调用回退到急切模式，并可选地以新配置重新编译。

守卫机制使 `torch.compile` 能与任意研究代码组合。它也带来代价：每次调用都要进行守卫检查，守卫失败时需要支付重编译代价。`torch._dynamo.config.cache_size_limit`（默认 8）控制停止重编译前可存储多少个编译变体。

**图断裂（graph break）** 是需要重点关注的失效模式。每当 Dynamo 遇到无法追踪的内容——`print` 语句、数据相关的控制流、调用非 `torch` 库——它就会产生一次图断裂：追踪过程拆分成一个编译段，后接一个急切段。每次断裂都是一个编译边界，也是一次性能悬崖。使用 `torch._dynamo.explain(fn, *args)` 诊断断裂点及其原因。

### 第二层：AOTAutograd

来自 Dynamo 的 FX 图只表示前向传播。AOTAutograd 将其提升为**前向-反向联合图**：它对前向计算和对应的反向（梯度）计算进行符号追踪，合并成一张图。

这里的关键优势是反向算子成为编译单元的一等公民。Inductor 可以融合反向的逐点算子、提升冗余计算、针对反向传播优化内存布局——与前向传播同等激进。如果没有 AOTAutograd，编译器只能看到前向图并生成融合的前向 kernel，然后对反向回退到急切模式，在训练负载中错失一半优化机会。

AOTAutograd 也处理保存张量的逻辑：它追踪哪些中间激活必须保存供反向使用、哪些可以重新计算（与 `torch.utils.checkpoint` 交互），并将这些信息编入联合图。

### 第三层：Inductor（后端）

Inductor 是代码生成后端。它接收联合 FX 图并执行：

1. **融合逐点与归约算子** — 相邻的逐元素操作（偏置加法、激活函数、残差、归一化）被合并成单个循环（CPU）或单个 Triton kernel（GPU）。
2. **优化内存布局** — Inductor 可能选择 channels-last 或自定义步幅以改善内存合并访问和 TensorCore 对齐。
3. **生成 Triton kernel 源码** — 对 GPU 目标，Inductor 自动生成 Triton kernel 代码（参见 `../triton/`）。生成的代码随后由 Triton 编译器 JIT 编译，后者负责线程映射、共享内存分配和软件流水。`torch.compile` 是 Triton DSL 之上的生产级编排层。
4. **对 CPU**，Inductor 生成带 OpenMP 并行的 C++ 代码。

融合效果在内存带宽受限的算子上最为显著。以融合的 LayerNorm + GELU + 残差加法为例：只需一次 kernel 启动——从 HBM 读取输入一次，计算均值和方差、归一化、应用 GELU、加残差，然后写回输出一次。三个独立 kernel 则会产生三次读取和三次写入。在 HBM 带宽 2 TB/s、kernel 启动延迟约 5 µs 的 A100 上，节省的既是带宽（这类算子提速 2–4×），也是启动开销。

## 编译模式与 CUDA Graphs

`torch.compile` 通过 `mode` 参数暴露三种编译模式：

**`default`**：标准 Inductor 优化——融合、布局优化、Triton kernel 生成。在编译时间（首次调用典型 Transformer 块约 10–30 秒）和运行时性能之间取得平衡。适用于大多数训练和推理负载。

**`reduce-overhead`**：在 Inductor 基础上加入 CUDA Graphs。CUDA Graph 将一系列 GPU 命令（kernel 启动、内存操作）录制为单个图对象。回放时 GPU 执行录制好的序列，CPU 完全不参与——消除了每 kernel ~5–10 µs 的 CPU 端启动开销。

CUDA Graphs 带来的额外加速是**小批次推理约 10–30%**，此时启动开销在运行时中占比显著。大批次下 kernel 运行时长达毫秒级，收益较小。关键约束是**静态形状**：CUDA Graph 在录制时捕获指针和启动参数，形状变化（不同批次大小、不同序列长度）会使图失效并强制重录。生产中的应对方案是将输入分桶到少数几种固定形状，或接受重录代价。

**`max-autotune`**：对 Triton kernel 配置（tile 大小、warp 数量、流水线阶段数）运行自动调优，在特定硬件上找到每个 kernel 的最优设置。编译时间增加到大模型的 2–10 分钟；运行时是可达到的最佳值。适用于一次编译、反复服务的推理部署。

## 预热与编译开销

`torch.compiled` 模型的第一次前向传播会产生全部 JIT 编译代价。对 70 亿参数规模的 Transformer，这通常是 **30–120 秒**，具体取决于唯一 FX 图段的数量、遇到的不同形状数量以及编译模式。使用动态序列长度（如来自 DataLoader 的可变长度批次）的训练循环可能在前几步触发多次重编译。

实践建议：

- **在计时或评估前显式预热**：运行若干批次后再开始测量吞吐。
- **尽量使用静态形状**：固定批次大小、填充序列、DataLoader 中设置 `drop_last=True`。
- **序列化编译产物**：`torch._dynamo.config.cache_size_limit` 和 `torch.compiler.reset()` 管理缓存；通过 `TORCHINDUCTOR_CACHE_DIR` 可启用持久化磁盘缓存。

编译代价是每个形状签名的一次性支出。训练时摊销到数百乃至数千步上。推理服务时，模型在启动时编译一次。

## 与 FSDP 的集成

`torch.compile` + FSDP（全分片数据并行）是不能放入单卡显存的模型的标准训练配置。关键要求是在 FSDP 包装器中使用 `use_orig_params=True`：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, use_orig_params=True)
model = torch.compile(model)
```

不使用 `use_orig_params=True` 时，FSDP 会将每个分片的参数展平为单个 `FlatParameter`，Dynamo 无法正确追踪——它会在每个 FSDP 单元边界处产生图断裂。使用 `use_orig_params=True` 后，从 Dynamo 视角看参数保留原始形状，FX 图能干净地捕获完整前向传播。编译随后**在分片后按 rank 进行**：每个 rank 编译自己的模型分片，集合通信算子（all-gather、reduce-scatter）在图中被视为不透明操作。FSDP 基础原理参见 `../zero-fsdp/`。

## 与 FlashAttention 的集成

`torch.nn.functional.scaled_dot_product_attention`（SDPA）在 PyTorch 调度器中注册，当满足以下条件时自动分发到 FlashAttention 后端：
- head 维度在支持范围内（通常 ≤128）
- 输入数据类型为 FP16 或 BF16
- 序列掩码不要求走非 flash 路径

`torch.compile` 将 SDPA 视为单个融合操作——它不会尝试将其拆解为 Triton 逐点算子，因为现有的 FlashAttention kernel 已实现接近最优的 HBM 利用率。因此，compile + FlashAttention 的组合效果是：Inductor 融合注意力前后的所有算子（QKV 投影、残差、LayerNorm），SDPA 算子内部调用 FlashAttention，两侧的融合均达到最优。FlashAttention 原理参见 `../flash-attention/`。

## 算子融合：核心性能驱动

`torch.compile` 在 LLM 负载中最主要的收益来自逐点算子融合。以单个 Transformer 块注意力后的前向传播为例：

1. 残差加法：`x = x + attn_output`
2. LayerNorm：`x = layer_norm(x)`
3. 线性投影：`x = F.linear(x, W1)`
4. GELU 激活：`x = F.gelu(x)`
5. 线性投影：`x = F.linear(x, W2)`
6. 残差加法：`x = x + ff_output`
7. LayerNorm：`x = layer_norm(x)`

步骤 1–2 和 6–7 是纯逐点/归约操作。急切模式下每个步骤是独立 kernel，各自有独立的 HBM 读写。Inductor 将步骤 1–2 融合为一个 kernel，将 6–7 融合为另一个（根据形状还可进一步与周围操作融合）。内存访问的节省可以用下式表达：

$$\text{带宽节省比} \approx \frac{N_\text{算子} \cdot T_\text{读写}}{T_\text{融合读写} + T_\text{融合计算}}$$

对于 `T_compute` 可忽略的内存受限算子，融合 N 个操作后 HBM 流量接近降低 N 倍。实测中，LayerNorm + 残差融合在该融合算子上实现 **2–4 倍加速**，对典型 LLM 训练配置的端到端训练加速贡献 15–30%。

## 工程 Tradeoff

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| `default` 模式 + 静态形状 | 编译时间适中（30–60 秒）；融合可靠；兼容 FSDP | 无 CUDA Graph 加速；形状动态时部分算子无法融合 | 通用训练、微调、开发阶段 |
| `reduce-overhead` + CUDA Graphs | 消除启动开销，额外提速 10–30% | 要求严格静态形状；形状变化触发重录 | 固定批次大小的推理服务；基准测试 |
| `max-autotune` | 可达最佳运行时性能；自动调优 Triton 配置 | 编译时间 2–10 分钟；仅在模型运行数小时以上时值得 | 一次编译长期服务的生产推理部署 |
| `torch.compile` + FSDP（`use_orig_params=True`）| 分片训练时完整融合；按 rank 编译 | 需要正确的 FSDP 包装；调试复杂度增加 | 多卡训练 7B+ 规模模型 |
| 急切模式（不编译） | 零预热；调试最简单；自由支持动态形状 | 慢 15–50%；每个算子独立 kernel 启动 | 调试、形状不稳定的研究代码、极短任务 |
| `torch.compile` + 梯度检查点 | 融合前向 + 选择性重计算；显存/计算平衡好 | AOTAutograd 与检查点的交互需要小心处理 | 长序列下显存受限的训练 |

## 实测数据

以下数据适用于 A100/H100 上 BF16/FP16 的 LLM 训练与推理：

- **训练加速（前向 + 反向）**：对密集 Transformer 层，比急切模式快 15–30%；LayerNorm/激活算子密集的模型偏高端。
- **推理加速**：视批次大小和算子分布，比急切模式快 20–50%；小批次推理受益最大，因为启动开销主导运行时。
- **CUDA Graphs 额外加速**：在融合 kernel 基础上，小批次推理（batch 1–8）额外提速 10–30%。
- **内存受限算子融合**：单独测试 LayerNorm、GELU、残差加法时提速 2–4 倍；matmul 的稀释使其对端到端的贡献减小。
- **编译预热时间**：70 亿参数规模模型在 `default` 模式下首次调用约 30–120 秒；`max-autotune` 下 2–10 分钟。

## 局限与失效模式

**图断裂**是性能损失的首要来源。Dynamo 遇到以下情况时产生断裂：`print` 语句、`if tensor.item() > threshold` 分支、调用非 PyTorch 库、`torch.Tensor.numpy()`。每次断裂在断裂点两侧各产生一个编译段，但断裂点本身始终是急切执行。一个有 5 次图断裂的模型获得的融合效果远不如干净图。诊断工具：`torch._dynamo.explain(fn, *args)` 或 `TORCH_COMPILE_DEBUG=1`。

**数据相关形状**（形状取决于张量值，如 `torch.nonzero` 或布尔掩码 `torch.where` 之后的形状）迫使 Dynamo 要么走守卫-重编译路径，要么产生断裂。将输入填充到静态形状通常是服务场景的正确工程响应。

**编译时间**对探索性工作可能令人望而却步。一个要测试 10 种不同模型配置的研究循环，如果每次编译耗时 2 分钟，开发体验会很差。推荐的工作流是在急切模式下开发，在基准测试或生产部署的最后步骤才加入 `torch.compile`。

## 交叉参考

- `../triton/` — Inductor 的 GPU 代码生成后端；`torch.compile` 是 Triton DSL 之上的生产级编排层；Inductor 生成的每个融合 kernel 都是一个 Triton kernel。
- `../flash-attention/` — SDPA 分发到 FlashAttention；`torch.compile` + FlashAttention 是标准训练和推理组合；compile 融合 SDPA 周围的算子，FA 自身处理 SDPA。
- `../zero-fsdp/` — FSDP 集成需要 `use_orig_params=True`；compile + FSDP 组合是 2024 年以后 PyTorch 多卡训练的默认配置。

## 参考文献

- [1] Ansel et al. _PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation._ ASPLOS '24 / arXiv:2311.13608.
- [2] TorchInductor 设计文档：https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
- [3] torch.compile 教程：https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- [4] torch.compile 文档：https://pytorch.org/docs/stable/torch.compiler.html
- [5] PyTorch 中的 CUDA Graphs：https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
